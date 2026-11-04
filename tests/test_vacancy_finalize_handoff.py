"""Regression tests for the GRACE -> MACE archive relocation handoff.

No ML calculator or GPU is needed for these checks.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import pytest

from dopingflow.vacancy_mc_staged import (
    _canonical_composition,
    _discover_finalize_parents,
    _parent_included,
    _relocated_legacy_fingerprint,
    _search_fingerprint,
    _verify_selected_archive,
)


@dataclass(frozen=True)
class FakeAnalysis:
    oxygen_reference_file: Path
    oxygen_reference_structure: Path


@dataclass(frozen=True)
class FakeCfg:
    parent_source: str
    parent_directory: Path
    outdir: Path
    vacancy_species: str
    supercell: tuple[int, int, int]
    oxidation_states: dict
    analysis: FakeAnalysis


@dataclass(frozen=True)
class FakeMcCfg:
    backend: str = "grace"
    model: str = "GRACE-1L-OMAT"
    task: str = ""
    device: str = "cuda"
    gpu_id: int = 0
    explicitly_configured: bool = True


def _cfg(root: Path) -> FakeCfg:
    return FakeCfg(
        parent_source="directory",
        parent_directory=root / "Ti-complete",
        outdir=root / "Ti-main",
        vacancy_species="O",
        supercell=(2, 2, 2),
        oxidation_states={"Sb": (3, 5)},
        analysis=FakeAnalysis(root / "refs.json", root / "O2.POSCAR"),
    )


def test_composition_selectors_are_order_and_notation_independent():
    parent = {"composition": "Sb5_Ti2p5", "candidate": "candidate_003"}
    assert _canonical_composition("Ti_2.5Sb_5") == _canonical_composition("Sb5_Ti2p5")
    assert _parent_included(parent, ["Ti_2.5Sb_5"])
    assert _parent_included(parent, ["Ti_2.5Sb_5/candidate_003"])
    assert not _parent_included(parent, ["Ti_2.5Sb_5/candidate_004"])


def test_discovery_joins_existing_archive_to_original_parent(tmp_path):
    cfg = _cfg(tmp_path)
    output = tmp_path / "vacancy-MC-2x2x2"
    composition = "Sb5_Ti2p5"
    for name, energy in [("candidate_001", -10.0), ("candidate_003", -12.0)]:
        source = cfg.parent_directory / composition / name
        (source / "01_scan").mkdir(parents=True)
        (source / "02_relax").mkdir(parents=True)
        (source / "01_scan" / "POSCAR").write_text("source scan")
        (source / "02_relax" / "POSCAR").write_text("source relax")
        (source / "02_relax" / "meta.json").write_text(
            json.dumps({"energy_relaxed_eV": energy})
        )
        archive = output / composition / name / "05_vacancies"
        archive.mkdir(parents=True)
        (archive / "mc_search_meta.json").write_text('{"status":"complete"}')
    parents = _discover_finalize_parents(
        {
            "output_directory": "vacancy-MC-2x2x2",
            "parent_include": ["Ti_2.5Sb_5"],
            "parent_pick": "lowest_energy",
        },
        tmp_path,
        cfg,
    )
    assert [p["parent_id"] for p in parents] == ["Sb5_Ti2p5/candidate_003"]
    assert parents[0]["candidate_dir"].is_relative_to(output)
    assert parents[0]["relaxed_path"].is_relative_to(cfg.parent_directory)


def test_legacy_hash_matches_with_transferred_source_bytes(tmp_path):
    root = tmp_path / "new-machine"
    cfg = _cfg(root)
    parent_id = "Sb5_Ti2p5/candidate_003"
    source_candidate = cfg.parent_directory / parent_id
    scan = source_candidate / "01_scan" / "POSCAR"
    relaxed = source_candidate / "02_relax" / "POSCAR"
    for file, content in ((scan, "original scan bytes"), (relaxed, "original relaxed bytes")):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)

    original_parent_root = Path("/home/shiv/old-run/Ti-complete")
    original_relaxed = original_parent_root / parent_id / "02_relax" / "POSCAR"
    original_scan = original_parent_root / parent_id / "01_scan" / "POSCAR"
    old_cfg = replace(
        cfg,
        outdir=original_parent_root.parent / "Ti-main",
        parent_directory=original_parent_root,
    )
    mc = FakeMcCfg()
    counts = (1, 2, 3, 4)
    expected = _search_fingerprint(
        old_cfg,
        mc,
        counts,
        scan,
        relaxed,
        source_identity_paths=(original_scan, original_relaxed),
    )

    archive = root / "vacancy-MC-2x2x2" / parent_id / "05_vacancies"
    group = archive / "V_O_01"
    group.mkdir(parents=True)
    (group / "selected_candidates.txt").write_text("mc_000001\n")
    gen_dir = group / "mc_000001" / "00_generate"
    gen_dir.mkdir(parents=True)
    (gen_dir / "meta.json").write_text(
        json.dumps({"source_parent_path": str(original_relaxed)})
    )
    parent = {
        "parent_id": parent_id,
        "candidate_dir": archive.parent,
        "symmetry_path": scan,
        "relaxed_path": relaxed,
    }
    assert (
        _relocated_legacy_fingerprint(parent, cfg, mc, counts, root, [1, 2, 3, 4])
        == expected
    )

    relaxed.write_text("CHANGED")
    assert (
        _relocated_legacy_fingerprint(parent, cfg, mc, counts, root, [1, 2, 3, 4])
        != expected
    )


def test_preflight_rejects_inconsistent_archived_candidates(tmp_path):
    cfg = _cfg(tmp_path)
    parent_id = "Sb5_Ti2p5/candidate_003"
    archive = tmp_path / "vacancy-MC-2x2x2" / parent_id / "05_vacancies"
    selection = archive / "V_O_01" / "selected_candidates.txt"
    selection.parent.mkdir(parents=True)
    selection.write_text("mc_000001\n")
    generate = selection.parent / "mc_000001" / "00_generate"
    generate.mkdir(parents=True)
    (generate / "POSCAR").write_text("POSCAR")
    (generate / "meta.json").write_text(
        json.dumps(
            {
                "parent_id": parent_id,
                "search_fingerprint": "different",
                "n_vacancies": 1,
            }
        )
    )
    with pytest.raises(ValueError, match="Inconsistent"):
        _verify_selected_archive(
            {"parent_id": parent_id, "candidate_dir": archive.parent},
            cfg,
            [1],
            "expected fingerprint",
        )

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
    _load_search_context,
)


@dataclass(frozen=True)
class FakeAnalysis:
    oxygen_reference_file: Path
    oxygen_reference_structure: Path
    oxygen_calibration_experimental_data: Path | None = None
    oxygen_calibration_dataset_cache_dir: Path | None = None


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


def test_legacy_checkpoint_requires_opt_in_and_still_audits_contents(tmp_path, monkeypatch):
    """Unknown historical search hash needs opt-in; damaged raw GRACE data still fails."""
    from dataclasses import asdict
    import numpy as np
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp import Poscar
    import dopingflow.vacancy_mc_staged as module

    cfg = _cfg(tmp_path)
    mc = FakeMcCfg()
    parent_id = "Sb5_Ti2p5/candidate_003"
    archive = tmp_path / "vacancy-MC-2x2x2" / parent_id / "05_vacancies"
    selection = archive / "V_O_01" / "selected_candidates.txt"
    selection.parent.mkdir(parents=True)
    selection.write_text("mc_000001\n")
    generated = selection.parent / "mc_000001" / "00_generate"
    generated.mkdir(parents=True)
    scan_dir = generated.parent / "01_scan"
    scan_dir.mkdir(parents=True)

    parent_structure = Structure(
        Lattice.cubic(12), ["Sn", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    defective = parent_structure.copy()
    defective.remove_sites([1])
    Poscar(defective).write_file(generated / "POSCAR")
    saved_fingerprint = "legacy-search-fingerprint-that-does-not-recompute"
    old_parent_path = (
        "/home/shiv/nfs/kaz/KERAMHyK/SnO2/Ti-complete/"
        "Sb5_Ti2p5/candidate_003/02_relax/POSCAR"
    )
    (generated / "meta.json").write_text(json.dumps({
        "configuration_id": "mc_000001",
        "parent_id": parent_id,
        "search_fingerprint": saved_fingerprint,
        "n_vacancies": 1,
        "supercell": list(cfg.supercell),
        "source_parent_path": old_parent_path,
    }))
    (scan_dir / "meta.json").write_text(json.dumps({
        "search_fingerprint": saved_fingerprint,
        "backend": "grace",
        "model": mc.model,
        "task": mc.task,
        "energy_sp_total_eV": -23.5,
    }))
    (archive / "mc_search_meta.json").write_text(json.dumps({
        "status": "complete",
        "parent_id": parent_id,
        "search_fingerprint": saved_fingerprint,
        "mc_search_calculator": asdict(mc),
        "supercell": list(cfg.supercell),
        "explicit_vacancy_counts": [1],
        "vacancy_range": {"explicit_counts": [1]},
    }))
    source = tmp_path / "Ti-complete" / parent_id
    for relative in ("01_scan/POSCAR", "02_relax/POSCAR"):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Poscar(parent_structure).write_file(target)
    parent = {
        "parent_id": parent_id,
        "candidate_dir": archive.parent,
        "symmetry_path": source / "01_scan/POSCAR",
        "relaxed_path": source / "02_relax/POSCAR",
    }
    monkeypatch.setattr(
        module, "_expanded_parent", lambda *_: (parent_structure, parent_structure, {})
    )
    monkeypatch.setattr(module, "_composition_fields", lambda *_: ({}, {}))
    monkeypatch.setattr(module, "_search_fingerprint", lambda *_args, **_kwargs: "different")
    monkeypatch.setattr(module, "_relocated_legacy_fingerprint", lambda *_args: None)

    with pytest.raises(ValueError, match="allow_legacy_archive_fingerprint_mismatch"):
        _load_search_context(
            parent, cfg, mc, (1,), config_root=tmp_path
        )
    result = _load_search_context(
        parent, cfg, mc, (1,), config_root=tmp_path,
        allow_legacy_fingerprint_mismatch=True,
    )
    assert result[1] == saved_fingerprint
    assert parent["legacy_fingerprint_status"] == "audited_not_reproduced"

    # A valid-looking archive with an incorrect GRACE scan energy is rejected.
    scan_metadata = json.loads((scan_dir / "meta.json").read_text())
    scan_metadata["energy_sp_total_eV"] = float("nan")
    (scan_dir / "meta.json").write_text(json.dumps(scan_metadata))
    with pytest.raises(ValueError, match="Inconsistent"):
        _load_search_context(
            parent, cfg, mc, (1,), config_root=tmp_path,
            allow_legacy_fingerprint_mismatch=True,
        )


def test_finalize_never_redirects_parent_source_to_archive(monkeypatch, tmp_path):
    """Archived MACE handoff owns its own directory mapping, unlike MC search."""
    from dopingflow import vacancies as base
    from dopingflow import vacancy_parent_selection_extensions as extension

    actual_parser = base.parse_vacancy_config
    seen = []

    def fake_staged_finalizer(raw, root, *, config_path=None):
        seen.append((raw, root, config_path))
        # The old search-stage wrapper replaced this parser, causing MACE
        # to discover original source POSCARs inside output_directory.
        assert base.parse_vacancy_config is actual_parser
        return tmp_path / "vacancy-MC-2x2x2" / "vacancies_database.csv"

    monkeypatch.setattr(extension, "_ORIGINAL_FINALIZE", fake_staged_finalizer)
    config = {
        "vacancies": {
            "output_directory": "vacancy-MC-2x2x2",
            "parent_directory": "Ti-Mn-Nb-Ta-Ce-Ru-In-Sn-Sb-complete",
            "parent_pick": "lowest_energy",
            "parent_include": ["Ce_2.5Sb_2.5"],
        }
    }
    expected = tmp_path / "vacancy-MC-2x2x2" / "vacancies_database.csv"
    assert extension.run_vacancy_finalize(config, tmp_path) == expected
    assert seen == [(config, tmp_path, None)]
    assert base.parse_vacancy_config is actual_parser


def test_finalization_metadata_serializes_enhanced_analysis_paths(tmp_path):
    """Production regression: optional calibration paths must not crash meta.json."""
    from dopingflow import vacancies as base

    cfg = _cfg(tmp_path)
    cfg = replace(
        cfg,
        analysis=replace(
            cfg.analysis,
            oxygen_calibration_experimental_data=tmp_path / "refs" / "oxides.json",
            oxygen_calibration_dataset_cache_dir=tmp_path / "cached-datasets",
        ),
    )
    resolved_config = base._jsonable_config(cfg)
    analysis = resolved_config["analysis"]
    assert analysis["oxygen_reference_file"] == str(cfg.analysis.oxygen_reference_file)
    assert analysis["oxygen_reference_structure"] == str(cfg.analysis.oxygen_reference_structure)
    assert analysis["oxygen_calibration_experimental_data"] == str(
        cfg.analysis.oxygen_calibration_experimental_data
    )
    assert analysis["oxygen_calibration_dataset_cache_dir"] == str(
        cfg.analysis.oxygen_calibration_dataset_cache_dir
    )

    final_meta = tmp_path / "finalize-meta.json"
    base._write_json(final_meta, {
        "status": "complete",
        "stage": "finalize",
        "resolved_config": resolved_config,
    })
    saved = json.loads(final_meta.read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["resolved_config"]["analysis"] == analysis


def test_finalization_metadata_serializes_optional_none_paths(tmp_path):
    """Unconfigured calibration paths remain JSON null, not string 'None'."""
    from dopingflow import vacancies as base

    cfg = _cfg(tmp_path)
    data = base._jsonable_config(cfg)
    assert data["analysis"]["oxygen_calibration_experimental_data"] is None
    assert data["analysis"]["oxygen_calibration_dataset_cache_dir"] is None
    assert json.loads(json.dumps(data)) == data

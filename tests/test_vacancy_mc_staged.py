from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import dopingflow.vacancy_mc_staged as staged
import dopingflow.vacancy_parent_selection_extensions as parent_selection
from dopingflow.vacancy_parent_selection_extensions import pick_parents
from dopingflow.vacancy_mc_extensions import MonteCarloSearchCalculatorConfig
from pymatgen.core import Lattice, Structure


def _cfg(tmp_path: Path):
    return SimpleNamespace(
        search_method="monte-carlo",
        backend="mace",
        model="mh-1",
        task="matpes_r2scan",
        device="cuda",
        gpu_id=0,
        tf_threads=1,
        omp_threads=1,
        parent_source="selected_candidates",
        parent_directory=None,
        outdir=tmp_path,
        analysis=SimpleNamespace(enabled=False),
    )


def _raw():
    return {
        "vacancies": {
            "search_method": "monte-carlo",
            "vacancy_counts": [1, 2],
            "mc_backend": "grace",
            "mc_model": "GRACE-1L-OMAT",
            "mc_task": "",
            "mc_device": "cuda",
            "mc_gpu_id": 0,
        }
    }


def _isolate_empty_parent_set(monkeypatch, tmp_path):
    monkeypatch.setattr(staged._base, "parse_vacancy_config", lambda raw, root: _cfg(tmp_path))
    monkeypatch.setattr(staged._base, "discover_selected_parents", lambda root: [])
    monkeypatch.setattr(staged._base, "_write_csv", lambda path, rows: None)
    monkeypatch.setattr(staged._base, "_write_json", lambda path, rows: None)


def test_search_stage_requires_only_mc_backend(monkeypatch, tmp_path):
    _isolate_empty_parent_set(monkeypatch, tmp_path)
    checked = []
    built = []
    monkeypatch.setattr(
        staged,
        "check_backend_dependency",
        lambda backend, stage_name: checked.append((backend, stage_name)),
    )
    monkeypatch.setattr(staged, "prepare_backend_runtime", lambda **kwargs: None)
    monkeypatch.setattr(
        staged,
        "build_ase_calculator",
        lambda **kwargs: built.append(kwargs) or object(),
    )

    staged.run_vacancy_mc_search(_raw(), tmp_path)

    assert [backend for backend, _ in checked] == ["grace"]
    assert [item["backend"] for item in built] == ["grace"]


def test_finalize_stage_requires_only_final_backend(monkeypatch, tmp_path):
    _isolate_empty_parent_set(monkeypatch, tmp_path)
    checked = []
    built = []
    monkeypatch.setattr(
        staged,
        "check_backend_dependency",
        lambda backend, stage_name: checked.append((backend, stage_name)),
    )
    monkeypatch.setattr(staged, "prepare_backend_runtime", lambda **kwargs: None)
    monkeypatch.setattr(
        staged,
        "build_ase_calculator",
        lambda **kwargs: built.append(kwargs) or object(),
    )

    staged.run_vacancy_finalize(_raw(), tmp_path)

    assert [backend for backend, _ in checked] == ["mace"]
    assert [item["backend"] for item in built] == ["mace"]


def test_pick_lowest_energy_parent_keeps_first_selected_per_composition():
    parents = [
        {"composition": "Sb5_Ti2p5", "parent_id": "Sb5_Ti2p5/candidate_014"},
        {"composition": "Sb5_Ti2p5", "parent_id": "Sb5_Ti2p5/candidate_022"},
        {"composition": "Sb5_Ce2p5", "parent_id": "Sb5_Ce2p5/candidate_029"},
        {"composition": "Sb5_Ce2p5", "parent_id": "Sb5_Ce2p5/candidate_031"},
    ]

    selected = pick_parents(parents, "lowest_energy")

    assert [p["parent_id"] for p in selected] == [
        "Sb5_Ti2p5/candidate_014",
        "Sb5_Ce2p5/candidate_029",
    ]


def test_dedicated_output_directory_is_shared_by_search_and_finalize(monkeypatch, tmp_path):
    source_root = tmp_path / "parents"
    output_root = tmp_path / "vacancy-MC-2x2x2"
    source_candidate = source_root / "Sb5_Ti2p5" / "candidate_014"

    @dataclass(frozen=True)
    class DummyConfig:
        parent_source: str
        parent_directory: Path
        outdir: Path

    def parse_config(_raw, _root):
        return DummyConfig(
            parent_source="directory",
            parent_directory=source_root,
            outdir=source_root,
        )

    discovered_parent = {
        "parent_id": "Sb5_Ti2p5/candidate_014",
        "composition": "Sb5_Ti2p5",
        "candidate": "candidate_014",
        "candidate_dir": source_candidate,
        "symmetry_path": source_candidate / "01_scan" / "POSCAR",
        "relaxed_path": source_candidate / "02_relax" / "POSCAR",
    }

    monkeypatch.setattr(parent_selection._base, "parse_vacancy_config", parse_config)
    monkeypatch.setattr(
        parent_selection._base,
        "discover_selected_parents",
        lambda root: [discovered_parent],
    )

    raw = {
        "vacancies": {
            "output_directory": str(output_root),
            "parent_pick": "lowest_energy",
        }
    }

    observed = []

    def fake_stage(_raw, _root, *, config_path=None):
        cfg = parent_selection._base.parse_vacancy_config(_raw, _root)
        parent_root = cfg.parent_directory
        parents = parent_selection._base.discover_selected_parents(parent_root)
        observed.append(
            (
                parent_root,
                parents[0]["candidate_dir"],
                parents[0]["source_candidate_dir"],
            )
        )
        return parent_root / "stage.csv"

    search_result = parent_selection._run_with_parent_filter(
        fake_stage, raw, tmp_path
    )
    finalize_result = parent_selection._run_with_parent_filter(
        fake_stage, raw, tmp_path
    )

    expected_candidate = output_root / "Sb5_Ti2p5" / "candidate_014"
    assert search_result == output_root / "stage.csv"
    assert finalize_result == output_root / "stage.csv"
    assert observed == [
        (output_root, expected_candidate, source_candidate),
        (output_root, expected_candidate, source_candidate),
    ]


def _search_reference_inputs(tmp_path):
    candidate_dir = tmp_path / "Sb5_Ti2p5" / "candidate_014"
    (candidate_dir / "02_relax").mkdir(parents=True)
    structure = Structure(
        Lattice.cubic(4.8),
        ["Sn", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    parent = {
        "parent_id": "Sb5_Ti2p5/candidate_014",
        "composition": "Sb5_Ti2p5",
        "candidate": "candidate_014",
        "candidate_dir": candidate_dir,
    }
    cfg = SimpleNamespace(
        resume=True,
        supercell=(1, 1, 1),
        vacancy_species="O",
    )
    mc_cfg = MonteCarloSearchCalculatorConfig(
        backend="grace",
        model="GRACE-1L-OMAT",
        task="",
        device="cuda",
        gpu_id=0,
    )
    fields = {
        "composition_directory": "Sb5_Ti2p5",
        "composition": "Sb5_Ti2p5",
        "candidate": "candidate_014",
        "host_species": "Sn",
        "dopant_counts_from_parent": {"Sb": 16, "Ti": 8},
        "dopant_counts_json": {"Sb": 16, "Ti": 8},
        "n_host": 296,
        "n_total_dopants": 24,
        "n_total_cations": 320,
        "n_oxygen_sites_parent": 640,
    }
    return parent, structure, fields, cfg, mc_cfg


def test_search_parent_reference_calculates_once_then_uses_cache(monkeypatch, tmp_path):
    parent, structure, fields, cfg, mc_cfg = _search_reference_inputs(tmp_path)
    calls = []
    monkeypatch.setattr(
        staged,
        "structure_energy_with_calculator",
        lambda structure, calculator: calls.append(structure) or -123.5,
    )

    first = staged._search_parent_reference_row(
        parent, structure, fields, cfg, mc_cfg, object(), "same-search"
    )
    second = staged._search_parent_reference_row(
        parent, structure, fields, cfg, mc_cfg, object(), "same-search"
    )

    assert len(calls) == 1
    assert first["n_vacancies"] == 0
    assert first["mc_search_energy_sp_total_eV"] == -123.5
    assert second["mc_search_energy_sp_total_eV"] == -123.5
    assert second["mc_search_energy_source"] == "mc_parent_reference_cache"


def test_search_parent_reference_reuses_matching_source_relaxation(monkeypatch, tmp_path):
    parent, structure, fields, cfg, mc_cfg = _search_reference_inputs(tmp_path)
    staged._base._write_json(
        Path(parent["candidate_dir"]) / "02_relax" / "meta.json",
        {
            "backend": "grace",
            "model": "GRACE-1L-OMAT",
            "task": "",
            "energy_relaxed_eV": -456.25,
        },
    )
    monkeypatch.setattr(
        staged,
        "structure_energy_with_calculator",
        lambda structure, calculator: (_ for _ in ()).throw(
            AssertionError("matching prior energy should be reused")
        ),
    )

    row = staged._search_parent_reference_row(
        parent, structure, fields, cfg, mc_cfg, object(), "search-fingerprint"
    )

    assert row["mc_search_energy_sp_total_eV"] == -456.25
    assert row["mc_search_energy_source"].endswith("02_relax/meta.json")


def test_merge_parent_reference_does_not_modify_defective_archive_rows():
    old_parent = {"parent_id": "A/candidate_001", "n_vacancies": 0, "old": True}
    defective = {
        "parent_id": "A/candidate_001",
        "n_vacancies": 2,
        "configuration_id": "mc_000001",
        "mc_search_energy_sp_total_eV": -10.0,
    }
    replacement = {
        "parent_id": "A/candidate_001",
        "n_vacancies": 0,
        "configuration_id": "parent_reference",
    }

    merged = staged._merge_parent_reference([old_parent, defective], replacement)

    assert merged == [replacement, defective]


def test_completed_search_backfills_parent_without_rerunning_monte_carlo(
    monkeypatch, tmp_path
):
    parent, structure, fields, _, mc_cfg = _search_reference_inputs(tmp_path)
    parent["symmetry_path"] = Path(parent["candidate_dir"]) / "01_scan" / "POSCAR"
    parent["relaxed_path"] = Path(parent["candidate_dir"]) / "02_relax" / "POSCAR"
    vacancy_root = Path(parent["candidate_dir"]) / staged._base.VACANCY_STAGE_DIR
    vacancy_root.mkdir(parents=True)
    defective = {
        "parent_id": parent["parent_id"],
        "n_vacancies": 1,
        "configuration_id": "mc_000001",
        "mc_search_energy_sp_total_eV": -100.0,
    }
    staged._base._write_json(
        vacancy_root / "mc_search_meta.json",
        {"status": "complete", "search_fingerprint": "existing-search"},
    )
    staged._base._write_json(vacancy_root / "mc_search_results.json", [defective])

    cfg = SimpleNamespace(
        skip_if_done=True,
        vacancy_species="O",
        oxidation_states={},
        host_oxidation_state=4,
    )
    monkeypatch.setattr(
        staged,
        "_expanded_parent",
        lambda parent, cfg: (structure, structure, {0: 0, 1: 1, 2: 2}),
    )
    monkeypatch.setattr(
        staged, "_composition_fields", lambda parent, structure, cfg: (fields, {})
    )
    monkeypatch.setattr(staged._base, "reachable_charge_scenarios", lambda *args: [])
    monkeypatch.setattr(staged, "_count_selection", lambda *args: ([1], {}))
    monkeypatch.setattr(staged, "_search_fingerprint", lambda *args: "existing-search")
    parent_row = {
        "parent_id": parent["parent_id"],
        "n_vacancies": 0,
        "configuration_id": "parent_reference",
        "mc_search_energy_sp_total_eV": -110.0,
    }
    monkeypatch.setattr(
        staged, "_search_parent_reference_row", lambda *args: parent_row
    )
    monkeypatch.setattr(
        staged._base,
        "monte_carlo_vacancy_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed vacancy search must not rerun")
        ),
    )

    rows = staged._run_search_parent(parent, cfg, mc_cfg, (1,), object())

    assert rows == [parent_row, defective]
    assert staged._base._load_json(vacancy_root / "mc_search_meta.json")[
        "parent_reference_included"
    ] is True

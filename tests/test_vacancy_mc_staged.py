from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import dopingflow.vacancy_mc_staged as staged
import dopingflow.vacancy_parent_selection_extensions as parent_selection
from dopingflow.vacancy_parent_selection_extensions import pick_parents


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

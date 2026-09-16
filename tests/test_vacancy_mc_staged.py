from pathlib import Path
from types import SimpleNamespace

import dopingflow.vacancy_mc_staged as staged


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

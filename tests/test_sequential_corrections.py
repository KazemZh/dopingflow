import csv

import pytest

import dopingflow.sequential as sequential_module


def test_sequential_fits_correction_once_before_all_composition_steps(
    tmp_path,
    monkeypatch,
):
    config = {
        "doping": {
            "mode": "explicit",
            "compositions": [{"Sb": 5.0}, {"Sb": 10.0}],
        },
        "sequential": {
            "mode": "recompute_energies",
            "outdir": "sequential_structures",
        },
        "structure": {"outdir": "random_structures"},
        "energy_correction": {"enabled": True},
    }
    sequential_root = tmp_path / "sequential_structures"
    for step in ("step_001_Sb5", "step_002_Sb10"):
        (sequential_root / step / "random_structures").mkdir(parents=True)
    stale_step = sequential_root / "step_999_obsolete"
    stale_step.mkdir(parents=True)
    (stale_step / "results_database.csv").write_text(
        "candidate,E_form_eV_total\nobsolete,999.0\n",
        encoding="utf-8",
    )

    events: list[str] = []
    stage_settings: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        sequential_module,
        "run_corrections_fit",
        lambda *args, **kwargs: events.append("fit"),
    )
    def fake_formation(cfg, *args, **kwargs):
        events.append("formation")
        stage_settings.append(
            ("formation", bool(cfg["formation"]["skip_if_done"]))
        )

    monkeypatch.setattr(sequential_module, "run_formation", fake_formation)

    def fake_collect(cfg, *args, **kwargs):
        events.append("collect")
        stage_settings.append(("database", bool(cfg["database"]["skip_if_done"])))
        pct = cfg["doping"]["compositions"][0]["Sb"]
        (tmp_path / "results_database.csv").write_text(
            f"candidate,E_form_eV_total\ncandidate_{pct:g},{pct:g}\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(sequential_module, "run_collect", fake_collect)

    sequential_module.run_sequential(config, tmp_path)

    assert events == [
        "fit",
        "formation",
        "collect",
        "formation",
        "collect",
    ]
    assert stage_settings == [
        ("formation", False),
        ("database", False),
        ("formation", False),
        ("database", False),
    ]
    with (tmp_path / "results_database.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["candidate"] for row in rows] == ["candidate_5", "candidate_10"]
    assert [row["sequential_step"] for row in rows] == [
        "step_001_Sb5",
        "step_002_Sb10",
    ]


def test_sequential_merge_rejects_mixed_model_selection_runs(tmp_path):
    paths = []
    for index, selection_hash in enumerate(("selection-a", "selection-b"), start=1):
        step = tmp_path / "sequential" / f"step_{index:03d}"
        step.mkdir(parents=True)
        path = step / "results_database.csv"
        path.write_text(
            "candidate,correction_model_family,correction_selection_run_hash\n"
            f"candidate_{index},m1,{selection_hash}\n",
            encoding="utf-8",
        )
        paths.append(path)

    with pytest.raises(ValueError, match="mixed correction provenance"):
        sequential_module._merge_step_databases(
            tmp_path / "sequential",
            tmp_path,
            {},
            step_databases=paths,
        )

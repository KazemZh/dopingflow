import csv
import json
from types import SimpleNamespace

import dopingflow.collect as collect_module
from dopingflow.collect import run_collect


def _write_candidate(root, *, fit_id="fit-new"):
    folder = root / "random_structures" / "Sb10"
    candidate = folder / "candidate_001"
    (candidate / "02_relax").mkdir(parents=True)
    (candidate / "04_formation").mkdir(parents=True)
    (folder / "selected_candidates.txt").write_text(
        "candidate_001\n",
        encoding="utf-8",
    )
    (candidate / "02_relax" / "meta.json").write_text(
        json.dumps({"energy_relaxed_eV": -10.0}),
        encoding="utf-8",
    )

    correction = {
        "enabled": True,
        "fit_id": fit_id,
        "method": "synthetic",
        "parameter_set": fit_id,
        "experimental_dataset": "synthetic",
        "backend_signature": {
            "backend": "mace",
            "model": "mh-1",
            "task": "omat_pbe",
        },
        "formation_input_hash": "current-input",
        "applied": True,
        "reason": "balanced_formation_reaction",
    }
    result_correction = {
        "fit_id": fit_id,
        "method": "synthetic",
        "experimental_dataset": "synthetic",
        "applied": True,
    }
    metadata = {
        "reference_mode": "metal",
        "primary_reference_label": "metal",
        "E_form_eV_total": 1.0,
        "reported": {"value": 1.0, "unit": "total_eV"},
        "dopant_counts": {"Sb": 1},
        "n_atoms_supercell": 3,
        "x_dopant": 0.5,
        "mixing": {},
        "energy_correction": correction,
        "energy_correction_eV_total": -0.2,
        "correction_uncertainty_eV_total": 0.1,
        "E_form_corrected_eV_total": 0.8,
        "reported_corrected": {
            "value": 0.8,
            "uncertainty": 0.1,
            "unit": "total_eV",
        },
        "reference_results": {
            "metal": {
                "E_form_eV_total": 1.0,
                "E_form_eV_per_atom": 1.0 / 3.0,
                "E_form_eV_per_cation": 0.5,
                "E_form_eV_per_dopant": 1.0,
                "formation_energy_raw_eV_total": 1.0,
                "energy_correction_eV_total": -0.2,
                "correction_uncertainty_eV_total": 0.1,
                "E_form_corrected_eV_total": 0.8,
                "E_form_corrected_eV_per_atom": 0.8 / 3.0,
                "E_form_corrected_eV_per_cation": 0.4,
                "E_form_corrected_eV_per_dopant": 0.8,
                "energy_correction": result_correction,
                "mixing": {},
            }
        },
    }
    (candidate / "04_formation" / "meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def _config(*, enabled):
    return {
        "structure": {"outdir": "random_structures"},
        "database": {"skip_if_done": True},
        "energy_correction": {
            "enabled": enabled,
            "correction_terms": ["oxide"],
        },
    }


def test_enabled_correction_rebuilds_existing_database(tmp_path, monkeypatch):
    _write_candidate(tmp_path, fit_id="fit-new")
    reference = tmp_path / "reference_structures" / "reference_energies.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        collect_module,
        "load_active_correction_model",
        lambda *args: SimpleNamespace(fit_id="fit-new"),
    )
    monkeypatch.setattr(
        collect_module,
        "_formation_correction_input_hash",
        lambda *args: "current-input",
    )
    output = tmp_path / "results_database.csv"
    output.write_text(
        "candidate,correction_fit_id\nold,fit-old\n",
        encoding="utf-8",
    )

    result = run_collect(_config(enabled=True), tmp_path)
    with result.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["candidate"] == "candidate_001"
    assert row["correction_fit_id"] == "fit-new"
    assert float(row["E_form_corrected_eV_total"]) == 0.8


def test_enabled_collection_rejects_stale_formation_fit(tmp_path, monkeypatch):
    _write_candidate(tmp_path, fit_id="fit-old")
    reference = tmp_path / "reference_structures" / "reference_energies.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        collect_module,
        "load_active_correction_model",
        lambda *args: SimpleNamespace(fit_id="fit-new"),
    )

    try:
        run_collect(_config(enabled=True), tmp_path)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:  # pragma: no cover - explicit fail keeps this dependency-free
        raise AssertionError("stale corrected formation metadata was accepted")


def test_disabling_correction_rebuilds_and_removes_stale_corrected_columns(tmp_path):
    # Deliberately retain stale corrected formation metadata.  The disabled
    # collector must still produce a legacy/raw-only database.
    _write_candidate(tmp_path)
    output = tmp_path / "results_database.csv"
    output.write_text(
        "candidate,correction_fit_id,E_form_corrected_eV_total\n"
        "old,fit-old,99\n",
        encoding="utf-8",
    )

    result = run_collect(_config(enabled=False), tmp_path)
    with result.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        fieldnames = reader.fieldnames or []

    assert row["candidate"] == "candidate_001"
    assert float(row["E_form_eV_total"]) == 1.0
    assert "correction_fit_id" not in fieldnames
    assert "E_form_corrected_eV_total" not in fieldnames
    assert "energy_correction_eV_total__metal" not in fieldnames
    assert "E_form_eV_total__metal" in fieldnames


def test_disabled_raw_database_still_honors_skip_if_done(tmp_path):
    output = tmp_path / "results_database.csv"
    output.write_text("candidate,E_form_eV_total\nkept,1.25\n", encoding="utf-8")

    result = run_collect(_config(enabled=False), tmp_path)

    assert result == output
    assert output.read_text(encoding="utf-8") == (
        "candidate,E_form_eV_total\nkept,1.25\n"
    )

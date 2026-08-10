import csv
import json

import pytest
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Composition, Lattice, Structure

import dopingflow.phase_diagram as phase_module
from dopingflow.corrections import CorrectionModel


def _model(
    *,
    terms=("oxide",),
    coefficients=(-0.5,),
    covariance=((0.01,),),
    signature_updates=None,
):
    signature = {
        "backend": "mace",
        "model": "mh-1",
        "task": "omat_pbe",
        "optimizer": "bfgs",
        "fmax_eV_per_A": 0.02,
        "max_steps": 300,
        "backend_package": "mace-torch",
        "backend_package_version": "synthetic-1",
    }
    signature.update(signature_updates or {})
    return CorrectionModel(
        schema_version=1,
        method="synthetic",
        fit_id="fit-hull",
        backend_signature=signature,
        correction_terms=tuple(terms),
        coefficients_eV_per_term=tuple(coefficients),
        covariance_eV2=tuple(tuple(row) for row in covariance),
        coefficient_uncertainties_eV_per_term=tuple(
            covariance[index][index] ** 0.5 for index in range(len(terms))
        ),
        experimental_dataset="synthetic",
        experimental_dataset_version="1",
        fit_input_hash="hash",
        units={"coefficient": "eV_per_term_atom"},
        calibration_formulas=("Li2O", "MgO"),
        fit_metrics={"rmse_eV_per_atom": 0.0},
    )


def _write_structure(path, species, coordinates):
    path.parent.mkdir(parents=True, exist_ok=True)
    structure = Structure(Lattice.cubic(6.0), species, coordinates)
    structure.to(fmt="poscar", filename=str(path))
    return path


def _entries(tmp_path):
    sno = _write_structure(
        tmp_path / "SnO.POSCAR",
        ["Sn", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    sno2 = _write_structure(
        tmp_path / "SnO2.POSCAR",
        ["Sn", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    references = [
        PDEntry(Composition("Sn"), 0.0, name="Sn", attribute={"entry_kind": "reference"}),
        PDEntry(Composition("O2"), 0.0, name="O2", attribute={"entry_kind": "reference"}),
        PDEntry(
            Composition("SnO"),
            -1.0,
            name="SnO",
            attribute={"entry_kind": "reference", "structure_path": str(sno)},
        ),
    ]
    candidate_dir = tmp_path / "SnO2_candidates" / "candidate_001"
    candidate = PDEntry(
        Composition("SnO2"),
        -0.8,
        name="SnO2_candidates/candidate_001",
        attribute={
            "entry_kind": "candidate",
            "structure_path": str(sno2),
            "backend": "mace",
            "model": "mh-1",
            "task": "omat_pbe",
            "backend_package": "mace-torch",
            "backend_package_version": "synthetic-1",
            "relaxed_poscar_sha256": phase_module._file_sha256(sno2),
            "optimizer": "bfgs",
            "fmax_target_eV_per_A": 0.02,
            "max_steps": 300,
            "converged": True,
        },
    )
    return references, [(candidate.name, candidate_dir, candidate)]


def _run(
    tmp_path,
    monkeypatch,
    references,
    candidates,
    *,
    model=None,
    skip_if_done=False,
    allow_legacy=False,
):
    active_model = model or _model()
    ref_path = tmp_path / phase_module.REF_JSON
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(
        phase_module,
        "load_active_correction_model",
        lambda *args: active_model,
    )
    monkeypatch.setattr(phase_module, "_reference_entries_from_ref", lambda ref: references)
    monkeypatch.setattr(
        phase_module,
        "_candidate_entries_from_database",
        lambda root: candidates,
    )
    return phase_module.run_phase_diagram(
        {
            "energy_correction": {
                "enabled": True,
                "correction_terms": list(active_model.correction_terms),
                "allow_element_terms": any(
                    term.startswith("element:")
                    for term in active_model.correction_terms
                ),
                "allow_legacy_candidate_provenance": allow_legacy,
            },
            "phase_diagram": {"skip_if_done": skip_if_done},
        },
        tmp_path,
    )


def test_corrected_hull_is_rebuilt_and_can_change_energy_above_hull(tmp_path, monkeypatch):
    output = _run(tmp_path, monkeypatch, *_entries(tmp_path))
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["energy_above_hull_raw_eV_per_atom"]) > 0
    assert float(row["energy_above_hull_corrected_eV_per_atom"]) == pytest.approx(0.0)
    assert float(row["energy_above_hull_correction_eV_per_atom"]) == pytest.approx(
        -float(row["energy_above_hull_raw_eV_per_atom"])
    )
    assert float(row["energy_above_hull_parameter_shift_eV_per_atom"]) == pytest.approx(
        0.0
    )
    assert float(row["energy_total_eV"]) == pytest.approx(-0.8)
    assert float(row["energy_corrected_eV"]) == pytest.approx(-1.8)


def test_corrected_hull_retains_raw_aliases_and_provenance(tmp_path, monkeypatch):
    output = _run(tmp_path, monkeypatch, *_entries(tmp_path))
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["energy_above_hull_eV_per_atom"] == row[
        "energy_above_hull_raw_eV_per_atom"
    ]
    assert row["correction_fit_id"] == "fit-hull"
    assert row["correction_method"] == "synthetic"
    assert row["experimental_dataset"] == "synthetic"


def test_corrected_hull_rejects_an_uncorrectable_competing_phase(tmp_path, monkeypatch):
    references, candidates = _entries(tmp_path)
    references[2] = PDEntry(Composition("SnO"), -1.0, name="SnO")
    with pytest.raises(ValueError, match="complete corrected phase diagram"):
        _run(tmp_path, monkeypatch, references, candidates)


def test_corrected_hull_rejects_candidate_backend_mismatch(tmp_path, monkeypatch):
    references, candidates = _entries(tmp_path)
    candidates[0][2].attribute["model"] = "different"
    with pytest.raises(ValueError, match="backend incompatible"):
        _run(tmp_path, monkeypatch, references, candidates)


def test_corrected_hull_explicitly_accepts_legacy_candidate_provenance(
    tmp_path, monkeypatch
):
    references, candidates = _entries(tmp_path)
    candidate_attributes = candidates[0][2].attribute
    candidate_attributes.pop("relaxed_poscar_sha256")
    candidate_attributes.pop("backend_package")
    candidate_attributes.pop("backend_package_version")
    candidate_attributes["device"] = "cuda"
    candidate_attributes["gpu_id"] = 0
    model = _model(signature_updates={"device": "cpu", "gpu_id": None})

    output = _run(
        tmp_path,
        monkeypatch,
        references,
        candidates,
        model=model,
        allow_legacy=True,
    )
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["candidate_energy_provenance_mode"] == (
        "legacy_explicitly_accepted"
    )
    assert "missing_original_relaxed_poscar_sha256" in row[
        "candidate_energy_provenance_assumptions_json"
    ]
    assert "execution_device_differs" in row[
        "candidate_energy_execution_differences_json"
    ]


def test_corrected_hull_rejects_candidate_structure_changed_after_relaxation(
    tmp_path, monkeypatch
):
    references, candidates = _entries(tmp_path)
    structure_path = candidates[0][2].attribute["structure_path"]
    with open(structure_path, "a", encoding="utf-8") as handle:
        handle.write("\n# tampered after relaxation\n")
    with pytest.raises(ValueError, match="structure changed after energy evaluation"):
        _run(tmp_path, monkeypatch, references, candidates)


def test_corrected_hull_rejects_candidate_backend_version_mismatch(
    tmp_path, monkeypatch
):
    references, candidates = _entries(tmp_path)
    candidates[0][2].attribute["backend_package_version"] = "different"
    with pytest.raises(ValueError, match="backend incompatible"):
        _run(tmp_path, monkeypatch, references, candidates)


def test_corrected_hull_requires_expected_checkpoint_provenance(
    tmp_path, monkeypatch
):
    references, candidates = _entries(tmp_path)
    model = _model(
        signature_updates={"model_checkpoint_sha256": "expected-checkpoint"}
    )
    with pytest.raises(ValueError, match="backend incompatible"):
        _run(
            tmp_path,
            monkeypatch,
            references,
            candidates,
            model=model,
        )


def test_enabled_correction_rebuilds_even_when_existing_fit_id_matches(
    tmp_path, monkeypatch
):
    output = tmp_path / phase_module.OUT_CSV
    output.write_text(
        "candidate,stable_corrected\n"
        "stale,True\n",
        encoding="utf-8",
    )
    result = _run(
        tmp_path,
        monkeypatch,
        *_entries(tmp_path),
        skip_if_done=True,
    )
    with result.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["candidate"] == "candidate_001"
    assert float(row["energy_total_eV"]) == pytest.approx(-0.8)


def test_disabling_correction_rebuilds_a_formerly_corrected_output(
    tmp_path, monkeypatch
):
    output = tmp_path / phase_module.OUT_CSV
    output.write_text(
        "candidate,correction_fit_id,energy_corrected_eV\n"
        "stale,fit-hull,999\n",
        encoding="utf-8",
    )
    ref_path = tmp_path / phase_module.REF_JSON
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps({}), encoding="utf-8")
    references, candidates = _entries(tmp_path)
    monkeypatch.setattr(
        phase_module,
        "_reference_entries_from_ref",
        lambda ref: references,
    )
    monkeypatch.setattr(
        phase_module,
        "_candidate_entries_from_database",
        lambda root: candidates,
    )

    result = phase_module.run_phase_diagram(
        {
            "energy_correction": {"enabled": False},
            "phase_diagram": {"skip_if_done": True},
        },
        tmp_path,
    )
    with result.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        assert "correction_fit_id" not in (reader.fieldnames or [])
        assert "energy_corrected_eV" not in (reader.fieldnames or [])
        assert "stable_corrected" not in (reader.fieldnames or [])
    assert row["candidate"] == "candidate_001"


def test_energy_above_hull_uncertainty_uses_full_covariance_q_vector(
    tmp_path, monkeypatch
):
    candidate_structure = _write_structure(
        tmp_path / "positive_SnO2.POSCAR",
        ["Sn", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    references = [
        PDEntry(
            Composition("Sn"),
            0.0,
            name="Sn",
            attribute={"entry_kind": "reference"},
        ),
        PDEntry(
            Composition("O2"),
            0.0,
            name="O2",
            attribute={"entry_kind": "reference"},
        ),
    ]
    candidate_dir = tmp_path / "positive_candidates" / "candidate_001"
    candidate = PDEntry(
        Composition("SnO2"),
        0.3,
        name="positive_candidates/candidate_001",
        attribute={
            "entry_kind": "candidate",
            "structure_path": str(candidate_structure),
            "backend": "mace",
            "model": "mh-1",
            "task": "omat_pbe",
            "backend_package": "mace-torch",
            "backend_package_version": "synthetic-1",
            "relaxed_poscar_sha256": phase_module._file_sha256(candidate_structure),
            "optimizer": "bfgs",
            "fmax_target_eV_per_A": 0.02,
            "max_steps": 300,
            "converged": True,
        },
    )
    model = _model(
        terms=("oxide", "element:Sn"),
        coefficients=(0.2, 0.1),
        covariance=((0.04, 0.015), (0.015, 0.09)),
    )
    output = _run(
        tmp_path,
        monkeypatch,
        references,
        [(candidate.name, candidate_dir, candidate)],
        model=model,
    )
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    q = [2.0 / 3.0, 1.0 / 3.0]
    expected_variance = (
        q[0] * q[0] * 0.04
        + 2.0 * q[0] * q[1] * 0.015
        + q[1] * q[1] * 0.09
    )
    assert json.loads(
        row["energy_above_hull_correction_q_vector_per_atom_json"]
    ) == pytest.approx(q)
    assert float(row["energy_above_hull_correction_eV_per_atom"]) == pytest.approx(
        q[0] * 0.2 + q[1] * 0.1
    )
    assert float(row["energy_above_hull_parameter_shift_eV_per_atom"]) == pytest.approx(
        q[0] * 0.2 + q[1] * 0.1
    )
    assert float(
        row["energy_above_hull_correction_uncertainty_eV_per_atom"]
    ) == pytest.approx(expected_variance**0.5)
    assert "q^T covariance q" in row["energy_above_hull_correction_provenance"]


def test_missing_experimental_target_value_does_not_block_model_application(tmp_path):
    structure = Structure(
        Lattice.cubic(20.0),
        ["Sn"] * 5 + ["O"] * 6,
        [
            [0.05, 0.05, 0.05],
            [0.25, 0.05, 0.05],
            [0.45, 0.05, 0.05],
            [0.65, 0.05, 0.05],
            [0.85, 0.05, 0.05],
            [0.10, 0.40, 0.40],
            [0.30, 0.40, 0.40],
            [0.50, 0.40, 0.40],
            [0.70, 0.40, 0.40],
            [0.20, 0.75, 0.75],
            [0.60, 0.75, 0.75],
        ],
    )
    application = phase_module.apply_energy_correction(
        _model(),
        structure.composition,
        structure=structure,
    )
    assert application.correction_eV == pytest.approx(-3.0)

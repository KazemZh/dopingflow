import csv
import json
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

import dopingflow.correction_calibration as calibration_module
from dopingflow.correction_calibration import (
    load_calibration_manifest,
    run_corrections_fit,
)
from dopingflow.corrections import (
    content_hash,
    load_active_correction_model,
    load_correction_model,
    model_path,
    parse_correction_config,
)
from dopingflow.refs import _file_sha256, _parse_ref_config, _relaxation_signature


def _structure(species):
    coordinates = {
        2: [[0, 0, 0], [0.5, 0.5, 0.5]],
        3: [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    }[len(species)]
    return Structure(Lattice.cubic(5.0), species, coordinates)


def _write_fixture(tmp_path):
    correction_dir = tmp_path / "reference_structures" / "corrections"
    correction_dir.mkdir(parents=True)
    li2o = correction_dir / "Li2O.POSCAR"
    mgo = correction_dir / "MgO.POSCAR"
    li = correction_dir / "Li.POSCAR"
    mg = correction_dir / "Mg.POSCAR"
    o2 = correction_dir / "O2.POSCAR"
    _structure(["Li", "Li", "O"]).to(fmt="poscar", filename=str(li2o))
    _structure(["Mg", "O"]).to(fmt="poscar", filename=str(mgo))
    _structure(["Li", "Li"]).to(fmt="poscar", filename=str(li))
    _structure(["Mg", "Mg"]).to(fmt="poscar", filename=str(mg))
    _structure(["O", "O"]).to(fmt="poscar", filename=str(o2))

    experimental = correction_dir / "experimental.csv"
    manifest = correction_dir / "calibration_manifest.csv"
    config = {
        "references": {
            "reference_mode": "metal",
            "host": "Li2O",
            "host_dir": "reference_structures/corrections",
            "supercell": [1, 1, 1],
            "metal_ref": ["Li", "Mg", "O"],
            "metals_dir": "reference_structures/metals",
            "backend": "m3gnet",
            "model": "default",
            "task": "",
            "optimizer": "bfgs",
            "fmax": 0.02,
            "max_steps": 300,
        },
        "energy_correction": {
            "enabled": True,
            "experimental_source": "custom",
            "experimental_data": str(experimental.relative_to(tmp_path)),
            "calibration_manifest": str(manifest.relative_to(tmp_path)),
            "correction_terms": ["oxide"],
            "max_calculated_e_above_hull_eV_per_atom": 0.1,
        },
    }
    relaxation_signature = _relaxation_signature(
        _parse_ref_config(config, tmp_path),
        root=tmp_path,
    )
    backend_version = relaxation_signature["backend_package_version"]
    settings_hash = content_hash(relaxation_signature)

    with experimental.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "formula",
                "formation_enthalpy",
                "uncertainty",
                "phase",
                "temperature",
                "units",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "formula": "Li2O",
                    "formation_enthalpy": -3.2 / 3.0,
                    "uncertainty": 0.01,
                    "phase": "synthetic-phase",
                    "temperature": "298 K",
                    "units": "eV/atom",
                    "source": "synthetic",
                },
                {
                    "formula": "MgO",
                    "formation_enthalpy": -2.2 / 2.0,
                    "uncertainty": 0.02,
                    "phase": "synthetic-phase",
                    "temperature": "298 K",
                    "units": "eV/atom",
                    "source": "synthetic",
                },
            ]
        )

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "formula",
                "phase",
                "structure_path",
                "energy_total_eV",
                "e_above_hull_eV_per_atom",
                "e_above_hull_provenance",
                "e_above_hull_backend",
                "e_above_hull_model",
                "e_above_hull_task",
                "e_above_hull_backend_version",
                "e_above_hull_calculation_settings_hash",
                "backend",
                "model",
                "task",
                "backend_version",
                "calculation_settings",
                "calculation_settings_hash",
                "converged",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "formula": "Li2O",
                    "phase": "synthetic-phase",
                    "structure_path": li2o.name,
                    "energy_total_eV": -3.0,
                    "e_above_hull_eV_per_atom": 0.0,
                    "e_above_hull_provenance": "same-backend synthetic hull",
                    "e_above_hull_backend": "m3gnet",
                    "e_above_hull_model": "default",
                    "e_above_hull_task": "",
                    "e_above_hull_backend_version": backend_version,
                    "e_above_hull_calculation_settings_hash": settings_hash,
                    "backend": "m3gnet",
                    "model": "default",
                    "task": "",
                    "backend_version": backend_version,
                    "calculation_settings": "pre-relaxed synthetic fixture",
                    "calculation_settings_hash": settings_hash,
                    "converged": True,
                },
                {
                    "formula": "MgO",
                    "phase": "synthetic-phase",
                    "structure_path": mgo.name,
                    "energy_total_eV": -2.0,
                    "e_above_hull_eV_per_atom": 0.0,
                    "e_above_hull_provenance": "same-backend synthetic hull",
                    "e_above_hull_backend": "m3gnet",
                    "e_above_hull_model": "default",
                    "e_above_hull_task": "",
                    "e_above_hull_backend_version": backend_version,
                    "e_above_hull_calculation_settings_hash": settings_hash,
                    "backend": "m3gnet",
                    "model": "default",
                    "task": "",
                    "backend_version": backend_version,
                    "calculation_settings": "pre-relaxed synthetic fixture",
                    "calculation_settings_hash": settings_hash,
                    "converged": True,
                },
            ]
        )

    reference_dir = tmp_path / "reference_structures"
    reference_data = {
        "backend": "m3gnet",
        "model": "default",
        "task": "",
        "optimizer": "bfgs",
        "fmax": 0.02,
        "max_steps": 300,
        "device": "cpu",
        "gpu_id": None,
        "backend_package": relaxation_signature["backend_package"],
        "backend_package_version": backend_version,
        "host": {
            "unit_converged": True,
            "supercell_converged": True,
            "relaxation_signature": relaxation_signature,
            "relaxed_unit_poscar": str(li2o),
            "relaxed_supercell_poscar": str(li2o),
            "relaxed_unit_sha256": _file_sha256(li2o),
            "relaxed_supercell_sha256": _file_sha256(li2o),
        },
        "references": {
            "Li": {
                "type": "metal",
                "E_per_atom_eV": 0.0,
                "converged": True,
                "relaxation_signature": relaxation_signature,
                "relaxed_poscar": str(li),
                "relaxed_sha256": _file_sha256(li),
            },
            "Mg": {
                "type": "metal",
                "E_per_atom_eV": 0.0,
                "converged": True,
                "relaxation_signature": relaxation_signature,
                "relaxed_poscar": str(mg),
                "relaxed_sha256": _file_sha256(mg),
            },
            "O2": {
                "type": "gas",
                "E_per_atom_eV": 0.0,
                "converged": True,
                "relaxation_signature": relaxation_signature,
                "relaxed_poscar": str(o2),
                "relaxed_sha256": _file_sha256(o2),
            },
        },
    }
    (reference_dir / "reference_energies.json").write_text(
        json.dumps(reference_data), encoding="utf-8"
    )

    return config, manifest


def _rewrite_csv(path, **updates):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(updates)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rejected_records(tmp_path):
    paths = list(
        (tmp_path / "reference_structures" / "corrections").glob(
            "*/calibration_rejected.json"
        )
    )
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_full_precomputed_calibration_fit_writes_reproducibility_artifacts(tmp_path):
    config, _ = _write_fixture(tmp_path)
    output = run_corrections_fit(config, tmp_path)
    assert output is not None and output.exists()
    model = load_correction_model(output)
    assert model.coefficients_eV_per_term == pytest.approx((-0.2,))
    for name in (
        "correction_fit_report.json",
        "experimental_calibration_used.json",
        "correction_metadata.json",
        "calibration_calculated_energies.json",
    ):
        assert (output.parent / name).exists()


def test_fit_cache_reuses_exact_input_hash(tmp_path):
    config, _ = _write_fixture(tmp_path)
    first = run_corrections_fit(config, tmp_path)
    first_text = first.read_text(encoding="utf-8")
    second = run_corrections_fit(config, tmp_path)
    assert second == first
    assert second.read_text(encoding="utf-8") == first_text


def test_generic_phase_is_rejected_unless_explicitly_overridden(tmp_path):
    config, _ = _write_fixture(tmp_path)
    experimental = tmp_path / config["energy_correction"]["experimental_data"]
    _rewrite_csv(experimental, phase="cryst")

    with pytest.raises(ValueError, match="No calibration records"):
        run_corrections_fit(config, tmp_path)
    assert {
        record["reason"] for record in _rejected_records(tmp_path)
    } == {"phase_not_verified"}

    config["energy_correction"]["allow_phase_mismatch"] = True
    output = run_corrections_fit(config, tmp_path)
    report = json.loads(
        (output.parent / "correction_fit_report.json").read_text(encoding="utf-8")
    )
    assert {
        row["phase_match_note"] for row in report["accepted_records"]
    } == {"phase_mismatch_explicitly_allowed"}


def test_eah_backend_provenance_mismatch_is_rejected(tmp_path):
    config, manifest = _write_fixture(tmp_path)
    _rewrite_csv(manifest, e_above_hull_backend="mace")
    with pytest.raises(ValueError, match="No calibration records"):
        run_corrections_fit(config, tmp_path)
    assert {
        record["reason"] for record in _rejected_records(tmp_path)
    } == {"calculated_e_above_hull_backend_mismatch"}


def test_nonconverged_automatic_calibration_is_rejected(tmp_path, monkeypatch):
    config, manifest = _write_fixture(tmp_path)
    _rewrite_csv(manifest, energy_total_eV="")

    def fake_relax(structure, _ref_config):
        return structure.copy(), -1.0, 3, 0.5, False

    monkeypatch.setattr(
        calibration_module,
        "_relax_structure_and_energy",
        fake_relax,
    )
    with pytest.raises(ValueError, match="No calibration records"):
        run_corrections_fit(config, tmp_path)
    assert {
        record["reason"] for record in _rejected_records(tmp_path)
    } == {"calibration_relaxation_not_converged"}


def test_manifest_formula_must_match_structure(tmp_path):
    config, manifest = _write_fixture(tmp_path)
    text = manifest.read_text(encoding="utf-8").replace(
        "Li2O,synthetic-phase",
        "Na2O,synthetic-phase",
    )
    manifest.write_text(text, encoding="utf-8")
    parsed = parse_correction_config(config, tmp_path)
    with pytest.raises(ValueError, match="does not match structure"):
        load_calibration_manifest(parsed, tmp_path)


def test_precomputed_calibration_requires_backend_provenance(tmp_path):
    config, manifest = _write_fixture(tmp_path)
    run_corrections_fit(config, tmp_path)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rows[0]["backend_version"] = ""
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="backend_version"):
        run_corrections_fit(config, tmp_path)


def test_precomputed_calibration_requires_convergence_provenance(tmp_path):
    config, manifest = _write_fixture(tmp_path)
    _rewrite_csv(manifest, converged="false")
    with pytest.raises(ValueError, match="converged=true"):
        run_corrections_fit(config, tmp_path)


def test_correction_fit_rejects_unconverged_reference_energy(tmp_path):
    config, _ = _write_fixture(tmp_path)
    reference_path = tmp_path / "reference_structures" / "reference_energies.json"
    reference_data = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_data["references"]["Mg"]["converged"] = False
    reference_path.write_text(json.dumps(reference_data), encoding="utf-8")

    with pytest.raises(ValueError, match="converged reference"):
        run_corrections_fit(config, tmp_path)


def test_correction_fit_rejects_changed_elemental_reference_structure(tmp_path):
    config, _ = _write_fixture(tmp_path)
    reference_path = tmp_path / "reference_structures" / "reference_energies.json"
    reference_data = json.loads(reference_path.read_text(encoding="utf-8"))
    li_path = reference_data["references"]["Li"]["relaxed_poscar"]
    with open(li_path, "a", encoding="utf-8") as handle:
        handle.write("\n# changed after reference energy evaluation\n")

    with pytest.raises(ValueError, match="changed for reference Li"):
        run_corrections_fit(config, tmp_path)


def test_correction_fit_rejects_mixed_reference_relaxation_signatures(tmp_path):
    config, _ = _write_fixture(tmp_path)
    reference_path = tmp_path / "reference_structures" / "reference_energies.json"
    reference_data = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_data["references"]["Mg"]["relaxation_signature"]["model"] = (
        "different-model"
    )
    reference_path.write_text(json.dumps(reference_data), encoding="utf-8")

    with pytest.raises(ValueError, match="relaxation signature inconsistent"):
        run_corrections_fit(config, tmp_path)


def test_model_path_is_backend_specific(tmp_path):
    config, _ = _write_fixture(tmp_path)
    output = run_corrections_fit(config, tmp_path)
    model = load_correction_model(output)
    assert output == model_path(tmp_path, model.backend_signature)


def test_active_model_is_invalidated_when_experimental_input_changes(tmp_path):
    config, _ = _write_fixture(tmp_path)
    run_corrections_fit(config, tmp_path)
    reference_path = tmp_path / "reference_structures" / "reference_energies.json"
    reference_data = json.loads(reference_path.read_text(encoding="utf-8"))
    assert load_active_correction_model(config, tmp_path, reference_data) is not None

    experimental = tmp_path / config["energy_correction"]["experimental_data"]
    experimental.write_text(
        experimental.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inputs/configuration changed"):
        load_active_correction_model(config, tmp_path, reference_data)


def test_m0_family_fit_publishes_selection_and_candidate_artifacts(
    tmp_path,
    monkeypatch,
):
    config, _ = _write_fixture(tmp_path)
    config["energy_correction"].update(
        {
            "model_family": "m0",
            "m1_elements": "workflow",
        }
    )
    synthetic_selection = SimpleNamespace(
        selected_family="M0",
        m0_model=SimpleNamespace(terms=("oxide",)),
        m1_model=None,
        report={
            "selected_family": "M0",
            "selection_reason": "synthetic integration fixture",
            "m1_unavailable_reason": "synthetic integration fixture",
        },
    )
    monkeypatch.setattr(
        calibration_module,
        "select_correction_model_family",
        lambda *args, **kwargs: synthetic_selection,
    )

    output = run_corrections_fit(config, tmp_path)
    assert output is not None
    model = load_correction_model(output)
    assert model.model_family == "m0"
    assert model.target_elements == ("Li",)
    assert model.selection_run_hash
    assert (output.parent / "correction_model_selection.json").is_file()
    assert (output.parent / "candidate_models" / "m0.json").is_file()
    assert (
        output.parent / "candidate_models" / "m0_fit_report.json"
    ).is_file()

    reference_data = json.loads(
        (tmp_path / "reference_structures" / "reference_energies.json").read_text(
            encoding="utf-8"
        )
    )
    active = load_active_correction_model(config, tmp_path, reference_data)
    assert active is not None and active.fit_id == model.fit_id


def test_deferred_same_backend_hull_supports_ternary_oxides():
    provisional = [
        {
            "calculated": {
                "reduced_formula": "MnO",
                "cache_key": "mno",
                "e_above_hull_eV_per_atom": None,
            },
            "calculated_formation": -2.0,
        },
        {
            "calculated": {
                "reduced_formula": "TiO2",
                "cache_key": "tio2",
                "e_above_hull_eV_per_atom": None,
            },
            "calculated_formation": -3.0,
        },
        {
            "calculated": {
                "reduced_formula": "MnTiO3",
                "cache_key": "mntio3",
                "e_above_hull_eV_per_atom": None,
            },
            "calculated_formation": -4.0,
        },
    ]
    calibration_module._assign_same_backend_hulls(
        provisional,
        {"references": {}},
        {"Mn": 0.0, "Ti": 0.0, "O": 0.0},
    )
    assert provisional[0]["calculated"]["e_above_hull_eV_per_atom"] == pytest.approx(
        0.0
    )
    assert provisional[1]["calculated"]["e_above_hull_eV_per_atom"] == pytest.approx(
        0.0
    )
    assert provisional[2]["calculated"]["e_above_hull_eV_per_atom"] == pytest.approx(
        0.2
    )

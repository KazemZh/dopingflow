import csv
import sys
import types
from dataclasses import replace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from dopingflow.corrections import (
    CorrectionModel,
    ExperimentalRecord,
    apply_energy_correction,
    backend_signature_from_reference,
    combine_feature_vectors,
    evaluate_feature_vector,
    feature_vector,
    fit_linear_correction_model,
    impute_missing_uncertainties,
    load_correction_model,
    load_custom_experimental_dataset,
    load_kingsbury_dataset,
    merge_experimental_datasets,
    parse_correction_config,
    save_correction_model,
    validate_backend_compatibility,
    validate_candidate_energy_provenance,
)


def _model(*, terms=("oxide",), coefficients=(-0.2,), covariance=((0.01,),)):
    return CorrectionModel(
        schema_version=1,
        method="synthetic_weighted_linear",
        fit_id="fit-123",
        backend_signature={
            "backend": "mace",
            "model": "mh-1",
            "task": "omat_pbe",
            "optimizer": "bfgs",
            "fmax_eV_per_A": 0.02,
            "max_steps": 300,
        },
        correction_terms=tuple(terms),
        coefficients_eV_per_term=tuple(coefficients),
        covariance_eV2=tuple(tuple(row) for row in covariance),
        coefficient_uncertainties_eV_per_term=tuple(
            float(np.sqrt(covariance[index][index]))
            for index in range(len(terms))
        ),
        experimental_dataset="synthetic",
        experimental_dataset_version="1",
        fit_input_hash="hash",
        units={"coefficient": "eV_per_term_atom"},
        calibration_formulas=("Li2O", "MgO"),
        fit_metrics={"rmse_eV_per_atom": 0.0},
    )


def _ordinary_oxide_structure(formula="SnO2"):
    if formula == "SnO2":
        return Structure(
            Lattice.cubic(5.0),
            ["Sn", "O", "O"],
            [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        )
    if formula == "Li2O":
        return Structure(
            Lattice.cubic(5.0),
            ["Li", "Li", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )
    raise AssertionError(formula)


def test_correction_absent_is_disabled_and_backward_compatible(tmp_path):
    config = parse_correction_config({}, tmp_path)
    assert config.enabled is False
    assert config.experimental_source == "kingsbury"


def test_disabled_custom_source_does_not_require_a_file(tmp_path):
    config = parse_correction_config(
        {"energy_correction": {"enabled": False, "experimental_source": "custom"}},
        tmp_path,
    )
    assert config.enabled is False


def test_disabled_malformed_correction_section_is_ignored(tmp_path):
    config = parse_correction_config(
        {
            "energy_correction": {
                "enabled": False,
                "experimental_source": "not-a-source",
                "correction_terms": [],
                "exclude_polyanions": "not-an-array",
                "max_relative_experimental_uncertainty": "not-a-number",
                "min_term_support": "not-an-integer",
                "calibration_manifest": ["not", "a", "path"],
            }
        },
        tmp_path,
    )
    assert config.enabled is False
    assert config.experimental_source == "kingsbury"
    assert config.correction_terms == ("oxide",)
    assert config.min_term_support == 2


def test_enabled_custom_source_requires_a_file(tmp_path):
    with pytest.raises(ValueError, match="experimental_data"):
        parse_correction_config(
            {"energy_correction": {"enabled": True, "experimental_source": "custom"}},
            tmp_path,
        )


def test_element_terms_require_explicit_scientific_opt_in(tmp_path):
    raw = {
        "energy_correction": {
            "enabled": True,
            "correction_terms": ["oxide", "element:Sn"],
        }
    }
    with pytest.raises(ValueError, match="allow_element_terms"):
        parse_correction_config(raw, tmp_path)
    raw["energy_correction"]["allow_element_terms"] = True
    assert parse_correction_config(raw, tmp_path).correction_terms == (
        "oxide",
        "element:Sn",
    )


def test_auto_family_infers_only_host_and_configured_dopants(tmp_path):
    raw = {
        "references": {
            "host": "SnO2",
            "oxides_ref": ["ZrO2"],
        },
        "doping": {
            "mode": "enumerate",
            "host_species": "Sn",
            "dopants": ["Mn", "Nb"],
            "must_include": ["Mn"],
        },
        "energy_correction": {
            "enabled": True,
            "model_family": "auto",
            "m1_elements": "workflow",
            "correction_terms": ["oxide"],
        },
    }
    config = parse_correction_config(raw, tmp_path)
    assert config.target_elements == ("Mn", "Nb", "Sn")
    assert config.m1_elements == ("Mn", "Nb", "Sn")
    assert "Zr" not in config.m1_elements


def test_auto_family_ignores_zero_amount_explicit_dopants(tmp_path):
    raw = {
        "references": {"host": "TiO2"},
        "doping": {
            "mode": "explicit",
            "host_species": "Ti",
            "compositions": [{"Mn": 2, "Nb": 0}, {"Sn": 1}],
        },
        "energy_correction": {
            "enabled": True,
            "model_family": "auto",
            "m1_elements": "workflow",
            "correction_terms": ["oxide"],
        },
    }
    config = parse_correction_config(raw, tmp_path)
    assert config.target_elements == ("Mn", "Sn", "Ti")
    assert config.m1_elements == ("Mn", "Sn", "Ti")


def test_oxide_cation_term_is_zero_outside_ordinary_oxides():
    terms = ("oxide", "oxide_cation:Mn")
    oxide_vector, matched, oxide_type = feature_vector(
        "MnO2",
        terms,
        known_oxide_type="oxide",
    )
    assert oxide_vector == (2.0, 1.0)
    assert matched == terms
    assert oxide_type == "oxide"

    sulfide_vector, matched, oxide_type = feature_vector("MnS", terms)
    assert sulfide_vector == (0.0, 0.0)
    assert matched == ()
    assert oxide_type is None


def test_legacy_candidate_provenance_requires_explicit_opt_in(tmp_path):
    path = tmp_path / "POSCAR"
    path.write_text("legacy structure bytes\n", encoding="utf-8")
    model = replace(
        _model(),
        backend_signature={
            "backend": "mace",
            "model": "mh-1",
            "task": "omat_pbe",
            "optimizer": "bfgs",
            "fmax_eV_per_A": 0.05,
            "max_steps": 300,
            "backend_package": "mace-torch",
            "backend_package_version": "0.3.15",
            "device": "cpu",
            "gpu_id": None,
        },
    )
    metadata = {
        "backend": "mace",
        "model": "mh-1",
        "task": "omat_pbe",
        "optimizer": "bfgs",
        "fmax_target_eV_per_A": 0.05,
        "max_steps": 300,
        "device": "cuda",
        "gpu_id": 0,
        "converged": True,
    }
    with pytest.raises(ValueError, match="explicitly enable legacy"):
        validate_candidate_energy_provenance(
            metadata,
            path,
            model,
            label="legacy candidate",
        )
    provenance = validate_candidate_energy_provenance(
        metadata,
        path,
        model,
        label="legacy candidate",
        allow_legacy=True,
    )
    assert provenance["mode"] == "legacy_explicitly_accepted"
    assert "missing_original_relaxed_poscar_sha256" in provenance["assumptions"]
    assert "missing_original_backend_package_version" in provenance["assumptions"]
    assert provenance["execution_differences"]


def test_legacy_opt_in_never_accepts_a_known_structure_hash_mismatch(tmp_path):
    path = tmp_path / "POSCAR"
    path.write_text("changed\n", encoding="utf-8")
    metadata = {
        "converged": True,
        "relaxed_poscar_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="structure changed after energy evaluation"):
        validate_candidate_energy_provenance(
            metadata,
            path,
            _model(),
            label="candidate",
            allow_legacy=True,
        )


def _write_custom(path, *, value, uncertainty, units, formula="SnO2"):
    with path.open("w", newline="", encoding="utf-8") as handle:
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
        writer.writerow(
            {
                "formula": formula,
                "formation_enthalpy": value,
                "uncertainty": uncertainty,
                "phase": "rutile",
                "temperature": "298 K",
                "units": units,
                "source": "test",
            }
        )


def test_custom_ev_per_atom_converts_energy_and_uncertainty_to_formula(tmp_path):
    path = tmp_path / "experimental.csv"
    _write_custom(path, value=-2.0, uncertainty=0.1, units="eV/atom")
    record = load_custom_experimental_dataset(path)[0]
    assert record.formation_enthalpy_eV_per_formula == pytest.approx(-6.0)
    assert record.uncertainty_eV_per_formula == pytest.approx(0.3)


def test_custom_ev_per_formula_converts_to_per_atom(tmp_path):
    path = tmp_path / "experimental.csv"
    _write_custom(path, value=-6.0, uncertainty=0.3, units="eV/formula_unit")
    record = load_custom_experimental_dataset(path)[0]
    assert record.formation_enthalpy_eV_per_atom == pytest.approx(-2.0)
    assert record.uncertainty_eV_per_atom == pytest.approx(0.1)


def test_custom_nonreduced_formula_unit_is_normalized_to_reduced_formula(tmp_path):
    path = tmp_path / "experimental.csv"
    _write_custom(
        path,
        formula="Fe4O6",
        value=-16.0,
        uncertainty=0.4,
        units="eV/formula_unit",
    )
    record = load_custom_experimental_dataset(path)[0]
    assert record.reduced_formula == "Fe2O3"
    assert record.formation_enthalpy_eV_per_formula == pytest.approx(-8.0)
    assert record.formation_enthalpy_eV_per_atom == pytest.approx(-1.6)
    assert record.uncertainty_eV_per_formula == pytest.approx(0.2)
    assert record.uncertainty_eV_per_atom == pytest.approx(0.04)


def test_custom_ambiguous_units_are_rejected(tmp_path):
    path = tmp_path / "experimental.csv"
    _write_custom(path, value=-6.0, uncertainty=0.3, units="eV")
    with pytest.raises(ValueError, match="ambiguous units"):
        load_custom_experimental_dataset(path)


def test_custom_nonstandard_temperature_is_rejected(tmp_path):
    path = tmp_path / "experimental.csv"
    _write_custom(path, value=-2.0, uncertainty=0.1, units="eV/atom")
    text = path.read_text(encoding="utf-8").replace("298 K", "500 K")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="does not mix temperatures"):
        load_custom_experimental_dataset(path)


def test_custom_missing_required_columns_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("formula,formation_enthalpy\nSnO2,-6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_custom_experimental_dataset(path)


def test_kingsbury_runtime_schema_is_normalized(monkeypatch):
    rows = [
            {
                "formula": "SnO2",
                "expt_form_e": -2.0,
                "uncertainty": 0.1,
                "phaseinfo": "rutile",
                "reference": "paper",
                "likely_mpid": "mp-1",
            }
        ]

    class FakeFrame:
        columns = tuple(rows[0])

        def to_dict(self, orient):
            assert orient == "records"
            return rows

    frame = FakeFrame()
    datasets = types.ModuleType("matminer.datasets")
    datasets.load_dataset = lambda name, **kwargs: frame
    package = types.ModuleType("matminer")
    package.datasets = datasets
    monkeypatch.setitem(sys.modules, "matminer", package)
    monkeypatch.setitem(sys.modules, "matminer.datasets", datasets)
    record = load_kingsbury_dataset()[0]
    assert record.formation_enthalpy_eV_per_formula == pytest.approx(-6.0)
    assert record.likely_mpid == "mp-1"


def test_kingsbury_nan_optional_text_is_normalized_to_empty(monkeypatch):
    rows = [
        {
            "formula": "SnO2",
            "expt_form_e": -2.0,
            "uncertainty": 0.1,
            "phaseinfo": np.nan,
            "reference": "paper",
            "likely_mpid": np.nan,
        }
    ]

    class FakeFrame:
        columns = tuple(rows[0])

        def to_dict(self, orient):
            assert orient == "records"
            return rows

    datasets = types.ModuleType("matminer.datasets")
    datasets.load_dataset = lambda name, **kwargs: FakeFrame()
    package = types.ModuleType("matminer")
    package.datasets = datasets
    monkeypatch.setitem(sys.modules, "matminer", package)
    monkeypatch.setitem(sys.modules, "matminer.datasets", datasets)
    record = load_kingsbury_dataset()[0]
    assert record.phase == ""
    assert record.likely_mpid == ""


def test_missing_and_zero_uncertainty_are_imputed_from_positive_population(tmp_path):
    path = tmp_path / "experimental.csv"
    rows = [
        ["Li2O", -2.0, 0.03, "cryst", "298 K", "eV/atom", "a"],
        ["MgO", -2.0, "", "cryst", "298 K", "eV/atom", "b"],
        ["CaO", -2.0, 0, "cryst", "298 K", "eV/atom", "c"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "formula",
                "formation_enthalpy",
                "uncertainty",
                "phase",
                "temperature",
                "units",
                "source",
            ]
        )
        writer.writerows(rows)
    imputed = impute_missing_uncertainties(load_custom_experimental_dataset(path))
    assert [record.uncertainty_eV_per_atom for record in imputed] == pytest.approx(
        [0.03, 0.03, 0.03]
    )
    assert [record.uncertainty_source for record in imputed] == [
        "reported",
        "imputed_mean",
        "imputed_mean",
    ]


def test_custom_override_uses_formula_and_phase(tmp_path):
    path = tmp_path / "base.csv"
    _write_custom(path, value=-2.0, uncertainty=0.1, units="eV/atom")
    base = load_custom_experimental_dataset(path)[0]
    replacement_data = base.to_dict()
    replacement_data["formation_enthalpy_eV_per_atom"] = -3.0
    replacement = ExperimentalRecord(**replacement_data)
    merged = merge_experimental_datasets([base], [replacement])
    assert merged[0].formation_enthalpy_eV_per_atom == -3.0


def test_weighted_linear_fit_recovers_known_oxide_coefficient():
    rows = [
        {
            "formula": "Li2O",
            "feature_vector": [1.0],
            "experimental_formation_eV_per_formula": -3.2,
            "calculated_formation_eV_per_formula": -3.0,
            "experimental_uncertainty_eV_per_formula": 0.03,
        },
        {
            "formula": "MgO",
            "feature_vector": [1.0],
            "experimental_formation_eV_per_formula": -2.2,
            "calculated_formation_eV_per_formula": -2.0,
            "experimental_uncertainty_eV_per_formula": 0.04,
        },
    ]
    model, report = fit_linear_correction_model(
        rows,
        correction_terms=["oxide"],
        backend_signature=_model().backend_signature,
        experimental_dataset="synthetic",
        experimental_dataset_version="1",
        fit_input_hash="hash",
    )
    assert model.coefficients_eV_per_term[0] == pytest.approx(-0.2)
    assert report["metrics"]["rmse_eV_per_atom"] == pytest.approx(0.0, abs=1e-12)


def test_fit_rejects_unidentifiable_terms():
    rows = [
        {
            "formula": "Li2O",
            "feature_vector": [1.0, 1.0],
            "experimental_formation_eV_per_formula": -3.2,
            "calculated_formation_eV_per_formula": -3.0,
            "experimental_uncertainty_eV_per_formula": 0.03,
        },
        {
            "formula": "MgO",
            "feature_vector": [1.0, 1.0],
            "experimental_formation_eV_per_formula": -2.2,
            "calculated_formation_eV_per_formula": -2.0,
            "experimental_uncertainty_eV_per_formula": 0.04,
        },
        {
            "formula": "CaO",
            "feature_vector": [1.0, 1.0],
            "experimental_formation_eV_per_formula": -1.2,
            "calculated_formation_eV_per_formula": -1.0,
            "experimental_uncertainty_eV_per_formula": 0.05,
        },
    ]
    with pytest.raises(ValueError, match="not identifiable"):
        fit_linear_correction_model(
            rows,
            correction_terms=["oxide", "element:Li"],
            backend_signature=_model().backend_signature,
            experimental_dataset="synthetic",
            experimental_dataset_version="1",
            fit_input_hash="hash",
        )


def test_fit_requires_replicated_support_for_every_term():
    rows = [
        {
            "formula": "Li2O",
            "feature_vector": [1.0, 1.0],
            "experimental_formation_eV_per_formula": -3.2,
            "calculated_formation_eV_per_formula": -3.0,
            "experimental_uncertainty_eV_per_formula": 0.03,
        },
        {
            "formula": "MgO",
            "feature_vector": [1.0, 0.0],
            "experimental_formation_eV_per_formula": -2.2,
            "calculated_formation_eV_per_formula": -2.0,
            "experimental_uncertainty_eV_per_formula": 0.04,
        },
        {
            "formula": "CaO",
            "feature_vector": [1.0, 0.0],
            "experimental_formation_eV_per_formula": -1.2,
            "calculated_formation_eV_per_formula": -1.0,
            "experimental_uncertainty_eV_per_formula": 0.05,
        },
    ]
    with pytest.raises(ValueError, match="replicated calibration support"):
        fit_linear_correction_model(
            rows,
            correction_terms=["oxide", "element:Li"],
            backend_signature=_model().backend_signature,
            experimental_dataset="synthetic",
            experimental_dataset_version="1",
            fit_input_hash="hash",
        )


def test_model_round_trip_and_backend_mismatch(tmp_path):
    model = _model()
    path = tmp_path / "model.json"
    save_correction_model(model, path)
    loaded = load_correction_model(path)
    assert loaded == model
    mismatch = dict(model.backend_signature)
    mismatch["model"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        validate_backend_compatibility(model, mismatch)


def test_model_loader_rejects_uncertainty_inconsistent_with_covariance(tmp_path):
    path = tmp_path / "model.json"
    save_correction_model(
        replace(
            _model(),
            coefficient_uncertainties_eV_per_term=(0.5,),
        ),
        path,
    )
    with pytest.raises(ValueError, match="do not match its covariance"):
        load_correction_model(path)


def test_model_loader_rejects_non_positive_semidefinite_covariance(tmp_path):
    path = tmp_path / "model.json"
    save_correction_model(
        _model(
            terms=("oxide", "element:Sn"),
            coefficients=(-0.2, 0.1),
            covariance=((0.04, 0.10), (0.10, 0.09)),
        ),
        path,
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        load_correction_model(path)


def test_oxide_feature_uses_actual_structure_stoichiometry():
    structure = _ordinary_oxide_structure("SnO2")
    vector, matched, kind = feature_vector(
        structure.composition,
        ["oxide"],
        structure=structure,
    )
    assert vector == (2.0,)
    assert matched == ("oxide",)
    assert kind == "oxide"


def test_oxygen_deficient_structure_uses_its_actual_oxygen_count():
    stoichiometric = _ordinary_oxide_structure("SnO2")
    oxygen_deficient = Structure(
        Lattice.cubic(5.0),
        ["Sn", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    full_vector, _, _ = feature_vector(
        stoichiometric.composition,
        ["oxide"],
        structure=stoichiometric,
    )
    deficient_vector, _, _ = feature_vector(
        oxygen_deficient.composition,
        ["oxide"],
        structure=oxygen_deficient,
    )
    assert full_vector == (2.0,)
    assert deficient_vector == (1.0,)


def test_backend_signature_hashes_a_local_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    signature = backend_signature_from_reference(
        {
            "backend": "custom",
            "model": str(checkpoint),
            "task": "",
            "optimizer": "bfgs",
            "fmax": 0.02,
            "max_steps": 300,
        }
    )
    assert signature["backend_package_version"] == "unknown"
    assert len(signature["model_checkpoint_sha256"]) == 64


def test_declared_oxide_type_cannot_override_structure_classification():
    structure = _ordinary_oxide_structure("SnO2")
    with pytest.raises(ValueError, match="disagrees"):
        feature_vector(
            structure.composition,
            ["oxide", "peroxide"],
            structure=structure,
            known_oxide_type="peroxide",
        )


def test_missing_required_oxygen_environment_term_is_rejected():
    # Two close O atoms classify this structure as a peroxide/superoxide,
    # which an oxide-only model must not silently correct.
    structure = Structure(
        Lattice.cubic(8.0),
        ["Li", "Li", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.40, 0.25, 0.25]],
    )
    with pytest.raises(ValueError, match="has no"):
        feature_vector(structure.composition, ["oxide"], structure=structure)


def test_elemental_reference_receives_no_compound_correction():
    application = apply_energy_correction(_model(), "O2")
    assert application.correction_eV == 0.0
    assert application.applied is False
    assert application.reason == "elemental_reference_not_corrected"


def test_polyanion_excluded_by_fit_is_rejected_at_application():
    model = replace(
        _model(),
        applicability_signature={"exclude_polyanions": ["CO3"]},
    )
    with pytest.raises(ValueError, match="polyanion excluded"):
        apply_energy_correction(
            model,
            "CaCO3",
            known_oxide_type="oxide",
        )


def test_shared_covariance_is_propagated_from_combined_reaction_vector():
    model = _model(
        terms=("oxide", "element:Sn"),
        coefficients=(-0.2, 0.1),
        covariance=((0.04, 0.015), (0.015, 0.09)),
    )
    vector = combine_feature_vectors(
        model.correction_terms,
        [(1.0, (2.0, 1.0)), (-1.0, (1.0, 1.0))],
    )
    application = evaluate_feature_vector(model, vector)
    assert vector == pytest.approx((1.0, 0.0))
    assert application.correction_eV == pytest.approx(-0.2)
    assert application.uncertainty_eV == pytest.approx(0.2)


def test_zero_balanced_reaction_is_explicitly_evaluated_but_not_applied():
    application = evaluate_feature_vector(_model(), (0.0,))
    assert application.correction_eV == 0.0
    assert application.applied is False
    assert application.reason == "model_evaluated_no_applicable_terms"

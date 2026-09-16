from __future__ import annotations

from pathlib import Path

import pytest

from dopingflow.corrections import CorrectionApplication, CorrectionModel
from dopingflow.vacancies import parse_vacancy_analysis_config
from dopingflow.vacancy_energy_correction_extensions import (
    parse_static_vacancy_thermodynamics_config,
    vacancy_reaction_correction,
)


def _model(
    *,
    family: str,
    terms: tuple[str, ...],
    coefficients: tuple[float, ...],
    covariance: tuple[tuple[float, ...], ...],
) -> CorrectionModel:
    return CorrectionModel(
        schema_version=1,
        method="kingsbury_weighted_linear_composition_correction",
        fit_id=f"test-{family}",
        backend_signature={},
        correction_terms=terms,
        coefficients_eV_per_term=coefficients,
        covariance_eV2=covariance,
        coefficient_uncertainties_eV_per_term=tuple(
            covariance[index][index] ** 0.5 for index in range(len(terms))
        ),
        experimental_dataset="test",
        experimental_dataset_version="test",
        fit_input_hash="test",
        units={
            "coefficient": "eV_per_term_atom",
            "covariance": "eV^2",
            "experimental_input": "eV_per_formula_unit",
            "reported_fit_residual": "eV_per_atom",
        },
        calibration_formulas=("SnO2",),
        fit_metrics={},
        model_family=family,
    )


def _application(vector: tuple[float, ...]) -> CorrectionApplication:
    return CorrectionApplication(
        correction_eV=0.0,
        uncertainty_eV=0.0,
        feature_vector=vector,
        matched_terms=(),
        applied=True,
        reason="test",
        oxide_type="oxide",
    )


def test_vacancy_fitted_correction_is_opt_in(tmp_path: Path) -> None:
    cfg = parse_static_vacancy_thermodynamics_config({}, tmp_path)
    assert cfg.apply_fitted_energy_correction is False
    assert cfg.allow_legacy_energy_correction_provenance is False


def test_vacancies_module_uses_enhanced_parser(tmp_path: Path) -> None:
    cfg = parse_vacancy_analysis_config(
        {"apply_fitted_energy_correction": True},
        tmp_path,
    )
    assert cfg.apply_fitted_energy_correction is True
    assert cfg.correction_workflow_root == tmp_path.resolve()


def test_rejects_fitted_correction_with_global_oxygen_calibration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="double count"):
        parse_static_vacancy_thermodynamics_config(
            {
                "apply_fitted_energy_correction": True,
                "oxygen_reference_mode": "global",
            },
            tmp_path,
        )


def test_rejects_fitted_correction_with_chemistry_specific_calibration(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="double count"):
        parse_static_vacancy_thermodynamics_config(
            {
                "apply_fitted_energy_correction": True,
                "oxygen_reference_mode": "chemistry-specific",
            },
            tmp_path,
        )


def test_m0_vacancy_reaction_changes_only_by_removed_oxygen() -> None:
    model = _model(
        family="m0",
        terms=("oxide",),
        coefficients=(-0.20,),
        covariance=((0.01,),),
    )
    parent = _application((80.0,))
    defect = _application((79.0,))

    reaction = vacancy_reaction_correction(model, defect, parent)

    assert reaction.feature_vector == pytest.approx((-1.0,))
    assert reaction.correction_eV == pytest.approx(0.20)
    assert reaction.uncertainty_eV == pytest.approx(0.10)


def test_m1_cation_terms_cancel_for_fixed_cation_composition() -> None:
    model = _model(
        family="m1",
        terms=("oxide", "oxide_cation:Sn", "oxide_cation:Sb"),
        coefficients=(-0.20, 0.03, -0.04),
        covariance=(
            (0.01, 0.002, -0.001),
            (0.002, 0.004, 0.0),
            (-0.001, 0.0, 0.003),
        ),
    )
    parent = _application((80.0, 38.0, 1.0))
    defect = _application((78.0, 38.0, 1.0))

    reaction = vacancy_reaction_correction(model, defect, parent)

    assert reaction.feature_vector == pytest.approx((-2.0, 0.0, 0.0))
    assert reaction.correction_eV == pytest.approx(0.40)
    assert reaction.uncertainty_eV == pytest.approx(0.20)

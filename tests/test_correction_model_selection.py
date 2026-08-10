import pytest

from dopingflow.correction_model_selection import (
    ModelSelectionConfig,
    correction_feature_vector,
    infer_workflow_target_elements,
    oxide_cation_term,
    select_correction_model_family,
)


def _row(formula, correction, *, sigma=0.05, oxide_type="oxide"):
    return {
        "formula": formula,
        "oxide_type": oxide_type,
        "experimental_formation_eV_per_formula": correction,
        "calculated_formation_eV_per_formula": 0.0,
        "experimental_uncertainty_eV_per_formula": sigma,
    }


def _synthetic_rows(*, oxygen_coefficient=-0.2, lithium_coefficient=0.4):
    formulas = ("MgO", "CaO", "Al2O3", "Li2O", "LiAlO2", "Li5AlO4")
    rows = []
    from pymatgen.core import Composition

    for formula in formulas:
        composition = Composition(formula)
        correction = (
            oxygen_coefficient * float(composition["O"])
            + lithium_coefficient * float(composition.get("Li", 0.0))
        )
        rows.append(_row(formula, correction))
    return rows


def _lithium_config():
    return {
        "doping": {
            "dopants": ["Li"],
            "compositions": [{"Li": 5.0}],
        },
        "scan": {"anion_species": ["O"]},
    }


def test_infer_workflow_target_elements_uses_host_dopants_and_references():
    raw = {
        "references": {
            "host": "SnO2",
            "metal_ref": ["Sn", "Sb", "O"],
            "oxides_ref": ["TiO2", "ZrO2"],
        },
        "doping": {
            "host_species": "Sn",
            "dopants": ["Sb", "Ti"],
            "must_include": ["Sb"],
            "compositions": [{"Nb": 5.0}],
        },
        "scan": {"anion_species": ["O"]},
        "vacancies": {
            "vacancy_species": "O",
            "oxidation_state_elements": ["Mn"],
        },
    }

    assert infer_workflow_target_elements(raw) == (
        "Mn",
        "Nb",
        "Sb",
        "Sn",
        "Ti",
        "Zr",
    )


def test_oxide_cation_features_are_scoped_to_ordinary_oxides():
    terms = ("oxide", oxide_cation_term("Li"))
    assert correction_feature_vector("Li2O", "oxide", terms) == (1.0, 2.0)
    assert correction_feature_vector("Li2O2", "peroxide", terms) == (0.0, 0.0)
    assert correction_feature_vector("Li", "oxide", terms) == (0.0, 0.0)


def test_m1_is_selected_only_after_independent_support_and_loo_improvement():
    result = select_correction_model_family(
        _synthetic_rows(),
        _lithium_config(),
        selection_config=ModelSelectionConfig(
            min_cv_rmse_improvement_eV_per_atom=1.0e-4
        ),
    )

    assert result.selected_family == "M1"
    assert result.m1_model is not None
    assert result.m1_model.terms == ("oxide", "oxide_cation:Li")
    assert result.m1_model.coefficients_eV_per_term == pytest.approx((-0.2, 0.4))
    assert result.m1_model.formulas == result.m0_model.formulas
    assert len(result.m1_model.loo_predictions_eV_per_formula) == len(
        result.m0_model.loo_predictions_eV_per_formula
    )
    assert result.report["cation_coverage"]["Li"]["unique_formulas"] == 3
    assert result.report["loo_rmse_improvement_eV_per_atom"] > 0
    assert result.report["one_standard_error_passed"] is True
    assert result.report["combined_m1_family_loo_diagnostics"]["Li"][
        "non_worsening"
    ] is True


def test_explicit_target_elements_override_reference_oxide_inventory():
    raw_config = {
        "references": {"oxides_ref": ["Na2O", "MgO", "Al2O3"]},
        "doping": {"dopants": ["Na"]},
    }

    result = select_correction_model_family(
        _synthetic_rows(),
        raw_config,
        target_elements=["Li"],
        selection_config=ModelSelectionConfig(
            min_cv_rmse_improvement_eV_per_atom=1.0e-4
        ),
    )

    assert result.selected_family == "M1"
    assert result.report["target_elements"] == ["Li"]
    assert result.report["target_element_source"] == "explicit_override"
    assert set(result.report["cation_coverage"]) == {"Li"}


def test_one_standard_error_gate_uses_paired_loo_squared_losses():
    corrections = (
        0.6134482969432957,
        -0.9651042455978449,
        -0.7457663186330873,
        0.10779847798459535,
        -0.588616077160142,
        0.34600212808266617,
    )
    formulas = ("MgO", "CaO", "Al2O3", "Li2O", "LiAlO2", "Li5AlO4")
    rows = [_row(formula, value) for formula, value in zip(formulas, corrections)]

    permissive = select_correction_model_family(
        rows,
        _lithium_config(),
        selection_config=ModelSelectionConfig(
            min_cv_rmse_improvement_eV_per_atom=0.0,
            require_one_standard_error=False,
        ),
    )
    conservative = select_correction_model_family(
        rows,
        _lithium_config(),
        selection_config=ModelSelectionConfig(
            min_cv_rmse_improvement_eV_per_atom=0.0,
            require_one_standard_error=True,
        ),
    )

    paired = conservative.report["paired_loo_squared_loss_improvement"]
    assert permissive.selected_family == "M1"
    assert conservative.selected_family == "M0"
    assert conservative.report["require_one_standard_error"] is True
    assert paired["mean_eV2_per_atom2"] > 0.0
    assert paired["mean_eV2_per_atom2"] < paired[
        "standard_error_eV2_per_atom2"
    ]
    assert paired["passes_one_standard_error"] is False


def test_cation_is_rejected_when_its_family_specific_loo_rmse_worsens():
    corrections = (
        -0.4143926157383104,
        1.1111738328305962,
        -0.3105472523914391,
        -0.425093760117127,
        1.1947861149929369,
        0.9229769176254901,
    )
    formulas = ("MgO", "CaO", "Al2O3", "Li2O", "LiAlO2", "Li5AlO4")
    rows = [_row(formula, value) for formula, value in zip(formulas, corrections)]

    result = select_correction_model_family(rows, _lithium_config())

    diagnostics = result.report["cation_coverage"]["Li"]
    assert result.selected_family == "M0"
    assert result.m1_model is None
    assert diagnostics["admitted"] is False
    assert diagnostics["m1_single_term_family_loo_rmse_eV_per_atom"] > (
        diagnostics["m0_family_loo_rmse_eV_per_atom"]
    )
    assert diagnostics["reasons"] == ["family_specific_loo_rmse_worsened"]


def test_m0_wins_a_tie_deterministically():
    result = select_correction_model_family(
        _synthetic_rows(lithium_coefficient=0.0),
        _lithium_config(),
        selection_config=ModelSelectionConfig(
            min_cv_rmse_improvement_eV_per_atom=0.0
        ),
    )

    assert result.m1_model is not None
    assert result.selected_family == "M0"
    assert "tie_break" in result.report["selection_method"]


def test_m1_element_is_rejected_without_three_independent_formulas():
    rows = _synthetic_rows()[:3] + _synthetic_rows()[3:5]
    result = select_correction_model_family(rows, _lithium_config())

    assert result.selected_family == "M0"
    assert result.m1_model is None
    assert result.report["admitted_m1_elements"] == []
    assert result.report["excluded_m1_elements"]["Li"] == [
        "insufficient_independent_formula_ratio_support"
    ]


def test_duplicate_polymorph_formula_does_not_count_as_independent_support():
    rows = _synthetic_rows()
    rows.append(_row("Li2O", -0.1))

    with pytest.raises(ValueError, match="one independent row per reduced formula"):
        select_correction_model_family(rows, _lithium_config())


def test_leave_one_out_rank_failure_makes_m1_unavailable():
    # Three Li oxides pass the nominal formula/ratio coverage gate, but after
    # holding out the sole non-Li anchor the O and Li columns become collinear.
    rows = [
        _row("MgO", -0.2),
        _row("Li2O", 0.6),
        _row("Li2MgO", 0.6),
        _row("Li2CaO", 0.6),
    ]
    result = select_correction_model_family(
        rows,
        _lithium_config(),
        selection_config=ModelSelectionConfig(min_unique_oxygen_ratios=1),
    )

    assert result.selected_family == "M0"
    assert result.m1_model is None
    assert "leaving out MgO" in result.report["m1_unavailable_reason"]


def test_condition_number_bound_is_applied_to_candidate_admission():
    result = select_correction_model_family(
        _synthetic_rows(),
        _lithium_config(),
        selection_config=ModelSelectionConfig(max_condition_number=1.01),
    )

    assert result.selected_family == "M0"
    assert result.m1_model is None
    assert any(
        "condition number" in reason or "condition bound" in reason
        for reason in result.report["excluded_m1_elements"]["Li"]
    )


def test_nonordinary_rows_are_rejected_before_identical_family_comparison():
    rows = _synthetic_rows()
    rows.append(_row("Na2O2", -1.0, oxide_type="peroxide"))
    result = select_correction_model_family(rows, _lithium_config())

    assert result.m0_model.n_observations == 6
    assert result.m1_model is not None
    assert result.report["rejected_rows"] == [
        {"index": 6, "formula": "Na2O2", "reason": "not_ordinary_oxide"}
    ]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"min_independent_cation_support": 2}, "must be >= 3"),
        ({"max_condition_number": 1.0}, "must be finite and > 1"),
        (
            {"min_cv_rmse_improvement_eV_per_atom": -0.1},
            "must be finite and >= 0",
        ),
        ({"require_one_standard_error": 1}, "must be a boolean"),
    ],
)
def test_selection_thresholds_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ModelSelectionConfig(**kwargs)

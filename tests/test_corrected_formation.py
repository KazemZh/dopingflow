import pytest

from dopingflow.corrections import CorrectionModel
from dopingflow.formation import (
    _attach_corrected_result,
    _flat_columns,
    _formation_result,
)


def _model():
    return CorrectionModel(
        schema_version=1,
        method="synthetic",
        fit_id="fit-formation",
        backend_signature={"backend": "mace", "model": "mh-1", "task": "omat_pbe"},
        correction_terms=("oxide",),
        coefficients_eV_per_term=(-0.5,),
        covariance_eV2=((0.04,),),
        coefficient_uncertainties_eV_per_term=(0.2,),
        experimental_dataset="synthetic",
        experimental_dataset_version="1",
        fit_input_hash="hash",
        units={"coefficient": "eV_per_term_atom"},
        calibration_formulas=("Li2O", "MgO"),
        fit_metrics={"rmse_eV_per_atom": 0.0},
    )


def _raw_result():
    return {
        "E_form_eV_total": 1.5,
        "E_form_eV_per_atom": 0.05,
        "E_form_eV_per_cation": 0.15,
        "E_form_eV_per_dopant": 1.5,
        "reported": {"value": 1.5, "unit": "eV_per_dopant_atom"},
        "mixing": {
            "E_mix_eV_total": 1.0,
            "E_mix_eV_per_atom": 1.0 / 30.0,
            "E_mix_eV_per_cation": 0.1,
            "E_mix_eV_per_dopant": 1.0,
        },
    }


def test_balanced_metal_reference_correction_preserves_raw_energy():
    result = _raw_result()
    _attach_corrected_result(
        result,
        model=_model(),
        formation_vector=(0.0,),
        mixing_vector=None,
        counts={"Sn": 9, "Sb": 1, "O": 20},
        dopant_counts={"Sb": 1},
        anion="O",
        n_atoms_supercell=30,
        normalize="per_dopant",
    )
    assert result["E_form_eV_total"] == 1.5
    assert result["E_form_corrected_eV_total"] == 1.5
    assert result["energy_correction"]["applied"] is False


def test_oxide_reaction_correction_and_uncertainty_use_reaction_vector():
    result = _raw_result()
    _attach_corrected_result(
        result,
        model=_model(),
        formation_vector=(-1.0,),
        mixing_vector=(-2.0,),
        counts={"Sn": 9, "Sb": 1, "O": 20},
        dopant_counts={"Sb": 1},
        anion="O",
        n_atoms_supercell=30,
        normalize="per_dopant",
    )
    assert result["E_form_eV_total"] == 1.5
    assert result["energy_correction_eV_total"] == pytest.approx(0.5)
    assert result["E_form_corrected_eV_total"] == pytest.approx(2.0)
    assert result["correction_uncertainty_eV_total"] == pytest.approx(0.2)
    assert result["mixing"]["E_mix_corrected_eV_total"] == pytest.approx(2.0)


def test_codoped_normalization_uses_all_dopant_atoms():
    result = _raw_result()
    _attach_corrected_result(
        result,
        model=_model(),
        formation_vector=(2.0,),
        mixing_vector=None,
        counts={"Sn": 8, "Sb": 1, "Ti": 1, "O": 20},
        dopant_counts={"Sb": 1, "Ti": 1},
        anion="O",
        n_atoms_supercell=30,
        normalize="per_dopant",
    )
    assert result["E_form_corrected_eV_total"] == pytest.approx(0.5)
    assert result["E_form_corrected_eV_per_dopant"] == pytest.approx(0.25)
    assert result["reported_corrected"]["value"] == pytest.approx(0.25)


def test_corrected_columns_are_added_without_overwriting_raw_columns():
    result = _raw_result()
    _attach_corrected_result(
        result,
        model=_model(),
        formation_vector=(1.0,),
        mixing_vector=(1.0,),
        counts={"Sn": 9, "Sb": 1, "O": 20},
        dopant_counts={"Sb": 1},
        anion="O",
        n_atoms_supercell=30,
        normalize="total",
    )
    flat = _flat_columns({"Sb2O5": result}, relative_enabled=False)
    assert flat["E_form_eV_total__Sb2O5"] == 1.5
    assert flat["E_form_corrected_eV_total__Sb2O5"] == pytest.approx(1.0)
    assert flat["correction_fit_id__Sb2O5"] == "fit-formation"


def test_formation_reaction_includes_host_and_oxygen_nonstoichiometry():
    host_ref = {
        "formula": "SnO2",
        "reduced_composition": {"Sn": 1.0, "O": 2.0},
        "E_per_formula_unit_eV": 8.0,
    }
    selected_oxides = {
        "Sb": (
            "Sb2O5",
            {
                "reduced_composition": {"Sb": 2.0, "O": 5.0},
                # Gives mu_Sb = (25 - 5 * 3) / 2 = 5 eV.
                "E_per_formula_unit_eV": 25.0,
            },
        )
    }
    result = _formation_result(
        E_doped=-99.0,
        E_pristine=-100.0,
        n_atoms_supercell=30,
        counts={"Sn": 8, "Sb": 1, "O": 19},
        pristine_counts={"Sn": 10, "O": 20},
        dopant_counts={"Sb": 1},
        host_species="Sn",
        anion="O",
        host_mu_value=2.0,
        oxygen_mu_value=3.0,
        selected_oxides=selected_oxides,
        host_ref=host_ref,
        E_O2=6.0,
        normalize="total",
    )

    # Delta n = {-2 Sn, +1 Sb, -1 O}; -sum(delta n_i mu_i) = +2 eV.
    assert result["stoichiometric_delta_atoms"] == {"O": -1, "Sb": 1, "Sn": -2}
    assert result["chemical_potential_term_eV"] == pytest.approx(2.0)
    assert result["E_form_eV_total"] == pytest.approx(3.0)


def test_general_reaction_reduces_to_legacy_substitution_expression():
    result = _formation_result(
        E_doped=-99.0,
        E_pristine=-100.0,
        n_atoms_supercell=30,
        counts={"Sn": 9, "Sb": 1, "O": 20},
        pristine_counts={"Sn": 10, "O": 20},
        dopant_counts={"Sb": 1},
        host_species="Sn",
        anion="O",
        host_mu_value=2.0,
        oxygen_mu_value=3.0,
        selected_oxides={
            "Sb": (
                "Sb2O5",
                {
                    "reduced_composition": {"Sb": 2.0, "O": 5.0},
                    "E_per_formula_unit_eV": 25.0,
                },
            )
        },
        host_ref={
            "formula": "SnO2",
            "reduced_composition": {"Sn": 1.0, "O": 2.0},
            "E_per_formula_unit_eV": 8.0,
        },
        E_O2=6.0,
        normalize="total",
    )
    expected = (-99.0) - (-100.0) + 1 * (2.0 - 5.0)
    assert result["E_form_eV_total"] == pytest.approx(expected)

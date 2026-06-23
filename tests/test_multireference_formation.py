from pathlib import Path

import pytest

from dopingflow.formation import CandidateRecord, _apply_relative_energies, _flat_columns


def _result(E_form_per_cation: float, E_mix_per_cation: float):
    return {
        "E_form_eV_total": E_form_per_cation * 10.0,
        "E_form_eV_per_atom": E_form_per_cation / 3.0,
        "E_form_eV_per_cation": E_form_per_cation,
        "E_form_eV_per_dopant": E_form_per_cation * 20.0,
        "mixing": {
            "E_mix_eV_total": E_mix_per_cation * 10.0,
            "E_mix_eV_per_atom": E_mix_per_cation / 3.0,
            "E_mix_eV_per_cation": E_mix_per_cation,
            "E_mix_eV_per_dopant": E_mix_per_cation * 20.0,
            "n_O2_out": 0.0,
            "reaction_reference": "reference reaction",
        },
    }


def _record(name: str, x: float, results: dict):
    return CandidateRecord(
        folder=Path("composition"),
        candidate_dir=Path(name),
        candidate=name,
        E_doped=0.0,
        counts={"Sn": 9, "Sb": 1, "O": 20},
        dopant_counts={"Sb": 1},
        x_dopant=x,
        reference_results=results,
    )


def test_relative_energies_use_lowest_endpoint_per_reference():
    low_x = _record(
        "candidate_low_x",
        0.05,
        {
            "SbO2": _result(0.30, 0.10),
            "Sb2O3": _result(0.50, 0.20),
        },
    )
    endpoint_high = _record(
        "candidate_endpoint_high",
        0.10,
        {
            "SbO2": _result(0.80, 0.40),
            "Sb2O3": _result(1.00, 0.60),
        },
    )
    endpoint_low = _record(
        "candidate_endpoint_low",
        0.10,
        {
            "SbO2": _result(0.60, 0.25),
            "Sb2O3": _result(0.90, 0.50),
        },
    )

    X, endpoints = _apply_relative_energies(
        [low_x, endpoint_high, endpoint_low],
        endpoint_x=None,
    )

    assert X == pytest.approx(0.10)
    assert endpoints["SbO2"]["E_form_eV_per_cation"] == pytest.approx(0.60)
    assert endpoints["Sb2O3"]["E_mix_eV_per_cation"] == pytest.approx(0.50)

    # E_rel(x=0.05) = E(x=0.05) - (0.05 / 0.10) * E_min(X)
    assert low_x.reference_results["SbO2"]["relative"]["E_form_rel_eV_per_cation"] == pytest.approx(0.0)
    assert low_x.reference_results["Sb2O3"]["relative"]["E_mix_rel_eV_per_cation"] == pytest.approx(-0.05)


def test_wide_columns_keep_one_row_per_candidate():
    values = _flat_columns(
        {
            "SbO2": _result(0.30, 0.10),
            "Sb2O5": _result(0.20, -0.05),
        },
        relative_enabled=False,
    )

    assert "E_form_eV_total__SbO2" in values
    assert "E_mix_eV_per_cation__Sb2O5" in values
    assert "E_form_eV_total__Sb2O3" not in values

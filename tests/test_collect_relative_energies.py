import pytest

from dopingflow.collect import _calculate_relative_columns


def test_collect_populates_relative_columns_after_global_collection():
    rows = [
        {
            "x_dopant": 0.05,
            "E_form_eV_per_cation__SbO2": 0.20,
            "E_mix_eV_per_cation__SbO2": 0.30,
        },
        {
            "x_dopant": 0.10,
            "E_form_eV_per_cation__SbO2": 0.40,
            "E_mix_eV_per_cation__SbO2": 0.10,
        },
    ]

    columns = _calculate_relative_columns(rows, endpoint_x=None)

    assert "E_form_rel_eV_per_cation__SbO2" in columns
    assert "E_mix_rel_eV_per_cation__SbO2" in columns

    # X = 0.10; both endpoint energies are 0.40 and 0.10, respectively.
    assert rows[0]["E_form_rel_eV_per_cation__SbO2"] == pytest.approx(0.0)
    assert rows[0]["E_mix_rel_eV_per_cation__SbO2"] == pytest.approx(0.25)
    assert rows[1]["E_form_rel_eV_per_cation__SbO2"] == pytest.approx(0.0)
    assert rows[1]["E_mix_rel_eV_per_cation__SbO2"] == pytest.approx(0.0)

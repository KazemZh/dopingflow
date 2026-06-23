import csv

import pytest

from dopingflow.relative_energy import populate_relative_energy_columns


def test_relative_energies_are_always_written_with_automatic_endpoint(tmp_path):
    path = tmp_path / "results_database.csv"
    rows = [
        {
            "x_dopant": "0.05",
            "E_form_eV_per_cation__SbO2": "0.20",
            "E_mix_eV_per_cation__SbO2": "0.30",
        },
        {
            "x_dopant": "0.10",
            "E_form_eV_per_cation__SbO2": "0.40",
            "E_mix_eV_per_cation__SbO2": "0.10",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    populate_relative_energy_columns(path, {"formation": {}})

    with path.open("r", newline="", encoding="utf-8") as handle:
        result = list(csv.DictReader(handle))

    assert float(result[0]["E_form_rel_eV_per_cation__SbO2"]) == pytest.approx(0.0)
    assert float(result[1]["E_form_rel_eV_per_cation__SbO2"]) == pytest.approx(0.0)
    assert float(result[0]["E_mix_rel_eV_per_cation__SbO2"]) == pytest.approx(0.25)
    assert float(result[1]["E_mix_rel_eV_per_cation__SbO2"]) == pytest.approx(0.0)


def test_explicit_endpoint_x_is_read_from_formation_section(tmp_path):
    path = tmp_path / "results_database.csv"
    rows = [
        {
            "x_dopant": "0.05",
            "E_form_eV_per_cation__SbO2": "0.20",
        },
        {
            "x_dopant": "0.10",
            "E_form_eV_per_cation__SbO2": "0.40",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    populate_relative_energy_columns(path, {"formation": {"endpoint_x": 0.10}})

    with path.open("r", newline="", encoding="utf-8") as handle:
        result = list(csv.DictReader(handle))

    assert float(result[0]["E_form_rel_eV_per_cation__SbO2"]) == pytest.approx(0.0)

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dopingflow.vacancy_analysis import (
    K_B_EV_PER_K,
    NEGLECTED_SOLID_TERMS,
    STATIC_LATTICE_APPROXIMATION,
    inverse_oxygen_pressure_log10,
    nist_o2_standard_state_delta_mu_eV_per_O,
    oxygen_pressure_delta_mu_eV_per_O,
    oxygen_standard_state_delta_mu,
)
from dopingflow.vacancy_static_thermodynamics import (
    analyze_static_vacancy_thermodynamics,
    parse_static_vacancy_thermodynamics_config as parse_vacancy_analysis_config,
)


def config(tmp_path: Path, **updates: object):
    section: dict[str, object] = {
        "static_thermodynamic_analysis": True,
        "static_energy_source": "relaxed_only",
        "oxygen_reference_mode": "explicit",
        "mu_O_reference_eV": -5.0,
        "delta_mu_O_points_eV": [0.0, -1.0, -2.0],
        "temperatures_K": [300.0, 900.0],
        "log10_pO2_min_bar": -2.0,
        "log10_pO2_max_bar": 0.0,
        "log10_pO2_step": 1.0,
    }
    section.update(updates)
    return parse_vacancy_analysis_config(section, tmp_path)


def row(n_vacancies: int, energy: float, *, converged: bool = True) -> dict[str, object]:
    return {
        "composition_directory": "Sb20",
        "parent_id": "parent",
        "configuration_id": "parent_reference" if n_vacancies == 0 else f"v{n_vacancies}",
        "host_species": "Sn",
        "vacancy_species": "O",
        "dopant_counts_from_parent": {"Sb": 2},
        "n_host": 8,
        "n_total_cations": 10,
        "n_oxygen_sites_parent": 20,
        "n_vacancies": n_vacancies,
        "energy_relaxed_total_eV": energy,
        "converged": converged,
        "relaxed_poscar_path": f"v{n_vacancies}/POSCAR",
        "backend": "mace",
        "model": "small",
        "task": "",
    }


def analyze(tmp_path: Path, **updates: object) -> dict[str, Path]:
    return analyze_static_vacancy_thermodynamics(
        rows=[row(0, -100.0), row(1, -94.0), row(2, -87.0)],
        cfg=config(tmp_path, **updates),
        parent_root=tmp_path,
        backend="mace",
        model="small",
        task="",
    )


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_static_keys_are_flat_and_legacy_enable_is_accepted(tmp_path: Path):
    static = parse_vacancy_analysis_config(
        {"static_thermodynamic_analysis": True, "static_energy_source": "relaxed_only"},
        tmp_path,
    )
    legacy = parse_vacancy_analysis_config({"thermodynamic_analysis": True}, tmp_path)
    assert static.enabled is True
    assert static.analysis_energy_source == "relaxed_only"
    assert legacy.enabled is True


def test_pressure_contribution_and_inverse_mapping_are_consistent():
    temperature = 900.0
    pressure = 1.0e-10
    delta = oxygen_pressure_delta_mu_eV_per_O(temperature, pressure)
    expected = 0.5 * K_B_EV_PER_K * temperature * math.log(pressure)
    assert delta == pytest.approx(expected)
    assert inverse_oxygen_pressure_log10(delta, temperature) == pytest.approx(-10.0)


def test_nist_shomate_matches_300_K_and_supports_non_tabulated_temperature():
    assert nist_o2_standard_state_delta_mu_eV_per_O(300.0) == pytest.approx(
        -0.27393606, abs=1.0e-8
    )
    value_873 = nist_o2_standard_state_delta_mu_eV_per_O(873.0)
    assert -0.97431406 < value_873 < -0.61120191


def test_nist_shomate_rejects_extrapolation_and_requires_one_bar(tmp_path: Path):
    with pytest.raises(ValueError, match="extrapolation"):
        nist_o2_standard_state_delta_mu_eV_per_O(99.0)
    with pytest.raises(ValueError, match="100 to 6000"):
        config(
            tmp_path,
            temperatures_K=[99.0, 300.0],
            oxygen_standard_state_mode="nist_shomate",
        )
    with pytest.raises(ValueError, match="1 bar"):
        config(
            tmp_path,
            oxygen_standard_state_mode="nist_shomate",
            standard_oxygen_pressure_bar=2.0,
        )


def test_user_standard_state_table_interpolates_and_must_cover_temperatures(tmp_path: Path):
    cfg = config(
        tmp_path,
        temperatures_K=[300.0, 600.0, 900.0],
        oxygen_standard_state_mode="user_table",
        oxygen_standard_state_temperatures_K=[300.0, 900.0],
        oxygen_standard_state_delta_mu_eV_per_O=[-0.1, -0.7],
    )
    assert oxygen_standard_state_delta_mu(cfg, 600.0) == pytest.approx(-0.4)
    with pytest.raises(ValueError, match="must cover"):
        config(
            tmp_path,
            temperatures_K=[300.0, 1200.0],
            oxygen_standard_state_mode="user_table",
            oxygen_standard_state_temperatures_K=[300.0, 900.0],
            oxygen_standard_state_delta_mu_eV_per_O=[-0.1, -0.7],
        )


def test_static_outputs_and_approximation_metadata(tmp_path: Path):
    outputs = analyze(tmp_path)
    assert outputs["vacancy_static_minima_csv"].exists()
    assert outputs["vacancy_static_stability_intervals_json"].exists()
    assert outputs["vacancy_static_best_counts_json"].exists()
    assert outputs["vacancy_static_pressure_map_json"].exists()
    metadata = read(outputs["static_metadata"])
    assert metadata["static_lattice_approximation"] == STATIC_LATTICE_APPROXIMATION
    assert metadata["neglected_solid_terms"] == list(NEGLECTED_SOLID_TERMS)
    assert metadata["pressure_mapping_is_approximate"] is True


def test_pressure_map_marks_omitted_standard_state_correction(tmp_path: Path):
    outputs = analyze(tmp_path)
    pressure = read(outputs["vacancy_static_pressure_map_json"])
    assert len(pressure) == 2 * 3
    assert all(item["pressure_mapping_is_approximate"] for item in pressure)
    assert all(
        item["pressure_mapping_approximation"]
        == "O2 standard-state thermal correction omitted"
        for item in pressure
    )
    assert all(item["static_lattice_approximation"] for item in pressure)


def test_user_table_pressure_map_is_not_marked_omitted(tmp_path: Path):
    outputs = analyze(
        tmp_path,
        oxygen_standard_state_mode="user_table",
        oxygen_standard_state_temperatures_K=[300.0, 900.0],
        oxygen_standard_state_delta_mu_eV_per_O=[-0.1, -0.7],
    )
    pressure = read(outputs["vacancy_static_pressure_map_json"])
    assert all(not item["pressure_mapping_is_approximate"] for item in pressure)
    row_900 = next(item for item in pressure if item["temperature_K"] == 900.0)
    assert row_900["delta_mu_O_standard_eV_per_O"] == pytest.approx(-0.7)


def test_nist_pressure_map_records_source_and_continuous_correction(tmp_path: Path):
    outputs = analyze(
        tmp_path,
        temperatures_K=[300.0, 873.0],
        oxygen_standard_state_mode="nist_shomate",
    )
    pressure = read(outputs["vacancy_static_pressure_map_json"])
    row_873 = next(item for item in pressure if item["temperature_K"] == 873.0)
    assert row_873["pressure_mapping_is_approximate"] is False
    assert row_873["oxygen_standard_state_mode"] == "nist_shomate"
    assert "NIST Chemistry WebBook" in row_873["oxygen_standard_state_source"]
    assert row_873["oxygen_standard_state_zpe_included"] is False
    assert row_873["delta_mu_O_standard_eV_per_O"] == pytest.approx(
        nist_o2_standard_state_delta_mu_eV_per_O(873.0)
    )


def test_disabled_pressure_mapping_writes_empty_table_without_approximation_claim(
    tmp_path: Path,
):
    outputs = analyze(tmp_path, pressure_mapping=False)
    assert read(outputs["vacancy_static_pressure_map_json"]) == []
    metadata = read(outputs["static_metadata"])
    assert metadata["pressure_mapping_is_approximate"] is False


def test_static_plotting_script_smoke(tmp_path: Path):
    outputs = analyze(tmp_path)
    minima = read(outputs["vacancy_static_minima_json"])
    composition = minima[0]["actual_composition_key"]
    output_dir = tmp_path / "plots"
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "vacancies"
        / "plot_static_vacancy_thermodynamics.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--minima", str(outputs["vacancy_static_minima_csv"]),
            "--intervals", str(outputs["vacancy_static_stability_intervals_csv"]),
            "--best-counts", str(outputs["vacancy_static_best_counts_csv"]),
            "--pressure-map", str(outputs["vacancy_static_pressure_map_csv"]),
            "--composition", composition,
            "--delta-mu-o", "0.0",
            "--temperature", "900",
            "--x-dopant", "Sb",
            "--output-dir", str(output_dir),
        ],
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert len(list(output_dir.glob("*.png"))) == 5
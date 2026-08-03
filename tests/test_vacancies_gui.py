from pathlib import Path
import sys

import pandas as pd
import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.gui_config import CHOICES, DEFAULTS, STEP_KEYS
from gui.io_project import ProjectIndex
from gui.vacancy_thermo_plots import (
    grand_potential_lines,
    grand_potential_lines_pressure,
    grand_potential_vs_count,
    grand_potential_vs_count_pressure,
    pressure_stability_map,
    pressure_stability_vs_composition,
    preferred_count_vs_doping,
    preferred_count_vs_doping_pressure,
    stability_map,
    vacancy_count_color,
)


def test_vacancy_gui_defaults_are_one_flat_section():
    assert STEP_KEYS.count("vacancies") == 1
    section = DEFAULTS["vacancies"]
    assert all(not isinstance(value, dict) for value in section.values())
    assert CHOICES["vacancies.parent_source"] == ["selected_candidates", "directory"]
    assert CHOICES["vacancies.enumeration_mode"] == ["auto", "exact", "sample"]
    assert CHOICES["vacancies.oxygen_reference_mode"] == [
        "reference_file", "same_calculator", "explicit", "none"
    ]
    assert section["static_thermodynamic_analysis"] is False
    assert section["static_energy_source"] == "relaxed_only"
    assert section["pressure_mapping"] is True
    assert section["oxygen_standard_state_mode"] == "nist_shomate"
    assert CHOICES["vacancies.oxygen_standard_state_mode"] == [
        "none", "nist_shomate", "user_table"
    ]
    assert section["exclude_unconverged"] is True


def test_vacancy_toml_round_trip_stays_flat():
    dumped = toml.dumps({"vacancies": DEFAULTS["vacancies"]})
    assert "[vacancies]" in dumped
    assert "[vacancies." not in dumped
    loaded = toml.loads(dumped)
    assert loaded["vacancies"]["oxidation_state_values"] == [
        [3, 5], [5], [3], [5], [3, 4], [3, 4, 5], [2, 3, 4]
    ]
    assert loaded["vacancies"]["delta_mu_O_points_eV"] == [
        0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0
    ]
    assert "static_thermodynamic_analysis" in loaded["vacancies"]


def test_project_index_detects_vacancy_results(tmp_path: Path):
    outdir = tmp_path / "random_structures"
    root = outdir / "Sb5" / "candidate_001" / "05_vacancies"
    group = root / "V_O_01"
    group.mkdir(parents=True)
    (root / "vacancy_results.csv").write_text("configuration_id\n", encoding="utf-8")
    (group / "ranking_scan.csv").write_text("configuration_id\n", encoding="utf-8")
    (outdir / "vacancies_database.csv").write_text("parent_id\n", encoding="utf-8")
    project = ProjectIndex(root=tmp_path, outdir=outdir)
    assert project.vacancy_database() == outdir / "vacancies_database.csv"
    assert project.vacancy_parents("Sb5") == ["candidate_001"]
    assert "V_O_01/ranking_scan.csv" in project.vacancy_rankings("Sb5", "candidate_001")


def test_vacancy_thermodynamic_gui_figures_use_categorical_legend():
    minima = pd.DataFrame(
        [
            {
                "actual_composition_key": "Sb10",
                "n_vacancies": count,
                "grand_potential_intercept_eV": intercept,
                "source_parent_id": "parent",
                "source_configuration_id": f"v{count}",
                "converged": True,
                "backend": "grace",
                "model": "model",
                "task": "",
                "oxygen_reference_mode": "same_calculator",
                "oxygen_reference_verified": True,
            }
            for count, intercept in ((0, 0.0), (1, 0.5), (3, 3.0))
        ]
    )
    intervals = pd.DataFrame(
        [
            {
                **minima.iloc[0].to_dict(),
                "stable_n_vacancies": count,
                "delta_mu_O_lower_eV": lower,
                "delta_mu_O_upper_eV": upper,
                "percent_Sb": 10.0,
                "total_dopant_percent": 10.0,
            }
            for count, lower, upper in ((3, -3.0, -1.5), (1, -1.5, -0.5), (0, -0.5, 0.0))
        ]
    )
    best = pd.DataFrame(
        [
            {
                **minima.iloc[0].to_dict(),
                "delta_mu_O_eV": -1.0,
                "best_n_vacancies": 1,
                "is_tied": False,
                "percent_Sb": 10.0,
                "total_dopant_percent": 10.0,
            }
        ]
    )

    heatmap = stability_map(intervals, "Sb")
    assert heatmap.data[0].showscale is False
    assert [trace.name for trace in heatmap.data[1:]] == ["0", "1", "3"]
    assert heatmap.data[2].marker.color == vacancy_count_color(1)
    assert grand_potential_lines(minima, intervals, "Sb10").data
    count_figure = grand_potential_vs_count(minima, "Sb10", -1.0)
    assert list(count_figure.data[0].x) == [0, 1, 3]
    preferred = count_figure.data[2]
    preferred_count = int(preferred.x[0])
    curve_index = list(count_figure.data[0].x).index(preferred_count)
    assert preferred.y[0] == count_figure.data[0].y[curve_index]
    assert preferred_count_vs_doping(best, -1.0, "Sb").data

    pressure = pd.DataFrame(
        [
            {
                **best.iloc[0].to_dict(),
                "temperature_K": temperature,
                "log10_oxygen_partial_pressure_bar": log_pressure,
                "best_n_vacancies": count,
                "pressure_mapping_approximation": (
                    "O2 standard-state thermal correction omitted"
                ),
                "pressure_mapping_is_approximate": True,
                "delta_mu_O_pressure_eV_per_O": 0.1 * log_pressure,
                "delta_mu_O_total_eV_per_O": -0.5 + 0.1 * log_pressure,
                "is_tied": False,
            }
            for temperature in (300.0, 900.0)
            for log_pressure, count in ((-2.0, 1), (0.0, 0))
        ]
    )
    pressure_figure = pressure_stability_map(pressure, "Sb10")
    assert pressure_figure.data[0].showscale is False
    assert pressure_figure.layout.title.text.startswith(
        "Approximate T–pO2 vacancy stability"
    )

    nist_pressure = pressure.assign(
        pressure_mapping_is_approximate=False,
        pressure_mapping_approximation=(
            "NIST O2 Shomate standard-state correction evaluated continuously"
        ),
        oxygen_standard_state_mode="nist_shomate",
        oxygen_standard_state_source="NIST Chemistry WebBook SRD 69",
        oxygen_standard_state_zpe_included=False,
    )
    nist_figure = pressure_stability_map(nist_pressure, "Sb10")
    assert nist_figure.layout.title.text.startswith("T–pO2 vacancy stability")
    assert "NIST Shomate, 1 bar" in nist_figure.layout.title.text
    assert "explicit ZPE excluded" in nist_figure.layout.title.text

    physical_stability = pressure_stability_vs_composition(
        minima, pressure, 300.0, "Sb", True
    )
    assert physical_stability.layout.yaxis.title.text == "log10(pO2/bar)"
    physical_lines = grand_potential_lines_pressure(
        minima, pressure, "Sb10", 300.0, True
    )
    assert len(physical_lines.data) == 3
    physical_count = grand_potential_vs_count_pressure(
        minima, pressure, "Sb10", 300.0, 0.0, True
    )
    assert "T=300 K" in physical_count.layout.title.text
    physical_doping = preferred_count_vs_doping_pressure(
        minima, pressure, 300.0, -2.0, "Sb", True
    )
    assert physical_doping.data

    omitted_lines = grand_potential_lines_pressure(
        minima, pressure, "Sb10", 300.0, False
    )
    assert "intentionally omitted" in omitted_lines.layout.title.text
    assert list(omitted_lines.data[1].y) != list(physical_lines.data[1].y)
    omitted_map = pressure_stability_map(pressure, "Sb10", minima, False)
    assert omitted_map.layout.title.text.startswith("Approximate T–pO2")

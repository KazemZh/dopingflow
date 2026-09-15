import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.phase_diagram_plots import (
    build_phase_composition_map,
    build_phase_diagram_figure,
    build_vacancy_hull_figure,
    find_codoped_system,
    prepare_phase_diagram_plot_data,
    prepare_vacancy_hull_data,
    select_hull_quantity,
)


def _frames():
    phase = pd.DataFrame(
        [
            {
                "candidate_path": "/tmp/a",
                "chemical_system": "O-Sb-Sn-Ti",
                "candidate": "candidate_001",
                "formula": "SnO2",
                "energy_above_hull_eV_per_atom": 0.010,
                "stable": False,
                "decomposition": "SnO2",
            },
            {
                "candidate_path": "/tmp/b",
                "chemical_system": "O-Sb-Sn-Ti",
                "candidate": "candidate_002",
                "formula": "SnO2",
                "energy_above_hull_eV_per_atom": 0.005,
                "stable": False,
                "decomposition": "SnO2",
            },
            {
                "candidate_path": "/tmp/c",
                "chemical_system": "O-Sb-Sn-Ti",
                "candidate": "candidate_003",
                "formula": "SnO2",
                "energy_above_hull_eV_per_atom": 0.0,
                "stable": True,
                "decomposition": "candidate_003",
            },
            {
                "candidate_path": "/tmp/d",
                "chemical_system": "O-Sb-Sn",
                "candidate": "candidate_004",
                "formula": "SnO2",
                "energy_above_hull_eV_per_atom": 0.002,
                "stable": False,
                "decomposition": "SnO2",
            },
        ]
    )
    database = pd.DataFrame(
        [
            {
                "candidate_path": "/tmp/a",
                "effective_pct_json": json.dumps({"Sb": 5, "Ti": 2.5}),
            },
            {
                "candidate_path": "/tmp/b",
                "effective_pct_json": json.dumps({"Sb": 5, "Ti": 2.5}),
            },
            {
                "candidate_path": "/tmp/c",
                "effective_pct_json": json.dumps({"Sb": 10, "Ti": 2.5}),
            },
            {
                "candidate_path": "/tmp/d",
                "effective_pct_json": json.dumps({"Sb": 5}),
            },
        ]
    )
    return phase, database


def test_prepare_phase_diagram_plot_data_joins_percentages_and_converts_units():
    phase, database = _frames()
    prepared, dopants = prepare_phase_diagram_plot_data(phase, database)

    assert dopants == ["Sb", "Ti"]
    assert prepared["percent_Sb"].tolist() == [5.0, 5.0, 10.0, 5.0]
    assert prepared["energy_above_hull_meV_per_atom"].tolist() == [10.0, 5.0, 0.0, 2.0]


def test_build_phase_diagram_figure_uses_best_candidate_and_stable_marker():
    phase, database = _frames()
    prepared, _ = prepare_phase_diagram_plot_data(phase, database)
    figure, best = build_phase_diagram_figure(
        prepared,
        chemical_system="O-Sb-Sn-Ti",
        x_dopant="Sb",
        series_dopant="Ti",
        selected_series=[0.0, 2.5],
        host_formula="SnO2",
    )

    assert best["candidate"].tolist() == [
        "candidate_004",
        "candidate_002",
        "candidate_003",
    ]
    assert best["energy_above_hull_meV_per_atom"].tolist() == [2.0, 5.0, 0.0]
    assert {trace.name for trace in figure.data} == {
        "Ti = 0%",
        "Ti = 2.5%",
        "Stable / within threshold",
        "SnO2",
    }


def test_find_codoped_system_selects_the_matching_pair():
    phase, database = _frames()
    prepared, _ = prepare_phase_diagram_plot_data(phase, database)

    assert find_codoped_system(prepared, "Sb", "Ti") == "O-Sb-Sn-Ti"


def test_metadata_join_survives_copied_absolute_path_prefixes():
    phase = pd.DataFrame(
        [
            {
                "candidate_path": "/local/project/Sb5/candidate_001",
                "chemical_system": "O-Sb-Sn",
                "candidate": "candidate_001",
                "formula": "Sn19SbO40",
                "energy_above_hull_eV_per_atom": 0.02,
                "stable": False,
            }
        ]
    )
    database = pd.DataFrame(
        [
            {
                "candidate_path": "/remote/hpc/run/Sb5/candidate_001",
                "effective_pct_json": json.dumps({"Sb": 5.0}),
            }
        ]
    )

    prepared, dopants = prepare_phase_diagram_plot_data(phase, database)

    assert dopants == ["Sb"]
    assert prepared.iloc[0]["percent_Sb"] == 5.0
    assert prepared.attrs["unmatched_phase_rows"] == 0


def test_unmatched_vacancy_rows_do_not_break_normal_composition_plot():
    phase = pd.DataFrame(
        [
            {
                "candidate_path": "/local/Sb5/candidate_001",
                "chemical_system": "O-Sb-Sn",
                "candidate": "candidate_001",
                "formula": "Sn19SbO40",
                "energy_above_hull_eV_per_atom": 0.02,
                "stable": False,
            },
            {
                "candidate_path": (
                    "/local/Sb5/candidate_001/05_vacancies/V_O_01/config_0001"
                ),
                "chemical_system": "O-Sb-Sn",
                "candidate": "config_0001",
                "formula": "Sn19SbO39",
                "energy_above_hull_eV_per_atom": 0.01,
                "stable": False,
            },
        ]
    )
    database = pd.DataFrame(
        [
            {
                "candidate_path": "/remote/Sb5/candidate_001",
                "effective_pct_json": json.dumps({"Sb": 5.0}),
            }
        ]
    )

    prepared, dopants = prepare_phase_diagram_plot_data(phase, database)

    assert dopants == ["Sb"]
    assert len(prepared) == 1
    assert prepared.attrs["unmatched_phase_rows"] == 1


def test_corrected_hull_selection_and_composition_map():
    phase = pd.DataFrame(
        [
            {
                "candidate_path": "/local/Ce2p5_Sb2p5/candidate_001",
                "chemical_system": "Ce-O-Sb-Sn",
                "candidate": "candidate_001",
                "formula": "CeSbSn38O80",
                "energy_above_hull_eV_per_atom": 0.04,
                "energy_above_hull_raw_eV_per_atom": 0.04,
                "energy_above_hull_corrected_eV_per_atom": 0.02,
                "stable": False,
                "stable_raw": False,
                "stable_corrected": True,
                "decomposition": "raw",
                "decomposition_corrected": "corrected",
            }
        ]
    )
    database = pd.DataFrame(
        [
            {
                "candidate_path": "/remote/Ce2p5_Sb2p5/candidate_001",
                "effective_pct_json": json.dumps({"Ce": 2.5, "Sb": 2.5}),
            }
        ]
    )

    selected = select_hull_quantity(phase, "Corrected")
    prepared, dopants = prepare_phase_diagram_plot_data(selected, database)
    figure, best = build_phase_composition_map(
        prepared,
        chemical_system="Ce-O-Sb-Sn",
        x_dopant="Sb",
        y_dopant="Ce",
    )

    assert dopants == ["Ce", "Sb"]
    assert best.iloc[0]["energy_above_hull_meV_per_atom"] == 20.0
    assert bool(best.iloc[0]["_stable_bool"])
    assert len(figure.data) >= 1


def test_vacancy_hull_plot_uses_corrected_energy():
    vacancy = pd.DataFrame(
        [
            {
                "actual_composition_key": "Ce2.5_Sb2.5",
                "n_vacancies": 0,
                "energy_above_hull_eV_per_atom": 0.06,
                "energy_above_hull_raw_eV_per_atom": 0.06,
                "energy_above_hull_corrected_eV_per_atom": 0.04,
                "stable": False,
                "stable_raw": False,
                "stable_corrected": False,
                "formula": "CeSbSn38O80",
            },
            {
                "actual_composition_key": "Ce2.5_Sb2.5",
                "n_vacancies": 1,
                "energy_above_hull_eV_per_atom": 0.03,
                "energy_above_hull_raw_eV_per_atom": 0.03,
                "energy_above_hull_corrected_eV_per_atom": 0.01,
                "stable": False,
                "stable_raw": False,
                "stable_corrected": True,
                "formula": "CeSbSn38O79",
            },
        ]
    )

    prepared = prepare_vacancy_hull_data(vacancy, quantity="Corrected")
    figure, shown = build_vacancy_hull_figure(
        prepared,
        selected_compositions=["Ce2.5_Sb2.5"],
    )

    assert shown["energy_above_hull_meV_per_atom"].tolist() == [40.0, 10.0]
    assert bool(shown.iloc[1]["_stable_bool"])
    assert len(figure.data) >= 1

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.phase_diagram_plots import (
    build_phase_diagram_figure,
    find_codoped_system,
    prepare_phase_diagram_plot_data,
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
        "On convex hull",
        "SnO2",
    }


def test_find_codoped_system_selects_the_matching_pair():
    phase, database = _frames()
    prepared, _ = prepare_phase_diagram_plot_data(phase, database)

    assert find_codoped_system(prepared, "Sb", "Ti") == "O-Sb-Sn-Ti"

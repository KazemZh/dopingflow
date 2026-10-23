from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.vacancy_mc_results import (
    lowest_energy_by_vacancy_count,
    preferred_vacancy_heatmap,
    preferred_vacancy_map,
    validate_search_database,
)


def _search_rows() -> pd.DataFrame:
    rows = []
    # At mu_O=-2 eV/O and U=0, n=0, 1 and 2 have grand energies 0, 1 and 2 eV.
    # Increasing U with z=2 eventually favors the more vacancy-rich structures.
    energies = {0: -100.0, 1: -97.0, 2: -94.0}
    for n_vacancies, energy in energies.items():
        rows.append(
            {
                "parent_id": "Sb5/candidate_001",
                "composition": "Sb5",
                "configuration_id": (
                    "parent_reference" if n_vacancies == 0 else f"mc_{n_vacancies:06d}"
                ),
                "n_vacancies": n_vacancies,
                "mc_search_energy_sp_total_eV": energy,
                "mc_search_backend": "grace",
                "mc_search_model": "GRACE-1L-OMAT",
                "search_fingerprint": "same-search",
                "generated_poscar_path": f"V_O_{n_vacancies:02d}/POSCAR",
            }
        )
    # A higher-energy duplicate verifies that only the minimum at each count is used.
    rows.append(
        {
            **rows[1],
            "configuration_id": "mc_999999",
            "mc_search_energy_sp_total_eV": -96.0,
        }
    )
    return pd.DataFrame(rows)


def test_search_database_requires_zero_vacancy_reference():
    frame = _search_rows()
    validate_search_database(frame)
    with pytest.raises(ValueError, match="Zero-vacancy"):
        validate_search_database(frame[frame["n_vacancies"] > 0])


def test_lowest_energy_candidate_is_used_for_each_count():
    minima = lowest_energy_by_vacancy_count(_search_rows(), "Sb5/candidate_001")
    assert minima["n_vacancies"].tolist() == [0, 1, 2]
    assert minima["configuration_id"].tolist() == [
        "parent_reference",
        "mc_000001",
        "mc_000002",
    ]
    assert minima["delta_energy_to_parent_eV"].tolist() == [0.0, 3.0, 6.0]


def test_pressure_potential_map_returns_preferred_vacancy_counts():
    result = preferred_vacancy_map(
        _search_rows(),
        "Sb5/candidate_001",
        temperature_K=1000.0,
        mu_O_reference_eV_per_O=-2.0,
        log10_pO2_min_bar=-1.0,
        log10_pO2_max_bar=1.0,
        potential_min_V=0.0,
        potential_max_V=2.0,
        electrons_released_per_vacancy=2.0,
        include_nist_standard_state=False,
        n_pressure_points=3,
        n_potential_points=3,
    )

    assert result.preferred_n_vacancies.shape == (3, 3)
    assert np.all(result.preferred_n_vacancies[0] == 0)
    assert np.all(result.preferred_n_vacancies[-1] == 2)
    figure = preferred_vacancy_heatmap(
        result,
        parent_id="Sb5/candidate_001",
        temperature_K=1000.0,
        electrons_released_per_vacancy=2.0,
        include_nist_standard_state=False,
    )
    assert figure.data[0].type == "heatmap"
    assert figure.layout.xaxis.title.text == "log10(pO2/bar)"
    assert "Applied potential" in figure.layout.yaxis.title.text


def test_map_rejects_inverted_ranges():
    with pytest.raises(ValueError, match="Minimum log10"):
        preferred_vacancy_map(
            _search_rows(),
            "Sb5/candidate_001",
            temperature_K=1000.0,
            mu_O_reference_eV_per_O=-2.0,
            log10_pO2_min_bar=1.0,
            log10_pO2_max_bar=-1.0,
            potential_min_V=0.0,
            potential_max_V=2.0,
            electrons_released_per_vacancy=2.0,
        )

"""Stage-1 vacancy Monte Carlo result preparation and interactive figures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dopingflow.vacancy_analysis import (
    K_B_EV_PER_K,
    nist_o2_standard_state_delta_mu_eV_per_O,
)

try:  # package import in tests and installed use
    from gui.vacancy_thermo_plots import vacancy_count_color
except ModuleNotFoundError:  # direct Streamlit page execution from gui/
    from vacancy_thermo_plots import vacancy_count_color


REQUIRED_SEARCH_COLUMNS = {
    "parent_id",
    "composition",
    "configuration_id",
    "n_vacancies",
    "mc_search_energy_sp_total_eV",
    "mc_search_backend",
    "mc_search_model",
    "search_fingerprint",
}


@dataclass(frozen=True)
class VacancyPotentialMap:
    """Numerical result used by the Stage-1 pressure-potential heatmap."""

    log10_pO2_bar: np.ndarray
    potential_V: np.ndarray
    preferred_n_vacancies: np.ndarray
    minimum_grand_energy_eV: np.ndarray
    minima: pd.DataFrame
    mu_O_standard_eV_per_O: float
    mu_O_grid_eV_per_O: np.ndarray


def validate_search_database(frame: pd.DataFrame) -> None:
    """Validate fields and the zero-vacancy references needed by the map."""
    missing = sorted(REQUIRED_SEARCH_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "Stage-1 database is missing required column(s): " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("Stage-1 search database is empty")
    if frame["mc_search_energy_sp_total_eV"].isna().any():
        raise ValueError("Stage-1 database contains missing GRACE search energies")

    counts = frame.assign(n_vacancies=frame["n_vacancies"].astype(int)).groupby(
        "parent_id"
    )["n_vacancies"]
    missing_parent = [str(parent) for parent, values in counts if 0 not in set(values)]
    if missing_parent:
        raise ValueError(
            "Zero-vacancy Stage-1 reference is missing for: "
            + ", ".join(missing_parent)
        )

    incompatible = []
    for parent_id, group in frame.groupby("parent_id", sort=False):
        identities = group[
            ["mc_search_backend", "mc_search_model", "search_fingerprint"]
        ].drop_duplicates()
        if len(identities) != 1:
            incompatible.append(str(parent_id))
    if incompatible:
        raise ValueError(
            "Stage-1 calculator/fingerprint is inconsistent within parent(s): "
            + ", ".join(incompatible)
        )


def lowest_energy_by_vacancy_count(
    frame: pd.DataFrame, parent_id: str
) -> pd.DataFrame:
    """Return the lowest Stage-1 single-point structure for every vacancy count."""
    validate_search_database(frame)
    selected = frame[frame["parent_id"].astype(str) == str(parent_id)].copy()
    if selected.empty:
        raise ValueError(f"No Stage-1 rows are available for parent {parent_id}")
    selected["n_vacancies"] = selected["n_vacancies"].astype(int)
    energy_column = "mc_search_energy_sp_total_eV"
    indices = selected.groupby("n_vacancies")[energy_column].idxmin()
    minima = selected.loc[indices].sort_values("n_vacancies").reset_index(drop=True)
    if int(minima.iloc[0]["n_vacancies"]) != 0:
        raise ValueError(f"Parent {parent_id} has no zero-vacancy GRACE reference")
    parent_energy = float(
        minima.loc[minima["n_vacancies"] == 0, energy_column].iloc[0]
    )
    minima["delta_energy_to_parent_eV"] = (
        minima[energy_column].astype(float) - parent_energy
    )
    return minima


def preferred_vacancy_map(
    frame: pd.DataFrame,
    parent_id: str,
    *,
    temperature_K: float,
    mu_O_reference_eV_per_O: float,
    log10_pO2_min_bar: float,
    log10_pO2_max_bar: float,
    potential_min_V: float,
    potential_max_V: float,
    electrons_released_per_vacancy: float,
    standard_pressure_bar: float = 1.0,
    include_nist_standard_state: bool = True,
    n_pressure_points: int = 321,
    n_potential_points: int = 241,
) -> VacancyPotentialMap:
    """Calculate the preferred vacancy count on a pO2-potential grid.

    The screening expression is

    ``DeltaOmega_n = E_n - E_0 + n*mu_O(T,p) - n*z*U``.

    Energies and ``mu_O_reference_eV_per_O`` must use the same calculator.
    The potential term assumes that formation of one vacancy releases ``z``
    electrons to an electron reservoir with ``mu_e = -e U``.
    """
    if temperature_K <= 0:
        raise ValueError("Temperature must be positive")
    if standard_pressure_bar <= 0:
        raise ValueError("Standard oxygen pressure must be positive")
    if log10_pO2_min_bar >= log10_pO2_max_bar:
        raise ValueError("Minimum log10(pO2) must be lower than the maximum")
    if potential_min_V >= potential_max_V:
        raise ValueError("Minimum potential must be lower than the maximum")
    if electrons_released_per_vacancy < 0:
        raise ValueError("Electrons released per vacancy cannot be negative")

    minima = lowest_energy_by_vacancy_count(frame, parent_id)
    counts = minima["n_vacancies"].to_numpy(dtype=int)
    delta_energies = minima["delta_energy_to_parent_eV"].to_numpy(dtype=float)

    log_pressure = np.linspace(
        log10_pO2_min_bar, log10_pO2_max_bar, n_pressure_points
    )
    potential = np.linspace(potential_min_V, potential_max_V, n_potential_points)
    standard_delta = (
        nist_o2_standard_state_delta_mu_eV_per_O(temperature_K)
        if include_nist_standard_state
        else 0.0
    )
    pressure_delta = (
        0.5
        * K_B_EV_PER_K
        * temperature_K
        * np.log(10.0)
        * (log_pressure - np.log10(standard_pressure_bar))
    )
    mu_O = float(mu_O_reference_eV_per_O) + standard_delta + pressure_delta

    grand_energies = (
        delta_energies[:, None, None]
        + counts[:, None, None] * mu_O[None, None, :]
        - counts[:, None, None]
        * float(electrons_released_per_vacancy)
        * potential[None, :, None]
    )
    preferred_indices = grand_energies.argmin(axis=0)
    preferred_counts = counts[preferred_indices]
    minimum = np.take_along_axis(
        grand_energies, preferred_indices[None, :, :], axis=0
    )[0]
    return VacancyPotentialMap(
        log10_pO2_bar=log_pressure,
        potential_V=potential,
        preferred_n_vacancies=preferred_counts,
        minimum_grand_energy_eV=minimum,
        minima=minima,
        mu_O_standard_eV_per_O=standard_delta,
        mu_O_grid_eV_per_O=mu_O,
    )


def preferred_vacancy_heatmap(
    result: VacancyPotentialMap,
    *,
    parent_id: str,
    temperature_K: float,
    electrons_released_per_vacancy: float,
    include_nist_standard_state: bool,
) -> go.Figure:
    """Return a discrete Plotly heatmap of the preferred vacancy number."""
    counts = sorted(int(value) for value in np.unique(result.preferred_n_vacancies))
    count_index = {count: index for index, count in enumerate(counts)}
    categorical = np.vectorize(count_index.__getitem__)(
        result.preferred_n_vacancies
    )
    colorscale: list[list[object]] = []
    for index, count in enumerate(counts):
        colorscale.extend(
            [
                [index / len(counts), vacancy_count_color(count)],
                [(index + 1) / len(counts), vacancy_count_color(count)],
            ]
        )
    custom = np.empty((*result.preferred_n_vacancies.shape, 2), dtype=float)
    custom[:, :, 0] = result.preferred_n_vacancies
    custom[:, :, 1] = result.minimum_grand_energy_eV
    figure = go.Figure(
        go.Heatmap(
            z=categorical,
            x=result.log10_pO2_bar,
            y=result.potential_V,
            customdata=custom,
            zmin=-0.5,
            zmax=len(counts) - 0.5,
            colorscale=colorscale,
            showscale=False,
            hovertemplate=(
                "log10(pO2/bar)=%{x:.2f}<br>Potential=%{y:.3f} V"
                "<br>Preferred vacancies=%{customdata[0]:.0f}"
                "<br>Minimum ΔΩ=%{customdata[1]:.4f} eV<extra></extra>"
            ),
        )
    )
    for count in counts:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 14,
                    "color": vacancy_count_color(count),
                    "symbol": "square",
                },
                name=str(count),
                hoverinfo="skip",
            )
        )
    standard_note = "NIST O2 standard-state correction included" if (
        include_nist_standard_state
    ) else "O2 standard-state correction omitted"
    figure.update_layout(
        title=(
            f"Preferred oxygen-vacancy count — {parent_id}"
            f"<br><sup>T={temperature_K:g} K; z={electrons_released_per_vacancy:g}; "
            f"{standard_note}</sup>"
        ),
        xaxis_title="log10(pO2/bar)",
        yaxis_title="Applied potential U (V; assumed reference)",
        template="plotly_white",
        legend_title="Preferred number of Vₒ",
        margin={"t": 95},
    )
    return figure


__all__ = [
    "VacancyPotentialMap",
    "lowest_energy_by_vacancy_count",
    "preferred_vacancy_heatmap",
    "preferred_vacancy_map",
    "validate_search_database",
]

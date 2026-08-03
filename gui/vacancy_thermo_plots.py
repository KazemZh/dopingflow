"""Interactive Plotly figures for compact vacancy thermodynamic outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


VACANCY_COUNT_COLORS = {
    0: "#d9d9d9",
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    6: "#8c564b",
    7: "#e377c2",
    8: "#bcbd22",
    9: "#17becf",
    10: "#393b79",
}
STATIC_PLOT_NOTE = (
    "Static-lattice approximation: T and p enter only through the O reservoir; "
    "solid vibrational and entropic terms are neglected."
)


def vacancy_count_color(count: int) -> str:
    """Return the same semantic color for a vacancy count in every figure."""
    if count in VACANCY_COUNT_COLORS:
        return VACANCY_COUNT_COLORS[count]
    return px.colors.sample_colorscale("Turbo", [(count * 0.61803398875) % 1.0])[0]


def metadata_label(frame: pd.DataFrame) -> str:
    first = frame.iloc[0]
    parts = []
    for key in ("backend", "model", "task"):
        value = first.get(key, "")
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())
    mode = first.get("oxygen_reference_mode", "unknown")
    verified = first.get("oxygen_reference_verified", "")
    return (
        f"O reference: {mode} (verified={verified}); model: {'/'.join(parts)}"
        f"<br>{STATIC_PLOT_NOTE}"
    )


def pressure_mapping_label(
    frame: pd.DataFrame, include_standard_state: bool = True
) -> tuple[str, str]:
    """Return an accuracy-aware title prefix and gas-reservoir description."""
    first = frame.iloc[0]
    approximation_text = str(first.get("pressure_mapping_approximation", ""))
    approximate = str(first.get("pressure_mapping_is_approximate", "")).lower() in {
        "true",
        "1",
    } or "omitted" in approximation_text.lower()
    mode = str(first.get("oxygen_standard_state_mode", "")).strip().lower()
    source = str(first.get("oxygen_standard_state_source", "")).strip()
    if not include_standard_state:
        return (
            "Approximate ",
            "O2 standard-state term ΔμOstandard(T) intentionally omitted.",
        )
    if mode == "nist_shomate":
        detail = "O2 standard state: NIST Shomate, 1 bar; explicit ZPE excluded."
    elif mode == "user_table":
        detail = "O2 standard state: user-supplied thermal correction."
    elif approximate:
        detail = "Approximate gas mapping: O2 standard-state thermal correction omitted."
    elif source and source.lower() != "none":
        detail = f"O2 standard state: {source}."
    else:
        detail = "O2 standard-state convention unavailable in this result file."
    return ("Approximate " if approximate else "", detail)


def pressure_condition_delta_mu(
    pressure: pd.DataFrame,
    composition: str,
    temperature_K: float,
    log10_pressure_bar: float,
    include_standard_state: bool = True,
) -> float:
    """Return the stored total per-O chemical-potential shift at one condition."""
    selected = pressure[
        (pressure["actual_composition_key"].astype(str) == composition)
        & np.isclose(pressure["temperature_K"].astype(float), temperature_K)
        & np.isclose(
            pressure["log10_oxygen_partial_pressure_bar"].astype(float),
            log10_pressure_bar,
        )
    ]
    if selected.empty:
        raise ValueError(
            f"No pressure-map row is available for {composition} at "
            f"T={temperature_K:g} K and log10(pO2/bar)={log10_pressure_bar:g}"
        )
    column = (
        "delta_mu_O_total_eV_per_O"
        if include_standard_state
        else "delta_mu_O_pressure_eV_per_O"
    )
    return float(selected.iloc[0][column])


def _best_count(minima: pd.DataFrame, delta_mu: float) -> int | None:
    values = (
        minima["grand_potential_intercept_eV"].astype(float)
        + minima["n_vacancies"].astype(float) * delta_mu
    )
    tied = minima.loc[np.isclose(values, values.min()), "n_vacancies"].astype(int)
    return int(tied.iloc[0]) if len(tied) == 1 else None


def _categorical_count_heatmap(
    grid: np.ndarray,
    x: list[object],
    y: list[float],
    counts: list[int],
    *,
    title: str,
    x_title: str,
    y_title: str,
    hovertemplate: str,
) -> go.Figure:
    scale = []
    for index, count in enumerate(counts):
        scale.extend(
            [
                [index / len(counts), vacancy_count_color(count)],
                [(index + 1) / len(counts), vacancy_count_color(count)],
            ]
        )
    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=x,
            y=y,
            zmin=-0.5,
            zmax=len(counts) - 0.5,
            colorscale=scale,
            showscale=False,
            hovertemplate=hovertemplate,
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
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        legend_title="Stable vacancy count",
    )
    return figure


def grand_potential_lines(
    minima: pd.DataFrame, intervals: pd.DataFrame, composition: str
) -> go.Figure:
    data = minima[minima["actual_composition_key"].astype(str) == composition].copy()
    data = data.sort_values("n_vacancies")
    selected = intervals[intervals["actual_composition_key"].astype(str) == composition]
    if data.empty or selected.empty:
        raise ValueError(f"No thermodynamic data are available for {composition}")
    x_min = float(selected["delta_mu_O_lower_eV"].min())
    x_max = float(selected["delta_mu_O_upper_eV"].max())
    x = np.linspace(x_min, x_max, 500)
    figure = go.Figure()
    for item in data.itertuples():
        count = int(item.n_vacancies)
        y = float(item.grand_potential_intercept_eV) + count * x
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"n={count}",
                line={"color": vacancy_count_color(count), "width": 2},
            )
        )
    for boundary in sorted(selected["delta_mu_O_upper_eV"].dropna().unique())[:-1]:
        figure.add_vline(x=float(boundary), line_dash="dash", line_color="gray")
    figure.update_layout(
        title=f"Oxygen grand potential — {composition}<br><sup>{metadata_label(data)}</sup>",
        xaxis_title="ΔμO (eV per O atom)",
        yaxis_title="ΔΩ (eV)",
        template="plotly_white",
        legend_title="Vacancy count",
    )
    return figure


def stability_map(intervals: pd.DataFrame, x_dopant: str | None) -> go.Figure:
    x_column = f"percent_{x_dopant}" if x_dopant else "total_dopant_percent"
    compositions = (
        intervals[["actual_composition_key", x_column]]
        .drop_duplicates()
        .sort_values([x_column, "actual_composition_key"])
    )
    y = np.linspace(
        float(intervals["delta_mu_O_lower_eV"].min()),
        float(intervals["delta_mu_O_upper_eV"].max()),
        400,
    )
    counts = sorted(int(value) for value in intervals["stable_n_vacancies"].dropna().unique())
    count_index = {count: index for index, count in enumerate(counts)}
    grid = np.full((len(y), len(compositions)), np.nan)
    for column, composition in enumerate(compositions["actual_composition_key"]):
        subset = intervals[intervals["actual_composition_key"] == composition]
        for item in subset.itertuples():
            mask = (y >= item.delta_mu_O_lower_eV) & (y <= item.delta_mu_O_upper_eV)
            grid[mask, column] = count_index[int(item.stable_n_vacancies)]
    scale = []
    for index, count in enumerate(counts):
        scale.extend(
            [
                [index / len(counts), vacancy_count_color(count)],
                [(index + 1) / len(counts), vacancy_count_color(count)],
            ]
        )
    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=compositions["actual_composition_key"],
            y=y,
            zmin=-0.5,
            zmax=len(counts) - 0.5,
            colorscale=scale,
            showscale=False,
            hovertemplate="Composition=%{x}<br>ΔμO=%{y:.3f} eV<extra></extra>",
        )
    )
    for count in counts:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 14, "color": vacancy_count_color(count), "symbol": "square"},
                name=str(count),
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        title=f"Stable oxygen-vacancy count<br><sup>{metadata_label(intervals)}</sup>",
        xaxis_title="Actual composition",
        yaxis_title="ΔμO (eV per O atom)",
        template="plotly_white",
        legend_title="Stable vacancy count",
    )
    return figure


def grand_potential_vs_count(
    minima: pd.DataFrame, composition: str, delta_mu_o: float
) -> go.Figure:
    data = minima[minima["actual_composition_key"].astype(str) == composition].copy()
    data = data.sort_values("n_vacancies")
    data["delta_grand_potential_eV"] = (
        data["grand_potential_intercept_eV"] + data["n_vacancies"] * delta_mu_o
    )
    minimum = float(data["delta_grand_potential_eV"].min())
    data["status"] = np.where(
        np.isclose(data["delta_grand_potential_eV"], minimum), "Preferred", "Other"
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["n_vacancies"],
            y=data["delta_grand_potential_eV"],
            mode="lines",
            line={"color": "#7f7f7f", "width": 2},
            name="Grand-potential curve",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for status, color in (("Other", "#1f77b4"), ("Preferred", "#d62728")):
        selected = data[data["status"] == status]
        customdata = selected[
            ["source_parent_id", "source_configuration_id", "converged"]
        ].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=selected["n_vacancies"],
                y=selected["delta_grand_potential_eV"],
                mode="markers",
                marker={"color": color, "size": 10},
                name=status,
                customdata=customdata,
                hovertemplate=(
                    "Vacancies=%{x}<br>ΔΩ=%{y:.6f} eV"
                    "<br>Parent=%{customdata[0]}"
                    "<br>Configuration=%{customdata[1]}"
                    "<br>Converged=%{customdata[2]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=f"{composition} at ΔμO={delta_mu_o:g} eV<br><sup>{metadata_label(data)}</sup>",
        xaxis_title="Number of oxygen vacancies",
        yaxis_title="ΔΩ (eV)",
        template="plotly_white",
    )
    return figure


def preferred_count_vs_doping(
    best: pd.DataFrame, delta_mu_o: float, x_dopant: str | None
) -> go.Figure:
    selected = best[np.isclose(best["delta_mu_O_eV"], delta_mu_o)].copy()
    tied = selected["is_tied"].astype(str).str.lower().isin({"true", "1"})
    selected = selected[~tied]
    x_column = f"percent_{x_dopant}" if x_dopant else "total_dopant_percent"
    other = sorted(
        column
        for column in selected
        if column.startswith("percent_")
        and column != x_column
        and selected[column].nunique() > 1
    )
    color = other[0] if other else None
    figure = px.line(
        selected.sort_values(x_column),
        x=x_column,
        y="best_n_vacancies",
        color=color,
        markers=True,
        hover_data=["actual_composition_key", "source_parent_id", "converged"],
    )
    figure.update_layout(
        title=(
            f"Preferred vacancy count at ΔμO={delta_mu_o:g} eV"
            f"<br><sup>{metadata_label(selected)}</sup>"
        ),
        xaxis_title=f"Actual {x_dopant or 'total dopant'} percentage (%)",
        yaxis_title="Preferred oxygen-vacancy count",
        template="plotly_white",
    )
    return figure


def pressure_stability_vs_composition(
    minima: pd.DataFrame,
    pressure: pd.DataFrame,
    temperature_K: float,
    x_dopant: str | None,
    include_standard_state: bool = True,
) -> go.Figure:
    """Plot stable counts by composition and oxygen pressure at fixed T."""
    selected = pressure[
        np.isclose(pressure["temperature_K"].astype(float), temperature_K)
    ].copy()
    if selected.empty:
        raise ValueError(f"No pressure-map data are available at T={temperature_K:g} K")
    x_column = f"percent_{x_dopant}" if x_dopant else "total_dopant_percent"
    compositions = (
        selected[["actual_composition_key", x_column]]
        .drop_duplicates()
        .sort_values([x_column, "actual_composition_key"])
    )
    pressures = sorted(
        float(value)
        for value in selected["log10_oxygen_partial_pressure_bar"].unique()
    )
    counts = sorted(int(value) for value in minima["n_vacancies"].dropna().unique())
    count_index = {count: index for index, count in enumerate(counts)}
    grid = np.full((len(pressures), len(compositions)), np.nan)
    composition_index = {
        str(value): index
        for index, value in enumerate(compositions["actual_composition_key"])
    }
    pressure_index = {value: index for index, value in enumerate(pressures)}
    for row in selected.itertuples():
        composition = str(row.actual_composition_key)
        lines = minima[minima["actual_composition_key"].astype(str) == composition]
        delta_mu = float(
            row.delta_mu_O_total_eV_per_O
            if include_standard_state
            else row.delta_mu_O_pressure_eV_per_O
        )
        best = _best_count(lines, delta_mu)
        if best is None:
            continue
        grid[
            pressure_index[float(row.log10_oxygen_partial_pressure_bar)],
            composition_index[composition],
        ] = count_index[best]
    prefix, gas_note = pressure_mapping_label(selected, include_standard_state)
    return _categorical_count_heatmap(
        grid,
        compositions["actual_composition_key"].astype(str).tolist(),
        pressures,
        counts,
        title=(
            f"{prefix}Stable oxygen-vacancy count at T={temperature_K:g} K"
            f"<br><sup>{metadata_label(selected)}<br>{gas_note}</sup>"
        ),
        x_title="Actual composition",
        y_title="log10(pO2/bar)",
        hovertemplate=(
            "Composition=%{x}<br>log10(pO2/bar)=%{y:g}<extra></extra>"
        ),
    )


def grand_potential_lines_pressure(
    minima: pd.DataFrame,
    pressure: pd.DataFrame,
    composition: str,
    temperature_K: float,
    include_standard_state: bool = True,
) -> go.Figure:
    """Plot every vacancy-count grand potential against pO2 at fixed T."""
    data = minima[minima["actual_composition_key"].astype(str) == composition].copy()
    conditions = pressure[
        (pressure["actual_composition_key"].astype(str) == composition)
        & np.isclose(pressure["temperature_K"].astype(float), temperature_K)
    ].sort_values("log10_oxygen_partial_pressure_bar")
    if data.empty or conditions.empty:
        raise ValueError(
            f"No thermodynamic pressure data are available for {composition} "
            f"at T={temperature_K:g} K"
        )
    x = conditions["log10_oxygen_partial_pressure_bar"].astype(float).to_numpy()
    delta_column = (
        "delta_mu_O_total_eV_per_O"
        if include_standard_state
        else "delta_mu_O_pressure_eV_per_O"
    )
    delta_mu = conditions[delta_column].astype(float).to_numpy()
    figure = go.Figure()
    for item in data.sort_values("n_vacancies").itertuples():
        count = int(item.n_vacancies)
        figure.add_trace(
            go.Scatter(
                x=x,
                y=float(item.grand_potential_intercept_eV) + count * delta_mu,
                mode="lines",
                name=f"n={count}",
                line={"color": vacancy_count_color(count), "width": 2},
                customdata=delta_mu,
                hovertemplate=(
                    "log10(pO2/bar)=%{x:g}<br>ΔμO=%{customdata:.4f} eV/O"
                    "<br>ΔΩ=%{y:.6f} eV<extra></extra>"
                ),
            )
        )
    prefix, gas_note = pressure_mapping_label(conditions, include_standard_state)
    figure.update_layout(
        title=(
            f"{prefix}Oxygen grand potential — {composition}, T={temperature_K:g} K"
            f"<br><sup>{metadata_label(data)}<br>{gas_note}</sup>"
        ),
        xaxis_title="log10(pO2/bar)",
        yaxis_title="ΔΩ (eV)",
        template="plotly_white",
        legend_title="Vacancy count",
    )
    return figure


def grand_potential_vs_count_pressure(
    minima: pd.DataFrame,
    pressure: pd.DataFrame,
    composition: str,
    temperature_K: float,
    log10_pressure_bar: float,
    include_standard_state: bool = True,
) -> go.Figure:
    """Plot grand potential by vacancy count at a physical T-pO2 condition."""
    delta_mu = pressure_condition_delta_mu(
        pressure,
        composition,
        temperature_K,
        log10_pressure_bar,
        include_standard_state,
    )
    figure = grand_potential_vs_count(minima, composition, delta_mu)
    selected = pressure[
        (pressure["actual_composition_key"].astype(str) == composition)
        & np.isclose(pressure["temperature_K"].astype(float), temperature_K)
    ]
    prefix, gas_note = pressure_mapping_label(selected, include_standard_state)
    figure.update_layout(
        title=(
            f"{prefix}{composition} at T={temperature_K:g} K, "
            f"log10(pO2/bar)={log10_pressure_bar:g}"
            f"<br><sup>ΔμO={delta_mu:.4f} eV/O; {gas_note}</sup>"
        )
    )
    return figure


def preferred_count_vs_doping_pressure(
    minima: pd.DataFrame,
    pressure: pd.DataFrame,
    temperature_K: float,
    log10_pressure_bar: float,
    x_dopant: str | None,
    include_standard_state: bool = True,
) -> go.Figure:
    """Plot preferred vacancy count versus doping at one T-pO2 condition."""
    selected = pressure[
        np.isclose(pressure["temperature_K"].astype(float), temperature_K)
        & np.isclose(
            pressure["log10_oxygen_partial_pressure_bar"].astype(float),
            log10_pressure_bar,
        )
    ].copy()
    if selected.empty:
        raise ValueError("No pressure-map data are available at the selected condition")
    delta_column = (
        "delta_mu_O_total_eV_per_O"
        if include_standard_state
        else "delta_mu_O_pressure_eV_per_O"
    )
    delta_mu = float(selected.iloc[0][delta_column])
    rows = []
    for composition, lines in minima.groupby("actual_composition_key", sort=False):
        best = _best_count(lines, delta_mu)
        if best is None:
            continue
        representative = lines[
            lines["n_vacancies"].astype(int) == best
        ].iloc[0].to_dict()
        condition_row = selected[
            selected["actual_composition_key"].astype(str) == str(composition)
        ]
        if not condition_row.empty:
            for column, value in condition_row.iloc[0].items():
                if column.startswith("percent_") or column == "total_dopant_percent":
                    representative[column] = value
        representative["best_n_vacancies"] = best
        representative["delta_mu_O_effective_eV_per_O"] = delta_mu
        rows.append(representative)
    selected = pd.DataFrame(rows)
    if selected.empty:
        raise ValueError("Every composition is tied at the selected condition")
    x_column = f"percent_{x_dopant}" if x_dopant else "total_dopant_percent"
    other = sorted(
        column
        for column in selected
        if column.startswith("percent_")
        and column != x_column
        and selected[column].nunique() > 1
    )
    color = other[0] if other else None
    figure = px.line(
        selected.sort_values(x_column),
        x=x_column,
        y="best_n_vacancies",
        color=color,
        markers=True,
        hover_data=[
            "actual_composition_key",
            "delta_mu_O_effective_eV_per_O",
            "source_parent_id",
            "converged",
        ],
    )
    prefix, gas_note = pressure_mapping_label(pressure, include_standard_state)
    figure.update_layout(
        title=(
            f"{prefix}Preferred vacancy count at T={temperature_K:g} K, "
            f"log10(pO2/bar)={log10_pressure_bar:g}"
            f"<br><sup>{gas_note}</sup>"
        ),
        xaxis_title=f"Actual {x_dopant or 'total dopant'} percentage (%)",
        yaxis_title="Preferred oxygen-vacancy count",
        template="plotly_white",
    )
    return figure


def pressure_stability_map(
    pressure: pd.DataFrame,
    composition: str,
    minima: pd.DataFrame | None = None,
    include_standard_state: bool = True,
) -> go.Figure:
    selected = pressure[
        pressure["actual_composition_key"].astype(str) == composition
    ].copy()
    if selected.empty:
        raise ValueError(f"No pressure-map data are available for {composition}")
    temperatures = sorted(float(value) for value in selected["temperature_K"].unique())
    pressures = sorted(
        float(value)
        for value in selected["log10_oxygen_partial_pressure_bar"].unique()
    )
    composition_minima = (
        minima[minima["actual_composition_key"].astype(str) == composition]
        if minima is not None
        else None
    )
    counts = sorted(
        int(value)
        for value in (
            composition_minima["n_vacancies"]
            if composition_minima is not None
            else selected["best_n_vacancies"]
        ).dropna().unique()
    )
    count_index = {count: index for index, count in enumerate(counts)}
    grid = np.full((len(temperatures), len(pressures)), np.nan)
    for row in selected.itertuples():
        if composition_minima is not None:
            delta_mu = float(
                row.delta_mu_O_total_eV_per_O
                if include_standard_state
                else row.delta_mu_O_pressure_eV_per_O
            )
            best = _best_count(composition_minima, delta_mu)
        else:
            best = None if pd.isna(row.best_n_vacancies) else int(row.best_n_vacancies)
        if best is None:
            continue
        y_index = temperatures.index(float(row.temperature_K))
        x_index = pressures.index(float(row.log10_oxygen_partial_pressure_bar))
        grid[y_index, x_index] = count_index[best]
    scale = []
    for index, count in enumerate(counts):
        scale.extend(
            [
                [index / len(counts), vacancy_count_color(count)],
                [(index + 1) / len(counts), vacancy_count_color(count)],
            ]
        )
    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=pressures,
            y=temperatures,
            zmin=-0.5,
            zmax=len(counts) - 0.5,
            colorscale=scale,
            showscale=False,
            hovertemplate=(
                "log10(pO2/bar)=%{x:g}<br>T=%{y:g} K<extra></extra>"
            ),
        )
    )
    for count in counts:
        figure.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker={"size": 14, "color": vacancy_count_color(count), "symbol": "square"},
                name=str(count), hoverinfo="skip",
            )
        )
    title_prefix, gas_note = pressure_mapping_label(
        selected, include_standard_state
    )
    figure.update_layout(
        title=(
            f"{title_prefix}T–pO2 vacancy stability — {composition}"
            f"<br><sup>{metadata_label(selected)}<br>{gas_note}</sup>"
        ),
        xaxis_title="log10(pO2/bar)",
        yaxis_title="Temperature (K)",
        template="plotly_white",
        legend_title="Stable vacancy count",
    )
    return figure

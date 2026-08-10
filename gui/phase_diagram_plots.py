"""Interactive phase-diagram composition plots for the Results Explorer."""

from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


ENERGY_COLUMN = "energy_above_hull_eV_per_atom"


def _percentage_mapping(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(raw, dict):
        return {}

    result: dict[str, float] = {}
    for element, percentage in raw.items():
        try:
            numeric = float(percentage)
        except (TypeError, ValueError):
            continue
        if pd.notna(numeric):
            result[str(element)] = numeric
    return result


def _stable_mask(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def prepare_phase_diagram_plot_data(
    phase_results: pd.DataFrame,
    results_database: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Join hull results to effective dopant percentages from the main database."""
    required_phase = {"candidate_path", "chemical_system", ENERGY_COLUMN, "stable"}
    missing_phase = sorted(required_phase - set(phase_results.columns))
    if missing_phase:
        raise ValueError(
            "Phase-diagram results are missing required columns: "
            + ", ".join(missing_phase)
        )

    percentage_column = next(
        (
            column
            for column in ("effective_pct_json", "requested_pct_json")
            if column in results_database.columns
        ),
        None,
    )
    if percentage_column is None or "candidate_path" not in results_database.columns:
        raise ValueError(
            "results_database.csv must contain candidate_path and either "
            "effective_pct_json or requested_pct_json"
        )

    metadata_columns = list(
        dict.fromkeys(
            column
            for column in (
                "candidate_path",
                percentage_column,
                "dopant_counts_json",
                "requested_pct_json",
                "effective_pct_json",
            )
            if column in results_database.columns
        )
    )
    metadata = results_database[metadata_columns].drop_duplicates(
        subset=["candidate_path"], keep="last"
    )
    joined = phase_results.merge(
        metadata,
        on="candidate_path",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int((joined["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(
            f"Could not match {unmatched} phase-diagram row(s) to results_database.csv"
        )
    joined = joined.drop(columns="_merge")
    joined["_dopant_percentages"] = joined[percentage_column].apply(
        _percentage_mapping
    )
    dopants = sorted(
        {
            element
            for mapping in joined["_dopant_percentages"]
            for element in mapping
        }
    )
    if not dopants:
        raise ValueError("No dopant percentages were found in the matched database rows")

    for element in dopants:
        joined[f"percent_{element}"] = joined["_dopant_percentages"].apply(
            lambda mapping, symbol=element: float(mapping.get(symbol, 0.0))
        )
    joined["energy_above_hull_meV_per_atom"] = (
        pd.to_numeric(joined[ENERGY_COLUMN], errors="coerce") * 1000.0
    )
    joined["_stable_bool"] = _stable_mask(joined["stable"])
    joined = joined.dropna(subset=["energy_above_hull_meV_per_atom"])
    return joined, dopants


def _format_percentage(value: float) -> str:
    return f"{value:g}%"


def filter_compatible_systems(
    data: pd.DataFrame,
    chemical_system: str,
) -> pd.DataFrame:
    """Return the selected exact system and its lower-dimensional subsystems."""
    target_elements = frozenset(chemical_system.split("-"))
    mask = data["chemical_system"].astype(str).apply(
        lambda label: frozenset(label.split("-")).issubset(target_elements)
    )
    return data[mask].copy()


def find_codoped_system(
    data: pd.DataFrame,
    dopant_x: str,
    dopant_fixed: str,
) -> str:
    """Find the smallest available chemical system containing both dopants."""
    matches = []
    for label in data["chemical_system"].dropna().astype(str).unique():
        elements = frozenset(label.split("-"))
        if {dopant_x, dopant_fixed}.issubset(elements):
            matches.append((len(elements), label))
    if not matches:
        raise ValueError(
            f"No phase-diagram system contains both {dopant_x} and {dopant_fixed}"
        )
    return min(matches)[1]


def build_phase_diagram_figure(
    data: pd.DataFrame,
    *,
    chemical_system: str,
    x_dopant: str,
    series_dopant: str | None,
    selected_series: Iterable[float] | None = None,
    show_all_candidates: bool = False,
    include_host_reference: bool = True,
    host_formula: str = "Host",
) -> tuple[go.Figure, pd.DataFrame]:
    """Plot the lowest-hull-energy candidate for each dopant composition."""
    x_column = f"percent_{x_dopant}"
    if x_column not in data:
        raise ValueError(f"No percentage data are available for {x_dopant}")

    plot_data = filter_compatible_systems(data, chemical_system)
    if plot_data.empty:
        raise ValueError(f"No rows are available for chemical system {chemical_system}")

    if series_dopant is None:
        series_column = "_series_percentage"
        plot_data[series_column] = 0.0
        series_title = "Series"
    else:
        series_column = f"percent_{series_dopant}"
        if series_column not in plot_data:
            raise ValueError(f"No percentage data are available for {series_dopant}")
        series_title = f"{series_dopant} concentration"

    plot_data[x_column] = pd.to_numeric(plot_data[x_column], errors="coerce")
    plot_data[series_column] = pd.to_numeric(plot_data[series_column], errors="coerce")
    plot_data = plot_data.dropna(
        subset=[x_column, series_column, "energy_above_hull_meV_per_atom"]
    )
    if selected_series is not None:
        allowed = {float(value) for value in selected_series}
        plot_data = plot_data[plot_data[series_column].isin(allowed)]
    if plot_data.empty:
        raise ValueError("No data remain after applying the series selection")

    group_columns = [x_column, series_column]
    best_indices = plot_data.groupby(group_columns, dropna=False)[
        "energy_above_hull_meV_per_atom"
    ].idxmin()
    best = plot_data.loc[best_indices].sort_values(group_columns).copy()

    palette = qualitative.Safe + qualitative.Plotly + qualitative.Dark24
    fig = go.Figure()
    series_values = sorted(float(value) for value in best[series_column].unique())
    hover_columns = [
        column
        for column in (
            "candidate",
            "formula",
            "candidate_path",
            "decomposition",
        )
        if column in best
    ]

    for index, series_value in enumerate(series_values):
        series_best = best[best[series_column] == series_value].sort_values(x_column)
        if series_dopant is None:
            trace_name = f"Best {x_dopant} structures"
        else:
            trace_name = f"{series_dopant} = {_format_percentage(series_value)}"

        customdata = series_best[hover_columns].astype(str).to_numpy()
        hover_lines = [
            f"{x_dopant}: %{{x:g}}%",
            "Energy above hull: %{y:.4g} meV/atom",
        ]
        if series_dopant is not None:
            hover_lines.insert(1, f"{series_dopant}: {_format_percentage(series_value)}")
        hover_lines.extend(
            f"{column}: %{{customdata[{column_index}]}}"
            for column_index, column in enumerate(hover_columns)
        )
        fig.add_trace(
            go.Scatter(
                x=series_best[x_column],
                y=series_best["energy_above_hull_meV_per_atom"],
                mode="lines+markers",
                name=trace_name,
                customdata=customdata,
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                line={"color": palette[index % len(palette)], "width": 2.2},
                marker={"symbol": "square", "size": 8},
            )
        )

        if show_all_candidates:
            all_series = plot_data[plot_data[series_column] == series_value]
            fig.add_trace(
                go.Scatter(
                    x=all_series[x_column],
                    y=all_series["energy_above_hull_meV_per_atom"],
                    mode="markers",
                    name=f"All candidates ({trace_name})",
                    legendgroup=trace_name,
                    showlegend=False,
                    hoverinfo="skip",
                    marker={
                        "color": palette[index % len(palette)],
                        "size": 5,
                        "opacity": 0.22,
                    },
                )
            )

    stable_best = best[best["_stable_bool"]]
    if not stable_best.empty:
        fig.add_trace(
            go.Scatter(
                x=stable_best[x_column],
                y=stable_best["energy_above_hull_meV_per_atom"],
                mode="markers",
                name="On convex hull",
                customdata=stable_best[["candidate", "formula"]].astype(str).to_numpy(),
                hovertemplate=(
                    f"{x_dopant}: %{{x:g}}%<br>"
                    "Energy above hull: %{y:.4g} meV/atom<br>"
                    "candidate: %{customdata[0]}<br>formula: %{customdata[1]}"
                    "<extra></extra>"
                ),
                marker={
                    "symbol": "star-open",
                    "size": 15,
                    "color": "#111827",
                    "line": {"color": "#111827", "width": 1.5},
                },
            )
        )

    if include_host_reference:
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers+text",
                name=host_formula,
                text=[host_formula],
                textposition="top right",
                hovertemplate=f"{host_formula} host reference<extra></extra>",
                marker={
                    "symbol": "square-open",
                    "size": 13,
                    "color": "#111827",
                    "line": {"color": "#111827", "width": 1.5},
                },
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=680,
        title={
            "text": (
                f"Energy above hull across {x_dopant}"
                + (f"–{series_dopant} co-doping" if series_dopant else " doping")
                + f"<br><sup>{chemical_system} and compatible subsystems; "
                "minimum-energy candidate at each composition</sup>"
            )
        },
        xaxis_title=f"{x_dopant} concentration (%)",
        yaxis_title="Energy above hull (meV/atom)",
        legend_title_text=series_title,
        hovermode="closest",
        margin={"l": 70, "r": 25, "t": 85, "b": 65},
        font={"size": 16},
        legend={"font": {"size": 12}, "bordercolor": "#D1D5DB", "borderwidth": 1},
    )
    max_x = float(best[x_column].max())
    x_upper = max(5.0, 5.0 * ((max_x + 4.999999) // 5.0))
    max_y = float(best["energy_above_hull_meV_per_atom"].max())
    y_upper = max(10.0, 5.0 * ((max_y + 4.999999) // 5.0))
    fig.update_xaxes(
        range=[0.0, x_upper],
        dtick=5.0,
        ticks="inside",
        mirror=True,
        showline=True,
        linecolor="#111827",
        minor={"ticks": "inside", "dtick": 2.5},
    )
    fig.update_yaxes(
        range=[0.0, y_upper],
        ticks="inside",
        mirror=True,
        showline=True,
        linecolor="#111827",
        minor={"ticks": "inside"},
        zeroline=True,
        zerolinecolor="#111827",
        zerolinewidth=1,
    )
    return fig, best

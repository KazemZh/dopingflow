"""Interactive phase-diagram plots used by the Streamlit GUI.

The helpers deliberately separate two related views:

* composition-space energy above hull for the ordinary phase-diagram output;
* energy above hull versus oxygen-vacancy count for ``vacancy_energy_above_hull.csv``.

The phase-diagram metadata join is path-portable: copied calculations may retain
absolute paths from another machine, so matching first uses the full path and
then falls back to the final ``composition/candidate`` path components.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


ENERGY_COLUMN = "energy_above_hull_eV_per_atom"
RAW_ENERGY_COLUMN = "energy_above_hull_raw_eV_per_atom"
CORRECTED_ENERGY_COLUMN = "energy_above_hull_corrected_eV_per_atom"


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


def _path_tail_key(value: Any, depth: int = 2) -> str:
    """Return a platform-independent tail key for a candidate directory path."""
    text = str(value or "").strip().replace("\\", "/").rstrip("/")
    if not text:
        return ""
    parts = [part for part in text.split("/") if part]
    return "/".join(parts[-depth:])


def _choose_hull_columns(
    frame: pd.DataFrame,
    quantity: str,
) -> tuple[str, str | None, str | None]:
    requested = str(quantity).strip().lower()
    if requested == "corrected" and CORRECTED_ENERGY_COLUMN in frame.columns:
        return (
            CORRECTED_ENERGY_COLUMN,
            "stable_corrected" if "stable_corrected" in frame.columns else None,
            "decomposition_corrected"
            if "decomposition_corrected" in frame.columns
            else None,
        )
    if requested == "raw" and RAW_ENERGY_COLUMN in frame.columns:
        return (
            RAW_ENERGY_COLUMN,
            "stable_raw" if "stable_raw" in frame.columns else None,
            "decomposition_raw" if "decomposition_raw" in frame.columns else None,
        )
    return (
        ENERGY_COLUMN,
        "stable" if "stable" in frame.columns else None,
        "decomposition" if "decomposition" in frame.columns else None,
    )


def available_hull_quantities(frame: pd.DataFrame) -> list[str]:
    """Return the hull-energy choices actually present in an output table."""
    options = ["Raw"]
    if CORRECTED_ENERGY_COLUMN in frame.columns:
        options.append("Corrected")
    return options


def select_hull_quantity(frame: pd.DataFrame, quantity: str) -> pd.DataFrame:
    """Return a copy exposing the requested hull through the legacy column names."""
    output = frame.copy()
    energy_column, stable_column, decomposition_column = _choose_hull_columns(
        output, quantity
    )
    output[ENERGY_COLUMN] = pd.to_numeric(output[energy_column], errors="coerce")
    if stable_column is not None:
        output["stable"] = output[stable_column]
    elif "stable" not in output.columns:
        output["stable"] = False
    if decomposition_column is not None:
        output["decomposition"] = output[decomposition_column]
    elif "decomposition" not in output.columns:
        output["decomposition"] = ""
    output["_hull_quantity"] = str(quantity).strip().capitalize()
    return output


def prepare_phase_diagram_plot_data(
    phase_results: pd.DataFrame,
    results_database: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Join phase results to dopant percentages from ``results_database.csv``.

    Exact absolute paths are preferred.  If results were copied from another
    machine, the final ``composition/candidate`` path components are used as a
    fallback.  Phase rows that represent vacancy configurations are allowed to
    remain unmatched; they are visualized separately by the vacancy-hull plot.
    """
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
    metadata = results_database[metadata_columns].copy()
    metadata["_candidate_path_text"] = metadata["candidate_path"].astype(str)
    metadata["_candidate_tail_key"] = metadata["candidate_path"].apply(_path_tail_key)
    metadata = metadata.drop_duplicates(subset=["_candidate_tail_key"], keep="last")

    joined = phase_results.copy()
    joined["_phase_path_text"] = joined["candidate_path"].astype(str)
    joined["_candidate_tail_key"] = joined["candidate_path"].apply(_path_tail_key)

    # Portable matching by final composition/candidate path components.  This
    # also works when the absolute prefix was generated on another machine.
    metadata_for_merge = metadata.drop(columns=["candidate_path"]).rename(
        columns={"_candidate_path_text": "_database_path_text"}
    )
    joined = joined.merge(
        metadata_for_merge,
        on="_candidate_tail_key",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = int((joined["_merge"] != "both").sum())
    joined.attrs["unmatched_phase_rows"] = unmatched
    joined = joined[joined["_merge"] == "both"].drop(columns="_merge")
    if joined.empty:
        raise ValueError(
            "No phase-diagram candidates could be matched to results_database.csv. "
            "The GUI now tolerates copied absolute paths, so verify that both files "
            "belong to the same workflow calculation."
        )

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
    target_elements = frozenset(str(chemical_system).split("-"))
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
    """Plot minimum energy above hull versus one dopant concentration."""
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
        stable_hover = [column for column in ("candidate", "formula") if column in stable_best]
        customdata = stable_best[stable_hover].astype(str).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=stable_best[x_column],
                y=stable_best["energy_above_hull_meV_per_atom"],
                mode="markers",
                name="Stable / within threshold",
                customdata=customdata,
                hovertemplate=(
                    f"{x_dopant}: %{{x:g}}%<br>"
                    "Energy above hull: %{y:.4g} meV/atom<extra></extra>"
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
                + f"<br><sup>{chemical_system}; minimum candidate at each plotted composition</sup>"
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
    fig.update_xaxes(range=[0.0, x_upper], ticks="inside", mirror=True, showline=True)
    fig.update_yaxes(
        range=[0.0, y_upper],
        ticks="inside",
        mirror=True,
        showline=True,
        zeroline=True,
    )
    return fig, best


def build_phase_composition_map(
    data: pd.DataFrame,
    *,
    chemical_system: str,
    x_dopant: str,
    y_dopant: str,
) -> tuple[go.Figure, pd.DataFrame]:
    """Plot a two-dopant composition map coloured by energy above hull."""
    if x_dopant == y_dopant:
        raise ValueError("Choose different dopants for the x and y axes")
    x_column = f"percent_{x_dopant}"
    y_column = f"percent_{y_dopant}"
    for column, dopant in ((x_column, x_dopant), (y_column, y_dopant)):
        if column not in data.columns:
            raise ValueError(f"No percentage data are available for {dopant}")

    plot_data = filter_compatible_systems(data, chemical_system)
    plot_data[x_column] = pd.to_numeric(plot_data[x_column], errors="coerce")
    plot_data[y_column] = pd.to_numeric(plot_data[y_column], errors="coerce")
    plot_data = plot_data.dropna(
        subset=[x_column, y_column, "energy_above_hull_meV_per_atom"]
    )
    if plot_data.empty:
        raise ValueError("No phase-diagram rows remain for the selected composition map")

    best_indices = plot_data.groupby([x_column, y_column], dropna=False)[
        "energy_above_hull_meV_per_atom"
    ].idxmin()
    best = plot_data.loc[best_indices].copy().sort_values([y_column, x_column])

    hover_columns = [
        column
        for column in ("candidate", "formula", "decomposition", "candidate_path")
        if column in best.columns
    ]
    customdata = best[hover_columns].astype(str).to_numpy()
    hover_lines = [
        f"{x_dopant}: %{{x:g}}%",
        f"{y_dopant}: %{{y:g}}%",
        "Energy above hull: %{marker.color:.4g} meV/atom",
    ]
    hover_lines.extend(
        f"{column}: %{{customdata[{index}]}}"
        for index, column in enumerate(hover_columns)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=best[x_column],
            y=best[y_column],
            mode="markers",
            name="Minimum candidate",
            customdata=customdata,
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            marker={
                "size": 13,
                "color": best["energy_above_hull_meV_per_atom"],
                "colorscale": "Viridis",
                "colorbar": {"title": "E above hull<br>(meV/atom)"},
                "showscale": True,
                "line": {"width": 0.8, "color": "#374151"},
            },
        )
    )

    stable = best[best["_stable_bool"]]
    if not stable.empty:
        fig.add_trace(
            go.Scatter(
                x=stable[x_column],
                y=stable[y_column],
                mode="markers",
                name="Stable / within threshold",
                hoverinfo="skip",
                marker={
                    "symbol": "star-open",
                    "size": 19,
                    "color": "#111827",
                    "line": {"width": 1.5, "color": "#111827"},
                },
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=700,
        title=(
            f"Energy above hull composition map: {x_dopant}–{y_dopant}"
            f"<br><sup>{chemical_system}; minimum candidate at each composition</sup>"
        ),
        xaxis_title=f"{x_dopant} concentration (%)",
        yaxis_title=f"{y_dopant} concentration (%)",
        hovermode="closest",
        margin={"l": 70, "r": 40, "t": 85, "b": 65},
        font={"size": 15},
    )
    fig.update_xaxes(ticks="inside", mirror=True, showline=True, rangemode="tozero")
    fig.update_yaxes(ticks="inside", mirror=True, showline=True, rangemode="tozero")
    return fig, best


def prepare_vacancy_hull_data(
    vacancy_results: pd.DataFrame,
    *,
    quantity: str = "Corrected",
) -> pd.DataFrame:
    """Prepare ``vacancy_energy_above_hull.csv`` for plotting."""
    required = {"actual_composition_key", "n_vacancies"}
    missing = sorted(required - set(vacancy_results.columns))
    if missing:
        raise ValueError(
            "Vacancy hull results are missing required columns: " + ", ".join(missing)
        )
    selected = select_hull_quantity(vacancy_results, quantity)
    selected["n_vacancies"] = pd.to_numeric(selected["n_vacancies"], errors="coerce")
    selected["energy_above_hull_meV_per_atom"] = (
        pd.to_numeric(selected[ENERGY_COLUMN], errors="coerce") * 1000.0
    )
    selected["_stable_bool"] = _stable_mask(selected["stable"])
    return selected.dropna(
        subset=["n_vacancies", "energy_above_hull_meV_per_atom"]
    ).copy()


def build_vacancy_hull_figure(
    data: pd.DataFrame,
    *,
    selected_compositions: Iterable[str] | None = None,
) -> tuple[go.Figure, pd.DataFrame]:
    """Plot energy above hull against oxygen-vacancy count."""
    plot_data = data.copy()
    if selected_compositions is not None:
        allowed = {str(value) for value in selected_compositions}
        plot_data = plot_data[
            plot_data["actual_composition_key"].astype(str).isin(allowed)
        ]
    if plot_data.empty:
        raise ValueError("No vacancy-hull rows remain for the selected compositions")

    palette = qualitative.Safe + qualitative.Plotly + qualitative.Dark24
    fig = go.Figure()
    compositions = sorted(plot_data["actual_composition_key"].astype(str).unique())

    for index, composition in enumerate(compositions):
        series = plot_data[
            plot_data["actual_composition_key"].astype(str) == composition
        ].sort_values("n_vacancies")
        hover_columns = [
            column
            for column in (
                "formula",
                "source_configuration_id",
                "vacancy_percent_of_parent_oxygen",
                "decomposition",
            )
            if column in series.columns
        ]
        customdata = series[hover_columns].astype(str).to_numpy()
        hover_lines = [
            "Vacancies: %{x:g}",
            "Energy above hull: %{y:.4g} meV/atom",
        ]
        hover_lines.extend(
            f"{column}: %{{customdata[{column_index}]}}"
            for column_index, column in enumerate(hover_columns)
        )
        fig.add_trace(
            go.Scatter(
                x=series["n_vacancies"],
                y=series["energy_above_hull_meV_per_atom"],
                mode="lines+markers",
                name=composition,
                customdata=customdata,
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                line={"color": palette[index % len(palette)], "width": 2.2},
                marker={"size": 8, "symbol": "circle"},
            )
        )

        stable = series[series["_stable_bool"]]
        if not stable.empty:
            fig.add_trace(
                go.Scatter(
                    x=stable["n_vacancies"],
                    y=stable["energy_above_hull_meV_per_atom"],
                    mode="markers",
                    name=f"Stable: {composition}",
                    legendgroup=composition,
                    showlegend=False,
                    hoverinfo="skip",
                    marker={
                        "symbol": "star-open",
                        "size": 16,
                        "color": palette[index % len(palette)],
                        "line": {"width": 1.5},
                    },
                )
            )

    quantity = (
        str(plot_data["_hull_quantity"].iloc[0])
        if "_hull_quantity" in plot_data.columns and not plot_data.empty
        else ""
    )
    fig.update_layout(
        template="plotly_white",
        height=700,
        title=(
            f"{quantity} energy above hull vs oxygen-vacancy count"
            "<br><sup>Each point uses the lowest-energy relaxed configuration at that vacancy count</sup>"
        ),
        xaxis_title="Number of oxygen vacancies",
        yaxis_title="Energy above hull (meV/atom)",
        hovermode="closest",
        margin={"l": 70, "r": 30, "t": 90, "b": 65},
        font={"size": 15},
        legend={"font": {"size": 11}},
    )
    fig.update_xaxes(dtick=1, ticks="inside", mirror=True, showline=True)
    fig.update_yaxes(
        rangemode="tozero",
        ticks="inside",
        mirror=True,
        showline=True,
        zeroline=True,
    )
    return fig, plot_data

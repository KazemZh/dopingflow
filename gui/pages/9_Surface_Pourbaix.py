Warning: truncated output (original token count: 19360)
Total output lines: 1941

"""Configure, run, and inspect electrochemical surface-state / Pourbaix analysis."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import toml

from dopingflow.surface_pourbaix import (
    preview_surface_pourbaix_targets,
    resolve_surface_pourbaix_output_dir,
    surface_state_display_label,
)
from dopingflow.surface_pourbaix_thermo import (
    build_surface_pourbaix_grid,
    summarize_stable_domains,
)
from gui_config import (
    BACKEND_CHOICES,
    DEVICE_CHOICES,
    GRACE_MODEL_CHOICES,
    MACE_MODEL_CHOICES,
    OPTIMIZER_CHOICES,
    UMA_MODEL_CHOICES,
    UMA_TASK_CHOICES,
)
from view_structure import show_structure, structure_viewer_controls

st.set_page_config(page_title="Surface Pourbaix", layout="wide")
st.title("Electrochemical surface states / Pourbaix")
st.caption(
    "Choose intact surfaces directly from Surface Screening/Refinement or, optionally, "
    "from Surface Segregation, or supply prepared slabs. Compare adsorbate coverages "
    "and find the stable surface state at each potential and pH."
)

project_root = Path(
    st.sidebar.text_input("Project root", value=str(Path.cwd()), key="surface_pourbaix_project_root")
).expanduser().resolve()
config_path = project_root / "input.toml"
if not config_path.exists():
    st.error(f"No input.toml found at {config_path}")
    st.stop()

cfg = toml.load(str(config_path))
saved = dict(cfg.get("surface_pourbaix", {}) or {})
saved_screen = dict(saved.get("screen", {}) or {})
saved_dft = dict(saved.get("dft", {}) or {})
surface_cfg = dict(cfg.get("surface", {}) or {})


def _csv(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ", ".join(str(x) for x in (value or []))


def _items(text: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in text.split(",") if x.strip()))


def _ints(text: str) -> list[int]:
    return sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))


def _floats(text: str) -> list[float]:
    return list(dict.fromkeys(float(x.strip()) for x in text.split(",") if x.strip()))


def _mixed_coverage_text(value: Any) -> str:
    pairs = value or [[25, 25], [25, 75], [50, 50], [75, 25]]
    return ", ".join(f"{float(pair[0]):g}:{float(pair[1]):g}" for pair in pairs)


def _mixed_coverages(text: str) -> list[list[float]]:
    pairs: list[list[float]] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        pairs.append([float(left.strip()), float(right.strip())])
    return pairs


@st.cache_data(show_spinner=False)
def _available_mace_models() -> tuple[str, ...]:
    """Use the installed MACE catalogue when available."""
    try:
        from dopingflow.ml_backends import get_mace_model_choices

        models = tuple(get_mace_model_choices())
        if models:
            return models
    except Exception:
        pass
    return tuple(MACE_MODEL_CHOICES)


def _backend_model_defaults(backend: str) -> tuple[str, str]:
    """Return valid defaults for the selected ML backend."""
    backend = str(backend).strip().lower()
    if backend == "m3gnet":
        return "default", ""
    if backend == "grace":
        return "GRACE-1L-OMAT", ""
    if backend == "uma":
        return "uma-s-1p2", "omat"
    if backend == "mace":
        return "mh-1", "matpes_r2scan"
    return "", ""


def _saved_screen_value(
    backend: str,
    key: str,
    fallback: str,
) -> str:
    """Never carry a model/task saved for another backend into this backend."""
    saved_backend = str(saved_screen.get("backend", "")).strip().lower()
    if saved_backend != str(backend).strip().lower():
        return fallback
    value = str(saved_screen.get(key, fallback)).strip()
    return value if value else fallback


def _valid_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _coverage_hover(row: pd.Series) -> str:
    family = str(row.get("stable_family", "")).strip()
    if family == "mixed-O-OH":
        o_cov = _valid_text(row.get("actual_o_coverage_pct"))
        oh_cov = _valid_text(row.get("actual_oh_coverage_pct"))
        if o_cov and oh_cov:
            return f"O*: {float(o_cov):.1f}% | OH*: {float(oh_cov):.1f}%"
    coverage = _valid_text(row.get("actual_coverage_pct"))
    return f"{float(coverage):.1f}%" if coverage else ""


def _enrich_grid_for_display(
    grid: pd.DataFrame,
    state_summary: pd.DataFrame | None,
) -> tuple[pd.DataFrame, bool]:
    """Add coverage-aware display labels, including graceful legacy fallback."""
    output = grid.copy()
    coverage_available = "stable_state_label" in output.columns and bool(
        output["stable_state_label"].map(_valid_text).any()
    )
    if coverage_available:
        return output, True

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if state_summary is not None and not state_summary.empty:
        for _, row in state_summary.iterrows():
            state = row.to_dict()
            key = (str(state.get("surface_id", "")), str(state.get("state_id", "")))
            lookup[key] = state

    labels: list[str] = []
    coverage_found = False
    metadata_keys = (
        "actual_coverage_pct",
        "requested_coverage_pct",
        "actual_o_coverage_pct",
        "actual_oh_coverage_pct",
        "requested_o_coverage_pct",
        "requested_oh_coverage_pct",
        "resolved_placement_side",
        "requested_protonation_side",
        "resolved_protonation_side",
        "eligible_protonation_oxygen_sites",
        "eligible_adsorbate_cation_sites",
        "arrangement_id",
        "symmetry_unique_arrangements",
        "symmetry_operations",
        "state_status",
        "pourbaix_eligible",
        "final_family",
        "final_state_label",
        "postprocess_reason",
        "n_surface_O2",
        "n_desorbed_O2",
        "n_desorbed_H2O",
        "n_unbound_H",
        "final_protonated_coverage_pct",
        "final_o_coverage_pct",
        "final_oh_coverage_pct",
        "final_h2o_coverage_pct",
    )
    for key in metadata_keys:
        if key not in output.columns:
            output[key] = None

    for index, row in output.iterrows():
        state = lookup.get(
            (str(row.get("surface_id", "")), str(row.get("stable_state_id", ""))),
            {},
        )
        if state:
            label = surface_state_display_label(state)
            for key in metadata_keys:
                if key in state:
                    output.at[index, key] = state[key]
            if _valid_text(state.get("actual_coverage_pct")) or (
                _valid_text(state.get("actual_o_coverage_pct"))
                and _valid_text(state.get("actual_oh_coverage_pct"))
            ):
                coverage_found = True
        else:
            label = surface_state_display_label(
                {"family": row.get("stable_family", "")}
            )
        labels.append(label)
    output["stable_state_label"] = labels
    return output, coverage_found



def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return bool(default)
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return bool(value)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _competition_group_label(row: pd.Series) -> str:
    """Group all arrangements generated for the same chemistry/coverage."""
    requested = _valid_text(row.get("requested_state_label"))
    if requested:
        return requested
    return surface_state_display_label(row.to_dict())


def _load_pourbaix_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _rebuild_selected_pourbaix_grid(
    states: pd.DataFrame,
    *,
    surface_id: str,
    selected_groups: list[str],
    pH_values: list[float],
    potential_values: list[float],
    potential_scale: str,
    temperature_K: float,
    energy_level: str,
    h2_energy_eV: float,
    h2o_energy_eV: float,
    co_energy_eV: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild a map from already calculated states only.

    Clean is always retained as the CHE reference and as a competing physical
    surface state. Selecting a chemistry/coverage group includes every eligible
    calculated arrangement in that group.
    """
    surface_states = states.copy()
    surface_states["_group_label"] = surface_states.apply(
        _competition_group_label,
        axis=1,
    )
    eligible = surface_states[
        surface_states.apply(
            lambda row: _truthy(row.get("pourbaix_eligible"), True),
            axis=1,
        )
    ].copy()

    clean_mask = eligible["family"].astype(str) == "clean"
    selected_mask = eligible["_group_label"].isin(selected_groups)
    chosen = eligible[clean_mask | selected_mask].copy()

    energy_key = "ml_energy_eV" if energy_level == "ML" else "dft_energy_eV"
    chosen[energy_key] = pd.to_numeric(chosen.get(energy_key), errors="coerce")
    chosen = chosen[np.isfinite(chosen[energy_key])].copy()

    if not (chosen["family"].astype(str) == "clean").any():
        raise ValueError(
            f"No finite clean-surface {energy_level} energy is available."
        )

    records = chosen.to_dict(orient="records")
    rebuilt, _ = build_surface_pourbaix_grid(
        records,
        h2_energy_eV=float(h2_energy_eV),
        h2o_energy_eV=float(h2o_energy_eV), co_energy_eV=co_energy_eV,
        potential_values=potential_values,
        pH_values=pH_values,
        potential_scale=potential_scale,
        temperature_K=float(temperature_K),
        energy_key=energy_key,
        energy_level=energy_level,
    )
    rebuilt.insert(0, "surface_id", str(surface_id))
    rebuilt, _ = _enrich_grid_for_display(rebuilt, chosen)
    domains = summarize_stable_domains(rebuilt)
    return rebuilt, domains, chosen


def _result_path(value: Any) -> Path | None:
    text = _valid_text(value)
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _initial_state_path(row: pd.Series) -> Path | None:
    explicit = _result_path(row.get("ml_initial_structure_path"))
    if explicit is not None and explicit.exists():
        return explicit
    # Backward-compatible fallback: screen_state has always written
    # POSCAR_initial next to POSCAR_relaxed even before the CSV exposed it.
    relaxed = _result_path(row.get("ml_structure_path"))
    if relaxed is not None:
        fallback = relaxed.parent / "POSCAR_initial"
        if fallback.exists():
            return fallback
    return explicit


def _state_option_label(row: pd.Series) -> str:
    final_label = _valid_text(row.get("final_state_label"))
    requested = _valid_text(row.get("requested_state_label"))
    status = _valid_text(row.get("state_status"))
    state_id = _valid_text(row.get("state_id"))
    label = final_label or requested or state_id or "Surface state"
    if status:
        label += f"  [{status}]"
    return label


def _show_structure_panel(
    path: Path | None,
    *,
    heading: str,
    title: str,
    viewer_options: dict[str, Any],
    surface_side: str | None,
) -> None:
    st.markdown(f"#### {heading}")
    if path is None:
        st.warning("No structure path is recorded for this state.")
        return
    st.caption("Saved structure path")
    st.code(str(path), language=None)
    if not path.exists():
        st.warning("The saved path does not currently exist on this machine.")
        return
    try:
        show_structure(
            path,
            title=title,
            width=560,
            height=480,
            viewer_mode="surface",
            surface_side=surface_side,
            **viewer_options,
        )
    except Exception as exc:
        st.warning(f"Could not render this structure: {exc}")


def _cell_edges(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return np.asarray([array[0] - 0.5, array[0] + 0.5], dtype=float)
    mid = (array[:-1] + array[1:]) / 2.0
    first = array[0] - (array[1] - array[0]) / 2.0
    last = array[-1] + (array[-1] - array[-2]) / 2.0
    return np.concatenate(([first], mid, [last]))


def _phase_map_figure(view: pd.DataFrame, surface_id: str) -> go.Figure:
    view = view.copy()
    view["stable_state_label"] = view["stable_state_label"].map(
        lambda value: _valid_text(value) or "Unknown surface state"
    )

    pH_values = sorted(float(value) for value in view["pH"].unique())
    potential_values = sorted(
        float(value) for value in view["applied_potential_V"].unique()
    )

    phase_order = list(
        view.groupby("stable_state_label", sort=False)["applied_potential_V"]
        .mean()
        .sort_values()
        .index
    )
    phase_to_code = {phase: index for index, phase in enumerate(phase_order)}

    label_grid = (
        view.pivot(
            index="applied_potential_V",
            columns="pH",
            values="stable_state_label",
        )
        .reindex(index=potential_values, columns=pH_values)
    )
    z = np.asarray(
        [
            [phase_to_code.get(str(value), 0) for value in row]
            for row in label_grid.to_numpy()
        ],
        dtype=float,
    )

    def hover_text(row: pd.Series) -> str:
        parts = [
            f"<b>{row['stable_state_label']}</b>",
            f"pH: {float(row['pH']):.2f}",
            (
                f"Potential: {float(row['applied_potential_V']):.3f} V "
                f"vs {row.get('potential_scale', '')}"
            ),
        ]
        coverage = _coverage_hover(row)
        if coverage:
            parts.append(f"Actual coverage: {coverage}")
        status = _valid_text(row.get("state_status"))
        if status:
            parts.append(f"Relaxed-state status: {status}")
        delta_g = row.get("deltaG_stable_eV")
        if pd.notna(delta_g):
            parts.append(f"Relative CHE free energy: {float(delta_g):.4f} eV")
        parts.append("See Structure comparison tab for the 3D before/after geometry")
        return "<br>".join(parts)

    view["_hover"] = view.apply(hover_text, axis=1)
    hover_grid = (
        view.pivot(
            index="applied_potential_V",
            columns="pH",
            values="_hover",
        )
        .reindex(index=potential_values, columns=pH_values)
        .to_numpy()
    )

    palette = list(px.colors.qualitative.Alphabet)
    colors = [palette[index % len(palette)] for index in range(len(phase_order))]
    n_phases = max(len(phase_order), 1)
    colorscale: list[list[Any]] = []
    for index, color in enumerate(colors):
        low = index / n_phases
        high = (index + 1) / n_phases
        colorscale.extend([[low, color], [high, color]])

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=pH_values,
            y=potential_values,
            z=z,
            text=hover_grid,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=-0.5,
            zmax=max(len(phase_order) - 0.5, 0.5),
            showscale=False,
            xgap=0,
            ygap=0,
        )
    )

    # Add one compact categorical legend entry per chemistry/coverage state.
    for index, phase in enumerate(phase_order):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 11, "symbol": "square", "color": colors[index]},
                name=phase,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # Draw boundaries between different chemistry/coverage regions.
    x_edges = _cell_edges(pH_values)
    y_edges = _cell_edges(potential_values)
    matrix = label_grid.to_numpy()
    boundary_x: list[float | None] = []
    boundary_y: list[float | None] = []

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1] - 1):
            if matrix[row, col] != matrix[row, col + 1]:
                x = float(x_edges[col + 1])
                boundary_x.extend([x, x, None])
                boundary_y.extend(
                    [float(y_edges[row]), float(y_edges[row + 1]), None]
                )
    for row in range(matrix.shape[0] - 1):
        for col in range(matrix.shape[1]):
            if matrix[row, col] != matrix[row + 1, col]:
                y = float(y_edges[row + 1])
                boundary_x.extend(
                    [float(x_edges[col]), float(x_edges[col + 1]), None]
                )
                boundary_y.extend([y, y, None])

    if boundary_x:
        fig.add_trace(
            go.Scatter(
                x=boundary_x,
                y=boundary_y,
                mode="lines",
                line={"width": 1.2, "color": "rgba(0,0,0,0.75)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    scale = str(view["potential_scale"].iloc[0]) if not view.empty else ""
    energy_levels = ", ".join(
        dict.fromkeys(str(value) for value in view["energy_level"].dropna())
    )
    fig.update_layout(
        title=(
            f"Surface Pourbaix map: {surface_id}"
            + (f" ({energy_levels})" if energy_levels else "")
        ),
        xaxis_title="pH",
        yaxis_title=f"Potential (V vs {scale})",
        height=650,
        legend_title_text="Stable surface state",
        hovermode="closest",
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
    )
    fig.update_xaxes(
        range=[float(x_edges[0]), float(x_edges[-1])],
        constrain="domain",
    )
    fig.update_yaxes(range=[float(y_edges[0]), float(y_edges[-1])])
    return fig


def _run(args: list[str]) -> int:
    st.code(" ".join(args))
    box = st.empty()
    lines: list[str] = []
    process = subprocess.Popen(
        args, cwd=str(project_root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-40:]))
    code = process.wait()
    if code == 0:
        st.success("Surface-Pourbaix stage finished successfully.")
    else:
        st.error(f"Surface-Pourbaix stage exited with return code {code}.")
    return code


with st.expander("Configuration & run controls", expanded=True):
    enabled = st.checkbox("Enable surface-Pourbaix stage", value=bool(saved.get("enabled", False)))

    st.subheader("Input surfaces")
    c1, c2, c3 = st.columns(3)
    source_modes = [
        "surface",
        "segregation",
        "auto",
        "final-selected",
        "refine-summary",
        "screen-selected",
        "screen-summary",
    ]
    source_labels = {
        "surface": "Surface stage — best available (default)",
        "segregation": "Surface segregation output",
        "au…9360 tokens truncated…view.get(
                                "energy_level",
                                pd.Series(dtype=str),
                            ).dropna()
                        ]
                        default_level = (
                            current_levels[0]
                            if current_levels
                            and current_levels[0] in energy_options
                            else energy_options[0]
                        )
                        energy_level = st.selectbox(
                            "Energy set for interactive plot",
                            energy_options,
                            index=energy_options.index(default_level),
                            key=f"surface_pourbaix_plot_energy_{selected}",
                        )

                        selected_groups = st.multiselect(
                            "Chemistry / coverage groups allowed to compete",
                            group_order,
                            default=group_order,
                            format_func=lambda label: (
                                f"{label}  "
                                f"({int(group_counts.get(label, 0))} calculated "
                                f"arrangement"
                                f"{'s' if int(group_counts.get(label, 0)) != 1 else ''})"
                            ),
                            key=(
                                f"surface_pourbaix_plot_groups_"
                                f"{selected}_{energy_level}"
                            ),
                            help=(
                                "For example, if 50% and 100% protonation were "
                                "calculated, keep both to reproduce their competition "
                                "or deselect either coverage to remove it from the plot."
                            ),
                        )

                        total_arrangements = int(
                            nonclean[
                                nonclean["_group_label"].isin(selected_groups)
                            ].shape[0]
                        )
                        st.caption(
                            f"{len(selected_groups)} selected coverage group(s), "
                            f"{total_arrangements} non-clean calculated arrangement(s) "
                            "entering the thermodynamic competition."
                        )

                        try:
                            if energy_level == "ML":
                                h2_ref, h2o_ref = h2_ml, h2o_ml
                            else:
                                h2_ref, h2o_ref = h2_dft, h2o_dft
                            assert h2_ref is not None and h2o_ref is not None

                            pH_values = sorted(
                                float(value) for value in view["pH"].unique()
                            )
                            potential_values = sorted(
                                float(value)
                                for value in view["applied_potential_V"].unique()
                            )
                            potential_scale = (
                                str(view["potential_scale"].iloc[0])
                                if not view.empty
                                else str(run_summary.get("potential_scale", "SHE"))
                            )
                            temperature_K = (
                                float(view["temperature_K"].iloc[0])
                                if "temperature_K" in view.columns
                                and not view.empty
                                else float(run_summary.get("temperature_K", 298.15))
                            )

                            plot_view, plot_domains, selected_state_rows = (
                                _rebuild_selected_pourbaix_grid(
                                    surface_states,
                                    surface_id=selected,
                                    selected_groups=selected_groups,
                                    pH_values=pH_values,
                                    potential_values=potential_values,
                                    potential_scale=potential_scale,
                                    temperature_K=temperature_K,
                                    energy_level=energy_level,
                                    h2_energy_eV=h2_ref,
                                    h2o_energy_eV=h2o_ref,
                                    co_energy_eV=run_summary.get(f"co_reference_eV_{energy_level.lower()}"),
                                )
                            )

                            competition_state_ids = set(
                                selected_state_rows["state_id"].astype(str)
                            )
                            available_selected = set(
                                selected_state_rows["_group_label"].astype(str)
                            )
                            unavailable_groups = [
                                group
                                for group in selected_groups
                                if group not in available_selected
                            ]
                            if unavailable_groups:
                                st.warning(
                                    f"{energy_level} energies are unavailable for "
                                    f"{len(unavailable_groups)} selected group(s); "
                                    "those groups cannot enter this plot."
                                )
                        except Exception as exc:
                            st.warning(
                                "Could not rebuild the interactive Pourbaix map; "
                                f"showing the saved map instead. {exc}"
                            )
                            plot_view = view.copy()
                            plot_domains = None
                    else:
                        st.info(
                            "The saved run does not contain the reference energies "
                            "needed to rebuild the map interactively. Rerun Surface "
                            "Pourbaix once with the current implementation."
                        )

            fig = _phase_map_figure(plot_view, selected)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Hover shows only the essential thermodynamic information. "
                "Use **Structure comparison** to inspect the generated and relaxed "
                "geometry of any stable state."
            )

            if plot_domains is None and domains_path.exists():
                domains = pd.read_csv(domains_path)
                plot_domains = domains[
                    domains["surface_id"].astype(str) == selected
                ].copy()

            if plot_domains is not None and not plot_domains.empty:
                domain_view = plot_domains.copy()
                label_map = dict(
                    plot_view[["stable_state_id", "stable_state_label"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
                if "surface_state" in domain_view.columns:
                    domain_view["surface_state"] = (
                        domain_view["state_id"].astype(str).map(label_map)
                    )
                else:
                    insert_at = min(2, len(domain_view.columns))
                    domain_view.insert(
                        insert_at,
                        "surface_state",
                        domain_view["state_id"].astype(str).map(label_map),
                    )
                preferred = [
                    "surface_state",
                    "family",
                    "energy_level",
                    "grid_fraction",
                    "pH_min_stable",
                    "pH_max_stable",
                    "potential_min_V_stable",
                    "potential_max_V_stable",
                    "state_id",
                ]
                st.markdown("**Stable surface-state domains**")
                st.dataframe(
                    domain_view[
                        [
                            column
                            for column in preferred
                            if column in domain_view.columns
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

        with result_tabs[1]:
            if state_summary is None or state_summary.empty:
                st.info("No surface-state summary is available.")
            else:
                surface_states = state_summary[
                    state_summary["surface_id"].astype(str) == selected
                ].copy()
                if surface_states.empty:
                    st.info("No calculated states were found for this surface.")
                else:
                    show_all_states = st.checkbox(
                        "Include all calculated states (also outside the current "
                        "Pourbaix-map selection)",
                        value=False,
                        key=f"surface_pourbaix_show_all_states_{selected}",
                        help=(
                            "When off, the list mirrors the chemistry/coverage groups "
                            "currently selected under **Chemistry / coverage groups "
                            "allowed to compete** in the Pourbaix-map tab, including "
                            "all eligible calculated arrangements in those groups and "
                            "the clean reference. Turn this on to inspect states that "
                            "were not included in the current map competition."
                        ),
                    )
                    selectable = (
                        surface_states
                        if show_all_states
                        else surface_states[
                            surface_states["state_id"]
                            .astype(str)
                            .isin(competition_state_ids)
                        ]
                    ).copy()

                    if selectable.empty:
                        st.warning(
                            "No inspectable states match the current Pourbaix-map "
                            "competition. Select at least one chemistry/coverage group "
                            "in the Pourbaix-map tab or enable all calculated states."
                        )
                        st.stop()

                    if not show_all_states:
                        st.caption(
                            "This list is synchronized with the current Pourbaix-map "
                            "competition selection."
                        )

                    selectable["_option_label"] = selectable.apply(
                        _state_option_label,
                        axis=1,
                    )
                    option_rows = list(selectable.index)
                    chosen_index = st.selectbox(
                        "Surface state to inspect",
                        option_rows,
                        format_func=lambda index: selectable.loc[
                            index, "_option_label"
                        ],
                        key=f"surface_pourbaix_structure_state_{selected}",
                        help=(
                            "By default this list contains the clean reference plus "
                            "all eligible calculated arrangements belonging to the "
                            "chemistry/coverage groups currently allowed to compete in "
                            "the Pourbaix-map tab."
                        ),
                    )
                    chosen = selectable.loc[chosen_index]

                    info_cols = st.columns(4)
                    info_cols[0].metric(
                        "Status",
                        _valid_text(chosen.get("state_status")) or "not recorded",
                    )
                    info_cols[1].metric(
                        "Pourbaix eligible",
                        (
                            "Yes"
                            if bool(chosen.get("pourbaix_eligible", True))
                            else "No"
                        ),
                    )
                    info_cols[2].metric(
                        "Requested state",
                        _valid_text(chosen.get("requested_state_label"))
                        or _valid_text(chosen.get("family"))
                        or "—",
                    )
                    info_cols[3].metric(
                        "Final relaxed state",
                        _valid_text(chosen.get("final_state_label"))
                        or _valid_text(chosen.get("family"))
                        or "—",
                    )

                    reason = _valid_text(chosen.get("postprocess_reason"))
                    if reason:
                        st.caption(f"Post-relaxation classification: {reason}")

                    initial_path = _initial_state_path(chosen)
                    relaxed_path = _result_path(chosen.get("ml_structure_path"))
                    chosen_family = _valid_text(chosen.get("family"))
                    if chosen_family == "protonated":
                        resolved_side = (
                            _valid_text(chosen.get("resolved_protonation_side"))
                            or _valid_text(chosen.get("resolved_placement_side"))
                            or None
                        )
                    else:
                        resolved_side = (
                            _valid_text(chosen.get("resolved_placement_side")) or None
                        )
                    viewer_options = structure_viewer_controls(
                        [initial_path, relaxed_path],
                        key_prefix=(
                            f"surface_pourbaix_viewer_{selected}_{chosen['state_id']}"
                        ),
                        expanded=False,
                        show_orientation_default=True,
                        show_unit_cell_default=True,
                    )

                    before_col, after_col = st.columns(2)
                    with before_col:
                        _show_structure_panel(
                            initial_path,
                            heading="Before relaxation",
                            title=(
                                f"{selected} | {chosen['state_id']} | "
                                "generated structure"
                            ),
                            viewer_options=viewer_options,
                            surface_side=resolved_side,
                        )
                    with after_col:
                        _show_structure_panel(
                            relaxed_path,
                            heading="After ML relaxation",
                            title=(
                                f"{selected} | {chosen['state_id']} | "
                                "relaxed structure"
                            ),
                            viewer_options=viewer_options,
                            surface_side=resolved_side,
                        )

                    source_path = _result_path(
                        chosen.get("source_structure_path")
                    )
                    with st.expander("Parent surface and calculation paths"):
                        st.markdown("**Parent surface used to generate this state**")
                        if source_path is not None:
                            st.code(str(source_path), language=None)
                        else:
                            st.write("No parent-surface path recorded.")
                        st.markdown("**State calculation directory**")
                        state_dir = (
                            relaxed_path.parent
                            if relaxed_path is not None
                            else (
                                initial_path.parent
                                if initial_path is not None
                                else None
                            )
                        )
                        if state_dir is not None:
                            st.code(str(state_dir), language=None)
                        else:
                            st.write("No state directory could be resolved.")
                        st.markdown("**Raw state ID**")
                        st.code(str(chosen["state_id"]), language=None)

        with result_tabs[2]:
            if state_summary is None or state_summary.empty:
                st.info("No state validation/provenance table is available.")
            else:
                surface_states = state_summary[
                    state_summary["surface_id"].astype(str) == selected
                ].copy()
                if "state_status" in surface_states.columns:
                    validation_summary = (
                        surface_states.groupby(
                            ["state_status", "pourbaix_eligible"],
                            dropna=False,
                        )
                        .size()
                        .reset_index(name="n_states")
                        .sort_values("n_states", ascending=False)
                    )
                    st.markdown("**Post-relaxation state validation**")
                    st.dataframe(
                        validation_summary,
                        width="stretch",
                        hide_index=True,
                    )

                    eligible_series = (
                        surface_states["pourbaix_eligible"]
                        .fillna(False)
                        .astype(bool)
                    )
                    excluded = surface_states[~eligible_series].copy()
                    if not excluded.empty:
                        st.warning(
                            f"{len(excluded)} relaxed state(s) for this surface were "
                            "excluded from the Pourbaix competition."
                        )
                        excluded_columns = [
                            "requested_state_label",
                            "final_state_label",
                            "state_status",
                            "postprocess_reason",
                            "n_desorbed_O2",
                            "n_desorbed_H2O",
                            "n_unbound_H",
                            "ml_initial_structure_path",
                            "ml_structure_path",
                            "state_id",
                        ]
                        st.dataframe(
                            excluded[
                                [
                                    column
                                    for column in excluded_columns
                                    if column in excluded.columns
                                ]
                            ],
                            width="stretch",
                            hide_index=True,
                        )

                with st.expander(
                    "All sampled states and full provenance",
                    expanded=False,
                ):
                    st.dataframe(
                        surface_states.drop(
                            columns=["_option_label"],
                            errors="ignore",
                        ),
                        width="stretch",
                        hide_index=True,
                    )
except Exception as exc:
    st.warning(str(exc))

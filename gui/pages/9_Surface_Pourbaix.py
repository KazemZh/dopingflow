"""Configure, run, and inspect electrochemical surface-state / Pourbaix analysis."""

from __future__ import annotations

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
from gui_config import (
    BACKEND_CHOICES,
    DEVICE_CHOICES,
    GRACE_MODEL_CHOICES,
    MACE_MODEL_CHOICES,
    OPTIMIZER_CHOICES,
    UMA_MODEL_CHOICES,
    UMA_TASK_CHOICES,
)

st.set_page_config(page_title="Surface Pourbaix", layout="wide")
st.title("Electrochemical surface states / Pourbaix")
st.caption(
    "Choose intact surfaces directly from Surface Screening/Refinement or, optionally, "
    "from Surface Segregation. Then sample protonated, O*, OH*, H2O*, and mixed O/OH "
    "states and determine the stable surface state versus potential and pH with the CHE."
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
        "arrangement_id",
        "symmetry_unique_arrangements",
        "symmetry_operations",
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
        requested = _valid_text(row.get("requested_coverage_pct"))
        if requested:
            parts.append(f"Requested coverage: {requested}%")
        side = _valid_text(row.get("resolved_placement_side"))
        if side:
            parts.append(f"Resolved side: {side}")
        parts.append(f"Raw state ID: {row['stable_state_id']}")
        arrangement = _valid_text(row.get("arrangement_id"))
        if arrangement:
            parts.append(f"Arrangement: {arrangement}")
        symmetry_unique = _valid_text(row.get("symmetry_unique_arrangements"))
        if symmetry_unique:
            parts.append(f"Symmetry-unique arrangements: {symmetry_unique}")
        delta_g = row.get("deltaG_stable_eV")
        if pd.notna(delta_g):
            parts.append(f"Relative CHE free energy: {float(delta_g):.4f} eV")
        parts.append(f"Energy level: {row.get('energy_level', '')}")
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
        "auto": "Auto — direct surface first, segregation fallback",
        "final-selected": "Surface stage — final selected",
        "refine-summary": "Surface stage — all refined",
        "screen-selected": "Surface stage — screened shortlist",
        "screen-summary": "Surface stage — all screened/generated",
    }
    saved_mode = str(saved.get("source_mode", "surface"))
    source_mode = c1.selectbox(
        "Surface source",
        source_modes,
        index=source_modes.index(saved_mode) if saved_mode in source_modes else 0,
        format_func=lambda mode: source_labels[mode],
        help=(
            "Surface segregation is optional. The default reads the best available "
            "table produced directly by the surface stage."
        ),
    )
    source_summary = c2.text_input(
        "Explicit source CSV (optional)", value=str(saved.get("source_summary", ""))
    ).strip()
    max_surfaces = c3.number_input(
        "Maximum surfaces", min_value=1, value=int(saved.get("max_surfaces", 10)), step=1
    )

    selector_text = st.text_input(
        "Surface selectors (optional, comma-separated exact IDs or globs)",
        value=_csv(saved.get("surface_include", [])),
        help="Leave empty for all surfaces in the chosen source table.",
    )
    requested_selectors = _items(selector_text)

    # First discover all surfaces from the selected source so users can pick
    # exact generated/refined surfaces without having to know their IDs.
    available_preview = pd.DataFrame()
    try:
        available_cfg = dict(cfg)
        available_section = dict(saved)
        available_section.update(
            enabled=True,
            source_mode=source_mode,
            source_summary=source_summary,
            surface_include=[],
            max_surfaces=10000,
        )
        available_cfg["surface_pourbaix"] = available_section
        available_preview = preview_surface_pourbaix_targets(available_cfg, project_root)
    except Exception:
        pass

    exact_surface_ids = (
        list(dict.fromkeys(available_preview["surface_id"].astype(str)))
        if not available_preview.empty and "surface_id" in available_preview.columns
        else []
    )
    exact_defaults = [value for value in requested_selectors if value in exact_surface_ids]
    picked_surfaces = st.multiselect(
        "Pick exact surfaces (optional)",
        exact_surface_ids,
        default=exact_defaults,
        help=(
            "This list is populated directly from the selected surface-stage or "
            "segregation table. If you select entries here, they override the text selectors above."
        ),
    )
    effective_surface_include = picked_surfaces or requested_selectors

    preview_cfg = dict(cfg)
    preview_section = dict(saved)
    preview_section.update(
        enabled=True, source_mode=source_mode, source_summary=source_summary,
        surface_include=effective_surface_include, max_surfaces=int(max_surfaces),
    )
    preview_cfg["surface_pourbaix"] = preview_section
    try:
        preview = preview_surface_pourbaix_targets(preview_cfg, project_root)
        st.caption(
            f"Selected source table: {preview.attrs.get('source_summary', '')} "
            f"({len(preview)} surface(s) selected)"
        )
        st.dataframe(preview, width="stretch", hide_index=True)
    except Exception as exc:
        st.warning(str(exc))

    st.subheader("Surface-state search")
    families = st.multiselect(
        "State families",
        ["clean", "protonated", "O", "OH", "H2O", "mixed-O-OH"],
        default=list(saved.get(
            "state_families",
            ["clean", "protonated", "O", "OH", "H2O", "mixed-O-OH"],
        )),
    )

    s1, s2, s3, s4 = st.columns(4)
    side_options = ["dopant-nearest", "top", "bottom", "both"]
    side_labels = {
        "dopant-nearest": "Near dopants — automatic (recommended)",
        "top": "Top",
        "bottom": "Bottom",
        "both": "Both",
    }
    saved_side = str(saved.get("placement_side", "dopant-nearest")).replace("_", "-")
    if saved_side not in side_options:
        saved_side = "dopant-nearest"
    placement_side = s1.selectbox(
        "Adsorbate / protonation side",
        side_options,
        index=side_options.index(saved_side),
        format_func=lambda value: side_labels[value],
        help=(
            "Automatic mode evaluates the actual doped slab and chooses the side "
            "closest to the selected dopant species independently for every surface."
        ),
    )
    surface_window = s2.number_input(
        "Exposed-site window (Å)", min_value=0.1,
        value=float(saved.get("surface_window_A", 2.0)), step=0.1,
    )
    max_o_sites = s3.number_input(
        "Max exposed O sites (0 = all)",
        min_value=0,
        value=int(saved.get("max_surface_oxygen_sites", 0)),
        step=1,
        help="0 uses every eligible exposed O site and is recommended for coverage definitions.",
    )
    max_cat_sites = s4.number_input(
        "Max exposed cation sites (0 = all)",
        min_value=0,
        value=int(saved.get("max_surface_cation_sites", 0)),
        step=1,
        help="0 uses every eligible exposed cation adsorption site.",
    )

    inferred_side_targets = (
        saved.get("side_target_species")
        or surface_cfg.get("dopant_species")
        or []
    )
    side_target_text = st.text_input(
        "Dopant species for automatic side selection",
        value=_csv(inferred_side_targets),
        disabled=placement_side != "dopant-nearest",
        help=(
            "Example: Sb, In. Leave blank to infer all non-host cations from each "
            "actual surface. The nearest depth is calculated per species, then "
            "averaged so one dopant type does not dominate simply by atom count."
        ),
    )
    side_target_species = _items(side_target_text)
    if placement_side == "dopant-nearest":
        st.caption(
            "If top and bottom are equally close within the tie tolerance, both sides "
            "are sampled rather than choosing one arbitrarily."
        )

    st.markdown("**Coverage grid (% of eligible sites on the selected side)**")
    cH, cO, cOH, cW = st.columns(4)
    proton_coverage_text = cH.text_input(
        "Protonated O coverage (%)",
        value=_csv(saved.get("proton_coverages_pct", [25, 50, 75, 100])),
    )
    o_coverage_text = cO.text_input(
        "O* coverage (%)",
        value=_csv(saved.get("o_coverages_pct", [25, 50, 75, 100])),
    )
    oh_coverage_text = cOH.text_input(
        "OH* coverage (%)",
        value=_csv(saved.get("oh_coverages_pct", [25, 50, 75, 100])),
    )
    h2o_coverage_text = cW.text_input(
        "H2O* coverage (%)",
        value=_csv(saved.get("h2o_coverages_pct", [25, 50, 100])),
    )
    mixed_coverage_text = st.text_input(
        "Mixed O*:OH* coverage pairs (%)",
        value=_mixed_coverage_text(saved.get("mixed_coverages_pct")),
        help=(
            "Format O%:OH%, separated by commas; e.g. 25:25, 25:75, 50:50, 75:25. "
            "The two coverages must sum to at most 100%."
        ),
    )
    st.caption(
        "Coverage is converted to the nearest realizable integer occupation for each "
        "surface. If the requested coverage lies exactly halfway between two integer "
        "counts (for example 25% of 10 sites), both realizations are kept: 20% and 30%."
    )

    q1, q2 = st.columns(2)
    max_arrangements = q1.number_input(
        "Max symmetry-distinct configurations / coverage",
        min_value=1,
        value=int(saved.get("max_arrangements_per_stoichiometry", 8)),
        step=1,
        help=(
            "Symmetry-equivalent patterns are removed first. If more configurations "
            "remain, a geometry/dopant-proximity diversity selection is applied."
        ),
    )
    symmetry_reduce = q2.checkbox(
        "Remove symmetry-equivalent arrangements",
        value=bool(saved.get("symmetry_reduce", True)),
        help=(
            "Symmetry is determined from the actual doped/vacancy slab, not the "
            "undoped parent crystal."
        ),
    )

    with st.expander("Advanced site/symmetry controls"):
        a1, a2, a3, a4 = st.columns(4)
        symmetry_symprec = a1.number_input(
            "Symmetry tolerance (Å)",
            min_value=0.001,
            value=float(saved.get("symmetry_symprec_A", 0.10)),
            step=0.01,
        )
        symmetry_mapping_tol = a2.number_input(
            "Site mapping tolerance (Å)",
            min_value=0.001,
            value=float(saved.get("symmetry_mapping_tolerance_A", 0.25)),
            step=0.01,
        )
        side_tie_tol = a3.number_input(
            "Dopant-side tie tolerance (Å)",
            min_value=0.0,
            value=float(saved.get("dopant_side_tie_tolerance_A", 0.25)),
            step=0.05,
        )
        max_raw_configs = a4.number_input(
            "Max raw patterns / coverage",
            min_value=100,
            value=int(saved.get("max_raw_configurations_per_stoichiometry", 100000)),
            step=1000,
        )

    st.subheader("Electrochemical grid")
    p1, p2, p3, p4 = st.columns(4)
    ph_min = p1.number_input("pH minimum", value=float(saved.get("pH_min", -1.0)), step=0.1)
    ph_max = p2.number_input("pH maximum", value=float(saved.get("pH_max", 3.0)), step=0.1)
    ph_step = p3.number_input(
        "pH step", min_value=0.01, value=float(saved.get("pH_step", 0.1)), step=0.05
    )
    potential_scale = p4.selectbox(
        "Potential scale", ["SHE", "RHE"],
        index=0 if str(saved.get("potential_scale", "SHE")).upper() == "SHE" else 1,
    )
    u1, u2, u3, u4 = st.columns(4)
    u_min = u1.number_input(
        "Potential minimum (V)", value=float(saved.get("potential_min_V", 0.0)), step=0.1
    )
    u_max = u2.number_input(
        "Potential maximum (V)", value=float(saved.get("potential_max_V", 2.0)), step=0.1
    )
    u_step = u3.number_input(
        "Potential step (V)", min_value=0.01,
        value=float(saved.get("potential_step_V", 0.05)), step=0.01,
    )
    temperature = u4.number_input(
        "Temperature (K)", min_value=1.0,
        value=float(saved.get("temperature_K", 298.15)), step=1.0,
    )
    if potential_scale == "RHE":
        st.info(
            "At fixed RHE potential, ideal CHE H+/e− terms are pH-independent. "
            "Use SHE for the conventional pH-sloped Pourbaix representation."
        )

    st.subheader("ML screening")
    m1, m2, m3, m4 = st.columns(4)

    backend_options = list(BACKEND_CHOICES)
    saved_backend = str(saved_screen.get("backend", "mace")).strip().lower()
    if saved_backend not in backend_options:
        saved_backend = "mace"
    backend = m1.selectbox(
        "Backend",
        backend_options,
        index=backend_options.index(saved_backend),
        key="surface_pourbaix_screen_backend",
    )

    backend_model_default, backend_task_default = _backend_model_defaults(backend)

    if backend == "m3gnet":
        model = "default"
        m2.text_input(
            "Model",
            value="default",
            disabled=True,
            key="surface_pourbaix_screen_model_m3gnet",
            help="M3GNet uses its default pretrained model in this workflow.",
        )
        task = ""
        m3.text_input(
            "Task / head",
            value="not used",
            disabled=True,
            key="surface_pourbaix_screen_task_m3gnet",
        )

    elif backend == "grace":
        current_model = _saved_screen_value(
            backend, "model", backend_model_default
        )
        if current_model not in GRACE_MODEL_CHOICES:
            current_model = "GRACE-1L-OMAT"
        model = m2.selectbox(
            "GRACE model",
            list(GRACE_MODEL_CHOICES),
            index=list(GRACE_MODEL_CHOICES).index(current_model),
            key="surface_pourbaix_screen_model_grace",
        )
        task = ""
        m3.text_input(
            "Task / head",
            value="not used",
            disabled=True,
            key="surface_pourbaix_screen_task_grace",
            help="GRACE model selection defines the potential; no separate task/head is used.",
        )

    elif backend == "uma":
        current_model = _saved_screen_value(
            backend, "model", backend_model_default
        )
        if current_model not in UMA_MODEL_CHOICES:
            current_model = "uma-s-1p2"
        model = m2.selectbox(
            "UMA model",
            list(UMA_MODEL_CHOICES),
            index=list(UMA_MODEL_CHOICES).index(current_model),
            key="surface_pourbaix_screen_model_uma",
        )

        current_task = _saved_screen_value(
            backend, "task", backend_task_default
        )
        if current_task not in UMA_TASK_CHOICES:
            current_task = "omat"
        task = m3.selectbox(
            "UMA task",
            list(UMA_TASK_CHOICES),
            index=list(UMA_TASK_CHOICES).index(current_task),
            key="surface_pourbaix_screen_task_uma",
        )

    else:  # MACE
        mace_models = list(_available_mace_models())
        custom_label = "Custom checkpoint path…"
        current_model = _saved_screen_value(
            backend, "model", backend_model_default
        )
        initial_model_choice = (
            current_model if current_model in mace_models else custom_label
        )

        selected_model = m2.selectbox(
            "MACE model",
            [*mace_models, custom_label],
            index=[*mace_models, custom_label].index(initial_model_choice),
            key="surface_pourbaix_screen_model_mace_choice",
        )
        if selected_model == custom_label:
            model = m2.text_input(
                "Checkpoint path",
                value=current_model if current_model not in mace_models else "",
                key="surface_pourbaix_screen_model_mace_custom",
                help="Path to a local MACE checkpoint.",
            ).strip()
        else:
            model = selected_model

        saved_mace_model = (
            str(saved_screen.get("model", "")).strip()
            if str(saved_screen.get("backend", "")).strip().lower() == "mace"
            else ""
        )
        if model == saved_mace_model:
            mace_task_default = str(
                saved_screen.get("task", backend_task_default)
            ).strip()
        elif model == "mh-1":
            mace_task_default = "matpes_r2scan"
        else:
            mace_task_default = ""

        is_multihead_or_custom = (
            model.startswith("mh-") or selected_model == custom_label
        )
        if is_multihead_or_custom:
            task = m3.text_input(
                "MACE head",
                value=mace_task_default,
                key=f"surface_pourbaix_screen_task_mace_{model}",
                help=(
                    "Head for a multi-head MACE checkpoint. For MH-1, "
                    "matpes_r2scan is the default. Leave blank only if the "
                    "selected checkpoint does not require a head."
                ),
            ).strip()
        else:
            task = ""
            m3.text_input(
                "MACE head",
                value="not used",
                disabled=True,
                key=f"surface_pourbaix_screen_task_mace_single_{model}",
            )

    saved_device = str(saved_screen.get("device", "cpu")).strip().lower()
    if saved_device not in DEVICE_CHOICES:
        saved_device = "cpu"
    device = m4.selectbox(
        "Device",
        DEVICE_CHOICES,
        index=DEVICE_CHOICES.index(saved_device),
        key="surface_pourbaix_screen_device",
    )

    st.caption(
        {
            "m3gnet": "M3GNet: fixed default pretrained model; no task/head.",
            "grace": "GRACE: choose only from GRACE checkpoints; no separate task/head.",
            "uma": "UMA: choose a UMA foundation model and its task.",
            "mace": "MACE: choose an installed packaged model or a custom checkpoint; a head is shown only where relevant.",
        }[backend]
    )

    m5, m6, m7 = st.columns(3)
    saved_optimizer = str(saved_screen.get("optimizer", "bfgs"))
    optimizer = m5.selectbox(
        "Optimizer", OPTIMIZER_CHOICES,
        index=OPTIMIZER_CHOICES.index(saved_optimizer)
        if saved_optimizer in OPTIMIZER_CHOICES else 0,
    )
    fmax = m6.number_input(
        "fmax (eV/Å)", min_value=0.001, value=float(saved_screen.get("fmax", 0.05)), step=0.01
    )
    max_steps = m7.number_input(
        "Max relaxation steps", min_value=1,
        value=int(saved_screen.get("max_steps", 300)), step=10,
    )

    with st.expander("Optional DFT refinement"):
        dft_enabled = st.checkbox(
            "Refine competitive surface states with GPAW",
            value=bool(saved_dft.get("enabled", False)),
        )
        d1, d2, d3, d4 = st.columns(4)
        dft_execute = d1.checkbox(
            "Execute missing GPAW calculations", value=bool(saved_dft.get("execute", False))
        )
        dft_ecut = d2.number_input(
            "PW cutoff (eV)", min_value=100.0,
            value=float(saved_dft.get("ecut_eV", 500.0)), step=50.0,
        )
        dft_kpts = d3.text_input(
            "k-points", value="x".join(str(x) for x in saved_dft.get("kpts", [3, 3, 1]))
        )
        dft_window = d4.number_input(
            "ML candidate window (eV)", min_value=0.0,
            value=float(saved_dft.get("candidate_window_eV", 0.30)), step=0.05,
        )
        dft_cap = st.number_input(
            "Max DFT states / surface", min_value=1,
            value=int(saved_dft.get("max_states_per_surface", 20)), step=1,
        )

    section = dict(saved)
    section.update(
        enabled=enabled, source_mode=source_mode, source_summary=source_summary,
        surface_include=effective_surface_include, max_surfaces=int(max_surfaces),
        placement_side=placement_side, side_target_species=side_target_species,
        state_families=families,
        proton_coverages_pct=_floats(proton_coverage_text),
        o_coverages_pct=_floats(o_coverage_text),
        oh_coverages_pct=_floats(oh_coverage_text),
        h2o_coverages_pct=_floats(h2o_coverage_text),
        mixed_coverages_pct=_mixed_coverages(mixed_coverage_text),
        surface_window_A=float(surface_window),
        max_surface_oxygen_sites=int(max_o_sites),
        max_surface_cation_sites=int(max_cat_sites),
        symmetry_reduce=bool(symmetry_reduce),
        symmetry_symprec_A=float(symmetry_symprec),
        symmetry_mapping_tolerance_A=float(symmetry_mapping_tol),
        dopant_side_tie_tolerance_A=float(side_tie_tol),
        max_raw_configurations_per_stoichiometry=int(max_raw_configs),
        max_arrangements_per_stoichiometry=int(max_arrangements),
        pH_min=float(ph_min), pH_max=float(ph_max), pH_step=float(ph_step),
        potential_scale=potential_scale, potential_min_V=float(u_min),
        potential_max_V=float(u_max), potential_step_V=float(u_step),
        temperature_K=float(temperature),
    )
    # Remove superseded count-based controls when an older input.toml is upgraded.
    for legacy_key in ("h_counts", "adsorbate_counts", "mixed_compositions"):
        section.pop(legacy_key, None)

    section["screen"] = {
        **saved_screen, "backend": backend, "model": model, "task": task,
        "device": device, "optimizer": optimizer, "fmax": float(fmax),
        "max_steps": int(max_steps), "relax": True,
    }
    try:
        kpts = [int(x) for x in dft_kpts.lower().replace("x", ",").split(",") if x.strip()]
    except ValueError:
        kpts = [3, 3, 1]
    section["dft"] = {
        **saved_dft, "enabled": dft_enabled, "execute": dft_execute,
        "ecut_eV": float(dft_ecut), "kpts": kpts,
        "candidate_window_eV": float(dft_window),
        "max_states_per_surface": int(dft_cap),
    }

    b1, b2, b3 = st.columns(3)
    if b1.button("Save settings", type="primary"):
        cfg["surface_pourbaix"] = section
        with config_path.open("w", encoding="utf-8") as handle:
            toml.dump(cfg, handle)
        st.success(f"Saved {config_path}")
    if b2.button("Dry run"):
        cfg["surface_pourbaix"] = section
        with config_path.open("w", encoding="utf-8") as handle:
            toml.dump(cfg, handle)
        _run(["dopingflow", "surface-pourbaix", "-c", str(config_path), "--dry-run"])
    if b3.button("Run surface-Pourbaix"):
        cfg["surface_pourbaix"] = section
        with config_path.open("w", encoding="utf-8") as handle:
            toml.dump(cfg, handle)
        _run(["dopingflow", "surface-pourbaix", "-c", str(config_path)])

st.divider()
st.subheader("Results")
try:
    resolved = resolve_surface_pourbaix_output_dir(cfg, project_root)
    grid_path = resolved / "pourbaix_grid.csv"
    state_summary_path = resolved / "surface_state_summary.csv"
    domains_path = resolved / "stable_surface_states.csv"

    state_summary = (
        pd.read_csv(state_summary_path) if state_summary_path.exists() else None
    )

    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        grid, coverage_labels_available = _enrich_grid_for_display(
            grid, state_summary
        )

        surface_ids = list(dict.fromkeys(grid["surface_id"].astype(str)))
        selected = st.selectbox("Surface", surface_ids)
        view = grid[grid["surface_id"].astype(str) == selected].copy()

        if not coverage_labels_available:
            st.warning(
                "These results were generated by an older Surface-Pourbaix run "
                "without coverage metadata. The phase map is still shown, but rerun "
                "the stage with the current implementation to obtain coverage-based "
                "legend labels and full symmetry/side provenance."
            )

        fig = _phase_map_figure(view, selected)
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Colors represent thermodynamically stable chemistry/coverage regions. "
            "Arrangement IDs are intentionally hidden from the legend and remain "
            "available in hover/provenance."
        )

        if domains_path.exists():
            domains = pd.read_csv(domains_path)
            domain_view = domains[
                domains["surface_id"].astype(str) == selected
            ].copy()
            if not domain_view.empty:
                label_map = dict(
                    view[["stable_state_id", "stable_state_label"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
                domain_view.insert(
                    2,
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
                        [column for column in preferred if column in domain_view.columns]
                    ],
                    width="stretch",
                    hide_index=True,
                )
    else:
        st.info("No surface-Pourbaix results yet.")

    if state_summary is not None and not state_summary.empty:
        with st.expander("All sampled surface states / provenance"):
            st.dataframe(state_summary, width="stretch", hide_index=True)
except Exception as exc:
    st.warning(str(exc))

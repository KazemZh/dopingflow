"""Configure, preview, run, and inspect dopant leaching from selected surfaces."""

from __future__ import annotations

import importlib
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
from pymatgen.core import Structure

from dopingflow.leaching import (
    analyze_site_environment,
    build_thermodynamic_grid,
    parse_leaching_config,
    preview_leaching_sites,
    resolve_leaching_output_dir,
)
from gui_config import BACKEND_CHOICES, DEVICE_CHOICES, OPTIMIZER_CHOICES
import view_structure as _view_structure

# Streamlit can keep helper modules cached while rerunning a page after git pulls.
# Reload this lightweight viewer module so the page and helper signature stay in sync.
_view_structure = importlib.reload(_view_structure)
show_site_environment = _view_structure.show_site_environment


st.set_page_config(page_title="Dopant leaching", layout="wide")
st.title("Dopant leaching")
st.caption(
    "Remove dopants from previously selected surfaces, relax the dopant-vacancy slab, "
    "calculate a metal-referenced extraction energy, and optionally evaluate a "
    "simple electrochemical dissolution model with user-supplied redox data."
)

project_root = Path(
    st.sidebar.text_input("Project root", value=str(Path.cwd()), key="leaching_project_root")
).expanduser().resolve()
config_path = project_root / "input.toml"
if not config_path.exists():
    st.error(f"No input.toml found at {config_path}")
    st.stop()

cfg = toml.load(str(config_path))
saved = dict(cfg.get("leaching", {}) or {})
saved_protonation = dict(saved.get("protonation", {}) or {})
saved_analysis = dict(saved.get("analysis_defaults", {}) or {})
surface = dict(cfg.get("surface", {}) or {})
refine = dict(surface.get("refine", {}) or {})
screen = dict(surface.get("screen", {}) or {})


PUBLICATION_PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "dopingflow_leaching_plot",
        "scale": 3,
    },
}


def _publication_style(
    fig: go.Figure,
    *,
    height: int = 620,
    is_3d: bool = False,
) -> go.Figure:
    """Apply a high-contrast paper/slide style to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial, Helvetica, sans-serif",
            size=16,
            color="black",
        ),
        title=dict(
            font=dict(size=22, color="black"),
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
        ),
        legend=dict(
            font=dict(size=15, color="black"),
            title_font=dict(size=15, color="black"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,0,0,0.35)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#333333",
            font=dict(size=14, color="black"),
        ),
        margin=dict(l=90, r=35, t=80, b=80),
    )

    if not is_3d:
        axis_style = dict(
            showline=True,
            linewidth=1.5,
            linecolor="black",
            mirror=True,
            ticks="outside",
            tickwidth=1.4,
            ticklen=6,
            tickcolor="black",
            tickfont=dict(size=16, color="black"),
            title_font=dict(size=20, color="black"),
            gridcolor="#E3E3E3",
            gridwidth=1,
            zeroline=False,
            automargin=True,
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)

        for trace in fig.data:
            if getattr(trace, "type", "") == "scatter":
                mode = str(getattr(trace, "mode", "") or "")
                if "lines" in mode:
                    trace.update(line=dict(width=3.0))
                if "markers" in mode:
                    trace.update(
                        marker=dict(
                            size=9,
                            line=dict(color="black", width=0.7),
                        )
                    )
            if getattr(trace, "type", "") in {"heatmap", "contour"}:
                colorbar = getattr(trace, "colorbar", None)
                if colorbar is not None:
                    trace.update(
                        colorbar=dict(
                            tickfont=dict(size=15, color="black"),
                            title=dict(font=dict(size=16, color="black")),
                            ticks="outside",
                            tickcolor="black",
                        )
                    )
    else:
        scene = (
            fig.layout.scene.to_plotly_json()
            if fig.layout.scene is not None
            else {}
        )
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = dict(scene.get(axis_name, {}) or {})
            axis_title = dict(axis.get("title", {}) or {})
            axis_title["font"] = dict(size=18, color="black")
            axis.update(
                showbackground=True,
                backgroundcolor="white",
                gridcolor="#D9D9D9",
                linecolor="black",
                zerolinecolor="#BDBDBD",
                tickfont=dict(size=14, color="black"),
                title=axis_title,
            )
            scene[axis_name] = axis
        fig.update_layout(scene=scene)
        for trace in fig.data:
            if getattr(trace, "type", "") == "surface":
                trace.update(
                    colorbar=dict(
                        tickfont=dict(size=15, color="black"),
                        title=dict(font=dict(size=16, color="black")),
                        ticks="outside",
                        tickcolor="black",
                    )
                )

    for annotation in fig.layout.annotations or []:
        annotation.update(font=dict(size=15, color="black"))

    return fig


def _show_publication_plot(
    fig: go.Figure,
    *,
    height: int = 620,
    is_3d: bool = False,
) -> None:
    _publication_style(fig, height=height, is_3d=is_3d)
    st.plotly_chart(
        fig,
        width="stretch",
        config=PUBLICATION_PLOT_CONFIG,
    )



def _csv(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ", ".join(str(x) for x in (value or []))


def _items(value: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in value.split(",") if x.strip()))


def _safe_surface_id(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in str(value).replace("/", "__")
    )


def _checkpoint_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("status") == "ok"


def _site_selector(surface_id: str, dopant: str, site_index: int) -> str:
    return f"{_safe_surface_id(surface_id)}/{dopant}_site_{int(site_index):04d}"


def _result_structure_path(value: Any, root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _existing_leaching_site_choices(
    root: Path,
    source_root_value: str,
    outdir_value: str,
) -> dict[str, str]:
    """Return label -> exact selector for already completed leaching sites."""
    source = Path(str(source_root_value or "")).expanduser()
    source_root = source.resolve() if source.is_absolute() else (root / source).resolve()
    out = Path(str(outdir_value or "09_leaching")).expanduser()
    result_dir = out.resolve() if out.is_absolute() else (source_root / out).resolve()
    summary_path = result_dir / "leaching_summary.csv"
    protonation_path = result_dir / "leaching_protonation_summary.csv"
    if not summary_path.exists():
        return {}

    try:
        summary = pd.read_csv(summary_path)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if summary.empty:
        return {}

    completed_h: dict[tuple[str, str, int], list[int]] = {}
    if protonation_path.exists():
        try:
            protonation = pd.read_csv(protonation_path)
        except (OSError, pd.errors.EmptyDataError):
            protonation = pd.DataFrame()
        if not protonation.empty:
            protonation = protonation.copy()
            protonation["site_index"] = pd.to_numeric(
                protonation["site_index"], errors="coerce"
            )
            protonation["h_count"] = pd.to_numeric(
                protonation["h_count"], errors="coerce"
            )
            if "status" in protonation.columns:
                protonation = protonation[
                    protonation["status"].astype(str).eq("ok")
                ]
            for keys, group in protonation.dropna(
                subset=["site_index", "h_count"]
            ).groupby(["surface_id", "dopant", "site_index"], dropna=False):
                completed_h[
                    (str(keys[0]), str(keys[1]), int(keys[2]))
                ] = sorted(
                    {
                        int(value)
                        for value in group["h_count"].tolist()
                    }
                )

    choices: dict[str, str] = {}
    for _, row in summary.iterrows():
        if "status" in summary.columns and str(row.get("status", "")) != "ok":
            continue
        try:
            sid = str(row["surface_id"])
            dopant = str(row["dopant"])
            idx = int(row["site_index"])
        except (KeyError, TypeError, ValueError):
            continue
        selector = _site_selector(sid, dopant, idx)
        target = str(row.get("target_id", ""))
        zone = str(row.get("initial_dopant_zone", ""))
        try:
            hkl = (
                f"({int(row.get('miller_h', 0))}"
                f"{int(row.get('miller_k', 0))}"
                f"{int(row.get('miller_l', 0))})"
            )
        except (TypeError, ValueError):
            hkl = ""
        h_counts = completed_h.get((sid, dopant, idx), [])
        h_text = ",".join(str(value) for value in h_counts) if h_counts else "none"
        label = (
            f"{dopant} site {idx} | {hkl} | {zone} | {target} "
            f"| completed H: {h_text}"
        )
        if label in choices:
            label = f"{label} | {selector}"
        choices[label] = selector
    return choices


def _recovery_candidates(result_dir: Path) -> tuple[dict[str, list[str]], int, int]:
    """Find already-started site jobs that do not have complete checkpoints."""
    surfaces_dir = result_dir / "surfaces"
    candidates: dict[str, list[str]] = {}
    n_bare_ok = 0
    n_protonation_ok = 0
    if not surfaces_dir.exists():
        return candidates, n_bare_ok, n_protonation_ok

    for site_dir in surfaces_dir.glob("*/*_site_*"):
        if not site_dir.is_dir():
            continue
        selector = f"{site_dir.parent.name}/{site_dir.name}"
        bare_checkpoint = site_dir / "leaching_result.json"
        if _checkpoint_ok(bare_checkpoint):
            n_bare_ok += 1
        elif (site_dir / "relax").exists() or bare_checkpoint.exists():
            candidates.setdefault(selector, []).append("bare leaching site incomplete")

        protonation_dir = site_dir / "protonation"
        if protonation_dir.exists():
            for arrangement_dir in sorted(protonation_dir.glob("H*_arr_*")):
                checkpoint = arrangement_dir / "protonation_result.json"
                if _checkpoint_ok(checkpoint):
                    n_protonation_ok += 1
                else:
                    candidates.setdefault(selector, []).append(
                        f"{arrangement_dir.name} incomplete"
                    )
    return candidates, n_bare_ok, n_protonation_ok


def _json_map(label: str, value: Any, help_text: str) -> tuple[dict[str, Any], str | None]:
    text = st.text_area(
        label,
        value=json.dumps(dict(value or {}), indent=2),
        help=help_text,
        height=120,
    )
    try:
        data = json.loads(text or "{}")
        if not isinstance(data, dict):
            raise ValueError("must be a JSON object")
        return data, None
    except (ValueError, json.JSONDecodeError) as exc:
        return dict(value or {}), f"{label}: {exc}"


def _default_calculator() -> dict[str, Any]:
    source = refine if bool(refine.get("enabled", False)) else screen
    if bool(refine.get("enabled", False)):
        fallback = {
            "backend": "mace", "model": "mh-1", "task": "matpes_r2scan",
            "device": "cpu", "gpu_id": 0, "optimizer": "bfgs",
            "fmax": 0.03, "max_steps": 500,
        }
    else:
        fallback = {
            "backend": "grace", "model": "GRACE-1L-OMAT", "task": "",
            "device": "cpu", "gpu_id": 0, "optimizer": "bfgs",
            "fmax": 0.05, "max_steps": 300,
        }
    fallback.update(source)
    return fallback


def _run(args: list[str]) -> None:
    with st.spinner("Running leaching stage..."):
        completed = subprocess.run(
            args,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
    st.session_state["leaching_stdout"] = completed.stdout
    st.session_state["leaching_stderr"] = completed.stderr
    st.session_state["leaching_returncode"] = completed.returncode
    if completed.returncode == 0:
        st.success("Leaching command finished successfully.")
    else:
        st.error(f"Leaching command exited with return code {completed.returncode}.")


with st.expander("Configuration & run controls", expanded=True):
    enabled = st.checkbox("Enable leaching stage", value=bool(saved.get("enabled", False)))

    st.subheader("Surface source and output")
    source_modes = ["auto", "final-selected", "screen-selected", "refine-summary", "screen-summary"]
    current_mode = str(saved.get("source_mode", "auto"))
    if current_mode not in source_modes:
        current_mode = "auto"

    inherited_source_root = (
        str(saved.get("source_root", "")).strip()
        or str(surface.get("source_root", "")).strip()
        or str((cfg.get("conductivity", {}) or {}).get("source_root", "")).strip()
        or str((cfg.get("oxidation", {}) or {}).get("source_root", "")).strip()
        or str((cfg.get("structure", {}) or {}).get("outdir", "random_structures")).strip()
    )

    c1, c2 = st.columns(2)
    source_root = c1.text_input(
        "Parent / source root",
        value=inherited_source_root,
        help=(
            "Relative leaching output is created inside this directory. "
            "By default it inherits the same source root used by the surface workflow."
        ),
    ).strip()
    source_mode = c2.selectbox("Surface table", source_modes, index=source_modes.index(current_mode))

    c3, c4 = st.columns(2)
    source_summary = c3.text_input(
        "Explicit surface CSV (optional)",
        value=str(saved.get("source_summary", "")),
        help=(
            "Overrides Surface table when set. Relative paths are resolved inside Parent / source root."
        ),
    ).strip()
    outdir = c4.text_input(
        "Output directory",
        value=str(saved.get("outdir", "09_leaching")),
        help="Relative paths are created inside Parent / source root; absolute paths are used exactly as entered.",
    ).strip()

    surface_include = _items(
        st.text_input(
            "Surface/target selector(s)",
            value=_csv(saved.get("surface_include", [])),
            help="Optional exact IDs or wildcards. Empty means every row in the selected surface table.",
        )
    )
    dopants = _items(
        st.text_input(
            "Dopant species",
            value=_csv(saved.get("dopant_species", [])),
            placeholder="Sb, Ti",
            help="Empty means infer dopants from the surface metadata.",
        )
    )
    q1, q2, q3 = st.columns(3)
    zones = q1.multiselect(
        "Depth zones",
        ["surface", "subsurface", "bulk"],
        default=[x for x in saved.get("zones", ["surface"]) if x in {"surface", "subsurface", "bulk"}]
        or ["surface"],
    )
    placement_side = q2.selectbox(
        "Surface side",
        ["top", "bottom", "both"],
        index=["top", "bottom", "both"].index(
            str(saved.get("placement_side", surface.get("placement_side", "top")))
            if str(saved.get("placement_side", surface.get("placement_side", "top"))) in {"top", "bottom", "both"}
            else "top"
        ),
    )
    max_sites = int(
        q3.number_input(
            "Max sites / surface / dopant",
            min_value=1,
            value=int(saved.get("max_sites_per_surface_species", 12)),
            step=1,
        )
    )

    g1, g2, g3 = st.columns(3)
    layer_tol = float(
        g1.number_input(
            "Cation-layer tolerance (Å)",
            min_value=0.01,
            value=float(saved.get("cation_layer_tolerance_A", surface.get("cation_layer_tolerance_A", 0.8))),
            step=0.05,
        )
    )
    layers_per_zone = int(
        g2.number_input(
            "Cation layers / zone",
            min_value=1,
            value=int(saved.get("layers_per_zone", surface.get("layers_per_zone", 1))),
            step=1,
        )
    )
    oxygen_cutoff = float(
        g3.number_input(
            "O-neighbor cutoff (Å)",
            min_value=0.1,
            value=float(saved.get("oxygen_neighbor_cutoff_A", 2.8)),
            step=0.1,
        )
    )

    st.subheader("Removal calculator")
    defaults = _default_calculator()
    r1, r2, r3 = st.columns(3)
    backend_default = str(saved.get("backend", defaults["backend"])).lower()
    backend = r1.selectbox(
        "Backend",
        BACKEND_CHOICES,
        index=BACKEND_CHOICES.index(backend_default) if backend_default in BACKEND_CHOICES else 0,
    )
    device_default = str(saved.get("device", defaults["device"])).lower()
    device = r2.selectbox(
        "Device",
        DEVICE_CHOICES,
        index=DEVICE_CHOICES.index(device_default) if device_default in DEVICE_CHOICES else 0,
    )
    gpu_id = int(
        r3.number_input(
            "GPU ID",
            min_value=0,
            value=int(saved.get("gpu_id", defaults.get("gpu_id", 0))),
            step=1,
            disabled=device != "cuda",
        )
    )
    m1, m2 = st.columns(2)
    model = m1.text_input("Model / checkpoint", value=str(saved.get("model", defaults["model"]))).strip()
    task = m2.text_input("Task / head", value=str(saved.get("task", defaults.get("task", "")))).strip()

    s1, s2, s3, s4 = st.columns(4)
    optimizer_default = str(saved.get("optimizer", defaults.get("optimizer", "bfgs"))).lower()
    optimizer = s1.selectbox(
        "Optimizer",
        OPTIMIZER_CHOICES,
        index=OPTIMIZER_CHOICES.index(optimizer_default) if optimizer_default in OPTIMIZER_CHOICES else 0,
    )
    fmax = float(
        s2.number_input(
            "fmax (eV/Å)",
            min_value=0.001,
            value=float(saved.get("fmax", defaults.get("fmax", 0.03))),
            step=0.005,
            format="%.3f",
        )
    )
    max_steps = int(
        s3.number_input(
            "Max steps",
            min_value=1,
            value=int(saved.get("max_steps", defaults.get("max_steps", 500))),
            step=25,
        )
    )
    relax_removed = s4.checkbox(
        "Relax removed slab",
        value=bool(saved.get("relax_removed_surface", True)),
    )

    t1, t2, t3, t4 = st.columns(4)
    reuse_parent = t1.checkbox(
        "Reuse compatible surface energy",
        value=bool(saved.get("reuse_surface_energy", True)),
    )
    relax_parent = t2.checkbox(
        "Relax parent if recomputed",
        value=bool(saved.get("relax_parent_if_recomputed", True)),
    )
    inherit_fixed = t3.checkbox(
        "Reuse surface fixed-atom rule",
        value=bool(saved.get("inherit_surface_fixed_atoms", True)),
    )
    resume_completed = t4.checkbox(
        "Resume completed sites",
        value=bool(saved.get("resume_completed", True)),
        help=(
            "Reuse compatible completed per-site leaching checkpoints after an interrupted run. "
            "Incomplete or failed sites are calculated again."
        ),
    )

    st.subheader("Post-leaching protonation (CHE)")
    st.caption(
        "Optionally protonate O atoms that were bonded to the removed dopant, relax those "
        "post-leaching structures, and reference the added H to H₂ using the computational "
        "hydrogen electrode (CHE). Bare-vacancy results are always retained for comparison."
    )
    protonation_enabled = st.checkbox(
        "Enable post-leaching protonation",
        value=bool(saved_protonation.get("enabled", False)),
    )

    p1, p2, p3, p4 = st.columns(4)
    saved_h_counts = [
        int(x) for x in saved_protonation.get("h_counts", [0, 1, 2, 3])
        if int(x) in range(0, 7)
    ]
    protonation_h_counts = p1.multiselect(
        "H counts to test",
        options=list(range(0, 7)),
        default=saved_h_counts or [0, 1, 2, 3],
        help="0 is the bare dopant-vacancy state. H atoms are added only to O neighbors of the removed dopant.",
        disabled=not protonation_enabled,
    )
    if 0 not in protonation_h_counts:
        protonation_h_counts = [0, *protonation_h_counts]

    protonation_neighbor_cutoff = float(
        p2.number_input(
            "Protonatable O cutoff (Å)",
            min_value=0.1,
            value=float(
                saved_protonation.get(
                    "neighbor_cutoff_A",
                    saved.get("oxygen_neighbor_cutoff_A", 2.8),
                )
            ),
            step=0.1,
            disabled=not protonation_enabled,
        )
    )
    protonation_oh_length = float(
        p3.number_input(
            "Initial O–H length (Å)",
            min_value=0.5,
            value=float(saved_protonation.get("oh_bond_length_A", 0.98)),
            step=0.01,
            format="%.2f",
            disabled=not protonation_enabled,
        )
    )
    protonation_max_arrangements = int(
        p4.number_input(
            "Max arrangements / H count",
            min_value=1,
            value=int(saved_protonation.get("max_arrangements_per_h_count", 5)),
            step=1,
            disabled=not protonation_enabled,
        )
    )

    pp1, pp2, pp3 = st.columns(3)
    relax_protonated = pp1.checkbox(
        "Relax protonated slabs",
        value=bool(saved_protonation.get("relax_protonated_surface", True)),
        disabled=not protonation_enabled,
    )
    compute_h2 = pp2.checkbox(
        "Calculate H₂ reference",
        value=bool(saved_protonation.get("compute_h2_reference", True)),
        help="Uses the same leaching MLFF and caches the H₂ energy.",
        disabled=not protonation_enabled,
    )
    relax_h2 = pp3.checkbox(
        "Relax H₂ reference",
        value=bool(saved_protonation.get("relax_h2_reference", True)),
        disabled=(not protonation_enabled) or (not compute_h2),
    )

    with st.expander("Advanced H₂ reference settings", expanded=False):
        hp1, hp2, hp3 = st.columns(3)
        manual_h2_default = saved_protonation.get("manual_h2_energy_eV", "")
        if manual_h2_default is None:
            manual_h2_default = ""
        manual_h2_text = hp1.text_input(
            "Manual E(H₂) (eV, optional)",
            value=str(manual_h2_default),
            help="If supplied, this overrides the calculated H₂ reference.",
            disabled=not protonation_enabled,
        ).strip()
        h2_bond_length = float(
            hp2.number_input(
                "Initial H–H length (Å)",
                min_value=0.2,
                value=float(saved_protonation.get("h2_bond_length_A", 0.74)),
                step=0.01,
                format="%.2f",
                disabled=(not protonation_enabled) or bool(manual_h2_text),
            )
        )
        h2_box = float(
            hp3.number_input(
                "H₂ box size (Å)",
                min_value=5.0,
                value=float(saved_protonation.get("h2_box_A", 15.0)),
                step=1.0,
                disabled=(not protonation_enabled) or bool(manual_h2_text),
            )
        )

    manual_h2_error = None
    manual_h2_value: float | str = ""
    if manual_h2_text:
        try:
            manual_h2_value = float(manual_h2_text)
        except ValueError:
            manual_h2_error = "Manual E(H₂) must be a number or left empty."
            st.error(manual_h2_error)

    st.subheader("Continuation / exact-site selection")
    st.caption(
        "Use this to extend protonation scans only for leaching sites that were already "
        "calculated. With Resume completed sites enabled, existing H-state checkpoints "
        "are reused and only missing requested H states are calculated."
    )

    existing_site_choices = _existing_leaching_site_choices(
        project_root,
        source_root,
        outdir,
    )
    saved_site_include = [
        str(value) for value in saved.get("site_include", []) or []
    ]
    continuation_default = bool(saved_site_include)
    restrict_existing_sites = st.checkbox(
        "Restrict this run to selected previously calculated sites",
        value=continuation_default,
        help=(
            "This is an exact atom-site filter, unlike Surface/target selector(s), "
            "which filters whole surface structures."
        ),
    )

    selected_existing_labels: list[str] = []
    if existing_site_choices:
        selector_to_label = {
            selector: label for label, selector in existing_site_choices.items()
        }
        default_labels = [
            selector_to_label[selector]
            for selector in saved_site_include
            if selector in selector_to_label
        ]
        selected_existing_labels = st.multiselect(
            "Previously calculated sites to continue",
            list(existing_site_choices.keys()),
            default=default_labels,
            disabled=not restrict_existing_sites,
            help=(
                "Choose the exact Sb/In atom sites to extend. The label also shows which "
                "H counts are already present in the protonation summary."
            ),
        )
    else:
        st.info(
            "No existing leaching_summary.csv was found in the current output directory, "
            "so there are no previous sites to select yet."
        )

    site_include = (
        [
            existing_site_choices[label]
            for label in selected_existing_labels
        ]
        if restrict_existing_sites
        else []
    )

    if restrict_existing_sites:
        if not site_include:
            st.warning(
                "Exact-site restriction is enabled but no previous site is selected."
            )
        if not resume_completed:
            st.error(
                "Continuation should use **Resume completed sites**; otherwise existing "
                "bare/protonation calculations can be recomputed."
            )
        if protonation_enabled:
            requested_counts = sorted(
                set(int(value) for value in protonation_h_counts) | {0}
            )
            st.info(
                "For an extension from the existing 0H–3H scan to 4H/5H, keep "
                "**H counts to test = 0,1,2,3,4,5**. Existing 1H–3H arrangements "
                "will be reused; missing 4H/5H arrangements will be calculated."
            )

    st.subheader("Metal reference")
    a1, a2, a3 = st.columns(3)
    reference_file = a1.text_input(
        "Reference-energy cache",
        value=str(saved.get("reference_energies_file", "reference_structures/reference_energies.json")),
    ).strip()
    metals_dir = a2.text_input(
        "Metal POSCAR directory",
        value=str(saved.get("metals_dir", "reference_structures/metals")),
    ).strip()
    compute_missing = a3.checkbox(
        "Calculate missing metal references",
        value=bool(saved.get("compute_missing_metal_references", True)),
    )
    relax_metal = st.checkbox(
        "Relax metal reference",
        value=bool(saved.get("relax_metal_reference", False)),
    )

    st.subheader("Electrochemical model definition")
    st.info(
        "Keep only the chemistry-defining redox data here. Temperature, ion activity, "
        "pH, applied-potential range, and SHE/RHE are post-processing controls and are "
        "now edited interactively in the Results → Thermodynamic explorer."
    )
    j1, j2 = st.columns(2)
    with j1:
        oxidation_states, err1 = _json_map(
            "Oxidation states / electron counts",
            saved.get("oxidation_states", {}),
            'JSON object, e.g. {"Sb": 5, "In": 3}.',
        )
        aqueous_species, err2 = _json_map(
            "Aqueous species labels",
            saved.get("aqueous_species", {}),
            'JSON object, e.g. {"Sb": "effective Sb(V) oxide/Sb 5e couple", "In": "In3+(aq)"}.',
        )
    with j2:
        standard_potentials, err3 = _json_map(
            "Standard reduction potentials (V vs SHE)",
            saved.get("standard_reduction_potentials_V_SHE", {}),
            "Use validated values for the exact redox reaction being modeled.",
        )
        st.caption(
            "These redox quantities define the thermodynamic model and remain saved with "
            "the calculation. Environmental analysis variables are intentionally not here."
        )
    redox_error = next((x for x in (err1, err2, err3) if x), None)
    if redox_error:
        st.error(redox_error)

    # Backward-compatible analysis defaults. They are saved for reproducibility but
    # are no longer exposed as calculation/run controls.
    legacy_potentials = [
        float(x) for x in saved.get("potentials_V", [1.23, 1.50, 1.70])
    ]
    analysis_defaults = dict(saved_analysis)
    analysis_defaults.setdefault(
        "temperature_K", float(saved.get("temperature_K", 298.15))
    )
    analysis_defaults.setdefault(
        "default_ion_activity", float(saved.get("default_ion_activity", 1e-6))
    )
    analysis_defaults.setdefault(
        "ion_activities", dict(saved.get("ion_activities", {}) or {})
    )
    analysis_defaults.setdefault(
        "potential_scale", str(saved.get("potential_scale", "RHE")).upper()
    )
    analysis_defaults.setdefault(
        "potential_min_V", min(legacy_potentials) if legacy_potentials else 1.0
    )
    analysis_defaults.setdefault(
        "potential_max_V", max(legacy_potentials) if legacy_potentials else 2.0
    )
    analysis_defaults.setdefault("potential_step_V", 0.02)
    analysis_defaults.setdefault("pH_min", 0.0)
    analysis_defaults.setdefault("pH_max", max(3.0, float(saved.get("pH", 0.0))))
    analysis_defaults.setdefault("pH_step", 0.1)
    analysis_defaults.setdefault(
        "selected_pH_values", [float(saved.get("pH", 0.0))]
    )
    analysis_defaults.setdefault(
        "selected_potential_values", legacy_potentials or [1.23, 1.50, 1.70]
    )

    resolved = dict(saved)
    resolved.update(
        enabled=bool(enabled),
        source_root=source_root,
        source_mode=source_mode,
        source_summary=source_summary,
        surface_include=surface_include,
        site_include=site_include,
        dopant_species=dopants,
        zones=zones,
        placement_side=placement_side,
        cation_layer_tolerance_A=layer_tol,
        layers_per_zone=layers_per_zone,
        max_sites_per_surface_species=max_sites,
        oxygen_neighbor_cutoff_A=oxygen_cutoff,
        outdir=outdir,
        backend=backend,
        model=model,
        task=task,
        device=device,
        gpu_id=gpu_id,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        reuse_surface_energy=bool(reuse_parent),
        relax_parent_if_recomputed=bool(relax_parent),
        relax_removed_surface=bool(relax_removed),
        inherit_surface_fixed_atoms=bool(inherit_fixed),
        resume_completed=bool(resume_completed),
        protonation={
            "enabled": bool(protonation_enabled),
            "h_counts": sorted(set(int(x) for x in protonation_h_counts) | {0}),
            "neighbor_cutoff_A": protonation_neighbor_cutoff,
            "oh_bond_length_A": protonation_oh_length,
            "max_arrangements_per_h_count": protonation_max_arrangements,
            "relax_protonated_surface": bool(relax_protonated),
            "manual_h2_energy_eV": manual_h2_value,
            "compute_h2_reference": bool(compute_h2),
            "relax_h2_reference": bool(relax_h2),
            "h2_bond_length_A": h2_bond_length,
            "h2_box_A": h2_box,
        },
        reference_energies_file=reference_file,
        metals_dir=metals_dir,
        compute_missing_metal_references=bool(compute_missing),
        relax_metal_reference=bool(relax_metal),
        oxidation_states=oxidation_states,
        standard_reduction_potentials_V_SHE=standard_potentials,
        aqueous_species=aqueous_species,
        analysis_defaults=analysis_defaults,
        # Legacy aliases remain synchronized so existing fixed summary/scan output
        # code keeps working without making these look like expensive run settings.
        ion_activities=dict(analysis_defaults.get("ion_activities", {}) or {}),
        default_ion_activity=float(analysis_defaults.get("default_ion_activity", 1e-6)),
        temperature_K=float(analysis_defaults.get("temperature_K", 298.15)),
        pH=float((analysis_defaults.get("selected_pH_values") or [0.0])[0]),
        potential_scale=str(analysis_defaults.get("potential_scale", "RHE")).upper(),
        potentials_V=[
            float(x)
            for x in (analysis_defaults.get("selected_potential_values") or [1.23, 1.50, 1.70])
        ],
    )
    resolved_cfg = dict(cfg)
    resolved_cfg["leaching"] = resolved

    validation_error = redox_error or manual_h2_error
    if restrict_existing_sites and not site_include and validation_error is None:
        validation_error = (
            "Exact-site continuation is enabled, but no previous site is selected."
        )
        st.error(validation_error)
    if restrict_existing_sites and not resume_completed and validation_error is None:
        validation_error = (
            "Exact-site continuation requires Resume completed sites to be enabled."
        )
        st.error(validation_error)
    if validation_error is None:
        try:
            parse_leaching_config(resolved_cfg, project_root)
        except Exception as exc:
            validation_error = str(exc)
            st.error(validation_error)

    preview = pd.DataFrame()
    with st.expander("Preview exact dopant-removal sites", expanded=False):
        if validation_error is None:
            try:
                preview = preview_leaching_sites(resolved_cfg, project_root)
            except Exception as exc:
                st.warning(str(exc))
            else:
                st.caption(f"{len(preview)} atom-removal job(s).")
                columns = [
                    c for c in (
                        "target_id", "miller_h", "miller_k", "miller_l", "termination_id",
                        "variant_label", "dopant", "site_index", "detected_zone",
                        "oxygen_coordination_within_cutoff", "nearest_oxygen_distance_A",
                        "protonatable_oxygen_count", "protonation_jobs_estimated",
                    ) if c in preview.columns
                ]
                st.dataframe(preview[columns], use_container_width=True, hide_index=True)

    recovery_mode = False
    recovery_selector = ""
    try:
        recovery_result_dir = resolve_leaching_output_dir(
            resolved_cfg,
            parse_leaching_config(resolved_cfg, project_root),
            project_root,
        )
    except Exception:
        recovery_result_dir = None

    with st.expander("Recovery / partial-result mode", expanded=False):
        st.caption(
            "Use this after an interrupted run when you want to finish one selected "
            "leaching site, stop cleanly there, and rebuild the CSV/JSON results only "
            "from sites completed up to that point."
        )
        detected: dict[str, list[str]] = {}
        bare_ok = proton_ok = 0
        if recovery_result_dir is not None:
            detected, bare_ok, proton_ok = _recovery_candidates(recovery_result_dir)
            st.write(
                f"Detected checkpoints: **{bare_ok} completed bare sites** and "
                f"**{proton_ok} completed protonation arrangements**."
            )
            if detected:
                for selector, reasons in detected.items():
                    st.warning(
                        f"Interrupted/incomplete site detected: `{selector}` — "
                        + ", ".join(reasons)
                    )
            else:
                st.info("No already-started incomplete site was detected in the output tree.")

        recovery_mode = st.checkbox(
            "Enable recovery / stop-after-site mode",
            value=False,
            help=(
                "Normal completed checkpoints are reused. The selected site is completed, "
                "then DopingFlow stops before starting any later site and writes partial summaries."
            ),
        )

        site_options: list[str] = []
        if not preview.empty:
            site_options = [
                _site_selector(
                    str(row["surface_id"]),
                    str(row["dopant"]),
                    int(row["site_index"]),
                )
                for _, row in preview.iterrows()
            ]
        for selector in detected:
            if selector not in site_options:
                site_options.append(selector)

        default_index = 0
        if detected and site_options:
            first_detected = next(iter(detected))
            if first_detected in site_options:
                default_index = site_options.index(first_detected)

        if site_options:
            recovery_selector = st.selectbox(
                "Finish through this site, then stop",
                site_options,
                index=default_index,
                disabled=not recovery_mode,
                help=(
                    "Use the full surface/site identifier. For an interrupted protonation "
                    "job, selecting its parent site finishes all remaining arrangements for "
                    "that site and then stops."
                ),
            )
        else:
            st.caption("No selected leaching sites are available for recovery.")

        if recovery_mode:
            st.info(
                "Normal **Run leaching** is disabled while recovery mode is active. "
                "Use **Run recovery to selected site** below."
            )

    with st.expander("Preview [leaching] TOML", expanded=False):
        st.code(toml.dumps({"leaching": resolved}), language="toml")

    save_col, dry_col, run_col, recovery_col = st.columns(4)
    with save_col:
        save = st.button(
            "Save leaching settings",
            type="primary",
            use_container_width=True,
            disabled=validation_error is not None,
        )
    with dry_col:
        dry = st.button(
            "Run leaching dry-run",
            use_container_width=True,
            disabled=(not enabled) or validation_error is not None,
        )
    with run_col:
        run = st.button(
            "Run leaching",
            use_container_width=True,
            disabled=(not enabled) or validation_error is not None or recovery_mode,
        )
    with recovery_col:
        recovery_run = st.button(
            "Run recovery to selected site",
            use_container_width=True,
            disabled=(
                (not enabled)
                or validation_error is not None
                or (not recovery_mode)
                or (not recovery_selector)
                or (not resume_completed)
            ),
            help=(
                "Requires Resume completed sites. Reuses compatible checkpoints, "
                "finishes the selected site, writes partial outputs, and stops."
            ),
        )

    if not enabled:
        st.caption(
            "Enable **leaching stage** to activate the dry-run and leaching run actions."
        )
    if recovery_mode and not resume_completed:
        st.error(
            "Recovery mode requires **Resume completed sites** to remain enabled; "
            "otherwise completed jobs would be recalculated."
        )

    if save or dry or run or recovery_run:
        cfg["leaching"] = resolved
        config_path.write_text(toml.dumps(cfg), encoding="utf-8")
        st.success(f"Saved {config_path}")
        if dry:
            _run(["dopingflow", "leaching", "-c", str(config_path), "--dry-run"])
        elif run:
            _run(["dopingflow", "leaching", "-c", str(config_path)])
        elif recovery_run:
            _run(
                [
                    "dopingflow",
                    "leaching",
                    "-c",
                    str(config_path),
                    "--stop-after-site",
                    recovery_selector,
                ]
            )


if "leaching_stdout" in st.session_state:
    with st.expander("Last command output", expanded=False):
        if st.session_state.get("leaching_stdout"):
            st.code(st.session_state["leaching_stdout"])
        if st.session_state.get("leaching_stderr"):
            st.code(st.session_state["leaching_stderr"])


st.divider()
st.subheader("Results")
try:
    parsed = parse_leaching_config(cfg, project_root)
    result_dir = resolve_leaching_output_dir(cfg, parsed, project_root)
except Exception:
    fallback_source = (
        str(saved.get("source_root", "")).strip()
        or str(surface.get("source_root", "")).strip()
        or str((cfg.get("structure", {}) or {}).get("outdir", "random_structures")).strip()
    )
    source_path = Path(fallback_source).expanduser()
    if not source_path.is_absolute():
        source_path = (project_root / source_path).resolve()
    result_dir = Path(saved.get("outdir", "09_leaching")).expanduser()
    if not result_dir.is_absolute():
        result_dir = (source_path / result_dir).resolve()

summary_path = result_dir / "leaching_summary.csv"
summary_json_path = result_dir / "leaching_results.json"
aggregate_path = result_dir / "leaching_surface_summary.csv"
potential_path = result_dir / "leaching_potential_scan.csv"
protonation_summary_path = result_dir / "leaching_protonation_summary.csv"
protonation_scan_path = result_dir / "leaching_protonation_potential_scan.csv"

if not summary_path.exists():
    st.info("No leaching_summary.csv exists yet.")
else:
    if summary_json_path.exists():
        try:
            result_meta = json.loads(summary_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result_meta = {}
        if bool(result_meta.get("partial_run", False)):
            stopped = str(result_meta.get("stopped_after_site", "")).strip()
            st.info(
                "These are **partial recovery results**. DopingFlow stopped cleanly "
                + (f"after `{stopped}`." if stopped else "at the requested recovery site.")
            )
    results = pd.read_csv(summary_path)
    try:
        parsed_for_results = parse_leaching_config(cfg, project_root)
        analysis_cfg = dict(parsed_for_results.get("analysis_defaults", {}) or {})
    except Exception:
        analysis_cfg = dict(saved_analysis)

    if protonation_summary_path.exists():
        try:
            protonation_table_all = pd.read_csv(protonation_summary_path)
        except pd.errors.EmptyDataError:
            protonation_table_all = pd.DataFrame()
    else:
        protonation_table_all = pd.DataFrame()

    tabs = st.tabs(
        [
            "Site results",
            "Surface summary",
            "Saved bare scan",
            "Protonation states",
            "Thermodynamic explorer",
            "Site environment",
            "Interpretation",
        ]
    )

    with tabs[0]:
        st.dataframe(results, use_container_width=True, hide_index=True)
        if "extraction_energy_eV" in results.columns:
            plot = results.copy()
            plot["extraction_energy_eV"] = pd.to_numeric(plot["extraction_energy_eV"], errors="coerce")
            plot = plot[plot["extraction_energy_eV"].notna()]
            if not plot.empty:
                fig = px.scatter(
                    plot,
                    x="dopant",
                    y="extraction_energy_eV",
                    color="initial_dopant_zone" if "initial_dopant_zone" in plot.columns else None,
                    hover_data=[c for c in ("target_id", "variant_label", "site_index", "initial_dopant_zone", "surface_variant_declared_zone", "initial_depth_from_selected_surface_A") if c in plot.columns],
                    title="Metal-referenced dopant extraction energy",
                )
                _show_publication_plot(fig)

    with tabs[1]:
        if aggregate_path.exists():
            st.dataframe(pd.read_csv(aggregate_path), use_container_width=True, hide_index=True)
        else:
            st.info("No aggregate table found.")

    with tabs[2]:
        selected_scale = str(
            (cfg.get("leaching", {}) or {}).get("potential_scale", "RHE")
        ).upper()
        threshold_col = (
            "dissolution_potential_V_RHE"
            if selected_scale == "RHE"
            else "dissolution_potential_V_SHE"
        )

        if threshold_col in results.columns:
            threshold = results.copy()
            threshold[threshold_col] = pd.to_numeric(
                threshold[threshold_col], errors="coerce"
            )
            threshold = threshold[threshold[threshold_col].notna()]
            if not threshold.empty:
                st.markdown("#### Bare-vacancy dissolution threshold potential")
                st.caption(
                    "Lower threshold potential means that the modeled dopant dissolution "
                    "becomes thermodynamically favorable at a lower applied potential."
                )
                fig_threshold = px.scatter(
                    threshold,
                    x="dopant",
                    y=threshold_col,
                    color=(
                        "initial_dopant_zone"
                        if "initial_dopant_zone" in threshold.columns
                        else None
                    ),
                    hover_data=[
                        col
                        for col in (
                            "target_id",
                            "variant_label",
                            "site_index",
                            "initial_dopant_zone",
                            "initial_depth_from_selected_surface_A",
                            "extraction_energy_eV",
                        )
                        if col in threshold.columns
                    ],
                    title=f"Bare-vacancy dissolution threshold vs {selected_scale}",
                    labels={
                        threshold_col: f"Dissolution potential (V vs {selected_scale})",
                        "dopant": "Dopant",
                    },
                )
                _show_publication_plot(fig_threshold)

        if potential_path.exists():
            try:
                scan = pd.read_csv(potential_path)
            except pd.errors.EmptyDataError:
                scan = pd.DataFrame()
            if scan.empty:
                st.info(
                    "No electrochemical ΔG_leach scan is available. Complete redox data "
                    "(oxidation state/electron count and standard reduction potential) "
                    "must be supplied for the dopant."
                )
            else:
                st.markdown("#### Bare-vacancy leaching free energy at operating potentials")
                st.caption(
                    "ΔG_leach < 0 means dissolution is thermodynamically favorable in "
                    "the current simple-ion model; ΔG_leach > 0 means it is unfavorable."
                )
                st.dataframe(scan, use_container_width=True, hide_index=True)
                fig = px.line(
                    scan,
                    x="applied_potential_V",
                    y="deltaG_leach_eV",
                    color="dopant",
                    line_group="surface_id",
                    markers=True,
                    hover_data=[
                        col
                        for col in (
                            "target_id",
                            "site_index",
                            "initial_dopant_zone",
                            "initial_depth_from_selected_surface_A",
                        )
                        if col in scan.columns
                    ],
                    title=f"Bare-vacancy leaching free energy vs potential ({selected_scale})",
                    labels={
                        "applied_potential_V": f"Applied potential (V vs {selected_scale})",
                        "deltaG_leach_eV": "ΔG_leach (eV)",
                        "dopant": "Dopant",
                    },
                )
                fig.add_hline(
                    y=0.0,
                    line_dash="dash",
                    annotation_text="ΔG_leach = 0",
                    annotation_position="top left",
                )
                _show_publication_plot(fig)
        else:
            st.info("No potential scan found.")

    with tabs[3]:
        selected_scale = str(
            (cfg.get("leaching", {}) or {}).get("potential_scale", "RHE")
        ).upper()
        adjusted_threshold_col = (
            "protonation_adjusted_dissolution_potential_V_RHE"
            if selected_scale == "RHE"
            else "protonation_adjusted_dissolution_potential_V_SHE"
        )
        bare_threshold_col = (
            "dissolution_potential_V_RHE"
            if selected_scale == "RHE"
            else "dissolution_potential_V_SHE"
        )

        if protonation_summary_path.exists():
            try:
                protonation_table = pd.read_csv(protonation_summary_path)
            except pd.errors.EmptyDataError:
                protonation_table = pd.DataFrame()
        else:
            protonation_table = pd.DataFrame()

        if protonation_table.empty:
            st.info(
                "No post-leaching protonation results are available. Enable "
                "**post-leaching protonation** and run the leaching stage."
            )
        else:
            st.markdown("#### Protonated post-leaching structures")
            st.caption(
                "ΔE_protonation(0 V) = E(defect+nH) − E(defect) − n/2 E(H₂). "
                "Negative values mean protonation stabilizes the dopant-vacancy surface "
                "relative to the bare vacancy at 0 V vs SHE and pH 0. The plot uses the "
                "lowest-energy arrangement for each H count; the table keeps all arrangements."
            )
            st.dataframe(
                protonation_table,
                use_container_width=True,
                hide_index=True,
            )

            structural = protonation_table.copy()
            structural["deltaE_protonation_zeroV_eV"] = pd.to_numeric(
                structural.get("deltaE_protonation_zeroV_eV"),
                errors="coerce",
            )
            structural["h_count"] = pd.to_numeric(
                structural.get("h_count"), errors="coerce"
            )
            structural = structural[
                structural["deltaE_protonation_zeroV_eV"].notna()
                & structural["h_count"].notna()
            ]
            if not structural.empty:
                structural["site_key"] = (
                    structural["surface_id"].astype(str)
                    + "::"
                    + structural["dopant"].astype(str)
                    + "::"
                    + structural["site_index"].astype(str)
                )
                # The table retains every generated arrangement. For the trend
                # plot, show only the lowest-energy arrangement at each H count.
                structural = (
                    structural.sort_values("deltaE_protonation_zeroV_eV")
                    .drop_duplicates(
                        subset=["site_key", "h_count"],
                        keep="first",
                    )
                    .sort_values(["site_key", "h_count"])
                )
                fig_prot = px.line(
                    structural,
                    x="h_count",
                    y="deltaE_protonation_zeroV_eV",
                    color="dopant",
                    line_group="site_key",
                    markers=True,
                    hover_data=[
                        col
                        for col in (
                            "surface_id",
                            "site_index",
                            "initial_dopant_zone",
                            "arrangement_id",
                            "oxygen_indices_json",
                        )
                        if col in structural.columns
                    ],
                    title="Post-leaching protonation stabilization",
                    labels={
                        "h_count": "Number of H atoms",
                        "deltaE_protonation_zeroV_eV": "ΔE protonation at 0 V (eV)",
                        "dopant": "Dopant",
                    },
                )
                fig_prot.add_hline(
                    y=0.0,
                    line_dash="dash",
                    annotation_text="Bare vacancy reference",
                    annotation_position="top left",
                )
                _show_publication_plot(fig_prot)

        if protonation_scan_path.exists():
            try:
                proton_scan = pd.read_csv(protonation_scan_path)
            except pd.errors.EmptyDataError:
                proton_scan = pd.DataFrame()
        else:
            proton_scan = pd.DataFrame()

        if not proton_scan.empty:
            st.markdown("#### Protonation-adjusted leaching free energy")
            st.caption(
                "At each operating potential, DopingFlow compares the bare vacancy with "
                "all successfully relaxed protonated states and reports the lowest "
                "ΔG_leach. The selected H count can therefore change with potential."
            )
            st.dataframe(proton_scan, use_container_width=True, hide_index=True)
            proton_scan["site_key"] = (
                proton_scan["surface_id"].astype(str)
                + "::"
                + proton_scan["dopant"].astype(str)
                + "::"
                + proton_scan["site_index"].astype(str)
            )
            fig_scan = px.line(
                proton_scan,
                x="applied_potential_V",
                y="best_deltaG_leach_eV",
                color="dopant",
                line_group="site_key",
                markers=True,
                hover_data=[
                    col
                    for col in (
                        "surface_id",
                        "site_index",
                        "initial_dopant_zone",
                        "best_h_count",
                        "best_arrangement_id",
                        "bare_deltaG_leach_eV",
                        "deltaG_change_vs_bare_eV",
                    )
                    if col in proton_scan.columns
                ],
                title=f"Minimum ΔG_leach over all available H states ({selected_scale})",
                labels={
                    "applied_potential_V": f"Applied potential (V vs {selected_scale})",
                    "best_deltaG_leach_eV": "Minimum ΔG_leach (eV)",
                    "dopant": "Dopant",
                },
            )
            fig_scan.add_hline(
                y=0.0,
                line_dash="dash",
                annotation_text="ΔG_leach = 0",
                annotation_position="top left",
            )
            _show_publication_plot(fig_scan)

        if (
            bare_threshold_col in results.columns
            and adjusted_threshold_col in results.columns
        ):
            threshold_columns = [
                col
                for col in (
                    "dopant",
                    "site_index",
                    "initial_dopant_zone",
                    bare_threshold_col,
                    adjusted_threshold_col,
                    "protonation_adjusted_threshold_h_count",
                    "protonation_adjusted_threshold_arrangement_id",
                )
                if col in results.columns
            ]
            threshold_compare = results[threshold_columns].copy()
            threshold_compare[bare_threshold_col] = pd.to_numeric(
                threshold_compare[bare_threshold_col], errors="coerce"
            )
            threshold_compare[adjusted_threshold_col] = pd.to_numeric(
                threshold_compare[adjusted_threshold_col], errors="coerce"
            )
            threshold_compare = threshold_compare.dropna(
                subset=[bare_threshold_col, adjusted_threshold_col],
                how="all",
            )
            if not threshold_compare.empty:
                st.markdown("#### Conventional dissolution-threshold comparison")
                st.caption(
                    "This plot is a **zero-crossing threshold comparison**, not a plot of the "
                    "globally lowest ΔG state at an operating potential. The protonation-adjusted "
                    "point is the lowest conventional anodic dissolution threshold among states "
                    "with **n_H < z**. States with **n_H ≥ z** are intentionally excluded because "
                    "they do not have the same conventional high-potential dissolution onset. "
                    "Use **Best post-leaching state: ΔG_leach vs potential** or the "
                    "**Thermodynamic explorer** to compare the true minimum ΔG across all "
                    "available H states at a chosen operating potential."
                )

                common_cols = [
                    col
                    for col in ("dopant", "site_index", "initial_dopant_zone")
                    if col in threshold_compare.columns
                ]

                bare_long = threshold_compare[
                    common_cols + [bare_threshold_col]
                ].copy()
                bare_long = bare_long.rename(
                    columns={bare_threshold_col: "dissolution_potential_V"}
                )
                bare_long["threshold_model"] = "Bare vacancy (0H)"
                bare_long["threshold_h_count"] = 0
                bare_long["threshold_arrangement_id"] = 0

                adjusted_keep = common_cols + [adjusted_threshold_col]
                if "protonation_adjusted_threshold_h_count" in threshold_compare.columns:
                    adjusted_keep.append("protonation_adjusted_threshold_h_count")
                if (
                    "protonation_adjusted_threshold_arrangement_id"
                    in threshold_compare.columns
                ):
                    adjusted_keep.append(
                        "protonation_adjusted_threshold_arrangement_id"
                    )
                adjusted_long = threshold_compare[adjusted_keep].copy()
                adjusted_long = adjusted_long.rename(
                    columns={
                        adjusted_threshold_col: "dissolution_potential_V",
                        "protonation_adjusted_threshold_h_count": "threshold_h_count",
                        "protonation_adjusted_threshold_arrangement_id": (
                            "threshold_arrangement_id"
                        ),
                    }
                )
                adjusted_long["threshold_model"] = (
                    "Lowest eligible protonated threshold (n_H < z)"
                )

                long_frames = [
                    bare_long.dropna(subset=["dissolution_potential_V"]),
                    adjusted_long.dropna(subset=["dissolution_potential_V"]),
                ]
                threshold_long = pd.concat(long_frames, ignore_index=True)

                hover_fields = [
                    col
                    for col in (
                        "site_index",
                        "initial_dopant_zone",
                        "threshold_h_count",
                        "threshold_arrangement_id",
                    )
                    if col in threshold_long.columns
                ]
                fig_threshold_compare = px.scatter(
                    threshold_long,
                    x="dopant",
                    y="dissolution_potential_V",
                    color="threshold_model",
                    symbol=(
                        "initial_dopant_zone"
                        if "initial_dopant_zone" in threshold_long.columns
                        else None
                    ),
                    hover_data=hover_fields,
                    title=(
                        f"Conventional dissolution thresholds ({selected_scale}; n_H < z)"
                    ),
                    labels={
                        "dissolution_potential_V": (
                            f"Dissolution threshold potential (V vs {selected_scale})"
                        ),
                        "threshold_model": "Threshold definition",
                        "threshold_h_count": "H count used for threshold",
                        "threshold_arrangement_id": "Arrangement",
                        "dopant": "Dopant",
                    },
                )
                _show_publication_plot(fig_threshold_compare)

    with tabs[4]:
        st.markdown("### Interactive thermodynamic explorer")
        st.caption(
            "These controls are pure post-processing. Changing temperature, activity, pH, "
            "potential range, or SHE/RHE does **not** rerun MLFF relaxations or invalidate "
            "the stored bare/protonated structures."
        )

        explorer_rows = results.copy()
        for col in ("oxidation_state", "standard_reduction_potential_V_SHE", "extraction_energy_eV"):
            if col in explorer_rows.columns:
                explorer_rows[col] = pd.to_numeric(explorer_rows[col], errors="coerce")
        required = [
            col
            for col in ("oxidation_state", "standard_reduction_potential_V_SHE", "extraction_energy_eV")
            if col in explorer_rows.columns
        ]
        if len(required) < 3:
            explorer_rows = pd.DataFrame()
        else:
            explorer_rows = explorer_rows.dropna(subset=required)

        if explorer_rows.empty:
            st.info(
                "No completed site with extraction energy, oxidation state, and standard "
                "reduction potential is available for interactive thermodynamic analysis."
            )
        else:
            site_labels: dict[str, int] = {}
            for ridx, row in explorer_rows.iterrows():
                hkl = ""
                if all(k in row for k in ("miller_h", "miller_k", "miller_l")):
                    hkl = f"({int(row['miller_h'])}{int(row['miller_k'])}{int(row['miller_l'])})"
                zone = str(row.get("initial_dopant_zone", ""))
                label = (
                    f"{row['dopant']} site {int(row['site_index'])} | {hkl} | "
                    f"{zone} | {row.get('target_id', row.get('surface_id', ''))}"
                )
                site_labels[label] = ridx

            def _states_for_explorer_row(row: pd.Series) -> list[dict[str, Any]]:
                surface_id = str(row["surface_id"])
                dopant = str(row["dopant"])
                site_index = int(row["site_index"])
                site_states: list[dict[str, Any]] = []
                if not protonation_table_all.empty:
                    table = protonation_table_all[
                        (protonation_table_all["surface_id"].astype(str) == surface_id)
                        & (protonation_table_all["dopant"].astype(str) == dopant)
                        & (
                            pd.to_numeric(
                                protonation_table_all["site_index"], errors="coerce"
                            )
                            == site_index
                        )
                    ].copy()
                    if "extraction_base_eV" in table.columns:
                        table["extraction_base_eV"] = pd.to_numeric(
                            table["extraction_base_eV"], errors="coerce"
                        )
                        table["h_count"] = pd.to_numeric(
                            table.get("h_count"), errors="coerce"
                        )
                        table = table.dropna(
                            subset=["extraction_base_eV", "h_count"]
                        )
                        site_states = table.to_dict("records")
                if not site_states:
                    site_states = [
                        {
                            "h_count": 0,
                            "arrangement_id": 0,
                            "extraction_base_eV": float(row["extraction_energy_eV"]),
                        }
                    ]
                return site_states

            def _h_mode_to_count(mode: str) -> int | None:
                return None if mode == "Best" else int(mode.removesuffix("H"))

            all_site_labels = list(site_labels.keys())
            comparison_labels = st.multiselect(
                "Structures/sites for 2D comparison",
                all_site_labels,
                default=all_site_labels[:1],
                help=(
                    "Select several surface structures/sites to overlay their ΔG curves. "
                    "Selections may include different dopants such as Sb and In."
                ),
                key="leaching_explorer_comparison_sites",
            )
            if not comparison_labels:
                st.warning("Select at least one structure/site for the 2D comparison plots.")
                comparison_labels = all_site_labels[:1]

            comparison_h_values: set[int] = set()
            for label in comparison_labels:
                row = explorer_rows.loc[site_labels[label]]
                comparison_h_values.update(
                    int(state["h_count"]) for state in _states_for_explorer_row(row)
                )
            comparison_h_options = ["Best"] + [
                f"{h}H" for h in sorted(comparison_h_values)
            ]
            comparison_h_modes = st.multiselect(
                "H state(s) for 2D comparison",
                comparison_h_options,
                default=["Best"],
                help=(
                    "Best minimizes ΔG over all calculated H states at every U/pH point. "
                    "Choose explicit 0H/1H/2H/... states to compare fixed protonation levels."
                ),
                key="leaching_explorer_comparison_h_states",
            )
            if not comparison_h_modes:
                comparison_h_modes = ["Best"]

            activity_map_saved = dict(analysis_cfg.get("ion_activities", {}) or {})
            default_activity_saved = float(
                analysis_cfg.get("default_ion_activity", 1e-6)
            )
            default_scale = str(analysis_cfg.get("potential_scale", "RHE")).upper()
            if default_scale not in {"RHE", "SHE"}:
                default_scale = "RHE"

            c1, c2, c3 = st.columns(3)
            explorer_temperature = float(
                c1.number_input(
                    "Temperature (K)",
                    min_value=1.0,
                    value=float(analysis_cfg.get("temperature_K", 298.15)),
                    step=5.0,
                    key="leaching_explorer_temperature",
                )
            )
            explorer_default_activity = float(
                c2.number_input(
                    "Default ion activity",
                    min_value=1e-20,
                    value=default_activity_saved,
                    format="%.3e",
                    key="leaching_explorer_default_activity",
                    help=(
                        "Used for selected dopants without a saved dopant-specific activity."
                    ),
                )
            )
            explorer_scale = c3.selectbox(
                "Potential scale",
                ["RHE", "SHE"],
                index=0 if default_scale == "RHE" else 1,
                key="leaching_explorer_scale",
            )

            selected_dopants = sorted(
                {
                    str(explorer_rows.loc[site_labels[label]]["dopant"])
                    for label in comparison_labels
                }
            )
            activity_by_dopant: dict[str, float] = {}
            with st.expander("Dopant-specific ion activities", expanded=False):
                st.caption(
                    "Each compared dopant can use its own activity. These are post-processing "
                    "values and do not trigger any structural calculation."
                )
                activity_cols = st.columns(min(4, max(1, len(selected_dopants))))
                for activity_index, dopant in enumerate(selected_dopants):
                    saved_value = float(
                        activity_map_saved.get(dopant, explorer_default_activity)
                    )
                    activity_by_dopant[dopant] = float(
                        activity_cols[activity_index % len(activity_cols)].number_input(
                            f"{dopant} activity",
                            min_value=1e-20,
                            value=saved_value,
                            format="%.3e",
                            key=f"leaching_explorer_activity_{dopant}",
                        )
                    )
            for dopant in selected_dopants:
                activity_by_dopant.setdefault(dopant, explorer_default_activity)

            r1, r2, r3 = st.columns(3)
            u_min = float(
                r1.number_input(
                    "Potential min (V)",
                    value=float(analysis_cfg.get("potential_min_V", 1.0)),
                    step=0.05,
                    key="leaching_explorer_umin",
                )
            )
            u_max = float(
                r2.number_input(
                    "Potential max (V)",
                    value=float(analysis_cfg.get("potential_max_V", 2.0)),
                    step=0.05,
                    key="leaching_explorer_umax",
                )
            )
            u_step = float(
                r3.number_input(
                    "Potential step (V)",
                    min_value=0.001,
                    value=float(analysis_cfg.get("potential_step_V", 0.02)),
                    step=0.005,
                    format="%.3f",
                    key="leaching_explorer_ustep",
                )
            )

            r4, r5, r6 = st.columns(3)
            ph_min = float(
                r4.number_input(
                    "pH min",
                    value=float(analysis_cfg.get("pH_min", 0.0)),
                    step=0.25,
                    key="leaching_explorer_phmin",
                )
            )
            ph_max = float(
                r5.number_input(
                    "pH max",
                    value=float(analysis_cfg.get("pH_max", 3.0)),
                    step=0.25,
                    key="leaching_explorer_phmax",
                )
            )
            ph_step = float(
                r6.number_input(
                    "pH step",
                    min_value=0.01,
                    value=float(analysis_cfg.get("pH_step", 0.1)),
                    step=0.05,
                    format="%.2f",
                    key="leaching_explorer_phstep",
                )
            )

            saved_ph_slices = analysis_cfg.get("selected_pH_values", [0.0, 1.0, 2.0])
            saved_u_slices = analysis_cfg.get(
                "selected_potential_values", [1.23, 1.50, 1.70]
            )

            selected_ph_slices: list[float] = []
            selected_u_slices: list[float] = []
            ranges_valid = u_max > u_min and ph_max > ph_min

            if not ranges_valid:
                st.error(
                    "Potential max and pH max must be larger than their corresponding minima."
                )
            else:
                u_values = np.arange(u_min, u_max + 0.5 * u_step, u_step)
                ph_values = np.arange(ph_min, ph_max + 0.5 * ph_step, ph_step)

                if len(u_values) > 300 or len(ph_values) > 300:
                    st.error(
                        "The requested grid is too dense for interactive plotting. Increase "
                        "the potential or pH step so each axis has at most 300 points."
                    )
                else:
                    comparison_specs: list[dict[str, Any]] = []
                    skipped_comparisons: list[str] = []

                    for comparison_label in comparison_labels:
                        comparison_row = explorer_rows.loc[
                            site_labels[comparison_label]
                        ]
                        comparison_states = _states_for_explorer_row(comparison_row)
                        comparison_dopant = str(comparison_row["dopant"])
                        comparison_site_index = int(comparison_row["site_index"])
                        comparison_z = int(comparison_row["oxidation_state"])
                        comparison_e0 = float(
                            comparison_row["standard_reduction_potential_V_SHE"]
                        )
                        comparison_activity = float(
                            activity_by_dopant.get(
                                comparison_dopant, explorer_default_activity
                            )
                        )
                        available_counts = {
                            int(state["h_count"]) for state in comparison_states
                        }

                        comparison_hkl = ""
                        if all(
                            key in comparison_row
                            for key in ("miller_h", "miller_k", "miller_l")
                        ):
                            comparison_hkl = (
                                f"({int(comparison_row['miller_h'])}"
                                f"{int(comparison_row['miller_k'])}"
                                f"{int(comparison_row['miller_l'])})"
                            )
                        comparison_zone = str(
                            comparison_row.get("initial_dopant_zone", "")
                        )

                        for h_mode in comparison_h_modes:
                            h_count = _h_mode_to_count(h_mode)
                            if h_count is not None and h_count not in available_counts:
                                skipped_comparisons.append(
                                    f"{comparison_dopant} site "
                                    f"{comparison_site_index}: {h_mode} unavailable"
                                )
                                continue

                            series_parts = [
                                f"{comparison_dopant} site {comparison_site_index}"
                            ]
                            if comparison_hkl:
                                series_parts.append(comparison_hkl)
                            if comparison_zone:
                                series_parts.append(comparison_zone)
                            series_parts.append(h_mode)

                            comparison_specs.append(
                                {
                                    "row": comparison_row,
                                    "states": comparison_states,
                                    "dopant": comparison_dopant,
                                    "site_index": comparison_site_index,
                                    "z": comparison_z,
                                    "e0": comparison_e0,
                                    "activity": comparison_activity,
                                    "h_mode": h_mode,
                                    "h_count": h_count,
                                    "series_label": " | ".join(series_parts),
                                }
                            )

                    if skipped_comparisons:
                        with st.expander(
                            f"Skipped unavailable H-state combinations "
                            f"({len(skipped_comparisons)})",
                            expanded=False,
                        ):
                            st.write(
                                "\n".join(
                                    f"- {item}" for item in skipped_comparisons
                                )
                            )

                    st.markdown("#### 2D comparison slices")
                    st.caption(
                        "Each line combines one selected structure/site with one requested "
                        "H state. Best means the minimum-ΔG H state can change along the curve."
                    )
                    left, right = st.columns(2)

                    with left:
                        st.markdown("##### ΔG vs potential")
                        potential_slice_mode = st.radio(
                            "pH selection mode",
                            ["Single interactive slice", "Compare multiple slices"],
                            horizontal=True,
                            key="leaching_explorer_potential_slice_mode",
                        )
                        if potential_slice_mode == "Single interactive slice":
                            default_ph = (
                                float(saved_ph_slices[0])
                                if saved_ph_slices
                                else ph_min
                            )
                            default_ph = min(max(default_ph, ph_min), ph_max)
                            selected_ph = st.slider(
                                "pH",
                                min_value=float(ph_min),
                                max_value=float(ph_max),
                                value=float(default_ph),
                                step=float(ph_step),
                                key="leaching_explorer_single_ph_slider",
                                help=(
                                    "Move the slider to update ΔG versus potential "
                                    "without rerunning any ML calculation."
                                ),
                            )
                            selected_ph_slices = [float(selected_ph)]
                        else:
                            ph_slice_text = st.text_input(
                                "pH values to compare",
                                value=", ".join(
                                    str(float(x)) for x in saved_ph_slices
                                ),
                                key="leaching_explorer_ph_slices",
                            )
                            try:
                                selected_ph_slices = [
                                    float(x.strip())
                                    for x in ph_slice_text.split(",")
                                    if x.strip()
                                ]
                            except ValueError:
                                st.error(
                                    "pH slice values must be comma-separated numbers."
                                )

                        if not selected_ph_slices:
                            st.info("Choose at least one pH value to draw this plot.")
                        else:
                            comparison_u_frames: list[pd.DataFrame] = []
                            for spec in comparison_specs:
                                grid_u = build_thermodynamic_grid(
                                    spec["states"],
                                    spec["z"],
                                    spec["e0"],
                                    u_values,
                                    selected_ph_slices,
                                    potential_scale=explorer_scale,
                                    ion_activity=spec["activity"],
                                    temperature_K=explorer_temperature,
                                    selected_h_count=spec["h_count"],
                                )
                                if grid_u.empty:
                                    continue
                                grid_u["dopant"] = spec["dopant"]
                                grid_u["site_index"] = spec["site_index"]
                                grid_u["surface_id"] = str(spec["row"]["surface_id"])
                                grid_u["requested_h_state"] = spec["h_mode"]
                                grid_u["series_label"] = spec["series_label"]
                                grid_u["ion_activity"] = spec["activity"]
                                comparison_u_frames.append(grid_u)

                            comparison_u_grid = (
                                pd.concat(comparison_u_frames, ignore_index=True)
                                if comparison_u_frames
                                else pd.DataFrame()
                            )
                            if comparison_u_grid.empty:
                                st.info(
                                    "No valid structure/H-state combination is "
                                    "available for ΔG vs potential."
                                )
                            else:
                                line_u = comparison_u_grid.copy().sort_values(
                                    ["series_label", "pH", "applied_potential_V"]
                                )
                                line_u["pH slice"] = line_u["pH"].map(
                                    lambda value: f"pH {value:g}"
                                )
                                if len(selected_ph_slices) == 1:
                                    st.caption(
                                        f"Current slice: **pH = "
                                        f"{float(selected_ph_slices[0]):g}**"
                                    )
                                line_kwargs: dict[str, Any] = dict(
                                    data_frame=line_u,
                                    x="applied_potential_V",
                                    y="deltaG_leach_eV",
                                    color="series_label",
                                    markers=False,
                                    hover_data=[
                                        "dopant",
                                        "site_index",
                                        "requested_h_state",
                                        "best_h_count",
                                        "net_electrons",
                                        "best_arrangement_id",
                                        "ion_activity",
                                    ],
                                    title=(
                                        "ΔG_leach vs potential — selected "
                                        "structures/H states"
                                    ),
                                    labels={
                                        "applied_potential_V": (
                                            f"Potential (V vs {explorer_scale})"
                                        ),
                                        "deltaG_leach_eV": "ΔG_leach (eV)",
                                        "series_label": "Structure / H state",
                                        "pH slice": "pH",
                                    },
                                )
                                if line_u["pH"].nunique() > 1:
                                    line_kwargs["line_dash"] = "pH slice"
                                fig_u = px.line(**line_kwargs)
                                fig_u.add_hline(y=0.0, line_dash="dash")
                                _show_publication_plot(fig_u, height=520)

                    with right:
                        st.markdown("##### ΔG vs pH")
                        ph_slice_mode = st.radio(
                            "Potential selection mode",
                            ["Single interactive slice", "Compare multiple slices"],
                            horizontal=True,
                            key="leaching_explorer_ph_slice_mode",
                        )
                        if ph_slice_mode == "Single interactive slice":
                            default_u = (
                                float(saved_u_slices[0])
                                if saved_u_slices
                                else u_min
                            )
                            default_u = min(max(default_u, u_min), u_max)
                            selected_u = st.slider(
                                f"Potential (V vs {explorer_scale})",
                                min_value=float(u_min),
                                max_value=float(u_max),
                                value=float(default_u),
                                step=float(u_step),
                                key="leaching_explorer_single_u_slider",
                                help=(
                                    "Move the slider to update ΔG versus pH "
                                    "without rerunning any ML calculation."
                                ),
                            )
                            selected_u_slices = [float(selected_u)]
                        else:
                            u_slice_text = st.text_input(
                                f"Potential values to compare "
                                f"(V vs {explorer_scale})",
                                value=", ".join(
                                    str(float(x)) for x in saved_u_slices
                                ),
                                key="leaching_explorer_u_slices",
                            )
                            try:
                                selected_u_slices = [
                                    float(x.strip())
                                    for x in u_slice_text.split(",")
                                    if x.strip()
                                ]
                            except ValueError:
                                st.error(
                                    "Potential slice values must be "
                                    "comma-separated numbers."
                                )

                        if not selected_u_slices:
                            st.info(
                                "Choose at least one potential value to draw this plot."
                            )
                        else:
                            comparison_ph_frames: list[pd.DataFrame] = []
                            for spec in comparison_specs:
                                grid_ph = build_thermodynamic_grid(
                                    spec["states"],
                                    spec["z"],
                                    spec["e0"],
                                    selected_u_slices,
                                    ph_values,
                                    potential_scale=explorer_scale,
                                    ion_activity=spec["activity"],
                                    temperature_K=explorer_temperature,
                                    selected_h_count=spec["h_count"],
                                )
                                if grid_ph.empty:
                                    continue
                                grid_ph["dopant"] = spec["dopant"]
                                grid_ph["site_index"] = spec["site_index"]
                                grid_ph["surface_id"] = str(spec["row"]["surface_id"])
                                grid_ph["requested_h_state"] = spec["h_mode"]
                                grid_ph["series_label"] = spec["series_label"]
                                grid_ph["ion_activity"] = spec["activity"]
                                comparison_ph_frames.append(grid_ph)

                            comparison_ph_grid = (
                                pd.concat(comparison_ph_frames, ignore_index=True)
                                if comparison_ph_frames
                                else pd.DataFrame()
                            )
                            if comparison_ph_grid.empty:
                                st.info(
                                    "No valid structure/H-state combination is "
                                    "available for ΔG vs pH."
                                )
                            else:
                                line_ph = comparison_ph_grid.copy().sort_values(
                                    [
                                        "series_label",
                                        "applied_potential_V",
                                        "pH",
                                    ]
                                )
                                line_ph["Potential slice"] = line_ph[
                                    "applied_potential_V"
                                ].map(lambda value: f"{value:g} V")
                                if len(selected_u_slices) == 1:
                                    st.caption(
                                        f"Current slice: **U = "
                                        f"{float(selected_u_slices[0]):g} V vs "
                                        f"{explorer_scale}**"
                                    )
                                ph_kwargs: dict[str, Any] = dict(
                                    data_frame=line_ph,
                                    x="pH",
                                    y="deltaG_leach_eV",
                                    color="series_label",
                                    markers=False,
                                    hover_data=[
                                        "dopant",
                                        "site_index",
                                        "requested_h_state",
                                        "best_h_count",
                                        "net_electrons",
                                        "best_arrangement_id",
                                        "ion_activity",
                                    ],
                                    title=(
                                        "ΔG_leach vs pH — selected "
                                        "structures/H states"
                                    ),
                                    labels={
                                        "deltaG_leach_eV": "ΔG_leach (eV)",
                                        "series_label": "Structure / H state",
                                        "Potential slice": "Potential",
                                    },
                                )
                                if (
                                    line_ph["applied_potential_V"].nunique() > 1
                                ):
                                    ph_kwargs["line_dash"] = "Potential slice"
                                fig_ph = px.line(**ph_kwargs)
                                fig_ph.add_hline(y=0.0, line_dash="dash")
                                _show_publication_plot(fig_ph, height=520)

                    st.markdown("#### U–pH landscape")
                    st.caption(
                        "Choose which of the structures/sites selected above should be shown "
                        "in the heatmap, preferred-H map, 3D surface, diagnostics, and grid export."
                    )
                    focus_col1, focus_col2 = st.columns(2)

                    focus_site_key = "leaching_explorer_focus_site"
                    current_focus_site = st.session_state.get(focus_site_key)
                    if current_focus_site not in comparison_labels:
                        st.session_state[focus_site_key] = comparison_labels[0]

                    focus_label = focus_col1.selectbox(
                        "Structure/site for U–pH and 3D",
                        comparison_labels,
                        disabled=len(comparison_labels) == 1,
                        help=(
                            "Only structures/sites selected in the 2D comparison are offered. "
                            "With one selected structure this choice is automatic."
                        ),
                        key=focus_site_key,
                    )
                    selected_row = explorer_rows.loc[site_labels[focus_label]]
                    selected_surface = str(selected_row["surface_id"])
                    selected_dopant = str(selected_row["dopant"])
                    selected_site_index = int(selected_row["site_index"])
                    z_value = int(selected_row["oxidation_state"])
                    e0_value = float(
                        selected_row["standard_reduction_potential_V_SHE"]
                    )
                    states = _states_for_explorer_row(selected_row)
                    available_h = sorted({int(state["h_count"]) for state in states})
                    focus_h_options = ["Best"] + [f"{h}H" for h in available_h]

                    focus_h_key = "leaching_explorer_focus_h_state"
                    current_focus_h = st.session_state.get(focus_h_key)
                    if current_focus_h not in focus_h_options:
                        st.session_state[focus_h_key] = "Best"

                    focus_h_mode = focus_col2.selectbox(
                        "H state for U–pH and 3D",
                        focus_h_options,
                        help=(
                            "Best minimizes ΔG over all calculated H states at every U/pH point. "
                            "Choose 0H/1H/2H/... to inspect a fixed protonation level."
                        ),
                        key=focus_h_key,
                    )
                    selected_h_count = _h_mode_to_count(focus_h_mode)
                    explorer_activity = float(
                        activity_by_dopant.get(
                            selected_dopant, explorer_default_activity
                        )
                    )

                    full_grid = build_thermodynamic_grid(
                        states,
                        z_value,
                        e0_value,
                        u_values,
                        ph_values,
                        potential_scale=explorer_scale,
                        ion_activity=explorer_activity,
                        temperature_K=explorer_temperature,
                        selected_h_count=selected_h_count,
                    )

                    # Build a common Delta-G range over the selected comparison
                    # structures using the same H-state definition as the focus map.
                    # This lets separately exported heatmaps/3D plots remain directly
                    # comparable on the same numerical color/z scale.
                    shared_dg_values: list[np.ndarray] = []
                    shared_scale_used: list[str] = []
                    shared_scale_skipped: list[str] = []
                    for scale_label in comparison_labels:
                        scale_row = explorer_rows.loc[site_labels[scale_label]]
                        scale_states = _states_for_explorer_row(scale_row)
                        scale_dopant = str(scale_row["dopant"])
                        scale_site_index = int(scale_row["site_index"])
                        scale_available_h = {
                            int(state["h_count"]) for state in scale_states
                        }
                        if (
                            selected_h_count is not None
                            and selected_h_count not in scale_available_h
                        ):
                            shared_scale_skipped.append(
                                f"{scale_dopant} site {scale_site_index}"
                            )
                            continue

                        scale_grid = build_thermodynamic_grid(
                            scale_states,
                            int(scale_row["oxidation_state"]),
                            float(
                                scale_row[
                                    "standard_reduction_potential_V_SHE"
                                ]
                            ),
                            u_values,
                            ph_values,
                            potential_scale=explorer_scale,
                            ion_activity=float(
                                activity_by_dopant.get(
                                    scale_dopant,
                                    explorer_default_activity,
                                )
                            ),
                            temperature_K=explorer_temperature,
                            selected_h_count=selected_h_count,
                        )
                        if scale_grid.empty:
                            shared_scale_skipped.append(
                                f"{scale_dopant} site {scale_site_index}"
                            )
                            continue

                        finite_values = pd.to_numeric(
                            scale_grid["deltaG_leach_eV"], errors="coerce"
                        ).to_numpy(dtype=float)
                        finite_values = finite_values[np.isfinite(finite_values)]
                        if finite_values.size:
                            shared_dg_values.append(finite_values)
                            shared_scale_used.append(
                                f"{scale_dopant} site {scale_site_index}"
                            )

                    current_dg_values = pd.to_numeric(
                        full_grid["deltaG_leach_eV"], errors="coerce"
                    ).to_numpy(dtype=float)
                    current_dg_values = current_dg_values[
                        np.isfinite(current_dg_values)
                    ]
                    if current_dg_values.size:
                        current_dg_min = float(np.min(current_dg_values))
                        current_dg_max = float(np.max(current_dg_values))
                    else:
                        current_dg_min, current_dg_max = -1.0, 1.0

                    if shared_dg_values:
                        all_shared_dg = np.concatenate(shared_dg_values)
                        shared_dg_min = float(np.min(all_shared_dg))
                        shared_dg_max = float(np.max(all_shared_dg))
                    else:
                        shared_dg_min = current_dg_min
                        shared_dg_max = current_dg_max

                    def _expand_equal_dg_limits(
                        lower: float, upper: float
                    ) -> tuple[float, float]:
                        if upper > lower:
                            return lower, upper
                        pad = max(0.5, abs(lower) * 0.05)
                        return lower - pad, upper + pad

                    shared_dg_min, shared_dg_max = _expand_equal_dg_limits(
                        shared_dg_min, shared_dg_max
                    )
                    current_dg_min, current_dg_max = _expand_equal_dg_limits(
                        current_dg_min, current_dg_max
                    )

                    scale_mode_labels = {
                        "shared_selected": "Shared across selected structures",
                        "current_auto": "Auto for current structure",
                        "manual": "Manual fixed range",
                    }
                    saved_scale_mode = str(
                        analysis_cfg.get(
                            "deltaG_color_scale_mode", "shared_selected"
                        )
                    )
                    if saved_scale_mode not in scale_mode_labels:
                        saved_scale_mode = "shared_selected"

                    st.markdown("##### ΔG scale for heatmap and 3D")
                    scale_col1, scale_col2, scale_col3 = st.columns(3)
                    scale_mode = scale_col1.selectbox(
                        "ΔG color/z scale",
                        list(scale_mode_labels.keys()),
                        index=list(scale_mode_labels.keys()).index(
                            saved_scale_mode
                        ),
                        format_func=lambda value: scale_mode_labels[value],
                        key="leaching_explorer_dg_scale_mode",
                        help=(
                            "Shared uses identical ΔG limits for every currently "
                            "selected structure/dopant, which is recommended when "
                            "exporting separate panels for side-by-side comparison."
                        ),
                    )

                    if scale_mode == "shared_selected":
                        dg_scale_min = shared_dg_min
                        dg_scale_max = shared_dg_max
                        scale_col2.metric(
                            "Shared ΔG min", f"{dg_scale_min:.3f} eV"
                        )
                        scale_col3.metric(
                            "Shared ΔG max", f"{dg_scale_max:.3f} eV"
                        )
                        st.caption(
                            f"Common scale calculated from "
                            f"{len(shared_scale_used)}/{len(comparison_labels)} "
                            f"selected structures using **{focus_h_mode}**: "
                            f"{dg_scale_min:.3f} to {dg_scale_max:.3f} eV."
                        )
                        if shared_scale_skipped:
                            st.caption(
                                "Not included in the shared scale because the "
                                f"requested H state is unavailable: "
                                + ", ".join(shared_scale_skipped)
                            )
                    elif scale_mode == "current_auto":
                        dg_scale_min = current_dg_min
                        dg_scale_max = current_dg_max
                        scale_col2.metric(
                            "Current ΔG min", f"{dg_scale_min:.3f} eV"
                        )
                        scale_col3.metric(
                            "Current ΔG max", f"{dg_scale_max:.3f} eV"
                        )
                    else:
                        manual_default_min = float(
                            analysis_cfg.get(
                                "deltaG_color_min_eV", shared_dg_min
                            )
                        )
                        manual_default_max = float(
                            analysis_cfg.get(
                                "deltaG_color_max_eV", shared_dg_max
                            )
                        )
                        dg_scale_min = float(
                            scale_col2.number_input(
                                "ΔG min (eV)",
                                value=manual_default_min,
                                step=0.25,
                                key="leaching_explorer_dg_manual_min",
                            )
                        )
                        dg_scale_max = float(
                            scale_col3.number_input(
                                "ΔG max (eV)",
                                value=manual_default_max,
                                step=0.25,
                                key="leaching_explorer_dg_manual_max",
                            )
                        )
                        if dg_scale_max <= dg_scale_min:
                            st.error(
                                "Manual ΔG maximum must be larger than the minimum."
                            )
                            dg_scale_min = shared_dg_min
                            dg_scale_max = shared_dg_max
                        elif (
                            current_dg_min < dg_scale_min
                            or current_dg_max > dg_scale_max
                        ):
                            st.caption(
                                "The manual range does not contain all current "
                                "ΔG values; values outside it will be clipped."
                            )

                    pivot_dg = full_grid.pivot(
                        index="pH",
                        columns="applied_potential_V",
                        values="deltaG_leach_eV",
                    ).sort_index()
                    pivot_h = full_grid.pivot(
                        index="pH",
                        columns="applied_potential_V",
                        values="best_h_count",
                    ).sort_index()

                    map_left, map_right = st.columns(2)
                    with map_left:
                        heat = go.Figure()
                        heat.add_trace(
                            go.Heatmap(
                                x=pivot_dg.columns.to_numpy(),
                                y=pivot_dg.index.to_numpy(),
                                z=pivot_dg.to_numpy(),
                                zmin=dg_scale_min,
                                zmax=dg_scale_max,
                                zsmooth="best",
                                colorbar=dict(title="ΔG (eV)"),
                                hovertemplate=(
                                    "U=%{x:.3f} V<br>pH=%{y:.2f}<br>ΔG=%{z:.3f} eV<extra></extra>"
                                ),
                            )
                        )
                        heat.add_trace(
                            go.Contour(
                                x=pivot_dg.columns.to_numpy(),
                                y=pivot_dg.index.to_numpy(),
                                z=pivot_dg.to_numpy(),
                                contours=dict(
                                    start=0.0,
                                    end=0.0,
                                    size=1.0,
                                    coloring="lines",
                                ),
                                line=dict(width=4),
                                showscale=False,
                                hoverinfo="skip",
                            )
                        )
                        heat.update_layout(
                            title=(
                                f"Smooth ΔG_leach(U, pH) — {selected_dopant} site "
                                f"{selected_site_index} | {focus_h_mode}"
                            ),
                            xaxis_title=f"Potential (V vs {explorer_scale})",
                            yaxis_title="pH",
                        )
                        _show_publication_plot(heat, height=520)
                    with map_right:
                        h_values = sorted(
                            {
                                int(x)
                                for x in pd.to_numeric(
                                    full_grid["best_h_count"], errors="coerce"
                                ).dropna()
                            }
                        )
                        if h_values:
                            h_min = min(h_values)
                            h_max = max(h_values)
                            if h_min == h_max:
                                h_zmin = h_min - 0.5
                                h_zmax = h_max + 0.5
                                h_colorscale = [[0.0, "#636EFA"], [1.0, "#636EFA"]]
                            else:
                                h_zmin = h_min - 0.5
                                h_zmax = h_max + 0.5
                                palette = px.colors.qualitative.Plotly
                                h_colorscale = []
                                span = h_zmax - h_zmin
                                for color_index, h_value in enumerate(h_values):
                                    left_edge = (h_value - 0.5 - h_zmin) / span
                                    right_edge = (h_value + 0.5 - h_zmin) / span
                                    color = palette[color_index % len(palette)]
                                    h_colorscale.extend(
                                        [
                                            [max(0.0, left_edge), color],
                                            [min(1.0, right_edge), color],
                                        ]
                                    )
                            hmap = go.Figure(
                                data=go.Heatmap(
                                    x=pivot_h.columns.to_numpy(),
                                    y=pivot_h.index.to_numpy(),
                                    z=pivot_h.to_numpy(),
                                    zmin=h_zmin,
                                    zmax=h_zmax,
                                    colorscale=h_colorscale,
                                    colorbar=dict(
                                        title="best nH",
                                        tickmode="array",
                                        tickvals=h_values,
                                        ticktext=[f"{h}H" for h in h_values],
                                    ),
                                    hovertemplate=(
                                        "U=%{x:.3f} V<br>pH=%{y:.2f}<br>best nH=%{z:.0f}<extra></extra>"
                                    ),
                                )
                            )
                            hmap.update_layout(
                                title=(
                                    f"Preferred post-leaching H count — {selected_dopant} "
                                    f"site {selected_site_index}"
                                ),
                                xaxis_title=f"Potential (V vs {explorer_scale})",
                                yaxis_title="pH",
                            )
                            _show_publication_plot(hmap, height=520)

                    surface_fig = go.Figure(
                        data=[
                            go.Surface(
                                x=pivot_dg.columns.to_numpy(),
                                y=pivot_dg.index.to_numpy(),
                                z=pivot_dg.to_numpy(),
                                cmin=dg_scale_min,
                                cmax=dg_scale_max,
                                colorbar=dict(title="ΔG (eV)"),
                                hovertemplate=(
                                    "U=%{x:.3f} V<br>pH=%{y:.2f}<br>ΔG=%{z:.3f} eV<extra></extra>"
                                ),
                            )
                        ]
                    )
                    surface_fig.update_layout(
                        title=(
                            f"3D thermodynamic surface — {selected_dopant} site "
                            f"{selected_site_index} | {focus_h_mode}"
                        ),
                        scene=dict(
                            xaxis_title=f"Potential (V vs {explorer_scale})",
                            yaxis_title="pH",
                            zaxis=dict(
                                title="ΔG_leach (eV)",
                                range=[dg_scale_min, dg_scale_max],
                            ),
                        ),
                    )
                    _show_publication_plot(
                        surface_fig,
                        height=620,
                        is_3d=True,
                    )

                    st.markdown("#### State diagnostics")
                    d1, d2 = st.columns(2)
                    diagnostic_u = float(
                        d1.number_input(
                            f"Diagnostic potential (V vs {explorer_scale})",
                            value=float(selected_u_slices[0]),
                            step=0.05,
                            key="leaching_explorer_diag_u",
                        )
                    )
                    diagnostic_ph = float(
                        d2.number_input(
                            "Diagnostic pH",
                            value=float(selected_ph_slices[0]),
                            step=0.25,
                            key="leaching_explorer_diag_ph",
                        )
                    )
                    diagnostic = build_thermodynamic_grid(
                        states,
                        z_value,
                        e0_value,
                        [diagnostic_u],
                        [diagnostic_ph],
                        potential_scale=explorer_scale,
                        ion_activity=explorer_activity,
                        temperature_K=explorer_temperature,
                        selected_h_count=selected_h_count,
                    )
                    if not diagnostic.empty:
                        point = diagnostic.iloc[0]
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Oxidation-state electrons z", f"{z_value}")
                        m2.metric("Preferred H count", f"{int(point['best_h_count'])}")
                        m3.metric("Net electrons z − nH", f"{int(point['net_electrons'])}")
                        m4.metric("ΔG_leach", f"{float(point['deltaG_leach_eV']):.3f} eV")
                        if int(point["net_electrons"]) == 0:
                            st.info(
                                "At this selected state, z = nH, so the implemented model has "
                                "zero ΔG slope with applied potential at fixed pH. This is the "
                                "same cancellation that produced the flat In curves."
                            )

                    export_grid = full_grid.copy()
                    export_grid.insert(0, "dopant", selected_dopant)
                    export_grid.insert(1, "site_index", selected_site_index)
                    export_grid.insert(2, "surface_id", selected_surface)
                    export_grid["oxidation_state"] = z_value
                    export_grid["standard_reduction_potential_V_SHE"] = e0_value
                    export_grid["ion_activity"] = explorer_activity
                    export_grid["temperature_K"] = explorer_temperature
                    export_grid["requested_h_state"] = focus_h_mode
                    export_grid["deltaG_scale_mode"] = scale_mode
                    export_grid["deltaG_scale_min_eV"] = dg_scale_min
                    export_grid["deltaG_scale_max_eV"] = dg_scale_max
                    dl_col, save_col = st.columns(2)
                    dl_col.download_button(
                        "Export current U–pH grid CSV",
                        data=export_grid.to_csv(index=False).encode("utf-8"),
                        file_name=f"leaching_thermodynamic_grid_{selected_dopant}_site_{selected_site_index}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    if save_col.button(
                        "Save current analysis defaults",
                        use_container_width=True,
                        help="Saves only visualization/post-processing defaults; no MLFF calculation is launched.",
                    ):
                        updated_activity_map = dict(activity_map_saved)
                        updated_activity_map.update(activity_by_dopant)
                        new_defaults = {
                            "temperature_K": explorer_temperature,
                            "default_ion_activity": explorer_default_activity,
                            "ion_activities": updated_activity_map,
                            "potential_scale": explorer_scale,
                            "potential_min_V": u_min,
                            "potential_max_V": u_max,
                            "potential_step_V": u_step,
                            "pH_min": ph_min,
                            "pH_max": ph_max,
                            "pH_step": ph_step,
                            "selected_pH_values": selected_ph_slices,
                            "selected_potential_values": selected_u_slices,
                            "deltaG_color_scale_mode": scale_mode,
                            "deltaG_color_min_eV": dg_scale_min,
                            "deltaG_color_max_eV": dg_scale_max,
                        }
                        save_cfg = toml.load(str(config_path))
                        save_cfg.setdefault("leaching", {})["analysis_defaults"] = new_defaults
                        # Keep legacy aliases synchronized for older scripts/outputs.
                        save_cfg["leaching"]["temperature_K"] = explorer_temperature
                        save_cfg["leaching"]["default_ion_activity"] = new_defaults[
                            "default_ion_activity"
                        ]
                        save_cfg["leaching"]["ion_activities"] = updated_activity_map
                        save_cfg["leaching"]["potential_scale"] = explorer_scale
                        save_cfg["leaching"]["pH"] = float(selected_ph_slices[0])
                        save_cfg["leaching"]["potentials_V"] = selected_u_slices
                        config_path.write_text(toml.dumps(save_cfg), encoding="utf-8")
                        st.success(
                            "Saved analysis defaults. No structural or protonation calculation was rerun."
                        )

    with tabs[5]:
        st.markdown("### Site environment / local structure")
        st.caption(
            "This analysis uses the **relaxed parent surface before dopant removal**. "
            "It is geometry-only post-processing: changing these cutoffs does not rerun "
            "GRACE/MACE or any leaching/protonation calculation."
        )

        env_control1, env_control2 = st.columns(2)
        default_coordination_cutoff = float(
            parsed_for_results.get("oxygen_neighbor_cutoff_A", 2.8)
            if "parsed_for_results" in locals()
            else 2.8
        )
        coordination_cutoff = float(
            env_control1.number_input(
                "O coordination cutoff (Å)",
                min_value=0.5,
                value=default_coordination_cutoff,
                step=0.05,
                format="%.2f",
                key="leaching_environment_coordination_cutoff",
                help=(
                    "O atoms within this distance define the first anion coordination "
                    "shell and its bond-length/angle descriptors."
                ),
            )
        )
        neighbor_cutoff = float(
            env_control2.number_input(
                "Local environment radius (Å)",
                min_value=coordination_cutoff,
                value=max(4.0, coordination_cutoff),
                step=0.1,
                format="%.2f",
                key="leaching_environment_neighbor_cutoff",
                help=(
                    "All atoms within this radius are included in the local-neighbor "
                    "description and can be shown in the 3D viewer."
                ),
            )
        )

        environment_rows: list[dict[str, Any]] = []
        environment_neighbors: dict[tuple[str, str, int], pd.DataFrame] = {}
        environment_paths: dict[tuple[str, str, int], Path] = {}
        environment_errors: list[str] = []
        structure_cache: dict[str, Structure] = {}

        anion_species = list(
            parsed_for_results.get("anion_species", ["O"])
            if "parsed_for_results" in locals()
            else ["O"]
        )
        dopant_species_all = sorted(
            {
                str(value)
                for value in results.get("dopant", pd.Series(dtype=str))
                .dropna()
                .astype(str)
            }
        )

        if "surface_structure_path" not in results.columns:
            st.warning(
                "The current leaching summary does not contain surface_structure_path, "
                "so the original local structures cannot be reconstructed from this result set."
            )
        else:
            for _, env_row in results.iterrows():
                try:
                    surface_id = str(env_row.get("surface_id", ""))
                    dopant = str(env_row.get("dopant", ""))
                    site_index = int(env_row.get("site_index"))
                    raw_path = str(env_row.get("surface_structure_path", "")).strip()
                    if not raw_path or raw_path.lower() == "nan":
                        raise FileNotFoundError("surface_structure_path is missing")
                    structure_path = _result_structure_path(raw_path, project_root)
                    if not structure_path.exists():
                        raise FileNotFoundError(str(structure_path))

                    cache_key = str(structure_path)
                    if cache_key not in structure_cache:
                        structure_cache[cache_key] = Structure.from_file(structure_path)
                    structure = structure_cache[cache_key]

                    env_summary, neighbor_table = analyze_site_environment(
                        structure,
                        site_index,
                        anion_species=anion_species,
                        dopant_species=dopant_species_all,
                        neighbor_cutoff_A=neighbor_cutoff,
                        coordination_cutoff_A=coordination_cutoff,
                    )
                    key = (surface_id, dopant, site_index)
                    environment_neighbors[key] = neighbor_table
                    environment_paths[key] = structure_path

                    record = {
                        "surface_id": surface_id,
                        "target_id": str(env_row.get("target_id", "")),
                        "dopant": dopant,
                        "site_index": site_index,
                        "initial_dopant_zone": str(
                            env_row.get("initial_dopant_zone", "")
                        ),
                        "miller_h": env_row.get("miller_h"),
                        "miller_k": env_row.get("miller_k"),
                        "miller_l": env_row.get("miller_l"),
                        "variant_label": str(env_row.get("variant_label", "")),
                        "initial_depth_from_selected_surface_A": env_row.get(
                            "initial_depth_from_selected_surface_A"
                        ),
                        "n_oxygen_vacancies": env_row.get("n_oxygen_vacancies"),
                        "extraction_energy_eV": env_row.get("extraction_energy_eV"),
                        "dissolution_potential_V_SHE": env_row.get(
                            "dissolution_potential_V_SHE"
                        ),
                        "dissolution_potential_V_RHE": env_row.get(
                            "dissolution_potential_V_RHE"
                        ),
                        "protonation_adjusted_dissolution_potential_V_SHE": env_row.get(
                            "protonation_adjusted_dissolution_potential_V_SHE"
                        ),
                        "protonation_adjusted_dissolution_potential_V_RHE": env_row.get(
                            "protonation_adjusted_dissolution_potential_V_RHE"
                        ),
                        **env_summary,
                    }
                    environment_rows.append(record)
                except Exception as exc:
                    environment_errors.append(
                        f"{env_row.get('dopant', '?')} site "
                        f"{env_row.get('site_index', '?')}: {exc}"
                    )

            environment_df = pd.DataFrame(environment_rows)

            if environment_errors:
                with st.expander(
                    f"Local environments that could not be reconstructed "
                    f"({len(environment_errors)})",
                    expanded=False,
                ):
                    st.write(
                        "\n".join(f"- {message}" for message in environment_errors)
                    )

            if environment_df.empty:
                st.info("No site environment could be reconstructed.")
            else:
                st.markdown("#### Compare environments across leaching sites")
                table_columns = [
                    column
                    for column in (
                        "dopant",
                        "site_index",
                        "initial_dopant_zone",
                        "miller_h",
                        "miller_k",
                        "miller_l",
                        "anion_coordination_number",
                        "mean_coordination_anion_distance_A",
                        "coordination_anion_distance_std_A",
                        "coordination_anion_distance_range_A",
                        "mean_coordination_angle_deg",
                        "coordination_angle_std_deg",
                        "neighbors_within_cutoff",
                        "cation_neighbors_within_cutoff",
                        "dopant_neighbors_within_cutoff",
                        "nearest_anion_distance_A",
                        "nearest_cation_distance_A",
                        "nearest_dopant_distance_A",
                        "local_environment_signature",
                        "initial_depth_from_selected_surface_A",
                        "n_oxygen_vacancies",
                        "extraction_energy_eV",
                        "dissolution_potential_V_SHE",
                        "dissolution_potential_V_RHE",
                        "protonation_adjusted_dissolution_potential_V_SHE",
                        "protonation_adjusted_dissolution_potential_V_RHE",
                        "target_id",
                    )
                    if column in environment_df.columns
                ]
                st.dataframe(
                    environment_df[table_columns],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Export site-environment comparison CSV",
                    data=environment_df.to_csv(index=False).encode("utf-8"),
                    file_name="leaching_site_environment.csv",
                    mime="text/csv",
                    key="leaching_environment_export",
                )

                numeric_descriptor_choices = [
                    column
                    for column in (
                        "anion_coordination_number",
                        "mean_coordination_anion_distance_A",
                        "coordination_anion_distance_std_A",
                        "coordination_anion_distance_range_A",
                        "mean_coordination_angle_deg",
                        "coordination_angle_std_deg",
                        "neighbors_within_cutoff",
                        "cation_neighbors_within_cutoff",
                        "dopant_neighbors_within_cutoff",
                        "nearest_anion_distance_A",
                        "nearest_cation_distance_A",
                        "nearest_dopant_distance_A",
                        "initial_depth_from_selected_surface_A",
                    )
                    if column in environment_df.columns
                ]
                response_choices = [
                    column
                    for column in (
                        "extraction_energy_eV",
                        "dissolution_potential_V_SHE",
                        "dissolution_potential_V_RHE",
                        "protonation_adjusted_dissolution_potential_V_SHE",
                        "protonation_adjusted_dissolution_potential_V_RHE",
                    )
                    if column in environment_df.columns
                    and pd.to_numeric(
                        environment_df[column], errors="coerce"
                    ).notna().any()
                ]
                if numeric_descriptor_choices and response_choices:
                    rel1, rel2 = st.columns(2)
                    descriptor_x = rel1.selectbox(
                        "Structural descriptor",
                        numeric_descriptor_choices,
                        key="leaching_environment_descriptor_x",
                    )
                    response_y = rel2.selectbox(
                        "Leaching metric",
                        response_choices,
                        key="leaching_environment_response_y",
                    )
                    relationship = environment_df.copy()
                    relationship[descriptor_x] = pd.to_numeric(
                        relationship[descriptor_x], errors="coerce"
                    )
                    relationship[response_y] = pd.to_numeric(
                        relationship[response_y], errors="coerce"
                    )
                    relationship = relationship.dropna(
                        subset=[descriptor_x, response_y]
                    )
                    if not relationship.empty:
                        fig_env = px.scatter(
                            relationship,
                            x=descriptor_x,
                            y=response_y,
                            color="dopant",
                            symbol=(
                                "initial_dopant_zone"
                                if "initial_dopant_zone"
                                in relationship.columns
                                else None
                            ),
                            hover_data=[
                                column
                                for column in (
                                    "site_index",
                                    "target_id",
                                    "local_environment_signature",
                                    "anion_coordination_number",
                                    "mean_coordination_anion_distance_A",
                                    "nearest_dopant_distance_A",
                                )
                                if column in relationship.columns
                            ],
                            title=(
                                f"{response_y} vs local structural environment"
                            ),
                        )
                        _show_publication_plot(fig_env)

                st.markdown("#### Inspect one site")
                site_option_map: dict[str, tuple[str, str, int]] = {}
                for _, item in environment_df.iterrows():
                    hkl = ""
                    try:
                        hkl = (
                            f"({int(item['miller_h'])}{int(item['miller_k'])}"
                            f"{int(item['miller_l'])})"
                        )
                    except (KeyError, TypeError, ValueError):
                        pass
                    label = (
                        f"{item['dopant']} site {int(item['site_index'])} | "
                        f"{item.get('initial_dopant_zone', '')} | {hkl} | "
                        f"{item.get('target_id', '')}"
                    )
                    site_option_map[label] = (
                        str(item["surface_id"]),
                        str(item["dopant"]),
                        int(item["site_index"]),
                    )

                selected_environment_label = st.selectbox(
                    "Site to inspect",
                    list(site_option_map.keys()),
                    key="leaching_environment_site_selector",
                )
                selected_environment_key = site_option_map[
                    selected_environment_label
                ]
                selected_environment_row = environment_df[
                    (environment_df["surface_id"].astype(str)
                     == selected_environment_key[0])
                    & (environment_df["dopant"].astype(str)
                       == selected_environment_key[1])
                    & (pd.to_numeric(
                        environment_df["site_index"], errors="coerce"
                    ) == selected_environment_key[2])
                ].iloc[0]
                selected_neighbors = environment_neighbors[
                    selected_environment_key
                ]
                selected_structure_path = environment_paths[
                    selected_environment_key
                ]

                selected_structure = structure_cache.get(
                    str(selected_structure_path)
                )
                if selected_structure is None:
                    selected_structure = Structure.from_file(
                        selected_structure_path
                    )
                    structure_cache[str(selected_structure_path)] = (
                        selected_structure
                    )

                elements_present = sorted(
                    {site.specie.symbol for site in selected_structure}
                )
                default_element_colors = {
                    "O": "#E41A1C",
                    "Sn": "#377EB8",
                    "Sb": "#984EA3",
                    "In": "#4DAF4A",
                    "H": "#F2F2F2",
                    "Ti": "#FF7F00",
                    "Zr": "#A6CEE3",
                    "Nb": "#A65628",
                    "Ba": "#FFD92F",
                    "Mn": "#F781BF",
                    "Ni": "#1B9E77",
                    "Fe": "#E6550D",
                    "Zn": "#66A61E",
                    "W": "#7570B3",
                }
                fallback_colors = [
                    "#4E79A7",
                    "#F28E2B",
                    "#59A14F",
                    "#B07AA1",
                    "#76B7B2",
                    "#EDC948",
                    "#9C755F",
                    "#BAB0AC",
                ]
                element_defaults = {
                    element: default_element_colors.get(
                        element,
                        fallback_colors[
                            elements_present.index(element)
                            % len(fallback_colors)
                        ],
                    )
                    for element in elements_present
                }

                with st.expander(
                    "Atom colors & interaction",
                    expanded=False,
                ):
                    reset_colors = st.button(
                        "Reset atom colors",
                        key="leaching_environment_reset_colors",
                    )
                    if reset_colors:
                        for element, default_color in element_defaults.items():
                            st.session_state[
                                f"leaching_environment_color_{element}"
                            ] = default_color

                    color_columns = st.columns(
                        min(4, max(1, len(elements_present)))
                    )
                    element_colors: dict[str, str] = {}
                    for element_index, element in enumerate(elements_present):
                        element_colors[element] = color_columns[
                            element_index % len(color_columns)
                        ].color_picker(
                            f"{element} atoms",
                            value=element_defaults[element],
                            key=f"leaching_environment_color_{element}",
                        )

                    enable_atom_hover = st.checkbox(
                        "Show element and site index on hover",
                        value=True,
                        key="leaching_environment_enable_hover",
                        help=(
                            "Hovering over a base atom shows its element and the "
                            "zero-based atom/site index used by DopingFlow."
                        ),
                    )
                    show_orientation = st.checkbox(
                        "Show coordinate axes and surface-normal direction",
                        value=True,
                        key="leaching_environment_show_orientation",
                        help=(
                            "Shows x/y/z arrows and marks the exposed surface/vacuum "
                            "direction along the slab-normal z axis."
                        ),
                    )

                detail_left, detail_right = st.columns([1.35, 1.0])
                with detail_left:
                    show_all_neighbors = st.checkbox(
                        "Show all neighbors within local radius",
                        value=True,
                        key="leaching_environment_show_all_neighbors",
                    )
                    label_neighbors = st.checkbox(
                        "Label neighbor atoms and distances",
                        value=False,
                        key="leaching_environment_label_neighbors",
                    )
                    displayed_neighbors = selected_neighbors
                    if not show_all_neighbors and not selected_neighbors.empty:
                        displayed_neighbors = selected_neighbors[
                            selected_neighbors["within_coordination_cutoff"]
                            | selected_neighbors["neighbor_class"].eq("dopant")
                        ].copy()
                    st.caption(
                        "Base atom colors follow your element palette. The selected "
                        "leaching site keeps the **same color as its element** and is "
                        "only slightly enlarged; its label identifies it explicitly. "
                        "Local-shell markers remain **gold** = coordinating anion, "
                        "**blue** = nearby dopant, **green** = other local atom."
                    )
                    if enable_atom_hover:
                        st.caption(
                            "Move the cursor over any base atom to show "
                            "**element + DopingFlow site index**."
                        )
                    if show_orientation:
                        placement_side = str(
                            parsed_for_results.get("placement_side", "top")
                            if "parsed_for_results" in locals()
                            else "top"
                        )
                        direction_text = {
                            "top": "+z",
                            "bottom": "−z",
                            "both": "±z",
                        }.get(placement_side, "z")
                        st.caption(
                            f"Surface-normal direction for this workflow: **{direction_text}**. "
                            "The orange arrow marks the exposed surface/vacuum side."
                        )
                    try:
                        show_site_environment(
                            selected_structure_path,
                            selected_environment_key[2],
                            displayed_neighbors,
                            title=selected_environment_label,
                            label_neighbors=label_neighbors,
                            element_colors=element_colors,
                            enable_hover=enable_atom_hover,
                            show_orientation=show_orientation,
                            surface_side=str(
                                parsed_for_results.get("placement_side", "top")
                                if "parsed_for_results" in locals()
                                else "top"
                            ),
                        )
                    except Exception as exc:
                        st.error(f"Could not render the 3D site environment: {exc}")

                with detail_right:
                    metric1, metric2 = st.columns(2)
                    metric1.metric(
                        "O coordination",
                        str(
                            int(
                                selected_environment_row[
                                    "anion_coordination_number"
                                ]
                            )
                        ),
                    )
                    mean_bond = selected_environment_row.get(
                        "mean_coordination_anion_distance_A"
                    )
                    metric2.metric(
                        "Mean M–O",
                        (
                            f"{float(mean_bond):.3f} Å"
                            if pd.notna(mean_bond)
                            else "n/a"
                        ),
                    )
                    metric3, metric4 = st.columns(2)
                    bond_std = selected_environment_row.get(
                        "coordination_anion_distance_std_A"
                    )
                    metric3.metric(
                        "M–O bond std.",
                        (
                            f"{float(bond_std):.3f} Å"
                            if pd.notna(bond_std)
                            else "n/a"
                        ),
                    )
                    nearest_dopant = selected_environment_row.get(
                        "nearest_dopant_distance_A"
                    )
                    metric4.metric(
                        "Nearest dopant",
                        (
                            f"{float(nearest_dopant):.3f} Å"
                            if pd.notna(nearest_dopant)
                            else "none in shell"
                        ),
                    )
                    st.markdown(
                        f"**Local shell:** "
                        f"{selected_environment_row.get('local_environment_signature', '')}"
                    )
                    st.markdown(
                        f"**Zone:** "
                        f"{selected_environment_row.get('initial_dopant_zone', '')}  "
                        f"\n**Depth from selected surface:** "
                        f"{float(selected_environment_row.get('initial_depth_from_selected_surface_A', 0.0)):.3f} Å"
                    )
                    extraction_value = pd.to_numeric(
                        pd.Series(
                            [selected_environment_row.get("extraction_energy_eV")]
                        ),
                        errors="coerce",
                    ).iloc[0]
                    if pd.notna(extraction_value):
                        st.markdown(
                            f"**Extraction energy:** {float(extraction_value):.3f} eV"
                        )
                    st.markdown(
                        "**Interpretive descriptors:** coordination number describes "
                        "under-/over-coordination; M–O bond spread and O–M–O angle spread "
                        "describe local distortion; nearest-dopant distance and local-shell "
                        "composition expose dopant–dopant/environment effects."
                    )

                st.markdown("##### Neighbor list for selected site")
                neighbor_columns = [
                    column
                    for column in (
                        "neighbor_index",
                        "species",
                        "neighbor_class",
                        "distance_A",
                        "within_coordination_cutoff",
                        "periodic_image_a",
                        "periodic_image_b",
                        "periodic_image_c",
                        "dx_A",
                        "dy_A",
                        "dz_A",
                    )
                    if column in selected_neighbors.columns
                ]
                st.dataframe(
                    selected_neighbors[neighbor_columns],
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Export selected-site neighbor list CSV",
                    data=selected_neighbors.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        f"site_environment_{selected_environment_key[1]}_"
                        f"{selected_environment_key[2]}.csv"
                    ),
                    mime="text/csv",
                    key="leaching_environment_neighbor_export",
                )

    with tabs[6]:
        st.markdown(
            "Every row keeps the dopant's **initial relaxed-surface zone and coordinates** before removal. "
            "**Lower extraction energy** means weaker retention relative to the elemental-metal "
            "reference. With a valid aqueous redox reference, **lower dissolution potential** "
            "means dissolution becomes thermodynamically favorable at a lower electrode potential."
        )
        st.warning(
            "The electrochemical extension still uses the user-supplied simple M^z+/M redox "
            "reference. When post-leaching protonation is enabled, local O-H termination is "
            "included through an H2/CHE correction, but explicit solvent, charged or "
            "constant-potential slabs, full aqueous speciation, kinetic barriers, and "
            "multi-atom dissolution pathways are not included."
        )

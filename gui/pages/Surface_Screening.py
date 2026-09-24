"""Configure, run, and inspect staged surface screening and refinement."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import toml

from dopingflow.surface_staged import (
    DEFAULT_MILLERS,
    parse_surface_config,
    preview_surface_candidates,
)
from gui_config import (
    BACKEND_CHOICES,
    DEVICE_CHOICES,
    GRACE_MODEL_CHOICES,
    OPTIMIZER_CHOICES,
    UMA_MODEL_CHOICES,
    UMA_TASK_CHOICES,
)
from view_structure import show_structure


st.set_page_config(page_title="Surface screening", layout="wide")
st.title("Surface screening")
st.caption(
    "Generate low-index slabs from selected stable bulk structures, enumerate terminations, "
    "scan representative co-dopant depth placements, rank them with a fast MLFF, and "
    "optionally refine the shortlist with a second higher-fidelity MLFF."
)

project_root = Path(
    st.sidebar.text_input(
        "Project root",
        value=str(Path.cwd()),
        key="surface_project_root",
    )
).expanduser().resolve()
config_path = project_root / "input.toml"

if not config_path.exists():
    st.error(f"No input.toml found at {config_path}")
    st.stop()

cfg = toml.load(str(config_path))
surface = dict(cfg.get("surface", {}) or {})
screen_saved = dict(surface.get("screen", {}) or {})
if not screen_saved:
    # Migrate the former Input Builder's flat surface-relaxation controls into
    # the new staged screen editor without losing the user's existing choices.
    screen_saved = {
        "enabled": True,
        "backend": surface.get("surface_backend", "grace"),
        "model": surface.get("surface_model", "GRACE-1L-OMAT"),
        "task": surface.get("surface_task", ""),
        "device": surface.get("surface_device", "cpu"),
        "gpu_id": surface.get("surface_gpu_id", 0),
        "tf_threads": surface.get("surface_tf_threads", 1),
        "omp_threads": surface.get("surface_omp_threads", 1),
        "relax": surface.get("relax_surface", True),
        "optimizer": surface.get("surface_optimizer", "bfgs"),
        "fmax": surface.get("surface_fmax", 0.05),
        "max_steps": surface.get("surface_max_steps", 300),
        "top_k_per_candidate": 10,
    }
refine_saved = dict(surface.get("refine", {}) or {})
doping = dict(cfg.get("doping", {}) or {})
scan_cfg = dict(cfg.get("scan", {}) or {})


def _csv_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return ""


def _parse_csv(text: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in text.split(",") if item.strip()))


def _miller_text(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        value = DEFAULT_MILLERS
    parts: list[str] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            parts.append(",".join(str(int(v)) for v in item))
    return "; ".join(parts) or "; ".join(",".join(str(v) for v in hkl) for hkl in DEFAULT_MILLERS)


def _parse_millers(text: str) -> list[list[int]]:
    result: list[list[int]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        values = [int(item.strip()) for item in part.split(",")]
        if len(values) != 3 or values == [0, 0, 0]:
            raise ValueError(
                "Each Miller index must contain three integers and cannot be 0,0,0."
            )
        result.append(values)
    if not result:
        raise ValueError("At least one Miller index is required.")
    return result


def _choice_index(options: list[str], value: Any, default: str) -> int:
    selected = str(value).strip()
    if selected not in options:
        selected = default
    return options.index(selected)


def _calculator_editor(
    label: str,
    key_prefix: str,
    saved: dict[str, Any],
    *,
    enabled_default: bool,
    backend_default: str,
    model_default: str,
    task_default: str,
    fmax_default: float,
    steps_default: int,
    topk_default: int,
) -> dict[str, Any]:
    st.markdown(f"#### {label}")
    c1, c2, c3 = st.columns(3)
    with c1:
        enabled = st.checkbox(
            f"Enable {label.lower()}",
            value=bool(saved.get("enabled", enabled_default)),
            key=f"{key_prefix}_enabled",
            disabled=(key_prefix == "surface_screen"),
            help=(
                "The broad surface screen is always required."
                if key_prefix == "surface_screen"
                else "When disabled, the workflow stops after the screen shortlist."
            ),
        )
        if key_prefix == "surface_screen":
            enabled = True
    with c2:
        backend = st.selectbox(
            "Backend",
            BACKEND_CHOICES,
            index=_choice_index(
                BACKEND_CHOICES,
                str(saved.get("backend", backend_default)).lower(),
                backend_default,
            ),
            key=f"{key_prefix}_backend",
        )
    with c3:
        device = st.selectbox(
            "Device",
            DEVICE_CHOICES,
            index=_choice_index(
                DEVICE_CHOICES,
                str(saved.get("device", "cpu")).lower(),
                "cpu",
            ),
            key=f"{key_prefix}_device",
        )

    m1, m2, m3 = st.columns(3)
    if backend == "m3gnet":
        model = "default"
        m1.text_input(
            "Model",
            value=model,
            disabled=True,
            key=f"{key_prefix}_model_m3gnet",
        )
        task = ""
        m2.text_input(
            "Task/head",
            value="not used",
            disabled=True,
            key=f"{key_prefix}_task_m3gnet",
        )
    elif backend == "grace":
        current_model = str(saved.get("model", model_default))
        if current_model not in GRACE_MODEL_CHOICES:
            current_model = (
                model_default
                if model_default in GRACE_MODEL_CHOICES
                else "GRACE-1L-OMAT"
            )
        model = m1.selectbox(
            "Model",
            GRACE_MODEL_CHOICES,
            index=GRACE_MODEL_CHOICES.index(current_model),
            key=f"{key_prefix}_model_grace",
        )
        task = ""
        m2.text_input(
            "Task/head",
            value="not used",
            disabled=True,
            key=f"{key_prefix}_task_grace",
        )
    elif backend == "uma":
        current_model = str(saved.get("model", model_default))
        if current_model not in UMA_MODEL_CHOICES:
            current_model = "uma-s-1p2"
        model = m1.selectbox(
            "Model",
            UMA_MODEL_CHOICES,
            index=UMA_MODEL_CHOICES.index(current_model),
            key=f"{key_prefix}_model_uma",
        )
        current_task = str(saved.get("task", task_default or "omat"))
        if current_task not in UMA_TASK_CHOICES:
            current_task = "omat"
        task = m2.selectbox(
            "Task",
            UMA_TASK_CHOICES,
            index=UMA_TASK_CHOICES.index(current_task),
            key=f"{key_prefix}_task_uma",
        )
    else:
        model = m1.text_input(
            "MACE model / checkpoint",
            value=str(saved.get("model", model_default)),
            key=f"{key_prefix}_model_mace",
            help="Examples: mh-1, mace-matpes-r2scan-0, or a local checkpoint path.",
        ).strip()
        task = m2.text_input(
            "MACE head",
            value=str(saved.get("task", task_default)),
            key=f"{key_prefix}_task_mace",
            help="For MH-1 refinement, matpes_r2scan is a useful high-fidelity materials head.",
        ).strip()

    gpu_id = int(
        m3.number_input(
            "GPU ID",
            min_value=0,
            value=int(saved.get("gpu_id", 0)),
            step=1,
            key=f"{key_prefix}_gpu",
            disabled=device != "cuda",
        )
    )

    r1, r2, r3, r4 = st.columns(4)
    relax = r1.checkbox(
        "Relax slabs",
        value=bool(saved.get("relax", True)),
        key=f"{key_prefix}_relax",
    )
    optimizer = r2.selectbox(
        "Optimizer",
        OPTIMIZER_CHOICES,
        index=_choice_index(
            OPTIMIZER_CHOICES,
            str(saved.get("optimizer", "bfgs")).lower(),
            "bfgs",
        ),
        key=f"{key_prefix}_optimizer",
        disabled=not relax,
    )
    fmax = float(
        r3.number_input(
            "fmax (eV/Å)",
            min_value=0.001,
            value=float(saved.get("fmax", fmax_default)),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_fmax",
            disabled=not relax,
        )
    )
    max_steps = int(
        r4.number_input(
            "Max relaxation steps",
            min_value=1,
            value=int(saved.get("max_steps", steps_default)),
            step=25,
            key=f"{key_prefix}_steps",
            disabled=not relax,
        )
    )

    p1, p2, p3 = st.columns(3)
    top_k = int(
        p1.number_input(
            "Top-k per bulk candidate",
            min_value=1,
            value=int(saved.get("top_k_per_candidate", topk_default)),
            step=1,
            key=f"{key_prefix}_topk",
        )
    )
    tf_threads = int(
        p2.number_input(
            "TensorFlow threads",
            min_value=1,
            value=int(saved.get("tf_threads", 1)),
            step=1,
            key=f"{key_prefix}_tf_threads",
        )
    )
    omp_threads = int(
        p3.number_input(
            "OpenMP threads",
            min_value=1,
            value=int(saved.get("omp_threads", 1)),
            step=1,
            key=f"{key_prefix}_omp_threads",
        )
    )

    return {
        **saved,
        "enabled": bool(enabled),
        "backend": backend,
        "model": model,
        "task": task,
        "device": device,
        "gpu_id": gpu_id,
        "tf_threads": tf_threads,
        "omp_threads": omp_threads,
        "relax": bool(relax),
        "optimizer": optimizer,
        "fmax": fmax,
        "max_steps": max_steps,
        "top_k_per_candidate": top_k,
    }


def _run_command(command: list[str], state_prefix: str, message: str) -> None:
    with st.spinner(message):
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
    st.session_state[f"{state_prefix}_stdout"] = completed.stdout
    st.session_state[f"{state_prefix}_stderr"] = completed.stderr
    st.session_state[f"{state_prefix}_returncode"] = completed.returncode
    if completed.returncode == 0:
        st.success("Calculation finished successfully.")
    else:
        st.error(f"Calculation exited with return code {completed.returncode}.")


def _resolved_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


with st.expander("Configuration & run controls", expanded=False):
    st.caption(
        "This page owns the complete surface-stage configuration. The main Input Builder "
        "only links here so the surface workflow is not duplicated in two places."
    )

    st.subheader("Bulk candidate selection")
    a1, a2, a3 = st.columns(3)
    enabled = a1.checkbox(
        "Enable surface stage",
        value=bool(surface.get("enabled", False)),
    )
    source_summary = a2.text_input(
        "Bulk results database",
        value=str(surface.get("source_summary", "results_database.csv")),
        help="Usually results_database.csv from the collect stage.",
    )
    outdir = a3.text_input(
        "Surface output directory",
        value=str(surface.get("outdir", "08_surfaces")),
    )

    composition_values = list(surface.get("composition_tags", []) or [])
    if surface.get("composition_tag"):
        composition_values.insert(0, str(surface["composition_tag"]))
    composition_text = st.text_input(
        "Composition filter(s), optional",
        value=", ".join(dict.fromkeys(str(x) for x in composition_values)),
        help="Leave empty to allow all compositions in the selected bulk database.",
    )
    composition_tags = _parse_csv(composition_text)

    selection_options = ["top_n", "id", "ids", "rank_range", "filters"]
    s1, s2, s3 = st.columns(3)
    selection_mode = s1.selectbox(
        "Selection mode",
        selection_options,
        index=_choice_index(
            selection_options,
            str(surface.get("selection_mode", "top_n")),
            "top_n",
        ),
    )

    candidate_id = int(surface.get("candidate_id", 1))
    candidate_ids = list(surface.get("candidate_ids", []) or [])
    rank_start = int(surface.get("rank_start", 1))
    rank_end = int(surface.get("rank_end", 10))
    top_n = int(surface.get("top_n", 3))
    formation_min = float(surface.get("formation_energy_min", -1e9))
    formation_max = float(surface.get("formation_energy_max", 1e9))
    bandgap_min = float(surface.get("bandgap_min", -1e9))
    bandgap_max = float(surface.get("bandgap_max", 1e9))

    if selection_mode == "top_n":
        top_n = int(
            s2.number_input(
                "Top bulk candidates",
                min_value=1,
                value=top_n,
                step=1,
            )
        )
    elif selection_mode == "id":
        candidate_id = int(
            s2.number_input(
                "Candidate ID",
                min_value=1,
                value=candidate_id,
                step=1,
            )
        )
    elif selection_mode == "ids":
        ids_text = s2.text_input(
            "Candidate IDs",
            value=", ".join(str(v) for v in candidate_ids),
            placeholder="1, 3, 7",
        )
        try:
            candidate_ids = [int(x) for x in _parse_csv(ids_text)]
        except ValueError:
            st.error("Candidate IDs must be comma-separated integers.")
            candidate_ids = []
    elif selection_mode == "rank_range":
        rank_start = int(
            s2.number_input("Rank start", min_value=1, value=rank_start, step=1)
        )
        rank_end = int(
            s3.number_input("Rank end", min_value=1, value=rank_end, step=1)
        )
    else:
        f1, f2, f3, f4 = st.columns(4)
        formation_min = float(f1.number_input("Formation energy min", value=formation_min))
        formation_max = float(f2.number_input("Formation energy max", value=formation_max))
        bandgap_min = float(f3.number_input("Band gap min (eV)", value=bandgap_min))
        bandgap_max = float(f4.number_input("Band gap max (eV)", value=bandgap_max))

    max_candidates = int(
        s3.number_input(
            "Maximum bulk candidates",
            min_value=1,
            value=int(surface.get("max_candidates", 20)),
            step=1,
            disabled=selection_mode == "rank_range",
        )
    )

    with st.expander("Surface orientations and slab construction", expanded=True):
        o1, o2, o3 = st.columns(3)
        orientation_mode = o1.selectbox(
            "Orientation mode",
            ["explicit", "automatic"],
            index=0 if str(surface.get("orientation_mode", "explicit")) == "explicit" else 1,
        )
        termination_mode = o2.selectbox(
            "Termination mode",
            ["all", "first"],
            index=0 if str(surface.get("termination_mode", "all")) == "all" else 1,
        )
        max_terms = int(
            o3.number_input(
                "Maximum terminations / orientation",
                min_value=1,
                value=int(surface.get("max_terminations_per_orientation", 12)),
                step=1,
            )
        )

        miller_list = list(surface.get("miller_list", DEFAULT_MILLERS))
        max_miller = int(surface.get("max_miller", 1))
        max_orientations = int(surface.get("max_orientations", 8))
        miller_error = None
        if orientation_mode == "explicit":
            miller_input = st.text_input(
                "Miller indices",
                value=_miller_text(miller_list),
                help="Use semicolons between facets, e.g. 1,1,0; 1,0,0; 1,0,1; 0,0,1.",
            )
            try:
                miller_list = _parse_millers(miller_input)
            except ValueError as exc:
                miller_error = str(exc)
                st.error(miller_error)
        else:
            m1, m2 = st.columns(2)
            max_miller = int(
                m1.number_input("Maximum Miller index", min_value=1, value=max_miller, step=1)
            )
            max_orientations = int(
                m2.number_input(
                    "Maximum symmetrically distinct orientations",
                    min_value=1,
                    value=max_orientations,
                    step=1,
                )
            )

        g1, g2, g3 = st.columns(3)
        min_slab = float(
            g1.number_input(
                "Minimum slab thickness (Å)",
                min_value=0.1,
                value=float(surface.get("min_slab_size", 12.0)),
                step=0.5,
            )
        )
        min_vacuum = float(
            g2.number_input(
                "Minimum vacuum thickness (Å)",
                min_value=0.1,
                value=float(surface.get("min_vacuum_size", 15.0)),
                step=0.5,
            )
        )
        max_total_surfaces = int(
            g3.number_input(
                "Maximum generated slab variants",
                min_value=1,
                value=int(surface.get("max_total_surfaces", 1000)),
                step=10,
            )
        )

        q1, q2, q3, q4 = st.columns(4)
        center_slab = q1.checkbox(
            "Center slab",
            value=bool(surface.get("center_slab", True)),
        )
        orthogonal_c = q2.checkbox(
            "Orthogonal c",
            value=bool(surface.get("orthogonal_c", True)),
        )
        symmetrize_slabs = q3.checkbox(
            "Request symmetric slabs",
            value=bool(surface.get("symmetrize_slabs", False)),
            help="May reduce or alter available terminations. Leave off when surveying all generated terminations.",
        )
        write_cif = q4.checkbox(
            "Also write CIF",
            value=bool(surface.get("write_cif", False)),
        )

        with st.expander("Advanced slab-generator controls", expanded=False):
            z1, z2, z3 = st.columns(3)
            in_unit_planes = z1.checkbox(
                "Thickness in unit planes",
                value=bool(surface.get("in_unit_planes", False)),
            )
            lll_reduce = z2.checkbox(
                "LLL reduce",
                value=bool(surface.get("lll_reduce", False)),
            )
            primitive = z3.checkbox(
                "Use primitive slab",
                value=bool(surface.get("primitive", False)),
            )
            reorient_lattice = st.checkbox(
                "Reorient lattice",
                value=bool(surface.get("reorient_lattice", True)),
            )

    with st.expander("Co-dopant depth / segregation scan", expanded=True):
        st.caption(
            "This is a representative depth scan, not an exhaustive same-species permutation search. "
            "One representative atom of each selected dopant is moved while total composition is preserved."
        )
        d1, d2, d3 = st.columns(3)
        variant_mode = d1.selectbox(
            "Dopant variant mode",
            ["co-dopant-depth", "none"],
            index=0 if str(surface.get("dopant_variant_mode", "co-dopant-depth")) == "co-dopant-depth" else 1,
        )
        host_species = d2.text_input(
            "Host cation",
            value=str(surface.get("host_species", doping.get("host_species", "Sn"))),
        ).strip()
        anion_text = d3.text_input(
            "Anion species",
            value=_csv_text(surface.get("anion_species", scan_cfg.get("anion_species", ["O"]))),
        )
        anion_species = _parse_csv(anion_text)

        dopants_text = st.text_input(
            "Dopant species",
            value=_csv_text(surface.get("dopant_species", [])),
            placeholder="Sb, Ti",
            help="Leave empty to infer all non-host, non-anion cations from each slab.",
            disabled=variant_mode == "none",
        )
        dopant_species = _parse_csv(dopants_text)

        v1, v2, v3, v4 = st.columns(4)
        zones_default = [
            str(x).lower()
            for x in surface.get("depth_zones", ["surface", "subsurface", "bulk"])
            if str(x).lower() in {"surface", "subsurface", "bulk"}
        ]
        depth_zones = v1.multiselect(
            "Depth zones",
            ["surface", "subsurface", "bulk"],
            default=zones_default or ["surface", "subsurface", "bulk"],
            disabled=variant_mode == "none",
        )
        placement_side = v2.selectbox(
            "Surface side",
            ["top", "bottom", "both"],
            index=_choice_index(
                ["top", "bottom", "both"],
                surface.get("placement_side", "top"),
                "top",
            ),
            disabled=variant_mode == "none",
        )
        layers_per_zone = int(
            v3.number_input(
                "Cation layers / zone",
                min_value=1,
                value=int(surface.get("layers_per_zone", 1)),
                step=1,
                disabled=variant_mode == "none",
            )
        )
        cation_layer_tol = float(
            v4.number_input(
                "Cation-layer tolerance (Å)",
                min_value=0.01,
                value=float(surface.get("cation_layer_tolerance_A", 0.8)),
                step=0.05,
                disabled=variant_mode == "none",
            )
        )
        w1, w2 = st.columns(2)
        include_original = w1.checkbox(
            "Keep original cut-slab dopant arrangement",
            value=bool(surface.get("include_original_variant", True)),
        )
        max_variants = int(
            w2.number_input(
                "Maximum depth variants / termination",
                min_value=1,
                value=int(surface.get("max_dopant_variants_per_termination", 18)),
                step=1,
                disabled=variant_mode == "none",
            )
        )
        st.info(
            "Segregation energy is reported relative to the same orientation/termination variant "
            "with all explicitly moved dopants in bulk-like layers. Negative E_seg means the "
            "selected surface/subsurface placement is preferred."
        )

    with st.expander("Fixed atoms during slab relaxation", expanded=False):
        c1, c2, c3 = st.columns(3)
        fix_atoms = c1.checkbox(
            "Fix part of the slab",
            value=bool(surface.get("fix_atoms", False)),
        )
        fix_region = c2.selectbox(
            "Fixed region",
            ["middle", "bottom"],
            index=0 if str(surface.get("fix_region", "middle")) == "middle" else 1,
            disabled=not fix_atoms,
        )
        fix_method = c3.selectbox(
            "Selection method",
            ["layers", "thickness"],
            index=0 if str(surface.get("fix_method", "layers")) == "layers" else 1,
            disabled=not fix_atoms,
        )
        fix_n_layers = int(surface.get("fix_n_layers", 2))
        fix_thickness = float(surface.get("fix_thickness_A", 4.0))
        f1, f2 = st.columns(2)
        if fix_method == "layers":
            fix_n_layers = int(
                f1.number_input(
                    "Fixed layers",
                    min_value=0,
                    value=fix_n_layers,
                    step=1,
                    disabled=not fix_atoms,
                )
            )
            fix_layer_tol = float(
                f2.number_input(
                    "Layer grouping tolerance (Å)",
                    min_value=0.01,
                    value=float(surface.get("fix_layer_tolerance_A", 0.6)),
                    step=0.05,
                    disabled=not fix_atoms,
                )
            )
        else:
            fix_thickness = float(
                f1.number_input(
                    "Fixed thickness (Å)",
                    min_value=0.0,
                    value=fix_thickness,
                    step=0.25,
                    disabled=not fix_atoms,
                )
            )
            fix_layer_tol = float(surface.get("fix_layer_tolerance_A", 0.6))

    st.divider()
    screen = _calculator_editor(
        "Fast surface screen",
        "surface_screen",
        screen_saved,
        enabled_default=True,
        backend_default="grace",
        model_default="GRACE-1L-OMAT",
        task_default="",
        fmax_default=0.05,
        steps_default=300,
        topk_default=10,
    )

    st.divider()
    refine = _calculator_editor(
        "Higher-fidelity refinement",
        "surface_refine",
        refine_saved,
        enabled_default=False,
        backend_default="mace",
        model_default="mh-1",
        task_default="matpes_r2scan",
        fmax_default=0.03,
        steps_default=500,
        topk_default=5,
    )

    resolved_surface = dict(surface)
    resolved_surface.update(
        {
            "enabled": bool(enabled),
            "source_summary": source_summary,
            "selection_mode": selection_mode,
            "candidate_id": candidate_id,
            "candidate_ids": candidate_ids,
            "rank_start": rank_start,
            "rank_end": rank_end,
            "top_n": top_n,
            "formation_energy_min": formation_min,
            "formation_energy_max": formation_max,
            "bandgap_min": bandgap_min,
            "bandgap_max": bandgap_max,
            "max_candidates": max_candidates,
            "orientation_mode": orientation_mode,
            "miller_list": miller_list,
            "max_miller": max_miller,
            "max_orientations": max_orientations,
            "min_slab_size": min_slab,
            "min_vacuum_size": min_vacuum,
            "center_slab": center_slab,
            "in_unit_planes": in_unit_planes,
            "lll_reduce": lll_reduce,
            "primitive": primitive,
            "reorient_lattice": reorient_lattice,
            "orthogonal_c": orthogonal_c,
            "termination_mode": termination_mode,
            "max_terminations_per_orientation": max_terms,
            "symmetrize_slabs": symmetrize_slabs,
            "outdir": outdir,
            "max_total_surfaces": max_total_surfaces,
            "write_cif": write_cif,
            "fix_atoms": fix_atoms,
            "fix_region": fix_region,
            "fix_method": fix_method,
            "fix_n_layers": fix_n_layers,
            "fix_thickness_A": fix_thickness,
            "fix_layer_tolerance_A": fix_layer_tol,
            "dopant_variant_mode": variant_mode,
            "host_species": host_species,
            "dopant_species": dopant_species,
            "anion_species": anion_species,
            "depth_zones": depth_zones,
            "placement_side": placement_side,
            "cation_layer_tolerance_A": cation_layer_tol,
            "layers_per_zone": layers_per_zone,
            "include_original_variant": include_original,
            "max_dopant_variants_per_termination": max_variants,
            "screen": screen,
            "refine": refine,
        }
    )
    resolved_surface.pop("composition_tag", None)
    resolved_surface["composition_tags"] = composition_tags

    # Once saved through the dedicated page, retire the former flat
    # surface-relaxation controls so there is one authoritative configuration.
    for legacy_key in (
        "relax_surface",
        "surface_backend",
        "surface_model",
        "surface_task",
        "surface_optimizer",
        "surface_device",
        "surface_gpu_id",
        "surface_tf_threads",
        "surface_omp_threads",
        "surface_fmax",
        "surface_max_steps",
        "surface_relaxed_filename",
        "surface_relax_log_filename",
        "surface_relax_traj_filename",
        "surface_relax_meta_filename",
        "write_poscar",
        "write_metadata_json",
        "summary_csv",
    ):
        resolved_surface.pop(legacy_key, None)

    resolved_cfg = dict(cfg)
    resolved_cfg["surface"] = resolved_surface

    validation_error = miller_error if orientation_mode == "explicit" else None
    parsed_surface = None
    if validation_error is None:
        try:
            parsed_surface = parse_surface_config(resolved_cfg)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            validation_error = str(exc)

    st.divider()
    st.subheader("Save and run")

    with st.expander("Preview [surface] TOML", expanded=False):
        st.code(toml.dumps({"surface": resolved_surface}), language="toml")

    if validation_error:
        st.error(validation_error)

    if parsed_surface is not None:
        with st.expander("Preview selected bulk structures", expanded=False):
            try:
                preview = preview_surface_candidates(resolved_cfg, project_root)
            except Exception as exc:
                st.warning(str(exc))
            else:
                preview_cols = [
                    col
                    for col in (
                        "composition_tag",
                        "candidate",
                        "rank_relax_filtered",
                        "E_relaxed_eV",
                        "E_form_norm",
                        "bandgap_eV",
                        "candidate_path",
                    )
                    if col in preview.columns
                ]
                st.dataframe(
                    preview[preview_cols],
                    use_container_width=True,
                    hide_index=True,
                )
                if len(preview):
                    estimated_upper = (
                        len(preview)
                        * (
                            len(miller_list)
                            if orientation_mode == "explicit"
                            else max_orientations
                        )
                        * max_terms
                        * (
                            max_variants
                            if variant_mode == "co-dopant-depth"
                            else 1
                        )
                    )
                    st.caption(
                        "Conservative pre-deduplication ceiling from the current caps: "
                        f"{estimated_upper:,} slab variants. Actual generation can be much smaller."
                    )

    expensive_confirm = st.checkbox(
        "I confirm that running the selected MLFF surface calculations may be computationally expensive",
        value=False,
        key="surface_expensive_confirm",
    )

    execution_mode = st.radio(
        "Execution environment",
        ["Current environment", "Named Conda environments"],
        horizontal=True,
        help=(
            "Use the current environment when this GUI is already running inside the required "
            "GRACE or MACE environment. Named Conda environments allow the screen and refinement "
            "to use separate installations while sharing the same project files."
        ),
    )

    screen_env = "dopingflow-grace"
    refine_env = "dopingflow-mace"
    if execution_mode == "Named Conda environments":
        e1, e2 = st.columns(2)
        screen_env = e1.text_input("Screen Conda environment", value=screen_env)
        refine_env = e2.text_input("Refinement Conda environment", value=refine_env)

    def command_for(stage: str) -> list[str]:
        command = ["dopingflow", stage, "-c", str(config_path)]
        if execution_mode == "Named Conda environments":
            env = screen_env if stage == "surface-scan" else refine_env
            command = ["conda", "run", "-n", env, *command]
        return command

    screen_command = command_for("surface-scan")
    refine_command = command_for("surface-refine")

    st.markdown("##### Commands")
    st.code(
        "\n".join(
            [
                " ".join(shlex.quote(token) for token in screen_command),
                " ".join(shlex.quote(token) for token in refine_command),
            ]
        ),
        language="bash",
    )

    save_col, scan_col, refine_col = st.columns(3)
    with save_col:
        if st.button(
            "Save surface settings",
            type="primary",
            use_container_width=True,
            disabled=validation_error is not None,
        ):
            config_path.write_text(toml.dumps(resolved_cfg), encoding="utf-8")
            st.success(f"Saved {config_path}")

    with scan_col:
        if st.button(
            "Run surface screen",
            use_container_width=True,
            disabled=(not enabled) or validation_error is not None or not expensive_confirm,
        ):
            config_path.write_text(toml.dumps(resolved_cfg), encoding="utf-8")
            _run_command(
                screen_command,
                "surface_screen_last",
                "Running surface screening...",
            )

    with refine_col:
        if st.button(
            "Run refinement",
            use_container_width=True,
            disabled=(
                (not enabled)
                or (not refine.get("enabled", False))
                or validation_error is not None
                or not expensive_confirm
            ),
        ):
            config_path.write_text(toml.dumps(resolved_cfg), encoding="utf-8")
            _run_command(
                refine_command,
                "surface_refine_last",
                "Running higher-fidelity surface refinement...",
            )

    if not refine.get("enabled", False):
        st.caption("Enable **Higher-fidelity refinement** to activate the refinement run action.")

    for state_prefix, title in (
        ("surface_screen_last", "Last screening run"),
        ("surface_refine_last", "Last refinement run"),
    ):
        if f"{state_prefix}_returncode" in st.session_state:
            with st.expander(title, expanded=False):
                stdout = st.session_state.get(f"{state_prefix}_stdout")
                stderr = st.session_state.get(f"{state_prefix}_stderr")
                if stdout:
                    st.text(stdout)
                if stderr:
                    st.text(stderr)


st.divider()
st.subheader("Results explorer")
results_root = _resolved_path(str(surface.get("outdir", "08_surfaces")))
if "resolved_surface" in locals():
    results_root = _resolved_path(str(resolved_surface.get("outdir", "08_surfaces")))
st.caption(f"Resolved surface output: `{results_root}`")

screen_summary_name = str(
    (resolved_surface if "resolved_surface" in locals() else surface).get(
        "screen_summary_csv", "surface_screen_summary.csv"
    )
)
screen_selected_name = str(
    (resolved_surface if "resolved_surface" in locals() else surface).get(
        "screen_selected_csv", "surface_screen_selected.csv"
    )
)
refine_summary_name = str(
    (resolved_surface if "resolved_surface" in locals() else surface).get(
        "refine_summary_csv", "surface_refine_summary.csv"
    )
)
final_selected_name = str(
    (resolved_surface if "resolved_surface" in locals() else surface).get(
        "refine_selected_csv", "surface_final_selected.csv"
    )
)

screen_df = _read_csv(results_root / screen_summary_name)
screen_selected_df = _read_csv(results_root / screen_selected_name)
refine_df = _read_csv(results_root / refine_summary_name)
final_df = _read_csv(results_root / final_selected_name)

if screen_df.empty and refine_df.empty:
    st.info(
        "No staged surface results are available yet. Run the surface screen to create "
        "surface_screen_summary.csv and the shortlist."
    )
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Screened variants", len(screen_df))
m2.metric(
    "Screen-rankable",
    int(
        screen_df.get("screen_rankable", pd.Series(dtype=object))
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )
    if not screen_df.empty
    else 0,
)
m3.metric("Refined variants", len(refine_df))
m4.metric("Final shortlist", len(final_df))

ranking_tab, structure_tab, raw_tab = st.tabs(
    ["Surface ranking", "Structure browser", "Raw data"]
)

with ranking_tab:
    available_stages = ["Screen"]
    if not refine_df.empty:
        available_stages.append("Refinement")
    ranking_stage = st.radio(
        "Energy level",
        available_stages,
        horizontal=True,
        key="surface_ranking_stage",
    )
    data = refine_df.copy() if ranking_stage == "Refinement" else screen_df.copy()
    prefix = "refine" if ranking_stage == "Refinement" else "screen"

    if not data.empty:
        compositions = data["composition_tag"].dropna().astype(str).drop_duplicates().tolist()
        selected_composition = st.selectbox(
            "Composition",
            compositions,
            key="surface_result_composition",
        )
        subset = data[data["composition_tag"].astype(str) == selected_composition].copy()
        candidates = subset["candidate"].dropna().astype(str).drop_duplicates().tolist()
        selected_candidate = st.selectbox(
            "Bulk parent",
            candidates,
            key="surface_result_candidate",
        )
        subset = subset[subset["candidate"].astype(str) == selected_candidate].copy()

        gamma_col = f"{prefix}_surface_energy_J_m2"
        status_col = f"{prefix}_surface_energy_status"
        rank_col = f"{prefix}_rank_overall"
        if gamma_col in subset.columns:
            subset[gamma_col] = pd.to_numeric(subset[gamma_col], errors="coerce")
        subset["Facet"] = subset.apply(
            lambda row: f"({int(row['miller_h'])}{int(row['miller_k'])}{int(row['miller_l'])})",
            axis=1,
        )

        rankable = subset[
            subset.get(status_col, pd.Series(index=subset.index, dtype=str)).eq("ok")
            & subset.get(gamma_col, pd.Series(index=subset.index, dtype=float)).notna()
        ].copy()

        if rankable.empty:
            st.warning(
                "No proportional/stoichiometric slabs are rankable for this bulk parent at "
                f"the {ranking_stage.lower()} energy level."
            )
        else:
            fig = px.scatter(
                rankable,
                x="Facet",
                y=gamma_col,
                color="variant_label",
                symbol="termination_id",
                hover_data=[
                    "termination_id",
                    "variant_label",
                    rank_col,
                    f"{prefix}_energy_eV",
                ],
                labels={
                    gamma_col: "Surface energy (J/m²)",
                    "variant_label": "Dopant-depth variant",
                },
                title=f"{ranking_stage} surface-energy ranking — {selected_composition} / {selected_candidate}",
            )
            st.plotly_chart(fig, use_container_width=True)

            top_columns = [
                col
                for col in (
                    rank_col,
                    "Facet",
                    "termination_id",
                    "variant_label",
                    gamma_col,
                    f"{prefix}_segregation_energy_eV",
                    f"{prefix}_final_fmax_eV_per_A",
                )
                if col in rankable.columns
            ]
            st.markdown("##### Lowest surface energies")
            st.dataframe(
                rankable.sort_values(gamma_col).head(20)[top_columns],
                use_container_width=True,
                hide_index=True,
            )

        segregation_col = f"{prefix}_segregation_energy_eV"
        segregation_status = f"{prefix}_segregation_status"
        if segregation_col in subset.columns:
            st.markdown("##### Dopant segregation within one termination")
            facet_options = subset["Facet"].drop_duplicates().tolist()
            chosen_facet = st.selectbox(
                "Facet for segregation view",
                facet_options,
                key=f"surface_segregation_facet_{prefix}",
            )
            seg_subset = subset[subset["Facet"] == chosen_facet].copy()
            term_options = sorted(
                int(v)
                for v in pd.to_numeric(
                    seg_subset["termination_id"], errors="coerce"
                ).dropna().unique()
            )
            if term_options:
                chosen_term = st.selectbox(
                    "Termination",
                    term_options,
                    key=f"surface_segregation_term_{prefix}",
                )
                seg_subset = seg_subset[
                    pd.to_numeric(seg_subset["termination_id"], errors="coerce")
                    == chosen_term
                ].copy()
                seg_subset[segregation_col] = pd.to_numeric(
                    seg_subset[segregation_col], errors="coerce"
                )
                if segregation_status in seg_subset.columns:
                    seg_subset = seg_subset[
                        seg_subset[segregation_status].astype(str) == "ok"
                    ]
                seg_subset = seg_subset[seg_subset[segregation_col].notna()]
                if seg_subset.empty:
                    st.info(
                        "No all-bulk-like reference variant is available for this termination, "
                        "so a segregation energy cannot be assigned."
                    )
                else:
                    seg_fig = px.scatter(
                        seg_subset,
                        x="variant_label",
                        y=segregation_col,
                        hover_data=[f"{prefix}_energy_eV", "target_zones_json"],
                        labels={
                            "variant_label": "Dopant-depth variant",
                            segregation_col: "Segregation energy (eV)",
                        },
                    )
                    seg_fig.add_hline(y=0.0, line_dash="dash")
                    st.plotly_chart(seg_fig, use_container_width=True)
                    st.caption(
                        "E_seg = E_variant − E_all-bulk-like for the same bulk parent, facet, "
                        "termination, composition, and calculator. Negative values indicate "
                        "surface/subsurface enrichment relative to the bulk-like placement."
                    )

with structure_tab:
    available = []
    if not refine_df.empty:
        available.append(("Refinement", refine_df, "refine"))
    if not screen_df.empty:
        available.append(("Screen", screen_df, "screen"))

    stage_names = [item[0] for item in available]
    structure_stage = st.selectbox(
        "Structure energy level",
        stage_names,
        key="surface_structure_stage",
    )
    _, structure_data, prefix = next(item for item in available if item[0] == structure_stage)
    browse = structure_data.copy()

    b1, b2 = st.columns(2)
    comp_options = browse["composition_tag"].dropna().astype(str).drop_duplicates().tolist()
    comp = b1.selectbox("Composition", comp_options, key="surface_browser_comp")
    browse = browse[browse["composition_tag"].astype(str) == comp]

    cand_options = browse["candidate"].dropna().astype(str).drop_duplicates().tolist()
    cand = b2.selectbox("Bulk parent", cand_options, key="surface_browser_candidate")
    browse = browse[browse["candidate"].astype(str) == cand].copy()
    browse["choice"] = browse.apply(
        lambda row: (
            f"({int(row['miller_h'])}{int(row['miller_k'])}{int(row['miller_l'])}) | "
            f"term {int(row['termination_id']):03d} | {row.get('variant_label', 'original')}"
        ),
        axis=1,
    )

    choice = st.selectbox(
        "Surface structure",
        browse["choice"].tolist(),
        key="surface_browser_choice",
    )
    row = browse[browse["choice"] == choice].iloc[0].to_dict()

    energy = row.get(f"{prefix}_energy_eV")
    gamma = row.get(f"{prefix}_surface_energy_J_m2")
    eseg = row.get(f"{prefix}_segregation_energy_eV")
    fmax = row.get(f"{prefix}_final_fmax_eV_per_A")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total energy", "-" if pd.isna(energy) else f"{float(energy):.5f} eV")
    k2.metric("Surface energy", "-" if pd.isna(gamma) else f"{float(gamma):.4f} J/m²")
    k3.metric("Segregation energy", "-" if pd.isna(eseg) else f"{float(eseg):+.4f} eV")
    k4.metric("Final fmax", "-" if pd.isna(fmax) else f"{float(fmax):.4f} eV/Å")

    relaxed_key = f"{prefix}_relaxed_structure_path"
    structure_path_text = str(row.get(relaxed_key, "") or "").strip()
    if not structure_path_text:
        structure_path_text = str(row.get("generated_structure_path", "") or "").strip()
    structure_path = Path(structure_path_text) if structure_path_text else Path()
    if structure_path_text and not structure_path.is_absolute():
        structure_path = (project_root / structure_path).resolve()

    st.caption(f"Structure file: `{structure_path}`")
    if structure_path_text and structure_path.exists():
        try:
            show_structure(
                structure_path,
                title=choice,
                width=900,
                height=500,
            )
        except Exception as exc:
            st.warning(f"Could not render the selected structure: {exc}")
    else:
        st.warning("The selected structure file is not available at the recorded path.")

    with st.expander("Selected structure metadata", expanded=False):
        clean_meta = {}
        for key, value in row.items():
            if isinstance(value, float) and pd.isna(value):
                clean_meta[key] = None
            elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
                try:
                    clean_meta[key] = value.item()
                except Exception:
                    clean_meta[key] = value
            else:
                clean_meta[key] = value
        st.json(clean_meta)

with raw_tab:
    table_choice = st.selectbox(
        "Table",
        [
            "Screen summary",
            "Screen shortlist",
            "Refinement summary",
            "Final shortlist",
        ],
        key="surface_raw_table",
    )
    tables = {
        "Screen summary": screen_df,
        "Screen shortlist": screen_selected_df,
        "Refinement summary": refine_df,
        "Final shortlist": final_df,
    }
    raw = tables[table_choice]
    if raw.empty:
        st.info(f"{table_choice} is not available yet.")
    else:
        st.dataframe(raw, use_container_width=True, hide_index=True)

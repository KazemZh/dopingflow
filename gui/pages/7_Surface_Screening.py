Warning: truncated output (original token count: 15265)
Total output lines: 1677

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
    ensure_surface_ids,
    parse_surface_config,
    preview_surface_candidates,
    resolve_surface_output_dir,
    select_refinement_candidates,
)
from gui_config import (
    BACKEND_CHOICES,
    DEVICE_CHOICES,
    GRACE_MODEL_CHOICES,
    OPTIMIZER_CHOICES,
    UMA_MODEL_CHOICES,
    UMA_TASK_CHOICES,
)
from view_structure import show_structure, structure_viewer_controls


st.set_page_config(page_title="Surface screening", layout="wide")
st.title("Surface screening")
st.caption(
    "Generate low-index slabs from selected vacancy-free or oxygen-vacancy structures, enumerate terminations, "
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
structure_cfg = dict(cfg.get("structure", {}) or {})
oxidation_cfg = dict(cfg.get("oxidation", {}) or {})
conductivity_cfg = dict(cfg.get("conductivity", {}) or {})


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
    show_top_k: bool = True,
    topk_label: str = "Top-k per source structure",
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

    top_k = int(saved.get("top_k_per_candidate", topk_default))
    if show_top_k:
        p1, p2, p3 = st.columns(3)
        top_k = int(
            p1.number_input(
                topk_label,
                min_value=1,
                value=top_k,
                step=1,
                key=f"{key_prefix}_topk",
            )
        )
        tf_widget, omp_widget = p2, p3
    else:
        tf_widget, omp_widget = st.columns(2)

    tf_threads = int(
        tf_widget.number_input(
            "TensorFlow threads",
            min_value=1,
            value=int(saved.get("tf_threads", 1)),
            step=1,
            key=f"{key_prefix}_tf_threads",
        )
    )
    omp_threads = int(
        omp_widget.number_input(
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

def _ensure_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "target_id" not in out.columns:
        composition = out.get("composition_tag", pd.Series("", index=out.index)).astype(str)
        candidate = out.get("candidate", pd.Series("", index=out.index)).astype(str)
        out["target_id"] = composition.str.rstrip("/") + "/" + candidate.str.lstrip("/")
    if "structure_kind" not in out.columns:
        out["structure_kind"] = "vacancy-free"
    if "n_oxygen_vacancies" not in out.columns:
        out["n_oxygen_vacancies"] = 0
    return out


def _merge_refinement_into_screen_raw(
    screen: pd.DataFrame,
    refine: pd.DataFrame,
) -> pd.DataFrame:
    """Add refinement-result columns to the complete screen table.

    The screen table remains the master list, so every screened termination is
    retained. Rows that were selected for refinement receive their ``refine_*``
    values by stable ``surface_id``; unrefined rows keep those columns empty.
    """
    if screen.empty:
        return screen.copy()
    if refine.empty:
        return screen.copy()

    screen_with_ids = ensure_surface_ids(screen)
    refine_with_ids = ensure_surface_ids(refine)
    refine_columns = [
        column
        for column in refine_with_ids.columns
        if column.startswith("refine_")
        and column not in screen_with_ids.columns
    ]
    if not refine_columns:
        return screen_with_ids

    refinement_data = (
        refine_with_ids[["surface_id", *refine_columns]]
        .drop_duplicates("surface_id", keep="last")
    )
    merged = screen_with_ids.merge(
        refinement_data,
        on="surface_id",
        how="left",
        validate="one_to_one",
    )

    # Put the most useful screen/refinement comparison fields next to each other
    # instead of leaving every refinement column at the far right of the table.
    preferred_pairs = [
        ("screen_energy_eV", "refine_energy_eV"),
        ("screen_energy_eV_atom", "refine_energy_eV_atom"),
        ("screen_surface_energy_J_m2", "refine_surface_energy_J_m2"),
        ("screen_surface_energy_status", "refine_surface_energy_status"),
        ("screen_rank_overall", "refine_rank_overall"),
        ("screen_rank_within_hkl", "refine_rank_within_hkl"),
        ("screen_converged", "refine_converged"),
        ("screen_final_fmax_eV_per_A", "refine_final_fmax_eV_per_A"),
        ("screen_optimizer_steps", "refine_optimizer_steps"),
    ]
    ordered: list[str] = []
    paired = {name for pair in preferred_pairs for name in pair}
    for column in merged.columns:
        if column in paired:
            continue
        ordered.append(column)
        if column == "termination_label":
            for left, right in preferred_pairs:
                if left in merged.columns and left not in ordered:
                    ordered.append(left)
                if right in merged.columns and right not in ordered:
                    ordered.append(right)
    for left, right in preferred_pairs:
        if left in merged.columns and left not in ordered:
            ordered.append(left)
        if right in merged.columns and right not in ordered:
            ordered.append(right)
    return merged[ordered]

def _target_label(row: pd.Series | dict[str, Any]) -> str:
    kind = str(row.get("structure_kind", "vacancy-free"))
    target_id = str(row.get("target_id", ""))
    n_vac = int(row.get("n_oxygen_vacancies", 0) or 0)
    if kind == "oxygen-vacancy":
        return f"O-vacancy ({n_vac}) — {target_id}"
    return f"Vacancy-free — {target_id}"


with st.expander("Configuration & run controls", expanded=False):
    st.caption(
        "This page owns the complete surface-stage configuration. The main Input Builder "
        "only links here so the surface workflow is not duplicated in two places."
    )

    st.subheader("Stage and structures")
    stage_left, stage_right = st.columns(2)
    enabled = stage_left.checkbox(
        "Enable surface stage",
        value=bool(surface.get("enabled", False)),
    )
    outdir = stage_right.text_input(
        "Surface output directory",
        value=str(surface.get("outdir", "08_surfaces")),
        help="Relative paths are created inside Source root; absolute paths are used exactly as entered.",
    )

    source_default = (
        str(surface.get("source_root", "")).strip()
        or str(conductivity_cfg.get("source_root", "")).strip()
        or str(oxidation_cfg.get("source_root", "")).strip()
        or str(structure_cfg.get("outdir", "random_structures")).strip()
    )

    include_left, include_right = st.columns(2)
    include_vacancy_free = include_left.checkbox(
        "Vacancy-free parents",
        value=bool(surface.get("include_vacancy_free", True)),
        help="Include the selected relaxed parent structures found under Source root.",
    )
    include_oxygen_vacancies = include_right.checkbox(
        "O-vacancy structures",
        value=bool(surface.get("include_oxygen_vacancies", False)),
        help=(
            "Include relaxed oxygen-vacancy structures listed in "
            "vacancies_database.json under Source root."
        ),
    )

    source_root = st.text_input(
        "Source root",
        value=source_default,
        help=(
            "Uses the same structure-discovery convention as Electronic Conductivity "
            "and Oxidation States. The directory normally contains composition folders "
            "with selected_candidates.txt and, when requested, vacancies_database.json."
        ),
    )

    saved_target_include = surface.get("target_include", [])
    target_include_text = st.text_input(
        "Target selector(s) (optional)",
        value=_csv_text(saved_target_include),
        help=(
            "Leave empty to use every discovered structure allowed by the vacancy toggles. "
            "Use exact IDs or wildcards, e.g. Sb5_Ti2p5/candidate_014, "
            "Sb5_Ti2p5/*, or a specific V_O target."
        ),
    )
    target_include = _parse_csv(target_include_text)
    if target_include:
        st.info(
            f"Target filtering is active: only structures matching {target_include} "
            "will be surface-screened."
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

        bulk_structure = st.text_input("Bulk unit-cell file (optional)", value=surface.get("bulk_structure", ""),
            help="Supply a periodic POSCAR or CIF relative to input.toml to build facets directly. Screen the slabs here, then select them in Surface Pourbaix for adsorbate coverage.")
        conventional = st.checkbox("Use conventional-cell Miller indices", value=surface.get("conventional_millers", False))
        repeat = surface.get("inplane_repeat", [1, 1])
        ra, rb = st.columns(2)
        repeat = [ra.number_input("In-plane repeats along a", 1, value=int(repeat[0])),
                  rb.number_input("In-plane repeats along b", 1, value=int(repeat[1]))]
        miller_list = list(surface.get("miller_list", DEFAULT_MILLERS))
        max_miller = int(surface.get("max_miller", 1))
        max_orientations = int(surface.get("max_orientations", 8))
        miller_error = Non…5265 tokens truncated…    )
    resolved_surface = dict(surface)
    resolved_surface.update(
        {
            "enabled": bool(enabled),
            "source_root": source_root, "bulk_structure": bulk_structure,
            "conventional_millers": conventional, "inplane_repeat": repeat,
            "include_vacancy_free": bool(include_vacancy_free),
            "include_oxygen_vacancies": bool(include_oxygen_vacancies),
            "target_include": target_include,
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
            "host_species": host_species,
            "dopant_species": dopant_species,
            "anion_species": anion_species,
            "cation_layer_tolerance_A": cation_layer_tol,
            "dopant_depth_layers": dopant_depth_layers,
            "screen": screen,
            "refine": refine,
        }
    )
    # Once saved through the dedicated page, retire the former flat
    # surface-relaxation controls so there is one authoritative configuration.
    for legacy_key in (
        "source_summary",
        "composition_tag",
        "composition_tags",
        "selection_mode",
        "candidate_id",
        "candidate_ids",
        "rank_start",
        "rank_end",
        "top_n",
        "formation_energy_min",
        "formation_energy_max",
        "bandgap_min",
        "bandgap_max",
        "max_candidates",
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
        "dopant_variant_mode",
        "depth_zones",
        "placement_side",
        "layers_per_zone",
        "include_original_variant",
        "max_dopant_variants_per_termination",
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
        with st.expander("Preview selected structures", expanded=False):
            try:
                preview = preview_surface_candidates(resolved_cfg, project_root)
            except Exception as exc:
                st.warning(str(exc))
            else:
                preview_cols = [
                    col
                    for col in (
                        "target_id",
                        "structure_kind",
                        "n_oxygen_vacancies",
                        "composition_tag",
                        "candidate",
                        "structure_path",
                    )
                    if col in preview.columns
                ]
                st.dataframe(
                    preview[preview_cols],
                    use_container_width=True,
                    hide_index=True,
                )
                for warning in preview.attrs.get("warnings", []):
                    st.warning(str(warning))
                if len(preview):
                    estimated_upper = (
                        len(preview)
                        * (
                            len(miller_list)
                            if orientation_mode == "explicit"
                            else max_orientations
                        )
                        * max_terms
                    )
                    st.caption(
                        "Conservative pre-deduplication ceiling from the current caps: "
                        f"{estimated_upper:,} natural terminations. Actual generation can be smaller."
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
results_config = resolved_cfg if "resolved_cfg" in locals() else cfg
results_surface = (
    resolved_surface
    if "resolved_surface" in locals()
    else parse_surface_config(results_config)
)
results_root = resolve_surface_output_dir(
    results_config,
    results_surface,
    project_root,
)
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

screen_df = _ensure_target_columns(_read_csv(results_root / screen_summary_name))
screen_selected_df = _ensure_target_columns(_read_csv(results_root / screen_selected_name))
refine_df = _ensure_target_columns(_read_csv(results_root / refine_summary_name))
final_df = _ensure_target_columns(_read_csv(results_root / final_selected_name))

if screen_df.empty and refine_df.empty:
    st.info(
        "No staged surface results are available yet. Run the surface screen to create "
        "surface_screen_summary.csv and the shortlist."
    )
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Screened terminations", len(screen_df))
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
m3.metric("Refined terminations", len(refine_df))
m4.metric("Retained refined", len(final_df))

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
        target_rows = (
            data.sort_values("target_id")
            .drop_duplicates("target_id")
            .set_index("target_id")
        )
        target_options = target_rows.index.astype(str).tolist()
        selected_target = st.selectbox(
            "Source structure",
            target_options,
            format_func=lambda target_id: _target_label(target_rows.loc[target_id]),
            key="surface_result_target",
        )
        subset = data[data["target_id"].astype(str) == selected_target].copy()
        selected_meta = target_rows.loc[selected_target]
        st.caption(
            f"Source: {selected_meta.get('structure_kind', 'vacancy-free')} | "
            f"O vacancies: {int(selected_meta.get('n_oxygen_vacancies', 0) or 0)} | "
            f"{selected_meta.get('source_structure_path', selected_meta.get('bulk_structure_path', ''))}"
        )

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
                "No proportional/stoichiometric slabs are rankable for this source structure at "
                f"the {ranking_stage.lower()} energy level."
            )
        else:
            fig = px.scatter(
                rankable,
                x="Facet",
                y=gamma_col,
                color="termination_label",
                symbol="termination_id",
                hover_data=[
                    "termination_id",
                    "termination_label",
                    rank_col,
                    f"{prefix}_energy_eV",
                ],
                labels={
                    gamma_col: "Surface energy (J/m²)",
                    "termination_label": "Natural dopant positions",
                },
                title=f"{ranking_stage} surface-energy ranking — {selected_target}",
            )
            st.plotly_chart(fig, use_container_width=True)

            top_columns = [
                col
                for col in (
                    rank_col,
                    "Facet",
                    "termination_id",
                    "termination_label",
                    gamma_col,
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

    browser_rows = (
        browse.sort_values("target_id")
        .drop_duplicates("target_id")
        .set_index("target_id")
    )
    browser_targets = browser_rows.index.astype(str).tolist()
    browser_target = st.selectbox(
        "Source structure",
        browser_targets,
        format_func=lambda target_id: _target_label(browser_rows.loc[target_id]),
        key="surface_browser_target",
    )
    browse = browse[browse["target_id"].astype(str) == browser_target].copy()
    browse["choice"] = browse.apply(
        lambda row: (
            f"({int(row['miller_h'])}{int(row['miller_k'])}{int(row['miller_l'])}) | "
            f"term {int(row['termination_id']):03d} | {row.get('termination_label', 'unlabeled')}"
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
    fmax = row.get(f"{prefix}_final_fmax_eV_per_A")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total energy", "-" if pd.isna(energy) else f"{float(energy):.5f} eV")
    k2.metric("Surface energy", "-" if pd.isna(gamma) else f"{float(gamma):.4f} J/m²")
    k3.metric("Final fmax", "-" if pd.isna(fmax) else f"{float(fmax):.4f} eV/Å")

    relaxed_key = f"{prefix}_relaxed_structure_path"
    structure_path_text = str(row.get(relaxed_key, "") or "").strip()
    if not structure_path_text:
        structure_path_text = str(row.get("generated_structure_path", "") or "").strip()
    structure_path = Path(structure_path_text) if structure_path_text else Path()
    if structure_path_text and not structure_path.is_absolute():
        structure_path = (project_root / structure_path).resolve()

    st.caption(f"Structure file: `{structure_path}`")
    if structure_path_text and structure_path.exists():
        viewer_options = structure_viewer_controls(
            [structure_path],
            key_prefix=f"surface_screening_viewer_{browser_target}_{choice}",
            expanded=False,
            show_orientation_default=True,
            show_unit_cell_default=True,
        )
        try:
            show_structure(
                structure_path,
                title=choice,
                width=900,
                height=500,
                viewer_mode="surface",
                **viewer_options,
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
            "Retained refined set",
        ],
        key="surface_raw_table",
    )
    combined_screen_df = _merge_refinement_into_screen_raw(
        screen_df,
        refine_df,
    )
    combined_screen_selected_df = _merge_refinement_into_screen_raw(
        screen_selected_df,
        refine_df,
    )
    tables = {
        "Screen summary": combined_screen_df,
        "Screen shortlist": combined_screen_selected_df,
        "Refinement summary": refine_df,
        "Retained refined set": final_df,
    }
    raw = tables[table_choice]
    if raw.empty:
        st.info(f"{table_choice} is not available yet.")
    else:
        if table_choice in {"Screen summary", "Screen shortlist"} and not refine_df.empty:
            st.caption(
                "This screen table also includes matching refinement-result columns "
                "(refine_*) for terminations that were refined. Screened terminations "
                "without refinement remain blank in those columns."
            )
        st.dataframe(raw, use_container_width=True, hide_index=True)

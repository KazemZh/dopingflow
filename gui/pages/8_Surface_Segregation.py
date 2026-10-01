"""Configure, run, and inspect finite-temperature surface segregation MC."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import toml

from dopingflow.surface_segregation import (
    parse_surface_segregation_config,
    preview_surface_segregation_targets,
    resolve_surface_segregation_output_dir,
    resolve_surface_segregation_source_summary,
)
from gui_config import (
    BACKEND_CHOICES,
    DEVICE_CHOICES,
    GRACE_MODEL_CHOICES,
    MACE_MODEL_CHOICES,
    UMA_MODEL_CHOICES,
    UMA_TASK_CHOICES,
)


st.set_page_config(page_title="Surface segregation MC", layout="wide")
st.title("Surface segregation Monte Carlo")
st.caption(
    "Sample fixed-composition host↔dopant swaps on user-selected natural surface "
    "terminations with a chosen ML force field. The MC reports site- and depth-resolved "
    "dopant occupancies and occupancy-derived finite-temperature segregation free energies."
)

project_root = Path(
    st.sidebar.text_input(
        "Project root",
        value=str(Path.cwd()),
        key="surface_segregation_project_root",
    )
).expanduser().resolve()
config_path = project_root / "input.toml"
if not config_path.exists():
    st.error(f"No input.toml found at {config_path}")
    st.stop()

cfg = toml.load(str(config_path))
saved = dict(cfg.get("surface_segregation", {}) or {})
surface = dict(cfg.get("surface", {}) or {})
surface_refine = dict(surface.get("refine", {}) or {})
surface_screen = dict(surface.get("screen", {}) or {})
doping = dict(cfg.get("doping", {}) or {})
scan = dict(cfg.get("scan", {}) or {})
structure_cfg = dict(cfg.get("structure", {}) or {})


def _csv(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ", ".join(str(x) for x in (value or []))


def _items(text: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in text.split(",") if x.strip()))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError) as exc:
        st.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def _resolved_source_root(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _default_backend() -> dict[str, Any]:
    source = surface_refine if bool(surface_refine.get("enabled", False)) else surface_screen
    fallback = {
        "backend": "mace",
        "model": "small",
        "task": "",
        "device": "cpu",
        "gpu_id": 0,
        "tf_threads": 1,
        "omp_threads": 1,
    }
    fallback.update(source)
    fallback.update(saved)
    return fallback


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
    """Return backend-valid model/task defaults without carrying stale values."""
    backend = str(backend).strip().lower()

    if backend == "m3gnet":
        return "default", ""
    if backend == "grace":
        return "GRACE-1L-OMAT", ""
    if backend == "uma":
        return "uma-s-1p2", "omat"
    if backend == "mace":
        refine_backend = str(surface_refine.get("backend", "")).strip().lower()
        if refine_backend == "mace":
            model = str(surface_refine.get("model", "mh-1")).strip() or "mh-1"
            task = str(surface_refine.get("task", "")).strip()
            if model == "mh-1" and not task:
                task = "matpes_r2scan"
            return model, task
        return "mh-1", "matpes_r2scan"

    return "", ""


def _saved_backend_value(
    backend: str,
    key: str,
    fallback: str,
) -> str:
    """Reuse a saved model/task only when it belongs to the active backend."""
    saved_backend = str(saved.get("backend", "")).strip().lower()
    if saved_backend != str(backend).strip().lower():
        return fallback
    value = str(saved.get(key, fallback)).strip()
    return value if value else fallback


def _run_streaming(args: list[str]) -> int:
    st.code(" ".join(args))
    output_box = st.empty()
    lines: list[str] = []
    process = subprocess.Popen(
        args,
        cwd=str(project_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        output_box.code("\\n".join(lines[-35:]))
    returncode = process.wait()
    st.session_state["surface_segregation_log"] = "\\n".join(lines)
    if returncode == 0:
        st.success("Surface segregation MC finished successfully.")
    else:
        st.error(f"Surface segregation MC exited with return code {returncode}.")
    return returncode


with st.expander("Configuration & run controls", expanded=True):
    enabled = st.checkbox(
        "Enable surface segregation MC",
        value=bool(saved.get("enabled", False)),
    )

    st.subheader("Selected surface terminations")
    inherited_source_root = (
        str(saved.get("source_root", "")).strip()
        or str(surface.get("source_root", "")).strip()
        or str(structure_cfg.get("outdir", "random_structures")).strip()
    )
    s1, s2, s3 = st.columns(3)
    source_root = s1.text_input(
        "Parent / source root",
        value=inherited_source_root,
        help="Normally the same source root used by the Surface Screening stage.",
    ).strip()
    source_modes = [
        "auto",
        "final-selected",
        "refine-summary",
        "screen-selected",
        "screen-summary",
    ]
    saved_mode = str(saved.get("source_mode", "auto"))
    if saved_mode not in source_modes:
        saved_mode = "auto"
    source_mode = s2.selectbox(
        "Surface table",
        source_modes,
        index=source_modes.index(saved_mode),
    )
    outdir = s3.text_input(
        "Output directory",
        value=str(saved.get("outdir", "09_surface_segregation")),
        help="Relative paths are created inside Parent / source root.",
    ).strip()

    source_summary = st.text_input(
        "Explicit surface CSV (optional)",
        value=str(saved.get("source_summary", "")),
        help="Overrides Surface table. Relative paths are resolved inside Parent / source root.",
    ).strip()

    preview_cfg = dict(cfg)
    preview_section = dict(saved)
    preview_section.update(
        {
            "enabled": True,
            "source_root": source_root,
            "source_mode": source_mode,
            "source_summary": source_summary,
            "surface_include": [],
            "max_surfaces": 10000,
            "outdir": outdir,
        }
    )
    preview_cfg["surface_segregation"] = preview_section

    available = pd.DataFrame()
    try:
        available = preview_surface_segregation_targets(preview_cfg, project_root)
    except Exception as exc:
        st.warning(str(exc))

    saved_include = set(str(x) for x in saved.get("surface_include", []) or [])
    if not available.empty:
        table = available.copy()
        table["Include"] = (
            table["surface_id"].astype(str).isin(saved_include)
            if saved_include
            else False
        )
        cols = [
            "Include",
            "surface_id",
            "target_id",
            "miller",
            "termination_id",
            "termination_label",
            "source_stage",
        ]
        edited = st.data_editor(
            table[cols],
            use_container_width=True,
            hide_index=True,
            disabled=[
                "surface_id",
                "target_id",
                "miller",
                "termination_id",
                "termination_label",
                "source_stage",
            ],
            column_config={
                "Include": st.column_config.CheckboxColumn(
                    "Run MC",
                    help="Select this exact surface termination for segregation MC.",
                ),
                "termination_label": st.column_config.TextColumn(
                    "Natural dopant positions"
                ),
            },
            key="surface_segregation_surface_selector",
        )
        surface_include = (
            edited.loc[edited["Include"], "surface_id"].astype(str).tolist()
        )
        st.caption(
            f"Available: {len(available)} terminations · selected for MC: "
            f"{len(surface_include)}."
        )
    else:
        surface_include = list(saved_include)

    max_surfaces = int(
        st.number_input(
            "Maximum selected surfaces",
            min_value=1,
            value=max(
                1,
                int(saved.get("max_surfaces", max(10, len(surface_include) or 1))),
            ),
            step=1,
            help="Safety cap applied after the exact surface selection.",
        )
    )

    st.subheader("Cation sublattice and depth zones")
    z1, z2, z3, z4 = st.columns(4)
    host_species = z1.text_input(
        "Host cation",
        value=str(
            saved.get(
                "host_species",
                surface.get("host_species", doping.get("host_species", "Sn")),
            )
        ),
    ).strip()
    anion_species = _items(
        z2.text_input(
            "Anion species",
            value=_csv(
                saved.get(
                    "anion_species",
                    surface.get("anion_species", scan.get("anion_species", ["O"])),
                )
            ),
        )
    )
    dopant_species = _items(
        z3.text_input(
            "Dopant species",
            value=_csv(saved.get("dopant_species", [])),
            placeholder="In, Sb",
            help="Leave empty to infer every non-host/non-anion cation in the selected surface.",
        )
    )
    depth_layers = int(
        z4.number_input(
            "Cation layers / depth zone",
            min_value=1,
            value=int(
                saved.get(
                    "dopant_depth_layers",
                    surface.get("dopant_depth_layers", 1),
                )
            ),
            step=1,
        )
    )
    layer_tolerance = float(
        st.number_input(
            "Cation-layer tolerance (Å)",
            min_value=0.01,
            value=float(
                saved.get(
                    "cation_layer_tolerance_A",
                    surface.get("cation_layer_tolerance_A", 0.8),
                )
            ),
            step=0.05,
        )
    )

    st.subheader("Monte Carlo")
    m1, m2, m3, m4 = st.columns(4)
    temperature = float(
        m1.number_input(
            "Temperature (K)",
            min_value=1.0,
            value=float(saved.get("temperature_K", 800.0)),
            step=50.0,
        )
    )
    steps = int(
        m2.number_input(
            "MC steps",
            min_value=2,
            value=int(saved.get("steps", 10000)),
            step=1000,
        )
    )
    burn_in = int(
        m3.number_input(
            "Burn-in steps",
            min_value=0,
            max_value=max(0, steps - 1),
            value=min(int(saved.get("burn_in", 2000)), max(0, steps - 1)),
            step=500,
        )
    )
    sample_interval = int(
        m4.number_input(
            "Sample every N steps",
            min_value=1,
            value=int(saved.get("sample_interval", 20)),
            step=5,
        )
    )

    m5, m6, m7 = st.columns(3)
    trace_interval = int(
        m5.number_input(
            "Energy trace interval",
            min_value=1,
            value=int(saved.get("trace_interval", max(20, sample_interval))),
            step=10,
        )
    )
    progress_interval = int(
        m6.number_input(
            "Progress print interval",
            min_value=1,
            value=int(saved.get("progress_interval", 500)),
            step=100,
            help="The GUI streams these progress messages while MC is running.",
        )
    )
    seed = int(
        m7.number_input(
            "Random seed",
            value=int(saved.get("seed", 42)),
            step=1,
        )
    )

    st.subheader("ML force field")
    defaults = _default_backend()
    b1, b2, b3, b4 = st.columns(4)
    backend_options = list(BACKEND_CHOICES)
    backend_default = str(defaults.get("backend", "mace")).lower()
    if backend_default not in backend_options:
        backend_default = "mace"
    backend = b1.selectbox(
        "Backend",
        backend_options,
        index=backend_options.index(backend_default),
        key="surface_segregation_backend",
    )

    backend_model_default, backend_task_default = _backend_model_defaults(backend)

    if backend == "m3gnet":
        model = "default"
        b2.text_input(
            "Model",
            value="default",
            disabled=True,
            key="surface_segregation_model_m3gnet",
            help="M3GNet uses its default pretrained potential in this workflow.",
        )
        task = ""
        b3.text_input(
            "Task / head",
            value="not used",
            disabled=True,
            key="surface_segregation_task_m3gnet",
        )

    elif backend == "grace":
        current_model = _saved_backend_value(
            backend,
            "model",
            backend_model_default,
        )
        if current_model not in GRACE_MODEL_CHOICES:
            current_model = "GRACE-1L-OMAT"
        model = b2.selectbox(
            "GRACE model",
            list(GRACE_MODEL_CHOICES),
            index=list(GRACE_MODEL_CHOICES).index(current_model),
            key="surface_segregation_model_grace",
        )
        task = ""
        b3.text_input(
            "Task / head",
            value="not used",
            disabled=True,
            key="surface_segregation_task_grace",
            help="GRACE model selection already defines the potential; no separate task/head is used.",
        )

    elif backend == "uma":
        current_model = _saved_backend_value(
            backend,
            "model",
            backend_model_default,
        )
        if current_model not in UMA_MODEL_CHOICES:
            current_model = "uma-s-1p2"
        model = b2.selectbox(
            "UMA model",
            list(UMA_MODEL_CHOICES),
            index=list(UMA_MODEL_CHOICES).index(current_model),
            key="surface_segregation_model_uma",
        )

        current_task = _saved_backend_value(
            backend,
            "task",
            backend_task_default,
        )
        if current_task not in UMA_TASK_CHOICES:
            current_task = "omat"
        task = b3.selectbox(
            "UMA task",
            list(UMA_TASK_CHOICES),
            index=list(UMA_TASK_CHOICES).index(current_task),
            key="surface_segregation_task_uma",
        )

    else:  # MACE
        mace_models = list(_available_mace_models())
        custom_label = "Custom checkpoint path…"
        current_model = _saved_backend_value(
            backend,
            "model",
            backend_model_default,
        )
        if current_model in mace_models:
            initial_model_choice = current_model
        else:
            initial_model_choice = custom_label

        selected_model = b2.selectbox(
            "MACE model",
            [*mace_models, custom_label],
            index=[*mace_models, custom_label].index(initial_model_choice),
            key="surface_segregation_model_mace_choice",
        )
        if selected_model == custom_label:
            model = b2.text_input(
                "Checkpoint path",
                value=(
                    current_model
                    if current_model not in mace_models
                    else ""
                ),
                key="surface_segregation_model_mace_custom",
                help="Path to a local MACE checkpoint.",
            ).strip()
        else:
            model = selected_model

        saved_mace_model = (
            str(saved.get("model", "")).strip()
            if str(saved.get("backend", "")).strip().lower() == "mace"
            else ""
        )
        if model == saved_mace_model:
            mace_task_default = str(saved.get("task", backend_task_default)).strip()
        elif model == "mh-1":
            mace_task_default = "matpes_r2scan"
        else:
            mace_task_default = ""

        is_multihead_or_custom = (
            model.startswith("mh-")
            or selected_model == custom_label
        )
        if is_multihead_or_custom:
            task = b3.text_input(
                "MACE head",
                value=mace_task_default,
                key=f"surface_segregation_task_mace_{model}",
                help=(
                    "Head name for a multi-head MACE checkpoint. "
                    "For MH-1, matpes_r2scan is the default used here. "
                    "Leave blank only when the selected checkpoint does not require a head."
                ),
            ).strip()
        else:
            task = ""
            b3.text_input(
                "MACE head",
                value="not used",
                disabled=True,
                key=f"surface_segregation_task_mace_single_{model}",
                help="This selected MACE model is treated as a single-head model.",
            )

    device_options = list(DEVICE_CHOICES)
    device_default = str(defaults.get("device", "cpu")).lower()
    if device_default not in device_options:
        device_default = "cpu"
    device = b4.selectbox(
        "Device",
        device_options,
        index=device_options.index(device_default),
        key="surface_segregation_device",
    )

    st.caption(
        {
            "m3gnet": "M3GNet: default pretrained model; no task/head.",
            "grace": "GRACE: choose a GRACE checkpoint; no separate task/head.",
            "uma": "UMA: choose both a foundation model and its task.",
            "mace": "MACE: choose a packaged model or custom checkpoint; a head is requested only for multi-head/custom checkpoints.",
        }[backend]
    )

    t1, t2, t3, t4 = st.columns(4)
    gpu_id = int(
        t1.number_input(
            "GPU ID",
            min_value=0,
            value=int(defaults.get("gpu_id", 0)),
            step=1,
            disabled=device != "cuda",
        )
    )
    tf_threads = int(
        t2.number_input(
            "TensorFlow threads",
            min_value=1,
            value=int(saved.get("tf_threads", defaults.get("tf_threads", 1))),
            step=1,
        )
    )
    omp_threads = int(
        t3.number_input(
            "OpenMP threads / surface",
            min_value=1,
            value=int(saved.get("omp_threads", defaults.get("omp_threads", 1))),
            step=1,
        )
    )
    parallel_surfaces = int(
        t4.number_input(
            "Parallel surfaces",
            min_value=1,
            value=int(saved.get("parallel_surfaces", 1)),
            step=1,
            help=(
                "Independent surfaces can run in parallel on CPU. One MC chain itself "
                "is sequential. CUDA currently uses one surface at a time."
            ),
        )
    )
    if device == "cuda" and parallel_surfaces > 1:
        st.warning("Set Parallel surfaces = 1 for CUDA.")

    resolved = dict(saved)
    resolved.update(
        {
            "enabled": bool(enabled),
            "source_root": source_root,
            "source_mode": source_mode,
            "source_summary": source_summary,
            "outdir": outdir,
            "surface_include": surface_include,
            "max_surfaces": max_surfaces,
            "host_species": host_species,
            "anion_species": anion_species,
            "dopant_species": dopant_species,
            "cation_layer_tolerance_A": layer_tolerance,
            "dopant_depth_layers": depth_layers,
            "temperature_K": temperature,
            "steps": steps,
            "burn_in": burn_in,
            "sample_interval": sample_interval,
            "trace_interval": trace_interval,
            "progress_interval": progress_interval,
            "seed": seed,
            "backend": backend,
            "model": model,
            "task": task,
            "device": device,
            "gpu_id": gpu_id,
            "tf_threads": tf_threads,
            "omp_threads": omp_threads,
            "parallel_surfaces": parallel_surfaces,
        }
    )
    resolved_cfg = dict(cfg)
    resolved_cfg["surface_segregation"] = resolved

    st.markdown("##### Configuration preview")
    st.code(toml.dumps({"surface_segregation": resolved}), language="toml")

    valid = True
    try:
        parse_surface_segregation_config(resolved_cfg, project_root)
        source_path, resolved_mode = resolve_surface_segregation_source_summary(
            resolved_cfg,
            project_root,
        )
        output_path = resolve_surface_segregation_output_dir(
            resolved_cfg,
            project_root,
        )
        st.caption(
            f"Resolved source: {source_path} ({resolved_mode}) · output: {output_path}"
        )
    except Exception as exc:
        valid = False
        st.error(str(exc))

    c1, c2, c3 = st.columns(3)
    if c1.button("Save to input.toml", disabled=not valid):
        cfg["surface_segregation"] = resolved
        config_path.write_text(toml.dumps(cfg), encoding="utf-8")
        st.success("Saved [surface_segregation] to input.toml.")

    if c2.button("Dry-run / write MC plan", disabled=not valid):
        cfg["surface_segregation"] = resolved
        config_path.write_text(toml.dumps(cfg), encoding="utf-8")
        _run_streaming(
            [
                "dopingflow",
                "surface-segregation",
                "-c",
                str(config_path),
                "--dry-run",
            ]
        )

    run_disabled = (
        not valid
        or not enabled
        or not surface_include
        or (device == "cuda" and parallel_surfaces > 1)
    )
    if c3.button("Run segregation MC", disabled=run_disabled):
        cfg["surface_segregation"] = resolved
        config_path.write_text(toml.dumps(cfg), encoding="utf-8")
        _run_streaming(
            [
                "dopingflow",
                "surface-segregation",
                "-c",
                str(config_path),
            ]
        )

    if not surface_include:
        st.info("Select at least one exact surface termination above before running MC.")

st.divider()
st.header("Results explorer")

try:
    result_root = resolve_surface_segregation_output_dir(cfg, project_root)
except Exception:
    result_root = _resolved_source_root(
        str(saved.get("source_root", inherited_source_root))
    ) / str(saved.get("outdir", "09_surface_segregation"))

summary = _read_csv(result_root / "surface_segregation_summary.csv")
if summary.empty:
    st.info(
        "No completed surface-segregation results found yet. Configure and run the stage above."
    )
    st.stop()

surface_options = summary["surface_id"].astype(str).tolist()
selected_surface = st.selectbox("Surface termination", surface_options)
row = summary[summary["surface_id"].astype(str) == selected_surface].iloc[0]

r1, r2, r3, r4 = st.columns(4)
r1.metric("Temperature", f"{float(row['temperature_K']):.0f} K")
r2.metric("Production samples", int(row["n_samples"]))
r3.metric("Acceptance", f"{100.0 * float(row['acceptance_fraction']):.1f}%")
r4.metric(
    "Best ΔE",
    f"{float(row['best_energy_eV']) - float(row['start_energy_eV']):+.4f} eV",
)
st.caption(
    f"Facet ({int(row['miller_h'])}{int(row['miller_k'])}{int(row['miller_l'])}) · "
    f"termination {int(row['termination_id'])} · "
    f"{row.get('termination_label', '')}"
)

safe_surface = "".join(
    ch if ch.isalnum() or ch in "._-" else "_"
    for ch in selected_surface.replace("/", "__")
)
surface_dir = result_root / "surfaces" / safe_surface
site_df = _read_csv(surface_dir / "site_occupancy.csv")
zone_df = _read_csv(surface_dir / "zone_occupancy.csv")
trace_df = _read_csv(surface_dir / "mc_trace.csv")
zone_trace_df = _read_csv(surface_dir / "zone_trace.csv")
swap_df = _read_csv(surface_dir / "swap_statistics.csv")

tab_zone, tab_site, tab_conv, tab_raw = st.tabs(
    ["Depth-zone preference", "Site heat map", "MC convergence", "Raw data"]
)

with tab_zone:
    st.subheader("Surface / subsurface / bulk occupancy")
    st.caption(
        "Mean site occupancy is normalized by the number of cation sites in each depth zone. "
        "This avoids falsely favoring the bulk simply because it contains more sites."
    )
    if not zone_df.empty:
        zone_order = ["surface", "subsurface", "bulk"]
        heat = zone_df.pivot(
            index="dopant",
            columns="zone",
            values="mean_site_occupancy",
        ).reindex(columns=zone_order)
        fig = px.imshow(
            heat,
            text_auto=".3f",
            aspect="auto",
            labels={
                "x": "Depth zone",
                "y": "Dopant",
                "color": "Mean site occupancy",
            },
            title="Normalized dopant occupancy by depth zone",
        )
        st.plotly_chart(fig, use_container_width=True)

        dg = zone_df[
            zone_df["zone"].isin(["surface", "subsurface"])
            & zone_df["delta_G_eff_vs_bulk_eV"].notna()
        ].copy()
        if not dg.empty:
            fig_dg = px.bar(
                dg,
                x="zone",
                y="delta_G_eff_vs_bulk_eV",
                color="dopant",
                barmode="group",
                labels={
                    "zone": "Depth zone",
                    "delta_G_eff_vs_bulk_eV": "Effective ΔGseg vs bulk (eV)",
                    "dopant": "Dopant",
                },
                title="Occupancy-derived effective segregation free energy",
            )
            fig_dg.add_hline(y=0.0, line_dash="dash")
            st.plotly_chart(fig_dg, use_container_width=True)
            st.caption(
                "Negative ΔG means that zone is enriched relative to bulk at the sampled "
                "temperature. This is an occupancy-derived effective PMF/free-energy preference, "
                "not a zero-temperature two-structure segregation energy."
            )

with tab_site:
    st.subheader("Which cation sites does each dopant visit?")
    if not site_df.empty:
        site_df = site_df.copy()
        site_df["site_label"] = site_df.apply(
            lambda x: f"{int(x['site_index'])} ({x['zone']})",
            axis=1,
        )
        pivot = site_df.pivot(
            index="dopant",
            columns="site_label",
            values="occupancy_probability",
        )
        fig = px.imshow(
            pivot,
            text_auto=".2f",
            aspect="auto",
            labels={
                "x": "Cation site index (zone)",
                "y": "Dopant",
                "color": "Occupancy probability",
            },
            title="Site-resolved dopant occupancy heat map",
        )
        st.plotly_chart(fig, use_container_width=True)

        pmf = site_df[site_df["site_pmf_eV_vs_most_occupied"].notna()].copy()
        if not pmf.empty:
            fig_pmf = px.scatter(
                pmf,
                x="site_index",
                y="site_pmf_eV_vs_most_occupied",
                color="dopant",
                symbol="zone",
                labels={
                    "site_index": "Cation site index",
                    "site_pmf_eV_vs_most_occupied": "Site PMF vs preferred site (eV)",
                },
                title="Site-resolved effective free-energy landscape",
            )
            st.plotly_chart(fig_pmf, use_container_width=True)

with tab_conv:
    st.subheader("Sampling diagnostics")
    if not trace_df.empty:
        fig = px.line(
            trace_df,
            x="step",
            y=["energy_eV", "best_energy_eV"],
            labels={"value": "Energy (eV)", "step": "MC step", "variable": "Trace"},
            title="MC energy trace",
        )
        st.plotly_chart(fig, use_container_width=True)
    if not swap_df.empty:
        st.dataframe(swap_df, use_container_width=True, hide_index=True)
    if not zone_trace_df.empty:
        fig_zone = px.line(
            zone_trace_df,
            x="step",
            y="count",
            color="dopant",
            line_dash="zone",
            labels={"count": "Dopant atoms in zone", "step": "MC step"},
            title="Depth-zone occupancy during production sampling",
        )
        st.plotly_chart(fig_zone, use_container_width=True)

with tab_raw:
    st.markdown("**Zone occupancy**")
    st.dataframe(zone_df, use_container_width=True, hide_index=True)
    st.markdown("**Site occupancy**")
    st.dataframe(site_df, use_container_width=True, hide_index=True)
    st.markdown("**MC trace**")
    st.dataframe(trace_df, use_container_width=True, hide_index=True)

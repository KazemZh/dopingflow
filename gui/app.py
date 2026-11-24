Warning: truncated output (original token count: 51360)
Total output lines: 4794

# gui/app.py
from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path
import json

import pandas as pd
import streamlit as st
import toml
import re


# Make sure gui/ is importable
GUI_DIR = Path(__file__).resolve().parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from gui_config import (
    SUPER_CELL_PRESETS,
    DOPING_MODE_CHOICES,
    ALLOWED_DOPANTS,
    RUN_PRESETS,
    DEFAULTS,
    CHOICES,
    MACE_MODEL_CHOICES,
    UMA_MODEL_CHOICES,
    UMA_TASK_CHOICES,
    GRACE_MODEL_CHOICES,
)
from io_project import ProjectIndex
from view_structure import show_structure, structure_viewer_controls


st.set_page_config(page_title="dopingflow GUI", layout="wide")

def sanitize_filename(name: str) -> str:
    """
    Make a safe filename by replacing problematic characters with underscores.
    """
    name = name.strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name or "output"

def load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    return toml.load(str(path))


def save_toml(path: Path, cfg: dict) -> None:
    path.write_text(toml.dumps(cfg))


@st.cache_data(show_spinner=False)
def available_mace_models() -> tuple[str, ...]:
    """Use the installed MACE catalogue, with GUI defaults as an offline fallback."""
    try:
        from dopingflow.ml_backends import get_mace_model_choices

        return get_mace_model_choices()
    except Exception:
        return tuple(MACE_MODEL_CHOICES)


def mace_model_input(label: str, current_model: str, *, key: str) -> str:
    """Render a shared MACE alias selector with a custom-checkpoint escape hatch."""
    current_model = str(current_model).strip()
    if current_model in {"", "default"}:
        current_model = "small"

    model_choices = list(available_mace_models())
    custom_label = "Custom checkpoint path…"
    choices = [*model_choices, custom_label]
    selected = current_model if current_model in model_choices else custom_label
    selected = st.selectbox(
        label,
        options=choices,
        index=choices.index(selected),
        key=key,
        help=(
            "Select a mace_mp alias. MACE downloads its weights on first use, or "
            "choose a local .model/.pt/.pth checkpoint path."
        ),
    )
    if selected != custom_label:
        return selected

    custom_model = st.text_input(
        f"{label} checkpoint path",
        value=current_model if current_model not in model_choices else "",
        key=f"{key}_custom",
        placeholder="/path/to/model.model",
    ).strip()
    if not custom_model:
        st.warning("Enter a MACE checkpoint path before saving or running.")
        return current_model
    return custom_model


def mace_head_input(label: str, current_task: str, *, key: str) -> str:
    return st.text_input(
        label,
        value=str(current_task).strip(),
        key=key,
        placeholder="Optional, e.g. omat_pbe for mh-1",
        help=(
            "Optional MACE multi-head model head. Leave blank to use a compatible "
            "dopingflow default; mh-1 resolves to omat_pbe. Stored in the existing "
            "task parameter."
        ),
    ).strip()


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n=== RUN {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return p.wait()


def tail_text(path: Path, n: int = 6000) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(errors="replace")
    return txt[-n:]


def file_browser(
    root: Path,
    label: str,
    exts: tuple[str, ...] = (".POSCAR", ".vasp", ".VASP"),
    key: str = "fb",
) -> str | None:
    """
    Simple in-app file picker rooted at `root`.
    Returns a relative path (string) or None.
    """
    root = root.resolve()
    state_key = f"{key}_cwd"
    if state_key not in st.session_state:
        st.session_state[state_key] = str(root)

    cwd = Path(st.session_state[state_key]).resolve()
    if root not in cwd.parents and cwd != root:
        cwd = root
        st.session_state[state_key] = str(root)

    st.markdown(f"**{label}**")
    st.caption(f"Browsing: `{cwd.relative_to(root) if cwd != root else '.'}`")

    colA, colB = st.columns([1, 3], vertical_alignment="center")
    with colA:
        if st.button("⬆ Up", key=f"{key}_up", disabled=(cwd == root)):
            st.session_state[state_key] = str(cwd.parent)
            st.rerun()

    # List directories + candidate files
    dirs = sorted([p for p in cwd.iterdir() if p.is_dir()])
    files = sorted(
        [p for p in cwd.iterdir() if p.is_file() and (p.suffix in exts or p.name == "POSCAR")]
    )

    # Directories selector
    dir_options = ["(stay here)"] + [d.name + "/" for d in dirs]
    with colB:
        chosen_dir = st.selectbox("Folders", dir_options, key=f"{key}_dirs")

    if chosen_dir != "(stay here)":
        target = cwd / chosen_dir.rstrip("/")
        st.session_state[state_key] = str(target)
        st.rerun()

    # Files selector
    file_options = ["(select a file)"] + [f.name for f in files]
    chosen_file = st.selectbox("Files", file_options, key=f"{key}_files")

    if chosen_file != "(select a file)":
        picked = (cwd / chosen_file).resolve()
        rel = str(picked.relative_to(root))
        st.success(f"Selected: `{rel}`")
        return rel

    return None


# Full periodic table symbols (1–118)
PERIODIC_TABLE = [
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
    "Nh","Fl","Mc","Lv","Ts","Og",
]

def normalize_symbol(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s[0].upper() + s[1:].lower()


# -----------------------
# Sidebar: project paths
# -----------------------
st.sidebar.title("Project")
project_root_str = st.sidebar.text_input("Project root", value=str(Path.cwd()))
project_root = Path(project_root_str).expanduser().resolve()

input_toml_path = project_root / "input.toml"
cfg = load_toml(input_toml_path)

# defaults
cfg.setdefault("structure", {})
cfg.setdefault("doping", {})
cfg.setdefault("generate", {})

outdir = cfg["structure"].get("outdir", "random_structures")
proj = ProjectIndex(root=project_root, outdir=project_root / outdir)

tab = st.sidebar.radio("Pages", ["Input Builder", "Run", "Results Explorer", "Structure Viewer"])


# -----------------------
# Page: Input Builder
# -----------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (does not mutate inputs)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure_config_defaults(cfg: dict) -> dict:
    cfg = cfg or {}
    merged = deep_merge(DEFAULTS, cfg)
    # Ensure missing top-level sections exist
    for sec in DEFAULTS:
        merged.setdefault(sec, dict(DEFAULTS[sec]))
    return merged


def normalize_compositions_list(val):
    # Support both formats:
    # 1) compositions = [ {Sb=5}, {Sb=5, Zr=5} ]
    # 2) [[doping.compositions]] blocks (toml library can load either way)
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return []


if tab == "Input Builder":
    st.title("Input Builder (input.toml)")

    cfg_raw = load_toml(input_toml_path)
    cfg_full = ensure_config_defaults(cfg_raw)

    # convenience
    sec = cfg_full

    # Reset button
    colR1, colR2 = st.columns([1, 1])
    with colR1:
        if st.button("↩ Reset to defaults"):
            cfg_full = ensure_config_defaults({})
            st.session_state["cfg_edit"] = cfg_full
            st.success("Reset done. Now click Save.")
    with colR2:
        st.caption("Tip: use expanders and Save at the bottom.")

    # Persist edited config in session_state
    if "cfg_edit" not in st.session_state:
        st.session_state["cfg_edit"] = cfg_full
    cfg_edit = st.session_state["cfg_edit"]
    # ===== PATCH: ensure section dicts exist =====
    for secname in [
        "hardware",
        "references",
        "structure",
        "doping",
        "generate",
        "sequential",
        "scan",
        "relax",
        "filter",
        "bandgap",
        "formation",
        "energy_correction",
        "database",
        "phase_diagram",
        "vacancies",
        "surface",
    ]:
        cfg_edit.setdefault(secname, {})
    # ===== END PATCH =====    

    # -----------------------------
    # STRUCTURE
    # -----------------------------
    
    with st.expander("Structure", expanded=True):
        st.subheader("Output")

        outdir_ui = st.text_input(
            "Output directory",
            value=cfg_edit["structure"].get("outdir", "random_structures"),
            help="Folder where generated structures will be written.",
        )

        cfg_edit["structure"]["outdir"] = outdir_ui


    # -----------------------------
    # REFERENCES (Step 00)
    # -----------------------------
    with st.expander("References (refs-build)", expanded=False):
        cfg_edit.setdefault("references", {})

        st.subheader("Common settings")

        # -----------------------------
        # Basic execution controls
        # -----------------------------
        colA1, colA2 = st.columns(2)

        with colA1:
            cfg_edit["references"]["skip_if_done"] = st.checkbox(
                "skip_if_done",
                value=bool(cfg_edit["references"].get("skip_if_done", True)),
                help="Skip refs-build if reference_energies.json already exists.",
            )

        with colA2:
            current_mode = str(
                cfg_edit["references"].get("reference_mode", "metal")
            ).strip().lower()
            if current_mode not in {"metal", "oxide"}:
                current_mode = "metal"

            cfg_edit["references"]["reference_mode"] = st.radio(
                "Reference mode",
                options=["metal", "oxide"],
                index=0 if current_mode == "metal" else 1,
                horizontal=True,
                help="Choose the reference construction scheme used for formation energies.",
            )

        colB1, colB2, colB3 = st.columns(3)

        with colB1:
            cfg_edit["references"]["fmax"] = st.number_input(
                "fmax",
                value=float(cfg_edit["references"].get("fmax", 0.02)),
                min_value=0.001,
                step=0.001,
                format="%.3f",
                help="Force convergence criterion in eV/Å.",
            )

        with colB2:
            cfg_edit["references"]["max_steps"] = st.number_input(
                "max_steps",
                value=int(cfg_edit["references"].get("max_steps", 300)),
                min_value=1,
                step=10,
                help="Maximum number of optimizer steps for each reference relaxation.",
            )

        with colB3:
            optimizer_choices = ["bfgs", "lbfgs", "fire", "mdmin", "quasinewton"]
            current_optimizer = str(
                cfg_edit["references"].get("optimizer", "bfgs")
            ).strip().lower()
            if current_optimizer not in optimizer_choices:
                current_optimizer = "bfgs"

            cfg_edit["references"]["optimizer"] = st.selectbox(
                "optimizer",
                options=optimizer_choices,
                index=optimizer_choices.index(current_optimizer),
                help="ASE optimizer used to relax host and reference structures.",
            )

        st.divider()
        st.subheader("Backend and device")

        backend_choices = ["m3gnet", "uma", "mace", "grace"]
        current_backend = str(
            cfg_edit["references"].get("backend", "m3gnet")
        ).strip().lower()
        if current_backend not in backend_choices:
            current_backend = "m3gnet"

        colC1, colC2, colC3 = st.columns(3)

        with colC1:
            cfg_edit["references"]["backend"] = st.selectbox(
                "backend",
                options=backend_choices,
                index=backend_choices.index(current_backend),
                help="ML backend used for refs-build structural relaxations.",
            )

        with colC2:
            device_choices = ["cpu", "cuda"]
            current_device = str(
                cfg_edit["references"].get("device", "cpu")
            ).strip().lower()
            if current_device not in device_choices:
                current_device = "cpu"

            cfg_edit["references"]["device"] = st.selectbox(
                "device",
                options=device_choices,
                index=device_choices.index(current_device),
                help="Execution device for the selected backend.",
            )

        with colC3:
            cfg_edit["references"]["gpu_id"] = st.number_input(
                "gpu_id",
                value=int(cfg_edit["references"].get("gpu_id", 0)),
                min_value=0,
                step=1,
                help="GPU index used when device='cuda'.",
            )

        selected_backend = cfg_edit["references"]["backend"]

        # Backend-dependent model/task controls
        uma_models = ["uma-s-1p2", "uma-s-1p1", "uma-m-1p1"]
        uma_tasks = ["omat", "oc20", "oc22", "oc25", "omol", "odac", "omc"]

        grace_models = [
            "GRACE-1L-OMAT",
            "GRACE-1L-OMAT-M-base",
            "GRACE-1L-OMAT-M",
            "GRACE-1L-OMAT-L-base",
            "GRACE-1L-OMAT-L",
            "GRACE-2L-OMAT",
            "GRACE-2L-OMAT-M-base",
            "GRACE-2L-OMAT-M",
            "GRACE-2L-OMAT-L-base",
            "GRACE-2L-OMAT-L",
            "GRACE-1L-OAM",
            "GRACE-1L-OAM-M",
            "GRACE-1L-OAM-L",
            "GRACE-2L-OAM",
            "GRACE-2L-OAM-M",
            "GRACE-2L-OAM-L",
            "GRACE-1L-SMAX-L",
            "GRACE-1L-SMAX-OMAT-L",
            "GRACE-2L-SMAX-M",
            "GRACE-2L-SMAX-L",
            "GRACE-2L-SMAX-OMAT-M",
            "GRACE-2L-SMAX-OMAT-L",
        ]

        colD1, colD2 = st.columns(2)

        with colD1:
            if selected_backend == "m3gnet":
                cfg_edit["references"]["model"] = st.text_input(
                    "model",
                    value=str(cfg_edit["references"].get("model", "default")),
                    disabled=True,
                    help="M3GNet currently uses the default model.",
                )
                cfg_edit["references"]["task"] = ""

            elif selected_backend == "uma":
                current_model = str(
                    cfg_edit["references"].get("model", "uma-s-1p2")
                ).strip()
                if current_model not in uma_models:
                    current_model = "uma-s-1p2"

                cfg_edit["references"]["model"] = st.selectbox(
                    "model",
                    options=uma_models,
                    index=uma_models.index(current_model),
                    help="Pretrained UMA model.",
                )

            elif selected_backend == "mace":
                cfg_edit["references"]["model"] = mace_model_input(
                    "model",
                    cfg_edit["references"].get("model", "small"),
                    key="references_model_mace",
                )

            elif selected_backend == "grace":
                current_model = str(
                    cfg_edit["references"].get("model", "GRACE-1L-OMAT")
                ).strip()
                if current_model not in grace_models:
                    current_model = "GRACE-1L-OMAT"

                cfg_edit["references"]["model"] = st.selectbox(
                    "model",
                    options=grace_models,
                    index=grace_models.index(current_model),
                    help="Pretrained GRACE model.",
                )
                cfg_edit["references"]["task"] = ""

        with colD2:
            if selected_backend == "uma":
                current_task = str(
                    cfg_edit["references"].get("task", "omat")
                ).strip()
                if current_task not in uma_tasks:
                    current_task = "omat"

                cfg_edit["references"]["task"] = st.selectbox(
                    "task",
                    options=uma_tasks,
                    index=uma_tasks.index(current_task),
                    help="Task family used by the UMA calculator.",
                )
            elif selected_backend == "mace":
                cfg_edit["references"]["task"] = mace_head_input(
                    "MACE head (optional)",
                    cfg_edit["references"].get("task", ""),
                    key="references_task_mace",
                )
            else:
                cfg_edit["references"]["task"] = st.text_input(
                    "task",
                    value="",
                    disabled=True,
                    help="Task is used by UMA and as an optional MACE head.",
                )

        colE1, colE2 = st.columns(2)

        with colE1:
            cfg_edit["references"]["tf_threads"] = st.number_input(
                "tf_threads",
                value=int(cfg_edit["references"].get("tf_threads", 1)),
                min_value=1,
                step=1,
                help="TensorFlow intra/inter-op thread count.",
            )

        with colE2:
            cfg_edit["references"]["omp_threads"] = st.number_input(
                "omp_threads",
                value=int(cfg_edit["references"].get("omp_threads", 1)),
                min_value=1,
                step=1,
                help="OMP thread count.",
            )

        st.divider()
        st.subheader("Host structure")

        cfg_edit["references"]["host"] = st.text_input(
            "host",
            value=str(cfg_edit["references"].get("host", "SnO2")),
            help='Host formula, e.g. "SnO2".',
        )

        cfg_edit["references"]["host_dir"] = st.text_input(
            "host_dir",
            value=str(cfg_edit["references"].get("host_dir", "reference_structures/oxides")),
            help="Directory containing <host>.POSCAR.",
        )

        sc = cfg_edit["references"].get("supercell", [5, 2, 1])
        if not isinstance(sc, list) or len(sc) != 3:
            sc = [5, 2, 1]

        c1, c2, c3 = st.columns(3)
        with c1:
            sx = st.number_input("supercell nx", min_value=1, value=int(sc[0]), step=1)
        with c2:
            sy = st.number_input("supercell ny", min_value=1, value=int(sc[1]), step=1)
        with c3:
            sz = st.number_input("supercell nz", min_value=1, value=int(sc[2]), step=1)
        cfg_edit["references"]["supercell"] = [int(sx), int(sy), int(sz)]

        st.divider()

        chosen_mode = cfg_edit["references"]["reference_mode"]

        if chosen_mode == "metal":
            st.subheader("Metal references")

            cfg_e…41360 tokens truncated…

        st.caption(f"Selected structures: **{len(plot_df):,}**")
        if plot_df.empty:
            st.warning("No matching structures.")
            st.stop()

    else:
        st.subheader("Generic plot studio")
        st.info("This dataset does not contain dopant_counts_json or dopant_counts. Using generic plotting mode.")
        plot_df = df_f.copy()

    # ============================================================
    # Plot builder with FULL hover
    # ============================================================
    st.divider()
    st.subheader("Plot")

    numeric_cols_sel = [c for c in plot_df.columns if pd.api.types.is_numeric_dtype(plot_df[c])]
    if not numeric_cols_sel:
        st.error("No numeric columns available to plot.")
        st.stop()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        x_col = st.selectbox("X axis", options=numeric_cols_sel, index=0)
    with col2:
        y_col = st.selectbox("Y axis", options=numeric_cols_sel, index=min(1, len(numeric_cols_sel) - 1))
    with col3:
        preferred_color_cols = [c for c in ["dopant_sig", "n_dopants", "composition_tag", "candidate", "surface_energy_status"] if c in plot_df.columns]
        other_color_cols = [c for c in plot_df.columns if c not in numeric_cols_sel and c not in preferred_color_cols]
        color_options = preferred_color_cols + other_color_cols
        if not color_options:
            color_options = ["(none)"]

    color_by = st.selectbox("Color by", options=color_options, index=0)
    # Appearance
    with st.expander("Appearance & axes", expanded=True):
        cA, cB, cC = st.columns([1.2, 1, 1])
        with cA:
            template = st.selectbox("Template", ["plotly_white", "plotly_simple_white", "plotly", "ggplot2"], index=0)
            opacity = st.slider("Opacity", 0.05, 1.0, 0.85, 0.05)
        with cB:
            x_min = st.text_input("X min (optional)", value="")
            x_max = st.text_input("X max (optional)", value="")
        with cC:
            y_min = st.text_input("Y min (optional)", value="")
            y_max = st.text_input("Y max (optional)", value="")

        x_min_v = to_float_or_none(x_min)
        x_max_v = to_float_or_none(x_max)
        y_min_v = to_float_or_none(y_min)
        y_max_v = to_float_or_none(y_max)

    plot_limit = 20_000
    if len(plot_df) > plot_limit:
        plot_render_df = plot_df.sample(plot_limit, random_state=42).sort_index()
        st.info(
            f"Plotting a reproducible sample of {plot_limit:,} from "
            f"{len(plot_df):,} selected rows. Apply filters for an unsampled plot."
        )
    else:
        plot_render_df = plot_df

    # FULL hover: include all columns EXCEPT non-serializable/internal ones
    internal_cols = {"_dopant_dict"}  # <- dict is serializable, but can get large; keep it out of hover
    hover_cols = [c for c in plot_render_df.columns.tolist() if c not in internal_cols]

    # Decide which column to use as hover title
    hover_name_col = None
    for c in ["candidate", "composition_tag"]:
        if c in plot_render_df.columns:
            hover_name_col = c
            break

    # Build scatter plot safely
    if color_by == "(none)":
        fig = px.scatter(
            plot_render_df,
            x=x_col,
            y=y_col,
            hover_name=hover_name_col,
            hover_data=hover_cols,
            labels={x_col: x_col, y_col: y_col},
        )
    else:
        fig = px.scatter(
            plot_render_df,
            x=x_col,
            y=y_col,
            color=color_by,
            hover_name=hover_name_col,
            hover_data=hover_cols,
            labels={x_col: x_col, y_col: y_col},
        )
    fig.update_layout(template=template, height=700)
    fig.update_traces(opacity=float(opacity), marker=dict(size=12))

    if x_min_v is not None or x_max_v is not None:
        fig.update_xaxes(range=[x_min_v if x_min_v is not None else None, x_max_v if x_max_v is not None else None], autorange=False)
    if y_min_v is not None or y_max_v is not None:
        fig.update_yaxes(range=[y_min_v if y_min_v is not None else None, y_max_v if y_max_v is not None else None], autorange=False)

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # Export
    # ============================================================
    st.divider()
    st.subheader("Export plot")

    base = sanitize_filename(st.text_input("Base filename", value="dopant_plot"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cE1, cE2, cE3, cE4 = st.columns(4)

    with cE1:
        st.download_button(
            "⬇ JSON",
            data=fig.to_json().encode("utf-8"),
            file_name=f"{base}_{ts}.json",
            mime="application/json",
        )

    with cE2:
        st.download_button(
            "⬇ HTML",
            data=fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
            file_name=f"{base}_{ts}.html",
            mime="text/html",
        )

    export_scale = st.number_input("Export scale (PNG/PDF)", min_value=1, max_value=6, value=2, step=1)

    def to_image_bytes(fmt: str):
        try:
            return pio.to_image(fig, format=fmt, scale=int(export_scale))
        except Exception as e:
            st.warning(f"{fmt.upper()} export needs `kaleido`. Error: {e}")
            return None

    with cE3:
        png = to_image_bytes("png")
        st.download_button(
            "⬇ PNG",
            data=png if png is not None else b"",
            file_name=f"{base}_{ts}.png",
            mime="image/png",
            disabled=(png is None),
        )

    with cE4:
        pdf = to_image_bytes("pdf")
        st.download_button(
            "⬇ PDF",
            data=pdf if pdf is not None else b"",
            file_name=f"{base}_{ts}.pdf",
            mime="application/pdf",
            disabled=(pdf is None),
        )


# -----------------------
# Page: Structure Viewer
# -----------------------
else:
    st.title("Structure Viewer (before/after)")

    view_mode = st.radio(
        "View mode",
        options=["Bulk candidate", "Vacancy configurations", "Surface slab"],
        horizontal=True,
        help="Choose whether to inspect bulk workflow structures or generated surface slabs.",
    )

    # ============================================================
    # BULK VIEWER
    # ============================================================
    if view_mode == "Bulk candidate":
        cfg_now = load_toml(input_toml_path)
        outdir_now = cfg_now.get("structure", {}).get("outdir", "random_structures")
        proj = ProjectIndex(root=project_root, outdir=project_root / outdir_now)

        comps = proj.compositions()
        if not comps:
            st.warning(f"No compositions found in `{proj.outdir}`.")
            st.stop()

        comp = st.selectbox("Composition", options=comps, key="bulk_comp")
        base = proj.composition_path(comp)

        selected = proj.selected_candidates(comp)
        if selected:
            cand_name = st.selectbox("Candidate", options=selected, key="bulk_cand")
        else:
            cand_dirs = sorted([p.name for p in base.glob("candidate_*") if p.is_dir()])
            if not cand_dirs:
                st.warning("No candidate folders found yet.")
                st.stop()
            cand_name = st.selectbox(
                "Candidate",
                options=cand_dirs,
                key="bulk_cand_fallback",
            )

        cand_dir = proj.find_candidate_dir(comp, cand_name)
        if cand_dir is None:
            st.error("Candidate folder not found.")
            st.stop()

        files = proj.find_structure_files(cand_dir)

        with st.expander("Structure files", expanded=False):
            st.json({k: str(v) for k, v in files.items()})

        viewer_options = structure_viewer_controls(
            [files.get("before"), files.get("after")],
            key_prefix=f"main_bulk_viewer_{comp}_{cand_name}",
            expanded=False,
            show_orientation_default=True,
            show_unit_cell_default=True,
        )

        colL, colR = st.columns(2)

        with colL:
            st.subheader("Before")
            if "before" in files:
                show_structure(
                    files["before"],
                    title=f"{comp} / {cand_name} — before",
                    viewer_mode="bulk",
                    width=700,
                    height=500,
                    **viewer_options,
                )
            else:
                st.info("No 'before' structure file found.")

        with colR:
            st.subheader("After")
            if "after" in files:
                show_structure(
                    files["after"],
                    title=f"{comp} / {cand_name} — after",
                    viewer_mode="bulk",
                    width=700,
                    height=500,
                )
            else:
                st.info("No 'after' structure file found.")

    # ============================================================
    # VACANCY VIEWER
    # ============================================================
    elif view_mode == "Vacancy configurations":
        cfg_now = load_toml(input_toml_path)
        outdir_now = cfg_now.get("structure", {}).get("outdir", "random_structures")
        vacancy_cfg_now = cfg_now.get("vacancies", {}) or {}
        vacancy_outdir = project_root / outdir_now
        if vacancy_cfg_now.get("parent_source") == "directory":
            configured_parent_root = Path(str(vacancy_cfg_now.get("parent_directory", ""))).expanduser()
            vacancy_outdir = (
                configured_parent_root
                if configured_parent_root.is_absolute()
                else project_root / configured_parent_root
            ).resolve()
        proj = ProjectIndex(root=project_root, outdir=vacancy_outdir)
        comps = [comp for comp in proj.compositions() if proj.vacancy_parents(comp)]
        if not comps:
            st.warning("No 05_vacancies results were found.")
            st.stop()
        comp = st.selectbox("Composition", comps, key="vac_view_comp")
        candidate = st.selectbox(
            "Parent candidate", proj.vacancy_parents(comp), key="vac_view_parent"
        )
        vacancy_root = proj.composition_path(comp) / candidate / "05_vacancies"
        groups = sorted(path for path in vacancy_root.glob("V_*_*") if path.is_dir())
        if not groups:
            st.info("This parent has only the zero-vacancy reference.")
            parent_path = vacancy_root / "parent_reference" / "relaxed" / "POSCAR"
            if parent_path.exists():
                parent_viewer_options = structure_viewer_controls(
                    [parent_path],
                    key_prefix=f"main_vacancy_parent_viewer_{comp}_{candidate}",
                    expanded=False,
                    show_orientation_default=True,
                    show_unit_cell_default=True,
                )
                show_structure(
                    parent_path,
                    title="Relaxed parent",
                    viewer_mode="bulk",
                    **parent_viewer_options,
                )
            st.stop()
        group = st.selectbox("Vacancy count", groups, format_func=lambda path: path.name, key="vac_view_count")
        mc_summary_path = group / "monte_carlo_summary.json"
        if mc_summary_path.exists():
            mc_summary = json.loads(mc_summary_path.read_text(encoding="utf-8"))
            st.caption(
                f"Monte Carlo: {mc_summary.get('mc_steps', 0)} steps; "
                f"{mc_summary.get('selection_reason', 'unknown stop reason')}"
            )
            move_rows = [
                {
                    "move": move,
                    "attempted": attempted,
                    "accepted": mc_summary.get("mc_accepted_moves", {}).get(move, 0),
                }
                for move, attempted in mc_summary.get("mc_attempted_moves", {}).items()
            ]
            if move_rows:
                move_frame = pd.DataFrame(move_rows)
                move_frame["acceptance_ratio"] = (
                    move_frame["accepted"] / move_frame["attempted"].replace(0, pd.NA)
                )
                st.dataframe(move_frame, use_container_width=True, hide_index=True)
        config_dirs = sorted(
            path
            for path in group.iterdir()
            if path.is_dir() and path.name.startswith(("config_", "mc_"))
        )
        if not config_dirs:
            st.warning("No generated vacancy configurations were found for this count.")
            st.stop()
        configuration = st.selectbox(
            "Configuration", config_dirs, format_func=lambda path: path.name, key="vac_view_config"
        )
        count = int(group.name.rsplit("_", 1)[1])
        files = proj.find_vacancy_structure_files(comp, candidate, count, configuration.name)
        st.json({label: str(path) for label, path in files.items()})
        vacancy_viewer_options = structure_viewer_controls(
            [files.get("parent"), files.get("generated"), files.get("relaxed")],
            key_prefix=(
                f"main_vacancy_viewer_{comp}_{candidate}_{group.name}_{configuration.name}"
            ),
            expanded=False,
            show_orientation_default=True,
            show_unit_cell_default=True,
        )
        columns = st.columns(3)
        for column, label in zip(columns, ("parent", "generated", "relaxed")):
            with column:
                st.subheader(label.title())
                if label in files:
                    show_structure(
                        files[label],
                        title=f"{candidate} — {label}",
                        viewer_mode="bulk",
                        width=500,
                        height=450,
                        **vacancy_viewer_options,
                    )
                else:
                    st.info(f"No {label} structure available.")

    # ============================================================
    # SURFACE VIEWER
    # ============================================================
    else:
        cfg_now = load_toml(input_toml_path)
        surface_cfg = cfg_now.get("surface", {}) or {}

        surface_outdir = surface_cfg.get("outdir", "generated_surfaces")
        surface_root = (project_root / surface_outdir).resolve()

        st.caption(f"Surface root: `{surface_root}`")

        if not surface_root.exists():
            st.warning(f"Surface output directory not found: `{surface_root}`")
            st.stop()

        # --------------------------------------------------------
        # composition
        # --------------------------------------------------------
        surface_comps = sorted([p.name for p in surface_root.iterdir() if p.is_dir()])
        if not surface_comps:
            st.warning("No surface composition folders found.")
            st.stop()

        comp = st.selectbox("Surface composition", options=surface_comps, key="surf_comp")
        comp_dir = surface_root / comp

        # --------------------------------------------------------
        # candidate
        # --------------------------------------------------------
        candidate_dirs = sorted([p.name for p in comp_dir.glob("candidate_*") if p.is_dir()])
        if not candidate_dirs:
            st.warning("No surface candidate folders found.")
            st.stop()

        cand_name = st.selectbox("Surface candidate", options=candidate_dirs, key="surf_cand")
        cand_dir = comp_dir / cand_name

        # --------------------------------------------------------
        # orientation
        # --------------------------------------------------------
        hkl_dirs = sorted([p.name for p in cand_dir.glob("hkl_*") if p.is_dir()])
        if not hkl_dirs:
            st.warning("No surface orientation folders found.")
            st.stop()

        hkl_name = st.selectbox("Surface orientation", options=hkl_dirs, key="surf_hkl")
        hkl_dir = cand_dir / hkl_name

        # --------------------------------------------------------
        # termination
        # --------------------------------------------------------
        term_dirs = sorted([p.name for p in hkl_dir.glob("term_*") if p.is_dir()])
        if not term_dirs:
            st.warning("No surface termination folders found.")
            st.stop()

        term_name = st.selectbox("Termination", options=term_dirs, key="surf_term")
        term_dir = hkl_dir / term_name

        # --------------------------------------------------------
        # locate files
        # --------------------------------------------------------
        relaxed_name = str(surface_cfg.get("surface_relaxed_filename", "CONTCAR"))

        before_path = term_dir / "POSCAR"
        after_path = term_dir / relaxed_name

        meta_path = term_dir / "meta.json"
        relax_meta_path = term_dir / str(
            surface_cfg.get("surface_relax_meta_filename", "surface_relax.json")
        )

        info_dict = {
            "before": str(before_path) if before_path.exists() else None,
            "after": str(after_path) if after_path.exists() else None,
            "meta": str(meta_path) if meta_path.exists() else None,
            "surface_relax_meta": str(relax_meta_path) if relax_meta_path.exists() else None,
        }

        with st.expander("Surface files", expanded=False):
            st.json(info_dict)

        # --------------------------------------------------------
        # optional metadata preview
        # --------------------------------------------------------
        with st.expander("Metadata", expanded=False):
            if meta_path.exists():
                try:
                    st.json(json.loads(meta_path.read_text(encoding="utf-8")))
                except Exception as e:
                    st.warning(f"Failed to read meta.json: {e}")
            else:
                st.info("No meta.json found.")

            if relax_meta_path.exists():
                try:
                    st.json(json.loads(relax_meta_path.read_text(encoding="utf-8")))
                except Exception as e:
                    st.warning(f"Failed to read surface_relax.json: {e}")
            else:
                st.info("No surface_relax.json found.")

        # --------------------------------------------------------
        # visualization
        # --------------------------------------------------------
        surface_viewer_options = structure_viewer_controls(
            [before_path, after_path],
            key_prefix=(
                f"main_surface_viewer_{comp}_{cand_name}_{hkl_name}_{term_name}"
            ),
            expanded=False,
            show_orientation_default=True,
            show_unit_cell_default=True,
        )
        colL, colR = st.columns(2)

        with colL:
            st.subheader("Before relaxation")
            if before_path.exists():
                show_structure(
                    before_path,
                    title=f"{comp} / {cand_name} / {hkl_name} / {term_name} — POSCAR",
                    viewer_mode="surface",
                    width=800,
                    height=550,
                    **surface_viewer_options,
                )
            else:
                st.info("No initial slab POSCAR found.")

        with colR:
            st.subheader("After relaxation")
            if after_path.exists():
                show_structure(
                    after_path,
                    title=f"{comp} / {cand_name} / {hkl_name} / {term_name} — {relaxed_name}",
                    viewer_mode="surface",
                    width=800,
                    height=550,
                )
            else:
                st.info(f"No relaxed slab file found (`{relaxed_name}`).")

# gui/app.py
from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import toml
import re


# Make sure gui/ is importable
GUI_DIR = Path(__file__).resolve().parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from gui_config import SUPER_CELL_PRESETS, DOPING_MODE_CHOICES, ALLOWED_DOPANTS, RUN_PRESETS, DEFAULTS, CHOICES
from io_project import ProjectIndex
from view_structure import show_structure


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
    for secname in ["hardware", "references", "structure", "doping", "generate", "scan", "relax", "filter", "bandgap", "formation", "database"]:
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

        mace_models = [
            "small",
            "medium",
            "large",
            "small-mpa-0",
            "medium-mpa-0",
            "large-mpa-0",
            "small-omat-0",
            "medium-omat-0",
        ]

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
                current_model = str(
                    cfg_edit["references"].get("model", "small")
                ).strip()
                if current_model not in mace_models:
                    current_model = "small"

                cfg_edit["references"]["model"] = st.selectbox(
                    "model",
                    options=mace_models,
                    index=mace_models.index(current_model),
                    help="Pretrained MACE model.",
                )
                cfg_edit["references"]["task"] = ""

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
            else:
                cfg_edit["references"]["task"] = st.text_input(
                    "task",
                    value="",
                    disabled=True,
                    help="Task is only used for the UMA backend.",
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

            cfg_edit["references"]["metals_dir"] = st.text_input(
                "metals_dir",
                value=str(cfg_edit["references"].get("metals_dir", "reference_structures/metals")),
                help="Directory containing <Element>.POSCAR files.",
            )

            metal_default = cfg_edit["references"].get(
                "metal_ref",
                ["Sn", "Sb", "Ti", "Zr", "Nb"],
            )
            metal_csv = st.text_input(
                "metal_ref (comma-separated)",
                value=",".join(metal_default),
                help='Example: "Sn,Sb,Ti,Zr,Nb"',
            )
            cfg_edit["references"]["metal_ref"] = [
                x.strip() for x in metal_csv.split(",") if x.strip()
            ]

        else:
            st.subheader("Oxide and gas references")

            cfg_edit["references"]["oxides_dir"] = st.text_input(
                "oxides_dir",
                value=str(cfg_edit["references"].get("oxides_dir", "reference_structures/oxides")),
                help="Directory containing <Oxide>.POSCAR files.",
            )

            ox_default = cfg_edit["references"].get(
                "oxides_ref",
                ["Sb2O5", "TiO2", "ZrO2", "Nb2O5"],
            )
            ox_csv = st.text_input(
                "oxides_ref (comma-separated)",
                value=",".join(ox_default),
                help='Example: "Sb2O5,TiO2,ZrO2,Nb2O5"',
            )
            cfg_edit["references"]["oxides_ref"] = [
                x.strip() for x in ox_csv.split(",") if x.strip()
            ]

            cfg_edit["references"]["gas_dir"] = st.text_input(
                "gas_dir",
                value=str(cfg_edit["references"].get("gas_dir", "reference_structures/gas")),
                help="Directory containing the gas reference POSCAR, typically O2.",
            )

            cfg_edit["references"]["gas_ref"] = st.text_input(
                "gas_ref",
                value=str(cfg_edit["references"].get("gas_ref", "O2")),
                help='Typically "O2".',
            )

            oxygen_choices = ["O-rich", "O-poor"]
            current_oxygen_mode = str(
                cfg_edit["references"].get("oxygen_mode", "O-rich")
            ).strip()
            if current_oxygen_mode not in oxygen_choices:
                current_oxygen_mode = "O-rich"

            cfg_edit["references"]["oxygen_mode"] = st.radio(
                "oxygen_mode",
                options=oxygen_choices,
                index=oxygen_choices.index(current_oxygen_mode),
                horizontal=True,
            )

            cfg_edit["references"]["muO_shift_ev"] = st.number_input(
                "muO_shift_ev",
                value=float(cfg_edit["references"].get("muO_shift_ev", 0.0)),
                step=0.1,
                format="%.3f",
                help="Optional oxygen chemical potential shift in eV.",
            )



    # -----------------------------
    # GENERATE
    # -----------------------------
    with st.expander("Generate", expanded=False):
        st.subheader("POSCAR writing")

        order = cfg_edit["generate"].get("poscar_order", DEFAULTS["generate"]["poscar_order"])
        if not isinstance(order, list):
            order = DEFAULTS["generate"]["poscar_order"]

        col1, col2 = st.columns([3, 2], vertical_alignment="bottom")

        with col1:
            order_text = st.text_input(
                "Species order in output POSCAR",
                value=",".join(order),
                placeholder="e.g. Zr,Ti,Sb,Sn,O",
                help=(
                    "Optional but recommended. Defining the species order ensures consistent "
                    "POSCAR formatting and helps with organized output storage. "
                    "Leave empty to keep the original species order."
                )
            )

        with col2:
            st.caption("Example: `Zr,Ti,Sb,Sn,O`")

        parsed_order = [x.strip() for x in order_text.split(",") if x.strip()]
        cfg_edit["generate"]["poscar_order"] = parsed_order

        if len(parsed_order) == 0:
            st.info("POSCAR species order: unchanged (original order will be kept).")
        else:
            st.success(f"POSCAR species order: {parsed_order}")

        st.divider()
        st.subheader("Reproducibility")

        cfg_edit["generate"]["seed_base"] = st.number_input(
            "Random seed (base)",
            min_value=0,
            value=int(cfg_edit["generate"].get("seed_base", DEFAULTS["generate"]["seed_base"])),
            step=1,
            help="Base seed used to generate deterministic randomness per composition.",
        )


    # -----------------------------
    # DOPING
    # -----------------------------
    with st.expander("Doping", expanded=True):
        st.subheader("Doping definition")

        col1, col2 = st.columns([2, 2], vertical_alignment="bottom")

        with col1:
            mode = st.selectbox(
                "Mode",
                CHOICES["doping.mode"],
                index=CHOICES["doping.mode"].index(cfg_edit["doping"].get("mode", "explicit")),
                help="Choose how compositions are provided: explicit list or automatic enumeration rules.",
            )
            cfg_edit["doping"]["mode"] = mode

        with col2:
            host_in = st.text_input(
                "Host species",
                value=cfg_edit["doping"].get("host_species", DEFAULTS["doping"]["host_species"]),
                help="Species to substitute during doping (e.g. Sn).",
            )

        # Normalize + store host
        host = normalize_symbol(host_in)
        cfg_edit["doping"]["host_species"] = host

        # Build element options (exclude host)
        element_options = [e for e in PERIODIC_TABLE if e != host]

        # Optional: warn if host isn't a valid element symbol
        if host and host not in PERIODIC_TABLE:
            st.warning(f"`host_species` looks unusual: `{host}` is not a standard element symbol.")

        st.divider()

        # -----------------------------
        # Explicit mode: composition builder
        # -----------------------------
        if mode == "explicit":
            st.subheader("Explicit compositions")
            st.caption("Build one composition at a time, then click **Add composition**.")

            # Ensure list exists
            if "compositions_ui" not in st.session_state:
                cfg_comps = normalize_compositions_list(cfg_edit["doping"].get("compositions"))
                st.session_state.compositions_ui = cfg_comps if cfg_comps else []

            colA, colB, colC = st.columns([2, 1, 1], vertical_alignment="bottom")

            with colA:
                dopant = st.selectbox(
                    "Dopant",
                    options=element_options,
                    index=0,
                    help=f"Choose any element except the host species ({host}).",
                )

            with colB:
                percent = st.number_input(
                    "Percent (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=1.0,
                    help="Dopant percentage relative to the host sublattice.",
                )

            with colC:
                add_btn = st.button("➕ Add dopant")

            # Temporary composition in session
            if "current_comp" not in st.session_state:
                st.session_state.current_comp = {}

            if add_btn:
                if dopant == host:
                    st.error("Dopant cannot be the same as the host species.")
                else:
                    st.session_state.current_comp[dopant] = float(percent)

            # Show current composition being built
            st.markdown("**Current composition (being built):**")
            if st.session_state.current_comp:
                st.json(st.session_state.current_comp)
            else:
                st.info("No dopants added yet. Use **Add dopant** to build a composition.")

            col_save, col_clear = st.columns([1, 1], vertical_alignment="bottom")

            with col_save:
                if st.button("✅ Add composition"):
                    if st.session_state.current_comp:
                        st.session_state.compositions_ui.append(dict(st.session_state.current_comp))
                        st.session_state.current_comp = {}
                    else:
                        st.warning("Add at least one dopant before adding a composition.")

            with col_clear:
                if st.button("🧹 Clear current composition"):
                    st.session_state.current_comp = {}

            st.divider()

            # Display compositions table
            st.markdown("### Current compositions")
            comps = st.session_state.compositions_ui

            if comps:
                df = pd.DataFrame(comps).fillna(0.0)
                df.index = [f"comp_{i:03d}" for i in range(len(df))]
                st.dataframe(df, use_container_width=True)

                colD, colE = st.columns([1, 1], vertical_alignment="bottom")
                with colD:
                    del_idx = st.number_input(
                        "Delete composition index",
                        min_value=0,
                        max_value=len(comps) - 1,
                        value=0,
                        step=1,
                        help="Index in the table (0-based).",
                    )
                    if st.button("🗑 Delete selected"):
                        st.session_state.compositions_ui.pop(int(del_idx))

                with colE:
                    if st.button("🧨 Clear all compositions"):
                        st.session_state.compositions_ui = []

            else:
                st.info("No compositions defined yet.")

            # Write back to config
            cfg_edit["doping"]["compositions"] = st.session_state.compositions_ui

            # keep enumerate fields available but not used
            for k in ["must_include", "dopants", "max_dopants_total", "allowed_totals", "levels"]:
                cfg_edit["doping"].setdefault(k, DEFAULTS["doping"][k])

        # -----------------------------
        # Enumerate mode: rule inputs (no parsing)
        # -----------------------------
        else:
            st.subheader("Enumerate mode rules")

            colA, colB = st.columns([2, 2], vertical_alignment="bottom")

            with colA:
                default_must = [x for x in cfg_edit["doping"].get("must_include", DEFAULTS["doping"]["must_include"]) if x != host]
                cfg_edit["doping"]["must_include"] = st.multiselect(
                    "Must include",
                    options=element_options,
                    default=default_must,
                    help="These dopants must appear in every generated composition.",
                )

            with colB:
                default_dops = [x for x in cfg_edit["doping"].get("dopants", DEFAULTS["doping"]["dopants"]) if x != host]
                cfg_edit["doping"]["dopants"] = st.multiselect(
                    "Allowed dopants",
                    options=element_options,
                    default=default_dops,
                    help="Dopants considered during enumeration.",
                )

            # Safety: enforce host exclusion (even if loaded from older TOML)
            cfg_edit["doping"]["must_include"] = [x for x in cfg_edit["doping"]["must_include"] if x != host]
            cfg_edit["doping"]["dopants"] = [x for x in cfg_edit["doping"]["dopants"] if x != host]

            colC, colD = st.columns([2, 2], vertical_alignment="bottom")

            with colC:
                cfg_edit["doping"]["max_dopants_total"] = st.number_input(
                    "Max dopants per composition",
                    min_value=1,
                    value=int(cfg_edit["doping"].get("max_dopants_total", DEFAULTS["doping"]["max_dopants_total"])),
                    step=1,
                    help="Maximum number of dopant species in any generated composition.",
                )

            with colD:
                st.caption("Example: 2 means co-doping at most (two dopant species).")

            # Totals and levels as multiselect (clean)
            def parse_number_list(text: str) -> list[float]:
                """
                Parse a comma/space/semicolon-separated list of numbers into sorted unique floats.
                Accepts: "5,10, 15;20 25" etc.
                """
                if text is None:
                    return []
                # split on commas/semicolons/whitespace
                parts = re.split(r"[,\s;]+", str(text).strip())
                out: list[float] = []
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    try:
                        out.append(float(p))
                    except ValueError:
                        raise ValueError(f"Invalid number: {p!r}")
                # unique + sorted
                out = sorted(set(out))
                return out

            def format_number_list(vals: list[float]) -> str:
                if not vals:
                    return ""
                # Keep integers pretty
                parts = []
                for v in vals:
                    if abs(v - round(v)) < 1e-12:
                        parts.append(str(int(round(v))))
                    else:
                        parts.append(str(v))
                return ", ".join(parts)

            # --- Allowed total doping (%) : FREE TEXT ---
            default_totals = cfg_edit["doping"].get("allowed_totals", DEFAULTS["doping"]["allowed_totals"])
            totals_text_default = format_number_list(default_totals)

            totals_text = st.text_input(
                "Allowed total doping (%)",
                value=totals_text_default,
                help="Enter any list of numbers (comma/space/semicolon separated). Example: 5, 10, 15, 20, 25, 30",
            )

            try:
                allowed_totals = parse_number_list(totals_text)
                # basic sanity (you can relax these if you truly want *anything*)
                if any(t < 0 or t > 100 for t in allowed_totals):
                    st.error("Allowed total doping values should be between 0 and 100.")
                cfg_edit["doping"]["allowed_totals"] = allowed_totals
                st.caption(f"Parsed totals: {allowed_totals}")
            except ValueError as e:
                st.error(str(e))
                # keep previous valid values
                cfg_edit["doping"]["allowed_totals"] = default_totals

            # --- Allowed levels (%) per dopant : FREE TEXT ---
            default_levels = cfg_edit["doping"].get("levels", DEFAULTS["doping"]["levels"])
            levels_text_default = format_number_list(default_levels)

            levels_text = st.text_input(
                "Allowed levels (%) per dopant",
                value=levels_text_default,
                help="Enter any list of numbers (comma/space/semicolon separated). Example: 2.5, 5, 7.5, 10",
            )

            try:
                levels = parse_number_list(levels_text)
                if any(x < 0 or x > 100 for x in levels):
                    st.error("Allowed level values should be between 0 and 100.")
                cfg_edit["doping"]["levels"] = levels
                st.caption(f"Parsed levels: {levels}")
            except ValueError as e:
                st.error(str(e))
                cfg_edit["doping"]["levels"] = default_levels            

            # compositions unused in enumerate mode
            cfg_edit["doping"].setdefault("compositions", [])

    # -----------------------------
    # SCAN
    # -----------------------------
    with st.expander("Scan", expanded=False):
        st.subheader("Screen candidates (Single-point energy)")

        st.caption(
            "Ranks doped configurations using the selected ML backend. "
            "In auto mode, the workflow uses exact symmetry-unique enumeration for manageable cases "
            "and switches to random symmetry-unique sampling for large configuration spaces."
        )

        cfg_edit.setdefault("scan", {})

        # -----------------------------
        # BACKEND
        # -----------------------------
        st.divider()
        st.subheader("Backend")

        backend_choices = ["m3gnet", "uma", "mace", "grace"]
        current_backend = str(cfg_edit["scan"].get("backend", DEFAULTS["scan"].get("backend", "m3gnet")))
        if current_backend not in backend_choices:
            current_backend = "m3gnet"

        colB1, colB2 = st.columns([1, 2], vertical_alignment="bottom")

        with colB1:
            cfg_edit["scan"]["backend"] = st.selectbox(
                "Scan backend",
                options=backend_choices,
                index=backend_choices.index(current_backend),
                help="ML backend used for single-point energy evaluation.",
                key="scan_backend",
            )

        scan_backend = cfg_edit["scan"]["backend"]

        with colB2:
            if scan_backend == "m3gnet":
                st.info("Stable general-purpose backend. Uses model='default' and ignores task.")
            elif scan_backend == "uma":
                st.info("Requires FAIR-Chem installation, Hugging Face login, and access to the gated UMA repository.")
            elif scan_backend == "mace":
                st.info("Uses MACE foundation models. The model is loaded once and reused during scan.")
            else:
                st.info("Uses GRACE foundation models. Task is not used.")

        # Model / task
        colBM, colBT = st.columns(2, vertical_alignment="bottom")

        if scan_backend == "m3gnet":
            cfg_edit["scan"]["model"] = "default"
            cfg_edit["scan"]["task"] = ""

            with colBM:
                st.text_input(
                    "Model",
                    value="default",
                    disabled=True,
                    key="scan_model_disabled_m3gnet",
                )
            with colBT:
                st.text_input(
                    "Task",
                    value="",
                    disabled=True,
                    key="scan_task_disabled_m3gnet",
                )

        elif scan_backend == "uma":
            uma_models = ["uma-s-1p2", "uma-s-1p1", "uma-m-1p1"]
            current_model = str(cfg_edit["scan"].get("model", "uma-s-1p2"))
            if current_model not in uma_models:
                current_model = "uma-s-1p2"

            uma_tasks = ["omat", "oc20", "oc22", "oc25", "omol", "odac", "omc"]
            current_task = str(cfg_edit["scan"].get("task", "omat"))
            if current_task not in uma_tasks:
                current_task = "omat"

            with colBM:
                cfg_edit["scan"]["model"] = st.selectbox(
                    "UMA model",
                    options=uma_models,
                    index=uma_models.index(current_model),
                    key="scan_model_uma",
                )
            with colBT:
                cfg_edit["scan"]["task"] = st.selectbox(
                    "UMA task",
                    options=uma_tasks,
                    index=uma_tasks.index(current_task),
                    key="scan_task_uma",
                )

        elif scan_backend == "mace":
            mace_models = [
                "small",
                "medium",
                "large",
                "small-mpa-0",
                "medium-mpa-0",
                "large-mpa-0",
                "small-omat-0",
                "medium-omat-0",
            ]
            current_model = str(cfg_edit["scan"].get("model", "small"))
            if current_model not in mace_models:
                current_model = "small"

            cfg_edit["scan"]["task"] = ""

            with colBM:
                cfg_edit["scan"]["model"] = st.selectbox(
                    "MACE model",
                    options=mace_models,
                    index=mace_models.index(current_model),
                    key="scan_model_mace",
                )
            with colBT:
                st.text_input(
                    "Task",
                    value="",
                    disabled=True,
                    key="scan_task_disabled_mace",
                )

        else:  # grace
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
            current_model = str(cfg_edit["scan"].get("model", "GRACE-2L-OAM"))
            if current_model not in grace_models:
                current_model = "GRACE-2L-OAM"

            cfg_edit["scan"]["task"] = ""

            with colBM:
                cfg_edit["scan"]["model"] = st.selectbox(
                    "GRACE model",
                    options=grace_models,
                    index=grace_models.index(current_model),
                    key="scan_model_grace",
                )
            with colBT:
                st.text_input(
                    "Task",
                    value="",
                    disabled=True,
                    key="scan_task_disabled_grace",
                )

        # -----------------------------
        # STRATEGY
        # -----------------------------
        st.divider()
        st.subheader("Scan strategy")

        colM1, colM2 = st.columns([1, 2], vertical_alignment="bottom")

        with colM1:
            scan_mode_choices = CHOICES.get("scan.mode", ["auto", "exact", "sample"])
            current_scan_mode = str(cfg_edit["scan"].get("mode", DEFAULTS["scan"]["mode"]))
            if current_scan_mode not in scan_mode_choices:
                current_scan_mode = DEFAULTS["scan"]["mode"]

            cfg_edit["scan"]["mode"] = st.selectbox(
                "Scan mode",
                options=scan_mode_choices,
                index=scan_mode_choices.index(current_scan_mode),
                help=(
                    "auto: use exact enumeration when manageable, otherwise sampling. "
                    "exact: force symmetry-unique enumeration. "
                    "sample: force random symmetry-unique sampling."
                ),
                key="scan_mode",
            )

        scan_mode = cfg_edit["scan"]["mode"]

        with colM2:
            if scan_mode == "auto":
                st.info(
                    "Auto mode is recommended for most cases. It stays exact for small problems "
                    "and automatically switches to sampling when the configuration space becomes too large."
                )
            elif scan_mode == "exact":
                st.warning(
                    "Exact mode can become very expensive for large supercells or many dopants."
                )
            else:
                st.info(
                    "Sample mode is more memory-friendly and is recommended for large supercells."
                )

        if scan_mode == "exact":
            colE1, colE2 = st.columns(2, vertical_alignment="bottom")

            with colE1:
                cfg_edit["scan"]["max_enum"] = st.number_input(
                    "Max exact raw configs",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("max_enum", DEFAULTS["scan"]["max_enum"])),
                    step=1000,
                    help="Hard limit for raw configuration count in exact mode.",
                    key="scan_max_enum",
                )

            with colE2:
                cfg_edit["scan"]["max_unique"] = st.number_input(
                    "Max exact symmetry-unique configs",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("max_unique", DEFAULTS["scan"]["max_unique"])),
                    step=1000,
                    help="Hard cap on the number of symmetry-unique configurations kept in exact mode.",
                    key="scan_max_unique",
                )

        elif scan_mode == "sample":
            colS1, colS2, colS3 = st.columns(3, vertical_alignment="bottom")

            with colS1:
                cfg_edit["scan"]["sample_budget"] = st.number_input(
                    "Sample budget",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("sample_budget", DEFAULTS["scan"]["sample_budget"])),
                    step=100,
                    help="Maximum number of random attempts in sampling mode.",
                    key="scan_sample_budget",
                )

            with colS2:
                cfg_edit["scan"]["sample_batch_size"] = st.number_input(
                    "Sample batch size",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("sample_batch_size", DEFAULTS["scan"]["sample_batch_size"])),
                    step=1,
                    help="Number of new unique sampled configurations evaluated per batch.",
                    key="scan_sample_batch_size",
                )

            with colS3:
                cfg_edit["scan"]["sample_patience"] = st.number_input(
                    "Sample patience",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("sample_patience", DEFAULTS["scan"]["sample_patience"])),
                    step=100,
                    help="Stop sampling after this many non-improving sampled candidates.",
                    key="scan_sample_patience",
                )

            colS4, colS5 = st.columns(2, vertical_alignment="bottom")

            with colS4:
                cfg_edit["scan"]["sample_seed"] = st.number_input(
                    "Sample seed",
                    min_value=0,
                    value=int(cfg_edit["scan"].get("sample_seed", DEFAULTS["scan"]["sample_seed"])),
                    step=1,
                    help="Random seed used in sampling mode.",
                    key="scan_sample_seed",
                )

            with colS5:
                cfg_edit["scan"]["sample_max_saved"] = st.number_input(
                    "Sample max saved",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("sample_max_saved", DEFAULTS["scan"]["sample_max_saved"])),
                    step=1000,
                    help="Maximum number of sampled canonical configurations remembered to avoid duplicates.",
                    key="scan_sample_max_saved",
                )

        else:  # auto
            colA1, colA2 = st.columns(2, vertical_alignment="bottom")

            with colA1:
                cfg_edit["scan"]["max_enum"] = st.number_input(
                    "Max exact raw configs",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("max_enum", DEFAULTS["scan"]["max_enum"])),
                    step=1000,
                    help="If exceeded in auto mode, the workflow switches to sampling.",
                    key="scan_max_enum",
                )

            with colA2:
                cfg_edit["scan"]["max_unique"] = st.number_input(
                    "Max exact symmetry-unique configs",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("max_unique", DEFAULTS["scan"]["max_unique"])),
                    step=1000,
                    help="Hard cap on the number of symmetry-unique configurations in exact mode.",
                    key="scan_max_unique",
                )

            colA3, colA4, colA5 = st.columns(3, vertical_alignment="bottom")

            with colA3:
                cfg_edit["scan"]["sample_budget"] = st.number_input(
                    "Sample budget",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("sample_budget", DEFAULTS["scan"]["sample_budget"])),
                    step=100,
                    help="Maximum number of random attempts if auto mode switches to sampling.",
                    key="scan_sample_budget",
                )

            with colA4:
                cfg_edit["scan"]["sample_batch_size"] = st.number_input(
                    "Sample batch size",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("sample_batch_size", DEFAULTS["scan"]["sample_batch_size"])),
                    step=1,
                    help="Batch size used if auto mode switches to sampling.",
                    key="scan_sample_batch_size",
                )

            with colA5:
                cfg_edit["scan"]["sample_patience"] = st.number_input(
                    "Sample patience",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("sample_patience", DEFAULTS["scan"]["sample_patience"])),
                    step=100,
                    help="Stop criterion used if auto mode switches to sampling.",
                    key="scan_sample_patience",
                )

            colA6, colA7 = st.columns(2, vertical_alignment="bottom")

            with colA6:
                cfg_edit["scan"]["sample_seed"] = st.number_input(
                    "Sample seed",
                    min_value=0,
                    value=int(cfg_edit["scan"].get("sample_seed", DEFAULTS["scan"]["sample_seed"])),
                    step=1,
                    help="Random seed used if auto mode switches to sampling.",
                    key="scan_sample_seed",
                )

            with colA7:
                cfg_edit["scan"]["sample_max_saved"] = st.number_input(
                    "Sample max saved",
                    min_value=1,
                    value=int(cfg_edit["scan"].get("sample_max_saved", DEFAULTS["scan"]["sample_max_saved"])),
                    step=1000,
                    help="Maximum number of sampled canonical configurations remembered to avoid duplicates.",
                    key="scan_sample_max_saved",
                )

        st.divider()
        st.subheader("Input & selection")

        col1, col2 = st.columns([2, 2], vertical_alignment="bottom")

        with col1:
            cfg_edit["scan"]["topk"] = st.number_input(
                "Keep top-K candidates",
                min_value=1,
                value=int(cfg_edit["scan"].get("topk", DEFAULTS["scan"]["topk"])),
                step=1,
                help="Number of lowest-energy configurations retained after the scan.",
                key="scan_topk",
            )

        with col2:
            st.caption(
                "Scan reads `POSCAR` automatically from each generated structure folder."
            )

        st.divider()
        st.subheader("Symmetry & uniqueness")

        colA, colB = st.columns([1, 2], vertical_alignment="bottom")

        with colA:
            cfg_edit["scan"]["symprec"] = st.number_input(
                "Symmetry tolerance (symprec)",
                min_value=1e-6,
                value=float(cfg_edit["scan"].get("symprec", DEFAULTS["scan"]["symprec"])),
                step=1e-3,
                format="%.6f",
                help="Tolerance used when identifying symmetry-equivalent configurations.",
                key="scan_symprec",
            )

        with colB:
            st.caption(
                "Smaller values are stricter and usually produce more unique structures, "
                "which can increase runtime."
            )

        st.divider()
        st.subheader("Execution")

        colX1, colX2, colX3 = st.columns(3, vertical_alignment="bottom")

        with colX1:
            device_choices = ["cpu", "cuda"]
            current_device = str(cfg_edit["scan"].get("device", DEFAULTS["scan"].get("device", "cpu")))
            if current_device not in device_choices:
                current_device = "cpu"

            cfg_edit["scan"]["device"] = st.selectbox(
                "Device",
                options=device_choices,
                index=device_choices.index(current_device),
                key="scan_device",
            )

        with colX2:
            cfg_edit["scan"]["n_workers"] = st.number_input(
                "Workers",
                min_value=1,
                value=int(cfg_edit["scan"].get("n_workers", DEFAULTS["scan"]["n_workers"])),
                step=1,
                key="scan_n_workers",
            )

        with colX3:
            cfg_edit["scan"]["gpu_id"] = st.number_input(
                "GPU ID",
                min_value=0,
                value=int(cfg_edit["scan"].get("gpu_id", DEFAULTS["scan"].get("gpu_id", 0))),
                step=1,
                key="scan_gpu_id",
            )

        cfg_edit["scan"]["chunksize"] = st.number_input(
            "Chunksize",
            min_value=1,
            value=int(cfg_edit["scan"].get("chunksize", DEFAULTS["scan"]["chunksize"])),
            step=1,
            help="Multiprocessing chunk size.",
            key="scan_chunksize",
        )

        st.divider()
        st.subheader("Sublattice definition")

        anions = cfg_edit["scan"].get("anion_species", DEFAULTS["scan"]["anion_species"])
        if not isinstance(anions, list):
            anions = DEFAULTS["scan"]["anion_species"]

        anion_text = st.text_input(
            "Anion species (comma-separated)",
            value=",".join(anions),
            help="Elements treated as anions and excluded from substitution. Example for oxides: O",
            placeholder="e.g. O",
            key="scan_anion_species",
        )
        cfg_edit["scan"]["anion_species"] = [x.strip() for x in anion_text.split(",") if x.strip()]

        st.divider()
        st.subheader("Caching")

        cfg_edit["scan"]["skip_if_done"] = st.checkbox(
            "Skip scan if results already exist",
            value=bool(cfg_edit["scan"].get("skip_if_done", True)),
            help="If enabled, scan is skipped when ranking_scan.csv already exists in the structure folder.",
            key="scan_skip_if_done",
        )

    # -----------------------------
    # RELAX
    # -----------------------------
    with st.expander("Relax", expanded=False):
        st.subheader("Doped structure relaxation")

        st.caption(
            "Relaxes the selected candidates using the chosen ML interatomic potential backend. "
            "You can select the backend, model, optimizer, convergence threshold, and maximum number of relaxation steps."
        )

        cfg_edit.setdefault("relax", {})

        # -----------------------------
        # Backend
        # -----------------------------
        st.divider()
        st.subheader("Backend")

        relax_backend_choices = ["m3gnet", "uma", "mace", "grace"]
        current_relax_backend = str(
            cfg_edit["relax"].get("backend", DEFAULTS["relax"].get("backend", "m3gnet"))
        ).strip().lower()
        if current_relax_backend not in relax_backend_choices:
            current_relax_backend = "m3gnet"

        cfg_edit["relax"]["backend"] = st.selectbox(
            "Relaxation backend",
            options=relax_backend_choices,
            index=relax_backend_choices.index(current_relax_backend),
            help="Choose the ML potential backend used for structural relaxation.",
            key="relax_backend",
        )

        relax_backend = cfg_edit["relax"]["backend"]

        # Model/task choices
        uma_model_choices = ["uma-s-1p2", "uma-s-1p1", "uma-m-1p1"]
        uma_task_choices = ["omat", "oc20", "oc22", "oc25", "omol", "odac", "omc"]

        mace_model_choices = [
            "small",
            "medium",
            "large",
            "small-mpa-0",
            "medium-mpa-0",
            "large-mpa-0",
            "small-omat-0",
            "medium-omat-0",
        ]

        grace_model_choices = [
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

        colB1, colB2 = st.columns(2, vertical_alignment="bottom")

        with colB1:
            if relax_backend == "m3gnet":
                cfg_edit["relax"]["model"] = "default"
                st.text_input(
                    "Model",
                    value="default",
                    disabled=True,
                    help="M3GNet currently uses the default pretrained model.",
                    key="relax_model_m3gnet_display",
                )

            elif relax_backend == "uma":
                current_model = str(cfg_edit["relax"].get("model", "uma-s-1p2")).strip()
                if current_model in {"", "default"} or current_model not in uma_model_choices:
                    current_model = "uma-s-1p2"

                cfg_edit["relax"]["model"] = st.selectbox(
                    "UMA model",
                    options=uma_model_choices,
                    index=uma_model_choices.index(current_model),
                    help="Choose the pretrained UMA model.",
                    key="relax_model_uma",
                )

            elif relax_backend == "mace":
                current_model = str(cfg_edit["relax"].get("model", "small")).strip()
                if current_model in {"", "default"} or current_model not in mace_model_choices:
                    current_model = "small"

                cfg_edit["relax"]["model"] = st.selectbox(
                    "MACE model",
                    options=mace_model_choices,
                    index=mace_model_choices.index(current_model),
                    help="Choose the pretrained MACE model.",
                    key="relax_model_mace",
                )

            elif relax_backend == "grace":
                current_model = str(cfg_edit["relax"].get("model", "GRACE-1L-OMAT")).strip()
                if current_model in {"", "default"} or current_model not in grace_model_choices:
                    current_model = "GRACE-1L-OMAT"

                cfg_edit["relax"]["model"] = st.selectbox(
                    "GRACE model",
                    options=grace_model_choices,
                    index=grace_model_choices.index(current_model),
                    help="Choose the pretrained GRACE model.",
                    key="relax_model_grace",
                )

        with colB2:
            if relax_backend == "uma":
                current_task = str(cfg_edit["relax"].get("task", "omat")).strip()
                if current_task == "" or current_task not in uma_task_choices:
                    current_task = "omat"

                cfg_edit["relax"]["task"] = st.selectbox(
                    "UMA task",
                    options=uma_task_choices,
                    index=uma_task_choices.index(current_task),
                    help="Task/domain used by the UMA predictor.",
                    key="relax_task_uma",
                )
            else:
                cfg_edit["relax"]["task"] = ""
                st.text_input(
                    "Task",
                    value="not used for this backend",
                    disabled=True,
                    key="relax_task_unused_display",
                )

        # -----------------------------
        # Optimizer & convergence
        # -----------------------------
        st.divider()
        st.subheader("Optimizer & convergence")

        optimizer_choices = ["bfgs", "lbfgs", "fire", "mdmin", "quasinewton"]
        current_optimizer = str(
            cfg_edit["relax"].get("optimizer", DEFAULTS["relax"].get("optimizer", "bfgs"))
        ).strip().lower()
        if current_optimizer not in optimizer_choices:
            current_optimizer = "bfgs"

        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")

        with col1:
            cfg_edit["relax"]["optimizer"] = st.selectbox(
                "Optimizer",
                options=optimizer_choices,
                index=optimizer_choices.index(current_optimizer),
                help="ASE optimizer used to move the atoms during relaxation.",
                key="relax_optimizer",
            )

        with col2:
            cfg_edit["relax"]["fmax"] = st.number_input(
                "Max force (fmax) [eV/Å]",
                min_value=0.0,
                value=float(cfg_edit["relax"].get("fmax", DEFAULTS["relax"]["fmax"])),
                step=0.01,
                help="Relaxation stops when the maximum atomic force drops below this threshold.",
                key="relax_fmax",
            )

        with col3:
            cfg_edit["relax"]["max_steps"] = int(
                st.number_input(
                    "Maximum steps",
                    min_value=1,
                    value=int(cfg_edit["relax"].get("max_steps", DEFAULTS["relax"].get("max_steps", 300))),
                    step=10,
                    help="Maximum number of optimizer steps before stopping the relaxation.",
                    key="relax_max_steps",
                )
            )

        st.caption(
            "Typical screening setup: **fmax = 0.05 eV/Å** and **100-300 steps**. "
            "Tighter relaxations may use **fmax = 0.02 eV/Å**."
        )

        # -----------------------------
        # Execution mode
        # -----------------------------
        st.divider()
        st.subheader("Execution mode")

        relax_device_options = ["cpu", "cuda"]
        current_relax_device = str(
            cfg_edit["relax"].get("device", DEFAULTS["relax"]["device"])
        ).lower()
        if current_relax_device not in relax_device_options:
            current_relax_device = "cpu"

        cfg_edit["relax"]["device"] = st.selectbox(
            "Device",
            relax_device_options,
            index=relax_device_options.index(current_relax_device),
            help="Choose CPU parallel execution or CUDA GPU execution.",
            key="relax_device",
        )

        relax_device = cfg_edit["relax"]["device"]

        if relax_device == "cpu":
            st.divider()
            st.subheader("Parallelism & performance")

            colA, colB, colC = st.columns(3, vertical_alignment="bottom")

            with colA:
                cfg_edit["relax"]["n_workers"] = int(
                    st.number_input(
                        "Workers",
                        min_value=1,
                        value=int(cfg_edit["relax"].get("n_workers", DEFAULTS["relax"]["n_workers"])),
                        step=1,
                        help="Number of parallel relaxation processes (one candidate per worker).",
                        key="relax_n_workers",
                    )
                )

            with colB:
                cfg_edit["relax"]["tf_threads"] = int(
                    st.number_input(
                        "TensorFlow threads / worker",
                        min_value=1,
                        value=int(cfg_edit["relax"].get("tf_threads", DEFAULTS["relax"]["tf_threads"])),
                        step=1,
                        help="TensorFlow threads per worker. Mainly relevant for M3GNet.",
                        key="relax_tf_threads",
                    )
                )

            with colC:
                cfg_edit["relax"]["omp_threads"] = int(
                    st.number_input(
                        "OpenMP threads / worker",
                        min_value=1,
                        value=int(cfg_edit["relax"].get("omp_threads", DEFAULTS["relax"]["omp_threads"])),
                        step=1,
                        help="OpenMP threads per worker. Keep small to avoid CPU oversubscription.",
                        key="relax_omp_threads",
                    )
                )

            try:
                total_threads = int(cfg_edit["relax"]["n_workers"]) * max(
                    int(cfg_edit["relax"]["tf_threads"]),
                    int(cfg_edit["relax"]["omp_threads"]),
                )
                st.caption(f"Rule of thumb: total CPU load ~ workers × threads ≈ **{total_threads}**")
            except Exception:
                pass

        else:
            col1, col2 = st.columns(2, vertical_alignment="bottom")

            with col1:
                cfg_edit["relax"]["gpu_id"] = int(
                    st.number_input(
                        "GPU ID",
                        min_value=0,
                        value=int(cfg_edit["relax"].get("gpu_id", DEFAULTS["relax"]["gpu_id"])),
                        step=1,
                        key="relax_gpu_id",
                    )
                )

            with col2:
                st.info("CUDA mode uses a single effective worker internally.")

            # keep these present in config even when hidden
            cfg_edit["relax"].setdefault("n_workers", DEFAULTS["relax"]["n_workers"])
            cfg_edit["relax"].setdefault("tf_threads", DEFAULTS["relax"]["tf_threads"])
            cfg_edit["relax"].setdefault("omp_threads", DEFAULTS["relax"]["omp_threads"])

        # -----------------------------
        # Caching
        # -----------------------------
        st.divider()
        st.subheader("Caching")

        cfg_edit["relax"]["skip_if_done"] = st.checkbox(
            "Skip relaxation if results already exist",
            value=bool(cfg_edit["relax"].get("skip_if_done", True)),
            help="If enabled, relaxation is skipped when ranking_relax.csv already exists in the folder.",
            key="relax_skip_if_done",
        )

        cfg_edit["relax"]["skip_candidate_if_done"] = st.checkbox(
            "Skip individual candidates already relaxed",
            value=bool(cfg_edit["relax"].get("skip_candidate_if_done", True)),
            help="If enabled, candidates with existing 02_relax/POSCAR and meta.json are skipped.",
            key="relax_skip_candidate_if_done",
        )

    # -----------------------------
    # FILTER
    # -----------------------------
    with st.expander("Filter", expanded=False):
        st.subheader("Select best relaxed candidates")

        st.caption(
            "Filters the relaxed candidates before running bandgap and formation energy. "
            "Choose either an energy window around the best candidate, or keep only the top-N lowest-energy structures."
        )

        st.divider()
        col1, col2 = st.columns([2, 3], vertical_alignment="bottom")

        with col1:
            fmode = st.selectbox(
                "Filter mode",
                CHOICES["filter.mode"],
                index=CHOICES["filter.mode"].index(cfg_edit["filter"].get("mode", "window")),
                help="Window: keep all candidates within an energy window above the minimum. Top-N: keep only N lowest-energy candidates.",
            )
            cfg_edit["filter"]["mode"] = fmode

        with col2:
            if fmode == "window":
                st.caption("**Window mode:** keeps everything within ΔE of the minimum relaxed energy.")
            else:
                st.caption("**Top-N mode:** keeps exactly N lowest-energy candidates.")

        st.divider()
        st.subheader("Threshold")

        if fmode == "window":
            cfg_edit["filter"]["window_meV"] = st.number_input(
                "Energy window ΔE [meV]",
                min_value=0.0,
                value=float(cfg_edit["filter"].get("window_meV", DEFAULTS["filter"]["window_meV"])),
                step=10.0,
                help="Candidates with relaxed energy ≤ (E_min + ΔE) are kept.",
            )
            # keep config consistent
            cfg_edit["filter"].setdefault("max_candidates", DEFAULTS["filter"]["max_candidates"])
        else:
            cfg_edit["filter"]["max_candidates"] = st.number_input(
                "Number of candidates (Top-N)",
                min_value=1,
                value=int(cfg_edit["filter"].get("max_candidates", DEFAULTS["filter"]["max_candidates"])),
                step=1,
                help="Keeps only the N lowest-energy relaxed candidates.",
            )
            # keep config consistent
            cfg_edit["filter"].setdefault("window_meV", DEFAULTS["filter"]["window_meV"])

        st.divider()
        st.subheader("Caching")

        cfg_edit["filter"]["skip_if_done"] = st.checkbox(
            "Skip filtering if results already exist",
            value=bool(cfg_edit["filter"].get("skip_if_done", True)),
            help="If enabled, filtering is skipped when ranking_relax_filtered.csv already exists in the folder.",
        )


    # -----------------------------
    # BANDGAP
    # -----------------------------
    with st.expander("Bandgap", expanded=False):
        st.subheader("Bandgap prediction")

        st.caption(
            "Predicts bandgaps for filtered candidates using ALIGNN. "
            "The parameters below control how the atomic graph is built before inference."
        )

        st.divider()
        st.subheader("Graph construction")

        col1, col2, col3 = st.columns([1, 1, 2], vertical_alignment="bottom")

        with col1:
            cfg_edit["bandgap"]["cutoff"] = st.number_input(
                "Cutoff radius [Å]",
                min_value=0.0,
                value=float(cfg_edit["bandgap"].get("cutoff", DEFAULTS["bandgap"]["cutoff"])),
                step=0.5,
                help="Neighbor search cutoff used to build edges in the atomic graph.",
            )

        with col2:
            cfg_edit["bandgap"]["max_neighbors"] = st.number_input(
                "Max neighbors",
                min_value=1,
                value=int(cfg_edit["bandgap"].get("max_neighbors", DEFAULTS["bandgap"]["max_neighbors"])),
                step=1,
                help="Maximum number of neighbors kept per atom (limits graph size).",
            )

        with col3:
            st.caption(
                "Typical values: cutoff 6–10 Å, max neighbors 12–24. "
                "Larger values may improve connectivity but increase runtime."
            )

        st.divider()
        st.subheader("Execution mode")

        cfg_edit.setdefault("bandgap", {})

        bandgap_device_options = ["cpu", "cuda"]
        current_device = str(cfg_edit["bandgap"].get("device", DEFAULTS["bandgap"]["device"])).lower()
        if current_device not in bandgap_device_options:
            current_device = "cpu"

        cfg_edit["bandgap"]["device"] = st.selectbox(
            "Device",
            bandgap_device_options,
            index=bandgap_device_options.index(current_device),
            help="Choose CPU parallel execution or CUDA GPU execution.",
            key="bandgap_device",
        )

        device = cfg_edit["bandgap"]["device"]

        if device == "cpu":
            cfg_edit["bandgap"]["n_workers"] = int(
                st.number_input(
                    "CPU workers",
                    min_value=1,
                    value=int(cfg_edit["bandgap"].get("n_workers", DEFAULTS["bandgap"]["n_workers"])),
                    step=1,
                    help="Number of parallel CPU processes for bandgap prediction.",
                    key="bandgap_n_workers",
                )
            )
            st.info("CPU mode uses parallel workers.")

        elif device == "cuda":
            col1, col2 = st.columns(2)

            with col1:
                cfg_edit["bandgap"]["gpu_id"] = int(
                    st.number_input(
                        "GPU ID",
                        min_value=0,
                        value=int(cfg_edit["bandgap"].get("gpu_id", DEFAULTS["bandgap"]["gpu_id"])),
                        step=1,
                        key="bandgap_gpu_id",
                    )
                )

            with col2:
                cfg_edit["bandgap"]["batch_size"] = int(
                    st.number_input(
                        "Batch size",
                        min_value=1,
                        value=int(cfg_edit["bandgap"].get("batch_size", DEFAULTS["bandgap"]["batch_size"])),
                        step=1,
                        help="Batch size for GPU inference.",
                        key="bandgap_batch_size",
                    )
                )

            cfg_edit["bandgap"]["n_workers"] = 1
            st.info("CUDA mode uses batched GPU inference.")
            
    # -----------------------------
    # FORMATION
    # -----------------------------
    with st.expander("Formation energy", expanded=False):
        st.subheader("Formation energy evaluation")

        st.write("Formation energy is computed as:")

        st.latex(
            r"E_{\mathrm{form}} = E_{\mathrm{doped}} - E_{\mathrm{pristine}}"
            r" - \sum_i n_i \mu_i + \sum_j n_j \mu_j"
        )

        st.write("where:")
        st.markdown(
            """
    - **$E_{\\mathrm{doped}}$**: relaxed doped structure energy  
    - **$E_{\\mathrm{pristine}}$**: pristine reference energy  
    - **$n_i,\\mu_i$**: number and chemical potential of dopants  
    - **$n_j,\\mu_j$**: number and chemical potential of removed host atoms  

    Chemical potentials are obtained from the **refs-build** stage.
            """
        )

        st.divider()
        st.subheader("Normalization scheme")

        norm = st.selectbox(
            "How to report formation energy",
            CHOICES["formation.normalize"],
            index=CHOICES["formation.normalize"].index(
                cfg_edit["formation"].get("normalize", "per_dopant")
            ),
            help="Controls how formation energy is reported in the final CSV/summary.",
        )
        cfg_edit["formation"]["normalize"] = norm

        # Show explanation ONLY for the selected normalization
        if norm == "per_dopant":
            st.info("**per_dopant**: compares stability per dopant atom (useful across different dopant counts).")
            st.latex(r"\frac{E_{\mathrm{form}}}{N_{\mathrm{dopant}}}")

        elif norm == "per_host":
            st.info("**per_host**: normalizes by the number of atoms (or host-sites, depending on your definition).")
            st.latex(r"\frac{E_{\mathrm{form}}}{N}")

        elif norm == "total":
            st.info("**total**: raw formation energy of the doped supercell (in eV).")
            st.latex(r"E_{\mathrm{form}} \; [\mathrm{eV}]")

    st.divider()
    st.subheader("Save input.toml")

    # Optional preview
    with st.expander("Preview TOML", expanded=False):
        st.code(toml.dumps(cfg_edit), language="toml")

    colS1, colS2 = st.columns([1, 2], vertical_alignment="center")

    with colS1:
        if st.button("💾 Save input.toml", use_container_width=True):
            try:
                save_toml(input_toml_path, cfg_edit)
                st.success(f"Saved: {input_toml_path}")
                st.session_state["cfg_edit"] = load_toml(input_toml_path)  # reload from disk
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save input.toml: {e}")

    with colS2:
        st.caption("After changing parameters above, click **Save input.toml** to write the file to disk.")

# ===== END PATCH =====

# -----------------------
# Page: Run
# -----------------------

elif tab == "Run":
    st.title("Workflow Execution")

    st.markdown(f"""
**Project root:** `{project_root}`  
**Config file:** `{input_toml_path}`
""")

    STEP_KEYS = ["refs", "generate", "scan", "relax", "filter", "bandgap", "formation", "collect"]

    st.divider()

    # -------------------------
    # Run mode (3 choices)
    # -------------------------
    st.subheader("Run mode")

    run_mode = st.radio(
        "What do you want to run?",
        options=["Full workflow", "Stage range", "Single stage"],
        horizontal=True,
        help=(
            "Full workflow runs everything. "
            "Stage range runs from a start stage to an end stage. "
            "Single stage runs exactly one chosen stage."
        ),
    )

    # Defaults
    step_from = "refs"
    step_until = "collect"
    only_steps = []

    if run_mode == "Full workflow":
        st.success("Will run: refs → collect (all stages).")
        step_from, step_until = "refs", "collect"

    elif run_mode == "Stage range":
        colA, colB = st.columns(2)
        with colA:
            step_from = st.selectbox("Start stage", options=STEP_KEYS, index=0)
        with colB:
            step_until = st.selectbox("End stage", options=STEP_KEYS, index=len(STEP_KEYS) - 1)

        # Optional: warn if user chose an invalid order
        if STEP_KEYS.index(step_from) > STEP_KEYS.index(step_until):
            st.error("Start stage must be before (or equal to) End stage.")
            st.stop()

        st.info(f"Will run: **{step_from} → {step_until}**")

    else:  # Single stage
        single = st.selectbox("Stage", options=STEP_KEYS, index=0)
        only_steps = [single]  # will use --only
        step_from, step_until = "refs", "collect"  # range irrelevant when using --only
        st.info(f"Will run only: **{single}**")

    st.divider()

    # -------------------------
    # Common options
    # -------------------------
    st.subheader("Options")

    col1, col2 = st.columns(2)
    with col1:
        dry_run = st.checkbox("Dry run (show planned steps only)", value=False)
    with col2:
        verbose = st.checkbox("Verbose logging", value=False)

    # Advanced controls (keep clean)
    with st.expander("Advanced controls", expanded=False):
        st.markdown("### Filter overrides (only relevant if filter runs)")
        filter_only = st.text_input("Filter only one composition (optional)", value="", placeholder="e.g. Sb5_Zr5")
        force = st.checkbox("Force recomputation (ignore existing outputs)", value=False)

        colF1, colF2 = st.columns(2)
        with colF1:
            window_mev = st.number_input("Energy window override (meV)", min_value=0.0, value=0.0, step=10.0)
        with colF2:
            topn = st.number_input("Top-N override", min_value=0, value=0, step=1)

        # If run_mode is Stage range, you may still want to let user specify --only subset
        # (optional; you can remove this if you want strict simplicity)
        if run_mode == "Stage range":
            extra_only = st.multiselect(
                "Run only selected stages (optional)",
                options=STEP_KEYS,
                default=[],
                help="If set, overrides the stage range and runs only these stages.",
            )
            if extra_only:
                only_steps = extra_only

    # -------------------------
    # Build command
    # -------------------------
    cmd = ["dopingflow", "run-all", "-c", str(input_toml_path), "--from", step_from, "--until", step_until]

    if only_steps:
        cmd += ["--only", ",".join(only_steps)]

    if dry_run:
        cmd += ["--dry-run"]
    if verbose:
        cmd += ["--verbose"]

    if filter_only.strip():
        cmd += ["--filter-only", filter_only.strip()]
    if force:
        cmd += ["--force"]
    if window_mev > 0:
        cmd += ["--window-mev", str(float(window_mev))]
    if topn > 0:
        cmd += ["--topn", str(int(topn))]

    st.subheader("Command preview")
    st.code(" ".join(cmd))

    # -------------------------
    # Run + Log
    # -------------------------
    log_path = project_root / "logs" / "gui_run.log"

    col_run, col_refresh = st.columns([1, 1])
    with col_run:
        if st.button("▶ Run", use_container_width=True):
            with st.spinner("Running workflow..."):
                rc = run_command(cmd, cwd=project_root, log_path=log_path)
            if rc == 0:
                st.success("Workflow finished successfully.")
            else:
                st.error(f"Workflow exited with return code {rc}.")

    with col_refresh:
        if st.button("🔄 Refresh log", use_container_width=True):
            pass

    st.subheader("Log tail")
    st.code(tail_text(log_path, n=10000))


# -----------------------
# Page: Results Explorer
# -----------------------
elif tab == "Results Explorer":
    from pathlib import Path
    from datetime import datetime
    import json

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import streamlit as st

    st.title("Results Explorer")

    # ============================================================
    # Load CSV
    # ============================================================
    st.subheader("Data source")

    default_csv = project_root / "results_database.csv"
    csv_path_str = st.text_input("Results CSV path", value=str(default_csv))
    csv_path = Path(csv_path_str).expanduser()
    if not csv_path.is_absolute():
        csv_path = (project_root / csv_path).resolve()

    if not csv_path.exists():
        st.error(f"CSV not found: `{csv_path}`")
        st.stop()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    if df.empty:
        st.warning("CSV loaded but is empty.")
        st.stop()

    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns from `{csv_path.name}`")

    # ============================================================
    # Generic filters (optional)
    # ============================================================
    numeric_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols_all = [c for c in df.columns if c not in numeric_cols_all]

    st.divider()
    st.subheader("Filters")

    df_f = df.copy()

    with st.expander("Filter controls", expanded=True):
        cat_col = st.selectbox(
            "Categorical filter column (optional)",
            options=["(none)"] + non_numeric_cols_all,
            index=0,
        )
        if cat_col != "(none)":
            vals = df_f[cat_col].dropna().astype(str).unique().tolist()
            vals = sorted(vals)[:1000]
            chosen = st.multiselect(f"Keep values in `{cat_col}`", options=vals, default=vals)
            if chosen:
                df_f = df_f[df_f[cat_col].astype(str).isin(set(chosen))]

        num_col = st.selectbox(
            "Numeric range filter column (optional)",
            options=["(none)"] + numeric_cols_all,
            index=0,
        )
        if num_col != "(none)" and len(df_f) > 0:
            col_min = float(np.nanmin(df_f[num_col].values))
            col_max = float(np.nanmax(df_f[num_col].values))
            if np.isfinite(col_min) and np.isfinite(col_max) and col_min != col_max:
                lo, hi = st.slider(
                    f"Range for `{num_col}`",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                )
                df_f = df_f[(df_f[num_col] >= lo) & (df_f[num_col] <= hi)]
            else:
                st.info(f"`{num_col}` has constant or non-finite range in current filtered set.")

    st.caption(f"Filtered rows: **{len(df_f):,}**")

    # Table
    st.subheader("Table")
    st.dataframe(df_f, use_container_width=True, height=360)
    st.download_button(
        "⬇ Download filtered CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="results_filtered.csv",
        mime="text/csv",
    )

    # ============================================================
    # Dopant parsing (based on your CSV)
    # ============================================================
    st.divider()
    st.subheader("Dopant-set plot studio (correct filtering + full hover)")

    has_json = "dopant_counts_json" in df_f.columns
    has_str = "dopant_counts" in df_f.columns

    if not (has_json or has_str):
        st.error("CSV must contain `dopant_counts_json` or `dopant_counts`.")
        st.stop()

    def parse_dopants_from_counts_json(s) -> dict:
        """
        s like: '{"Sb": 2, "Zr": 1}'  (as a string in CSV)
        Returns dict[str,int]
        """
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return {}
        try:
            d = json.loads(str(s))
            if isinstance(d, dict):
                # keep only positive counts
                out = {}
                for k, v in d.items():
                    try:
                        vv = int(v)
                    except Exception:
                        continue
                    if vv > 0:
                        out[str(k)] = vv
                return out
        except Exception:
            return {}
        return {}

    def parse_dopants_from_counts_str(s) -> dict:
        """
        s like: 'Sb:2;Zr:1'
        Returns dict[str,int]
        """
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return {}
        txt = str(s).strip()
        if not txt:
            return {}
        out = {}
        for part in txt.split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                # if someone writes "Sb" only
                out[part] = max(out.get(part, 0), 1)
                continue
            el, cnt = part.split(":", 1)
            el = el.strip()
            try:
                cnt_i = int(str(cnt).strip())
            except Exception:
                cnt_i = 1
            if el:
                out[el] = max(out.get(el, 0), cnt_i)
        return out

    def dopant_sig_from_dict(d: dict) -> str:
        keys = sorted(d.keys())
        return "+".join(keys) if keys else "Undoped"
    
    def to_float_or_none(value):
        try:
            value = str(value).strip()
            if value == "":
                return None
            return float(value)
        except Exception:
            return None    

    # Build parsing columns (JSON-safe strings only)
    dfp = df_f.copy()

    if has_json:
        dfp["_dopant_dict"] = dfp["dopant_counts_json"].apply(parse_dopants_from_counts_json)
        # fallback to dopant_counts if json empty
        if has_str:
            mask_empty = dfp["_dopant_dict"].apply(lambda d: len(d) == 0)
            dfp.loc[mask_empty, "_dopant_dict"] = dfp.loc[mask_empty, "dopant_counts"].apply(parse_dopants_from_counts_str)
    else:
        dfp["_dopant_dict"] = dfp["dopant_counts"].apply(parse_dopants_from_counts_str)

    dfp["dopant_sig"] = dfp["_dopant_dict"].apply(dopant_sig_from_dict)
    dfp["n_dopants"] = dfp["_dopant_dict"].apply(lambda d: len(d.keys()))
    dfp["dopant_list_str"] = dfp["_dopant_dict"].apply(lambda d: ", ".join(sorted(d.keys())) if d else "Undoped")

    # Universe of dopants
    all_elements = sorted({el for d in dfp["_dopant_dict"].values for el in d.keys()})

    # ============================================================
    # Correct filtering UI
    # ============================================================
    st.markdown("### Select structures")

    selected_classes = st.multiselect(
        "Dopant count class",
        options=["0 (undoped)", "1 (single)", "2 (double)", "3 (triple)", "4+ (>=4)"],
        default=["1 (single)", "2 (double)", "3 (triple)"],
    )

    required = set(
        st.multiselect(
            "Required dopants (optional)",
            options=all_elements,
            default=[],
            help=(
                "Rules:\n"
                "- For 1/2 dopants: EXACT match (only these dopants)\n"
                "- For >=3 dopants: CONTAINS (must include these, plus others allowed)\n"
                "- Leave empty: show ALL structures in the selected dopant-count class(es)"
            ),
        )
    )

    excluded = set(
        st.multiselect(
            "Exclude dopants (optional)",
            options=all_elements,
            default=[],
        )
    )

    def class_selected(n: int) -> bool:
        if n == 0:
            return "0 (undoped)" in selected_classes
        if n == 1:
            return "1 (single)" in selected_classes
        if n == 2:
            return "2 (double)" in selected_classes
        if n == 3:
            return "3 (triple)" in selected_classes
        return "4+ (>=4)" in selected_classes

    def keep_row(d: dict, n: int) -> bool:
        ds = set(d.keys())

        # class
        if not class_selected(n):
            return False

        # exclude always
        if excluded and (ds & excluded):
            return False

        # no required -> keep all in class
        if not required:
            return True

        # exact for 1/2
        if n in (1, 2):
            return ds == required

        # contains for >=3
        if n >= 3:
            return required.issubset(ds)

        # undoped only if required empty (handled)
        return False

    df_sel = dfp[dfp.apply(lambda r: keep_row(r["_dopant_dict"], int(r["n_dopants"])), axis=1)].copy()

    st.caption(f"Selected structures: **{len(df_sel):,}**")
    if df_sel.empty:
        st.warning("No matching structures.")
        st.stop()

    # ============================================================
    # Plot builder with FULL hover
    # ============================================================
    st.divider()
    st.subheader("Plot")

    numeric_cols_sel = [c for c in df_sel.columns if pd.api.types.is_numeric_dtype(df_sel[c])]
    if not numeric_cols_sel:
        st.error("No numeric columns available to plot.")
        st.stop()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        x_col = st.selectbox("X axis", options=numeric_cols_sel, index=0)
    with col2:
        y_col = st.selectbox("Y axis", options=numeric_cols_sel, index=min(1, len(numeric_cols_sel) - 1))
    with col3:
        color_by = st.selectbox("Color by", options=["dopant_sig", "n_dopants"] + [c for c in df_sel.columns if c not in numeric_cols_sel], index=0)

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

    # FULL hover: include all columns EXCEPT non-serializable/internal ones
    internal_cols = {"_dopant_dict"}  # <- dict is serializable, but can get large; keep it out of hover
    hover_cols = [c for c in df_sel.columns.tolist() if c not in internal_cols]

    fig = px.scatter(
        df_sel,
        x=x_col,
        y=y_col,
        color=color_by if color_by in df_sel.columns else "dopant_sig",
        hover_name="candidate" if "candidate" in df_sel.columns else ("composition_tag" if "composition_tag" in df_sel.columns else None),
        hover_data=hover_cols,  # ✅ everything safe
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

    cfg_now = load_toml(input_toml_path)
    outdir_now = cfg_now.get("structure", {}).get("outdir", "random_structures")
    proj = ProjectIndex(root=project_root, outdir=project_root / outdir_now)

    comps = proj.compositions()
    if not comps:
        st.warning(f"No compositions found in `{proj.outdir}`.")
        st.stop()

    comp = st.selectbox("Composition", options=comps)
    base = proj.composition_path(comp)

    selected = proj.selected_candidates(comp)
    if selected:
        cand_name = st.selectbox("Candidate", options=selected)
    else:
        cand_dirs = sorted([p.name for p in base.glob("candidate_*") if p.is_dir()])
        if not cand_dirs:
            st.warning("No candidate folders found yet.")
            st.stop()
        cand_name = st.selectbox("Candidate", options=cand_dirs)

    cand_dir = proj.find_candidate_dir(comp, cand_name)
    if cand_dir is None:
        st.error("Candidate folder not found.")
        st.stop()

    files = proj.find_structure_files(cand_dir)
    st.write({k: str(v) for k, v in files.items()})

    colL, colR = st.columns(2)

    with colL:
        st.subheader("Before")
        if "before" in files:
            show_structure(files["before"], title=f"{comp} / {cand_name} — before")
        else:
            st.info("No 'before' structure file found.")

    with colR:
        st.subheader("After")
        if "after" in files:
            show_structure(files["after"], title=f"{comp} / {cand_name} — after")
        else:
            st.info("No 'after' structure file found.")

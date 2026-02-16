# gui/app.py
from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import toml

# Make sure gui/ is importable
GUI_DIR = Path(__file__).resolve().parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from gui_config import SUPER_CELL_PRESETS, DOPING_MODE_CHOICES, ALLOWED_DOPANTS, RUN_PRESETS, DEFAULTS, CHOICES
from io_project import ProjectIndex
from view_structure import show_structure


st.set_page_config(page_title="dopingflow GUI", layout="wide")


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

from gui_config import DEFAULTS, CHOICES  # add this import


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

    # -----------------------------
    # STRUCTURE
    # -----------------------------
    
    with st.expander("Structure", expanded=True):
        st.subheader("Base structure & output")

        col1, col2 = st.columns([3, 2], vertical_alignment="bottom")

        with col1:
            base_poscar = st.text_input(
                "Base structure (POSCAR)",
                value=cfg_edit["structure"].get("base_poscar", DEFAULTS["structure"]["base_poscar"]),
                placeholder="e.g. reference_structures/base.POSCAR",
                help="Path to the pristine structure file (VASP POSCAR). Relative to project root or absolute.",
            )

        with col2:
            outdir_ui = st.text_input(
                "Output directory",
                value=cfg_edit["structure"].get("outdir", DEFAULTS["structure"]["outdir"]),
                placeholder="random_structures",
                help="Folder where the workflow writes generated structures (relative to project root).",
            )

        # Resolve and validate base POSCAR
        base_path = (project_root / base_poscar).resolve() if not Path(base_poscar).is_absolute() else Path(base_poscar)

        if base_poscar.strip() == "":
            st.warning("Please provide a POSCAR path.")
        elif not base_path.exists():
            st.error(f"Base POSCAR not found: `{base_path}`")
        else:
            st.success(f"Base POSCAR found: `{base_path}`")

        cfg_edit["structure"]["base_poscar"] = base_poscar
        cfg_edit["structure"]["outdir"] = outdir_ui

        st.divider()
        st.subheader("Supercell")

        current_supercell = cfg_edit["structure"].get("supercell", [1, 1, 1])
        if not isinstance(current_supercell, list) or len(current_supercell) != 3:
            current_supercell = [1, 1, 1]

        col_a, col_b, col_c, col_info = st.columns([1, 1, 1, 2], vertical_alignment="bottom")

        with col_a:
            a = st.number_input("a", min_value=1, step=1, value=int(current_supercell[0]))
        with col_b:
            b = st.number_input("b", min_value=1, step=1, value=int(current_supercell[1]))
        with col_c:
            c = st.number_input("c", min_value=1, step=1, value=int(current_supercell[2]))

        cfg_edit["structure"]["supercell"] = [int(a), int(b), int(c)]

        with col_info:
            vol_factor = int(a) * int(b) * int(c)
            st.caption(f"Supercell: `{cfg_edit['structure']['supercell']}`  |  size factor: ×{vol_factor}")



    # -----------------------------
    # REFERENCES (Step 00)
    # -----------------------------
    with st.expander("References (refs-build)", expanded=False):
        st.subheader("Reference source")

        col1, col2 = st.columns([2, 3], vertical_alignment="bottom")

        with col1:
            src = st.selectbox(
                "Source",
                CHOICES["references.source"],
                index=CHOICES["references.source"].index(cfg_edit["references"].get("source", "local")),
                help='Where to get elemental bulk references: local folder or Materials Project (mp).',
            )
            cfg_edit["references"]["source"] = src

        with col2:
            bulk_dir = st.text_input(
                "Reference folder",
                value=cfg_edit["references"].get("bulk_dir", DEFAULTS["references"]["bulk_dir"]),
                placeholder="reference_structures/",
                help="Folder containing element POSCARs, e.g. Sn.POSCAR, Sb.POSCAR, Ti.POSCAR …",
            )
            cfg_edit["references"]["bulk_dir"] = bulk_dir

        # Validate local folder
        if src == "local":
            bulk_path = (project_root / bulk_dir).resolve() if not Path(bulk_dir).is_absolute() else Path(bulk_dir)
            if bulk_dir.strip() == "":
                st.warning("Please provide a reference folder path.")
            elif not bulk_path.exists():
                st.error(f"Reference folder not found: `{bulk_path}`")
            elif not bulk_path.is_dir():
                st.error(f"Reference folder path is not a directory: `{bulk_path}`")
            else:
                # show how many references it contains
                n_poscar = len(list(bulk_path.glob("*.POSCAR")))
                st.success(f"Reference folder found: `{bulk_path}`  |  POSCAR files: {n_poscar}")

        st.divider()
        st.subheader("Relaxation & caching")

        colA, colB, colC = st.columns([1, 1, 2], vertical_alignment="bottom")

        with colA:
            cfg_edit["references"]["fmax"] = st.number_input(
                "fmax",
                min_value=0.0,
                value=float(cfg_edit["references"].get("fmax", DEFAULTS["references"]["fmax"])),
                step=0.01,
                help="Force threshold used when relaxing reference bulks with M3GNet+FIRE.",
            )

        with colB:
            cfg_edit["references"]["skip_if_done"] = st.checkbox(
                "Skip if done",
                value=bool(cfg_edit["references"].get("skip_if_done", True)),
                help="If enabled, refs-build will not recompute if reference_energies.json already exists.",
            )

        with colC:
            st.caption(
                    "Maximum force (eV/Å) used to relax reference unit cells "
                    "with M3GNet + FIRE. Relaxation stops below this threshold."
            )

        # Optional MP IDs dict editor
        if src == "mp":
            st.divider()
            st.subheader("Materials Project IDs (optional)")

            st.caption("One per line: `Element=mp-xxxx` (only needed for elements you want to fetch from MP).")

            mp_ids_text = st.text_area(
                "mp_ids",
                value="\n".join(
                    [f"{k}={v}" for k, v in (cfg_edit["references"].get("mp_ids", {}) or {}).items()]
                ),
                height=140,
                placeholder="Sn=mp-xxxxx\nSb=mp-yyyyy\nTi=mp-zzzzz",
            )

            mp_ids = {}
            for line in mp_ids_text.splitlines():
                line = line.strip()
                if not line or "=" not in line:
                    continue
                k, v = [x.strip() for x in line.split("=", 1)]
                if k and v:
                    mp_ids[k] = v
            cfg_edit["references"]["mp_ids"] = mp_ids
        else:
            cfg_edit["references"].pop("mp_ids", None)


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
            cfg_edit["doping"]["allowed_totals"] = st.multiselect(
                "Allowed total doping (%)",
                options=[0, 5, 10, 15, 20, 25, 30],
                default=cfg_edit["doping"].get("allowed_totals", DEFAULTS["doping"]["allowed_totals"]),
                help="Total dopant percentage (sum of dopants) used to generate compositions.",
            )

            cfg_edit["doping"]["levels"] = st.multiselect(
                "Allowed levels (%) per dopant",
                options=[1, 2, 3, 4, 5, 10, 15, 20],
                default=cfg_edit["doping"].get("levels", DEFAULTS["doping"]["levels"]),
                help="Discrete percentage values assigned to each dopant during enumeration.",
            )

            # compositions unused in enumerate mode
            cfg_edit["doping"].setdefault("compositions", [])


    # -----------------------------
    # SCAN
    # -----------------------------
    with st.expander("Scan", expanded=False):
        st.subheader("Screen candidates (Single-point energy)")

        st.caption(
            "Enumerates symmetry-unique doped configurations and ranks them using M3GNet single-point energies. "
            "The best candidates are kept for relaxation."
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
                help="Number of lowest-energy configurations retained after M3GNet single-point scan.",
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
                min_value=0.0,
                value=float(cfg_edit["scan"].get("symprec", DEFAULTS["scan"]["symprec"])),
                step=1e-3,
                format="%.6f",
                help="Tolerance used when identifying symmetry-unique configurations.",
            )

        with colB:
            st.caption(
                "Smaller values are stricter (more unique structures, slower). "
                "Typical: 1e-3 to 1e-2."
            )

        st.divider()
        st.subheader("Parallel execution")

        colP1, colP2, colP3 = st.columns([1, 1, 2], vertical_alignment="bottom")

        with colP1:
            cfg_edit["scan"]["nproc"] = st.number_input(
                "Processes (nproc)",
                min_value=1,
                value=int(cfg_edit["scan"].get("nproc", DEFAULTS["scan"]["nproc"])),
                step=1,
                help="Number of parallel processes used for M3GNet energy evaluation.",
            )

        with colP2:
            cfg_edit["scan"]["chunksize"] = st.number_input(
                "Chunk size",
                min_value=1,
                value=int(cfg_edit["scan"].get("chunksize", DEFAULTS["scan"]["chunksize"])),
                step=1,
                help="Work chunk size per process. Larger chunks reduce overhead; too large can increase memory spikes.",
            )

        with colP3:
            st.caption("Rule of thumb: start with chunksize 50–200, then tune for your machine.")

        st.divider()
        st.subheader("Safety limits")

        colS1, colS2 = st.columns(2, vertical_alignment="bottom")

        with colS1:
            cfg_edit["scan"]["max_enum"] = st.number_input(
                "Max enumerated configs",
                min_value=1,
                value=int(cfg_edit["scan"].get("max_enum", DEFAULTS["scan"]["max_enum"])),
                step=1000,
                help="Hard cap on how many raw configurations may be generated before aborting.",
            )

        with colS2:
            cfg_edit["scan"]["max_unique"] = st.number_input(
                "Max symmetry-unique configs",
                min_value=1,
                value=int(cfg_edit["scan"].get("max_unique", DEFAULTS["scan"]["max_unique"])),
                step=1000,
                help="Hard cap on symmetry-unique configurations (prevents exploding output size).",
            )

        st.divider()
        st.subheader("Sublattice definition")

        anions = cfg_edit["scan"].get("anion_species", DEFAULTS["scan"]["anion_species"])
        if not isinstance(anions, list):
            anions = DEFAULTS["scan"]["anion_species"]

        anion_text = st.text_input(
            "Anion species (comma-separated)",
            value=",".join(anions),
            help="Elements treated as anions (excluded from substitution). Example for oxides: O",
            placeholder="e.g. O",
        )
        cfg_edit["scan"]["anion_species"] = [x.strip() for x in anion_text.split(",") if x.strip()]

        st.divider()
        st.subheader("Caching")

        cfg_edit["scan"]["skip_if_done"] = st.checkbox(
            "Skip scan if results already exist",
            value=bool(cfg_edit["scan"].get("skip_if_done", True)),
            help="If enabled, scan is skipped when ranking_scan.csv already exists in the folder.",
        )



    # -----------------------------
    # RELAX
    # -----------------------------
    with st.expander("Relax", expanded=False):
        st.subheader("Doped Structure relaxation")

        st.caption(
            "Relaxes the selected candidates using the M3GNet potential. "
            "You can control the force convergence threshold and parallel execution."
        )

        st.divider()
        st.subheader("Convergence")

        col1, col2 = st.columns([1, 2], vertical_alignment="bottom")
        with col1:
            cfg_edit["relax"]["fmax"] = st.number_input(
                "Max force (fmax) [eV/Å]",
                min_value=0.0,
                value=float(cfg_edit["relax"].get("fmax", DEFAULTS["relax"]["fmax"])),
                step=0.01,
                help="Relaxation stops when the maximum atomic force drops below this threshold.",
            )
        with col2:
            st.caption(
                "Typical values: 0.05 eV/Å (fast screening) to 0.02 eV/Å (tighter relaxation)."
            )

        st.divider()
        st.subheader("Parallelism & performance")

        colA, colB, colC = st.columns(3, vertical_alignment="bottom")

        with colA:
            cfg_edit["relax"]["n_workers"] = st.number_input(
                "Workers",
                min_value=1,
                value=int(cfg_edit["relax"].get("n_workers", DEFAULTS["relax"]["n_workers"])),
                step=1,
                help="Number of parallel relaxation processes (one candidate per worker).",
            )

        with colB:
            cfg_edit["relax"]["tf_threads"] = st.number_input(
                "TensorFlow threads / worker",
                min_value=1,
                value=int(cfg_edit["relax"].get("tf_threads", DEFAULTS["relax"]["tf_threads"])),
                step=1,
                help="TensorFlow threads per worker. Keep small (often 1) when using multiple workers.",
            )

        with colC:
            cfg_edit["relax"]["omp_threads"] = st.number_input(
                "OpenMP threads / worker",
                min_value=1,
                value=int(cfg_edit["relax"].get("omp_threads", DEFAULTS["relax"]["omp_threads"])),
                step=1,
                help="OpenMP threads per worker. Keep small to avoid CPU oversubscription.",
            )

        # Quick computed hint
        try:
            total_threads = int(cfg_edit["relax"]["n_workers"]) * max(
                int(cfg_edit["relax"]["tf_threads"]), int(cfg_edit["relax"]["omp_threads"])
            )
            st.caption(f"Rule of thumb: total CPU load ~ workers × threads ≈ **{total_threads}**")
        except Exception:
            pass

        st.divider()
        st.subheader("Caching")

        cfg_edit["relax"]["skip_if_done"] = st.checkbox(
            "Skip relaxation if results already exist",
            value=bool(cfg_edit["relax"].get("skip_if_done", True)),
            help="If enabled, relaxation is skipped when ranking_relax.csv already exists in the folder.",
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
        st.subheader("Caching")

        cfg_edit["bandgap"]["skip_if_done"] = st.checkbox(
            "Skip bandgap prediction if results already exist",
            value=bool(cfg_edit["bandgap"].get("skip_if_done", True)),
            help="If enabled, bandgap prediction is skipped when output files already exist.",
        )


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
        if norm == "per dopant":
            st.info("**per_dopant**: compares stability per dopant atom (useful across different dopant counts).")
            st.latex(r"\frac{E_{\mathrm{form}}}{N_{\mathrm{dopant}}}")

        elif norm == "per host":
            st.info("**per_host**: normalizes by the number of host sites in the pristine supercell.")
            st.latex(r"\frac{E_{\mathrm{form}}}{N_{\mathrm{host}}}")

        elif norm == "total":
            st.info("**total**: raw formation energy of the doped supercell (in eV).")
            st.latex(r"E_{\mathrm{form}} \; [\mathrm{eV}]")



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
    import numpy as np
    import plotly.express as px

    st.title("Results Explorer")

    # -----------------------------
    # Locate CSV
    # -----------------------------
    st.subheader("Data source")

    default_csv = project_root / "results_database.csv"
    csv_path_str = st.text_input(
        "Results CSV path",
        value=str(default_csv),
        help="Path to a CSV file produced by dopingflow (default: results_database.csv in project root).",
    )
    csv_path = Path(csv_path_str).expanduser()
    if not csv_path.is_absolute():
        csv_path = (project_root / csv_path).resolve()

    if not csv_path.exists():
        st.error(f"CSV not found: `{csv_path}`")
        st.stop()

    # -----------------------------
    # Read CSV
    # -----------------------------
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    if df.empty:
        st.warning("CSV loaded but is empty.")
        st.stop()

    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns from `{csv_path.name}`")

    # -----------------------------
    # Basic cleaning helpers
    # -----------------------------
    # Identify numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    # -----------------------------
    # Sidebar filters
    # -----------------------------
    st.divider()
    st.subheader("Filters")

    with st.expander("Filter controls", expanded=True):
        # Choose one categorical column to filter (optional)
        cat_col = st.selectbox(
            "Categorical filter column (optional)",
            options=["(none)"] + non_numeric_cols,
            index=0,
        )

        df_f = df.copy()

        if cat_col != "(none)":
            # limit unique values to keep UI snappy
            values = df_f[cat_col].dropna().astype(str).unique().tolist()
            values = sorted(values)[:500]
            chosen = st.multiselect(f"Keep values in `{cat_col}`", options=values, default=values)
            if chosen:
                df_f = df_f[df_f[cat_col].astype(str).isin(set(chosen))]

        # Numeric range filter (optional)
        num_col = st.selectbox(
            "Numeric range filter column (optional)",
            options=["(none)"] + numeric_cols,
            index=0,
        )

        if num_col != "(none)":
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

    # -----------------------------
    # Table view
    # -----------------------------
    st.subheader("Table")
    st.dataframe(df_f, use_container_width=True, height=360)

    # Optional download
    st.download_button(
        "⬇ Download filtered CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="results_filtered.csv",
        mime="text/csv",
    )

    # -----------------------------
    # Plot builder
    # -----------------------------
    st.divider()
    st.subheader("Plot builder")

    plot_type = st.selectbox(
        "Plot type",
        options=["Scatter", "Line", "Histogram", "Box"],
        index=0,
    )

    # Common aesthetics
    color_col = st.selectbox("Color (optional)", options=["(none)"] + df_f.columns.tolist(), index=0)
    hover_cols = st.multiselect(
        "Hover columns",
        options=df_f.columns.tolist(),
        default=[c for c in ["composition", "candidate", "E_form_eV", "bandgap_eV"] if c in df_f.columns],
    )

    if plot_type in ("Scatter", "Line"):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            x_col = st.selectbox("X axis", options=numeric_cols if numeric_cols else df_f.columns.tolist())
        with col2:
            y_col = st.selectbox("Y axis", options=numeric_cols if numeric_cols else df_f.columns.tolist())
        with col3:
            size_col = st.selectbox("Size (optional)", options=["(none)"] + numeric_cols, index=0)

        symbol_col = st.selectbox("Marker symbol (optional)", options=["(none)"] + non_numeric_cols, index=0)

        fig = px.scatter(
            df_f,
            x=x_col,
            y=y_col,
            color=None if color_col == "(none)" else color_col,
            size=None if size_col == "(none)" else size_col,
            symbol=None if symbol_col == "(none)" else symbol_col,
            hover_data=hover_cols if hover_cols else None,
        ) if plot_type == "Scatter" else px.line(
            df_f.sort_values(by=x_col),
            x=x_col,
            y=y_col,
            color=None if color_col == "(none)" else color_col,
            hover_data=hover_cols if hover_cols else None,
        )

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Histogram":
        col1, col2 = st.columns([2, 1])
        with col1:
            x_col = st.selectbox("Column", options=numeric_cols if numeric_cols else df_f.columns.tolist())
        with col2:
            nbins = st.number_input("Bins", min_value=5, max_value=300, value=40, step=5)

        fig = px.histogram(
            df_f,
            x=x_col,
            color=None if color_col == "(none)" else color_col,
            nbins=int(nbins),
            hover_data=hover_cols if hover_cols else None,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # Box
        col1, col2 = st.columns([1, 1])
        with col1:
            y_col = st.selectbox("Y (numeric)", options=numeric_cols if numeric_cols else df_f.columns.tolist())
        with col2:
            x_col = st.selectbox("Group by (optional)", options=["(none)"] + non_numeric_cols, index=0)

        fig = px.box(
            df_f,
            y=y_col,
            x=None if x_col == "(none)" else x_col,
            color=None if color_col == "(none)" else color_col,
            hover_data=hover_cols if hover_cols else None,
        )
        st.plotly_chart(fig, use_container_width=True)


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

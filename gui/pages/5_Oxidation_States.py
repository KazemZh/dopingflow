from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import toml


STRATEGY_METHODS = {
    "structural": ["bond-valence"],
    "ml": ["toss-gnn", "chgnet", "bertos"],
    "dft": ["dft-auto", "dft-electronic", "bader", "wannier", "eos"],
}
ALL_METHODS = [
    "bond-valence",
    "toss-gnn",
    "chgnet",
    "bertos",
    "dft-auto",
    "dft-electronic",
    "bader",
    "wannier",
    "eos",
]
DFT_METHODS = ["dft-auto", "dft-electronic", "bader", "wannier", "eos"]


st.set_page_config(page_title="Oxidation-state analysis", layout="wide")
st.title("Oxidation-state analysis")
st.caption(
    "Configure and run site/composition oxidation analysis independently of the MLFF "
    "used to relax the structures. Parent and oxygen-vacancy structures can be analyzed together."
)

project_root = Path(
    st.sidebar.text_input(
        "Project root",
        value=str(Path.cwd()),
        key="oxidation_project_root",
    )
).expanduser().resolve()
config_path = project_root / "input.toml"

if not config_path.exists():
    st.error(f"No input.toml found at {config_path}")
    st.stop()

cfg = toml.load(str(config_path))
oxidation = dict(cfg.get("oxidation", {}) or {})

# Migrate the legacy GUI-generated GPAW output root. The migration is limited
# to the old default name so unrelated custom paths remain untouched.
dft_saved = oxidation.get("dft_electronic")
if isinstance(dft_saved, dict):
    dft_saved = dict(dft_saved)
    if str(dft_saved.get("output_root", "")).strip() == "gpaw_oxidation":
        dft_saved["output_root"] = "dft_oxidation"
    oxidation["dft_electronic"] = dft_saved

bader_saved = oxidation.get("bader")
if isinstance(bader_saved, dict):
    bader_saved = dict(bader_saved)
    if str(bader_saved.get("output_root", "")).strip() == "gpaw_oxidation":
        inherited_root = "dft_oxidation"
        current_dft = oxidation.get("dft_electronic")
        if isinstance(current_dft, dict):
            inherited_root = str(current_dft.get("output_root") or inherited_root).strip()
        bader_saved["output_root"] = inherited_root
    oxidation["bader"] = bader_saved


def _table(name: str) -> dict:
    value = oxidation.get(name, {}) or {}
    return dict(value) if isinstance(value, dict) else {}


def _parse_command(text: str) -> list[str]:
    text = text.strip()
    return shlex.split(text) if text else []


def _command_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return shlex.join(str(item) for item in value)
    return str(value or "")


def _parse_json_mapping(text: str, label: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        st.error(f"{label} must be valid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        st.error(f"{label} must be a JSON object.")
        return {}
    return value


st.subheader("Strategy and targets")
left, middle, right = st.columns(3)
with left:
    enabled = st.checkbox(
        "Enable oxidation stage",
        value=bool(oxidation.get("enabled", True)),
    )
with middle:
    strategy_options = ["structural", "ml", "dft", "combined"]
    strategy = str(oxidation.get("strategy", "structural")).strip().lower()
    if strategy not in strategy_options:
        strategy = "structural"
    strategy = st.selectbox(
        "Strategy",
        strategy_options,
        index=strategy_options.index(strategy),
        help="Use combined only when selecting methods from more than one strategy group.",
    )
with right:
    fail_fast = st.checkbox(
        "Fail fast",
        value=bool(oxidation.get("fail_fast", False)),
        help="Normally an unavailable optional method is recorded and the remaining methods continue.",
    )

allowed_methods = ALL_METHODS if strategy == "combined" else STRATEGY_METHODS[strategy]
existing_methods = oxidation.get("methods", [])
if isinstance(existing_methods, str):
    existing_methods = [item.strip() for item in existing_methods.split(",") if item.strip()]
existing_methods = [m for m in existing_methods if m in allowed_methods]
if not existing_methods and strategy != "combined":
    existing_methods = ["dft-auto"] if strategy == "dft" else list(allowed_methods)

methods = st.multiselect(
    "Methods",
    options=allowed_methods,
    default=existing_methods,
    help=(
        "Each method is retained independently. Combined mode reports agreement/disagreement; "
        "it does not average labels or use majority voting."
    ),
)
if strategy == "combined" and not methods:
    st.error("Combined strategy requires at least one explicitly selected method.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    include_vacancy_free = st.checkbox(
        "Vacancy-free parents",
        value=bool(oxidation.get("include_vacancy_free", True)),
    )
with c2:
    include_oxygen_vacancies = st.checkbox(
        "O-vacancy structures",
        value=bool(oxidation.get("include_oxygen_vacancies", True)),
    )
with c3:
    mapping_tolerance = st.number_input(
        "Parent mapping tolerance (Å)",
        min_value=0.01,
        value=float(oxidation.get("mapping_tolerance", 1.2)),
        step=0.1,
    )
with c4:
    output_dir = st.text_input(
        "Output directory",
        value=str(oxidation.get("output_dir", "06_oxidation")),
        help="Relative paths are created under the oxidation source root.",
    )

source_default = str((cfg.get("structure", {}) or {}).get("outdir", "random_structures"))
source_root = st.text_input(
    "Source root",
    value=str(oxidation.get("source_root", source_default)),
    help="Root containing selected relaxed parents and, when enabled, vacancies_database.json.",
)

saved_target_include = oxidation.get("target_include", [])
if isinstance(saved_target_include, str):
    saved_target_include = [
        item.strip() for item in saved_target_include.split(",") if item.strip()
    ]
elif not isinstance(saved_target_include, (list, tuple)):
    saved_target_include = []
target_include_text = st.text_input(
    "Target selector(s) (optional)",
    value=", ".join(str(item) for item in saved_target_include),
    help=(
        "Leave empty to analyze every discovered target. Use exact target IDs such as "
        "Sb5_Ti2p5/candidate_014, safe IDs such as Sb5_Ti2p5__candidate_014, or shell-style "
        "wildcards such as Sb5_Ti2p5/*. Multiple selectors can be comma-separated."
    ),
)
target_include = list(
    dict.fromkeys(
        item.strip() for item in target_include_text.split(",") if item.strip()
    )
)
if target_include:
    st.info(
        f"Target filtering is active: only structures matching {target_include} will be analyzed."
    )

st.info(
    "Bader charge, DOS/orbital populations, magnetic moments, and static Wannier centers remain "
    "supporting descriptors when used alone. The dft-auto method combines them with local bond-valence "
    "chemistry into an automated per-atom oxidation-state suggestion, confidence, and explicit electronic compensation."
)


if "bond-valence" in methods:
    with st.expander("Bond valence (pymatgen)", expanded=False):
        bv = _table("bond_valence")
        b1, b2 = st.columns(2)
        bv["symm_tol"] = float(
            b1.number_input("symm_tol", min_value=0.0, value=float(bv.get("symm_tol", 0.1)), step=0.05)
        )
        bv["max_radius"] = float(
            b2.number_input("max_radius (Å)", min_value=0.1, value=float(bv.get("max_radius", 4.0)), step=0.1)
        )
        oxidation["bond_valence"] = bv

if "toss-gnn" in methods:
    with st.expander("TOSS-GNN", expanded=True):
        toss = _table("toss_gnn")
        toss["repo_path"] = st.text_input(
            "Local TOSS repository",
            value=str(toss.get("repo_path", "")),
            key="oxidation_toss_gnn_repo",
        )
        t1, t2 = st.columns(2)
        toss["lp_checkpoint"] = t1.text_input(
            "LP checkpoint",
            value=str(toss.get("lp_checkpoint", "models/pyg_Hetero_GCN_s_0608.pth")),
        )
        toss["nc_checkpoint"] = t2.text_input(
            "NC checkpoint",
            value=str(toss.get("nc_checkpoint", "models/pyg_GCN_s_0609.pth")),
        )
        toss["device"] = "cpu"
        st.warning(
            "The verified upstream TOSS-GNN interface is CPU-only in this adapter. CUDA is not silently substituted."
        )
        oxidation["toss_gnn"] = toss

if "chgnet" in methods:
    with st.expander("CHGNet magnetic-moment analysis", expanded=False):
        chgnet = _table("chgnet")
        c1, c2 = st.columns(2)
        chgnet["model_name"] = c1.text_input(
            "Model name", value=str(chgnet.get("model_name", "0.3.0"))
        )
        device_options = ["cpu", "cuda"]
        current_device = str(chgnet.get("device", "cpu")).lower()
        if current_device not in device_options:
            current_device = "cpu"
        chgnet["device"] = c2.selectbox(
            "Device", device_options, index=device_options.index(current_device), key="oxidation_chgnet_device"
        )
        chgnet["use_upstream_mn_mapping"] = st.checkbox(
            "Use verified upstream Mn moment → oxidation mapping",
            value=bool(chgnet.get("use_upstream_mn_mapping", True)),
        )
        chgnet["use_absolute_moment"] = st.checkbox(
            "Use absolute moment for user-defined mappings",
            value=bool(chgnet.get("use_absolute_moment", False)),
            help="Off reproduces the upstream raw-moment Mn helper semantics.",
        )
        mapping_text = json.dumps(chgnet.get("moment_oxidation_ranges", {}), indent=2)
        mapping_text = st.text_area(
            "Optional element-specific moment ranges (JSON)",
            value=mapping_text,
            height=120,
            help='Example shape: {"Fe": [[min_moment, max_moment, oxidation_state]]}. Add only validated mappings.',
        )
        chgnet["moment_oxidation_ranges"] = _parse_json_mapping(mapping_text, "Moment oxidation ranges")
        st.caption(
            "Magnetic moments remain separate from formal oxidation labels. Near-zero moments are not used "
            "to distinguish Sn2+/Sn4+ or Sb3+/Sb5+."
        )
        oxidation["chgnet"] = chgnet

if "bertos" in methods:
    with st.expander("BERTOS composition-level prediction", expanded=False):
        bertos = _table("bertos")
        bertos["repo_path"] = st.text_input("Local BERTOS repository", value=str(bertos.get("repo_path", "")))
        b1, b2 = st.columns(2)
        bertos["model_path"] = b1.text_input(
            "Extracted model directory", value=str(bertos.get("model_path", "trained_models/ICSD_CN"))
        )
        bertos["tokenizer_path"] = b2.text_input(
            "Tokenizer path", value=str(bertos.get("tokenizer_path", "tokenizer"))
        )
        device_options = ["cpu", "cuda"]
        current_device = str(bertos.get("device", "cpu")).lower()
        if current_device not in device_options:
            current_device = "cpu"
        bertos["device"] = st.selectbox(
            "Device", device_options, index=device_options.index(current_device), key="oxidation_bertos_device"
        )
        st.warning(
            "BERTOS is composition-token level, not crystallographic-site level. Different vacancy arrangements "
            "at the same composition are indistinguishable to this method."
        )
        oxidation["bertos"] = bertos


def dft_method_panel(method: str, table_name: str, *, extra: str = "") -> None:
    settings = _table(table_name)
    settings["output_root"] = st.text_input(
        "Per-target DFT output root",
        value=str(settings.get("output_root", "dft_oxidation")),
        key=f"oxidation_{table_name}_root",
    )
    settings["execute"] = st.checkbox(
        "Execute a new external calculation",
        value=bool(settings.get("execute", False)),
        key=f"oxidation_{table_name}_execute",
        help="Off means post-process existing outputs only.",
    )
    command_text = st.text_input(
        "External command",
        value=_command_text(settings.get("command", [])),
        key=f"oxidation_{table_name}_command",
        disabled=not settings["execute"],
        help="Required when execute is enabled. The command is tokenized with shell-like quoting; no shell is used.",
    )
    settings["command"] = _parse_command(command_text)
    if settings["execute"] and not settings["command"]:
        st.error(f"{method}: execute is enabled but no command is configured.")
    if extra:
        st.caption(extra)
    oxidation[table_name] = settings


if "dft-auto" in methods:
    with st.expander("Automatic DFT oxidation-state assignment", expanded=True):
        auto = _table("dft_auto")
        inherited = _table("dft_electronic")
        auto["output_root"] = st.text_input(
            "Automatic DFT output root",
            value=str(auto.get("output_root") or inherited.get("output_root") or "dft_oxidation"),
            key="oxidation_dft_auto_root",
        )
        a1, a2 = st.columns(2)
        auto["execute"] = a1.checkbox(
            "Run missing DFT/Bader/Wannier steps automatically",
            value=bool(auto.get("execute", False)),
            key="oxidation_dft_auto_execute",
            help="Existing compatible files are reused by default. Turn this on only when new calculations may be launched.",
        )
        auto["reuse_existing"] = a2.checkbox(
            "Reuse existing compatible outputs",
            value=bool(auto.get("reuse_existing", True)),
            key="oxidation_dft_auto_reuse",
        )

        st.markdown("**Core GPAW settings**")
        g1, g2, g3 = st.columns(3)
        auto["xc"] = g1.text_input(
            "XC functional",
            value=str(auto.get("xc", inherited.get("xc", "PBE"))),
            key="oxidation_dft_auto_xc",
        )
        auto["ecut_eV"] = float(
            g2.number_input(
                "Plane-wave cutoff (eV)",
                min_value=50.0,
                value=float(auto.get("ecut_eV", inherited.get("ecut_eV", 500.0))),
                step=25.0,
                key="oxidation_dft_auto_ecut",
            )
        )
        auto["smearing_eV"] = float(
            g3.number_input(
                "Smearing (eV)",
                min_value=0.0,
                value=float(auto.get("smearing_eV", inherited.get("smearing_eV", 0.05))),
                step=0.01,
                key="oxidation_dft_auto_smearing",
            )
        )
        raw_kpts = auto.get("kpts", inherited.get("kpts", [1, 1, 1]))
        if not isinstance(raw_kpts, (list, tuple)) or len(raw_kpts) != 3:
            raw_kpts = [1, 1, 1]
        k1, k2, k3, kg = st.columns(4)
        auto["kpts"] = [
            int(k1.number_input("k₁", min_value=1, value=int(raw_kpts[0]), step=1, key="oxidation_dft_auto_k1")),
            int(k2.number_input("k₂", min_value=1, value=int(raw_kpts[1]), step=1, key="oxidation_dft_auto_k2")),
            int(k3.number_input("k₃", min_value=1, value=int(raw_kpts[2]), step=1, key="oxidation_dft_auto_k3")),
        ]
        auto["gamma"] = kg.checkbox(
            "Gamma-centered",
            value=bool(auto.get("gamma", inherited.get("gamma", True))),
            key="oxidation_dft_auto_gamma",
        )
        spin_options = ["auto", "true", "false"]
        current_spin = str(auto.get("spinpol", inherited.get("spinpol", "auto"))).lower()
        if current_spin not in spin_options:
            current_spin = "auto"
        auto["spinpol"] = st.selectbox(
            "Spin polarization",
            spin_options,
            index=spin_options.index(current_spin),
            key="oxidation_dft_auto_spin",
        )

        q1, q2, q3 = st.columns(3)
        auto["band_edge_analysis"] = q1.checkbox(
            "Analyze HOMO/LUMO localization",
            value=bool(auto.get("band_edge_analysis", True)),
            key="oxidation_dft_auto_band_edges",
        )
        wannier_modes = ["auto", "always", "never"]
        current_wannier = str(auto.get("wannier_mode", "auto")).lower()
        if current_wannier not in wannier_modes:
            current_wannier = "auto"
        auto["wannier_mode"] = q2.selectbox(
            "Wannier follow-up",
            wannier_modes,
            index=wannier_modes.index(current_wannier),
            key="oxidation_dft_auto_wannier",
            help="auto runs/reuses Wannier only when the preliminary integer assignment leaves electronic compensation.",
        )
        auto["save_wavefunctions"] = q3.checkbox(
            "Store wavefunctions",
            value=bool(auto.get("save_wavefunctions", True)),
            key="oxidation_dft_auto_wavefunctions",
            help="Needed for automatic band-edge and Wannier analysis.",
        )
        with st.expander("Automatic oxidation-state reference calibration", expanded=True):
            calibration_modes = ["auto", "off", "require"]
            calibration_mode = str(auto.get("reference_calibration", "auto")).lower()
            if calibration_mode not in calibration_modes:
                calibration_mode = "auto"
            r1, r2, r3 = st.columns(3)
            auto["reference_calibration"] = r1.selectbox(
                "Reference calibration",
                calibration_modes,
                index=calibration_modes.index(calibration_mode),
                key="oxidation_dft_auto_reference_calibration",
                help=(
                    "auto discovers/caches suitable binary oxide references; require stops if "
                    "two oxidation-state references are not available for every cation."
                ),
            )
            auto["run_missing_references"] = r2.checkbox(
                "Run missing reference DFT/Bader",
                value=bool(auto.get("run_missing_references", True)),
                key="oxidation_dft_auto_run_reference",
                disabled=auto["reference_calibration"] == "off",
                help=(
                    "Reference calculations are still protected by the main dft-auto execution "
                    "confirmation. Existing compatible reference fingerprints are reused."
                ),
            )
            kpoint_modes = ["match-density", "same-grid"]
            kpoint_mode = str(auto.get("reference_kpoint_mode", "match-density")).lower()
            if kpoint_mode not in kpoint_modes:
                kpoint_mode = "match-density"
            auto["reference_kpoint_mode"] = r3.selectbox(
                "Reference k-point mode",
                kpoint_modes,
                index=kpoint_modes.index(kpoint_mode),
                disabled=auto["reference_calibration"] == "off",
                help="match-density scales the reference mesh to approximately preserve reciprocal-space sampling density.",
            )

            default_roots = auto.get(
                "reference_roots",
                [
                    "reference_structures/relaxed/refs",
                    "reference_structures/oxidation_states",
                    "reference_structures/oxides",
                ],
            )
            if isinstance(default_roots, str):
                roots_text = default_roots
            else:
                roots_text = "\n".join(str(item) for item in default_roots)
            roots_text = st.text_area(
                "Reference structure roots (one per line)",
                value=roots_text,
                height=80,
                key="oxidation_dft_auto_reference_roots",
                disabled=auto["reference_calibration"] == "off",
            )
            auto["reference_roots"] = [
                line.strip() for line in roots_text.splitlines() if line.strip()
            ]
            ir1, ir2 = st.columns(2)
            auto["reference_include_project_relaxed_refs"] = ir1.checkbox(
                "Always include project's relaxed references",
                value=bool(auto.get("reference_include_project_relaxed_refs", True)),
                disabled=auto["reference_calibration"] == "off",
                help=(
                    "Automatically includes reference_structures/relaxed/refs even for "
                    "older saved configurations that do not list it explicitly."
                ),
            )
            auto["reference_include_correction_calibration"] = ir2.checkbox(
                "Supplement missing OS from correction references",
                value=bool(auto.get("reference_include_correction_calibration", True)),
                disabled=auto["reference_calibration"] == "off",
                help=(
                    "Uses one existing relaxed_calibration structure only when an "
                    "element/oxidation-state pair is missing from the primary reference roots."
                ),
            )
            rr1, rr2 = st.columns(2)
            auto["reference_manifest"] = rr1.text_input(
                "Optional reference manifest JSON",
                value=str(auto.get("reference_manifest", "")),
                key="oxidation_dft_auto_reference_manifest",
                disabled=auto["reference_calibration"] == "off",
                help=(
                    "Optional explicit references for difficult or non-binary cases. Relative paths "
                    "inside the manifest are resolved from the manifest directory."
                ),
            )
            auto["reference_cache_root"] = rr2.text_input(
                "Reference calibration cache",
                value=str(auto.get("reference_cache_root", "")),
                key="oxidation_dft_auto_reference_cache",
                disabled=auto["reference_calibration"] == "off",
                help="Empty uses <dft output root>/reference_calibration.",
            )
            rc1, rc2, rc3 = st.columns(3)
            auto["reference_min_states"] = int(
                rc1.number_input(
                    "Minimum OS references / element",
                    min_value=2,
                    value=max(2, int(auto.get("reference_min_states", 2))),
                    step=1,
                    disabled=auto["reference_calibration"] == "off",
                )
            )
            auto["reference_kpts_max"] = int(
                rc2.number_input(
                    "Maximum reference k-point mesh",
                    min_value=1,
                    value=max(1, int(auto.get("reference_kpts_max", 8))),
                    step=1,
                    disabled=auto["reference_calibration"] == "off",
                )
            )
            auto["reference_auto_magnetic_seed"] = rc3.checkbox(
                "Auto magnetic seed for transition-metal references",
                value=bool(auto.get("reference_auto_magnetic_seed", True)),
                disabled=auto["reference_calibration"] == "off",
                help="Used only as an SCF initialization; it is not treated as oxidation-state evidence.",
            )
            reference_spin_options = ["auto", "true", "false"]
            reference_spin = str(auto.get("reference_spinpol", "auto")).lower()
            if reference_spin not in reference_spin_options:
                reference_spin = "auto"
            auto["reference_spinpol"] = st.selectbox(
                "Reference spin polarization",
                reference_spin_options,
                index=reference_spin_options.index(reference_spin),
                disabled=auto["reference_calibration"] == "off",
                help=(
                    "Independent from the target spin setting. 'auto' lets nonzero reference "
                    "magnetic seeds activate spin when needed."
                ),
            )

            with st.expander("Calibration decision thresholds", expanded=False):
                ct1, ct2, ct3, ct4 = st.columns(4)
                auto["reference_max_z"] = float(
                    ct1.number_input(
                        "Max z-distance",
                        min_value=0.1,
                        value=float(auto.get("reference_max_z", 2.5)),
                        step=0.1,
                    )
                )
                auto["reference_min_z_gap"] = float(
                    ct2.number_input(
                        "Min z-gap",
                        min_value=0.0,
                        value=float(auto.get("reference_min_z_gap", 0.75)),
                        step=0.05,
                    )
                )
                auto["reference_min_probability"] = float(
                    ct3.number_input(
                        "Min relative likelihood",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(auto.get("reference_min_probability", 0.70)),
                        step=0.05,
                    )
                )
                auto["reference_min_probability_margin"] = float(
                    ct4.number_input(
                        "Min likelihood margin",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(auto.get("reference_min_probability_margin", 0.20)),
                        step=0.05,
                    )
                )

        auto["bader_reference_file"] = st.text_input(
            "Optional manual Bader fingerprint JSON override",
            value=str(auto.get("bader_reference_file", "")),
            key="oxidation_dft_auto_reference_file",
            help="Manual fingerprints override automatically generated fingerprints for matching element/oxidation-state entries.",
        )
        st.caption(
            "dft-auto runs/reuses GPAW, Bader, automatic same-method reference calibration, "
            "band-edge localization, and—only when useful—Wannier analysis. It does not force "
            "charge neutrality by inventing localized mixed valence."
        )
        oxidation["dft_auto"] = auto


if "dft-electronic" in methods:
    with st.expander("GPAW electronic descriptors", expanded=False):
        dft = _table("dft_electronic")
        dft["code"] = "gpaw"
        st.info(
            "GPAW is the supported DFT backend for oxidation analysis. It is open source and runs "
            "directly from the relaxed structure without INCAR/KPOINTS/POTCAR-style per-structure inputs."
        )
        dft["output_root"] = st.text_input(
            "Per-target GPAW output root",
            value=str(dft.get("output_root", "dft_oxidation")),
            key="oxidation_dft_electronic_root",
        )
        dft["execute"] = st.checkbox(
            "Run GPAW single-point calculation",
            value=bool(dft.get("execute", False)),
            key="oxidation_dft_electronic_execute",
            help="Off means post-process an existing oxidation.gpw file only.",
        )
        f1, f2 = st.columns(2)
        dft["gpw_file"] = f1.text_input(
            "GPAW restart file", value=str(dft.get("gpw_file", "oxidation.gpw"))
        )
        dft["txt_file"] = f2.text_input(
            "GPAW log file", value=str(dft.get("txt_file", "gpaw.txt"))
        )
        g1, g2, g3 = st.columns(3)
        dft["xc"] = g1.text_input("XC functional", value=str(dft.get("xc", "PBE")))
        dft["mode"] = "pw"
        dft["ecut_eV"] = float(
            g2.number_input(
                "Plane-wave cutoff (eV)",
                min_value=50.0,
                value=float(dft.get("ecut_eV", 500.0)),
                step=25.0,
            )
        )
        dft["smearing_eV"] = float(
            g3.number_input(
                "Fermi-Dirac smearing (eV)",
                min_value=0.0,
                value=float(dft.get("smearing_eV", 0.05)),
                step=0.01,
            )
        )
        raw_kpts = dft.get("kpts", [1, 1, 1])
        if not isinstance(raw_kpts, (list, tuple)) or len(raw_kpts) != 3:
            raw_kpts = [1, 1, 1]
        k1, k2, k3, kg = st.columns(4)
        kx = int(k1.number_input("k₁", min_value=1, value=int(raw_kpts[0]), step=1))
        ky = int(k2.number_input("k₂", min_value=1, value=int(raw_kpts[1]), step=1))
        kz = int(k3.number_input("k₃", min_value=1, value=int(raw_kpts[2]), step=1))
        dft["kpts"] = [kx, ky, kz]
        dft["gamma"] = kg.checkbox("Gamma-centered", value=bool(dft.get("gamma", True)))
        c1, c2, c3 = st.columns(3)
        dft["convergence_density"] = float(
            c1.number_input(
                "Density convergence",
                min_value=1.0e-10,
                value=float(dft.get("convergence_density", 1.0e-5)),
                format="%.1e",
            )
        )
        dft["maxiter"] = int(
            c2.number_input("SCF max iterations", min_value=1, value=int(dft.get("maxiter", 333)), step=10)
        )
        dft["charge"] = float(
            c3.number_input("Net cell charge (e)", value=float(dft.get("charge", 0.0)), step=1.0)
        )
        spin_options = ["auto", "true", "false"]
        spin_current = str(dft.get("spinpol", "auto")).lower()
        if spin_current not in spin_options:
            spin_current = "auto"
        dft["spinpol"] = st.selectbox(
            "Spin polarization",
            spin_options,
            index=spin_options.index(spin_current),
            help="auto lets GPAW follow supplied initial magnetic moments; set true explicitly for magnetic systems when needed.",
        )
        magmom_text = json.dumps(dft.get("initial_magmoms", {}), indent=2)
        magmom_text = st.text_area(
            "Optional initial magnetic moments by element (JSON)",
            value=magmom_text,
            height=100,
            help='Example: {"Mn": 4.0, "Fe": 4.0, "Ni": 2.0}. Unlisted elements start at 0 μB.',
        )
        dft["initial_magmoms"] = _parse_json_mapping(magmom_text, "Initial magnetic moments")
        d1, d2, d3, d4 = st.columns(4)
        dft["dos_emin_eV"] = float(d1.number_input("DOS Emin (E-EF, eV)", value=float(dft.get("dos_emin_eV", -10.0))))
        dft["dos_emax_eV"] = float(d2.number_input("DOS Emax (E-EF, eV)", value=float(dft.get("dos_emax_eV", 5.0))))
        dft["dos_npoints"] = int(d3.number_input("DOS points", min_value=51, value=int(dft.get("dos_npoints", 601)), step=50))
        dft["dos_width_eV"] = float(d4.number_input("DOS width (eV)", min_value=0.0, value=float(dft.get("dos_width_eV", 0.10)), step=0.05))
        dft["save_wavefunctions"] = st.checkbox(
            "Store wavefunctions in .gpw (larger file)",
            value=bool(dft.get("save_wavefunctions", False)),
        )
        st.caption(
            "The calculation is single-point only. Energy cutoff, k-points, spin treatment, smearing, and convergence must be converged for the target chemistry before production use."
        )
        oxidation["dft_electronic"] = dft

if "bader" in methods:
    with st.expander("GPAW + Bader charge analysis", expanded=False):
        bader = _table("bader")
        bader["output_root"] = st.text_input(
            "Per-target GPAW/Bader output root",
            value=str(bader.get("output_root", (_table("dft_electronic").get("output_root") or "dft_oxidation"))),
            key="oxidation_bader_root",
            help="Use the same root as dft-electronic so Bader can reuse oxidation.gpw.",
        )
        bader["execute"] = st.checkbox(
            "Generate GPAW all-electron density and run Bader",
            value=bool(bader.get("execute", False)),
            key="oxidation_bader_execute",
            help="Requires an existing GPAW .gpw file in the same per-target directory and the free Bader executable.",
        )
        b1, b2, b3 = st.columns(3)
        bader["gpw_file"] = b1.text_input(
            "GPAW restart file", value=str(bader.get("gpw_file", "oxidation.gpw"))
        )
        bader["density_file"] = b2.text_input(
            "All-electron density cube", value=str(bader.get("density_file", "density.cube"))
        )
        bader["acf_file"] = b3.text_input(
            "Bader ACF file", value=str(bader.get("acf_file", "ACF.dat"))
        )
        grid_options = [1, 2, 4]
        grid_current = int(bader.get("gridrefinement", 2))
        if grid_current not in grid_options:
            grid_current = 2
        bader["gridrefinement"] = st.selectbox(
            "GPAW all-electron density grid refinement",
            grid_options,
            index=grid_options.index(grid_current),
        )
        command_text = st.text_input(
            "Bader command",
            value=_command_text(bader.get("command", ["bader", "density.cube"])),
            disabled=not bader["execute"],
            help="Default uses the free 'bader' executable on density.cube; no shell is used.",
        )
        bader["command"] = _parse_command(command_text)
        st.caption(
            "dopingflow reconstructs GPAW's all-electron density and reports continuous Bader charges as Z − basin electrons. It never converts them automatically into formal integer oxidation states."
        )
        oxidation["bader"] = bader

if "wannier" in methods:
    with st.expander("GPAW + Wannier90 descriptors", expanded=False):
        wannier = _table("wannier")
        inherited_root = _table("dft_electronic").get("output_root") or "dft_oxidation"
        wannier["output_root"] = st.text_input(
            "Per-target GPAW/Wannier output root",
            value=str(wannier.get("output_root") or inherited_root),
            key="oxidation_wannier_root",
            help="Use the same root as dft-electronic so Wannier can reuse oxidation.gpw.",
        )
        wannier["execute"] = st.checkbox(
            "Generate Wannier centres from the GPAW restart",
            value=bool(wannier.get("execute", False)),
            key="oxidation_wannier_execute",
            help="Requires oxidation.gpw written with stored wavefunctions and the wannier90.x executable.",
        )
        mode_options = ["native-gpaw", "external-command"]
        current_mode = str(wannier.get("execution_mode", "native-gpaw")).lower()
        if current_mode not in mode_options:
            current_mode = "native-gpaw"
        wannier["execution_mode"] = st.selectbox(
            "Wannier execution mode",
            mode_options,
            index=mode_options.index(current_mode),
            help="native-gpaw currently handles isolated non-spin-polarized Gamma-only occupied manifolds without disentanglement.",
        )
        w1, w2, w3 = st.columns(3)
        wannier["gpw_file"] = w1.text_input(
            "GPAW restart file", value=str(wannier.get("gpw_file", "oxidation.gpw"))
        )
        wannier["seed"] = w2.text_input(
            "Wannier90 seed", value=str(wannier.get("seed", "wannier90"))
        )
        wannier["executable"] = w3.text_input(
            "Wannier90 executable", value=str(wannier.get("executable", "wannier90.x"))
        )
        w4, w5, w6 = st.columns(3)
        wannier["num_iter"] = int(
            w4.number_input(
                "Wannier localization iterations",
                min_value=1,
                value=int(wannier.get("num_iter", 1000)),
                step=100,
            )
        )
        wannier["occupation_tolerance"] = float(
            w5.number_input(
                "Occupation tolerance",
                min_value=1.0e-8,
                max_value=0.1,
                value=float(wannier.get("occupation_tolerance", 1.0e-4)),
                format="%.1e",
            )
        )
        wannier["min_gap_eV"] = float(
            w6.number_input(
                "Minimum insulating gap (eV)",
                min_value=0.0,
                value=float(wannier.get("min_gap_eV", 1.0e-3)),
                format="%.3e",
            )
        )
        wannier["less_memory"] = st.checkbox(
            "Low-memory GPAW overlap generation",
            value=bool(wannier.get("less_memory", False)),
            help=(
                "For the current single-Gamma route leave this off unless needed. "
                "The option is retained for future multi-k workflows."
            ),
        )
        st.markdown("**Wannier-center classification thresholds**")
        wc1, wc2, wc3, wc4 = st.columns(4)
        wannier["atom_center_cutoff_angstrom"] = float(
            wc1.number_input(
                "Atom-center cutoff (Å)",
                min_value=0.01,
                value=float(wannier.get("atom_center_cutoff_angstrom", 0.45)),
                step=0.05,
            )
        )
        wannier["bond_center_cutoff_angstrom"] = float(
            wc2.number_input(
                "Bond-center cutoff (Å)",
                min_value=0.05,
                value=float(wannier.get("bond_center_cutoff_angstrom", 1.35)),
                step=0.05,
            )
        )
        wannier["bond_distance_balance_angstrom"] = float(
            wc3.number_input(
                "Bond distance balance (Å)",
                min_value=0.0,
                value=float(wannier.get("bond_distance_balance_angstrom", 0.30)),
                step=0.05,
            )
        )
        wannier["delocalized_spread_threshold_ang2"] = float(
            wc4.number_input(
                "Delocalized spread threshold (Å²)",
                min_value=0.01,
                value=float(wannier.get("delocalized_spread_threshold_ang2", 3.0)),
                step=0.25,
            )
        )
        st.caption(
            "These cutoffs classify periodic center geometry and spread outliers only; they do not define formal oxidation states."
        )
        default_centres = f"{wannier['seed']}_centres.xyz"
        wannier["centres_file"] = st.text_input(
            "Wannier centres file",
            value=str(wannier.get("centres_file", default_centres)),
        )
        if wannier["execution_mode"] == "external-command":
            command_text = st.text_input(
                "External Wannier command",
                value=_command_text(wannier.get("command", [])),
                disabled=not wannier["execute"],
                help="Legacy/custom route. The native GPAW route does not need this field.",
            )
            wannier["command"] = _parse_command(command_text)
            if wannier["execute"] and not wannier["command"]:
                st.error(
                    "wannier: external-command mode requires a command when execution is enabled."
                )
        else:
            wannier.pop("command", None)
            st.info(
                "Native mode detects the fully occupied bands, verifies a finite gap, uses Bloch phases "
                "as the initial gauge (no arbitrary atomic projection choice), writes .eig/.mmn through "
                "GPAW, runs wannier90.x, and parses the resulting *_centres.xyz file."
            )
        st.caption(
            "Static Wannier centers remain descriptors only. Formal oxidation states require a validated EOS/charge-pumping procedure."
        )
        oxidation["wannier"] = wannier

if "eos" in methods:
    with st.expander("EOS / charge-pumping formal assignment", expanded=False):
        dft_method_panel(
            "eos",
            "eos",
            extra=(
                "Formal DFT oxidation labels are accepted only from an external result marked validated=true "
                "with a documented assignment_procedure."
            ),
        )
        eos = _table("eos")
        eos["results_file"] = st.text_input(
            "Validated EOS results JSON",
            value=str(eos.get("results_file", "eos_results.json")),
        )
        oxidation["eos"] = eos


st.divider()
st.subheader("Optional DFT follow-up for unresolved/conflicting cases")
followup = dict(oxidation.get("dft_followup", {}) or {})
f1, f2 = st.columns(2)
with f1:
    followup["enabled"] = st.checkbox(
        "Create DFT follow-up candidates",
        value=bool(followup.get("enabled", False)),
        help="When execute is off, this only writes the candidate list.",
    )
with f2:
    followup["candidate_limit"] = int(
        st.number_input(
            "Candidate limit",
            min_value=1,
            value=int(followup.get("candidate_limit", 10)),
            step=1,
        )
    )
followup_methods = followup.get("methods", ["dft-electronic", "bader", "eos"])
if isinstance(followup_methods, str):
    followup_methods = [m.strip() for m in followup_methods.split(",") if m.strip()]
followup["methods"] = st.multiselect(
    "Follow-up methods",
    DFT_METHODS,
    default=[m for m in followup_methods if m in DFT_METHODS],
    disabled=not followup["enabled"],
)
followup["execute"] = st.checkbox(
    "Execute configured DFT follow-up calculations",
    value=bool(followup.get("execute", False)),
    disabled=not followup["enabled"],
    help="This is a separate execution gate. Keep it off to generate candidates only.",
)
oxidation["dft_followup"] = followup


oxidation.update(
    {
        "enabled": bool(enabled),
        "strategy": strategy,
        "methods": methods,
        "include_vacancy_free": bool(include_vacancy_free),
        "include_oxygen_vacancies": bool(include_oxygen_vacancies),
        "target_include": target_include,
        "source_root": source_root,
        "output_dir": output_dir,
        "mapping_tolerance": float(mapping_tolerance),
        "fail_fast": bool(fail_fast),
    }
)

st.divider()
st.subheader("Save and run")
resolved_cfg = dict(cfg)
resolved_cfg["oxidation"] = oxidation

with st.expander("Preview [oxidation] TOML", expanded=False):
    st.code(toml.dumps({"oxidation": oxidation}), language="toml")

contains_dft_execution = any(
    bool((_table(name) if name in oxidation else {}).get("execute", False))
    for name in ("dft_auto", "dft_electronic", "bader", "wannier", "eos")
) or bool(followup.get("execute", False))

if contains_dft_execution and not target_include:
    st.warning(
        "DFT execution is enabled but no target selector is active. The calculation can run for every "
        "discovered structure. For an expensive smoke test, select one exact target first."
    )

confirm_dft = True
if contains_dft_execution:
    st.warning(
        "At least one explicit DFT execution gate is ON. Running this page may launch external calculations "
        "for selected targets. No DFT calculation is launched when all execute flags are off."
    )
    confirm_dft = st.checkbox(
        "I confirm that I want the configured external DFT commands to be allowed to run",
        value=False,
    )

save_col, run_col = st.columns(2)
with save_col:
    if st.button("Save oxidation settings", type="primary", use_container_width=True):
        config_path.write_text(toml.dumps(resolved_cfg), encoding="utf-8")
        st.success(f"Saved {config_path}")

command = ["dopingflow", "oxidation", "-c", str(config_path), "--strategy", strategy]
if methods:
    command.extend(["--methods", ",".join(methods)])

with run_col:
    run_disabled = (strategy == "combined" and not methods) or (contains_dft_execution and not confirm_dft)
    if st.button("Run oxidation analysis", use_container_width=True, disabled=run_disabled):
        config_path.write_text(toml.dumps(resolved_cfg), encoding="utf-8")
        with st.spinner("Running oxidation analysis..."):
            completed = subprocess.run(
                command,
                cwd=str(project_root),
                text=True,
                capture_output=True,
                check=False,
            )
        st.session_state["oxidation_last_stdout"] = completed.stdout
        st.session_state["oxidation_last_stderr"] = completed.stderr
        st.session_state["oxidation_last_returncode"] = completed.returncode
        if completed.returncode == 0:
            st.success("Oxidation analysis finished successfully.")
        else:
            st.error(f"Oxidation analysis exited with return code {completed.returncode}.")

st.code(" ".join(shlex.quote(token) for token in command), language="bash")
if "oxidation_last_returncode" in st.session_state:
    with st.expander("Last run output", expanded=True):
        if st.session_state.get("oxidation_last_stdout"):
            st.text(st.session_state["oxidation_last_stdout"])
        if st.session_state.get("oxidation_last_stderr"):
            st.text(st.session_state["oxidation_last_stderr"])


st.divider()
st.subheader("Results")
source_path = Path(source_root).expanduser()
if not source_path.is_absolute():
    source_path = (project_root / source_path).resolve()
results_root = Path(output_dir).expanduser()
if not results_root.is_absolute():
    results_root = (source_path / results_root).resolve()

sites_csv = results_root / "oxidation_sites.csv"
index_csv = results_root / "oxidation_structure_index.csv"
results_json = results_root / "oxidation_results.json"
comparison_json = results_root / "oxidation_comparison.json"
followup_json = results_root / "dft_followup_candidates.json"

st.caption(f"Resolved output: `{results_root}`")

if not index_csv.exists():
    st.info(
        "No oxidation_structure_index.csv found yet. Run the oxidation stage with the "
        "updated workflow to create per-structure result folders and the structure browser."
    )
else:
    try:
        structure_index = pd.read_csv(index_csv)
    except Exception as exc:
        st.warning(f"Could not read {index_csv.name}: {exc}")
    else:
        st.markdown("#### Analysed structures")
        overview_columns = [
            column
            for column in (
                "target_id",
                "structure_kind",
                "n_oxygen_vacancies",
                "n_atoms",
                "overall_status",
                "n_successful_methods",
                "n_unresolved_methods",
            )
            if column in structure_index.columns
        ]
        st.dataframe(
            structure_index[overview_columns],
            use_container_width=True,
            hide_index=True,
        )

        target_ids = structure_index["target_id"].astype(str).tolist()
        selected_target = st.selectbox(
            "Choose a structure",
            target_ids,
            key="oxidation_result_target",
        )
        selected_meta = structure_index[
            structure_index["target_id"].astype(str) == selected_target
        ].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Kind", str(selected_meta.get("structure_kind", "")))
        m2.metric("O vacancies", int(selected_meta.get("n_oxygen_vacancies", 0)))
        n_atoms_value = selected_meta.get("n_atoms")
        m3.metric("Atoms", "-" if pd.isna(n_atoms_value) else int(n_atoms_value))
        m4.metric("Status", str(selected_meta.get("overall_status", "unknown")))

        st.caption(f"Structure file: `{selected_meta.get('structure_path', '')}`")
        st.caption(
            f"Per-structure results: `{selected_meta.get('output_directory', '')}`"
        )

        target_result_dir = Path(str(selected_meta.get("output_directory", "")))
        target_results_file = target_result_dir / "oxidation_results.json"
        target_sites_file = target_result_dir / "oxidation_sites.csv"
        target_summary_file = target_result_dir / "summary.json"

        target_results = []
        if target_results_file.exists():
            try:
                target_results = json.loads(
                    target_results_file.read_text(encoding="utf-8")
                )
            except Exception as exc:
                st.warning(f"Could not read {target_results_file}: {exc}")

        if target_results:
            method_names = [str(item.get("method")) for item in target_results]
            selected_method = st.selectbox(
                "Oxidation method",
                method_names,
                key="oxidation_result_method",
            )
            method_result = next(
                item
                for item in target_results
                if str(item.get("method")) == selected_method
            )
            status = str(method_result.get("assignment_status", "unknown"))
            if status in {"assigned", "descriptors-only"}:
                st.success(f"{selected_method}: {status}")
            else:
                detail = method_result.get("error") or "; ".join(
                    method_result.get("limitations", [])
                )
                st.warning(
                    f"{selected_method}: {status}. "
                    + (
                        str(detail)
                        if detail
                        else "No assignment was produced for this structure."
                    )
                )

            if selected_method == "dft-auto":
                summary = method_result.get("assignment_summary", {}) or {}
                compensation = summary.get("electronic_compensation", {}) or {}
                st.markdown("#### Automated DFT oxidation-state suggestion")
                am1, am2, am3, am4 = st.columns(4)
                am1.metric(
                    "Atomic formal-charge sum",
                    f"{float(summary.get('formal_charge_sum_e', 0.0)):+.2f} e",
                )
                am2.metric(
                    "Electronic compensation",
                    f"{float(compensation.get('charge_e', 0.0)):+.2f} e",
                )
                am3.metric(
                    "Compensation type",
                    str(compensation.get("type", "none")),
                )
                am4.metric(
                    "Localization",
                    str(compensation.get("localization", "not-evaluated")),
                )
                calibration = method_result.get("reference_calibration", {}) or {}
                calibration_summary = summary.get("reference_calibration_summary", {}) or {}
                if calibration:
                    st.markdown("##### Automatic reference calibration")
                    cr1, cr2, cr3, cr4 = st.columns(4)
                    states_available = calibration.get("states_available", {}) or {}
                    cr1.metric("Calibration status", str(calibration.get("status", "unknown")))
                    cr2.metric(
                        "Calibrated sites",
                        int(calibration_summary.get("n_calibrated_sites", 0)),
                    )
                    cr3.metric(
                        "Ambiguous sites",
                        int(calibration_summary.get("n_ambiguous_sites", 0)),
                    )
                    cr4.metric(
                        "Reference elements",
                        len(states_available),
                    )
                    missing_refs = calibration.get("elements_missing_minimum_states", []) or []
                    if missing_refs:
                        st.warning(
                            "Insufficient oxidation-state reference coverage for: "
                            + ", ".join(str(item) for item in missing_refs)
                        )
                    if states_available:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "element": element,
                                        "reference_oxidation_states": states,
                                    }
                                    for element, states in states_available.items()
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                component_status = method_result.get("component_status", {}) or {}
                if component_status:
                    st.markdown("##### Automatic diagnostic stages")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"stage": name, **(value if isinstance(value, dict) else {"status": value})}
                                for name, value in component_status.items()
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.caption(
                    "These are formal oxidation-state suggestions supported by DFT descriptors. "
                    "Residual electrons/holes are shown explicitly rather than being assigned arbitrarily to atoms."
                )

            if selected_method == "wannier":
                analysis = method_result.get("wannier_analysis", {}) or {}
                if analysis:
                    st.markdown("#### Wannier-center analysis")
                    wm1, wm2, wm3, wm4 = st.columns(4)
                    wm1.metric("Wannier centers", int(analysis.get("n_wannier_centres", 0)))
                    represented = analysis.get("represented_electrons")
                    wm2.metric(
                        "Electron equivalent",
                        "-" if represented is None else f"{float(represented):.0f}",
                    )
                    wm3.metric(
                        "Spread outliers",
                        int(analysis.get("n_delocalized_spread_outliers", 0)),
                    )
                    max_spread = analysis.get("max_spread_ang2")
                    wm4.metric(
                        "Max spread (Å²)",
                        "-" if max_spread is None else f"{float(max_spread):.3f}",
                    )
                    counts = analysis.get("classification_counts", {}) or {}
                    if counts:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {"classification": name, "count": count}
                                    for name, count in counts.items()
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    if analysis.get("center_count_consistent_with_occupied_bands") is False:
                        st.warning("Wannier-center count does not match the occupied-band count recorded for this run.")

                site_summary = method_result.get("wannier_site_summary", []) or []
                if site_summary:
                    st.markdown("##### Per-site geometric Wannier descriptors")
                    st.dataframe(
                        pd.DataFrame(site_summary),
                        use_container_width=True,
                        hide_index=True,
                    )

                centre_rows = method_result.get("wannier_descriptors", []) or []
                if centre_rows:
                    with st.expander("Wannier centers and spreads", expanded=False):
                        center_df = pd.DataFrame(centre_rows)
                        if "spread_ang2" in center_df.columns:
                            center_df = center_df.sort_values(
                                "spread_ang2", ascending=False, na_position="last"
                            )
                        st.dataframe(center_df, use_container_width=True, hide_index=True)

                parent_changes = method_result.get("parent_relative_changes", []) or []
                if parent_changes:
                    st.markdown("##### Parent-relative Wannier descriptor changes")
                    st.dataframe(
                        pd.DataFrame(parent_changes),
                        use_container_width=True,
                        hide_index=True,
                    )
                elif str(selected_meta.get("structure_kind", "")) == "vacancy-free":
                    st.info(
                        "This is a standalone vacancy-free Wannier analysis. No oxygen-vacancy result is required. "
                        "Parent-relative deltas will appear only after a matched vacancy structure is also analyzed."
                    )

            if target_sites_file.exists():
                try:
                    target_sites = pd.read_csv(target_sites_file)
                except Exception as exc:
                    st.warning(f"Could not read {target_sites_file}: {exc}")
                else:
                    method_sites = target_sites[
                        target_sites["method"].astype(str) == selected_method
                    ].copy()
                    if "site_index" in method_sites.columns:
                        method_sites = method_sites.sort_values(
                            "site_index", na_position="last"
                        )
                    preferred = [
                        "site_index",
                        "element",
                        "formal_oxidation_state",
                        "confidence",
                        "confidence_label",
                        "oxidation_state_status",
                        "reference_calibration_status",
                        "calibrated_oxidation_state",
                        "calibration_confidence",
                        "structural_prior_oxidation_state",
                        "coordination_number",
                        "bader_partial_charge",
                        "magnetic_moment",
                        "parent_site_index",
                        "parent_formal_oxidation_state",
                        "delta_formal_oxidation_state",
                        "assignment_status",
                    ]
                    display_columns = [
                        col for col in preferred if col in method_sites.columns
                    ]
                    extra_columns = [
                        col
                        for col in method_sites.columns
                        if col not in display_columns
                        and col
                        not in {
                            "target_id",
                            "parent_id",
                            "structure_kind",
                            "method",
                            "strategy",
                            "prediction_scope",
                        }
                    ]
                    display_columns.extend(extra_columns)
                    st.markdown("#### Atom-by-atom results")
                    if method_sites.empty:
                        st.info(
                            "This method produced no site/composition records for the "
                            "selected structure."
                        )
                    else:
                        st.dataframe(
                            method_sites[display_columns],
                            use_container_width=True,
                            hide_index=True,
                        )

            with st.expander("Selected method details", expanded=False):
                st.json(method_result)

        if target_summary_file.exists():
            with st.expander("Structure summary", expanded=False):
                try:
                    st.json(
                        json.loads(target_summary_file.read_text(encoding="utf-8"))
                    )
                except Exception as exc:
                    st.warning(f"Could not read {target_summary_file}: {exc}")

with st.expander("Aggregate result files", expanded=False):
    if sites_csv.exists():
        try:
            all_sites = pd.read_csv(sites_csv)
            st.caption(f"Global atom/composition table: {len(all_sites):,} rows")
        except Exception as exc:
            st.warning(f"Could not read {sites_csv}: {exc}")
    for path in (results_json, comparison_json, followup_json):
        if not path.exists():
            continue
        st.markdown(f"**{path.name}**")
        try:
            st.json(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            st.warning(f"Could not read {path}: {exc}")


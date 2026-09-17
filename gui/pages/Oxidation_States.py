from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import toml


STRATEGY_METHODS = {
    "structural": ["bond-valence", "toss-bayesian"],
    "ml": ["toss-gnn", "chgnet", "bertos"],
    "dft": ["dft-electronic", "bader", "wannier", "eos"],
}
ALL_METHODS = [
    "bond-valence",
    "toss-bayesian",
    "toss-gnn",
    "chgnet",
    "bertos",
    "dft-electronic",
    "bader",
    "wannier",
    "eos",
]
DFT_METHODS = ["dft-electronic", "bader", "wannier", "eos"]


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
    existing_methods = list(allowed_methods)

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

st.info(
    "Bader charge, DOS/orbital populations, magnetic moments, and static Wannier centers are "
    "supporting descriptors. They are not automatically converted into formal integer oxidation states."
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

if "toss-bayesian" in methods:
    with st.expander("Conventional TOSS Bayesian/MAP", expanded=False):
        toss = _table("toss_bayesian")
        toss["repo_path"] = st.text_input(
            "Local TOSS repository",
            value=str(toss.get("repo_path", "")),
            key="oxidation_toss_bayesian_repo",
        )
        st.caption("This is the conventional TOSS assignment, separate from the pretrained TOSS-GNN model.")
        oxidation["toss_bayesian"] = toss

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


if "dft-electronic" in methods:
    with st.expander("DFT electronic descriptors", expanded=False):
        dft = _table("dft_electronic")
        dft["code"] = st.selectbox(
            "Electronic-structure code",
            ["vasp"],
            index=0,
            help="The current parser supports VASP vasprun.xml/OUTCAR outputs.",
        )
        oxidation["dft_electronic"] = dft
        dft_method_panel(
            "dft-electronic",
            "dft_electronic",
            extra="Reports electronic descriptors only; it does not create a formal integer oxidation assignment.",
        )

if "bader" in methods:
    with st.expander("Bader charge analysis", expanded=False):
        dft_method_panel(
            "bader",
            "bader",
            extra="Bader produces continuous charge descriptors only and never creates formal oxidation labels by itself.",
        )
        bader = _table("bader")
        valence_text = json.dumps(bader.get("valence_electrons", {}), indent=2)
        valence_text = st.text_area(
            "Optional valence-electron references (JSON)",
            value=valence_text,
            height=100,
            help='Example: {"Sn": 4, "Sb": 5, "O": 6}. Leave empty when POTCAR supplies the values.',
        )
        bader["valence_electrons"] = _parse_json_mapping(valence_text, "Valence-electron references")
        oxidation["bader"] = bader

if "wannier" in methods:
    with st.expander("Wannier descriptors", expanded=False):
        dft_method_panel(
            "wannier",
            "wannier",
            extra="Static Wannier centers are retained as descriptors and are not automatically converted to formal oxidation states.",
        )
        wannier = _table("wannier")
        wannier["centres_file"] = st.text_input(
            "Wannier centres file",
            value=str(wannier.get("centres_file", "wannier90_centres.xyz")),
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
    for name in ("dft_electronic", "bader", "wannier", "eos")
) or bool(followup.get("execute", False))

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


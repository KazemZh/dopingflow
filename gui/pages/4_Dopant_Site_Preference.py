"""Configure, run, and inspect dopant site-preference / ordering analysis."""

from __future__ import annotations

import html
import json
import shlex
import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import toml

from dopingflow.site_preference import (
    discover_site_preference_targets,
    parse_site_preference_config,
)


st.set_page_config(page_title="Dopant site preference", layout="wide")
st.title("Dopant site preference")
st.caption(
    "Determine where dopants and co-dopants prefer to sit relative to one another and to "
    "oxygen vacancies. Existing relaxed structures are analysed first; controlled pair-shell "
    "scans and finite-temperature cation-ordering Monte Carlo are optional."
)

project_root = Path(
    st.sidebar.text_input(
        "Project root",
        value=str(Path.cwd()),
        key="site_preference_project_root",
    )
).expanduser().resolve()
config_path = project_root / "input.toml"

if not config_path.exists():
    st.error(f"No input.toml found at {config_path}")
    st.stop()

cfg = toml.load(str(config_path))
section = dict(cfg.get("site_preference", {}) or {})
doping = dict(cfg.get("doping", {}) or {})
scan = dict(cfg.get("scan", {}) or {})
structure = dict(cfg.get("structure", {}) or {})


def _csv_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return ""


def _parse_csv(text: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in text.split(",") if item.strip()))


def _pairs_text(value) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, (list, tuple)):
        return ""
    output = []
    for item in value:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            output.append(f"{item[0]}-{item[1]}")
    return ", ".join(output)


with st.expander("Configuration & run controls", expanded=False):
    st.caption(
        "Open this only when you want to change settings or launch a calculation. "
        "The scientific results explorer stays uncluttered below."
    )
    st.subheader("Analysis")
    a1, a2, a3 = st.columns(3)
    with a1:
        enabled = st.checkbox(
            "Enable site-preference stage",
            value=bool(section.get("enabled", False)),
        )
    with a2:
        include_vacancy_free = st.checkbox(
            "Vacancy-free structures",
            value=bool(section.get("include_vacancy_free", True)),
        )
    with a3:
        include_oxygen_vacancies = st.checkbox(
            "O-vacancy structures",
            value=bool(section.get("include_oxygen_vacancies", True)),
        )

    b1, b2, b3 = st.columns(3)
    host_species = b1.text_input(
        "Host cation",
        value=str(section.get("host_species", doping.get("host_species", "Sn"))),
    )
    anion_species_text = b2.text_input(
        "Anion species",
        value=_csv_text(section.get("anion_species", scan.get("anion_species", ["O"]))),
    )
    output_dir = b3.text_input(
        "Output directory",
        value=str(section.get("output_dir", "06_site_preference")),
    )

    source_root = st.text_input(
        "Source root",
        value=str(section.get("source_root", structure.get("outdir", "random_structures"))),
        help="Contains selected_candidates.txt files and, when applicable, vacancies_database.json.",
    )

    c1, c2 = st.columns(2)
    max_shells = int(
        c1.number_input(
            "Maximum coordination shells",
            min_value=1,
            value=int(section.get("max_shells", 6)),
            step=1,
        )
    )
    shell_tolerance = float(
        c2.number_input(
            "Shell clustering tolerance (Å)",
            min_value=0.001,
            value=float(section.get("shell_tolerance_angstrom", 0.12)),
            step=0.01,
            format="%.3f",
        )
    )

    d1, d2, d3 = st.columns(3)
    mapping_tolerance = float(
        d1.number_input(
            "Parent ↔ vacancy mapping tolerance (Å)",
            min_value=0.05,
            value=float(section.get("mapping_tolerance_angstrom", 1.5)),
            step=0.05,
        )
    )
    motif_neighbor_shell_max = int(
        d2.number_input(
            "Triplet motif neighbor shell",
            min_value=1,
            max_value=max_shells,
            value=min(
                int(section.get("motif_neighbor_shell_max", 1)),
                max_shells,
            ),
            step=1,
            help=(
                "A pair counts as connected in Sb–X–Y motif classification when its "
                "separation is within this cation coordination shell."
            ),
        )
    )
    target_include_text = d3.text_input(
        "Target selector(s), optional",
        value=_csv_text(section.get("target_include", [])),
        help="Exact IDs or wildcards, e.g. Sb5_Ti2p5/*.",
    )

    pair_saved = dict(section.get("pair_scan", {}) or {})
    with st.expander("Controlled dopant-pair shell scan", expanded=False):
        st.caption(
            "Build a host-only cation sublattice from a selected parent, place one dopant pair "
            "at representative coordination shells, then optionally evaluate/relax each shell "
            "with the selected MLFF. ΔE is reported relative to the farthest evaluated shell."
        )
        p1, p2 = st.columns(2)
        pair_enabled = p1.checkbox(
            "Enable pair scan",
            value=bool(pair_saved.get("enabled", False)),
            key="site_pair_enabled",
        )
        pair_execute = p2.checkbox(
            "Execute MLFF energies/relaxations",
            value=bool(pair_saved.get("execute", False)),
            key="site_pair_execute",
        )
        pair_source = st.text_input(
            "Source parent target (optional)",
            value=str(pair_saved.get("source_target", "")),
            help="Leave empty to use the lowest-energy selected vacancy-free parent.",
        )
        pair_text = st.text_input(
            "Pairs",
            value=_pairs_text(pair_saved.get("pairs", [])),
            placeholder="Sb-Ti, Sb-Nb, Ti-Nb",
            help="Leave empty to infer all dopant pair types present in the selected structures.",
        )
        p3, p4, p5 = st.columns(3)
        pair_max_shells = int(
            p3.number_input(
                "Pair-scan shells",
                min_value=1,
                value=int(pair_saved.get("max_shells", max_shells)),
                step=1,
            )
        )
        pair_relax = p4.checkbox(
            "Relax each shell",
            value=bool(pair_saved.get("relax", True)),
        )
        pair_backend = p5.selectbox(
            "MLFF backend",
            ["mace", "grace", "m3gnet", "uma"],
            index=["mace", "grace", "m3gnet", "uma"].index(
                str(pair_saved.get("backend", "mace")).lower()
                if str(pair_saved.get("backend", "mace")).lower()
                in ["mace", "grace", "m3gnet", "uma"]
                else "mace"
            ),
        )
        p6, p7, p8 = st.columns(3)
        pair_model = p6.text_input("Model", value=str(pair_saved.get("model", "small")))
        pair_device = p7.selectbox(
            "Device",
            ["cpu", "cuda"],
            index=1 if str(pair_saved.get("device", "cpu")).lower() == "cuda" else 0,
        )
        pair_fmax = float(
            p8.number_input(
                "Relax fmax (eV/Å)",
                min_value=0.001,
                value=float(pair_saved.get("fmax", 0.05)),
                step=0.01,
            )
        )
        p9, p10 = st.columns(2)
        pair_max_steps = int(
            p9.number_input(
                "Relax max steps",
                min_value=1,
                value=int(pair_saved.get("max_steps", 300)),
                step=10,
            )
        )
        pair_task = p10.text_input("Backend task (optional)", value=str(pair_saved.get("task", "")))

    mc_saved = dict(section.get("ordering_mc", {}) or {})
    with st.expander("Finite-temperature cation-ordering Monte Carlo", expanded=False):
        st.caption(
            "Composition is fixed. MC swaps cation identities without force calculations; "
            "single-point MLFF energies drive Metropolis acceptance. The best occupation can "
            "optionally be relaxed at the end."
        )
        m1, m2 = st.columns(2)
        mc_enabled = m1.checkbox(
            "Enable ordering MC",
            value=bool(mc_saved.get("enabled", False)),
            key="site_mc_enabled",
        )
        mc_execute = m2.checkbox(
            "Execute MC",
            value=bool(mc_saved.get("execute", False)),
            key="site_mc_execute",
        )
        mc_targets = st.text_input(
            "MC target selector(s), optional",
            value=_csv_text(mc_saved.get("target_include", [])),
            help="Leave empty to use the lowest-energy vacancy-free parent from each composition.",
        )
        m3, m4, m5, m6 = st.columns(4)
        mc_temperature = float(
            m3.number_input(
                "Temperature (K)",
                min_value=1.0,
                value=float(mc_saved.get("temperature_K", 800.0)),
                step=50.0,
            )
        )
        mc_steps = int(
            m4.number_input(
                "MC steps",
                min_value=2,
                value=int(mc_saved.get("steps", 10000)),
                step=1000,
            )
        )
        mc_burn = int(
            m5.number_input(
                "Burn-in",
                min_value=0,
                value=int(mc_saved.get("burn_in", 2000)),
                step=500,
            )
        )
        mc_interval = int(
            m6.number_input(
                "Sample interval",
                min_value=1,
                value=int(mc_saved.get("sample_interval", 20)),
                step=5,
            )
        )
        m7, m8, m9 = st.columns(3)
        mc_max_targets = int(
            m7.number_input(
                "Maximum MC targets",
                min_value=1,
                value=int(mc_saved.get("max_targets", 5)),
                step=1,
                help="Maximum number of independent compositions/targets included in the MC run.",
            )
        )
        mc_parallel_targets = int(
            m8.number_input(
                "Parallel MC targets",
                min_value=1,
                value=int(mc_saved.get("parallel_targets", 1)),
                step=1,
                help=(
                    "Number of independent compositions run simultaneously. "
                    "Each individual MC chain remains sequential."
                ),
            )
        )
        mc_omp_threads = int(
            m9.number_input(
                "CPU threads per target",
                min_value=1,
                value=int(mc_saved.get("omp_threads", 1)),
                step=1,
                help="OpenMP CPU threads available to each MLFF worker.",
            )
        )

        m10, m11, m12 = st.columns(3)
        mc_backend = m10.selectbox(
            "MC MLFF backend",
            ["mace", "grace", "m3gnet", "uma"],
            index=["mace", "grace", "m3gnet", "uma"].index(
                str(mc_saved.get("backend", "mace")).lower()
                if str(mc_saved.get("backend", "mace")).lower()
                in ["mace", "grace", "m3gnet", "uma"]
                else "mace"
            ),
            key="site_mc_backend",
        )
        mc_model = m11.text_input(
            "MC model",
            value=str(mc_saved.get("model", "small")),
        )
        mc_device = m12.selectbox(
            "MC device",
            ["cpu", "cuda"],
            index=1 if str(mc_saved.get("device", "cpu")).lower() == "cuda" else 0,
            key="site_mc_device",
        )

        m13, m14 = st.columns(2)
        mc_relax_best = m13.checkbox(
            "Relax best MC occupation",
            value=bool(mc_saved.get("relax_best", True)),
        )
        mc_seed = int(
            m14.number_input(
                "MC random seed",
                value=int(mc_saved.get("seed", 42)),
                step=1,
            )
        )

        effective_workers_preview = min(mc_parallel_targets, mc_max_targets)
        if mc_device == "cpu":
            st.info(
                f"CPU plan: up to {effective_workers_preview} MC target(s) in parallel "
                f"× {mc_omp_threads} thread(s) per target = "
                f"up to {effective_workers_preview * mc_omp_threads} requested CPU threads."
            )
        elif mc_parallel_targets > 1:
            st.warning(
                "Parallel MC targets > 1 is currently supported only for device='cpu'. "
                "Set Parallel MC targets to 1 for CUDA."
            )

    target_include = _parse_csv(target_include_text)
    anion_species = _parse_csv(anion_species_text)
    pairs = _parse_csv(pair_text)
    mc_target_include = _parse_csv(mc_targets)

    resolved_section = dict(section)
    resolved_section.update(
        {
            "enabled": enabled,
            "source_root": source_root,
            "output_dir": output_dir,
            "host_species": host_species.strip(),
            "anion_species": anion_species,
            "include_vacancy_free": include_vacancy_free,
            "include_oxygen_vacancies": include_oxygen_vacancies,
            "target_include": target_include,
            "max_shells": max_shells,
            "shell_tolerance_angstrom": shell_tolerance,
            "mapping_tolerance_angstrom": mapping_tolerance,
            "motif_neighbor_shell_max": motif_neighbor_shell_max,
            "pair_scan": {
                **pair_saved,
                "enabled": pair_enabled,
                "execute": pair_execute,
                "source_target": pair_source.strip(),
                "pairs": pairs,
                "max_shells": pair_max_shells,
                "relax": pair_relax,
                "backend": pair_backend,
                "model": pair_model.strip(),
                "task": pair_task.strip(),
                "device": pair_device,
                "fmax": pair_fmax,
                "max_steps": pair_max_steps,
            },
            "ordering_mc": {
                **mc_saved,
                "enabled": mc_enabled,
                "execute": mc_execute,
                "target_include": mc_target_include,
                "max_targets": mc_max_targets,
                "parallel_targets": mc_parallel_targets,
                "temperature_K": mc_temperature,
                "steps": mc_steps,
                "burn_in": mc_burn,
                "sample_interval": mc_interval,
                "seed": mc_seed,
                "backend": mc_backend,
                "model": mc_model.strip(),
                "device": mc_device,
                "omp_threads": mc_omp_threads,
                "relax_best": mc_relax_best,
            },
        }
    )
    resolved_cfg = dict(cfg)
    resolved_cfg["site_preference"] = resolved_section

    validation_error = None
    parsed_cfg = None
    try:
        parsed_cfg = parse_site_preference_config(resolved_cfg, project_root)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        validation_error = str(exc)

    st.divider()
    st.subheader("Save and run")

    with st.expander("Preview site_preference TOML", expanded=False):
        st.code(toml.dumps({"site_preference": resolved_section}), language="toml")

    if validation_error:
        st.error(validation_error)

    if parsed_cfg is not None:
        with st.expander("Preview selected structures", expanded=False):
            try:
                preview_targets, _, preview_warnings = discover_site_preference_targets(parsed_cfg)
            except Exception as exc:
                st.warning(str(exc))
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "target_id": target.target_id,
                                "kind": target.kind,
                                "O vacancies": target.n_vacancies,
                                "energy (eV)": target.energy_eV,
                                "structure": str(target.structure_path),
                            }
                            for target in preview_targets
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                for warning in preview_warnings:
                    st.warning(warning)

    expensive = (pair_enabled and pair_execute) or (mc_enabled and mc_execute)
    confirm = True
    if expensive:
        st.warning(
            "MLFF execution is enabled. Pair scans may relax several coordination shells, and "
            "ordering MC may require many single-point energy evaluations."
        )
        confirm = st.checkbox(
            "I confirm that the configured MLFF calculations may run",
            value=False,
        )

    save_col, run_col = st.columns(2)
    with save_col:
        if st.button(
            "Save site-preference settings",
            type="primary",
            use_container_width=True,
            disabled=validation_error is not None,
        ):
            config_path.write_text(toml.dumps(resolved_cfg), encoding="utf-8")
            st.success(f"Saved {config_path}")

    command = ["dopingflow", "site-preference", "-c", str(config_path)]
    with run_col:
        if st.button(
            "Run site-preference analysis",
            use_container_width=True,
            disabled=(not enabled) or validation_error is not None or not confirm,
        ):
            config_path.write_text(toml.dumps(resolved_cfg), encoding="utf-8")
            with st.spinner("Running dopant site-preference analysis..."):
                completed = subprocess.run(
                    command,
                    cwd=str(project_root),
                    text=True,
                    capture_output=True,
                    check=False,
                )
            st.session_state["site_pref_stdout"] = completed.stdout
            st.session_state["site_pref_stderr"] = completed.stderr
            st.session_state["site_pref_returncode"] = completed.returncode
            if completed.returncode == 0:
                st.success("Site-preference analysis finished successfully.")
            else:
                st.error(f"Analysis exited with return code {completed.returncode}.")

    st.code(" ".join(shlex.quote(token) for token in command), language="bash")
    if "site_pref_returncode" in st.session_state:
        with st.expander("Last run output", expanded=True):
            if st.session_state.get("site_pref_stdout"):
                st.text(st.session_state["site_pref_stdout"])
            if st.session_state.get("site_pref_stderr"):
                st.text(st.session_state["site_pref_stderr"])


st.divider()
st.subheader("Results explorer")
st.caption(
    "The primary view is intentionally interpretation-first: one structure, one scientific "
    "question at a time. Detailed CSV tables are kept in the Raw data tab."
)

st.markdown(
    """
    <style>
    .df-hero {
        border: 1px solid rgba(128,128,128,0.28);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin: 0.35rem 0 0.8rem 0;
        background: rgba(128,128,128,0.055);
    }
    .df-hero-title {
        font-size: 1.12rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .df-hero-sub {
        opacity: 0.78;
        font-size: 0.92rem;
        line-height: 1.35;
    }
    .df-takeaway {
        border: 1px solid rgba(128,128,128,0.25);
        border-left: 5px solid rgba(55,110,180,0.85);
        border-radius: 10px;
        padding: 0.78rem 0.9rem;
        margin: 0.45rem 0;
        background: rgba(128,128,128,0.035);
        min-height: 5.2rem;
    }
    .df-takeaway-title {
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.70;
        margin-bottom: 0.22rem;
        font-weight: 650;
    }
    .df-takeaway-main {
        font-size: 1.02rem;
        font-weight: 650;
        line-height: 1.33;
    }
    .df-takeaway-note {
        margin-top: 0.22rem;
        font-size: 0.82rem;
        opacity: 0.72;
        line-height: 1.30;
    }
    .df-section-note {
        border-radius: 9px;
        padding: 0.72rem 0.85rem;
        margin: 0.35rem 0 0.7rem 0;
        background: rgba(128,128,128,0.07);
        font-size: 0.92rem;
        line-height: 1.38;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

source_path = Path(source_root).expanduser()
if not source_path.is_absolute():
    source_path = (project_root / source_path).resolve()
results_root = Path(output_dir).expanduser()
if not results_root.is_absolute():
    results_root = (source_path / results_root).resolve()

summary_path = results_root / "site_preference_summary.json"
targets_csv = results_root / "site_preference_targets.csv"
pairs_csv = results_root / "dopant_pairs.csv"
nearest_csv = results_root / "nearest_pair_by_target.csv"
preference_csv = results_root / "pair_preference_summary.csv"
sro_csv = results_root / "warren_cowley_sro.csv"
vacancy_csv = results_root / "dopant_vacancy_pairs.csv"
nearest_vacancy_csv = results_root / "nearest_dopant_vacancy_by_target.csv"
vacancy_preference_csv = results_root / "dopant_vacancy_preference_summary.csv"
triplet_target_csv = results_root / "triplet_target_motifs.csv"
triplet_csv = results_root / "triplet_motif_summary.csv"
pair_scan_csv = results_root / "pair_scan" / "pair_scan.csv"
mc_summary_path = results_root / "ordering_mc" / "ordering_mc_summary.json"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()


def _read_json(path: Path, default):
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _one_target(df: pd.DataFrame, target_id: str) -> pd.DataFrame:
    if df.empty or "target_id" not in df.columns:
        return pd.DataFrame()
    return df[df["target_id"].astype(str) == str(target_id)].copy()


def _fmt(value, digits: int = 3, suffix: str = "") -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _shell_text(value) -> str:
    try:
        if pd.isna(value):
            return "unassigned"
        return f"{int(float(value))}NN"
    except (TypeError, ValueError):
        return str(value)


def _plain_motif(value: str) -> str:
    return {
        "compact_triangle": "compact triangle",
        "connected_chain": "connected chain",
        "isolated_pair_plus_third": "isolated pair + third",
        "dispersed": "dispersed",
    }.get(str(value), str(value).replace("_", " "))


def _safe(value) -> str:
    return html.escape(str(value), quote=True)


def _takeaway(title: str, main: str, note: str = "") -> str:
    note_html = (
        f'<div class="df-takeaway-note">{_safe(note)}</div>' if note else ""
    )
    return (
        '<div class="df-takeaway">'
        f'<div class="df-takeaway-title">{_safe(title)}</div>'
        f'<div class="df-takeaway-main">{_safe(main)}</div>'
        f"{note_html}</div>"
    )


def _pair_priority(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "pair" not in frame.columns:
        return frame
    out = frame.copy()
    parts = out["pair"].astype(str).str.split("-", n=1, expand=True)
    if parts.shape[1] == 2:
        out["_hetero"] = parts[0] != parts[1]
        out["_sb"] = parts.apply(
            lambda row: "Sb" in {str(row.iloc[0]), str(row.iloc[1])},
            axis=1,
        )
        out = out.sort_values(
            ["_sb", "_hetero", "nearest_distance_angstrom"],
            ascending=[False, False, True],
        )
    return out


targets_df = _read_csv(targets_csv)
pairs_df = _read_csv(pairs_csv)
nearest_df = _read_csv(nearest_csv)
pref_df = _read_csv(preference_csv)
sro_df = _read_csv(sro_csv)
vacancy_df = _read_csv(vacancy_csv)
nearest_vacancy_df = _read_csv(nearest_vacancy_csv)
vacancy_pref_df = _read_csv(vacancy_preference_csv)
triplet_target_df = _read_csv(triplet_target_csv)
triplet_pref_df = _read_csv(triplet_csv)
pair_scan_df = _read_csv(pair_scan_csv)
summary_payload = _read_json(summary_path, {})
mc_rows = _read_json(mc_summary_path, [])
if isinstance(mc_rows, dict):
    mc_rows = list(mc_rows.values())
if not isinstance(mc_rows, list):
    mc_rows = []

if summary_payload:
    with st.expander("Run summary", expanded=False):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Structures", int(summary_payload.get("n_targets", 0)))
        k2.metric("Dopant-pair records", int(summary_payload.get("n_dopant_pair_records", 0)))
        k3.metric("Dopant–Vₒ records", int(summary_payload.get("n_dopant_vacancy_records", 0)))
        k4.metric("Pair-scan records", int(summary_payload.get("n_pair_scan_rows", 0)))
        for warning in summary_payload.get("warnings", []) or []:
            st.warning(str(warning))
        for error in summary_payload.get("analysis_errors", []) or []:
            st.error(f"{error.get('target_id')}: {error.get('error')}")

structure_tab, scan_tab, mc_tab, raw_tab = st.tabs(
    ["Structure result", "Controlled pair scan", "Ordering MC", "Raw data"]
)

with structure_tab:
    if targets_df.empty or "target_id" not in targets_df.columns:
        st.info("No structure-level site-preference results are available yet.")
    else:
        st.markdown("### Select a structure")
        s1, s2, s3 = st.columns([1.0, 0.9, 2.1])

        compositions = (
            sorted(targets_df["composition"].dropna().astype(str).unique().tolist())
            if "composition" in targets_df.columns
            else ["all"]
        )
        selected_composition = s1.selectbox(
            "Composition",
            compositions,
            key="site_result_composition_v3",
        )
        filtered = targets_df.copy()
        if "composition" in filtered.columns:
            filtered = filtered[
                filtered["composition"].astype(str) == selected_composition
            ]

        kind_options = ["All"]
        if "structure_kind" in filtered.columns:
            kind_options += sorted(
                filtered["structure_kind"].dropna().astype(str).unique().tolist()
            )
        selected_kind = s2.selectbox(
            "Type",
            kind_options,
            key="site_result_kind_v3",
        )
        if selected_kind != "All" and "structure_kind" in filtered.columns:
            filtered = filtered[
                filtered["structure_kind"].astype(str) == selected_kind
            ]

        if filtered.empty:
            st.warning("No structures match the selected composition/type.")
        else:
            filtered = filtered.sort_values(
                "delta_energy_within_group_eV"
                if "delta_energy_within_group_eV" in filtered.columns
                else "target_id"
            ).copy()

            def _structure_label(row) -> str:
                delta = _fmt(row.get("delta_energy_within_group_eV"), 3, " eV")
                n_vac = int(row.get("n_oxygen_vacancies", 0))
                return f"{row.get('target_id')}   |   ΔE {delta}   |   Vₒ {n_vac}"

            label_map = {
                _structure_label(row): str(row["target_id"])
                for _, row in filtered.iterrows()
            }
            selected_label = s3.selectbox(
                "Structure",
                list(label_map.keys()),
                key="site_result_target_v3",
            )
            selected_target = label_map[selected_label]
            selected_meta = filtered[
                filtered["target_id"].astype(str) == selected_target
            ].iloc[0]

            selected_nearest = _one_target(nearest_df, selected_target)
            selected_pairs = _one_target(pairs_df, selected_target)
            selected_sro = _one_target(sro_df, selected_target)
            selected_vacancy = _one_target(nearest_vacancy_df, selected_target)
            selected_triplets = _one_target(triplet_target_df, selected_target)

            hero_kind = str(selected_meta.get("structure_kind", "structure"))
            hero_formula = str(selected_meta.get("formula", ""))
            hero_delta = selected_meta.get("delta_energy_within_group_eV")
            hero_text = (
                f"{hero_kind} • {hero_formula} • "
                f"{int(selected_meta.get('n_oxygen_vacancies', 0))} oxygen vacancy(ies)"
            )
            st.markdown(
                (
                    '<div class="df-hero">'
                    f'<div class="df-hero-title">{_safe(selected_target)}</div>'
                    f'<div class="df-hero-sub">{_safe(hero_text)}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Relative energy",
                _fmt(hero_delta, 3, " eV"),
                help="Relative to the lowest-energy structure with the same composition, structure type and vacancy count.",
            )
            m2.metric("Total energy", _fmt(selected_meta.get("energy_total_eV"), 3, " eV"))
            m3.metric("Atoms", str(selected_meta.get("n_atoms", "—")))
            m4.metric("Dopant pairs", str(selected_meta.get("n_dopant_pairs", "—")))

            st.markdown("### What this structure tells us")
            takeaways: list[tuple[str, str, str]] = []

            try:
                delta_value = float(hero_delta)
            except (TypeError, ValueError):
                delta_value = None
            if delta_value is not None and pd.notna(delta_value):
                if abs(delta_value) <= 1e-8:
                    takeaways.append(
                        (
                            "Energetic position",
                            "Lowest-energy structure in its directly comparable group.",
                            "This compares only the same composition, structure type and vacancy count.",
                        )
                    )
                else:
                    takeaways.append(
                        (
                            "Energetic position",
                            f"{delta_value:.3f} eV above the lowest-energy comparable structure.",
                            "Smaller ΔE means a more favorable configuration within this group.",
                        )
                    )

            prioritized_pairs = _pair_priority(selected_nearest)
            if not prioritized_pairs.empty:
                pair_row = prioritized_pairs.iloc[0]
                takeaways.append(
                    (
                        "Dopant geometry",
                        f"{pair_row['pair']}: {_shell_text(pair_row['nearest_shell'])} "
                        f"at {float(pair_row['nearest_distance_angstrom']):.2f} Å.",
                        "This is the nearest occurrence of the highlighted dopant pair in this structure.",
                    )
                )

            if not selected_sro.empty and "warren_cowley_alpha" in selected_sro.columns:
                valid_sro = selected_sro.dropna(subset=["warren_cowley_alpha"]).copy()
                if not valid_sro.empty:
                    sro_row = valid_sro.loc[
                        valid_sro["warren_cowley_alpha"].abs().idxmax()
                    ]
                    alpha = float(sro_row["warren_cowley_alpha"])
                    meaning = (
                        "association"
                        if alpha < -0.05
                        else "avoidance"
                        if alpha > 0.05
                        else "approximately random mixing"
                    )
                    takeaways.append(
                        (
                            "Local ordering",
                            f"{sro_row['pair']}: {meaning} at {_shell_text(sro_row['shell'])} "
                            f"(α = {alpha:.2f}).",
                            "This is a local occupancy statistic, not by itself an energetic pair-binding result.",
                        )
                    )

            if (
                str(selected_meta.get("structure_kind", "")) == "oxygen-vacancy"
                and not selected_vacancy.empty
            ):
                vrow = selected_vacancy.sort_values(
                    "nearest_distance_angstrom"
                ).iloc[0]
                takeaways.append(
                    (
                        "Vacancy environment",
                        f"Closest relation is {vrow['pair']} at "
                        f"{_shell_text(vrow['nearest_shell'])}, "
                        f"{float(vrow['nearest_distance_angstrom']):.2f} Å.",
                        "This describes the selected relaxed vacancy configuration.",
                    )
                )

            if not selected_triplets.empty:
                trow = selected_triplets.sort_values(
                    "n_instances", ascending=False
                ).iloc[0]
                takeaways.append(
                    (
                        "Three-dopant motif",
                        f"{trow['species_triplet']}: {_plain_motif(trow['motif'])}.",
                        f"Observed {int(trow['n_instances'])} time(s) in this structure.",
                    )
                )

            if not takeaways:
                st.info("No interpretable site-preference descriptors are available for this structure.")
            else:
                for start in range(0, len(takeaways), 2):
                    cols = st.columns(2)
                    for col, item in zip(cols, takeaways[start : start + 2]):
                        with col:
                            st.markdown(
                                _takeaway(item[0], item[1], item[2]),
                                unsafe_allow_html=True,
                            )

            st.divider()
            st.markdown("### Pair geometry")
            st.caption(
                "Read this section as geometry only: which dopant pairs are closest, "
                "at which coordination shell, and at what distance."
            )
            if selected_nearest.empty:
                st.info("No dopant-pair geometry is available for this structure.")
            else:
                pair_geom = selected_nearest[
                    ["pair", "nearest_shell", "nearest_distance_angstrom"]
                ].copy()
                pair_geom["Shell"] = pair_geom["nearest_shell"].map(_shell_text)
                pair_geom["Distance (Å)"] = pd.to_numeric(
                    pair_geom["nearest_distance_angstrom"], errors="coerce"
                ).round(3)
                pair_geom = pair_geom.rename(columns={"pair": "Pair"})
                pair_geom = pair_geom.sort_values("Distance (Å)")

                left, right = st.columns([1.15, 1.85])
                with left:
                    st.dataframe(
                        pair_geom[["Pair", "Shell", "Distance (Å)"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                with right:
                    geom_plot = pair_geom.dropna(subset=["Distance (Å)"]).copy()
                    if not geom_plot.empty:
                        fig = px.scatter(
                            geom_plot,
                            x="Distance (Å)",
                            y="Pair",
                            text="Shell",
                            labels={"Distance (Å)": "Nearest separation (Å)"},
                        )
                        fig.update_traces(
                            marker={"size": 14},
                            textposition="middle right",
                        )
                        fig.update_layout(
                            height=max(260, 58 * len(geom_plot)),
                            margin={"l": 10, "r": 45, "t": 20, "b": 35},
                            showlegend=False,
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"site_geom_v3_{selected_target}",
                        )

            st.divider()
            st.markdown("### Local ordering")
            st.caption(
                "For one selected dopant pair, α < 0 means association, α ≈ 0 random-like "
                "mixing, and α > 0 avoidance."
            )
            if selected_sro.empty:
                st.info("No Warren–Cowley local-ordering result is available for this structure.")
            else:
                sro_pairs = selected_sro["pair"].dropna().astype(str).unique().tolist()
                selected_sro_pair = st.selectbox(
                    "Pair to inspect",
                    sro_pairs,
                    key=f"site_sro_pair_v3_{selected_target}",
                )
                one_sro = selected_sro[
                    selected_sro["pair"].astype(str) == selected_sro_pair
                ].copy()
                one_sro = one_sro.dropna(
                    subset=["shell", "warren_cowley_alpha"]
                ).sort_values("shell")

                if one_sro.empty:
                    st.info("No valid shell-resolved α values are available for this pair.")
                else:
                    strongest = one_sro.loc[
                        one_sro["warren_cowley_alpha"].abs().idxmax()
                    ]
                    alpha = float(strongest["warren_cowley_alpha"])
                    meaning = (
                        "ASSOCIATES"
                        if alpha < -0.05
                        else "AVOIDS"
                        if alpha > 0.05
                        else "IS APPROXIMATELY RANDOM"
                    )
                    st.markdown(
                        _takeaway(
                            "Strongest local-order signal",
                            f"{selected_sro_pair} {meaning} at {_shell_text(strongest['shell'])}.",
                            f"α = {alpha:.2f}. The sign gives the direction of ordering; |α| gives its strength.",
                        ),
                        unsafe_allow_html=True,
                    )

                    sro_left, sro_right = st.columns([1.75, 1.0])
                    with sro_left:
                        fig = px.line(
                            one_sro,
                            x="shell",
                            y="warren_cowley_alpha",
                            markers=True,
                            labels={
                                "shell": "Coordination shell",
                                "warren_cowley_alpha": "α",
                            },
                        )
                        fig.add_hline(y=0, line_dash="dash")
                        fig.update_layout(
                            height=330,
                            margin={"l": 20, "r": 15, "t": 20, "b": 45},
                            showlegend=False,
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"site_sro_v3_{selected_target}_{selected_sro_pair}",
                        )
                    with sro_right:
                        sro_table = one_sro[
                            ["shell", "warren_cowley_alpha", "interpretation"]
                        ].copy()
                        sro_table["Shell"] = sro_table["shell"].map(_shell_text)
                        sro_table["α"] = pd.to_numeric(
                            sro_table["warren_cowley_alpha"], errors="coerce"
                        ).round(3)
                        sro_table["Meaning"] = (
                            sro_table["interpretation"]
                            .astype(str)
                            .str.replace("approximately-random", "random-like", regex=False)
                        )
                        st.dataframe(
                            sro_table[["Shell", "α", "Meaning"]],
                            use_container_width=True,
                            hide_index=True,
                        )

            if str(selected_meta.get("structure_kind", "")) == "oxygen-vacancy":
                st.divider()
                st.markdown("### Oxygen-vacancy environment")
                st.caption(
                    "This section answers which dopant lies closest to Vₒ in this particular relaxed structure."
                )
                if selected_vacancy.empty:
                    st.info("No mapped dopant–oxygen-vacancy distances are available.")
                else:
                    vplot = selected_vacancy[
                        ["pair", "nearest_shell", "nearest_distance_angstrom"]
                    ].copy()
                    vplot["Shell"] = vplot["nearest_shell"].map(_shell_text)
                    vplot["Distance (Å)"] = pd.to_numeric(
                        vplot["nearest_distance_angstrom"], errors="coerce"
                    ).round(3)
                    vplot = vplot.sort_values("Distance (Å)")
                    closest = vplot.iloc[0]
                    st.markdown(
                        _takeaway(
                            "Nearest dopant to Vₒ",
                            f"{closest['pair']} at {closest['Shell']}, {closest['Distance (Å)']:.2f} Å.",
                            "A lower distance means the dopant is geometrically closer to the vacancy in this structure.",
                        ),
                        unsafe_allow_html=True,
                    )
                    vleft, vright = st.columns([1.0, 1.8])
                    with vleft:
                        st.dataframe(
                            vplot.rename(columns={"pair": "Pair"})[
                                ["Pair", "Shell", "Distance (Å)"]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    with vright:
                        fig = px.bar(
                            vplot,
                            x="Distance (Å)",
                            y="pair",
                            orientation="h",
                            text="Shell",
                            labels={"pair": "Dopant–Vₒ"},
                        )
                        fig.update_layout(
                            height=max(250, 55 * len(vplot)),
                            margin={"l": 10, "r": 20, "t": 20, "b": 35},
                            showlegend=False,
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"site_vac_v3_{selected_target}",
                        )

            if not selected_triplets.empty:
                st.divider()
                st.markdown("### Three-dopant arrangement")
                st.caption(
                    "Shown only when three-dopant motifs actually exist in the selected structure."
                )
                triplet_species = (
                    selected_triplets["species_triplet"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                triplet_choice = st.selectbox(
                    "Triplet",
                    triplet_species,
                    key=f"site_triplet_v3_{selected_target}",
                )
                one_triplet = selected_triplets[
                    selected_triplets["species_triplet"].astype(str)
                    == triplet_choice
                ].copy()
                motif_counts = (
                    one_triplet.groupby("motif", as_index=False)["n_instances"].sum()
                    .sort_values("n_instances", ascending=False)
                )
                if not motif_counts.empty:
                    dominant = motif_counts.iloc[0]
                    st.markdown(
                        _takeaway(
                            "Dominant motif",
                            f"{triplet_choice}: {_plain_motif(dominant['motif'])}.",
                            f"{int(dominant['n_instances'])} occurrence(s) in this structure.",
                        ),
                        unsafe_allow_html=True,
                    )
                    motif_counts["Motif"] = motif_counts["motif"].map(_plain_motif)
                    motif_counts["Count"] = motif_counts["n_instances"].astype(int)
                    st.dataframe(
                        motif_counts[["Motif", "Count"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            st.divider()
            st.markdown("### Energetic context")
            st.caption(
                "This ranking contains only directly comparable structures: same composition, "
                "structure type, and oxygen-vacancy count. 0 eV is the best structure in that group."
            )
            comparable = targets_df.copy()
            mask = pd.Series(True, index=comparable.index)
            if "composition" in comparable.columns:
                mask &= comparable["composition"].astype(str) == str(
                    selected_meta.get("composition", "")
                )
            if "structure_kind" in comparable.columns:
                mask &= comparable["structure_kind"].astype(str) == str(
                    selected_meta.get("structure_kind", "")
                )
            if "n_oxygen_vacancies" in comparable.columns:
                mask &= pd.to_numeric(
                    comparable["n_oxygen_vacancies"], errors="coerce"
                ).fillna(-1) == float(selected_meta.get("n_oxygen_vacancies", 0))
            comparable = comparable[mask].dropna(
                subset=["delta_energy_within_group_eV"]
            ).copy()
            if comparable.empty:
                st.info("No comparable structure energies are available.")
            else:
                comparable["Selected"] = comparable["target_id"].astype(str).map(
                    lambda value: "Selected structure" if value == selected_target else "Other"
                )
                comparable["Short ID"] = comparable["target_id"].astype(str).map(
                    lambda value: value[-42:] if len(value) > 42 else value
                )
                comparable = comparable.sort_values(
                    "delta_energy_within_group_eV", ascending=True
                )
                fig = px.bar(
                    comparable,
                    x="delta_energy_within_group_eV",
                    y="Short ID",
                    orientation="h",
                    color="Selected",
                    labels={
                        "delta_energy_within_group_eV": "Relative energy ΔE (eV)",
                        "Short ID": "Structure",
                    },
                    category_orders={"Selected": ["Selected structure", "Other"]},
                )
                fig.update_layout(
                    height=max(300, min(700, 36 * len(comparable) + 120)),
                    margin={"l": 10, "r": 20, "t": 20, "b": 40},
                    legend_title_text="",
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"site_energy_rank_v3_{selected_target}",
                )

with scan_tab:
    st.markdown("### Controlled pair interaction")
    st.caption(
        "This is the clean energetic test: all other cation sites are returned to the host, "
        "then one dopant pair is placed at symmetry-distinct separations/orientations."
    )
    if pair_scan_df.empty or "pair" not in pair_scan_df.columns:
        st.info("No controlled pair-scan energies are available yet.")
    else:
        scan_pairs = pair_scan_df["pair"].dropna().astype(str).unique().tolist()
        scan_pair = st.selectbox(
            "Dopant pair",
            scan_pairs,
            key="site_pair_scan_v3",
        )
        scan_one = pair_scan_df[
            pair_scan_df["pair"].astype(str) == scan_pair
        ].copy()
        if "delta_E_vs_farthest_eV" not in scan_one.columns:
            st.warning("Pair structures exist, but evaluated ΔE values are not present.")
        else:
            evaluated = scan_one.dropna(
                subset=["delta_E_vs_farthest_eV"]
            ).copy()
            if evaluated.empty:
                st.warning("Pair structures were generated, but MLFF energies have not been evaluated.")
            else:
                distance_col = (
                    "final_distance_angstrom"
                    if "final_distance_angstrom" in evaluated.columns
                    and evaluated["final_distance_angstrom"].notna().any()
                    else "initial_distance_angstrom"
                )
                evaluated = evaluated.dropna(subset=[distance_col]).copy()
                best = evaluated.loc[
                    evaluated["delta_E_vs_farthest_eV"].idxmin()
                ]
                farthest_shell = int(
                    pd.to_numeric(evaluated["shell"], errors="coerce").max()
                )
                best_shell = int(float(best["shell"]))
                best_delta = float(best["delta_E_vs_farthest_eV"])
                if best_shell < farthest_shell and best_delta < -0.01:
                    interaction = "A closer dopant arrangement is energetically preferred."
                elif best_shell == farthest_shell:
                    interaction = "The largest tested separation is energetically preferred."
                else:
                    interaction = "No strong preference relative to the farthest tested separation is resolved."

                st.markdown(
                    _takeaway(
                        "Pair-scan conclusion",
                        interaction,
                        f"Best {scan_pair}: {_shell_text(best_shell)}, orbit {int(best['orbit'])}, "
                        f"{float(best[distance_col]):.2f} Å, ΔE = {best_delta:.3f} eV.",
                    ),
                    unsafe_allow_html=True,
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Preferred shell", _shell_text(best_shell))
                c2.metric("Separation", _fmt(best[distance_col], 2, " Å"))
                c3.metric("ΔE vs farthest", _fmt(best_delta, 3, " eV"))
                c4.metric("Configurations tested", str(len(evaluated)))

                evaluated["Label"] = [
                    f"{_shell_text(shell)} / o{int(orbit)}"
                    for shell, orbit in zip(evaluated["shell"], evaluated["orbit"])
                ]
                fig = px.scatter(
                    evaluated,
                    x=distance_col,
                    y="delta_E_vs_farthest_eV",
                    text="Label",
                    hover_data=[
                        col
                        for col in ["energy_total_eV", "degeneracy", "converged"]
                        if col in evaluated.columns
                    ],
                    labels={
                        distance_col: "Dopant separation (Å)",
                        "delta_E_vs_farthest_eV": "ΔE relative to farthest tested arrangement (eV)",
                    },
                )
                fig.add_hline(y=0, line_dash="dash")
                fig.update_traces(marker={"size": 13}, textposition="top center")
                fig.update_layout(
                    height=430,
                    margin={"l": 20, "r": 25, "t": 20, "b": 45},
                    showlegend=False,
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"site_pair_scan_plot_v3_{scan_pair}",
                )

                shell_summary = (
                    evaluated.groupby("shell", as_index=False)
                    .agg(
                        best_delta_eV=("delta_E_vs_farthest_eV", "min"),
                        mean_distance_angstrom=(distance_col, "mean"),
                        n_orientations=("orbit", "count"),
                    )
                    .sort_values("shell")
                )
                shell_summary["Shell"] = shell_summary["shell"].map(_shell_text)
                shell_summary["Best ΔE (eV)"] = shell_summary["best_delta_eV"].round(3)
                shell_summary["Mean distance (Å)"] = shell_summary[
                    "mean_distance_angstrom"
                ].round(3)
                shell_summary["Orientations"] = shell_summary[
                    "n_orientations"
                ].astype(int)
                st.dataframe(
                    shell_summary[
                        ["Shell", "Mean distance (Å)", "Best ΔE (eV)", "Orientations"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

with mc_tab:
    st.markdown("### Finite-temperature cation ordering")
    st.caption(
        "Monte Carlo keeps the composition fixed and samples cation occupations. "
        "Select one structure and one pair; the GUI translates the shell-resolved α values."
    )
    if not mc_rows:
        st.info("No ordering-MC result is available yet.")
    else:
        mc_labels = [
            str(row.get("target_id", f"target {idx + 1}"))
            for idx, row in enumerate(mc_rows)
        ]
        mc_choice = st.selectbox(
            "MC structure",
            mc_labels,
            key="site_mc_target_v3",
        )
        mc_row = next(
            row for row in mc_rows
            if str(row.get("target_id", "")) == mc_choice
        )

        start_energy = mc_row.get("mc_start_energy_eV")
        best_energy = mc_row.get("best_mc_energy_eV")
        try:
            improvement = float(best_energy) - float(start_energy)
        except (TypeError, ValueError):
            improvement = None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature", _fmt(mc_row.get("temperature_K"), 0, " K"))
        c2.metric(
            "Acceptance",
            _fmt(100.0 * float(mc_row.get("acceptance_fraction", 0.0)), 1, "%"),
        )
        c3.metric("Samples", str(mc_row.get("samples", "—")))
        c4.metric(
            "Best vs MC start",
            _fmt(improvement, 3, " eV") if improvement is not None else "—",
        )

        safe_target = (
            mc_choice.replace("\\", "__").replace("/", "__")
        )
        safe_target = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in safe_target
        )
        mc_sro = _read_csv(
            results_root
            / "ordering_mc"
            / safe_target
            / "sro_temperature_average.csv"
        )
        if mc_sro.empty:
            embedded = mc_row.get("sro_temperature_average", [])
            if isinstance(embedded, list) and embedded:
                mc_sro = pd.DataFrame(embedded)

        if mc_sro.empty or "pair" not in mc_sro.columns:
            st.info("No finite-temperature SRO values are available for this MC target.")
        else:
            mc_pairs = mc_sro["pair"].dropna().astype(str).unique().tolist()
            mc_pair = st.selectbox(
                "Pair to inspect",
                mc_pairs,
                key=f"site_mc_pair_v3_{safe_target}",
            )
            one_mc = mc_sro[
                mc_sro["pair"].astype(str) == mc_pair
            ].copy()
            alpha_col = next(
                (
                    col
                    for col in [
                        "mean_warren_cowley_alpha",
                        "warren_cowley_alpha",
                        "alpha_mean",
                    ]
                    if col in one_mc.columns
                ),
                None,
            )
            if alpha_col is None or "shell" not in one_mc.columns:
                st.warning("The MC SRO table does not contain shell-resolved α values.")
            else:
                one_mc = one_mc.dropna(
                    subset=[alpha_col, "shell"]
                ).sort_values("shell")
                if one_mc.empty:
                    st.info("No valid α values are available for this pair.")
                else:
                    strongest = one_mc.loc[one_mc[alpha_col].abs().idxmax()]
                    alpha = float(strongest[alpha_col])
                    meaning = (
                        "association"
                        if alpha < -0.05
                        else "avoidance"
                        if alpha > 0.05
                        else "approximately random mixing"
                    )
                    st.markdown(
                        _takeaway(
                            "Finite-temperature ordering",
                            f"{mc_pair}: {meaning} at {_shell_text(strongest['shell'])}.",
                            f"Average α = {alpha:.2f} at {float(mc_row.get('temperature_K', 0)):.0f} K.",
                        ),
                        unsafe_allow_html=True,
                    )

                    left, right = st.columns([1.8, 1.0])
                    with left:
                        fig = px.line(
                            one_mc,
                            x="shell",
                            y=alpha_col,
                            markers=True,
                            labels={
                                "shell": "Coordination shell",
                                alpha_col: "Average α",
                            },
                        )
                        fig.add_hline(y=0, line_dash="dash")
                        fig.update_layout(
                            height=350,
                            margin={"l": 20, "r": 15, "t": 20, "b": 45},
                            showlegend=False,
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"site_mc_plot_v3_{safe_target}_{mc_pair}",
                        )
                    with right:
                        mc_table = one_mc[["shell", alpha_col]].copy()
                        mc_table["Shell"] = mc_table["shell"].map(_shell_text)
                        mc_table["Average α"] = pd.to_numeric(
                            mc_table[alpha_col], errors="coerce"
                        ).round(3)
                        mc_table["Meaning"] = mc_table["Average α"].map(
                            lambda value: (
                                "association"
                                if value < -0.05
                                else "avoidance"
                                if value > 0.05
                                else "random-like"
                            )
                        )
                        st.dataframe(
                            mc_table[["Shell", "Average α", "Meaning"]],
                            use_container_width=True,
                            hide_index=True,
                        )

with raw_tab:
    st.markdown("### Raw result tables")
    st.caption(
        f"Output directory: {results_root}. These tables are for auditing/export; "
        "they are not intended as the primary scientific view."
    )
    raw_tables = [
        ("Structures", targets_df),
        ("All dopant pairs", pairs_df),
        ("Nearest pair by structure", nearest_df),
        ("Pair preference summary", pref_df),
        ("Warren–Cowley SRO", sro_df),
        ("All dopant–Vₒ pairs", vacancy_df),
        ("Nearest dopant–Vₒ by structure", nearest_vacancy_df),
        ("Dopant–Vₒ preference summary", vacancy_pref_df),
        ("Triplet motifs by structure", triplet_target_df),
        ("Triplet preference summary", triplet_pref_df),
        ("Controlled pair scan", pair_scan_df),
    ]
    any_raw = False
    for title, frame in raw_tables:
        if not frame.empty:
            any_raw = True
            with st.expander(title, expanded=False):
                st.dataframe(frame, use_container_width=True, hide_index=True)
    if not any_raw:
        st.info("No raw result tables are available yet.")

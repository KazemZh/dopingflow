"""Configure, run, and inspect dopant site-preference / ordering analysis."""

from __future__ import annotations

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
        )
    )
    mc_backend = m8.selectbox(
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
    mc_model = m9.text_input("MC model", value=str(mc_saved.get("model", "small")))
    m10, m11, m12 = st.columns(3)
    mc_relax_best = m10.checkbox(
        "Relax best MC occupation",
        value=bool(mc_saved.get("relax_best", True)),
    )
    mc_device = m11.selectbox(
        "MC device",
        ["cpu", "cuda"],
        index=1 if str(mc_saved.get("device", "cpu")).lower() == "cuda" else 0,
        key="site_mc_device",
    )
    mc_seed = int(
        m12.number_input(
            "MC random seed",
            value=int(mc_saved.get("seed", 42)),
            step=1,
        )
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
            "temperature_K": mc_temperature,
            "steps": mc_steps,
            "burn_in": mc_burn,
            "sample_interval": mc_interval,
            "seed": mc_seed,
            "backend": mc_backend,
            "model": mc_model.strip(),
            "device": mc_device,
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
st.subheader("Results")
st.caption(
    "Results are shown one structure at a time. Select a structure first, then inspect "
    "only the analyses that apply to it. Global/raw tables are kept at the bottom for auditing."
)

source_path = Path(source_root).expanduser()
if not source_path.is_absolute():
    source_path = (project_root / source_path).resolve()
results_root = Path(output_dir).expanduser()
if not results_root.is_absolute():
    results_root = (source_path / results_root).resolve()
st.caption(f"Resolved output: {results_root}")

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
mc_summary = results_root / "ordering_mc" / "ordering_mc_summary.json"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()


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

if summary_path.exists():
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        payload = {}
        st.warning(f"Could not read site-preference summary: {exc}")
    if payload:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Structures analysed", int(payload.get("n_targets", 0)))
        k2.metric("Dopant-pair records", int(payload.get("n_dopant_pair_records", 0)))
        k3.metric("Dopant–Vₒ records", int(payload.get("n_dopant_vacancy_records", 0)))
        k4.metric("Pair-scan records", int(payload.get("n_pair_scan_rows", 0)))
        for warning in payload.get("warnings", []) or []:
            st.warning(str(warning))
        for error in payload.get("analysis_errors", []) or []:
            st.error(f"{error.get('target_id')}: {error.get('error')}")
elif targets_df.empty:
    st.info("No site-preference results found yet. Save the settings and run the stage.")

if not targets_df.empty and "target_id" in targets_df.columns:
    st.markdown("### 1. Structure browser")
    st.write(
        "Choose one composition and one relaxed structure. The panels below answer the "
        "site-preference questions for that structure only."
    )

    composition_options = sorted(
        targets_df["composition"].dropna().astype(str).unique().tolist()
    ) if "composition" in targets_df.columns else ["all"]
    selected_composition = st.selectbox(
        "Composition",
        composition_options,
        key="site_pref_result_composition",
    )
    subset = targets_df.copy()
    if "composition" in subset.columns:
        subset = subset[subset["composition"].astype(str) == selected_composition]

    kind_options = ["All"]
    if "structure_kind" in subset.columns:
        kind_options += sorted(subset["structure_kind"].dropna().astype(str).unique().tolist())
    selected_kind = st.selectbox(
        "Structure type",
        kind_options,
        key="site_pref_result_kind",
    )
    if selected_kind != "All" and "structure_kind" in subset.columns:
        subset = subset[subset["structure_kind"].astype(str) == selected_kind]

    if subset.empty:
        st.warning("No structures match the selected filters.")
    else:
        def _target_label(row) -> str:
            target = str(row.get("target_id", ""))
            kind = str(row.get("structure_kind", ""))
            n_vac = row.get("n_oxygen_vacancies", 0)
            delta = row.get("delta_energy_within_group_eV")
            delta_text = _fmt(delta, 3, " eV")
            return f"{target}  |  {kind}  |  Vₒ={n_vac}  |  ΔE={delta_text}"

        label_to_target = {
            _target_label(row): str(row["target_id"])
            for _, row in subset.iterrows()
        }
        selected_label = st.selectbox(
            "Structure",
            list(label_to_target.keys()),
            key="site_pref_result_target",
        )
        selected_target = label_to_target[selected_label]
        selected_meta = subset[
            subset["target_id"].astype(str) == selected_target
        ].iloc[0]

        st.markdown("#### At a glance")
        a, b, c, d, e = st.columns(5)
        a.metric("Composition", str(selected_meta.get("composition", "—")))
        b.metric("Type", str(selected_meta.get("structure_kind", "—")))
        c.metric("O vacancies", int(selected_meta.get("n_oxygen_vacancies", 0)))
        d.metric(
            "ΔE in same group",
            _fmt(selected_meta.get("delta_energy_within_group_eV"), 3, " eV"),
            help="Energy above the lowest-energy structure with the same composition, structure type and vacancy count.",
        )
        e.metric(
            "Total energy",
            _fmt(selected_meta.get("energy_total_eV"), 3, " eV"),
        )

        selected_nearest = _one_target(nearest_df, selected_target)
        selected_sro = _one_target(sro_df, selected_target)
        selected_vacancy = _one_target(nearest_vacancy_df, selected_target)
        selected_triplets = _one_target(triplet_target_df, selected_target)

        tab_pair, tab_order, tab_vac, tab_triplet, tab_compare = st.tabs(
            [
                "Dopant pairs",
                "Local ordering",
                "O-vacancy relation",
                "Three-dopant motifs",
                "Compare same composition",
            ]
        )

        with tab_pair:
            st.markdown("#### Where are the dopants relative to each other?")
            if selected_nearest.empty:
                st.info("No dopant-pair records are available for this structure.")
            else:
                simple = selected_nearest.copy()
                simple["Nearest distance (Å)"] = simple["nearest_distance_angstrom"].map(
                    lambda x: round(float(x), 3) if pd.notna(x) else None
                )
                simple["Nearest shell"] = simple["nearest_shell"].map(_shell_text)
                simple = simple.rename(columns={"pair": "Pair"})
                st.dataframe(
                    simple[["Pair", "Nearest shell", "Nearest distance (Å)"]],
                    use_container_width=True,
                    hide_index=True,
                )

                pair_choices = simple["Pair"].astype(str).unique().tolist()
                selected_pair = st.selectbox(
                    "Inspect pair",
                    pair_choices,
                    key=f"site_pref_pair_{selected_target}",
                )
                row = simple[simple["Pair"].astype(str) == selected_pair].iloc[0]
                st.success(
                    f"In this structure, the closest **{selected_pair}** pair is "
                    f"**{_shell_text(row['nearest_shell'])}** at "
                    f"**{_fmt(row['nearest_distance_angstrom'], 3, ' Å')}**."
                )

                pair_detail = _one_target(pairs_df, selected_target)
                if not pair_detail.empty and "pair" in pair_detail.columns:
                    pair_detail = pair_detail[pair_detail["pair"].astype(str) == selected_pair].copy()
                    if not pair_detail.empty:
                        pair_detail["label"] = [
                            f"{_shell_text(shell)} · {float(dist):.2f} Å"
                            for shell, dist in zip(
                                pair_detail["shell"],
                                pair_detail["distance_angstrom"],
                            )
                        ]
                        fig = px.bar(
                            pair_detail.sort_values("distance_angstrom"),
                            x="label",
                            y="distance_angstrom",
                            labels={
                                "label": "Pair occurrence",
                                "distance_angstrom": "Distance (Å)",
                            },
                            title=f"{selected_pair} separations in the selected structure",
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"site_pref_pair_dist_{selected_target}_{selected_pair}",
                        )

        with tab_order:
            st.markdown("#### Do the dopants locally associate or avoid each other?")
            st.caption(
                "Warren–Cowley α is translated here into plain language: "
                "negative = association, near zero = roughly random, positive = avoidance."
            )
            if selected_sro.empty:
                st.info(
                    "No local-ordering result is available for this structure. "
                    "This can happen when there are too few atoms of a dopant type."
                )
            else:
                sro_pairs = selected_sro["pair"].dropna().astype(str).unique().tolist()
                sro_pair = st.selectbox(
                    "Dopant pair",
                    sro_pairs,
                    key=f"site_pref_sro_pair_{selected_target}",
                )
                one_sro = selected_sro[
                    selected_sro["pair"].astype(str) == sro_pair
                ].copy().sort_values("shell")
                if not one_sro.empty:
                    strongest = one_sro.loc[
                        one_sro["warren_cowley_alpha"].abs().idxmax()
                    ]
                    alpha = float(strongest["warren_cowley_alpha"])
                    if alpha < -0.05:
                        message = "association — the pair occurs together more than expected randomly"
                    elif alpha > 0.05:
                        message = "avoidance — the pair occurs together less than expected randomly"
                    else:
                        message = "approximately random local mixing"
                    st.success(
                        f"Strongest signal for **{sro_pair}**: **{message}** at "
                        f"**{_shell_text(strongest['shell'])}** "
                        f"(α = {alpha:.2f})."
                    )
                    fig = px.bar(
                        one_sro,
                        x="shell",
                        y="warren_cowley_alpha",
                        labels={
                            "shell": "Coordination shell",
                            "warren_cowley_alpha": "α  (− associate, + avoid)",
                        },
                        title=f"{sro_pair}: local ordering by coordination shell",
                    )
                    fig.add_hline(y=0)
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"site_pref_sro_{selected_target}_{sro_pair}",
                    )
                    display_sro = one_sro[
                        [
                            "shell",
                            "shell_center_angstrom",
                            "warren_cowley_alpha",
                            "interpretation",
                        ]
                    ].copy()
                    display_sro.columns = [
                        "Shell",
                        "Shell distance (Å)",
                        "α",
                        "Meaning",
                    ]
                    st.dataframe(display_sro, use_container_width=True, hide_index=True)

        with tab_vac:
            st.markdown("#### Which dopant is closest to the oxygen vacancy?")
            if str(selected_meta.get("structure_kind", "")) != "oxygen-vacancy":
                st.info("This is a vacancy-free structure, so this analysis does not apply.")
            elif selected_vacancy.empty:
                st.info("No mapped dopant–oxygen-vacancy records are available for this structure.")
            else:
                vac_simple = selected_vacancy.copy().sort_values("nearest_distance_angstrom")
                closest = vac_simple.iloc[0]
                st.success(
                    f"Closest dopant–vacancy relation: **{closest['pair']}**, "
                    f"**{_shell_text(closest['nearest_shell'])}**, "
                    f"**{_fmt(closest['nearest_distance_angstrom'], 3, ' Å')}**."
                )
                vac_display = vac_simple[
                    ["pair", "nearest_shell", "nearest_distance_angstrom"]
                ].copy()
                vac_display["nearest_shell"] = vac_display["nearest_shell"].map(_shell_text)
                vac_display.columns = ["Dopant–Vₒ", "Nearest shell", "Nearest distance (Å)"]
                st.dataframe(vac_display, use_container_width=True, hide_index=True)
                fig = px.bar(
                    vac_simple,
                    x="pair",
                    y="nearest_distance_angstrom",
                    labels={
                        "pair": "Dopant–Vₒ pair",
                        "nearest_distance_angstrom": "Nearest distance (Å)",
                    },
                    title="Nearest dopant–oxygen-vacancy distances",
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"site_pref_vac_{selected_target}",
                )

        with tab_triplet:
            st.markdown("#### How are groups of three dopants arranged?")
            st.caption(
                "Compact triangle = all three close; connected chain = two neighbor links; "
                "isolated pair + third = only one neighbor link; dispersed = no close links."
            )
            if selected_triplets.empty:
                st.info("No three-dopant motif is present or available for this structure.")
            else:
                triplet_species = selected_triplets[
                    "species_triplet"
                ].dropna().astype(str).unique().tolist()
                triplet_choice = st.selectbox(
                    "Dopant triplet",
                    triplet_species,
                    key=f"site_pref_triplet_{selected_target}",
                )
                one_triplet = selected_triplets[
                    selected_triplets["species_triplet"].astype(str) == triplet_choice
                ].copy()
                if not one_triplet.empty:
                    best_count = one_triplet.sort_values(
                        "n_instances", ascending=False
                    ).iloc[0]
                    st.success(
                        f"For **{triplet_choice}**, the most common motif in this structure is "
                        f"**{_plain_motif(best_count['motif'])}** "
                        f"({int(best_count['n_instances'])} occurrence(s))."
                    )
                    plot_triplet = (
                        one_triplet.groupby("motif", as_index=False)["n_instances"].sum()
                    )
                    plot_triplet["motif"] = plot_triplet["motif"].map(_plain_motif)
                    fig = px.bar(
                        plot_triplet,
                        x="motif",
                        y="n_instances",
                        labels={"motif": "Motif", "n_instances": "Count"},
                        title=f"{triplet_choice}: motif counts in selected structure",
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"site_pref_triplet_plot_{selected_target}_{triplet_choice}",
                    )
                    st.dataframe(
                        plot_triplet.rename(
                            columns={"motif": "Motif", "n_instances": "Count"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

        with tab_compare:
            st.markdown("#### Does a particular arrangement correlate with lower energy?")
            st.caption(
                "This comparison uses only structures with the same composition, structure type "
                "and oxygen-vacancy count. It is a configuration-energy trend, not an isolated "
                "pair-binding energy."
            )
            if selected_nearest.empty:
                st.info("No pair information is available for comparison.")
            else:
                compare_pairs = selected_nearest["pair"].dropna().astype(str).unique().tolist()
                compare_pair = st.selectbox(
                    "Pair to compare across structures",
                    compare_pairs,
                    key=f"site_pref_compare_pair_{selected_target}",
                )
                comp_rows = nearest_df.copy()
                conditions = pd.Series(True, index=comp_rows.index)
                if "composition" in comp_rows.columns:
                    conditions &= comp_rows["composition"].astype(str) == str(
                        selected_meta.get("composition", "")
                    )
                if "structure_kind" in comp_rows.columns:
                    conditions &= comp_rows["structure_kind"].astype(str) == str(
                        selected_meta.get("structure_kind", "")
                    )
                if "n_oxygen_vacancies" in comp_rows.columns:
                    conditions &= pd.to_numeric(
                        comp_rows["n_oxygen_vacancies"], errors="coerce"
                    ).fillna(-1) == float(selected_meta.get("n_oxygen_vacancies", 0))
                conditions &= comp_rows["pair"].astype(str) == compare_pair
                comp_rows = comp_rows[conditions].copy()
                comp_rows = comp_rows.dropna(
                    subset=["nearest_distance_angstrom", "delta_energy_within_group_eV"]
                )
                if len(comp_rows) < 2:
                    st.info(
                        "At least two comparable structures are needed to infer an energetic "
                        "site-preference trend."
                    )
                else:
                    best = comp_rows.loc[
                        comp_rows["delta_energy_within_group_eV"].idxmin()
                    ]
                    st.success(
                        f"Lowest-energy observed **{compare_pair}** arrangement: "
                        f"**{_shell_text(best['nearest_shell'])}** at "
                        f"**{_fmt(best['nearest_distance_angstrom'], 3, ' Å')}**."
                    )
                    fig = px.scatter(
                        comp_rows,
                        x="nearest_distance_angstrom",
                        y="delta_energy_within_group_eV",
                        hover_name="target_id",
                        symbol="nearest_shell",
                        labels={
                            "nearest_distance_angstrom": f"Nearest {compare_pair} distance (Å)",
                            "delta_energy_within_group_eV": "ΔE within comparable structures (eV)",
                        },
                        title=f"{compare_pair}: distance versus relative configuration energy",
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"site_pref_compare_{selected_target}_{compare_pair}",
                    )

                pref_match = pref_df.copy()
                if not pref_match.empty:
                    mask = pd.Series(True, index=pref_match.index)
                    if "composition" in pref_match.columns:
                        mask &= pref_match["composition"].astype(str) == str(
                            selected_meta.get("composition", "")
                        )
                    if "structure_kind" in pref_match.columns:
                        mask &= pref_match["structure_kind"].astype(str) == str(
                            selected_meta.get("structure_kind", "")
                        )
                    if "n_oxygen_vacancies" in pref_match.columns:
                        mask &= pd.to_numeric(
                            pref_match["n_oxygen_vacancies"], errors="coerce"
                        ).fillna(-1) == float(selected_meta.get("n_oxygen_vacancies", 0))
                    if "pair" in pref_match.columns:
                        mask &= pref_match["pair"].astype(str) == compare_pair
                    pref_match = pref_match[mask]
                    if not pref_match.empty:
                        p = pref_match.iloc[0]
                        st.info(
                            f"Summary across this group: preferred observed shell = "
                            f"**{_shell_text(p.get('preferred_shell'))}** "
                            f"(mean distance {_fmt(p.get('preferred_shell_mean_distance_angstrom'), 3, ' Å')})."
                        )

    st.markdown("### 2. Controlled pair scan")
    st.caption(
        "This is the cleaner energetic test. Choose one dopant pair at a time; "
        "the plot compares only its symmetry-distinct shell/orientation calculations."
    )
    if pair_scan_df.empty:
        st.info("No controlled pair-scan results are available yet.")
    elif "pair" not in pair_scan_df.columns:
        st.warning("pair_scan.csv does not contain a pair column.")
    else:
        scan_pairs = pair_scan_df["pair"].dropna().astype(str).unique().tolist()
        scan_pair = st.selectbox(
            "Controlled pair",
            scan_pairs,
            key="site_pref_controlled_pair",
        )
        scan_one = pair_scan_df[
            pair_scan_df["pair"].astype(str) == scan_pair
        ].copy()
        evaluated = scan_one.dropna(subset=["delta_E_vs_farthest_eV"]).copy()
        if evaluated.empty:
            st.warning(
                "Structures were generated for this pair, but no evaluated MLFF energies are "
                "available yet."
            )
            cols = [
                col for col in
                ["shell", "orbit", "initial_distance_angstrom", "run_directory"]
                if col in scan_one.columns
            ]
            if cols:
                st.dataframe(scan_one[cols], use_container_width=True, hide_index=True)
        else:
            distance_col = (
                "final_distance_angstrom"
                if "final_distance_angstrom" in evaluated.columns
                and evaluated["final_distance_angstrom"].notna().any()
                else "initial_distance_angstrom"
            )
            best = evaluated.loc[evaluated["delta_E_vs_farthest_eV"].idxmin()]
            sign_text = (
                "favored relative to the farthest tested separation"
                if float(best["delta_E_vs_farthest_eV"]) < -1e-6
                else "similar in energy to the farthest tested separation"
            )
            st.success(
                f"Best **{scan_pair}** arrangement: **{_shell_text(best['shell'])}**, "
                f"orbit **{int(best['orbit'])}**, "
                f"{_fmt(best[distance_col], 3, ' Å')}; "
                f"ΔE = **{_fmt(best['delta_E_vs_farthest_eV'], 3, ' eV')}**, {sign_text}."
            )
            fig = px.scatter(
                evaluated,
                x=distance_col,
                y="delta_E_vs_farthest_eV",
                symbol="shell",
                hover_data=[
                    col for col in ["orbit", "degeneracy", "energy_total_eV"]
                    if col in evaluated.columns
                ],
                labels={
                    distance_col: "Dopant separation (Å)",
                    "delta_E_vs_farthest_eV": "ΔE vs farthest tested shell (eV)",
                },
                title=f"{scan_pair}: controlled interaction scan",
            )
            fig.add_hline(y=0)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"site_pref_pair_scan_{scan_pair}",
            )
            scan_cols = [
                col for col in
                [
                    "shell",
                    "orbit",
                    distance_col,
                    "delta_E_vs_farthest_eV",
                    "energy_total_eV",
                    "converged",
                ]
                if col in evaluated.columns
            ]
            st.dataframe(
                evaluated.sort_values("delta_E_vs_farthest_eV")[scan_cols],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### 3. Finite-temperature ordering MC")
    st.caption(
        "Choose one MC target at a time. This summarizes collective cation ordering at the "
        "configured temperature rather than mixing all compositions in one table."
    )
    if not mc_summary.exists():
        st.info("No ordering-MC summary is available yet.")
    else:
        try:
            mc_rows = json.loads(mc_summary.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            st.warning(f"Could not read MC summary: {exc}")
            mc_rows = []
        if isinstance(mc_rows, dict):
            mc_rows = list(mc_rows.values())
        if not mc_rows:
            st.info("The ordering-MC summary is empty.")
        else:
            mc_labels = [
                str(row.get("target_id", f"target {idx + 1}"))
                for idx, row in enumerate(mc_rows)
            ]
            mc_choice = st.selectbox(
                "MC structure",
                mc_labels,
                key="site_pref_mc_target",
            )
            mc_row = next(
                row for row in mc_rows
                if str(row.get("target_id", "")) == mc_choice
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Temperature", _fmt(mc_row.get("temperature_K"), 0, " K"))
            m2.metric(
                "Acceptance",
                _fmt(
                    100.0 * float(mc_row.get("acceptance_fraction", 0.0)),
                    1,
                    "%",
                ),
            )
            m3.metric("Samples", str(mc_row.get("samples", "—")))
            m4.metric(
                "Best MC energy",
                _fmt(mc_row.get("best_mc_energy_eV"), 3, " eV"),
            )

            safe_target = mc_choice.replace("/", "__")
            sro_average_path = (
                results_root / "ordering_mc" / safe_target / "sro_temperature_average.csv"
            )
            mc_sro = _read_csv(sro_average_path)
            if not mc_sro.empty and "pair" in mc_sro.columns:
                mc_pair = st.selectbox(
                    "MC ordering pair",
                    mc_sro["pair"].dropna().astype(str).unique().tolist(),
                    key=f"site_pref_mc_pair_{safe_target}",
                )
                mc_one = mc_sro[mc_sro["pair"].astype(str) == mc_pair].copy()
                alpha_col = next(
                    (
                        col for col in
                        ["mean_warren_cowley_alpha", "warren_cowley_alpha", "alpha_mean"]
                        if col in mc_one.columns
                    ),
                    None,
                )
                if alpha_col and "shell" in mc_one.columns:
                    mc_one = mc_one.dropna(subset=[alpha_col]).copy()
                    if not mc_one.empty:
                        strongest_mc = mc_one.loc[mc_one[alpha_col].abs().idxmax()]
                        alpha_mc = float(strongest_mc[alpha_col])
                        if alpha_mc < -0.05:
                            mc_meaning = "association"
                        elif alpha_mc > 0.05:
                            mc_meaning = "avoidance"
                        else:
                            mc_meaning = "approximately random mixing"
                        st.success(
                            f"At **{_fmt(mc_row.get('temperature_K'), 0, ' K')}**, "
                            f"**{mc_pair}** shows its strongest average signal at "
                            f"**{_shell_text(strongest_mc['shell'])}**: "
                            f"**{mc_meaning}** (α = {alpha_mc:.2f})."
                        )
                        fig = px.bar(
                            mc_one.sort_values("shell"),
                            x="shell",
                            y=alpha_col,
                            labels={
                                "shell": "Coordination shell",
                                alpha_col: "Average α  (− associate, + avoid)",
                            },
                            title=f"{mc_pair}: finite-temperature local ordering",
                        )
                        fig.add_hline(y=0)
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"site_pref_mc_sro_{safe_target}_{mc_pair}",
                        )
                with st.expander("MC SRO values", expanded=False):
                    st.dataframe(mc_one, use_container_width=True, hide_index=True)

    with st.expander("Raw / global result tables", expanded=False):
        st.caption(
            "These tables are retained for auditing and export. They are intentionally not "
            "the main visualization."
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
        for title, frame in raw_tables:
            if not frame.empty:
                st.markdown(f"**{title}**")
                st.dataframe(frame, use_container_width=True, hide_index=True)

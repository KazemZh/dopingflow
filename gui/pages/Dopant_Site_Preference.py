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

source_path = Path(source_root).expanduser()
if not source_path.is_absolute():
    source_path = (project_root / source_path).resolve()
results_root = Path(output_dir).expanduser()
if not results_root.is_absolute():
    results_root = (source_path / results_root).resolve()
st.caption(f"Resolved output: {results_root}")

summary_path = results_root / "site_preference_summary.json"
preference_csv = results_root / "pair_preference_summary.csv"
nearest_csv = results_root / "nearest_pair_by_target.csv"
sro_csv = results_root / "warren_cowley_sro.csv"
vacancy_csv = results_root / "dopant_vacancy_pairs.csv"
vacancy_preference_csv = results_root / "dopant_vacancy_preference_summary.csv"
triplet_csv = results_root / "triplet_motif_summary.csv"
pair_scan_csv = results_root / "pair_scan" / "pair_scan.csv"
mc_summary = results_root / "ordering_mc" / "ordering_mc_summary.json"

if summary_path.exists():
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Structures", int(payload.get("n_targets", 0)))
    k2.metric("Dopant pairs", int(payload.get("n_dopant_pair_records", 0)))
    k3.metric("Triplets", int(payload.get("n_triplet_records", 0)))
    k4.metric("Dopant–Vₒ pairs", int(payload.get("n_dopant_vacancy_records", 0)))
    k5.metric("SRO records", int(payload.get("n_sro_records", 0)))
    for warning in payload.get("warnings", []) or []:
        st.warning(str(warning))
    for error in payload.get("analysis_errors", []) or []:
        st.error(f"{error.get('target_id')}: {error.get('error')}")
else:
    st.info("No site_preference_summary.json found yet. Save the settings and run the stage.")

if preference_csv.exists() and preference_csv.stat().st_size:
    pref = pd.read_csv(preference_csv)
    st.markdown("#### Preferred dopant–dopant shells")
    st.dataframe(pref, use_container_width=True, hide_index=True)
    if not pref.empty and "median_delta_energy_within_group_eV" in pref:
        plot_df = pref.dropna(subset=["median_delta_energy_within_group_eV"])
        if not plot_df.empty:
            fig = px.scatter(
                plot_df,
                x="preferred_shell",
                y="median_delta_energy_within_group_eV",
                color="pair",
                hover_data=["composition", "preferred_shell_mean_distance_angstrom"],
                labels={
                    "preferred_shell": "Preferred coordination shell",
                    "median_delta_energy_within_group_eV": "Median ΔE within composition (eV)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

if nearest_csv.exists() and nearest_csv.stat().st_size:
    with st.expander("Nearest pair in every structure", expanded=False):
        st.dataframe(pd.read_csv(nearest_csv), use_container_width=True, hide_index=True)

if sro_csv.exists() and sro_csv.stat().st_size:
    st.markdown("#### Warren–Cowley short-range order")
    sro = pd.read_csv(sro_csv)
    st.dataframe(sro, use_container_width=True, hide_index=True)
    if not sro.empty:
        sro_plot = sro.dropna(subset=["warren_cowley_alpha"])
        if not sro_plot.empty:
            fig = px.scatter(
                sro_plot,
                x="shell",
                y="warren_cowley_alpha",
                color="pair",
                symbol="structure_kind",
                hover_data=["target_id", "shell_center_angstrom"],
                labels={
                    "shell": "Coordination shell",
                    "warren_cowley_alpha": "Warren–Cowley α",
                },
            )
            fig.add_hline(y=0)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("α < 0: association; α ≈ 0: random; α > 0: avoidance.")

if triplet_csv.exists() and triplet_csv.stat().st_size:
    st.markdown("#### Triple-dopant motif preferences")
    triplets = pd.read_csv(triplet_csv)
    st.dataframe(triplets, use_container_width=True, hide_index=True)
    st.caption(
        "Motifs: compact triangle = 3 neighbor edges; connected chain = 2; "
        "isolated pair + third = 1; dispersed = 0. The neighbor-shell cutoff is configurable above."
    )

if vacancy_preference_csv.exists() and vacancy_preference_csv.stat().st_size:
    st.markdown("#### Preferred dopant–oxygen-vacancy shells")
    vacancy_pref = pd.read_csv(vacancy_preference_csv)
    st.dataframe(vacancy_pref, use_container_width=True, hide_index=True)

if vacancy_csv.exists() and vacancy_csv.stat().st_size:
    with st.expander("All dopant–oxygen-vacancy distances", expanded=False):
        vacancy = pd.read_csv(vacancy_csv)
        st.dataframe(vacancy, use_container_width=True, hide_index=True)

if pair_scan_csv.exists() and pair_scan_csv.stat().st_size:
    st.markdown("#### Controlled pair-shell scan")
    pair_scan = pd.read_csv(pair_scan_csv)
    st.dataframe(pair_scan, use_container_width=True, hide_index=True)
    if "delta_E_vs_farthest_eV" in pair_scan:
        evaluated = pair_scan.dropna(subset=["delta_E_vs_farthest_eV"])
        if not evaluated.empty:
            fig = px.line(
                evaluated.sort_values(["pair", "final_distance_angstrom"]),
                x="final_distance_angstrom",
                y="delta_E_vs_farthest_eV",
                color="pair",
                markers=True,
                labels={
                    "final_distance_angstrom": "Dopant separation (Å)",
                    "delta_E_vs_farthest_eV": "ΔE vs farthest shell (eV)",
                },
            )
            fig.add_hline(y=0)
            st.plotly_chart(fig, use_container_width=True)

if mc_summary.exists():
    st.markdown("#### Finite-temperature ordering MC")
    try:
        mc_rows = json.loads(mc_summary.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read MC summary: {exc}")
    else:
        flattened = [
            {
                "target_id": row.get("target_id"),
                "temperature_K": row.get("temperature_K"),
                "acceptance_fraction": row.get("acceptance_fraction"),
                "samples": row.get("samples"),
                "best_mc_energy_eV": row.get("best_mc_energy_eV"),
            }
            for row in mc_rows
        ]
        if flattened:
            st.dataframe(pd.DataFrame(flattened), use_container_width=True, hide_index=True)

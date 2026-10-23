from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd
import streamlit as st
import toml

from gui_config import GRACE_MODEL_CHOICES, MACE_MODEL_CHOICES
from vacancy_staged import (
    build_staged_command,
    format_int_list,
    parse_parent_selectors,
    parse_positive_int_list,
)
from vacancy_mc_results import (
    lowest_energy_by_vacancy_count,
    preferred_vacancy_heatmap,
    preferred_vacancy_map,
    validate_search_database,
)


st.set_page_config(page_title="Staged Vacancy MC", layout="wide")
st.title("Staged Vacancy Monte Carlo: GRACE → MACE")
st.caption(
    "Configure and run the two-environment vacancy workflow: joint cation/vacancy "
    "Monte Carlo with GRACE, followed by MACE single points, relaxation, reranking, "
    "and vacancy thermodynamics."
)

project_root = Path(
    st.text_input("Project root", value=str(Path.cwd()), key="staged_vac_project_root")
).expanduser().resolve()
config_path = project_root / "input.toml"

if config_path.is_file():
    cfg = toml.load(str(config_path))
else:
    cfg = {}
    st.warning(f"No input.toml found yet at `{config_path}`. Saving will create it.")

vac = cfg.setdefault("vacancies", {})

st.info(
    "The staged commands intentionally run in separate environments. GRACE only needs "
    "to be installed in the search environment; MACE only needs to be installed in the "
    "finalize environment. Both stages use the same input.toml and, when configured, the "
    "same [vacancies].output_directory handoff tree."
)

with st.form("staged_vacancy_config"):
    st.subheader("Parent selection and staged output")
    c1, c2 = st.columns(2)
    parent_source = c1.selectbox(
        "parent_source",
        ["directory", "selected_candidates"],
        index=0 if str(vac.get("parent_source", "directory")) == "directory" else 1,
        help="Use directory for an existing multi-composition parent tree.",
    )
    parent_directory = c2.text_input(
        "parent_directory",
        value=str(vac.get("parent_directory", "vacancy-selected")),
        disabled=parent_source != "directory",
    )

    c3, c4 = st.columns(2)
    parent_pick = c3.selectbox(
        "parent_pick",
        ["lowest_energy", "all"],
        index=0 if str(vac.get("parent_pick", "lowest_energy")) == "lowest_energy" else 1,
        help=(
            "lowest_energy keeps the first selected candidate per composition. The normal "
            "filtering stage writes selected_candidates.txt in ascending relaxed-energy order."
        ),
    )
    output_directory = c4.text_input(
        "output_directory",
        value=str(vac.get("output_directory", "vacancy-mc-grace-mace")),
        help=(
            "Dedicated result tree shared by vacancies-mc-search and vacancies-finalize. "
            "Source parent structures remain untouched."
        ),
    )

    parent_text = st.text_area(
        "parent_include (one composition or exact composition/candidate ID per line)",
        value="\n".join(str(x) for x in vac.get("parent_include", [])),
        height=180,
        help=(
            "Composition selectors are matched order/notation-insensitively for common labels, "
            "for example Ti_2.5Sb_5, Ti2p5_Sb5, and Sb5_Ti2p5. Use composition/candidate_XXX "
            "for an exact parent ID. Leave blank to process every discovered composition."
        ),
    )

    st.subheader("Search space")
    counts_text = st.text_input(
        "vacancy_counts",
        value=format_int_list(vac.get("vacancy_counts", [1, 2, 3, 4])),
        help="Explicit fixed vacancy counts. The current large-cell study uses 1, 2, 3, 4.",
    )
    sc = list(vac.get("supercell", [2, 2, 2]))
    if len(sc) != 3:
        sc = [2, 2, 2]
    s1, s2, s3 = st.columns(3)
    super_a = s1.number_input("supercell a", min_value=1, value=int(sc[0]), step=1)
    super_b = s2.number_input("supercell b", min_value=1, value=int(sc[1]), step=1)
    super_c = s3.number_input("supercell c", min_value=1, value=int(sc[2]), step=1)

    st.subheader("GRACE Monte Carlo search")
    g1, g2, g3 = st.columns(3)
    mc_backend = g1.selectbox(
        "mc_backend",
        ["grace", "mace", "m3gnet", "uma"],
        index=["grace", "mace", "m3gnet", "uma"].index(str(vac.get("mc_backend", "grace"))),
    )
    current_grace_model = str(vac.get("mc_model", "GRACE-1L-OMAT"))
    grace_options = list(GRACE_MODEL_CHOICES)
    if current_grace_model not in grace_options:
        grace_options = [current_grace_model, *grace_options]
    mc_model = g2.selectbox(
        "mc_model",
        grace_options,
        index=grace_options.index(current_grace_model),
        disabled=mc_backend != "grace",
    )
    mc_task = g3.text_input("mc_task", value=str(vac.get("mc_task", "")))

    g4, g5 = st.columns(2)
    mc_device = g4.selectbox(
        "mc_device",
        ["cuda", "cpu"],
        index=0 if str(vac.get("mc_device", "cuda")) == "cuda" else 1,
    )
    mc_gpu_id = g5.number_input(
        "mc_gpu_id", min_value=0, value=int(vac.get("mc_gpu_id", 0)), step=1
    )

    annealing = st.checkbox("mc_annealing", value=bool(vac.get("mc_annealing", True)))
    t1, t2 = st.columns(2)
    initial_temperature = t1.number_input(
        "mc_initial_temperature_K",
        min_value=1.0,
        value=float(vac.get("mc_initial_temperature_K", 1500.0)),
        step=50.0,
        disabled=not annealing,
    )
    target_temperature = t2.number_input(
        "mc_temperature_K",
        min_value=1.0,
        value=float(vac.get("mc_temperature_K", 600.0)),
        step=50.0,
    )

    a1, a2 = st.columns(2)
    hold_steps = a1.number_input(
        "mc_annealing_hold_steps",
        min_value=0,
        value=int(vac.get("mc_annealing_hold_steps", 5000)),
        step=1000,
        disabled=not annealing,
    )
    annealing_steps = a2.number_input(
        "mc_annealing_steps",
        min_value=0,
        value=int(vac.get("mc_annealing_steps", 50000)),
        step=5000,
        disabled=not annealing,
    )

    r1, r2, r3 = st.columns(3)
    run_mode = r1.selectbox(
        "mc_run_mode",
        ["combined", "fixed", "converged"],
        index=["combined", "fixed", "converged"].index(str(vac.get("mc_run_mode", "combined"))),
    )
    max_steps = r2.number_input(
        "mc_max_steps",
        min_value=1,
        value=int(vac.get("mc_max_steps", 500000)),
        step=10000,
    )
    patience = r3.number_input(
        "mc_patience",
        min_value=1,
        value=int(vac.get("mc_patience", 105000)),
        step=5000,
    )
    st.caption(
        "mc_max_steps is the total trajectory length, including the hot hold and cooling ramp. "
        "The current implementation counts patience from step 1. With 5,000 hold + 50,000 "
        "cooling steps, mc_patience=105,000 guarantees that a no-improvement stop cannot occur "
        "before the schedule plus roughly 50,000 additional trial moves at 600 K."
    )

    p1, p2, p3 = st.columns(3)
    improvement_tol = p1.number_input(
        "mc_improvement_tolerance_eV",
        min_value=0.0,
        value=float(vac.get("mc_improvement_tolerance_eV", 0.001)),
        step=0.001,
        format="%.6f",
        help="Best-so-far energy improvement required to reset the no-improvement counter.",
    )
    energy_window = p2.number_input(
        "mc_energy_window_eV",
        min_value=0.0,
        value=float(vac.get("mc_energy_window_eV", 1.0)),
        step=0.1,
    )
    sample_seed = p3.number_input(
        "sample_seed", value=int(vac.get("sample_seed", 42)), step=1
    )

    w1, w2, w3 = st.columns(3)
    cation_weight = w1.number_input(
        "mc_cation_move_weight",
        min_value=0.0,
        value=float(vac.get("mc_cation_move_weight", 0.5)),
        step=0.1,
    )
    vacancy_weight = w2.number_input(
        "mc_vacancy_move_weight",
        min_value=0.0,
        value=float(vac.get("mc_vacancy_move_weight", 0.5)),
        step=0.1,
    )
    sample_max_saved = w3.number_input(
        "sample_max_saved",
        min_value=1,
        value=int(vac.get("sample_max_saved", 100)),
        step=10,
    )

    st.subheader("MACE finalization")
    m1, m2, m3 = st.columns(3)
    final_backend = m1.selectbox(
        "backend",
        ["mace", "grace", "m3gnet", "uma"],
        index=["mace", "grace", "m3gnet", "uma"].index(str(vac.get("backend", "mace"))),
    )
    current_mace_model = str(vac.get("model", "mh-1"))
    mace_options = list(MACE_MODEL_CHOICES)
    if current_mace_model not in mace_options:
        mace_options = [current_mace_model, *mace_options]
    final_model = m2.selectbox(
        "model",
        mace_options,
        index=mace_options.index(current_mace_model),
        disabled=final_backend != "mace",
    )
    final_task = m3.text_input("task", value=str(vac.get("task", "matpes_r2scan")))

    m4, m5, m6 = st.columns(3)
    final_device = m4.selectbox(
        "device",
        ["cuda", "cpu"],
        index=0 if str(vac.get("device", "cuda")) == "cuda" else 1,
    )
    final_gpu_id = m5.number_input(
        "gpu_id", min_value=0, value=int(vac.get("gpu_id", 0)), step=1
    )
    topk = m6.number_input(
        "topk_per_vacancy_count",
        min_value=1,
        value=int(vac.get("topk_per_vacancy_count", 20)),
        step=1,
    )

    q1, q2, q3 = st.columns(3)
    optimizer = q1.selectbox(
        "optimizer",
        ["bfgs", "lbfgs", "fire", "mdmin", "quasinewton"],
        index=["bfgs", "lbfgs", "fire", "mdmin", "quasinewton"].index(
            str(vac.get("optimizer", "bfgs"))
        ),
    )
    fmax = q2.number_input(
        "fmax", min_value=0.001, value=float(vac.get("fmax", 0.05)), step=0.01
    )
    relax_steps = q3.number_input(
        "max_steps", min_value=1, value=int(vac.get("max_steps", 300)), step=25
    )

    save = st.form_submit_button("Save staged vacancy settings to input.toml", type="primary")

if save:
    try:
        vacancy_counts = parse_positive_int_list(counts_text, field_name="vacancy_counts")
        selectors = parse_parent_selectors(parent_text)
        if cation_weight + vacancy_weight <= 0:
            raise ValueError("At least one MC move weight must be positive")
        if annealing and initial_temperature < target_temperature:
            raise ValueError("mc_initial_temperature_K must be >= mc_temperature_K")
        if run_mode in {"combined", "converged"} and annealing:
            schedule_steps = int(hold_steps) + int(annealing_steps)
            if int(patience) <= schedule_steps:
                st.warning(
                    "mc_patience is not larger than the annealing schedule. The current MC "
                    "implementation can therefore satisfy the patience criterion before cooling "
                    "has completed. Increase patience if the full schedule is required."
                )

        vac.update(
            {
                "enabled": True,
                "parent_source": parent_source,
                "parent_pick": parent_pick,
                "output_directory": output_directory.strip(),
                "include_parent_reference": bool(vac.get("include_parent_reference", True)),
                "skip_if_done": bool(vac.get("skip_if_done", True)),
                "resume": bool(vac.get("resume", True)),
                "search_method": "monte-carlo",
                "vacancy_counts": vacancy_counts,
                "supercell": [int(super_a), int(super_b), int(super_c)],
                "mc_backend": mc_backend,
                "mc_model": mc_model,
                "mc_task": mc_task,
                "mc_device": mc_device,
                "mc_gpu_id": int(mc_gpu_id),
                "mc_annealing": annealing,
                "mc_initial_temperature_K": float(initial_temperature),
                "mc_annealing_hold_steps": int(hold_steps),
                "mc_annealing_steps": int(annealing_steps),
                "mc_temperature_K": float(target_temperature),
                "mc_run_mode": run_mode,
                "mc_max_steps": int(max_steps),
                "mc_patience": int(patience),
                "mc_improvement_tolerance_eV": float(improvement_tol),
                "mc_energy_window_eV": float(energy_window),
                "mc_cation_move_weight": float(cation_weight),
                "mc_vacancy_move_weight": float(vacancy_weight),
                "sample_seed": int(sample_seed),
                "sample_max_saved": int(sample_max_saved),
                "backend": final_backend,
                "model": final_model,
                "task": final_task,
                "device": final_device,
                "gpu_id": int(final_gpu_id),
                "topk_per_vacancy_count": int(topk),
                "optimizer": optimizer,
                "fmax": float(fmax),
                "max_steps": int(relax_steps),
            }
        )
        if parent_source == "directory":
            vac["parent_directory"] = parent_directory.strip()
        else:
            vac.pop("parent_directory", None)
        if selectors:
            vac["parent_include"] = selectors
        else:
            vac.pop("parent_include", None)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(toml.dumps(cfg), encoding="utf-8")
        st.success(f"Saved `{config_path}`")
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.subheader("Run the two stages in their own environments")
run1, run2, run3 = st.columns(3)
grace_env = run1.text_input("GRACE conda environment", value="dopingflow-grace")
mace_env = run2.text_input("MACE conda environment", value="dopingflow-mace")
conda_executable = run3.text_input("conda executable", value="conda")

search_cmd = build_staged_command(
    "search",
    env_name=grace_env,
    config_path=config_path,
    conda_executable=conda_executable,
)
finalize_cmd = build_staged_command(
    "finalize",
    env_name=mace_env,
    config_path=config_path,
    conda_executable=conda_executable,
)

st.markdown("**GRACE search command**")
st.code(" ".join(search_cmd), language="bash")
st.markdown("**MACE finalize command**")
st.code(" ".join(finalize_cmd), language="bash")

b1, b2 = st.columns(2)
if b1.button("Run GRACE MC search", type="primary"):
    if not config_path.is_file():
        st.error("Save input.toml first.")
    else:
        with st.spinner("Running staged GRACE Monte Carlo search..."):
            proc = subprocess.run(
                search_cmd,
                cwd=str(project_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        st.code(proc.stdout or "", language="text")
        if proc.returncode == 0:
            st.success("GRACE MC search completed.")
        else:
            st.error(f"GRACE MC search exited with code {proc.returncode}.")

if b2.button("Run MACE finalize", type="primary"):
    if not config_path.is_file():
        st.error("Save input.toml first.")
    else:
        with st.spinner("Running MACE finalization..."):
            proc = subprocess.run(
                finalize_cmd,
                cwd=str(project_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        st.code(proc.stdout or "", language="text")
        if proc.returncode == 0:
            st.success("MACE finalization completed.")
        else:
            st.error(f"MACE finalization exited with code {proc.returncode}.")

st.warning(
    "Current staged selection is GRACE archive → GRACE top-k → MACE single point/relaxation "
    "of those selected candidates. MACE does not yet rescore every archived GRACE candidate "
    "before top-k selection. Validate GRACE/MACE ranking agreement on a small test before a "
    "large production campaign."
)

st.divider()
st.subheader("Results")
st.caption(
    "Inspect the Stage-1 GRACE archive and map the preferred number of oxygen "
    "vacancies versus oxygen partial pressure and applied potential."
)

result_root = Path(output_directory.strip()).expanduser()
if not result_root.is_absolute():
    result_root = project_root / result_root
search_database_path = result_root / "vacancy_mc_search_database.csv"

if not search_database_path.is_file():
    st.info(
        "No Stage-1 search database is available yet. Expected: "
        f"`{search_database_path}`"
    )
else:
    try:
        search_results = pd.read_csv(search_database_path)
        validate_search_database(search_results)
        zero_rows = search_results[search_results["n_vacancies"].astype(int) == 0]

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Archived structures", f"{len(search_results):,}")
        metric2.metric("Parents", search_results["parent_id"].nunique())
        metric3.metric("Zero-vacancy references", len(zero_rows))
        available_counts = sorted(
            int(value) for value in search_results["n_vacancies"].unique()
        )
        metric4.metric("Vacancy counts", ", ".join(map(str, available_counts)))

        st.markdown("#### Preferred vacancy number: pO₂–potential heatmap")
        st.warning(
            "Preliminary Stage-1 screening only. The map uses unrelaxed GRACE "
            "single-point energies. Its absolute pressure and potential boundaries "
            "depend on the supplied model-consistent oxygen reference and assumed "
            "vacancy charge/electron-transfer model."
        )

        parent_labels = (
            search_results[["parent_id", "composition"]]
            .drop_duplicates()
            .sort_values(["composition", "parent_id"])
        )
        label_to_parent = {
            f"{row.composition} — {row.parent_id}": str(row.parent_id)
            for row in parent_labels.itertuples(index=False)
        }
        selected_label = st.selectbox(
            "Parent structure",
            list(label_to_parent),
            key="staged_vacancy_heatmap_parent",
        )
        selected_parent = label_to_parent[selected_label]

        h1, h2, h3 = st.columns(3)
        map_temperature = h1.number_input(
            "Temperature (K)",
            min_value=100.0,
            max_value=6000.0,
            value=float(vac.get("mc_heatmap_temperature_K", 1000.0)),
            step=50.0,
            key="staged_vacancy_heatmap_temperature",
        )
        configured_mu = vac.get("mu_O_reference_eV")
        default_mu = float(configured_mu) if configured_mu is not None else -7.5
        mu_o_reference = h2.number_input(
            "Model-consistent μO reference (eV/O)",
            value=default_mu,
            step=0.05,
            format="%.6f",
            help=(
                "Use one half of the O2 energy calculated with the same Stage-1 "
                "backend/model, or a validated calibrated per-O reference."
            ),
            key="staged_vacancy_heatmap_mu_o_reference",
        )
        electrons_per_vacancy = h3.number_input(
            "Electrons released per vacancy, z",
            min_value=0.0,
            value=float(vac.get("mc_heatmap_electrons_per_vacancy", 2.0)),
            step=1.0,
            help=(
                "The potential term is -n·z·U. z=2 corresponds to an assumed "
                "V_O^(2+) vacancy reaction releasing two electrons."
            ),
            key="staged_vacancy_heatmap_electrons",
        )
        if configured_mu is None:
            st.error(
                "No `[vacancies].mu_O_reference_eV` is configured. The displayed "
                "-7.5 eV/O is only an editable plotting placeholder; do not interpret "
                "the pressure boundaries quantitatively until it is replaced by a "
                "GRACE-consistent or calibrated oxygen reference."
            )

        p1, p2, u1, u2 = st.columns(4)
        logp_min = p1.number_input(
            "Minimum log10(pO2/bar)",
            value=float(vac.get("log10_pO2_min_bar", -20.0)),
            step=1.0,
            key="staged_vacancy_heatmap_logp_min",
        )
        logp_max = p2.number_input(
            "Maximum log10(pO2/bar)",
            value=float(vac.get("log10_pO2_max_bar", 1.0)),
            step=1.0,
            key="staged_vacancy_heatmap_logp_max",
        )
        potential_min = u1.number_input(
            "Minimum potential (V)",
            value=float(vac.get("mc_heatmap_potential_min_V", -1.0)),
            step=0.1,
            key="staged_vacancy_heatmap_potential_min",
        )
        potential_max = u2.number_input(
            "Maximum potential (V)",
            value=float(vac.get("mc_heatmap_potential_max_V", 2.0)),
            step=0.1,
            key="staged_vacancy_heatmap_potential_max",
        )
        include_nist = st.checkbox(
            "Include NIST O2 standard-state thermal correction",
            value=True,
            help=(
                "Adds the per-O ideal-gas standard-state correction derived from "
                "NIST Shomate data. Explicit O2 zero-point energy remains excluded."
            ),
            key="staged_vacancy_heatmap_nist",
        )

        heatmap_result = preferred_vacancy_map(
            search_results,
            selected_parent,
            temperature_K=float(map_temperature),
            mu_O_reference_eV_per_O=float(mu_o_reference),
            log10_pO2_min_bar=float(logp_min),
            log10_pO2_max_bar=float(logp_max),
            potential_min_V=float(potential_min),
            potential_max_V=float(potential_max),
            electrons_released_per_vacancy=float(electrons_per_vacancy),
            include_nist_standard_state=include_nist,
        )
        heatmap_figure = preferred_vacancy_heatmap(
            heatmap_result,
            parent_id=selected_parent,
            temperature_K=float(map_temperature),
            electrons_released_per_vacancy=float(electrons_per_vacancy),
            include_nist_standard_state=include_nist,
        )
        st.plotly_chart(heatmap_figure, use_container_width=True)
        st.latex(
            r"\Delta\Omega_n(U,p_{O_2},T)="
            r"E_n-E_0+n\mu_O(T,p_{O_2})-nzU"
        )
        st.caption(
            "Each color gives the vacancy count with the lowest screening grand "
            "energy at that pressure and potential. Solid vibrational, electronic, "
            "magnetic and configurational free-energy terms are not included."
        )

        with st.expander("Lowest GRACE-energy candidate used for each vacancy count"):
            minima = lowest_energy_by_vacancy_count(
                search_results, selected_parent
            )
            visible_columns = [
                "n_vacancies",
                "configuration_id",
                "mc_search_energy_sp_total_eV",
                "delta_energy_to_parent_eV",
                "generated_poscar_path",
            ]
            st.dataframe(minima[visible_columns], use_container_width=True)
    except Exception as exc:
        st.error(f"Could not build the Stage-1 vacancy heatmap: {exc}")

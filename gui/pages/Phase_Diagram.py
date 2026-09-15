from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import toml


GUI_DIR = Path(__file__).resolve().parents[1]
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from phase_diagram_plots import (  # noqa: E402
    available_hull_quantities,
    build_phase_composition_map,
    build_phase_diagram_figure,
    build_vacancy_hull_figure,
    prepare_phase_diagram_plot_data,
    prepare_vacancy_hull_data,
    select_hull_quantity,
)


st.set_page_config(page_title="dopingflow Phase Diagram", layout="wide")
st.title("Phase Diagram & Vacancy Hull")
st.caption(
    "Inspect raw/corrected energy above hull in composition space and as a function "
    "of oxygen-vacancy count."
)

project_root = Path(
    st.sidebar.text_input("Project root", value=str(Path.cwd()), key="pd_project_root")
).expanduser().resolve()
input_toml = project_root / "input.toml"


def _load_config() -> dict:
    return toml.load(str(input_toml)) if input_toml.exists() else {}


def _resolve_vacancy_root(cfg: dict) -> Path:
    phase = cfg.get("phase_diagram", {}) or {}
    explicit = str(phase.get("vacancy_results_directory", "")).strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()
    vacancies = cfg.get("vacancies", {}) or {}
    if str(vacancies.get("parent_source", "")).strip() == "directory":
        configured = str(vacancies.get("parent_directory", "")).strip()
        if configured:
            path = Path(configured).expanduser()
            return path.resolve() if path.is_absolute() else (project_root / path).resolve()
    structure = cfg.get("structure", {}) or {}
    return (project_root / str(structure.get("outdir", "random_structures"))).resolve()


cfg = _load_config()
phase_cfg = dict(cfg.get("phase_diagram", {}) or {})

with st.expander("Phase-diagram configuration", expanded=False):
    st.caption(
        "These controls edit only the [phase_diagram] section of input.toml. "
        "The vacancy-hull option reuses the existing corrected phase-diagram machinery."
    )
    col1, col2 = st.columns(2)
    include_vacancies = col1.checkbox(
        "Include lowest-energy vacancy minima",
        value=bool(phase_cfg.get("include_vacancy_minima", False)),
        help=(
            "Add the lowest-energy relaxed structure at every investigated oxygen-vacancy "
            "count to the same raw/corrected phase diagram."
        ),
    )
    skip_if_done = col2.checkbox(
        "Skip existing phase diagram",
        value=bool(phase_cfg.get("skip_if_done", True)),
        help=(
            "When vacancy minima are included, dopingflow forces a rebuild because an old "
            "phase_diagram_results.csv does not contain the vacancy entries."
        ),
    )
    vacancy_directory = st.text_input(
        "Vacancy results directory",
        value=str(phase_cfg.get("vacancy_results_directory", "")),
        placeholder="vacancy-selected",
        help=(
            "Directory containing vacancy_static_minima.csv and the corresponding relaxed "
            "vacancy structures. Leave blank to use the vacancy workflow location."
        ),
    )
    threshold = st.number_input(
        "Stability threshold (eV/atom)",
        min_value=0.0,
        value=float(phase_cfg.get("stable_threshold_eV_per_atom", 1.0e-8)),
        step=0.01,
        format="%.4f",
    )
    if st.button("Save phase-diagram settings", type="primary"):
        cfg.setdefault("phase_diagram", {})
        cfg["phase_diagram"]["include_vacancy_minima"] = include_vacancies
        cfg["phase_diagram"]["skip_if_done"] = skip_if_done
        cfg["phase_diagram"]["stable_threshold_eV_per_atom"] = float(threshold)
        if vacancy_directory.strip():
            cfg["phase_diagram"]["vacancy_results_directory"] = vacancy_directory.strip()
        else:
            cfg["phase_diagram"].pop("vacancy_results_directory", None)
        input_toml.write_text(toml.dumps(cfg), encoding="utf-8")
        st.success(f"Saved {input_toml}")
        cfg = _load_config()

phase_path = project_root / "phase_diagram_results.csv"
vacancy_root = _resolve_vacancy_root(cfg)
vacancy_hull_path = project_root / "vacancy_energy_above_hull.csv"
if not vacancy_hull_path.exists():
    alternate = vacancy_root / "vacancy_energy_above_hull.csv"
    if alternate.exists():
        vacancy_hull_path = alternate

phase_tab, vacancy_tab = st.tabs(["Phase diagram", "Vacancy E above hull"])

with phase_tab:
    if not phase_path.exists():
        st.info(
            "No phase_diagram_results.csv was found. Run `dopingflow phase-diagram -c input.toml` first."
        )
    else:
        try:
            raw_phase = pd.read_csv(phase_path, low_memory=False)
        except Exception as exc:
            st.error(f"Could not read {phase_path}: {exc}")
        else:
            if raw_phase.empty:
                st.warning("phase_diagram_results.csv is empty.")
            elif "chemical_system" not in raw_phase.columns:
                st.error("phase_diagram_results.csv has no chemical_system column.")
            else:
                systems = sorted(raw_phase["chemical_system"].dropna().astype(str).unique())
                controls_a, controls_b = st.columns(2)
                selected_system = controls_a.selectbox(
                    "Chemical system",
                    systems,
                    key="pd_system",
                )
                quantity_options = available_hull_quantities(raw_phase)
                default_quantity = "Corrected" if "Corrected" in quantity_options else "Raw"
                quantity = controls_b.radio(
                    "Hull energy",
                    quantity_options,
                    index=quantity_options.index(default_quantity),
                    horizontal=True,
                    key="pd_quantity",
                )

                system_raw = raw_phase[
                    raw_phase["chemical_system"].astype(str) == selected_system
                ].copy()
                system_selected = select_hull_quantity(system_raw, quantity)

                database_path = project_root / "results_database.csv"
                if not database_path.exists():
                    st.warning(
                        "results_database.csv is missing, so composition-coordinate plots "
                        "cannot be prepared. The phase table is still available below."
                    )
                    plot_data = pd.DataFrame()
                    dopants: list[str] = []
                else:
                    try:
                        database = pd.read_csv(
                            database_path,
                            usecols=lambda column: column
                            in {
                                "candidate_path",
                                "effective_pct_json",
                                "requested_pct_json",
                                "dopant_counts_json",
                            },
                            low_memory=False,
                        )
                        plot_data, dopants = prepare_phase_diagram_plot_data(
                            system_selected,
                            database,
                        )
                    except Exception as exc:
                        st.warning(f"Could not prepare composition plot: {exc}")
                        plot_data = pd.DataFrame()
                        dopants = []

                if not plot_data.empty and dopants:
                    unmatched = int(plot_data.attrs.get("unmatched_phase_rows", 0))
                    if unmatched:
                        st.caption(
                            f"{unmatched} phase row(s) were not present in the main results "
                            "database. This is expected for vacancy configurations; they are "
                            "shown in the Vacancy E above hull tab."
                        )

                    if len(dopants) >= 2:
                        view_mode = st.radio(
                            "Composition view",
                            ["2D composition map", "Concentration curves"],
                            horizontal=True,
                            key="pd_view_mode",
                        )
                        axis_a, axis_b = st.columns(2)
                        default_x = "Sb" if "Sb" in dopants else dopants[0]
                        x_dopant = axis_a.selectbox(
                            "x dopant",
                            dopants,
                            index=dopants.index(default_x),
                            key="pd_x",
                        )
                        other = [dopant for dopant in dopants if dopant != x_dopant]
                        y_default = "Ti" if "Ti" in other else other[0]
                        second_dopant = axis_b.selectbox(
                            "Second dopant",
                            other,
                            index=other.index(y_default),
                            key="pd_y",
                        )
                        try:
                            if view_mode == "2D composition map":
                                figure, shown = build_phase_composition_map(
                                    plot_data,
                                    chemical_system=selected_system,
                                    x_dopant=x_dopant,
                                    y_dopant=second_dopant,
                                )
                            else:
                                figure, shown = build_phase_diagram_figure(
                                    plot_data,
                                    chemical_system=selected_system,
                                    x_dopant=x_dopant,
                                    series_dopant=second_dopant,
                                    include_host_reference=False,
                                )
                            st.plotly_chart(
                                figure,
                                use_container_width=True,
                                key="pd_main_plot",
                            )
                        except ValueError as exc:
                            st.warning(str(exc))
                    else:
                        try:
                            figure, shown = build_phase_diagram_figure(
                                plot_data,
                                chemical_system=selected_system,
                                x_dopant=dopants[0],
                                series_dopant=None,
                                include_host_reference=False,
                            )
                            st.plotly_chart(
                                figure,
                                use_container_width=True,
                                key="pd_single_dopant_plot",
                            )
                        except ValueError as exc:
                            st.warning(str(exc))

                display_columns = [
                    column
                    for column in (
                        "candidate",
                        "formula",
                        "energy_above_hull_eV_per_atom",
                        "stable",
                        "decomposition",
                        "candidate_path",
                    )
                    if column in system_selected.columns
                ]
                st.subheader("Selected-system data")
                st.dataframe(
                    system_selected[display_columns].sort_values(
                        "energy_above_hull_eV_per_atom"
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

with vacancy_tab:
    if not vacancy_hull_path.exists():
        st.info(
            "No vacancy_energy_above_hull.csv was found. Enable `include_vacancy_minima` "
            "and rerun `dopingflow phase-diagram -c input.toml`."
        )
    else:
        try:
            vacancy_raw = pd.read_csv(vacancy_hull_path, low_memory=False)
        except Exception as exc:
            st.error(f"Could not read {vacancy_hull_path}: {exc}")
        else:
            if vacancy_raw.empty:
                st.warning("vacancy_energy_above_hull.csv is empty.")
            else:
                quantity_options = available_hull_quantities(vacancy_raw)
                default_quantity = "Corrected" if "Corrected" in quantity_options else "Raw"
                quantity = st.radio(
                    "Vacancy hull energy",
                    quantity_options,
                    index=quantity_options.index(default_quantity),
                    horizontal=True,
                    key="vac_hull_quantity",
                )
                try:
                    vacancy_data = prepare_vacancy_hull_data(
                        vacancy_raw,
                        quantity=quantity,
                    )
                except Exception as exc:
                    st.error(f"Could not prepare vacancy-hull plot: {exc}")
                else:
                    compositions = sorted(
                        vacancy_data["actual_composition_key"].dropna().astype(str).unique()
                    )
                    selected = st.multiselect(
                        "Compositions",
                        compositions,
                        default=compositions,
                        key="vac_hull_compositions",
                    )
                    if selected:
                        try:
                            figure, shown = build_vacancy_hull_figure(
                                vacancy_data,
                                selected_compositions=selected,
                            )
                            st.plotly_chart(
                                figure,
                                use_container_width=True,
                                key="vacancy_hull_plot",
                            )
                        except ValueError as exc:
                            st.warning(str(exc))
                    else:
                        shown = vacancy_data.iloc[0:0]
                        st.info("Select at least one composition.")

                    st.info(
                        "This is a closed-system energy-above-hull descriptor for each "
                        "oxygen-deficient composition. It is different from the vacancy "
                        "formation free energy and from an oxygen-open grand-potential hull."
                    )
                    table_columns = [
                        column
                        for column in (
                            "actual_composition_key",
                            "n_vacancies",
                            "vacancy_percent_of_parent_oxygen",
                            "formula",
                            "energy_above_hull_eV_per_atom",
                            "stable",
                            "decomposition",
                            "source_configuration_id",
                        )
                        if column in vacancy_data.columns
                    ]
                    st.subheader("Vacancy hull data")
                    table = vacancy_data[
                        vacancy_data["actual_composition_key"].astype(str).isin(selected)
                    ] if selected else vacancy_data.iloc[0:0]
                    st.dataframe(
                        table[table_columns].sort_values(
                            ["actual_composition_key", "n_vacancies"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

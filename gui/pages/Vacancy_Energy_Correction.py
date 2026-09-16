from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import toml


st.set_page_config(page_title="Vacancy Energy Correction", layout="wide")
st.title("Vacancy M0/M1 Energy Correction")
st.caption(
    "Optionally apply the already fitted M0/M1 formation-energy correction to "
    "oxygen-vacancy thermodynamics."
)

project_root = Path(
    st.sidebar.text_input(
        "Project root",
        value=str(Path.cwd()),
        key="vac_corr_project_root",
    )
).expanduser().resolve()
config_path = project_root / "input.toml"

if not config_path.exists():
    st.error(f"No input.toml found at {config_path}")
    st.stop()

cfg = toml.load(str(config_path))
vacancies = dict(cfg.get("vacancies", {}) or {})
energy_correction = dict(cfg.get("energy_correction", {}) or {})

correction_enabled = bool(energy_correction.get("enabled", False))
configured_family = str(energy_correction.get("model_family", "manual")).strip().lower()
oxygen_mode = str(vacancies.get("oxygen_reference_mode", "reference_file")).strip().lower()
oxygen_mode = oxygen_mode.replace("_", "-")

left, right = st.columns(2)
with left:
    st.metric("Energy correction", "Enabled" if correction_enabled else "Disabled")
with right:
    st.metric("Configured family", configured_family.upper())

st.write(f"Current vacancy oxygen-reference mode: `{oxygen_mode}`")

apply_correction = st.checkbox(
    "Apply fitted M0/M1 correction to vacancy thermodynamics",
    value=bool(vacancies.get("apply_fitted_energy_correction", False)),
    help=(
        "Correct the selected n=0 parent and every selected vacancy minimum with the "
        "already fitted backend-specific model before comparing different vacancy counts."
    ),
)
allow_legacy = st.checkbox(
    "Allow legacy vacancy-energy provenance",
    value=bool(vacancies.get("allow_legacy_energy_correction_provenance", False)),
    help=(
        "Use only for older vacancy calculations that predate package-version and "
        "relaxed-POSCAR hashes. Known backend/model/task/settings mismatches are still rejected."
    ),
)

invalid_reasons: list[str] = []
if apply_correction:
    if not correction_enabled:
        invalid_reasons.append(
            "[energy_correction].enabled must be true and a fitted correction model must exist."
        )
    if configured_family not in {"m0", "m1", "auto"}:
        invalid_reasons.append(
            "Set [energy_correction].model_family to 'm0', 'm1', or 'auto' and run "
            "dopingflow corrections-fit first."
        )
    if oxygen_mode in {"global", "chemistry-specific"}:
        invalid_reasons.append(
            "The fitted M0/M1 vacancy correction cannot be combined with the global or "
            "chemistry-specific experimental oxygen calibration because that would double "
            "count the oxygen-related calibration. Use reference_file or same_calculator."
        )

if invalid_reasons:
    for reason in invalid_reasons:
        st.error(reason)
else:
    if apply_correction:
        st.success(
            "The vacancy analysis will use the fitted correction. If model_family='auto', "
            "the M0 or M1 family selected by corrections-fit is used."
        )
    else:
        st.info("Vacancy thermodynamics will continue to use the raw relaxed ML solid energies.")

st.markdown(
    """
**What changes when enabled**

The workflow keeps the raw values but evaluates the cross-vacancy-count energy as

`ΔE_corrected = ΔE_raw + C(vacancy structure) - C(parent)`.

Correction uncertainty is evaluated from the reaction feature-vector difference,
so the same fitted parameters in parent and defect are treated as correlated.
The finite-temperature oxygen term uses the 298 K enthalpy origin consistent with
the experimental M0/M1 calibration.
"""
)

if st.button(
    "Save vacancy correction settings",
    type="primary",
    disabled=bool(invalid_reasons),
):
    cfg.setdefault("vacancies", {})
    cfg["vacancies"]["apply_fitted_energy_correction"] = bool(apply_correction)
    cfg["vacancies"]["allow_legacy_energy_correction_provenance"] = bool(allow_legacy)
    config_path.write_text(toml.dumps(cfg), encoding="utf-8")
    st.success(f"Saved {config_path}")

st.divider()
st.subheader("Run")
st.code("dopingflow corrections-fit -c input.toml\ndopingflow vacancies -c input.toml", language="bash")
st.caption(
    "With resume/skip enabled, existing vacancy screening and relaxations are reused; "
    "the thermodynamic tables are rebuilt from the available vacancy database."
)


def _vacancy_results_root() -> Path:
    if str(vacancies.get("parent_source", "")).strip() == "directory":
        value = str(vacancies.get("parent_directory", "")).strip()
        if value:
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (project_root / path).resolve()
    outdir = str((cfg.get("structure", {}) or {}).get("outdir", "random_structures"))
    return (project_root / outdir).resolve()


st.divider()
st.subheader("Raw vs corrected vacancy free energy")
results_root = _vacancy_results_root()
free_energy_path = results_root / "vacancy_formation_free_energy.csv"

if not free_energy_path.exists():
    st.info(
        "No vacancy_formation_free_energy.csv was found yet. Run the vacancy analysis "
        "after saving the settings above."
    )
else:
    try:
        free_energy = pd.read_csv(free_energy_path, low_memory=False)
    except Exception as exc:
        st.error(f"Could not read {free_energy_path}: {exc}")
    else:
        required = {
            "actual_composition_key",
            "n_vacancies",
            "temperature_K",
            "log10_oxygen_partial_pressure_bar",
        }
        if not required.issubset(free_energy.columns):
            st.warning("The vacancy free-energy file does not contain the expected finite-T columns.")
        else:
            compositions = sorted(
                free_energy["actual_composition_key"].dropna().astype(str).unique()
            )
            composition = st.selectbox(
                "Composition",
                compositions,
                key="vac_corr_plot_composition",
            )
            subset = free_energy[
                free_energy["actual_composition_key"].astype(str) == composition
            ].copy()
            temperatures = sorted(subset["temperature_K"].dropna().astype(float).unique())
            temperature = st.selectbox(
                "Temperature (K)",
                temperatures,
                key="vac_corr_plot_temperature",
            )
            subset = subset[subset["temperature_K"].astype(float) == float(temperature)]
            pressures = sorted(
                subset["log10_oxygen_partial_pressure_bar"].dropna().astype(float).unique()
            )
            pressure = st.selectbox(
                "log10(pO2 / bar)",
                pressures,
                key="vac_corr_plot_pressure",
            )
            selected = subset[
                subset["log10_oxygen_partial_pressure_bar"].astype(float) == float(pressure)
            ].sort_values("n_vacancies")

            figure = go.Figure()
            if "vacancy_formation_free_energy_raw_eV" in selected.columns:
                figure.add_trace(
                    go.Scatter(
                        x=selected["n_vacancies"],
                        y=selected["vacancy_formation_free_energy_raw_eV"],
                        mode="lines+markers",
                        name="Raw",
                    )
                )
            corrected_column = (
                "vacancy_formation_free_energy_corrected_eV"
                if "vacancy_formation_free_energy_corrected_eV" in selected.columns
                else "vacancy_formation_free_energy_eV"
            )
            if corrected_column in selected.columns:
                figure.add_trace(
                    go.Scatter(
                        x=selected["n_vacancies"],
                        y=selected[corrected_column],
                        mode="lines+markers",
                        name="M0/M1 corrected" if "corrected" in corrected_column else "Active",
                    )
                )
            figure.update_layout(
                title=(
                    f"{composition}: vacancy free energy at T={float(temperature):g} K, "
                    f"log10(pO2/bar)={float(pressure):g}"
                ),
                xaxis_title="Number of oxygen vacancies",
                yaxis_title="Vacancy formation free energy (eV)",
                template="plotly_white",
            )
            st.plotly_chart(figure, use_container_width=True)

            table_columns = [
                column
                for column in (
                    "n_vacancies",
                    "vacancy_formation_free_energy_raw_eV",
                    "vacancy_formation_free_energy_corrected_eV",
                    "vacancy_reaction_correction_eV",
                    "vacancy_reaction_correction_uncertainty_eV",
                    "correction_model_family",
                    "correction_fit_id",
                )
                if column in selected.columns
            ]
            if table_columns:
                st.dataframe(
                    selected[table_columns],
                    use_container_width=True,
                    hide_index=True,
                )

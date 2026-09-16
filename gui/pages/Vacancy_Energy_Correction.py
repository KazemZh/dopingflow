from __future__ import annotations

from pathlib import Path

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

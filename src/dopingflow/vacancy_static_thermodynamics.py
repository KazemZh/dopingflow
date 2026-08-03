"""Level-1 static-lattice thermodynamics for oxygen-vacancy screening.

The implementation reuses the established vacancy-analysis core so the legacy
compact outputs and the explicitly named static-lattice outputs remain
numerically identical. No additional public CLI command is exposed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from dopingflow.vacancy_analysis import (
    K_B_EV_PER_K,
    NEGLECTED_SOLID_TERMS,
    NIST_O2_SHOMATE_SOURCE,
    STATIC_LATTICE_APPROXIMATION,
    VacancyAnalysisConfig,
    analyze_vacancies_database,
    analyze_vacancy_thermodynamics,
    exact_stability_intervals,
    inverse_oxygen_pressure_log10,
    nist_o2_standard_state_delta_mu_eV_per_O,
    oxygen_pressure_delta_mu_eV_per_O,
    oxygen_standard_state_delta_mu,
    parse_vacancy_analysis_config,
)

StaticVacancyThermodynamicsConfig = VacancyAnalysisConfig
parse_static_vacancy_thermodynamics_config = parse_vacancy_analysis_config


def analyze_static_vacancy_thermodynamics(
    *,
    rows: Sequence[dict[str, Any]],
    cfg: StaticVacancyThermodynamicsConfig,
    parent_root: Path,
    calculator: Any = None,
    backend: str,
    model: str,
    task: str,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    max_steps: int = 300,
    source_database: Path | None = None,
) -> dict[str, Path]:
    """Create compact static-lattice outputs from detailed vacancy rows."""
    return analyze_vacancy_thermodynamics(
        rows=rows,
        analysis_cfg=cfg,
        parent_root=parent_root,
        backend=backend,
        model=model,
        task=task,
        calculator=calculator,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        source_database=source_database,
    )


def analyze_vacancies_database_static(
    database_path: Path, config_path: Path
) -> dict[str, Path]:
    """Reprocess an existing detailed vacancy database without a new CLI."""
    return analyze_vacancies_database(database_path, config_path)


__all__ = [
    "K_B_EV_PER_K",
    "NEGLECTED_SOLID_TERMS",
    "NIST_O2_SHOMATE_SOURCE",
    "STATIC_LATTICE_APPROXIMATION",
    "StaticVacancyThermodynamicsConfig",
    "analyze_static_vacancy_thermodynamics",
    "analyze_vacancies_database_static",
    "exact_stability_intervals",
    "inverse_oxygen_pressure_log10",
    "nist_o2_standard_state_delta_mu_eV_per_O",
    "oxygen_pressure_delta_mu_eV_per_O",
    "oxygen_standard_state_delta_mu",
    "parse_static_vacancy_thermodynamics_config",
]

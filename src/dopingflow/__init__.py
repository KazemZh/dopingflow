# src/dopingflow/__init__.py
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

# Install the optional vacancy M0/M1 thermodynamic extension before workflow
# modules import the vacancy-analysis entry points. The extension patches the
# parser/analyzer functions only; keep the original base config class bound in
# the source module so the original parser remains a stable construction step.
from dopingflow import vacancy_energy_correction_extensions as _vacancy_energy_correction_extensions  # noqa: E402,F401
from dopingflow import vacancy_static_thermodynamics as _vacancy_static_thermodynamics  # noqa: E402

_base_vacancy_config = (
    _vacancy_energy_correction_extensions.EnergyCorrectedStaticVacancyThermodynamicsConfig.__mro__[1]
)
_vacancy_static_thermodynamics.StaticVacancyThermodynamicsConfig = _base_vacancy_config
_vacancy_static_thermodynamics.VacancyAnalysisConfig = _base_vacancy_config

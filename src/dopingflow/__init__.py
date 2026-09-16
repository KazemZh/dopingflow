# src/dopingflow/__init__.py
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

# Install the optional vacancy M0/M1 thermodynamic extension before workflow
# modules import the vacancy-analysis entry points.
from dopingflow import vacancy_energy_correction_extensions as _vacancy_energy_correction_extensions  # noqa: E402,F401

# Add opt-in explicit vacancy counts and a dedicated Monte Carlo search
# calculator while preserving the established final-calculator workflow.
from dopingflow import vacancy_mc_extensions as _vacancy_mc_extensions  # noqa: E402,F401

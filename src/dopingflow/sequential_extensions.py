"""Sequential-workflow wiring for the extended correction/oxygen stages."""

from __future__ import annotations

from pathlib import Path

from dopingflow import sequential as _base
from dopingflow.correction_workflow import run_corrections_fit
from dopingflow.formation_oxygen_extensions import run_formation

# sequential.py stores imported stage callables as module globals. Replace only
# the two extended stages; all other sequential behavior remains unchanged.
_base.run_corrections_fit = run_corrections_fit
_base.run_formation = run_formation


def run_sequential_from_toml(config_path: Path) -> Path:
    return _base.run_sequential_from_toml(config_path)

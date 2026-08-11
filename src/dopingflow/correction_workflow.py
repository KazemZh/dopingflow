"""Public correction-fit entry point with oxygen-reference compatibility checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from dopingflow.correction_calibration_extensions import (
    run_corrections_fit as _run_corrections_fit,
)
from dopingflow.oxygen_thermodynamics import parse_oxygen_thermodynamics_config

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def run_corrections_fit(
    raw_cfg: Mapping[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> Path | None:
    # This rejects a non-zero empirical O-reference correction when the fitted
    # correction model is enabled, while still allowing a physical delta_mu_O.
    parse_oxygen_thermodynamics_config(raw_cfg)
    return _run_corrections_fit(
        raw_cfg,
        root,
        config_path=config_path,
    )


def run_corrections_fit_from_toml(config_path: Path) -> Path | None:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_corrections_fit(
        raw,
        config_path.resolve().parent,
        config_path=config_path,
    )

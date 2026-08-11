"""Collection entry points with optional legacy relative-energy fallback."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from dopingflow.collect import run_collect as _run_collect
from dopingflow.oxygen_thermodynamics import (
    oxygen_settings_hash,
    parse_oxygen_thermodynamics_config,
)
from dopingflow.relative_energy import populate_relative_energy_columns, relative_energy_enabled

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _with_resolved_oxygen_hash(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Mirror the formation-stage oxygen hash used in correction provenance.

    ``formation_oxygen_extensions`` injects ``_oxygen_settings_hash`` into the
    effective formation configuration before computing each candidate's
    ``formation_input_hash``.  Collect must reconstruct that same effective
    configuration, otherwise a freshly generated corrected formation result is
    incorrectly reported as stale.
    """

    forwarded = copy.deepcopy(raw_cfg)
    settings = parse_oxygen_thermodynamics_config(raw_cfg)
    formation = forwarded.setdefault("formation", {})
    formation["_oxygen_settings_hash"] = oxygen_settings_hash(settings)
    return forwarded


def run_collect(
    raw_cfg: dict[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> Path:
    """Collect results and add legacy relative columns only when requested."""

    forwarded = _with_resolved_oxygen_hash(raw_cfg)
    out_csv = _run_collect(forwarded, root, config_path=config_path)
    if relative_energy_enabled(raw_cfg):
        return populate_relative_energy_columns(out_csv, raw_cfg)
    return out_csv


def run_collect_from_toml(config_path: Path) -> Path:
    raw_cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parent
    return run_collect(raw_cfg, root, config_path=config_path)

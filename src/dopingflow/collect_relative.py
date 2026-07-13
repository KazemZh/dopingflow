"""Collection entry points with optional legacy relative-energy fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dopingflow.collect import run_collect as _run_collect
from dopingflow.relative_energy import populate_relative_energy_columns, relative_energy_enabled

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def run_collect(
    raw_cfg: dict[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> Path:
    """Collect results and add legacy relative columns only when requested."""
    out_csv = _run_collect(raw_cfg, root, config_path=config_path)
    if relative_energy_enabled(raw_cfg):
        return populate_relative_energy_columns(out_csv, raw_cfg)
    return out_csv


def run_collect_from_toml(config_path: Path) -> Path:
    raw_cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parent
    return run_collect(raw_cfg, root, config_path=config_path)

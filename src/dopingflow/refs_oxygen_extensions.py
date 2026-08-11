"""Reference-stage extension for explicit oxygen-energy semantics.

The base reference builder is left responsible for all ML relaxations and raw
energies.  This wrapper resolves the oxygen convention before the run, preserves
legacy numerical behavior where safe, and annotates ``reference_energies.json``
with unambiguous electronic-reference and thermodynamic chemical-potential fields.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from dopingflow import refs as _base
from dopingflow.oxygen_thermodynamics import parse_oxygen_thermodynamics_config

_BASE_RUN_REFS_BUILD = _base.run_refs_build


def run_refs_build(
    raw_cfg: dict[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> Path:
    settings = parse_oxygen_thermodynamics_config(raw_cfg)

    # The historical base writer knows only muO_shift_ev. Feed it the numerical
    # total shift so any legacy consumer remains numerically consistent, then
    # overwrite/augment the JSON with explicit semantics below.
    forwarded = copy.deepcopy(raw_cfg)
    refs = forwarded.setdefault("references", {})
    refs["muO_shift_ev"] = settings.effective_total_shift_eV_per_O

    output = _BASE_RUN_REFS_BUILD(
        forwarded,
        root,
        config_path=config_path,
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    oxide_mode = data.setdefault("oxide_mode", {})
    if not isinstance(oxide_mode, dict):
        oxide_mode = {}
        data["oxide_mode"] = oxide_mode

    oxide_mode.update(
        {
            "oxygen_mode": settings.oxygen_mode,
            "oxygen_reference_correction_ev": (
                settings.oxygen_reference_correction_eV_per_O
            ),
            "delta_mu_O_ev": settings.delta_mu_O_eV_per_O,
            # Retain a derived legacy field for old readers; it is no longer the
            # authoritative semantic representation.
            "muO_shift_ev": settings.effective_total_shift_eV_per_O,
            "oxygen_energy_convention": settings.to_dict(),
        }
    )
    data["oxygen_energy_convention"] = settings.to_dict()
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output


def run_refs_build_from_toml(config_path: Path) -> Path:
    raw = _base._load_raw_toml(config_path)
    return run_refs_build(
        raw,
        config_path.resolve().parent,
        config_path=config_path,
    )


def install_extensions() -> None:
    # Base run_refs_build_from_toml resolves its global ``run_refs_build`` at
    # runtime, so patching this global also protects callers outside the CLI.
    _base.run_refs_build = run_refs_build


install_extensions()

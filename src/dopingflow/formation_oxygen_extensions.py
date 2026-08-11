"""Formation-stage oxygen thermodynamics extension.

This wrapper keeps the existing formation-energy implementation intact while
replacing only the oxygen-reference resolver.  It also invalidates cached
formation outputs whenever the oxygen convention changes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from dopingflow import formation as _base
from dopingflow.oxygen_thermodynamics import (
    OxygenThermodynamicsConfig,
    oxygen_config_from_reference_json,
    oxygen_energy_state,
    oxygen_settings_hash,
    parse_oxygen_thermodynamics_config,
)

_BASE_RUN_FORMATION = _base.run_formation
_ACTIVE_OXYGEN_SETTINGS: OxygenThermodynamicsConfig | None = None
_STATE_FILENAME = "reference_structures/formation_oxygen_state.json"


def _raw_o2_energy(ref: dict[str, Any]) -> float:
    oxide_mode = ref.get("oxide_mode", {}) or {}
    inventory = ref.get("reference_inventory", {}) or {}
    refs = ref.get("references", {}) or {}
    gas_ref = str(
        (oxide_mode.get("gas_ref") if isinstance(oxide_mode, dict) else None)
        or inventory.get("gas_ref")
        or "O2"
    ).strip()
    gas = refs.get(gas_ref)
    if not isinstance(gas, dict):
        raise KeyError(
            f"Missing gas reference {gas_ref!r}; an oxygen-nonstoichiometric "
            "formation reaction requires a same-backend O2 reference"
        )
    if "E_per_molecule_eV" in gas:
        return float(gas["E_per_molecule_eV"])
    if "E_total_eV" in gas:
        return float(gas["E_total_eV"])
    raise KeyError(f"Gas reference {gas_ref} missing E_total_eV or E_per_molecule_eV")


def _oxygen_mu_extended(ref: dict[str, Any]) -> tuple[float, float]:
    """Return physical mu_O and effective O2 energy with separated semantics."""

    settings = _ACTIVE_OXYGEN_SETTINGS
    if settings is None:
        settings = oxygen_config_from_reference_json(ref)
    state = oxygen_energy_state(_raw_o2_energy(ref), settings)
    return state.mu_O_used_eV_per_O, state.E_O2_effective_eV_per_molecule


def _state_path(root: Path) -> Path:
    return root / _STATE_FILENAME


def _previous_hash(root: Path) -> str | None:
    path = _state_path(root)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return str(value.get("oxygen_settings_hash") or "") or None


def _write_state(root: Path, settings: OxygenThermodynamicsConfig) -> None:
    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "oxygen_settings_hash": oxygen_settings_hash(settings),
        "oxygen_energy_convention": settings.to_dict(),
        "note": (
            "Formation outputs must be rebuilt when this hash changes. The empirical "
            "oxygen-reference correction and the physical delta_mu_O are distinct."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_formation(
    raw_cfg: dict[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> None:
    global _ACTIVE_OXYGEN_SETTINGS

    settings = parse_oxygen_thermodynamics_config(raw_cfg)
    current_hash = oxygen_settings_hash(settings)
    changed = _previous_hash(root) != current_hash

    forwarded = copy.deepcopy(raw_cfg)
    formation = forwarded.setdefault("formation", {})
    # Existing formation skip logic predates explicit oxygen thermodynamics.
    # Force one rebuild whenever the convention changes, then persist the hash.
    if changed:
        formation["skip_if_done"] = False
    # Include the convention in correction-sensitive hashes used by the base
    # implementation without adding a public formation option.
    formation["_oxygen_settings_hash"] = current_hash

    _ACTIVE_OXYGEN_SETTINGS = settings
    try:
        _BASE_RUN_FORMATION(
            forwarded,
            root,
            config_path=config_path,
        )
    finally:
        _ACTIVE_OXYGEN_SETTINGS = None

    _write_state(root, settings)


def run_formation_from_toml(config_path: Path) -> None:
    raw = _base.tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_formation(
        raw,
        config_path.resolve().parent,
        config_path=config_path,
    )


def install_extensions() -> None:
    _base._oxygen_mu = _oxygen_mu_extended
    _base.run_formation = run_formation


install_extensions()

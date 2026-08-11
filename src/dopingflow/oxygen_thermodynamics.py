"""Oxygen reference-energy and chemical-potential semantics.

This module separates two quantities that are numerically similar but
scientifically different:

``oxygen_reference_correction_ev``
    An empirical/electronic correction to the raw same-backend oxygen reference,
    expressed per O atom.  This changes the reference energy itself.

``delta_mu_O_ev``
    A physical thermodynamic shift of the oxygen chemical potential relative to
    the O-rich reference, also expressed per O atom.  This represents an
    environment (for example O-poor conditions), not an electronic-structure
    correction.

A fitted ``[energy_correction]`` model is calibrated against raw same-backend
formation energies.  Therefore a non-zero empirical oxygen-reference correction
must not be silently combined with it; doing so would double count an
oxygen-linear correction.  A physical ``delta_mu_O_ev`` remains allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

_TOL = 1.0e-12


@dataclass(frozen=True)
class OxygenThermodynamicsConfig:
    """Resolved oxygen-energy convention, in eV per O atom where applicable."""

    oxygen_mode: str
    oxygen_reference_correction_eV_per_O: float
    delta_mu_O_eV_per_O: float
    energy_correction_enabled: bool
    legacy_muO_shift_ev: float | None = None
    legacy_interpretation: str | None = None
    explicit_new_keys: bool = False

    @property
    def effective_total_shift_eV_per_O(self) -> float:
        """Numerical total shift relative to 1/2 raw O2, for legacy consumers."""

        return (
            float(self.oxygen_reference_correction_eV_per_O)
            + float(self.delta_mu_O_eV_per_O)
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["effective_total_shift_eV_per_O"] = self.effective_total_shift_eV_per_O
        data["schema_version"] = 1
        data["semantics"] = {
            "oxygen_reference_correction_eV_per_O": (
                "empirical/electronic shift of the raw same-backend O reference"
            ),
            "delta_mu_O_eV_per_O": (
                "physical oxygen chemical-potential shift relative to the O-rich reference"
            ),
        }
        return data


@dataclass(frozen=True)
class OxygenEnergyState:
    """Resolved O2 reference and oxygen chemical potential for one raw O2 energy."""

    E_O2_raw_eV_per_molecule: float
    E_O2_reference_eV_per_molecule: float
    mu_O_rich_eV_per_O: float
    delta_mu_O_eV_per_O: float
    mu_O_used_eV_per_O: float
    E_O2_effective_eV_per_molecule: float
    oxygen_reference_correction_eV_per_O: float
    oxygen_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_float(value: Any, *, key: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[references].{key} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"[references].{key} must be a finite number")
    return result


def _validate_resolved(
    *,
    oxygen_mode: str,
    reference_correction: float,
    delta_mu: float,
    energy_correction_enabled: bool,
) -> None:
    if oxygen_mode not in {"O-rich", "O-poor"}:
        raise ValueError("[references].oxygen_mode must be 'O-rich' or 'O-poor'")

    if energy_correction_enabled and abs(reference_correction) > _TOL:
        raise ValueError(
            "A non-zero [references].oxygen_reference_correction_ev cannot be combined "
            "with [energy_correction].enabled=true. The fitted correction model is "
            "calibrated against the raw same-backend oxygen reference, so an additional "
            "empirical O-reference shift would double count an oxygen-linear correction. "
            "Set oxygen_reference_correction_ev=0.0. If the intended shift represents "
            "O-poor thermodynamic conditions, use delta_mu_O_ev instead."
        )

    if oxygen_mode == "O-rich" and abs(delta_mu) > _TOL:
        raise ValueError(
            "[references].oxygen_mode='O-rich' defines delta_mu_O_ev=0.0. "
            "Use oxygen_mode='O-poor' for a negative physical oxygen chemical-potential shift."
        )
    if oxygen_mode == "O-poor" and delta_mu > _TOL:
        raise ValueError(
            "[references].delta_mu_O_ev must be <= 0 for oxygen_mode='O-poor'."
        )


def parse_oxygen_thermodynamics_config(
    raw_cfg: Mapping[str, Any],
) -> OxygenThermodynamicsConfig:
    """Resolve new oxygen settings and the legacy ``muO_shift_ev`` input.

    Migration policy for ``muO_shift_ev``:

    * zero is harmless and accepted for backward compatibility;
    * a non-zero legacy value with energy corrections enabled is rejected as
      ambiguous/double-count-prone;
    * a non-zero legacy value with energy corrections disabled preserves the
      historical interpretation as an empirical oxygen-reference correction;
    * a non-zero legacy value must not be mixed with either new key.
    """

    references = raw_cfg.get("references", {}) or {}
    energy_correction = raw_cfg.get("energy_correction", {}) or {}
    if not isinstance(references, Mapping):
        raise ValueError("[references] must be a TOML table")
    if not isinstance(energy_correction, Mapping):
        raise ValueError("[energy_correction] must be a TOML table")

    oxygen_mode = str(references.get("oxygen_mode", "O-rich")).strip()
    correction_enabled = bool(energy_correction.get("enabled", False))

    has_reference = "oxygen_reference_correction_ev" in references
    has_delta = "delta_mu_O_ev" in references
    has_legacy = "muO_shift_ev" in references
    explicit_new = has_reference or has_delta

    reference_correction = _finite_float(
        references.get("oxygen_reference_correction_ev", 0.0),
        key="oxygen_reference_correction_ev",
    )
    delta_mu = _finite_float(
        references.get("delta_mu_O_ev", 0.0),
        key="delta_mu_O_ev",
    )

    legacy_value: float | None = None
    legacy_interpretation: str | None = None
    if has_legacy:
        legacy_value = _finite_float(references.get("muO_shift_ev", 0.0), key="muO_shift_ev")
        if abs(legacy_value) > _TOL:
            if explicit_new:
                raise ValueError(
                    "Non-zero legacy [references].muO_shift_ev cannot be combined with "
                    "oxygen_reference_correction_ev or delta_mu_O_ev. Replace the legacy "
                    "setting with exactly one scientifically explicit new setting."
                )
            if correction_enabled:
                raise ValueError(
                    "Non-zero legacy [references].muO_shift_ev is ambiguous while "
                    "[energy_correction].enabled=true and may double count the oxygen "
                    "correction. Set muO_shift_ev=0.0/remove it. Use delta_mu_O_ev for a "
                    "physical O-poor chemical-potential shift; do not use an additional "
                    "empirical oxygen-reference correction with the fitted model."
                )
            reference_correction = legacy_value
            delta_mu = 0.0
            legacy_interpretation = "historical_empirical_oxygen_reference_correction"
        else:
            legacy_interpretation = "zero_legacy_alias"

    _validate_resolved(
        oxygen_mode=oxygen_mode,
        reference_correction=reference_correction,
        delta_mu=delta_mu,
        energy_correction_enabled=correction_enabled,
    )

    return OxygenThermodynamicsConfig(
        oxygen_mode=oxygen_mode,
        oxygen_reference_correction_eV_per_O=reference_correction,
        delta_mu_O_eV_per_O=delta_mu,
        energy_correction_enabled=correction_enabled,
        legacy_muO_shift_ev=legacy_value,
        legacy_interpretation=legacy_interpretation,
        explicit_new_keys=explicit_new,
    )


def oxygen_config_from_reference_json(
    reference_data: Mapping[str, Any],
    *,
    energy_correction_enabled: bool = False,
) -> OxygenThermodynamicsConfig:
    """Recover oxygen semantics from ``reference_energies.json``.

    New schema fields take precedence. Older files containing only
    ``muO_shift_ev`` are treated conservatively as a legacy empirical reference
    correction and are rejected when a fitted correction is simultaneously active.
    """

    oxide_mode = reference_data.get("oxide_mode", {}) or {}
    if not isinstance(oxide_mode, Mapping):
        oxide_mode = {}

    if (
        "oxygen_reference_correction_ev" in oxide_mode
        or "delta_mu_O_ev" in oxide_mode
    ):
        raw = {
            "references": {
                "oxygen_mode": oxide_mode.get("oxygen_mode", "O-rich"),
                "oxygen_reference_correction_ev": oxide_mode.get(
                    "oxygen_reference_correction_ev", 0.0
                ),
                "delta_mu_O_ev": oxide_mode.get("delta_mu_O_ev", 0.0),
            },
            "energy_correction": {"enabled": bool(energy_correction_enabled)},
        }
        return parse_oxygen_thermodynamics_config(raw)

    legacy = _finite_float(oxide_mode.get("muO_shift_ev", 0.0), key="muO_shift_ev")
    if energy_correction_enabled and abs(legacy) > _TOL:
        raise ValueError(
            "reference_energies.json contains a non-zero legacy muO_shift_ev while an "
            "energy-correction model is active. Rerun refs-build with explicit oxygen "
            "settings and oxygen_reference_correction_ev=0.0 to avoid double counting."
        )
    return OxygenThermodynamicsConfig(
        oxygen_mode=str(oxide_mode.get("oxygen_mode", "O-rich")),
        oxygen_reference_correction_eV_per_O=legacy,
        delta_mu_O_eV_per_O=0.0,
        energy_correction_enabled=bool(energy_correction_enabled),
        legacy_muO_shift_ev=legacy,
        legacy_interpretation=(
            "historical_empirical_oxygen_reference_correction"
            if abs(legacy) > _TOL
            else "zero_legacy_alias"
        ),
        explicit_new_keys=False,
    )


def oxygen_energy_state(
    E_O2_raw_eV_per_molecule: float,
    config: OxygenThermodynamicsConfig,
) -> OxygenEnergyState:
    """Apply the electronic reference shift first, then the physical ``delta_mu_O``."""

    raw = float(E_O2_raw_eV_per_molecule)
    if not math.isfinite(raw):
        raise ValueError("Raw O2 energy must be finite")
    reference = raw + 2.0 * config.oxygen_reference_correction_eV_per_O
    mu_rich = 0.5 * reference
    mu_used = mu_rich + config.delta_mu_O_eV_per_O
    effective = 2.0 * mu_used
    return OxygenEnergyState(
        E_O2_raw_eV_per_molecule=raw,
        E_O2_reference_eV_per_molecule=reference,
        mu_O_rich_eV_per_O=mu_rich,
        delta_mu_O_eV_per_O=config.delta_mu_O_eV_per_O,
        mu_O_used_eV_per_O=mu_used,
        E_O2_effective_eV_per_molecule=effective,
        oxygen_reference_correction_eV_per_O=(
            config.oxygen_reference_correction_eV_per_O
        ),
        oxygen_mode=config.oxygen_mode,
    )


def oxygen_settings_hash(config: OxygenThermodynamicsConfig) -> str:
    """Stable hash used to invalidate formation outputs when oxygen conditions change."""

    payload = {
        "schema_version": 1,
        "oxygen_mode": config.oxygen_mode,
        "oxygen_reference_correction_eV_per_O": (
            config.oxygen_reference_correction_eV_per_O
        ),
        "delta_mu_O_eV_per_O": config.delta_mu_O_eV_per_O,
        "energy_correction_enabled": config.energy_correction_enabled,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

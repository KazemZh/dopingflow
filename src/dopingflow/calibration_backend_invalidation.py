"""Backend-aware invalidation of phase-resolved calibration manifest caches.

A phase-matched structure/experimental identity can be reused when the active
ML backend/model/task changes.  A precomputed ML total energy or calculated
energy-above-hull cannot: both belong to the backend signature that produced
them.  In phase-resolved calibration mode, stale backend-specific values are
therefore cleared and recomputed instead of causing otherwise valid structures
to be rejected.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from dopingflow import correction_calibration as _base
from dopingflow.corrections import CorrectionConfig, content_hash
from dopingflow.refs import REF_JSON

log = logging.getLogger(__name__)

_BASE_LOAD_CALIBRATION_MANIFEST = _base.load_calibration_manifest


def _active_backend_identity(reference_data: Mapping[str, Any]) -> dict[str, str]:
    """Return the identity used by calibration energy/hull provenance checks."""

    signature: Mapping[str, Any] | None = None
    host = reference_data.get("host", {}) or {}
    if isinstance(host, Mapping) and isinstance(host.get("relaxation_signature"), Mapping):
        signature = host["relaxation_signature"]
    if signature is None:
        for entry in (reference_data.get("references", {}) or {}).values():
            if isinstance(entry, Mapping) and isinstance(
                entry.get("relaxation_signature"), Mapping
            ):
                signature = entry["relaxation_signature"]
                break
    if signature is None:
        raise ValueError(
            "reference_energies.json lacks a relaxation_signature needed to "
            "invalidate backend-specific calibration caches"
        )

    return {
        "backend": str(reference_data.get("backend") or ""),
        "model": str(reference_data.get("model") or ""),
        "task": str(reference_data.get("task") or ""),
        "backend_version": str(
            reference_data.get("backend_package_version")
            or signature.get("backend_package_version")
            or ""
        ),
        "calculation_settings_hash": content_hash(dict(signature)),
    }


def _energy_identity(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        "backend": str(record.get("backend") or ""),
        "model": str(record.get("model") or ""),
        "task": str(record.get("task") or ""),
        "backend_version": str(record.get("backend_version") or ""),
        "calculation_settings_hash": str(
            record.get("calculation_settings_hash") or ""
        ),
    }


def _hull_identity(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        "backend": str(record.get("e_above_hull_backend") or ""),
        "model": str(record.get("e_above_hull_model") or ""),
        "task": str(record.get("e_above_hull_task") or ""),
        "backend_version": str(record.get("e_above_hull_backend_version") or ""),
        "calculation_settings_hash": str(
            record.get("e_above_hull_calculation_settings_hash") or ""
        ),
    }


def sanitize_phase_resolved_manifest_record(
    record: Mapping[str, Any],
    expected_identity: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Clear stale backend-specific values while preserving phase/structure data."""

    cleaned = dict(record)
    invalidated: dict[str, Any] = {}

    if cleaned.get("energy_total_eV") is not None:
        declared = _energy_identity(cleaned)
        if declared != dict(expected_identity):
            invalidated["precomputed_energy"] = {
                "declared": declared,
                "expected": dict(expected_identity),
                "action": "cleared_for_same_backend_recalculation",
            }
            cleaned["energy_total_eV"] = None
            cleaned["backend"] = ""
            cleaned["model"] = ""
            cleaned["task"] = ""
            cleaned["backend_version"] = ""
            cleaned["calculation_settings"] = ""
            cleaned["calculation_settings_hash"] = ""
            cleaned["converged"] = None

    if cleaned.get("e_above_hull_eV_per_atom") is not None:
        declared = _hull_identity(cleaned)
        identity_columns_present = bool(
            cleaned.get("e_above_hull_identity_columns_present")
        )
        if not identity_columns_present or declared != dict(expected_identity):
            invalidated["calculated_e_above_hull"] = {
                "declared": declared,
                "expected": dict(expected_identity),
                "identity_columns_present": identity_columns_present,
                "action": "cleared_for_same_backend_hull_recalculation",
            }
            cleaned["e_above_hull_eV_per_atom"] = None
            cleaned["e_above_hull_provenance"] = ""
            cleaned["e_above_hull_backend"] = ""
            cleaned["e_above_hull_model"] = ""
            cleaned["e_above_hull_task"] = ""
            cleaned["e_above_hull_backend_version"] = ""
            cleaned["e_above_hull_calculation_settings_hash"] = ""
            cleaned["e_above_hull_identity_columns_present"] = False

    # The base fitter defers a missing hull only for records materialized by the
    # automatic phase-resolved path.  A manifest structure is still part of that
    # path, so mark it accordingly even when no stale value had to be cleared.
    cleaned["selection_source"] = "phase_resolved_manifest"

    if invalidated:
        provenance = dict(cleaned.get("expansion_provenance") or {})
        provenance["backend_cache_invalidation"] = invalidated
        cleaned["expansion_provenance"] = provenance

    return cleaned, invalidated


def load_calibration_manifest_backend_aware(
    config: CorrectionConfig,
    root: Path,
) -> list[dict[str, Any]]:
    """Load manifest and invalidate stale backend data in phase-resolved mode."""

    records = _BASE_LOAD_CALIBRATION_MANIFEST(config, root)
    if config.calibration_selection != "phase_resolved" or not records:
        return records

    reference_path = root / REF_JSON
    if not reference_path.is_file():
        # The base fit will emit the canonical missing-reference error later.
        return records
    try:
        reference_data = json.loads(reference_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {reference_path}") from exc

    expected = _active_backend_identity(reference_data)
    output: list[dict[str, Any]] = []
    invalidated_formulas: list[str] = []
    for record in records:
        cleaned, invalidated = sanitize_phase_resolved_manifest_record(record, expected)
        output.append(cleaned)
        if invalidated:
            invalidated_formulas.append(str(cleaned.get("reduced_formula") or cleaned.get("formula")))

    if invalidated_formulas:
        log.info(
            "Invalidated stale backend-specific calibration energy/hull metadata for %d "
            "phase-resolved manifest structure(s) under active %s/%s/%s: %s. "
            "Structures and phase identity are retained; energies/hulls will be "
            "recomputed with the active backend.",
            len(invalidated_formulas),
            expected["backend"],
            expected["model"],
            expected["task"],
            sorted(invalidated_formulas),
        )
    return output


def install_extensions() -> None:
    _base.load_calibration_manifest = load_calibration_manifest_backend_aware


install_extensions()

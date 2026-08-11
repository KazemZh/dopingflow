from __future__ import annotations

from dopingflow.calibration_backend_invalidation import (
    sanitize_phase_resolved_manifest_record,
)


EXPECTED = {
    "backend": "mace",
    "model": "mh-1",
    "task": "matpes_r2scan",
    "backend_version": "0.3.15",
    "calculation_settings_hash": "r2scan-hash",
}


def _record() -> dict:
    return {
        "formula": "CeO2",
        "reduced_formula": "CeO2",
        "structure_path": "/tmp/CeO2.POSCAR",
        "energy_total_eV": -100.0,
        "backend": "mace",
        "model": "mh-1",
        "task": "omat_pbe",
        "backend_version": "0.3.15",
        "calculation_settings": "old",
        "calculation_settings_hash": "pbe-hash",
        "converged": True,
        "e_above_hull_eV_per_atom": 0.01,
        "e_above_hull_provenance": "old backend hull",
        "e_above_hull_backend": "mace",
        "e_above_hull_model": "mh-1",
        "e_above_hull_task": "omat_pbe",
        "e_above_hull_backend_version": "0.3.15",
        "e_above_hull_calculation_settings_hash": "pbe-hash",
        "e_above_hull_identity_columns_present": True,
    }


def test_stale_backend_energy_and_hull_are_cleared_but_structure_is_retained():
    cleaned, invalidated = sanitize_phase_resolved_manifest_record(_record(), EXPECTED)

    assert cleaned["structure_path"] == "/tmp/CeO2.POSCAR"
    assert cleaned["energy_total_eV"] is None
    assert cleaned["e_above_hull_eV_per_atom"] is None
    assert cleaned["backend"] == ""
    assert cleaned["e_above_hull_backend"] == ""
    assert cleaned["selection_source"] == "phase_resolved_manifest"
    assert set(invalidated) == {"precomputed_energy", "calculated_e_above_hull"}


def test_current_backend_values_are_preserved():
    record = _record()
    record.update(
        {
            "task": EXPECTED["task"],
            "calculation_settings_hash": EXPECTED["calculation_settings_hash"],
            "e_above_hull_task": EXPECTED["task"],
            "e_above_hull_calculation_settings_hash": EXPECTED[
                "calculation_settings_hash"
            ],
        }
    )

    cleaned, invalidated = sanitize_phase_resolved_manifest_record(record, EXPECTED)

    assert cleaned["energy_total_eV"] == -100.0
    assert cleaned["e_above_hull_eV_per_atom"] == 0.01
    assert cleaned["selection_source"] == "phase_resolved_manifest"
    assert invalidated == {}


def test_missing_hull_identity_is_recomputed_in_phase_resolved_mode():
    record = _record()
    record["energy_total_eV"] = None
    record["e_above_hull_identity_columns_present"] = False

    cleaned, invalidated = sanitize_phase_resolved_manifest_record(record, EXPECTED)

    assert cleaned["e_above_hull_eV_per_atom"] is None
    assert "calculated_e_above_hull" in invalidated
    assert cleaned["selection_source"].startswith("phase_resolved_")

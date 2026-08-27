from __future__ import annotations

import json
from pathlib import Path

import pytest

from dopingflow.vacancy_configurational_thermodynamics import (
    configurational_partition_thermodynamics,
)
from dopingflow.vacancy_static_thermodynamics import (
    analyze_static_vacancy_thermodynamics,
    parse_static_vacancy_thermodynamics_config,
)


def _row(
    configuration_id: str,
    n_vacancies: int,
    relaxed: float,
    *,
    sp: float | None = None,
    degeneracy: int | None = None,
    enumeration_mode: str = "exact",
    degeneracy_is_exact: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "composition_directory": "Sb20",
        "parent_id": "parent",
        "configuration_id": configuration_id,
        "host_species": "Sn",
        "vacancy_species": "O",
        "dopant_counts_from_parent": {"Sb": 2},
        "dopant_counts_json": {"Sb": 2},
        "n_host": 8,
        "n_total_cations": 10,
        "n_oxygen_sites_parent": 20,
        "n_vacancies": n_vacancies,
        "energy_relaxed_total_eV": relaxed,
        "energy_sp_total_eV": relaxed if sp is None else sp,
        "converged": True,
        "relaxed_poscar_path": f"{configuration_id}/POSCAR",
        "backend": "mace",
        "model": "small",
        "task": "",
    }
    if n_vacancies > 0:
        row.update(
            {
                "enumeration_mode": enumeration_mode,
                "degeneracy": degeneracy,
                "degeneracy_is_exact": degeneracy_is_exact,
            }
        )
    return row


def _minimum_row() -> dict[str, object]:
    return {
        "actual_composition_key": "Sb20",
        "source_parent_id": "parent",
        "dopant_counts_json": {"Sb": 2},
        "n_total_cations": 10,
        "n_oxygen_sites_parent": 20,
        "n_vacancies": 1,
    }


def test_partition_function_uses_exact_orbit_degeneracies():
    rows = [
        _row("v1a", 1, -94.0, degeneracy=2),
        _row("v1b", 1, -93.9, degeneracy=3),
    ]
    result = configurational_partition_thermodynamics(
        rows, _minimum_row(), 600.0
    )
    assert result["free_energy_correction_eV"] < 0.0
    assert result["entropy_eV_per_K"] > 0.0
    assert result["partition_total_degeneracy"] == 5
    assert result["partition_orbit_count"] == 2
    assert result["partition_exact"] is True
    assert result["partition_energy_basis"] == "complete_exact_relaxed_spectrum"


def test_partition_function_uses_full_single_point_spectrum_if_relaxations_incomplete():
    rows = [
        _row("v1a", 1, -94.0, sp=-93.8, degeneracy=2),
        _row("v1b", 1, -93.9, sp=-93.7, degeneracy=3),
    ]
    rows[1]["energy_relaxed_total_eV"] = None
    result = configurational_partition_thermodynamics(
        rows, _minimum_row(), 600.0
    )
    assert result["free_energy_correction_eV"] < 0.0
    assert result["partition_energy_basis"].startswith("complete_exact_single_point")


def test_partition_function_rejects_sampled_degeneracies():
    rows = [
        _row(
            "v1a",
            1,
            -94.0,
            degeneracy=None,
            enumeration_mode="sample",
            degeneracy_is_exact=False,
        )
    ]
    with pytest.raises(ValueError, match="requires exact vacancy enumeration"):
        configurational_partition_thermodynamics(rows, _minimum_row(), 600.0)


def test_configurational_mode_changes_reported_vacancy_formation_free_energy(
    tmp_path: Path,
):
    section = {
        "static_thermodynamic_analysis": True,
        "oxygen_reference_mode": "explicit",
        "mu_O_reference_eV": -2.0,
        "solid_configurational_entropy": "configurational",
        "oxygen_standard_state_mode": "none",
        "temperatures_K": [600.0],
        "standard_oxygen_pressure_bar": 1.0,
        "log10_pO2_min_bar": 0.0,
        "log10_pO2_max_bar": 0.0,
        "log10_pO2_step": 1.0,
        "delta_mu_O_points_eV": [0.0],
    }
    cfg = parse_static_vacancy_thermodynamics_config(section, tmp_path)
    rows = [
        _row("parent_reference", 0, -100.0),
        _row("v1a", 1, -94.0, degeneracy=2),
        _row("v1b", 1, -93.9, degeneracy=3),
    ]
    outputs = analyze_static_vacancy_thermodynamics(
        rows=rows,
        cfg=cfg,
        parent_root=tmp_path,
        backend="mace",
        model="small",
        task="",
    )
    free_rows = json.loads(
        outputs["vacancy_formation_free_energy_json"].read_text(encoding="utf-8")
    )
    one = next(row for row in free_rows if row["n_vacancies"] == 1)
    # Static: (-94)-(-100) + 1*(-2) = 4 eV. The configurational
    # partition function lowers the finite-T free energy.
    assert one["vacancy_formation_energy_static_reference_eV"] == pytest.approx(4.0)
    assert one["solid_configurational_free_energy_correction_eV"] < 0.0
    assert one["vacancy_formation_free_energy_eV"] < 4.0
    assert one["configurational_partition_total_degeneracy"] == 5

    pressure = json.loads(
        outputs["vacancy_static_pressure_map_json"].read_text(encoding="utf-8")
    )
    assert pressure[0]["solid_configurational_entropy_applied"] is True
    metadata = json.loads(outputs["static_metadata"].read_text(encoding="utf-8"))
    assert metadata["solid_configurational_entropy"] == "configurational"
    assert "vacancy_formation_free_energy" in metadata["configurational_entropy_applied_to"]

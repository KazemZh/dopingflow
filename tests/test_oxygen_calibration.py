from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

import dopingflow.oxygen_calibration as oxygen_calibration
from dopingflow.oxygen_calibration import (
    OxygenCalibrationRequest,
    fit_oxygen_reference,
)
from dopingflow.vacancy_static_thermodynamics import (
    analyze_static_vacancy_thermodynamics,
    ideal_vacancy_configurational_entropy_eV_per_K,
    parse_static_vacancy_thermodynamics_config,
)


def _write_structure(path: Path, species: list[str]) -> None:
    lattice = Lattice.cubic(8.0)
    coords = [
        [0.10 + 0.12 * index, 0.10 + 0.09 * index, 0.10 + 0.07 * index]
        for index in range(len(species))
    ]
    Structure(lattice, species, coords).to(fmt="poscar", filename=str(path))


def _experimental_csv(path: Path) -> None:
    rows = [
        # These values give mu_O = -2 eV/O for SnO2 and Sb2O5.
        {
            "formula": "SnO2",
            "formation_enthalpy": -5.0,
            "uncertainty": 0.02,
            "phase": "solid",
            "temperature": "298 K",
            "units": "eV/formula_unit",
            "source": "synthetic-test",
        },
        {
            "formula": "Sb2O5",
            "formation_enthalpy": -12.0,
            "uncertainty": 0.02,
            "phase": "solid",
            "temperature": "298 K",
            "units": "eV/formula_unit",
            "source": "synthetic-test",
        },
        # TiO2 intentionally gives mu_O = -3 eV/O.
        {
            "formula": "TiO2",
            "formation_enthalpy": -5.0,
            "uncertainty": 0.02,
            "phase": "solid",
            "temperature": "298 K",
            "units": "eV/formula_unit",
            "source": "synthetic-test",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reference_file(tmp_path: Path) -> Path:
    refs_dir = tmp_path / "reference_structures"
    relaxed = refs_dir / "relaxed"
    relaxed.mkdir(parents=True)
    sn = relaxed / "SnO2.POSCAR"
    sb = relaxed / "Sb2O5.POSCAR"
    ti = relaxed / "TiO2.POSCAR"
    _write_structure(sn, ["Sn", "O", "O"])
    _write_structure(sb, ["Sb", "Sb", "O", "O", "O", "O", "O"])
    _write_structure(ti, ["Ti", "O", "O"])

    data = {
        "backend": "mace",
        "model": "small",
        "task": "",
        "references": {
            "Sn": {"type": "metal", "E_per_atom_eV": -3.0},
            "Sb": {"type": "metal", "E_per_atom_eV": -4.0},
            "Ti": {"type": "metal", "E_per_atom_eV": -5.0},
            "SnO2": {
                "type": "oxide",
                "E_per_formula_unit_eV": -12.0,
                "relaxed_poscar": str(sn),
            },
            "Sb2O5": {
                "type": "oxide",
                "E_per_formula_unit_eV": -30.0,
                "relaxed_poscar": str(sb),
            },
            "TiO2": {
                "type": "oxide",
                "E_per_formula_unit_eV": -16.0,
                "relaxed_poscar": str(ti),
            },
        },
    }
    path = refs_dir / "reference_energies.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_global_and_chemistry_specific_select_different_reference_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reference_file = _reference_file(tmp_path)
    experimental = tmp_path / "experimental.csv"
    _experimental_csv(experimental)
    # Unit-test the thermochemical fit independently of pymatgen's geometric
    # oxide classifier; production still requires an ordinary binary oxide.
    monkeypatch.setattr(oxygen_calibration, "_normal_binary_oxide", lambda structure: True)

    common = dict(
        reference_file=reference_file,
        experimental_source="custom",
        experimental_data=experimental,
        min_references=2,
        include_host_oxide=False,
    )
    global_result = fit_oxygen_reference(
        OxygenCalibrationRequest(scope="global", target_elements=(), **common),
        backend="mace",
        model="small",
        task="",
    )
    local_result = fit_oxygen_reference(
        OxygenCalibrationRequest(
            scope="chemistry-specific", target_elements=("Sn", "Sb"), **common
        ),
        backend="mace",
        model="small",
        task="",
    )

    assert global_result["n_references"] == 3
    assert global_result["mu_O_reference_eV"] == pytest.approx(-7.0 / 3.0)
    assert local_result["n_references"] == 2
    assert local_result["mu_O_reference_eV"] == pytest.approx(-2.0)
    assert {item["reduced_formula"] for item in local_result["references_used"]} == {
        "SnO2",
        "Sb2O5",
    }


def test_calibration_requires_same_backend_and_enough_real_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reference_file = _reference_file(tmp_path)
    experimental = tmp_path / "experimental.csv"
    _experimental_csv(experimental)
    monkeypatch.setattr(oxygen_calibration, "_normal_binary_oxide", lambda structure: True)

    request = OxygenCalibrationRequest(
        reference_file=reference_file,
        scope="chemistry-specific",
        target_elements=("Sn",),
        experimental_source="custom",
        experimental_data=experimental,
        min_references=2,
        include_host_oxide=False,
    )
    with pytest.raises(ValueError, match="found 1 eligible"):
        fit_oxygen_reference(request, backend="mace", model="small", task="")
    with pytest.raises(ValueError, match="different calculator"):
        fit_oxygen_reference(
            OxygenCalibrationRequest(
                **{**request.__dict__, "min_references": 1}
            ),
            backend="mace",
            model="medium",
            task="",
        )


def test_ideal_vacancy_configurational_entropy_has_expected_limits():
    assert ideal_vacancy_configurational_entropy_eV_per_K(0, 20) == 0.0
    assert ideal_vacancy_configurational_entropy_eV_per_K(20, 20) == 0.0
    value = ideal_vacancy_configurational_entropy_eV_per_K(10, 20)
    expected = 20 * 8.617333262145e-5 * math.log(2.0)
    assert value == pytest.approx(expected)


def _vacancy_row(n_vacancies: int, energy: float) -> dict[str, object]:
    return {
        "composition_directory": "Sb20",
        "parent_id": "parent",
        "configuration_id": "parent_reference" if n_vacancies == 0 else f"v{n_vacancies}",
        "host_species": "Sn",
        "vacancy_species": "O",
        "dopant_counts_from_parent": {"Sb": 2},
        "n_host": 8,
        "n_total_cations": 10,
        "n_oxygen_sites_parent": 20,
        "n_vacancies": n_vacancies,
        "energy_relaxed_total_eV": energy,
        "converged": True,
        "relaxed_poscar_path": f"v{n_vacancies}/POSCAR",
        "backend": "mace",
        "model": "small",
        "task": "",
    }


def test_calibrated_analysis_records_reference_and_optional_entropy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    section = {
        "static_thermodynamic_analysis": True,
        "oxygen_reference_mode": "chemistry-specific",
        "oxygen_reference_file": str(tmp_path / "reference_structures/reference_energies.json"),
        "oxygen_calibration_experimental_source": "custom",
        "oxygen_calibration_experimental_data": str(tmp_path / "experimental.csv"),
        "oxygen_calibration_min_references": 2,
        "solid_configurational_entropy": "ideal",
        "oxygen_standard_state_mode": "none",
        "temperatures_K": [600.0],
        "log10_pO2_min_bar": 0.0,
        "log10_pO2_max_bar": 0.0,
        "log10_pO2_step": 1.0,
        "delta_mu_O_points_eV": [0.0],
    }
    cfg = parse_static_vacancy_thermodynamics_config(section, tmp_path)

    def fake_fit(request, *, backend, model, task):
        assert request.scope == "chemistry-specific"
        assert set(request.target_elements) == {"Sn", "Sb"}
        return {
            "scope": request.scope,
            "target_elements": ["Sb", "Sn"],
            "mu_O_reference_eV": -2.0,
            "n_references": 2,
            "rmse_mu_spread_eV_per_O": 0.05,
            "formation_enthalpy_rmse_eV_per_formula": 0.10,
        }

    monkeypatch.setattr(
        "dopingflow.vacancy_static_thermodynamics.fit_oxygen_reference", fake_fit
    )
    outputs = analyze_static_vacancy_thermodynamics(
        rows=[_vacancy_row(0, -100.0), _vacancy_row(1, -94.0)],
        cfg=cfg,
        parent_root=tmp_path,
        backend="mace",
        model="small",
        task="",
    )
    minima = json.loads(outputs["vacancy_static_minima_json"].read_text(encoding="utf-8"))
    one_vacancy = next(row for row in minima if row["n_vacancies"] == 1)
    assert one_vacancy["mu_O_reference_eV"] == pytest.approx(-2.0)
    assert one_vacancy["grand_potential_intercept_eV"] == pytest.approx(4.0)
    assert one_vacancy["solid_configurational_entropy_eV_per_K"] > 0

    pressure = json.loads(
        outputs["vacancy_static_pressure_map_json"].read_text(encoding="utf-8")
    )
    assert pressure[0]["solid_configurational_entropy_applied"] is True
    metadata = json.loads(outputs["static_metadata"].read_text(encoding="utf-8"))
    assert metadata["oxygen_reference_mode"] == "chemistry-specific"
    assert metadata["solid_configurational_entropy"] == "ideal"

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dopingflow.vacancy_analysis import (
    VacancyAnalysisConfig,
    analyze_vacancy_thermodynamics,
    exact_stability_intervals,
    parse_oxygen_reference_file,
    parse_vacancy_analysis_config,
    resolve_oxygen_reference,
)


def analysis_config(tmp_path: Path, **updates: object) -> VacancyAnalysisConfig:
    section: dict[str, object] = {
        "thermodynamic_analysis": True,
        "oxygen_reference_mode": "explicit",
        "mu_O_reference_eV": -5.0,
        "delta_mu_O_min_eV": -3.0,
        "delta_mu_O_max_eV": 0.0,
        "delta_mu_O_points_eV": [0.0, -1.0, -2.0],
    }
    section.update(updates)
    return parse_vacancy_analysis_config(section, tmp_path)


def row(
    *,
    parent: str,
    configuration: str,
    n_vacancies: int,
    energy: float | None,
    converged: bool = True,
    dopants: dict[str, int] | None = None,
    n_host: int = 8,
    n_oxygen: int = 16,
    single_point: float | None = None,
) -> dict[str, object]:
    dopants = dopants or {"Sb": 2}
    return {
        "composition_directory": "rounded_folder",
        "parent_id": parent,
        "configuration_id": configuration,
        "host_species": "Sn",
        "vacancy_species": "O",
        "dopant_counts_from_parent": dopants,
        "n_host": n_host,
        "n_total_cations": n_host + sum(dopants.values()),
        "n_oxygen_sites_parent": n_oxygen,
        "n_vacancies": n_vacancies,
        "energy_relaxed_total_eV": energy,
        "energy_sp_total_eV": single_point,
        "converged": converged,
        "relaxed_poscar_path": f"{parent}/{configuration}/POSCAR",
        "delta_Q_values": [-2],
        "residual_charge_values": [0],
        "has_fully_compensated_scenario": True,
        "backend": "mace",
        "model": "small",
        "task": "",
    }


def run_analysis(tmp_path: Path, rows: list[dict[str, object]], **updates: object):
    return analyze_vacancy_thermodynamics(
        rows=rows,
        analysis_cfg=analysis_config(tmp_path, **updates),
        parent_root=tmp_path,
        backend="mace",
        model="small",
        task="",
    )


def load_json(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_actual_percentages_and_dynamic_columns_use_integer_counts(tmp_path: Path):
    outputs = run_analysis(
        tmp_path,
        [row(parent="p", configuration="parent_reference", n_vacancies=0, energy=-100), row(parent="p", configuration="v1", n_vacancies=1, energy=-94)],
    )
    minima = load_json(outputs["vacancy_minima_by_composition_json"])
    assert minima[0]["percent_Sb"] == pytest.approx(20.0)
    assert minima[0]["n_Sb"] == 2
    assert minima[0]["total_dopant_percent"] == pytest.approx(20.0)


def test_same_display_percentage_different_integer_compositions_are_not_combined(tmp_path: Path):
    rows = [
        row(parent="p1", configuration="parent_reference", n_vacancies=0, energy=-10, dopants={"Sb": 1}, n_host=9),
        row(parent="p2", configuration="parent_reference", n_vacancies=0, energy=-20, dopants={"Sb": 2}, n_host=18),
    ]
    outputs = run_analysis(tmp_path, rows)
    minima = load_json(outputs["vacancy_minima_by_composition_json"])
    assert len(minima) == 2
    assert len({item["actual_composition_key"] for item in minima}) == 2


def test_multiple_parents_combine_and_lowest_converged_wins(tmp_path: Path):
    rows = [
        row(parent="p1", configuration="parent_reference", n_vacancies=0, energy=-100),
        row(parent="p2", configuration="parent_reference", n_vacancies=0, energy=-101),
        row(parent="p1", configuration="v1", n_vacancies=1, energy=-94),
        row(parent="p2", configuration="v2", n_vacancies=1, energy=-96),
        row(parent="p2", configuration="bad", n_vacancies=1, energy=-99, converged=False),
    ]
    outputs = run_analysis(tmp_path, rows)
    minima = load_json(outputs["vacancy_minima_by_composition_json"])
    parent = next(item for item in minima if item["n_vacancies"] == 0)
    vacancy = next(item for item in minima if item["n_vacancies"] == 1)
    assert parent["source_parent_id"] == "p2"
    assert vacancy["source_configuration_id"] == "v2"
    assert vacancy["delta_energy_to_parent_eV"] == pytest.approx(5.0)
    assert vacancy["grand_potential_intercept_eV"] == pytest.approx(0.0)


def test_relaxed_only_never_falls_back_and_reports_missing_count(tmp_path: Path):
    rows = [
        row(parent="p", configuration="parent_reference", n_vacancies=0, energy=-100),
        row(parent="p", configuration="v1", n_vacancies=1, energy=None, single_point=-110),
    ]
    outputs = run_analysis(tmp_path, rows)
    minima = load_json(outputs["vacancy_minima_by_composition_json"])
    metadata = json.loads(outputs["metadata"].read_text())
    assert [item["n_vacancies"] for item in minima] == [0]
    assert metadata["exclusion_reasons"]["missing_relaxed_energy"] == 1
    assert list(metadata["missing_vacancy_counts_by_composition"].values()) == [[1]]


def test_csv_false_convergence_is_not_treated_as_truthy(tmp_path: Path):
    parent = row(
        parent="p",
        configuration="parent_reference",
        n_vacancies=0,
        energy=-100,
    )
    defective = row(
        parent="p", configuration="v1", n_vacancies=1, energy=-110
    )
    defective["converged"] = "False"
    outputs = run_analysis(tmp_path, [parent, defective])
    minima = load_json(outputs["vacancy_minima_by_composition_json"])
    assert [item["n_vacancies"] for item in minima] == [0]


def test_zero_vacancy_has_no_per_vacancy_division(tmp_path: Path):
    outputs = run_analysis(
        tmp_path,
        [row(parent="p", configuration="parent_reference", n_vacancies=0, energy=-100)],
    )
    item = load_json(outputs["vacancy_minima_by_composition_json"])[0]
    assert item["delta_energy_to_parent_eV"] == 0
    assert item["grand_potential_intercept_eV"] == 0
    assert item["grand_potential_intercept_per_vacancy_eV"] is None


def reference_file(tmp_path: Path, metadata: dict[str, str] | None = None) -> Path:
    gas = {"E_per_molecule_eV": -10.0}
    if metadata:
        gas.update(metadata)
    data = {
        "oxide_mode": {"gas_ref": "O2", "muO_shift_ev": -0.2},
        "references": {"O2": gas},
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_reference_file_shift_and_compatibility(tmp_path: Path):
    parsed = parse_oxygen_reference_file(
        reference_file(tmp_path, {"backend": "mace", "model": "small", "task": ""}),
        backend="mace", model="small", task="", allow_unverified=False,
    )
    assert parsed["oxygen_reference_energy_eV"] == pytest.approx(-10.4)
    assert parsed["mu_O_reference_eV"] == pytest.approx(-5.2)
    assert parsed["oxygen_reference_verified"] is True


def test_incompatible_and_unverifiable_references_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="incompatible"):
        parse_oxygen_reference_file(
            reference_file(tmp_path, {"backend": "uma", "model": "uma-s-1p2", "task": "omat"}),
            backend="mace", model="small", task="", allow_unverified=False,
        )
    with pytest.raises(ValueError, match="cannot be verified"):
        parse_oxygen_reference_file(
            reference_file(tmp_path), backend="mace", model="small", task="", allow_unverified=False,
        )


def test_unverified_override_and_explicit_mode_are_recorded(tmp_path: Path):
    parsed = parse_oxygen_reference_file(
        reference_file(tmp_path), backend="mace", model="small", task="", allow_unverified=True,
    )
    assert parsed["oxygen_reference_verified"] is False
    explicit = resolve_oxygen_reference(
        analysis_config(tmp_path, mu_O_reference_eV=-4.8), backend="mace", model="small", task="",
    )
    assert explicit["mu_O_reference_eV"] == -4.8
    assert explicit["oxygen_reference_verified"] is False


def test_none_mode_writes_minima_without_stability_claim(tmp_path: Path):
    with pytest.warns(RuntimeWarning, match="no cross-count stability"):
        outputs = run_analysis(
            tmp_path,
            [row(parent="p", configuration="parent_reference", n_vacancies=0, energy=-100)],
            oxygen_reference_mode="none",
        )
    assert load_json(outputs["vacancy_minima_by_composition_json"])[0]["mu_O_reference_eV"] is None
    assert load_json(outputs["vacancy_stability_intervals_json"]) == []
    assert load_json(outputs["vacancy_best_counts_json"]) == []


def test_same_calculator_mode_uses_existing_calculator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    structure = tmp_path / "O2.POSCAR"
    structure.write_text("unused", encoding="utf-8")
    cfg = analysis_config(tmp_path, oxygen_reference_mode="same_calculator", oxygen_reference_structure=str(structure))
    monkeypatch.setattr(
        "dopingflow.vacancy_analysis.Structure.from_file",
        lambda _: SimpleNamespace(
            composition=SimpleNamespace(get_el_amt_dict=lambda: {"O": 2}),
            as_dict=lambda: {"species": ["O", "O"]},
        ),
    )
    monkeypatch.setattr("dopingflow.vacancy_analysis.structure_energy_with_calculator", lambda structure, calculator: -9.6)
    result = resolve_oxygen_reference(cfg, backend="mace", model="small", task="", calculator=object())
    assert result["mu_O_reference_eV"] == pytest.approx(-4.8)
    assert result["oxygen_reference_verified"] is True


def test_exact_lower_envelope_crossings_bypass_unstable_count_and_record_ties():
    lines = [
        {"n_vacancies": 0, "grand_potential_intercept_eV": 0.0},
        {"n_vacancies": 1, "grand_potential_intercept_eV": 3.0},
        {"n_vacancies": 2, "grand_potential_intercept_eV": 2.0},
    ]
    intervals = exact_stability_intervals(lines, -2.0, 0.0, 1e-8)
    assert [item["stable_n_vacancies"] for item in intervals] == [2, 0]
    assert intervals[0]["delta_mu_O_upper_eV"] == pytest.approx(-1.0)
    assert intervals[0]["upper_boundary_tied_counts"] == [0, 2]


def test_best_counts_handle_ties_at_requested_points(tmp_path: Path):
    rows = [
        row(parent="p", configuration="parent_reference", n_vacancies=0, energy=-100),
        row(parent="p", configuration="v1", n_vacancies=1, energy=-94),
    ]
    outputs = run_analysis(tmp_path, rows, delta_mu_O_points_eV=[-1.0])
    best = load_json(outputs["vacancy_best_counts_json"])[0]
    # A1 = (-94 + 100) - 5 = 1, so x=-1 is the exact crossing.
    assert best["is_tied"] is True
    assert best["tied_n_vacancies"] == [0, 1]
    assert best["best_n_vacancies"] is None


def test_flat_config_validation_defaults_to_backward_compatible_disabled(tmp_path: Path):
    cfg = parse_vacancy_analysis_config({}, tmp_path)
    assert cfg.enabled is False
    with pytest.raises(ValueError, match="must be <= 0"):
        parse_vacancy_analysis_config({"delta_mu_O_max_eV": 0.1}, tmp_path)


def test_plotting_script_runs_from_compact_csvs(tmp_path: Path):
    outputs = run_analysis(
        tmp_path,
        [
            row(parent="p", configuration="parent_reference", n_vacancies=0, energy=-100),
            row(parent="p", configuration="v1", n_vacancies=1, energy=-94),
        ],
    )
    composition = load_json(outputs["vacancy_minima_by_composition_json"])[0]["actual_composition_key"]
    output_dir = tmp_path / "plots"
    script = Path(__file__).parents[1] / "examples" / "vacancies" / "plot_vacancy_analysis.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--minima",
            str(outputs["vacancy_minima_by_composition_csv"]),
            "--intervals",
            str(outputs["vacancy_stability_intervals_csv"]),
            "--best-counts",
            str(outputs["vacancy_best_counts_csv"]),
            "--composition",
            str(composition),
            "--delta-mu-o",
            "-2.0",
            "--x-dopant",
            "Sb",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert len(list(output_dir.glob("*.png"))) == 4

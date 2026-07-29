from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from typer.testing import CliRunner

from dopingflow.cli import app
from dopingflow.vacancies import (
    charge_scenarios_for_count,
    determine_vacancy_counts,
    enumerate_vacancy_orbits,
    parse_vacancy_config,
    reachable_charge_scenarios,
    vacancy_config_fingerprint,
    run_vacancies,
)


def vacancy_raw(**updates):
    section = {
        "enabled": True,
        "parent_source": "selected_candidates",
        "count_mode": "all_reachable",
        "host_species": "Sn",
        "host_oxidation_state": 4,
        "vacancy_species": "O",
        "vacancy_compensation_charge": 2,
        "oxidation_state_elements": ["Sb", "Nb"],
        "oxidation_state_values": [[3, 5], [5]],
        "backend": "m3gnet",
        "model": "default",
        "device": "cpu",
    }
    section.update(updates)
    return {"structure": {"outdir": "out"}, "vacancies": section}


def test_parse_flat_vacancy_config(tmp_path):
    cfg = parse_vacancy_config(vacancy_raw(), tmp_path)
    assert cfg.oxidation_states == {"Sb": (3, 5), "Nb": (5,)}
    assert cfg.relax_mode == "atoms"
    assert cfg.outdir == tmp_path / "out"


def test_parse_directory_parent_source(tmp_path):
    parent_root = tmp_path / "many_compositions"
    parent_root.mkdir()
    cfg = parse_vacancy_config(
        vacancy_raw(parent_source="directory", parent_directory="many_compositions"),
        tmp_path,
    )
    assert cfg.parent_source == "directory"
    assert cfg.parent_directory == parent_root


def test_directory_parent_source_requires_existing_path(tmp_path):
    with pytest.raises(ValueError, match="parent_directory"):
        parse_vacancy_config(
            vacancy_raw(parent_source="directory", parent_directory="missing"), tmp_path
        )


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"oxidation_state_values": [[3]]}, "equal lengths"),
        ({"oxidation_state_elements": ["Sb", "Sb"]}, "duplicate"),
        ({"enumeration_mode": "bad"}, "enumeration_mode"),
        ({"extra_vacancies": -1}, "extra_vacancies"),
        ({"fmax": 0}, "fmax"),
    ],
)
def test_invalid_config(updates, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        parse_vacancy_config(vacancy_raw(**updates), tmp_path)


def test_four_and_eight_sb_continuous_counts():
    for atoms, expected in ((4, [1, 2]), (8, [1, 2, 3, 4])):
        scenarios = reachable_charge_scenarios({"Sb": atoms}, {"Sb": [3]}, 4)
        counts, metadata = determine_vacancy_counts(
            scenarios,
            compensation_charge=2,
            extra_vacancies=0,
            max_vacancies_cap=8,
            available_sites=20,
        )
        assert counts == expected
        assert metadata["charge_based_max"] == expected[-1]


def test_mixed_states_and_codopant_cancellation():
    mixed = reachable_charge_scenarios({"Sb": 2}, {"Sb": [3, 5]}, 4)
    assert [item["delta_Q"] for item in mixed] == [-2, 0, 2]
    cancelled = reachable_charge_scenarios(
        {"Sb": 4, "Nb": 4}, {"Sb": [3], "Nb": [5]}, 4
    )
    assert [item["delta_Q"] for item in cancelled] == [0]
    counts, _ = determine_vacancy_counts(
        cancelled,
        compensation_charge=2,
        extra_vacancies=0,
        max_vacancies_cap=8,
        available_sites=10,
    )
    assert counts == []


def test_odd_charge_residuals_and_range():
    scenarios = [{"delta_Q": -3, "population_scenarios": []}]
    counts, _ = determine_vacancy_counts(
        scenarios,
        compensation_charge=2,
        extra_vacancies=0,
        max_vacancies_cap=8,
        available_sites=10,
    )
    assert counts == [1, 2]
    assert charge_scenarios_for_count(scenarios, 1, 2)[0]["residual_charge"] == -1
    assert charge_scenarios_for_count(scenarios, 2, 2)[0]["residual_charge"] == 1


def test_extra_cap_and_warning():
    scenarios = [{"delta_Q": -8, "population_scenarios": []}]
    with warnings.catch_warnings(record=True) as caught:
        counts, metadata = determine_vacancy_counts(
            scenarios,
            compensation_charge=2,
            extra_vacancies=3,
            max_vacancies_cap=5,
            available_sites=20,
        )
    assert counts == [1, 2, 3, 4, 5]
    assert metadata["requested_max"] == 7
    assert metadata["truncation_reasons"] == ["max_vacancies_cap"]
    assert caught


def test_missing_present_dopant_is_rejected():
    with pytest.raises(ValueError, match="missing present dopant"):
        reachable_charge_scenarios({"Sb": 2, "Mn": 1}, {"Sb": [3]}, 4)


def small_symmetric_parent():
    return Structure(
        Lattice.cubic(4.0),
        ["Sn", "O", "O"],
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
    )


def test_exact_symmetry_reduction_and_degeneracy(tmp_path):
    parent = small_symmetric_parent()
    cfg = parse_vacancy_config(
        vacancy_raw(
            oxidation_state_elements=["Sb"],
            oxidation_state_values=[[3]],
            enumeration_mode="exact",
        ),
        tmp_path,
    )
    configs, metadata = enumerate_vacancy_orbits(parent, parent, [1, 2], [1, 2], 1, cfg)
    assert len(configs) == 1
    assert configs[0]["degeneracy"] == 2
    assert configs[0]["degeneracy_is_exact"] is True
    assert metadata["enumeration_mode"] == "exact"


def test_sample_is_reproducible_and_not_exact(tmp_path):
    parent = small_symmetric_parent()
    cfg = parse_vacancy_config(
        vacancy_raw(
            oxidation_state_elements=["Sb"],
            oxidation_state_values=[[3]],
            enumeration_mode="sample",
            sample_budget=20,
            sample_patience=5,
        ),
        tmp_path,
    )
    first, _ = enumerate_vacancy_orbits(parent, parent, [1, 2], [1, 2], 1, cfg)
    second, _ = enumerate_vacancy_orbits(parent, parent, [1, 2], [1, 2], 1, cfg)
    assert first == second
    assert all(item["degeneracy"] is None for item in first)


def test_fingerprint_is_stable(tmp_path):
    source = tmp_path / "POSCAR"
    source.write_text("same", encoding="utf-8")
    cfg = parse_vacancy_config(vacancy_raw(), tmp_path)
    assert vacancy_config_fingerprint(cfg, source) == vacancy_config_fingerprint(cfg, source)


def test_only_one_public_vacancy_command():
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "vacancies" in help_result.output
    for forbidden in ("vacancy-generate", "vacancy-scan", "vacancy-relax"):
        assert forbidden not in help_result.output
    vacancy_help = runner.invoke(app, ["vacancies", "--help"])
    assert vacancy_help.exit_code == 0


def test_run_all_dry_run_supports_vacancies(tmp_path):
    config = tmp_path / "input.toml"
    config.write_text("[vacancies]\nenabled=true\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        app, ["run-all", "-c", str(config), "--only", "vacancies", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "vacancies" in result.output


def test_orchestrator_writes_parent_and_fixed_count_results(tmp_path, monkeypatch):
    from pymatgen.io.vasp import Poscar
    import dopingflow.vacancies as module

    parent = small_symmetric_parent()
    candidate = tmp_path / "out" / "pristine" / "candidate_001"
    (candidate / "01_scan").mkdir(parents=True)
    (candidate / "02_relax").mkdir(parents=True)
    Poscar(parent).write_file(candidate / "01_scan" / "POSCAR")
    Poscar(parent).write_file(candidate / "02_relax" / "POSCAR")
    (candidate.parent / "selected_candidates.txt").write_text("candidate_001\n")

    raw = vacancy_raw(
        oxidation_state_elements=[],
        oxidation_state_values=[],
        extra_vacancies=1,
        enumeration_mode="exact",
        topk_per_vacancy_count=1,
    )
    monkeypatch.setattr(module, "check_backend_dependency", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "prepare_backend_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "build_ase_calculator", lambda **kwargs: object())
    monkeypatch.setattr(
        module, "structure_energy_with_calculator", lambda structure, calculator: -float(len(structure))
    )
    monkeypatch.setattr(
        module,
        "relax_structure_with_calculator",
        lambda structure, **kwargs: (structure.copy(), -float(len(structure)) - 0.1, 2, 0.01, True),
    )

    database = run_vacancies(raw, tmp_path)
    assert database.exists()
    rows = module._load_json(tmp_path / "out" / "vacancies_database.json")
    assert sum(row["n_vacancies"] == 0 for row in rows) == 1
    defective = [row for row in rows if row["n_vacancies"] == 1]
    assert len(defective) == 1
    assert defective[0]["selected_for_relaxation"] is True
    assert defective[0]["rank_relaxed_within_vacancy_count"] == 1
    assert (candidate / "05_vacancies" / "V_O_01" / "ranking_scan.csv").exists()

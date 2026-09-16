from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

import dopingflow.vacancy_mc_extensions as ext


def _cfg(**overrides):
    values = {
        "backend": "mace",
        "model": "mh-1",
        "task": "matpes_r2scan",
        "device": "cuda",
        "gpu_id": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parent():
    return Structure(
        Lattice.cubic(6.0),
        ["Sn", "Sb", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.25, 0.25, 0.75],
        ],
    )


def test_explicit_vacancy_counts_are_sorted_and_validated():
    assert ext.parse_explicit_vacancy_counts({}) is None
    assert ext.parse_explicit_vacancy_counts({"vacancy_counts": [2, 1]}) == (1, 2)

    with pytest.raises(ValueError, match="non-empty"):
        ext.parse_explicit_vacancy_counts({"vacancy_counts": []})
    with pytest.raises(ValueError, match="positive integer"):
        ext.parse_explicit_vacancy_counts({"vacancy_counts": [0, 1]})
    with pytest.raises(ValueError, match="duplicates"):
        ext.parse_explicit_vacancy_counts({"vacancy_counts": [1, 1]})


def test_mc_search_calculator_inherits_final_calculator_by_default():
    cfg = _cfg()
    resolved = ext.parse_monte_carlo_search_calculator({}, cfg)

    assert resolved.backend == "mace"
    assert resolved.model == "mh-1"
    assert resolved.task == "matpes_r2scan"
    assert resolved.device == "cuda"
    assert resolved.gpu_id == 0
    assert resolved.explicitly_configured is False


def test_mc_search_calculator_can_use_grace_while_final_is_mace():
    cfg = _cfg()
    resolved = ext.parse_monte_carlo_search_calculator(
        {
            "mc_backend": "grace",
            "mc_model": "GRACE-1L-OMAT",
            "mc_task": "",
            "mc_device": "cuda",
            "mc_gpu_id": 0,
        },
        cfg,
    )

    assert resolved.backend == "grace"
    assert resolved.model == "GRACE-1L-OMAT"
    assert resolved.task == ""
    assert resolved.device == "cuda"
    assert resolved.gpu_id == 0
    assert resolved.explicitly_configured is True


def test_two_cuda_calculators_must_share_gpu_id():
    cfg = _cfg(gpu_id=0)
    with pytest.raises(ValueError, match="same gpu_id"):
        ext.parse_monte_carlo_search_calculator(
            {
                "mc_backend": "grace",
                "mc_model": "GRACE-1L-OMAT",
                "mc_device": "cuda",
                "mc_gpu_id": 1,
            },
            cfg,
        )


def test_explicit_counts_override_charge_derived_range():
    previous = ext._ACTIVE_EXPLICIT_COUNTS
    ext._ACTIVE_EXPLICIT_COUNTS = (1, 2)
    try:
        counts, metadata = ext.determine_vacancy_counts(
            [{"delta_Q": -8}],
            compensation_charge=2,
            extra_vacancies=3,
            max_vacancies_cap=8,
            available_sites=160,
        )
    finally:
        ext._ACTIVE_EXPLICIT_COUNTS = previous

    assert counts == [1, 2]
    assert metadata["mode"] == "explicit"
    assert metadata["charge_based_max"] == 4
    assert metadata["explicit_counts"] == [1, 2]


def test_mc_search_routes_energies_through_active_search_calculator(monkeypatch):
    calls = []
    search_calculator = object()

    def fake_energy(structure, calculator):
        calls.append(calculator)
        return float(sum(site.specie.Z for site in structure))

    def forbidden_energy(_structure):
        raise AssertionError("final-calculator energy function should not be used by MC")

    monkeypatch.setattr(ext, "structure_energy_with_calculator", fake_energy)
    previous = ext._ACTIVE_MC_CALCULATOR
    ext._ACTIVE_MC_CALCULATOR = search_calculator
    try:
        result = ext.monte_carlo_vacancy_search(
            _parent(),
            vacancy_species="O",
            n_vacancies=1,
            energy_function=forbidden_energy,
            max_steps=3,
            patience=3,
            run_mode="fixed",
            seed=3,
            max_candidates=3,
            energy_window_eV=None,
            cation_move_weight=0.5,
            vacancy_move_weight=0.5,
        )
    finally:
        ext._ACTIVE_MC_CALCULATOR = previous

    assert result.candidates
    assert calls
    assert all(calculator is search_calculator for calculator in calls)

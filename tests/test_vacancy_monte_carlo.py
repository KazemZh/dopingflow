from collections import Counter

from pymatgen.core import Lattice, Structure

from dopingflow.vacancy_monte_carlo import (
    annealing_temperature,
    monte_carlo_vacancy_search,
)
from dopingflow.vacancies import periodic_vacancy_distances


def _parent():
    return Structure(
        Lattice.cubic(6.0),
        ["Sn", "Sb", "Nb", "Ta", "O", "O", "O", "O"],
        [
            [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25], [0.25, 0.25, 0.75],
        ],
    )


def test_mc_preserves_arbitrary_cation_composition_and_vacancy_count():
    parent = _parent()

    def energy(structure):
        # Occupation-dependent deterministic toy potential.
        return sum((index + 1) * site.specie.Z for index, site in enumerate(structure)) / 1000

    result = monte_carlo_vacancy_search(
        parent,
        vacancy_species="O",
        n_vacancies=1,
        energy_function=energy,
        max_steps=30,
        patience=30,
        run_mode="fixed",
        seed=7,
        max_candidates=10,
        energy_window_eV=None,
    )
    assert result.candidates
    expected_cations = Counter({"Sn": 1, "Sb": 1, "Nb": 1, "Ta": 1})
    for candidate in result.candidates:
        counts = Counter(site.species_string for site in candidate.structure)
        assert Counter({key: counts[key] for key in expected_cations}) == expected_cations
        assert counts["O"] == 3
        assert len(candidate.vacancy_indices) == 1


def test_mc_is_reproducible_for_a_seed():
    kwargs = dict(
        vacancy_species="O",
        n_vacancies=1,
        energy_function=lambda structure: float(sum(site.specie.Z for site in structure)),
        max_steps=12,
        patience=12,
        seed=11,
        max_candidates=5,
    )
    left = monte_carlo_vacancy_search(_parent(), **kwargs)
    right = monte_carlo_vacancy_search(_parent(), **kwargs)
    assert [item.vacancy_indices for item in left.candidates] == [
        item.vacancy_indices for item in right.candidates
    ]
    assert left.accepted_moves == right.accepted_moves


def test_annealing_temperature_holds_cools_and_reaches_target():
    settings = dict(
        initial_temperature_K=1200.0,
        target_temperature_K=300.0,
        hold_steps=2,
        cooling_steps=3,
    )
    assert annealing_temperature(1, **settings) == 1200.0
    assert annealing_temperature(2, **settings) == 1200.0
    assert annealing_temperature(3, **settings) == 900.0
    assert annealing_temperature(4, **settings) == 600.0
    assert annealing_temperature(5, **settings) == 300.0
    assert annealing_temperature(20, **settings) == 300.0


def test_single_vacancy_has_no_pair_distance():
    minimum, mean, maximum = periodic_vacancy_distances(_parent(), [4])
    assert minimum is None
    assert mean is None
    assert maximum is None

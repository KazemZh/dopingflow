"""Metropolis sampling of vacancy and multi-species cation occupations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from pymatgen.core import Structure

K_B_EV_PER_K = 8.617333262145e-5


def annealing_temperature(
    step: int,
    *,
    initial_temperature_K: float,
    target_temperature_K: float,
    hold_steps: int,
    cooling_steps: int,
) -> float:
    """Return the hold/linear-cooling/target temperature for a 1-based step."""
    if step <= hold_steps:
        return float(initial_temperature_K)
    if cooling_steps > 0 and step <= hold_steps + cooling_steps:
        fraction = (step - hold_steps) / cooling_steps
        return float(
            initial_temperature_K
            + fraction * (target_temperature_K - initial_temperature_K)
        )
    return float(target_temperature_K)


@dataclass(frozen=True)
class MonteCarloCandidate:
    structure: Structure
    energy_eV: float
    step: int
    move: str
    vacancy_indices: tuple[int, ...]


@dataclass(frozen=True)
class MonteCarloResult:
    candidates: tuple[MonteCarloCandidate, ...]
    steps: int
    stop_reason: str
    best_energy_eV: float
    best_step: int
    attempted_moves: dict[str, int]
    accepted_moves: dict[str, int]


def _physical_structure(
    template: Structure, occupations: Sequence[str | None]
) -> Structure:
    result = template.copy()
    for index, species in enumerate(occupations):
        if species is not None and result[index].species_string != species:
            result.replace(index, species)
    result.remove_sites(
        [index for index, species in enumerate(occupations) if species is None]
    )
    return result


def monte_carlo_vacancy_search(
    parent: Structure,
    *,
    vacancy_species: str,
    n_vacancies: int,
    energy_function: Callable[[Structure], float],
    temperature_K: float = 300.0,
    initial_temperature_K: float | None = None,
    annealing_hold_steps: int = 0,
    annealing_steps: int = 0,
    max_steps: int = 10_000,
    patience: int = 2_000,
    run_mode: str = "combined",
    seed: int = 42,
    max_candidates: int = 20,
    energy_window_eV: float | None = 0.5,
    improvement_tolerance_eV: float = 1.0e-5,
    cation_move_weight: float = 0.5,
    vacancy_move_weight: float = 0.5,
) -> MonteCarloResult:
    """Sample fixed-site occupations and return unique low-energy structures.

    Every non-vacancy species is treated generically. Cation swaps are restricted
    to the complement of the vacancy-species sublattice; vacancy markers remain
    internal and are removed before every calculator call.
    """
    if run_mode not in {"fixed", "converged", "combined"}:
        raise ValueError("run_mode must be 'fixed', 'converged', or 'combined'")
    if temperature_K <= 0 or max_steps <= 0 or patience <= 0 or max_candidates <= 0:
        raise ValueError("temperature, step limits, patience, and max_candidates must be > 0")
    initial_temperature = (
        temperature_K if initial_temperature_K is None else initial_temperature_K
    )
    if initial_temperature <= 0:
        raise ValueError("initial_temperature_K must be > 0")
    if initial_temperature < temperature_K:
        raise ValueError("initial_temperature_K must be >= temperature_K")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (annealing_hold_steps, annealing_steps)
    ):
        raise ValueError("annealing hold and ramp steps must be non-negative integers")
    if not 0 <= n_vacancies:
        raise ValueError("n_vacancies must be non-negative")
    if energy_window_eV is not None and energy_window_eV < 0:
        raise ValueError("energy_window_eV must be non-negative or None")
    if min(cation_move_weight, vacancy_move_weight) < 0:
        raise ValueError("Monte Carlo move weights must be non-negative")

    anion_sites = np.array(
        [i for i, site in enumerate(parent) if site.species_string == vacancy_species],
        dtype=int,
    )
    if n_vacancies > len(anion_sites):
        raise ValueError("n_vacancies exceeds the available vacancy-species sites")
    cation_sites = np.array(
        [i for i, site in enumerate(parent) if site.species_string != vacancy_species],
        dtype=int,
    )
    rng = np.random.default_rng(seed)
    occupations: list[str | None] = [site.species_string for site in parent]
    if n_vacancies:
        for index in rng.choice(anion_sites, size=n_vacancies, replace=False):
            occupations[int(index)] = None

    def evaluate() -> tuple[Structure, float]:
        structure = _physical_structure(parent, occupations)
        return structure, float(energy_function(structure))

    structure, current_energy = evaluate()
    best_energy, best_step = current_energy, 0
    archive: dict[tuple[str, ...], MonteCarloCandidate] = {}

    def occupation_key() -> tuple[str, ...]:
        return tuple("X" if value is None else value for value in occupations)

    def archive_current(
        step: int, move: str, candidate_structure: Structure, energy: float
    ) -> None:
        key = occupation_key()
        prior = archive.get(key)
        vacancies = tuple(i for i, value in enumerate(occupations) if value is None)
        if prior is None or energy < prior.energy_eV:
            archive[key] = MonteCarloCandidate(
                candidate_structure.copy(), energy, step, move, vacancies
            )
        ordered = sorted(archive.items(), key=lambda item: item[1].energy_eV)
        if ordered and energy_window_eV is not None:
            cutoff = ordered[0][1].energy_eV + energy_window_eV
            ordered = [item for item in ordered if item[1].energy_eV <= cutoff]
        archive.clear()
        archive.update(ordered[:max_candidates])

    archive_current(0, "initial", structure, current_energy)
    attempted = {"cation_swap": 0, "vacancy_swap": 0}
    accepted = {"cation_swap": 0, "vacancy_swap": 0}
    weights = np.array([cation_move_weight, vacancy_move_weight], dtype=float)
    # Disable impossible moves without making simple chemistries fail.
    if len({occupations[i] for i in cation_sites}) < 2:
        weights[0] = 0.0
    if n_vacancies == 0 or n_vacancies == len(anion_sites):
        weights[1] = 0.0
    if weights.sum() == 0:
        return MonteCarloResult(
            tuple(archive.values()),
            0,
            "no available swaps",
            best_energy,
            0,
            attempted,
            accepted,
        )
    probabilities = weights / weights.sum()
    stalled = 0
    stop_reason = "maximum steps reached"

    for step in range(1, max_steps + 1):
        step_temperature = annealing_temperature(
            step,
            initial_temperature_K=initial_temperature,
            target_temperature_K=temperature_K,
            hold_steps=annealing_hold_steps,
            cooling_steps=annealing_steps,
        )
        beta = 1.0 / (K_B_EV_PER_K * step_temperature)
        move = str(rng.choice(["cation_swap", "vacancy_swap"], p=probabilities))
        attempted[move] += 1
        if move == "vacancy_swap":
            vacant = np.array([i for i in anion_sites if occupations[i] is None])
            occupied = np.array([i for i in anion_sites if occupations[i] is not None])
            left, right = int(rng.choice(vacant)), int(rng.choice(occupied))
        else:
            # Pick two different species, then a random site occupied by each.
            species = sorted({str(occupations[i]) for i in cation_sites})
            species_a, species_b = rng.choice(species, size=2, replace=False)
            left = int(rng.choice([i for i in cation_sites if occupations[i] == species_a]))
            right = int(rng.choice([i for i in cation_sites if occupations[i] == species_b]))
        occupations[left], occupations[right] = occupations[right], occupations[left]
        try:
            trial_structure, trial_energy = evaluate()
        except Exception:
            occupations[left], occupations[right] = occupations[right], occupations[left]
            stalled += 1
            continue
        delta = trial_energy - current_energy
        keep = delta <= 0 or rng.random() < np.exp(-beta * delta)
        if keep:
            accepted[move] += 1
            current_energy = trial_energy
            structure = trial_structure
            archive_current(step, move, structure, current_energy)
        else:
            occupations[left], occupations[right] = occupations[right], occupations[left]
        if current_energy < best_energy - improvement_tolerance_eV:
            best_energy, best_step, stalled = current_energy, step, 0
        else:
            stalled += 1
        if run_mode in {"converged", "combined"} and stalled >= patience:
            stop_reason = "convergence patience reached"
            break
        if run_mode == "converged":
            continue
    else:
        step = max_steps
    candidates = tuple(sorted(archive.values(), key=lambda item: item.energy_eV))
    return MonteCarloResult(
        candidates, step, stop_reason, best_energy, best_step, attempted, accepted
    )

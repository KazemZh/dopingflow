from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from dopingflow.ml_backends import (
    build_ase_calculator,
    check_backend_dependency,
    normalize_backend_config,
    prepare_backend_runtime,
)
from dopingflow.ml_relaxation import (
    get_optimizer_class,
    relax_structure_with_calculator,
    structure_energy_with_calculator,
)
from dopingflow.utils.symmetry import (
    build_sublattice_symmetry_permutations,
    canonical_occupancy_key,
)
from dopingflow.vacancy_analysis import (
    VacancyAnalysisConfig,
    analyze_vacancy_thermodynamics,
    parse_vacancy_analysis_config,
)

log = logging.getLogger(__name__)

_WORKER_CALCULATOR: Any = None

VACANCY_STAGE_DIR = "05_vacancies"
_SCENARIOS_PER_CHARGE = 32
_RELAX_MODES = {"atoms", "full", "isotropic", "volume", "shape", "xy", "cell_only"}
_CELL_FILTERS = {"frechet", "unit", "exp"}
_ENERGY_NORMALIZATIONS = {"total", "per_atom", "per_vacancy"}


def _vacancy_worker_initializer(
    backend: str,
    model: str,
    task: str,
    device: str,
    gpu_id: int,
    tf_threads: int,
    omp_threads: int,
) -> None:
    global _WORKER_CALCULATOR
    prepare_backend_runtime(
        backend=backend,
        device=device,
        gpu_id=gpu_id,
        tf_threads=tf_threads,
        omp_threads=omp_threads,
    )
    _WORKER_CALCULATOR = build_ase_calculator(
        backend=backend, model=model, task=task, device=device
    )


def _single_point_worker(structure_dict: dict[str, Any]) -> float:
    structure = Structure.from_dict(structure_dict)
    return structure_energy_with_calculator(structure, _WORKER_CALCULATOR)


def _relax_worker(job: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    structure_dict, settings = job
    structure = Structure.from_dict(structure_dict)
    start = time.time()
    relaxed, energy, nsteps, final_force, converged = relax_structure_with_calculator(
        structure,
        calculator=_WORKER_CALCULATOR,
        optimizer_name=settings["optimizer"],
        fmax=settings["fmax"],
        max_steps=settings["max_steps"],
        relax_mode=settings["relax_mode"],
        cell_filter=settings["cell_filter"],
    )
    return {
        "structure": relaxed.as_dict(),
        "energy_relaxed_total_eV": energy,
        "nsteps": nsteps,
        "final_fmax": final_force,
        "converged": converged,
        "walltime_s": time.time() - start,
    }


@dataclass(frozen=True)
class VacancyConfig:
    outdir: Path
    enabled: bool
    parent_source: str
    parent_directory: Path | None
    include_parent_reference: bool
    skip_if_done: bool
    resume: bool
    count_mode: str
    host_species: str
    host_oxidation_state: int
    vacancy_species: str
    vacancy_compensation_charge: int
    oxidation_states: dict[str, tuple[int, ...]]
    extra_vacancies: int
    max_vacancies_cap: int
    symprec: float
    angle_tolerance: float
    mapping_tolerance: float
    enumeration_mode: str
    max_exact_raw_configs: int
    max_exact_unique_configs: int
    sample_budget: int
    sample_batch_size: int
    sample_patience: int
    sample_seed: int
    sample_max_saved: int
    minimum_vacancy_distance: float
    backend: str
    model: str
    task: str
    device: str
    gpu_id: int
    n_workers: int
    tf_threads: int
    omp_threads: int
    chunksize: int
    topk_per_vacancy_count: int
    energy_normalization: str
    optimizer: str
    fmax: float
    max_steps: int
    relax_mode: str
    cell_filter: str
    analysis: VacancyAnalysisConfig


def parse_oxidation_state_arrays(section: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Parse the required flat parallel oxidation-state arrays."""
    if "oxidation_state_elements" not in section:
        raise ValueError("[vacancies].oxidation_state_elements is required")
    if "oxidation_state_values" not in section:
        raise ValueError("[vacancies].oxidation_state_values is required")
    elements = section["oxidation_state_elements"]
    values = section["oxidation_state_values"]
    if not isinstance(elements, list) or not isinstance(values, list):
        raise ValueError(
            "[vacancies].oxidation_state_elements and oxidation_state_values must be arrays"
        )
    if len(elements) != len(values):
        raise ValueError(
            "[vacancies].oxidation_state_elements and oxidation_state_values must have equal lengths"
        )
    parsed: dict[str, tuple[int, ...]] = {}
    for index, (element_raw, states_raw) in enumerate(zip(elements, values)):
        element = str(element_raw).strip()
        if not element:
            raise ValueError(f"[vacancies].oxidation_state_elements[{index}] must be non-empty")
        if element in parsed:
            raise ValueError(f"[vacancies].oxidation_state_elements contains duplicate '{element}'")
        if not isinstance(states_raw, list) or not states_raw:
            raise ValueError(
                f"[vacancies].oxidation_state_values[{index}] for {element} must be a non-empty array"
            )
        states: list[int] = []
        for state in states_raw:
            if isinstance(state, bool) or not isinstance(state, int):
                raise ValueError(
                    f"[vacancies].oxidation_state_values for {element} must contain integers"
                )
            states.append(int(state))
        parsed[element] = tuple(sorted(set(states)))
    return parsed


def _positive_int(section: dict[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"[vacancies].{key} must be a positive integer")
    return int(value)


def parse_vacancy_config(raw: dict[str, Any], root: Path) -> VacancyConfig:
    if "vacancies" not in raw:
        raise ValueError("Missing required [vacancies] section")
    section = raw.get("vacancies") or {}
    if not isinstance(section, dict):
        raise ValueError("[vacancies] must be a TOML table")
    enabled = bool(section.get("enabled", True))
    if not enabled:
        raise ValueError("[vacancies].enabled is false; enable it before running vacancies")
    include_parent_reference = bool(section.get("include_parent_reference", True))
    analysis_config = parse_vacancy_analysis_config(section, root)
    if analysis_config.enabled and not include_parent_reference:
        raise ValueError(
            "[vacancies].include_parent_reference must be true when "
            "thermodynamic_analysis=true so E_min(c,0) is available"
        )

    parent_source = str(section.get("parent_source", "selected_candidates")).strip()
    if parent_source not in {"selected_candidates", "directory"}:
        raise ValueError(
            "[vacancies].parent_source must be 'selected_candidates' or 'directory'"
        )
    parent_directory: Path | None = None
    if parent_source == "directory":
        directory_value = str(section.get("parent_directory", "")).strip()
        if not directory_value:
            raise ValueError(
                "[vacancies].parent_directory is required when parent_source='directory'"
            )
        candidate_directory = Path(directory_value).expanduser()
        parent_directory = (
            candidate_directory
            if candidate_directory.is_absolute()
            else root / candidate_directory
        ).resolve()
        if not parent_directory.is_dir():
            raise ValueError(
                f"[vacancies].parent_directory does not exist or is not a directory: "
                f"{parent_directory}"
            )
    host_species = str(section.get("host_species", "")).strip()
    vacancy_species = str(section.get("vacancy_species", "O")).strip()
    if not host_species:
        raise ValueError("[vacancies].host_species is required")
    if not vacancy_species:
        raise ValueError("[vacancies].vacancy_species must be non-empty")
    if host_species == vacancy_species:
        raise ValueError("[vacancies].host_species and vacancy_species must differ")

    host_oxidation_state = section.get("host_oxidation_state", 4)
    vacancy_charge = section.get("vacancy_compensation_charge", 2)
    if isinstance(host_oxidation_state, bool) or not isinstance(host_oxidation_state, int):
        raise ValueError("[vacancies].host_oxidation_state must be an integer")
    if isinstance(vacancy_charge, bool) or not isinstance(vacancy_charge, int) or vacancy_charge <= 0:
        raise ValueError("[vacancies].vacancy_compensation_charge must be a positive integer")

    oxidation_states = parse_oxidation_state_arrays(section)
    if host_species in oxidation_states or vacancy_species in oxidation_states:
        raise ValueError(
            "[vacancies].oxidation_state_elements must contain dopants only, not host_species or vacancy_species"
        )
    count_mode = str(section.get("count_mode", "all_reachable")).strip().lower()
    if count_mode not in {"all_reachable", "nominal"}:
        raise ValueError("[vacancies].count_mode must be 'all_reachable' or 'nominal'")
    extra = section.get("extra_vacancies", 0)
    if isinstance(extra, bool) or not isinstance(extra, int) or extra < 0:
        raise ValueError("[vacancies].extra_vacancies must be a non-negative integer")
    cap = _positive_int(section, "max_vacancies_cap", 8)
    symprec = float(section.get("symprec", 1e-3))
    angle_tolerance = float(section.get("angle_tolerance", 5.0))
    mapping_tolerance = float(section.get("mapping_tolerance", 1.0))
    if symprec <= 0:
        raise ValueError("[vacancies].symprec must be > 0")
    if angle_tolerance <= 0:
        raise ValueError("[vacancies].angle_tolerance must be > 0")
    if mapping_tolerance <= 0:
        raise ValueError("[vacancies].mapping_tolerance must be > 0")
    enumeration_mode = str(section.get("enumeration_mode", "auto")).strip().lower()
    if enumeration_mode not in {"auto", "exact", "sample"}:
        raise ValueError("[vacancies].enumeration_mode must be 'auto', 'exact', or 'sample'")

    minimum_distance = float(section.get("minimum_vacancy_distance", 0.0))
    if minimum_distance < 0:
        raise ValueError("[vacancies].minimum_vacancy_distance must be >= 0")
    device = str(section.get("device", "cpu")).strip().lower()
    gpu_id = section.get("gpu_id", 0)
    if device not in {"cpu", "cuda"}:
        raise ValueError("[vacancies].device must be 'cpu' or 'cuda'")
    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ValueError("[vacancies].gpu_id must be a non-negative integer")

    backend, model, task = normalize_backend_config(
        backend=str(section.get("backend", "m3gnet")),
        model=str(section.get("model", "default")),
        task=str(section.get("task", "")),
        section_name="vacancies",
    )
    normalization = str(section.get("energy_normalization", "per_vacancy")).strip().lower()
    if normalization not in _ENERGY_NORMALIZATIONS:
        raise ValueError(
            "[vacancies].energy_normalization must be 'total', 'per_atom', or 'per_vacancy'"
        )
    optimizer = str(section.get("optimizer", "bfgs")).strip().lower()
    try:
        get_optimizer_class(optimizer)
    except ValueError as exc:
        raise ValueError(f"[vacancies].optimizer: {exc}") from exc
    fmax = float(section.get("fmax", 0.05))
    if fmax <= 0:
        raise ValueError("[vacancies].fmax must be > 0")
    relax_mode = str(section.get("relax_mode", "atoms")).strip().lower()
    if relax_mode not in _RELAX_MODES:
        raise ValueError(f"[vacancies].relax_mode must be one of {sorted(_RELAX_MODES)}")
    cell_filter = str(section.get("cell_filter", "frechet")).strip().lower()
    if cell_filter not in _CELL_FILTERS:
        raise ValueError(f"[vacancies].cell_filter must be one of {sorted(_CELL_FILTERS)}")

    outdir = (root / str((raw.get("structure") or {}).get("outdir", "random_structures"))).resolve()
    sample_seed = section.get("sample_seed", 42)
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
        raise ValueError("[vacancies].sample_seed must be an integer")

    return VacancyConfig(
        outdir=outdir,
        enabled=enabled,
        parent_source=parent_source,
        parent_directory=parent_directory,
        include_parent_reference=include_parent_reference,
        skip_if_done=bool(section.get("skip_if_done", True)),
        resume=bool(section.get("resume", True)),
        count_mode=count_mode,
        host_species=host_species,
        host_oxidation_state=int(host_oxidation_state),
        vacancy_species=vacancy_species,
        vacancy_compensation_charge=int(vacancy_charge),
        oxidation_states=oxidation_states,
        extra_vacancies=int(extra),
        max_vacancies_cap=cap,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        mapping_tolerance=mapping_tolerance,
        enumeration_mode=enumeration_mode,
        max_exact_raw_configs=_positive_int(section, "max_exact_raw_configs", 300_000),
        max_exact_unique_configs=_positive_int(section, "max_exact_unique_configs", 100_000),
        sample_budget=_positive_int(section, "sample_budget", 20_000),
        sample_batch_size=_positive_int(section, "sample_batch_size", 256),
        sample_patience=_positive_int(section, "sample_patience", 4_000),
        sample_seed=int(sample_seed),
        sample_max_saved=_positive_int(section, "sample_max_saved", 50_000),
        minimum_vacancy_distance=minimum_distance,
        backend=backend,
        model=model,
        task=task,
        device=device,
        gpu_id=int(gpu_id),
        n_workers=_positive_int(section, "n_workers", 1),
        tf_threads=_positive_int(section, "tf_threads", 1),
        omp_threads=_positive_int(section, "omp_threads", 1),
        chunksize=_positive_int(section, "chunksize", 25),
        topk_per_vacancy_count=_positive_int(section, "topk_per_vacancy_count", 15),
        energy_normalization=normalization,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=_positive_int(section, "max_steps", 300),
        relax_mode=relax_mode,
        cell_filter=cell_filter,
        analysis=analysis_config,
    )


def _population_options(count: int, states: Sequence[int]) -> Iterable[dict[int, int]]:
    def visit(index: int, remaining: int, current: dict[int, int]):
        if index == len(states) - 1:
            yield {**current, int(states[index]): remaining}
            return
        for amount in range(remaining + 1):
            yield from visit(index + 1, remaining - amount, {**current, int(states[index]): amount})

    yield from visit(0, count, {})


def reachable_charge_scenarios(
    dopant_counts: dict[str, int],
    oxidation_states: dict[str, Sequence[int]],
    host_oxidation_state: int,
    *,
    max_examples_per_charge: int = _SCENARIOS_PER_CHARGE,
) -> list[dict[str, Any]]:
    """Combine compact oxidation-state populations and deduplicate total charges."""
    missing = sorted(set(dopant_counts) - set(oxidation_states))
    if missing:
        raise ValueError(
            "[vacancies].oxidation_state_elements is missing present dopant(s): "
            + ", ".join(missing)
        )
    charge_map: dict[int, list[dict[str, dict[str, int]]]] = {0: [{}]}
    for element in sorted(dopant_counts):
        count = int(dopant_counts[element])
        options = list(_population_options(count, oxidation_states[element]))
        updated: dict[int, list[dict[str, dict[str, int]]]] = {}
        for prior_charge, prior_examples in charge_map.items():
            for population in options:
                contribution = sum(
                    amount * (state - host_oxidation_state)
                    for state, amount in population.items()
                )
                charge = prior_charge + contribution
                bucket = updated.setdefault(charge, [])
                for example in prior_examples:
                    if len(bucket) >= max_examples_per_charge:
                        break
                    bucket.append(
                        {
                            **example,
                            element: {str(state): amount for state, amount in population.items() if amount},
                        }
                    )
        charge_map = updated
    return [
        {"delta_Q": charge, "population_scenarios": charge_map[charge]}
        for charge in sorted(charge_map)
    ]


def determine_vacancy_counts(
    scenarios: Sequence[dict[str, Any]],
    *,
    compensation_charge: int,
    extra_vacancies: int,
    max_vacancies_cap: int,
    available_sites: int,
) -> tuple[list[int], dict[str, Any]]:
    negative = [int(item["delta_Q"]) for item in scenarios if int(item["delta_Q"]) < 0]
    charge_based_max = max(
        (math.ceil(-charge / compensation_charge) for charge in negative), default=0
    )
    requested_max = charge_based_max + extra_vacancies
    applied_max = min(requested_max, max_vacancies_cap, available_sites)
    reasons: list[str] = []
    if applied_max < requested_max and max_vacancies_cap < requested_max:
        reasons.append("max_vacancies_cap")
    if applied_max < requested_max and available_sites < requested_max:
        reasons.append("available_vacancy_species_sites")
    if reasons:
        warnings.warn(
            f"Vacancy maximum truncated from {requested_max} to {applied_max} by "
            + " and ".join(reasons),
            RuntimeWarning,
            stacklevel=2,
        )
    return list(range(1, applied_max + 1)), {
        "charge_based_max": charge_based_max,
        "requested_max": requested_max,
        "applied_max": applied_max,
        "available_vacancy_species_sites": available_sites,
        "truncated": bool(reasons),
        "truncation_reasons": reasons,
    }


def charge_scenarios_for_count(
    scenarios: Sequence[dict[str, Any]], n_vacancies: int, compensation_charge: int
) -> list[dict[str, Any]]:
    output = []
    for scenario in scenarios:
        residual = int(scenario["delta_Q"]) + compensation_charge * n_vacancies
        output.append(
            {
                **scenario,
                "n_vacancies": n_vacancies,
                "residual_charge": residual,
                "fully_compensated": residual == 0,
            }
        )
    return output


def periodic_vacancy_distances(
    structure: Structure, removed_indices: Sequence[int]
) -> tuple[float | None, float | None, float | None]:
    if len(removed_indices) < 2:
        return None, None, None
    distances = [
        float(structure.get_distance(i, j)) for i, j in combinations(removed_indices, 2)
    ]
    return min(distances), float(sum(distances) / len(distances)), max(distances)


class _TooManyUnique(RuntimeError):
    pass


def enumerate_vacancy_orbits(
    symmetry_parent: Structure,
    relaxed_parent: Structure,
    symmetry_vacancy_indices: Sequence[int],
    relaxed_vacancy_indices: Sequence[int],
    n_vacancies: int,
    cfg: VacancyConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate exact or sampled vacancy combinations, reduced by parent symmetry."""
    n_sites = len(symmetry_vacancy_indices)
    raw_count = math.comb(n_sites, n_vacancies)
    permutations = build_sublattice_symmetry_permutations(
        symmetry_parent,
        symmetry_vacancy_indices,
        symprec=cfg.symprec,
        angle_tolerance=cfg.angle_tolerance,
    )
    selected_mode = cfg.enumeration_mode
    reason = f"explicit {selected_mode} mode"
    if selected_mode == "auto":
        selected_mode = "exact" if raw_count <= cfg.max_exact_raw_configs else "sample"
        reason = (
            f"raw combinations {raw_count} <= max_exact_raw_configs"
            if selected_mode == "exact"
            else f"raw combinations {raw_count} > max_exact_raw_configs"
        )

    def record(combination: tuple[int, ...]) -> tuple[bytes, dict[str, Any]] | None:
        relaxed_removed = tuple(relaxed_vacancy_indices[position] for position in combination)
        minimum, mean, maximum = periodic_vacancy_distances(relaxed_parent, relaxed_removed)
        if minimum is not None and minimum < cfg.minimum_vacancy_distance:
            return None
        labels = np.zeros(n_sites, dtype=np.int8)
        labels[list(combination)] = 1
        key = canonical_occupancy_key(labels, permutations)
        return key, {
            "canonical_key": key.hex(),
            "sublattice_positions": list(combination),
            "removed_site_indices": list(relaxed_removed),
            "removed_fractional_coordinates": [
                relaxed_parent[index].frac_coords.tolist() for index in relaxed_removed
            ],
            "removed_cartesian_coordinates": [
                relaxed_parent[index].coords.tolist() for index in relaxed_removed
            ],
            "minimum_vacancy_distance_angstrom": minimum,
            "mean_vacancy_distance_angstrom": mean,
            "maximum_vacancy_distance_angstrom": maximum,
        }

    def exact() -> list[dict[str, Any]]:
        orbits: dict[bytes, dict[str, Any]] = {}
        for combination in combinations(range(n_sites), n_vacancies):
            item = record(combination)
            if item is None:
                continue
            key, data = item
            if key in orbits:
                orbits[key]["degeneracy"] += 1
            else:
                data.update({"degeneracy": 1, "degeneracy_is_exact": True})
                orbits[key] = data
                if len(orbits) > cfg.max_exact_unique_configs:
                    raise _TooManyUnique
        return [orbits[key] for key in sorted(orbits)]

    if selected_mode == "exact":
        try:
            configs = exact()
        except _TooManyUnique:
            if cfg.enumeration_mode == "exact":
                raise RuntimeError(
                    "[vacancies].max_exact_unique_configs exceeded during exact enumeration"
                )
            selected_mode = "sample"
            reason = "exact enumeration exceeded max_exact_unique_configs; restarted in sample mode"

    if selected_mode == "sample":
        rng = random.Random(cfg.sample_seed + n_vacancies)
        seen: dict[bytes, dict[str, Any]] = {}
        attempted = 0
        without_new = 0
        while (
            attempted < cfg.sample_budget
            and without_new < cfg.sample_patience
            and len(seen) < cfg.sample_max_saved
        ):
            batch = min(cfg.sample_batch_size, cfg.sample_budget - attempted)
            for _ in range(batch):
                attempted += 1
                combination = tuple(sorted(rng.sample(range(n_sites), n_vacancies)))
                item = record(combination)
                if item is None:
                    without_new += 1
                    continue
                key, data = item
                if key in seen:
                    without_new += 1
                else:
                    data.update({"degeneracy": None, "degeneracy_is_exact": False})
                    seen[key] = data
                    without_new = 0
                if without_new >= cfg.sample_patience or len(seen) >= cfg.sample_max_saved:
                    break
        configs = [seen[key] for key in sorted(seen)]
    else:
        attempted = raw_count

    for identifier, item in enumerate(configs, start=1):
        item["configuration_id"] = f"config_{identifier:04d}"
        item["enumeration_mode"] = selected_mode
        item["generation_seed"] = cfg.sample_seed if selected_mode == "sample" else None
    return configs, {
        "raw_combination_count": raw_count,
        "unique_configuration_count": len(configs),
        "enumeration_mode": selected_mode,
        "selection_reason": reason,
        "attempted_combinations": attempted,
        "symmetry_permutation_count": len(permutations),
    }


def _jsonable_config(cfg: VacancyConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["outdir"] = str(cfg.outdir)
    data["parent_directory"] = (
        str(cfg.parent_directory) if cfg.parent_directory is not None else None
    )
    data["oxidation_states"] = {key: list(value) for key, value in cfg.oxidation_states.items()}
    data["analysis"]["oxygen_reference_file"] = str(cfg.analysis.oxygen_reference_file)
    data["analysis"]["oxygen_reference_structure"] = str(
        cfg.analysis.oxygen_reference_structure
    )
    return data


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def vacancy_config_fingerprint(cfg: VacancyConfig, *source_paths: Path) -> str:
    generation_config = _jsonable_config(cfg)
    # Analysis settings affect only derived compact outputs and must not force
    # expensive vacancy enumeration/screening/relaxation to rerun.
    generation_config.pop("analysis", None)
    payload = {
        "config": generation_config,
        "sources": {str(path.resolve()): _file_sha256(path) for path in source_paths},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def discover_selected_parents(outdir: Path) -> list[dict[str, Path | str]]:
    if not outdir.exists():
        raise FileNotFoundError(f"[vacancies] structure output directory does not exist: {outdir}")
    found: list[dict[str, Path | str]] = []
    selection_files = sorted(outdir.glob("*/selected_candidates.txt"))
    if not selection_files:
        raise FileNotFoundError(
            f"[vacancies]: no composition-level selected_candidates.txt under parent root {outdir}"
        )
    for selection_file in selection_files:
        composition_dir = selection_file.parent
        names = [line.strip() for line in selection_file.read_text().splitlines() if line.strip()]
        for name in names:
            candidate_dir = composition_dir / name
            scan_path = candidate_dir / "01_scan" / "POSCAR"
            relaxed_path = candidate_dir / "02_relax" / "POSCAR"
            if not scan_path.exists() or not relaxed_path.exists():
                missing = scan_path if not scan_path.exists() else relaxed_path
                raise FileNotFoundError(f"[vacancies] parent {composition_dir.name}/{name}: missing {missing}")
            found.append(
                {
                    "parent_id": f"{composition_dir.name}/{name}",
                    "composition": composition_dir.name,
                    "candidate": name,
                    "candidate_dir": candidate_dir,
                    "symmetry_path": scan_path,
                    "relaxed_path": relaxed_path,
                }
            )
    if not found:
        raise RuntimeError("[vacancies] selected_candidates.txt files contain no candidates")
    return found


def map_parent_sites(
    symmetry_parent: Structure,
    relaxed_parent: Structure,
    *,
    tolerance: float,
    parent_id: str,
) -> dict[int, int]:
    if len(symmetry_parent) != len(relaxed_parent):
        raise ValueError(f"[vacancies] parent {parent_id}: symmetry/relaxed site counts differ")
    if symmetry_parent.composition != relaxed_parent.composition:
        raise ValueError(f"[vacancies] parent {parent_id}: symmetry/relaxed compositions differ")
    length_change = max(
        abs(source - target)
        for source, target in zip(symmetry_parent.lattice.abc, relaxed_parent.lattice.abc)
    )
    angle_change = max(
        abs(source - target)
        for source, target in zip(symmetry_parent.lattice.angles, relaxed_parent.lattice.angles)
    )
    if length_change > tolerance or angle_change > 15.0:
        raise ValueError(
            f"[vacancies] parent {parent_id}: symmetry/relaxed lattices are incompatible "
            f"(maximum length change {length_change:.3f} Å, angle change {angle_change:.3f}°)"
        )
    mapping: dict[int, int] = {}
    unused = set(range(len(relaxed_parent)))
    for source_index, source_site in enumerate(symmetry_parent):
        candidates = [
            target_index
            for target_index in unused
            if relaxed_parent[target_index].species_string == source_site.species_string
        ]
        if not candidates:
            raise ValueError(
                f"[vacancies] parent {parent_id}: cannot map species {source_site.species_string}"
            )
        distances = [
            relaxed_parent.lattice.get_distance_and_image(
                source_site.frac_coords, relaxed_parent[target].frac_coords
            )[0]
            for target in candidates
        ]
        best_position = int(np.argmin(distances))
        if distances[best_position] > tolerance:
            raise ValueError(
                f"[vacancies] parent {parent_id}: unreliable site mapping; nearest "
                f"{source_site.species_string} is {distances[best_position]:.3f} Å "
                f"> [vacancies].mapping_tolerance={tolerance:.3f} Å"
            )
        target_index = candidates[best_position]
        mapping[source_index] = target_index
        unused.remove(target_index)
    return mapping


def _dopant_counts(structure: Structure, cfg: VacancyConfig, parent_id: str) -> dict[str, int]:
    counts = Counter(site.species_string for site in structure)
    if counts[cfg.host_species] == 0:
        raise ValueError(f"[vacancies] parent {parent_id}: host species '{cfg.host_species}' is absent")
    if counts[cfg.vacancy_species] == 0:
        raise ValueError(
            f"[vacancies] parent {parent_id}: vacancy species '{cfg.vacancy_species}' is absent"
        )
    dopants = {
        element: count
        for element, count in counts.items()
        if element not in {cfg.host_species, cfg.vacancy_species}
    }
    missing = sorted(set(dopants) - set(cfg.oxidation_states))
    if missing:
        raise ValueError(
            f"[vacancies] parent {parent_id}: missing oxidation-state data for " + ", ".join(missing)
        )
    if cfg.count_mode == "nominal":
        ambiguous = sorted(element for element in dopants if len(cfg.oxidation_states[element]) != 1)
        if ambiguous:
            raise ValueError(
                f"[vacancies] parent {parent_id}: count_mode='nominal' requires exactly "
                "one oxidation state for present dopant(s): " + ", ".join(ambiguous)
            )
    return dopants


def _defective_structure(parent: Structure, removed_indices: Sequence[int]) -> Structure:
    structure = parent.copy()
    structure.remove_sites(sorted(removed_indices, reverse=True))
    return structure


def _energy_columns(energy: float, n_atoms: int, n_vacancies: int) -> dict[str, float | None]:
    return {
        "energy_sp_total_eV": float(energy),
        "energy_sp_per_atom_eV": float(energy) / n_atoms,
        "energy_sp_per_vacancy_eV": float(energy) / n_vacancies if n_vacancies else None,
    }


def _reported_energy(row: dict[str, Any], normalization: str) -> float | None:
    return row[f"energy_sp_{normalization}_eV"]


def _relaxation_matches(meta: dict[str, Any], cfg: VacancyConfig) -> bool:
    return all(
        [
            meta.get("backend") == cfg.backend,
            meta.get("model") == cfg.model,
            str(meta.get("task", "")) == cfg.task,
            meta.get("optimizer") == cfg.optimizer,
            meta.get("relax_mode", "atoms") == cfg.relax_mode,
            meta.get("cell_filter", "frechet") == cfg.cell_filter,
            float(meta.get("fmax_target_eV_per_A", -1)) == cfg.fmax,
            int(meta.get("max_steps", -1)) == cfg.max_steps,
            meta.get("energy_relaxed_eV") is not None,
        ]
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _parent_reference(
    parent: dict[str, Path | str],
    relaxed_parent: Structure,
    root: Path,
    cfg: VacancyConfig,
    calculator: Any,
    fingerprint: str,
) -> dict[str, Any]:
    ref_dir = root / "parent_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    Poscar(relaxed_parent).write_file(ref_dir / "POSCAR")
    previous_sp = _load_json(ref_dir / "single_point.json")
    if previous_sp.get("configuration_fingerprint") == fingerprint:
        energy_sp = float(previous_sp["energy_sp_total_eV"])
    else:
        energy_sp = structure_energy_with_calculator(relaxed_parent, calculator)
    sp_meta = {
        "configuration_fingerprint": fingerprint,
        "energy_sp_total_eV": energy_sp,
        "backend": cfg.backend,
        "model": cfg.model,
        "task": cfg.task,
        "device": cfg.device,
    }
    _write_json(ref_dir / "single_point.json", sp_meta)
    source_meta_path = Path(parent["candidate_dir"]) / "02_relax" / "meta.json"
    source_meta = _load_json(source_meta_path)
    reused = _relaxation_matches(source_meta, cfg)
    if reused:
        energy_relaxed = float(source_meta["energy_relaxed_eV"])
        relaxed_structure = relaxed_parent
        energy_source = str(source_meta_path)
        converged = bool(source_meta.get("converged", False))
    else:
        prior_consistency = _load_json(ref_dir / "relaxed" / "meta.json")
        prior_poscar = ref_dir / "relaxed" / "POSCAR"
        if (
            prior_consistency.get("configuration_fingerprint") == fingerprint
            and prior_poscar.exists()
        ):
            relaxed_structure = Structure.from_file(prior_poscar)
            energy_relaxed = float(prior_consistency["energy_relaxed_eV"])
            converged = bool(prior_consistency.get("converged", False))
        else:
            relaxed_structure, energy_relaxed, nsteps, final_force, converged = (
                relax_structure_with_calculator(
                    relaxed_parent,
                    calculator=calculator,
                    optimizer_name=cfg.optimizer,
                    fmax=cfg.fmax,
                    max_steps=cfg.max_steps,
                    relax_mode=cfg.relax_mode,
                    cell_filter=cfg.cell_filter,
                )
            )
            relax_meta = {
                "configuration_fingerprint": fingerprint,
                "energy_relaxed_eV": energy_relaxed,
                "optimizer_steps": nsteps,
                "final_fmax_eV_per_A": final_force,
                "converged": converged,
                "backend": cfg.backend,
                "model": cfg.model,
                "task": cfg.task,
                "optimizer": cfg.optimizer,
                "relax_mode": cfg.relax_mode,
                "cell_filter": cfg.cell_filter,
                "fmax_target_eV_per_A": cfg.fmax,
                "max_steps": cfg.max_steps,
            }
            _write_json(ref_dir / "relaxed" / "meta.json", relax_meta)
        energy_source = str(ref_dir / "relaxed" / "meta.json")
    (ref_dir / "relaxed").mkdir(parents=True, exist_ok=True)
    Poscar(relaxed_structure).write_file(ref_dir / "relaxed" / "POSCAR")
    source = {
        "parent_id": parent["parent_id"],
        "parent_geometry_source": str(parent["relaxed_path"]),
        "parent_energy_source": energy_source,
        "parent_energy_sp_eV": energy_sp,
        "parent_energy_relaxed_eV": energy_relaxed,
        "parent_backend": cfg.backend,
        "parent_model": cfg.model,
        "parent_task": cfg.task,
        "parent_relaxation_reused": reused,
        "parent_converged": converged,
    }
    _write_json(ref_dir / "source.json", source)
    return source


def _run_parent(
    parent: dict[str, Path | str], cfg: VacancyConfig, calculator: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_id = str(parent["parent_id"])
    symmetry_parent = Structure.from_file(str(parent["symmetry_path"]))
    relaxed_parent = Structure.from_file(str(parent["relaxed_path"]))
    mapping = map_parent_sites(
        symmetry_parent,
        relaxed_parent,
        tolerance=cfg.mapping_tolerance,
        parent_id=parent_id,
    )
    fingerprint = vacancy_config_fingerprint(
        cfg, Path(parent["symmetry_path"]), Path(parent["relaxed_path"])
    )
    vacancy_root = Path(parent["candidate_dir"]) / VACANCY_STAGE_DIR
    previous = _load_json(vacancy_root / "meta.json")
    if (
        cfg.skip_if_done
        and previous.get("status") == "complete"
        and previous.get("configuration_fingerprint") == fingerprint
        and (vacancy_root / "vacancy_results.json").exists()
        and (vacancy_root / "vacancy_results.csv").exists()
    ):
        log.info("SKIP %s: compatible vacancy workflow is complete", parent_id)
        prior_rows = json.loads((vacancy_root / "vacancy_results.json").read_text())
        dopant_counts = _dopant_counts(relaxed_parent, cfg, parent_id)
        species_counts = Counter(site.species_string for site in relaxed_parent)
        n_host = int(species_counts[cfg.host_species])
        common = {
            "composition_directory": str(parent["composition"]),
            "composition": str(parent["composition"]),
            "candidate": str(parent["candidate"]),
            "host_species": cfg.host_species,
            "dopant_counts_from_parent": dopant_counts,
            "dopant_counts_json": dopant_counts,
            "n_host": n_host,
            "n_total_dopants": sum(dopant_counts.values()),
            "n_total_cations": n_host + sum(dopant_counts.values()),
            "n_oxygen_sites_parent": int(species_counts[cfg.vacancy_species]),
        }
        for row in prior_rows:
            for key, value in common.items():
                row.setdefault(key, value)
            if int(row.get("n_vacancies", -1)) == 0 and "converged" not in row:
                source = _load_json(vacancy_root / "parent_reference" / "source.json")
                row["converged"] = bool(source.get("parent_converged", False))
        return prior_rows, previous
    vacancy_root.mkdir(parents=True, exist_ok=True)

    log.info("[1/7] Parent analysis: %s", parent_id)
    dopant_counts = _dopant_counts(relaxed_parent, cfg, parent_id)
    species_counts = Counter(site.species_string for site in relaxed_parent)
    n_host = int(species_counts[cfg.host_species])
    n_total_dopants = sum(dopant_counts.values())
    n_total_cations = n_host + n_total_dopants
    composition_fields = {
        "composition_directory": str(parent["composition"]),
        "composition": str(parent["composition"]),
        "candidate": str(parent["candidate"]),
        "host_species": cfg.host_species,
        "dopant_counts_from_parent": dopant_counts,
        "dopant_counts_json": dopant_counts,
        "n_host": n_host,
        "n_total_dopants": n_total_dopants,
        "n_total_cations": n_total_cations,
        "n_oxygen_sites_parent": int(species_counts[cfg.vacancy_species]),
    }
    scenarios = reachable_charge_scenarios(
        dopant_counts, cfg.oxidation_states, cfg.host_oxidation_state
    )
    symmetry_vacancy_indices = [
        index
        for index, site in enumerate(symmetry_parent)
        if site.species_string == cfg.vacancy_species
    ]
    relaxed_vacancy_indices = [mapping[index] for index in symmetry_vacancy_indices]
    counts, count_meta = determine_vacancy_counts(
        scenarios,
        compensation_charge=cfg.vacancy_compensation_charge,
        extra_vacancies=cfg.extra_vacancies,
        max_vacancies_cap=cfg.max_vacancies_cap,
        available_sites=len(relaxed_vacancy_indices),
    )
    count_rows = [
        {
            "n_vacancies": count,
            "charge_scenarios": charge_scenarios_for_count(
                scenarios, count, cfg.vacancy_compensation_charge
            ),
        }
        for count in counts
    ]
    _write_json(vacancy_root / "vacancy_counts.json", {**count_meta, "counts": count_rows})
    _write_csv(vacancy_root / "vacancy_counts.csv", count_rows)

    rows: list[dict[str, Any]] = []
    if cfg.include_parent_reference:
        parent_ref = _parent_reference(
            parent, relaxed_parent, vacancy_root, cfg, calculator, fingerprint
        )
        parent_row = {
                "parent_id": parent_id,
                "parent_path": str(parent["candidate_dir"]),
                "configuration_id": "parent_reference",
                "vacancy_species": cfg.vacancy_species,
                "n_vacancies": 0,
                **composition_fields,
                "vacancy_fraction": 0.0,
                "vacancy_percent": 0.0,
                "selected_for_relaxation": False,
                **_energy_columns(parent_ref["parent_energy_sp_eV"], len(relaxed_parent), 0),
                "energy_relaxed_total_eV": parent_ref["parent_energy_relaxed_eV"],
                "converged": parent_ref["parent_converged"],
                "backend": cfg.backend,
                "model": cfg.model,
                "task": cfg.task,
                "generated_poscar_path": str(vacancy_root / "parent_reference" / "POSCAR"),
                "relaxed_poscar_path": str(
                    vacancy_root / "parent_reference" / "relaxed" / "POSCAR"
                ),
                **parent_ref,
            }
        parent_row["energy_sp_reported_eV"] = _reported_energy(
            parent_row, cfg.energy_normalization
        )
        rows.append(parent_row)

    log.info("[2/7] Vacancy enumeration: %s counts=%s", parent_id, counts)
    for n_vacancies in counts:
        group_dir = vacancy_root / f"V_{cfg.vacancy_species}_{n_vacancies:02d}"
        configs, enumeration_meta = enumerate_vacancy_orbits(
            symmetry_parent,
            relaxed_parent,
            symmetry_vacancy_indices,
            relaxed_vacancy_indices,
            n_vacancies,
            cfg,
        )
        log.info(
            "%s n=%d: %s unique=%d (%s)",
            parent_id,
            n_vacancies,
            enumeration_meta["enumeration_mode"],
            len(configs),
            enumeration_meta["selection_reason"],
        )
        charge_rows = charge_scenarios_for_count(
            scenarios, n_vacancies, cfg.vacancy_compensation_charge
        )
        for item in configs:
            config_dir = group_dir / item["configuration_id"]
            generate_dir = config_dir / "00_generate"
            defective = _defective_structure(relaxed_parent, item["removed_site_indices"])
            generate_dir.mkdir(parents=True, exist_ok=True)
            Poscar(defective).write_file(generate_dir / "POSCAR")
            generation_meta = {
                **item,
                "parent_id": parent_id,
                "vacancy_species": cfg.vacancy_species,
                "n_vacancies": n_vacancies,
                "vacancy_species_sites_in_parent": len(relaxed_vacancy_indices),
                "vacancy_fraction": n_vacancies / len(relaxed_vacancy_indices),
                "vacancy_percent": 100.0 * n_vacancies / len(relaxed_vacancy_indices),
                "symprec": cfg.symprec,
                "angle_tolerance": cfg.angle_tolerance,
                "charge_scenarios": charge_rows,
                "source_parent_path": str(parent["relaxed_path"]),
                "output_path": str(config_dir),
                "configuration_fingerprint": fingerprint,
                **enumeration_meta,
            }
            _write_json(generate_dir / "meta.json", generation_meta)
            rows.append(
                {
                    "parent_id": parent_id,
                    "parent_path": str(parent["candidate_dir"]),
                    "configuration_id": item["configuration_id"],
                    "vacancy_species": cfg.vacancy_species,
                    "n_vacancies": n_vacancies,
                    **composition_fields,
                    "vacancy_fraction": generation_meta["vacancy_fraction"],
                    "vacancy_percent": generation_meta["vacancy_percent"],
                    "enumeration_mode": item["enumeration_mode"],
                    "degeneracy": item["degeneracy"],
                    "degeneracy_is_exact": item["degeneracy_is_exact"],
                    "minimum_vacancy_distance_angstrom": item[
                        "minimum_vacancy_distance_angstrom"
                    ],
                    "delta_Q_values": [scenario["delta_Q"] for scenario in charge_rows],
                    "residual_charge_values": [
                        scenario["residual_charge"] for scenario in charge_rows
                    ],
                    "has_fully_compensated_scenario": any(
                        scenario["fully_compensated"] for scenario in charge_rows
                    ),
                    "backend": cfg.backend,
                    "model": cfg.model,
                    "task": cfg.task,
                    "generated_poscar_path": str(generate_dir / "POSCAR"),
                    "relaxed_poscar_path": "",
                    "configuration_fingerprint": fingerprint,
                }
            )

    log.info("[3/7] Single-point screening: %s", parent_id)
    defective_rows = [row for row in rows if row["n_vacancies"] > 0]
    pending_sp: list[tuple[dict[str, Any], Structure, Path]] = []
    for row in defective_rows:
        scan_dir = Path(row["generated_poscar_path"]).parent.parent / "01_scan"
        old_meta = _load_json(scan_dir / "meta.json")
        if (
            cfg.resume
            and old_meta.get("configuration_fingerprint") == fingerprint
            and old_meta.get("energy_sp_total_eV") is not None
        ):
            energy = float(old_meta["energy_sp_total_eV"])
            row.update(
                _energy_columns(
                    energy,
                    len(relaxed_parent) - row["n_vacancies"],
                    row["n_vacancies"],
                )
            )
            row["energy_sp_reported_eV"] = _reported_energy(row, cfg.energy_normalization)
        else:
            structure = Structure.from_file(row["generated_poscar_path"])
            pending_sp.append((row, structure, scan_dir))

    if pending_sp and cfg.n_workers > 1 and cfg.device == "cpu":
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=cfg.n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_vacancy_worker_initializer,
            initargs=(
                cfg.backend,
                cfg.model,
                cfg.task,
                cfg.device,
                cfg.gpu_id,
                cfg.tf_threads,
                cfg.omp_threads,
            ),
        ) as executor:
            energies = executor.map(
                _single_point_worker,
                [structure.as_dict() for _, structure, _ in pending_sp],
                chunksize=cfg.chunksize,
            )
            evaluated_sp = zip(pending_sp, energies)
            evaluated_sp = list(evaluated_sp)
    else:
        evaluated_sp = [
            ((row, structure, scan_dir), structure_energy_with_calculator(structure, calculator))
            for row, structure, scan_dir in pending_sp
        ]

    for (row, structure, scan_dir), energy in evaluated_sp:
        scan_meta = {
                "configuration_fingerprint": fingerprint,
                **_energy_columns(energy, len(structure), row["n_vacancies"]),
                "backend": cfg.backend,
                "model": cfg.model,
                "task": cfg.task,
                "device": cfg.device,
            }
        _write_json(scan_dir / "meta.json", scan_meta)
        row.update(
            _energy_columns(
                energy, len(relaxed_parent) - row["n_vacancies"], row["n_vacancies"]
            )
        )
        row["energy_sp_reported_eV"] = _reported_energy(row, cfg.energy_normalization)

    log.info("[4/7] Top-k selection: %s", parent_id)
    for n_vacancies in counts:
        group = [row for row in defective_rows if row["n_vacancies"] == n_vacancies]
        group.sort(key=lambda row: (row["energy_sp_total_eV"], row["configuration_id"]))
        if group:
            minimum = group[0]["energy_sp_total_eV"]
        for rank, row in enumerate(group, start=1):
            relative = row["energy_sp_total_eV"] - minimum
            row.update(
                {
                    "relative_energy_same_count_eV": relative,
                    "relative_energy_same_count_meV": 1000.0 * relative,
                    "rank_within_vacancy_count": rank,
                    "selected_for_relaxation": rank <= cfg.topk_per_vacancy_count,
                }
            )
        group_dir = vacancy_root / f"V_{cfg.vacancy_species}_{n_vacancies:02d}"
        _write_csv(group_dir / "ranking_scan.csv", group)
        selected = [row["configuration_id"] for row in group if row["selected_for_relaxation"]]
        (group_dir / "selected_candidates.txt").write_text(
            "\n".join(selected) + ("\n" if selected else ""), encoding="utf-8"
        )

    log.info("[5/7] Relaxation: %s", parent_id)
    pending_relax: list[tuple[dict[str, Any], Structure, Path]] = []
    for row in defective_rows:
        if not row["selected_for_relaxation"]:
            continue
        config_dir = Path(row["generated_poscar_path"]).parent.parent
        relax_dir = config_dir / "02_relax"
        old_meta = _load_json(relax_dir / "meta.json")
        if (
            cfg.resume
            and old_meta.get("configuration_fingerprint") == fingerprint
            and (relax_dir / "POSCAR").exists()
        ):
            row.update(old_meta)
            row["relaxed_poscar_path"] = str(relax_dir / "POSCAR")
        else:
            structure = Structure.from_file(row["generated_poscar_path"])
            pending_relax.append((row, structure, relax_dir))

    relax_settings = {
        "optimizer": cfg.optimizer,
        "fmax": cfg.fmax,
        "max_steps": cfg.max_steps,
        "relax_mode": cfg.relax_mode,
        "cell_filter": cfg.cell_filter,
    }
    if pending_relax and cfg.n_workers > 1 and cfg.device == "cpu":
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=cfg.n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_vacancy_worker_initializer,
            initargs=(
                cfg.backend,
                cfg.model,
                cfg.task,
                cfg.device,
                cfg.gpu_id,
                cfg.tf_threads,
                cfg.omp_threads,
            ),
        ) as executor:
            outputs = executor.map(
                _relax_worker,
                [(structure.as_dict(), relax_settings) for _, structure, _ in pending_relax],
                chunksize=cfg.chunksize,
            )
            evaluated_relax = list(zip(pending_relax, outputs))
    else:
        evaluated_relax = []
        for pending, structure, relax_dir in pending_relax:
            start = time.time()
            relaxed, energy, nsteps, final_force, converged = relax_structure_with_calculator(
                structure,
                calculator=calculator,
                optimizer_name=cfg.optimizer,
                fmax=cfg.fmax,
                max_steps=cfg.max_steps,
                relax_mode=cfg.relax_mode,
                cell_filter=cfg.cell_filter,
            )
            evaluated_relax.append(
                (
                    (pending, structure, relax_dir),
                    {
                        "structure": relaxed.as_dict(),
                        "energy_relaxed_total_eV": energy,
                        "nsteps": nsteps,
                        "final_fmax": final_force,
                        "converged": converged,
                        "walltime_s": time.time() - start,
                    },
                )
            )

    for (row, structure, relax_dir), output in evaluated_relax:
        relaxed = Structure.from_dict(output.pop("structure"))
        relax_dir.mkdir(parents=True, exist_ok=True)
        Poscar(relaxed).write_file(relax_dir / "POSCAR")
        energy = output["energy_relaxed_total_eV"]
        relax_meta = {
                "configuration_fingerprint": fingerprint,
                "energy_initial_eV": row["energy_sp_total_eV"],
                "energy_relaxed_total_eV": energy,
                "energy_change_eV": energy - row["energy_sp_total_eV"],
                "optimizer": cfg.optimizer,
                "fmax_target": cfg.fmax,
                "max_steps": cfg.max_steps,
                "relax_mode": cfg.relax_mode,
                "cell_filter": cfg.cell_filter,
                "backend": cfg.backend,
                "model": cfg.model,
                "task": cfg.task,
                "device": cfg.device,
                **output,
            }
        _write_json(relax_dir / "meta.json", relax_meta)
        row.update(relax_meta)
        row["relaxed_poscar_path"] = str(relax_dir / "POSCAR")

    for n_vacancies in counts:
        relaxed_group = [
            row
            for row in defective_rows
            if row["n_vacancies"] == n_vacancies and row.get("energy_relaxed_total_eV") is not None
        ]
        relaxed_group.sort(
            key=lambda row: (row["energy_relaxed_total_eV"], row["configuration_id"])
        )
        if relaxed_group:
            minimum = relaxed_group[0]["energy_relaxed_total_eV"]
        for rank, row in enumerate(relaxed_group, start=1):
            relative = row["energy_relaxed_total_eV"] - minimum
            row.update(
                {
                    "relative_relaxed_energy_same_count_eV": relative,
                    "relative_relaxed_energy_same_count_meV": relative * 1000.0,
                    "rank_relaxed_within_vacancy_count": rank,
                }
            )
        _write_csv(
            vacancy_root / f"V_{cfg.vacancy_species}_{n_vacancies:02d}" / "ranking_relax.csv",
            relaxed_group,
        )

    log.info("Writing per-parent vacancy results: %s", parent_id)
    _write_json(vacancy_root / "vacancy_results.json", rows)
    _write_csv(vacancy_root / "vacancy_results.csv", rows)
    metadata = {
        "status": "complete",
        "parent_id": parent_id,
        "configuration_fingerprint": fingerprint,
        "resolved_config": _jsonable_config(cfg),
        "dopant_counts_from_parent": dopant_counts,
        "n_host": n_host,
        "n_total_cations": n_total_cations,
        "reachable_charge_scenarios": scenarios,
        "vacancy_range": count_meta,
        "n_result_rows": len(rows),
    }
    _write_json(vacancy_root / "meta.json", metadata)
    return rows, metadata


def run_vacancies(raw: dict[str, Any], root: Path, *, config_path: Path | None = None) -> Path:
    cfg = parse_vacancy_config(raw, root)
    check_backend_dependency(cfg.backend, stage_name="Vacancies")
    if cfg.device == "cuda" and cfg.n_workers != 1:
        log.warning("[vacancies].device='cuda' forces one worker for safe model reuse")
    prepare_backend_runtime(
        backend=cfg.backend,
        device=cfg.device,
        gpu_id=cfg.gpu_id,
        tf_threads=cfg.tf_threads,
        omp_threads=cfg.omp_threads,
    )
    calculator = build_ase_calculator(
        backend=cfg.backend, model=cfg.model, task=cfg.task, device=cfg.device
    )
    parent_root = cfg.parent_directory if cfg.parent_source == "directory" else cfg.outdir
    assert parent_root is not None
    parents = discover_selected_parents(parent_root)
    all_rows: list[dict[str, Any]] = []
    for index, parent in enumerate(parents, start=1):
        log.info("Vacancy parent %d/%d: %s", index, len(parents), parent["parent_id"])
        rows, _ = _run_parent(parent, cfg, calculator)
        all_rows.extend(rows)
    csv_path = parent_root / "vacancies_database.csv"
    _write_csv(csv_path, all_rows)
    _write_json(parent_root / "vacancies_database.json", all_rows)
    if cfg.analysis.enabled:
        log.info("[6/7] Vacancy thermodynamic analysis")
        analyze_vacancy_thermodynamics(
            rows=all_rows,
            analysis_cfg=cfg.analysis,
            parent_root=parent_root,
            backend=cfg.backend,
            model=cfg.model,
            task=cfg.task,
            calculator=calculator,
            optimizer=cfg.optimizer,
            fmax=cfg.fmax,
            max_steps=cfg.max_steps,
            source_database=csv_path,
        )
    log.info("[7/7] Summary writing complete: %s", csv_path)
    return csv_path


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def run_vacancies_from_toml(config_path: Path) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_vacancies(raw, config_path.resolve().parent, config_path=config_path)

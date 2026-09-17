"""Two-stage Monte Carlo vacancy workflow for incompatible ML environments.

The search stage runs only the Monte Carlo calculator (for example GRACE) and
writes a low-energy archive plus selected candidate structures. The finalize
stage runs only the ordinary vacancy calculator (for example MACE), evaluates
the selected structures consistently, relaxes/reranks them, and performs the
existing thermodynamic analysis.

The two dependency checks live in separate entry points so GRACE and MACE do
not need to be installed in the same Python environment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from dopingflow import vacancies as _base
from dopingflow.ml_backends import (
    build_ase_calculator,
    check_backend_dependency,
    prepare_backend_runtime,
)
from dopingflow.ml_relaxation import (
    relax_structure_with_calculator,
    structure_energy_with_calculator,
)
from dopingflow.vacancy_mc_extensions import (
    MonteCarloSearchCalculatorConfig,
    parse_explicit_vacancy_counts,
    parse_monte_carlo_search_calculator,
)

log = logging.getLogger(__name__)


def _sha256_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _search_fingerprint(
    cfg: _base.VacancyConfig,
    mc_cfg: MonteCarloSearchCalculatorConfig,
    explicit_counts: tuple[int, ...] | None,
    *source_paths: Path,
) -> str:
    """Fingerprint only search-space and MC-search settings, not final relaxation."""
    data = _base._jsonable_config(cfg)
    data.pop("analysis", None)
    for key in (
        "backend",
        "model",
        "task",
        "device",
        "gpu_id",
        "n_workers",
        "tf_threads",
        "omp_threads",
        "chunksize",
        "optimizer",
        "fmax",
        "max_steps",
        "relax_mode",
        "cell_filter",
    ):
        data.pop(key, None)
    payload = {
        "search_config": data,
        "mc_search_calculator": asdict(mc_cfg),
        "explicit_vacancy_counts": (
            list(explicit_counts) if explicit_counts is not None else None
        ),
        "sources": {
            str(path.resolve()): _base._file_sha256(path) for path in source_paths
        },
    }
    return _sha256_payload(payload)


def _final_fingerprint(cfg: _base.VacancyConfig, search_fingerprint: str) -> str:
    return _sha256_payload(
        {
            "search_fingerprint": search_fingerprint,
            "final_calculator": {
                "backend": cfg.backend,
                "model": cfg.model,
                "task": cfg.task,
                "device": cfg.device,
                "gpu_id": cfg.gpu_id,
            },
            "relaxation": {
                "optimizer": cfg.optimizer,
                "fmax": cfg.fmax,
                "max_steps": cfg.max_steps,
                "relax_mode": cfg.relax_mode,
                "cell_filter": cfg.cell_filter,
            },
        }
    )


def _count_selection(
    cfg: _base.VacancyConfig,
    scenarios: list[dict[str, Any]],
    explicit_counts: tuple[int, ...] | None,
    available_sites: int,
) -> tuple[list[int], dict[str, Any]]:
    if explicit_counts is None:
        return _base.determine_vacancy_counts(
            scenarios,
            compensation_charge=cfg.vacancy_compensation_charge,
            extra_vacancies=cfg.extra_vacancies,
            max_vacancies_cap=cfg.max_vacancies_cap,
            available_sites=available_sites,
        )
    counts = list(explicit_counts)
    if counts[-1] > available_sites:
        raise ValueError(
            "[vacancies].vacancy_counts requests more vacancies than available "
            f"vacancy-species sites: requested {counts[-1]}, available {available_sites}"
        )
    negative = [
        int(item["delta_Q"]) for item in scenarios if int(item["delta_Q"]) < 0
    ]
    charge_based_max = max(
        (
            math.ceil(-charge / cfg.vacancy_compensation_charge)
            for charge in negative
        ),
        default=0,
    )
    return counts, {
        "mode": "explicit",
        "explicit_counts": counts,
        "charge_based_max": charge_based_max,
        "requested_max": max(counts),
        "applied_max": max(counts),
        "available_vacancy_species_sites": available_sites,
        "truncated": False,
        "truncation_reasons": [],
    }


def _expanded_parent(
    parent: dict[str, Path | str], cfg: _base.VacancyConfig
) -> tuple[Structure, Structure, dict[int, int]]:
    parent_id = str(parent["parent_id"])
    symmetry_parent = Structure.from_file(str(parent["symmetry_path"]))
    relaxed_parent = Structure.from_file(str(parent["relaxed_path"]))
    if cfg.supercell != (1, 1, 1):
        symmetry_parent.make_supercell(cfg.supercell)
        relaxed_parent.make_supercell(cfg.supercell)
    mapping = _base.map_parent_sites(
        symmetry_parent,
        relaxed_parent,
        tolerance=cfg.mapping_tolerance,
        parent_id=parent_id,
    )
    return symmetry_parent, relaxed_parent, mapping


def _composition_fields(
    parent: dict[str, Path | str],
    relaxed_parent: Structure,
    cfg: _base.VacancyConfig,
) -> tuple[dict[str, Any], dict[str, int]]:
    dopant_counts = _base._dopant_counts(relaxed_parent, cfg, str(parent["parent_id"]))
    species_counts = Counter(site.species_string for site in relaxed_parent)
    n_host = int(species_counts[cfg.host_species])
    n_total_dopants = sum(dopant_counts.values())
    fields = {
        "composition_directory": str(parent["composition"]),
        "composition": str(parent["composition"]),
        "candidate": str(parent["candidate"]),
        "host_species": cfg.host_species,
        "dopant_counts_from_parent": dopant_counts,
        "dopant_counts_json": dopant_counts,
        "n_host": n_host,
        "n_total_dopants": n_total_dopants,
        "n_total_cations": n_host + n_total_dopants,
        "n_oxygen_sites_parent": int(species_counts[cfg.vacancy_species]),
    }
    return fields, dopant_counts


def _run_search_parent(
    parent: dict[str, Path | str],
    cfg: _base.VacancyConfig,
    mc_cfg: MonteCarloSearchCalculatorConfig,
    explicit_counts: tuple[int, ...] | None,
    calculator: Any,
) -> list[dict[str, Any]]:
    parent_id = str(parent["parent_id"])
    symmetry_parent, relaxed_parent, mapping = _expanded_parent(parent, cfg)
    composition_fields, dopant_counts = _composition_fields(parent, relaxed_parent, cfg)
    scenarios = _base.reachable_charge_scenarios(
        dopant_counts, cfg.oxidation_states, cfg.host_oxidation_state
    )
    symmetry_vacancy_indices = [
        i
        for i, site in enumerate(symmetry_parent)
        if site.species_string == cfg.vacancy_species
    ]
    relaxed_vacancy_indices = [mapping[i] for i in symmetry_vacancy_indices]
    counts, count_meta = _count_selection(
        cfg, scenarios, explicit_counts, len(relaxed_vacancy_indices)
    )
    search_fp = _search_fingerprint(
        cfg,
        mc_cfg,
        explicit_counts,
        Path(parent["symmetry_path"]),
        Path(parent["relaxed_path"]),
    )
    vacancy_root = Path(parent["candidate_dir"]) / _base.VACANCY_STAGE_DIR
    vacancy_root.mkdir(parents=True, exist_ok=True)
    prior = _base._load_json(vacancy_root / "mc_search_meta.json")
    result_json = vacancy_root / "mc_search_results.json"
    if (
        cfg.skip_if_done
        and prior.get("status") == "complete"
        and prior.get("search_fingerprint") == search_fp
        and result_json.is_file()
    ):
        log.info("SKIP MC search %s: compatible search is complete", parent_id)
        return json.loads(result_json.read_text(encoding="utf-8"))

    count_rows = [
        {
            "n_vacancies": count,
            "charge_scenarios": _base.charge_scenarios_for_count(
                scenarios, count, cfg.vacancy_compensation_charge
            ),
        }
        for count in counts
    ]
    _base._write_json(
        vacancy_root / "vacancy_counts.json", {**count_meta, "counts": count_rows}
    )
    _base._write_csv(vacancy_root / "vacancy_counts.csv", count_rows)

    rows: list[dict[str, Any]] = []
    for n_vacancies in counts:
        log.info("MC search %s n=%d", parent_id, n_vacancies)
        group_dir = vacancy_root / f"V_{cfg.vacancy_species}_{n_vacancies:02d}"
        mc_result = _base.monte_carlo_vacancy_search(
            relaxed_parent,
            vacancy_species=cfg.vacancy_species,
            n_vacancies=n_vacancies,
            energy_function=lambda structure: structure_energy_with_calculator(
                structure, calculator
            ),
            temperature_K=cfg.mc_temperature_K,
            initial_temperature_K=(
                cfg.mc_initial_temperature_K if cfg.mc_annealing else cfg.mc_temperature_K
            ),
            annealing_hold_steps=(cfg.mc_annealing_hold_steps if cfg.mc_annealing else 0),
            annealing_steps=(cfg.mc_annealing_steps if cfg.mc_annealing else 0),
            max_steps=cfg.mc_max_steps,
            patience=cfg.mc_patience,
            run_mode=cfg.mc_run_mode,
            seed=cfg.sample_seed + n_vacancies,
            max_candidates=cfg.sample_max_saved,
            energy_window_eV=cfg.mc_energy_window_eV,
            improvement_tolerance_eV=cfg.mc_improvement_tolerance_eV,
            cation_move_weight=cfg.mc_cation_move_weight,
            vacancy_move_weight=cfg.mc_vacancy_move_weight,
        )
        summary = {
            "search_fingerprint": search_fp,
            "enumeration_mode": "monte_carlo",
            "selection_reason": mc_result.stop_reason,
            "mc_steps": mc_result.steps,
            "mc_best_energy_eV": mc_result.best_energy_eV,
            "mc_best_step": mc_result.best_step,
            "mc_attempted_moves": mc_result.attempted_moves,
            "mc_accepted_moves": mc_result.accepted_moves,
            "mc_annealing": cfg.mc_annealing,
            "mc_initial_temperature_K": (
                cfg.mc_initial_temperature_K if cfg.mc_annealing else cfg.mc_temperature_K
            ),
            "mc_target_temperature_K": cfg.mc_temperature_K,
            "mc_annealing_hold_steps": (
                cfg.mc_annealing_hold_steps if cfg.mc_annealing else 0
            ),
            "mc_annealing_steps": cfg.mc_annealing_steps if cfg.mc_annealing else 0,
            "mc_search_backend": mc_cfg.backend,
            "mc_search_model": mc_cfg.model,
            "mc_search_task": mc_cfg.task,
            "mc_search_device": mc_cfg.device,
            "mc_search_gpu_id": mc_cfg.gpu_id,
        }
        _base._write_json(group_dir / "monte_carlo_summary.json", summary)

        group_rows: list[dict[str, Any]] = []
        for rank, candidate in enumerate(mc_result.candidates, start=1):
            config_id = f"mc_{rank:06d}"
            config_dir = group_dir / config_id
            generate_dir = config_dir / "00_generate"
            generate_dir.mkdir(parents=True, exist_ok=True)
            Poscar(candidate.structure).write_file(generate_dir / "POSCAR")
            minimum_distance = _base.periodic_vacancy_distances(
                relaxed_parent, candidate.vacancy_indices
            )[0]
            charge_rows = _base.charge_scenarios_for_count(
                scenarios, n_vacancies, cfg.vacancy_compensation_charge
            )
            generation_meta = {
                "configuration_id": config_id,
                "parent_id": parent_id,
                "vacancy_species": cfg.vacancy_species,
                "n_vacancies": n_vacancies,
                "removed_site_indices": list(candidate.vacancy_indices),
                "minimum_vacancy_distance_angstrom": minimum_distance,
                "vacancy_species_sites_in_parent": len(relaxed_vacancy_indices),
                "vacancy_fraction": n_vacancies / len(relaxed_vacancy_indices),
                "vacancy_percent": 100.0 * n_vacancies / len(relaxed_vacancy_indices),
                "charge_scenarios": charge_rows,
                "source_parent_path": str(parent["relaxed_path"]),
                "search_method": "monte-carlo",
                "supercell": list(cfg.supercell),
                "search_fingerprint": search_fp,
                "mc_step": candidate.step,
                "mc_move": candidate.move,
                **summary,
            }
            _base._write_json(generate_dir / "meta.json", generation_meta)
            search_columns = _base._energy_columns(
                candidate.energy_eV, len(candidate.structure), n_vacancies
            )
            search_meta = {
                "search_fingerprint": search_fp,
                **search_columns,
                "backend": mc_cfg.backend,
                "model": mc_cfg.model,
                "task": mc_cfg.task,
                "device": mc_cfg.device,
                "energy_role": "monte_carlo_search_single_point",
                "mc_step": candidate.step,
                "mc_move": candidate.move,
            }
            _base._write_json(config_dir / "01_scan" / "meta.json", search_meta)
            row = {
                "parent_id": parent_id,
                "parent_path": str(parent["candidate_dir"]),
                "configuration_id": config_id,
                "vacancy_species": cfg.vacancy_species,
                "n_vacancies": n_vacancies,
                **composition_fields,
                "vacancy_fraction": generation_meta["vacancy_fraction"],
                "vacancy_percent": generation_meta["vacancy_percent"],
                "minimum_vacancy_distance_angstrom": minimum_distance,
                "generated_poscar_path": str(generate_dir / "POSCAR"),
                "search_method": "monte-carlo",
                "supercell": list(cfg.supercell),
                "search_fingerprint": search_fp,
                "mc_step": candidate.step,
                "mc_move": candidate.move,
                "mc_search_backend": mc_cfg.backend,
                "mc_search_model": mc_cfg.model,
                "mc_search_task": mc_cfg.task,
                "mc_search_device": mc_cfg.device,
                "mc_search_energy_sp_total_eV": candidate.energy_eV,
                "mc_search_energy_sp_per_atom_eV": search_columns["energy_sp_per_atom_eV"],
                "mc_search_energy_sp_per_vacancy_eV": search_columns[
                    "energy_sp_per_vacancy_eV"
                ],
            }
            group_rows.append(row)

        group_rows.sort(
            key=lambda row: (
                row["mc_search_energy_sp_total_eV"],
                row["configuration_id"],
            )
        )
        minimum = group_rows[0]["mc_search_energy_sp_total_eV"] if group_rows else None
        for rank, row in enumerate(group_rows, start=1):
            relative = row["mc_search_energy_sp_total_eV"] - minimum
            row.update(
                {
                    "relative_mc_search_energy_same_count_eV": relative,
                    "relative_mc_search_energy_same_count_meV": 1000.0 * relative,
                    "rank_within_vacancy_count": rank,
                    "selected_for_finalization": rank <= cfg.topk_per_vacancy_count,
                }
            )
        _base._write_csv(group_dir / "ranking_scan.csv", group_rows)
        selected = [
            str(row["configuration_id"])
            for row in group_rows
            if row["selected_for_finalization"]
        ]
        (group_dir / "selected_candidates.txt").write_text(
            "\n".join(selected) + ("\n" if selected else ""), encoding="utf-8"
        )
        rows.extend(group_rows)

    metadata = {
        "status": "complete",
        "stage": "mc_search",
        "parent_id": parent_id,
        "search_fingerprint": search_fp,
        "mc_search_calculator": asdict(mc_cfg),
        "explicit_vacancy_counts": (
            list(explicit_counts) if explicit_counts is not None else None
        ),
        "vacancy_range": count_meta,
        "supercell": list(cfg.supercell),
        "n_result_rows": len(rows),
    }
    _base._write_json(vacancy_root / "mc_search_meta.json", metadata)
    _base._write_json(vacancy_root / "mc_search_results.json", rows)
    _base._write_csv(vacancy_root / "mc_search_results.csv", rows)
    return rows


def run_vacancy_mc_search(
    raw: dict[str, Any], root: Path, *, config_path: Path | None = None
) -> Path:
    """Run only the MC occupation search; the final backend is never built."""
    cfg = _base.parse_vacancy_config(raw, root)
    if cfg.search_method != "monte-carlo":
        raise ValueError(
            "vacancies-mc-search requires [vacancies].search_method='monte-carlo'"
        )
    section = raw.get("vacancies") or {}
    explicit_counts = parse_explicit_vacancy_counts(section)
    mc_cfg = parse_monte_carlo_search_calculator(section, cfg)

    check_backend_dependency(mc_cfg.backend, stage_name="Vacancies Monte Carlo search")
    prepare_backend_runtime(
        backend=mc_cfg.backend,
        device=mc_cfg.device,
        gpu_id=mc_cfg.gpu_id,
        tf_threads=cfg.tf_threads,
        omp_threads=cfg.omp_threads,
    )
    calculator = build_ase_calculator(
        backend=mc_cfg.backend,
        model=mc_cfg.model,
        task=mc_cfg.task,
        device=mc_cfg.device,
    )

    parent_root = cfg.parent_directory if cfg.parent_source == "directory" else cfg.outdir
    assert parent_root is not None
    parents = _base.discover_selected_parents(parent_root)
    all_rows: list[dict[str, Any]] = []
    for index, parent in enumerate(parents, start=1):
        log.info(
            "MC vacancy-search parent %d/%d: %s",
            index,
            len(parents),
            parent["parent_id"],
        )
        all_rows.extend(
            _run_search_parent(parent, cfg, mc_cfg, explicit_counts, calculator)
        )

    output = parent_root / "vacancy_mc_search_database.csv"
    _base._write_csv(output, all_rows)
    _base._write_json(parent_root / "vacancy_mc_search_database.json", all_rows)
    return output


def _load_search_context(
    parent: dict[str, Path | str],
    cfg: _base.VacancyConfig,
    mc_cfg: MonteCarloSearchCalculatorConfig,
    explicit_counts: tuple[int, ...] | None,
) -> tuple[Structure, str, list[int], dict[str, Any], dict[str, int]]:
    _, relaxed_parent, _ = _expanded_parent(parent, cfg)
    composition_fields, dopant_counts = _composition_fields(parent, relaxed_parent, cfg)
    search_fp = _search_fingerprint(
        cfg,
        mc_cfg,
        explicit_counts,
        Path(parent["symmetry_path"]),
        Path(parent["relaxed_path"]),
    )
    vacancy_root = Path(parent["candidate_dir"]) / _base.VACANCY_STAGE_DIR
    search_meta = _base._load_json(vacancy_root / "mc_search_meta.json")
    if search_meta.get("status") != "complete":
        raise FileNotFoundError(
            f"MC search is not complete for {parent['parent_id']}; "
            "run dopingflow vacancies-mc-search first"
        )
    if search_meta.get("search_fingerprint") != search_fp:
        raise ValueError(
            f"MC search fingerprint mismatch for {parent['parent_id']}; "
            "the search settings or parent structure changed. Rerun "
            "dopingflow vacancies-mc-search."
        )
    counts = [
        int(value)
        for value in search_meta.get("vacancy_range", {}).get("explicit_counts", [])
    ]
    if not counts:
        count_data = _base._load_json(vacancy_root / "vacancy_counts.json")
        counts = [int(item["n_vacancies"]) for item in count_data.get("counts", [])]
    return relaxed_parent, search_fp, counts, composition_fields, dopant_counts


def _run_finalize_parent(
    parent: dict[str, Path | str],
    cfg: _base.VacancyConfig,
    mc_cfg: MonteCarloSearchCalculatorConfig,
    explicit_counts: tuple[int, ...] | None,
    calculator: Any,
) -> list[dict[str, Any]]:
    parent_id = str(parent["parent_id"])
    (
        relaxed_parent,
        search_fp,
        counts,
        composition_fields,
        dopant_counts,
    ) = _load_search_context(parent, cfg, mc_cfg, explicit_counts)
    final_fp = _final_fingerprint(cfg, search_fp)
    vacancy_root = Path(parent["candidate_dir"]) / _base.VACANCY_STAGE_DIR

    prior = _base._load_json(vacancy_root / "meta.json")
    result_json = vacancy_root / "vacancy_results.json"
    if (
        cfg.skip_if_done
        and prior.get("status") == "complete"
        and prior.get("configuration_fingerprint") == final_fp
        and result_json.is_file()
    ):
        log.info("SKIP finalization %s: compatible final results exist", parent_id)
        return json.loads(result_json.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    if cfg.include_parent_reference:
        parent_ref = _base._parent_reference(
            parent, relaxed_parent, vacancy_root, cfg, calculator, final_fp
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
            **_base._energy_columns(parent_ref["parent_energy_sp_eV"], len(relaxed_parent), 0),
            "energy_relaxed_total_eV": parent_ref["parent_energy_relaxed_eV"],
            "converged": parent_ref["parent_converged"],
            "backend": cfg.backend,
            "model": cfg.model,
            "task": cfg.task,
            "generated_poscar_path": str(vacancy_root / "parent_reference" / "POSCAR"),
            "relaxed_poscar_path": str(
                vacancy_root / "parent_reference" / "relaxed" / "POSCAR"
            ),
            "configuration_fingerprint": final_fp,
            "search_fingerprint": search_fp,
            **parent_ref,
        }
        parent_row["energy_sp_reported_eV"] = _base._reported_energy(
            parent_row, cfg.energy_normalization
        )
        rows.append(parent_row)

    for n_vacancies in counts:
        group_dir = vacancy_root / f"V_{cfg.vacancy_species}_{n_vacancies:02d}"
        selection_file = group_dir / "selected_candidates.txt"
        if not selection_file.is_file():
            raise FileNotFoundError(
                f"Missing MC selection file for {parent_id}, n={n_vacancies}: "
                f"{selection_file}"
            )
        selected_ids = [
            line.strip()
            for line in selection_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        group_rows: list[dict[str, Any]] = []
        for config_id in selected_ids:
            config_root = group_dir / config_id
            generated_path = config_root / "00_generate" / "POSCAR"
            if not generated_path.is_file():
                raise FileNotFoundError(f"Missing MC-generated structure: {generated_path}")
            structure = Structure.from_file(generated_path)
            search_meta = _base._load_json(config_root / "01_scan" / "meta.json")
            search_energy = search_meta.get("energy_sp_total_eV")

            final_scan_dir = config_root / "01_final_scan"
            final_scan_meta = _base._load_json(final_scan_dir / "meta.json")
            if (
                cfg.resume
                and final_scan_meta.get("configuration_fingerprint") == final_fp
                and final_scan_meta.get("energy_sp_total_eV") is not None
            ):
                final_sp = float(final_scan_meta["energy_sp_total_eV"])
            else:
                final_sp = structure_energy_with_calculator(structure, calculator)
                final_scan_meta = {
                    "configuration_fingerprint": final_fp,
                    **_base._energy_columns(final_sp, len(structure), n_vacancies),
                    "backend": cfg.backend,
                    "model": cfg.model,
                    "task": cfg.task,
                    "device": cfg.device,
                    "energy_role": "final_backend_single_point_before_relaxation",
                    "mc_search_energy_sp_total_eV": search_energy,
                    "mc_search_backend": mc_cfg.backend,
                    "mc_search_model": mc_cfg.model,
                    "mc_search_task": mc_cfg.task,
                }
                _base._write_json(final_scan_dir / "meta.json", final_scan_meta)

            relax_dir = config_root / "02_relax"
            relax_meta = _base._load_json(relax_dir / "meta.json")
            if (
                cfg.resume
                and relax_meta.get("configuration_fingerprint") == final_fp
                and (relax_dir / "POSCAR").is_file()
                and relax_meta.get("energy_relaxed_total_eV") is not None
            ):
                relaxed_energy = float(relax_meta["energy_relaxed_total_eV"])
                converged = bool(relax_meta.get("converged", False))
            else:
                start = time.time()
                relaxed, relaxed_energy, nsteps, final_force, converged = (
                    relax_structure_with_calculator(
                        structure,
                        calculator=calculator,
                        optimizer_name=cfg.optimizer,
                        fmax=cfg.fmax,
                        max_steps=cfg.max_steps,
                        relax_mode=cfg.relax_mode,
                        cell_filter=cfg.cell_filter,
                    )
                )
                relax_dir.mkdir(parents=True, exist_ok=True)
                Poscar(relaxed).write_file(relax_dir / "POSCAR")
                relax_meta = {
                    "configuration_fingerprint": final_fp,
                    "energy_initial_eV": final_sp,
                    "energy_relaxed_total_eV": relaxed_energy,
                    "energy_change_eV": relaxed_energy - final_sp,
                    "optimizer_steps": nsteps,
                    "final_fmax": final_force,
                    "converged": converged,
                    "walltime_s": time.time() - start,
                    "optimizer": cfg.optimizer,
                    "fmax_target": cfg.fmax,
                    "fmax_target_eV_per_A": cfg.fmax,
                    "max_steps": cfg.max_steps,
                    "relax_mode": cfg.relax_mode,
                    "cell_filter": cfg.cell_filter,
                    "backend": cfg.backend,
                    "model": cfg.model,
                    "task": cfg.task,
                    "device": cfg.device,
                    "mc_search_energy_sp_total_eV": search_energy,
                    "mc_search_backend": mc_cfg.backend,
                    "mc_search_model": mc_cfg.model,
                    "mc_search_task": mc_cfg.task,
                }
                _base._write_json(relax_dir / "meta.json", relax_meta)

            generation_meta = _base._load_json(config_root / "00_generate" / "meta.json")
            final_columns = _base._energy_columns(final_sp, len(structure), n_vacancies)
            row = {
                "parent_id": parent_id,
                "parent_path": str(parent["candidate_dir"]),
                "configuration_id": config_id,
                "vacancy_species": cfg.vacancy_species,
                "n_vacancies": n_vacancies,
                **composition_fields,
                "vacancy_fraction": generation_meta.get("vacancy_fraction"),
                "vacancy_percent": generation_meta.get("vacancy_percent"),
                "minimum_vacancy_distance_angstrom": generation_meta.get(
                    "minimum_vacancy_distance_angstrom"
                ),
                "delta_Q_values": [
                    scenario["delta_Q"]
                    for scenario in generation_meta.get("charge_scenarios", [])
                ],
                "residual_charge_values": [
                    scenario["residual_charge"]
                    for scenario in generation_meta.get("charge_scenarios", [])
                ],
                "has_fully_compensated_scenario": any(
                    scenario.get("fully_compensated", False)
                    for scenario in generation_meta.get("charge_scenarios", [])
                ),
                "selected_for_relaxation": True,
                "search_method": "monte-carlo",
                "supercell": list(cfg.supercell),
                "search_fingerprint": search_fp,
                "configuration_fingerprint": final_fp,
                "mc_search_backend": mc_cfg.backend,
                "mc_search_model": mc_cfg.model,
                "mc_search_task": mc_cfg.task,
                "mc_search_device": mc_cfg.device,
                "mc_search_energy_sp_total_eV": search_energy,
                "backend": cfg.backend,
                "model": cfg.model,
                "task": cfg.task,
                "generated_poscar_path": str(generated_path),
                "final_scan_meta_path": str(final_scan_dir / "meta.json"),
                "relaxed_poscar_path": str(relax_dir / "POSCAR"),
                **final_columns,
                "energy_relaxed_total_eV": relaxed_energy,
                "converged": converged,
                **relax_meta,
            }
            row["energy_sp_reported_eV"] = _base._reported_energy(
                row, cfg.energy_normalization
            )
            group_rows.append(row)

        group_rows.sort(
            key=lambda row: (
                float(row["energy_relaxed_total_eV"]),
                row["configuration_id"],
            )
        )
        if group_rows:
            minimum = float(group_rows[0]["energy_relaxed_total_eV"])
            for rank, row in enumerate(group_rows, start=1):
                relative = float(row["energy_relaxed_total_eV"]) - minimum
                row["relative_relaxed_energy_same_count_eV"] = relative
                row["relative_relaxed_energy_same_count_meV"] = relative * 1000.0
                row["rank_relaxed_within_vacancy_count"] = rank
        _base._write_csv(group_dir / "ranking_relax.csv", group_rows)
        rows.extend(group_rows)

    metadata = {
        "status": "complete",
        "stage": "finalize",
        "parent_id": parent_id,
        "search_fingerprint": search_fp,
        "configuration_fingerprint": final_fp,
        "mc_search_calculator": asdict(mc_cfg),
        "final_calculator": {
            "backend": cfg.backend,
            "model": cfg.model,
            "task": cfg.task,
            "device": cfg.device,
            "gpu_id": cfg.gpu_id,
        },
        "resolved_config": _base._jsonable_config(cfg),
        "dopant_counts_from_parent": dopant_counts,
        "n_result_rows": len(rows),
    }
    _base._write_json(vacancy_root / "vacancy_results.json", rows)
    _base._write_csv(vacancy_root / "vacancy_results.csv", rows)
    _base._write_json(vacancy_root / "meta.json", metadata)
    return rows


def run_vacancy_finalize(
    raw: dict[str, Any], root: Path, *, config_path: Path | None = None
) -> Path:
    """Finalize a completed MC search using only the ordinary vacancy backend."""
    cfg = _base.parse_vacancy_config(raw, root)
    if cfg.search_method != "monte-carlo":
        raise ValueError(
            "vacancies-finalize requires [vacancies].search_method='monte-carlo'"
        )
    section = raw.get("vacancies") or {}
    explicit_counts = parse_explicit_vacancy_counts(section)
    mc_cfg = parse_monte_carlo_search_calculator(section, cfg)

    check_backend_dependency(cfg.backend, stage_name="Vacancies final calculator")
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
    parents = _base.discover_selected_parents(parent_root)
    all_rows: list[dict[str, Any]] = []
    for index, parent in enumerate(parents, start=1):
        log.info(
            "Vacancy finalization parent %d/%d: %s",
            index,
            len(parents),
            parent["parent_id"],
        )
        all_rows.extend(
            _run_finalize_parent(parent, cfg, mc_cfg, explicit_counts, calculator)
        )

    csv_path = parent_root / "vacancies_database.csv"
    _base._write_csv(csv_path, all_rows)
    _base._write_json(parent_root / "vacancies_database.json", all_rows)
    if cfg.analysis.enabled:
        log.info("Static-lattice vacancy thermodynamic analysis")
        _base.analyze_static_vacancy_thermodynamics(
            rows=all_rows,
            cfg=cfg.analysis,
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
    return csv_path


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def run_vacancy_mc_search_from_toml(config_path: Path) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_vacancy_mc_search(
        raw, config_path.resolve().parent, config_path=config_path
    )


def run_vacancy_finalize_from_toml(config_path: Path) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_vacancy_finalize(
        raw, config_path.resolve().parent, config_path=config_path
    )


__all__ = [
    "run_vacancy_mc_search",
    "run_vacancy_mc_search_from_toml",
    "run_vacancy_finalize",
    "run_vacancy_finalize_from_toml",
]

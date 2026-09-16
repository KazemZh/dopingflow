"""Opt-in extensions for large-cell Monte Carlo vacancy searches.

Two controls are added without changing legacy vacancy inputs:

* ``vacancy_counts = [1, 2, ...]`` searches exactly the requested positive
  vacancy counts instead of deriving a continuous range from formal charges.
* ``mc_backend`` / ``mc_model`` / ``mc_task`` / ``mc_device`` / ``mc_gpu_id``
  select a calculator used only for Metropolis occupation energies.  The normal
  vacancy calculator remains authoritative for parent/reference calculations,
  final top-k relaxations, thermodynamic analysis, and correction provenance.

The Monte Carlo archive and top-k selection use the MC-search energies.  When
search and final calculators differ, those search energies are explicitly
labelled and are not presented as same-calculator relaxation energy changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

from dopingflow import vacancies as _base
from dopingflow.ml_backends import (
    build_ase_calculator,
    check_backend_dependency,
    normalize_backend_config,
    prepare_backend_runtime,
)
from dopingflow.ml_relaxation import structure_energy_with_calculator

log = logging.getLogger(__name__)

_ORIGINAL_RUN = getattr(_base, "_mc_extension_original_run", _base.run_vacancies)
_ORIGINAL_RUN_PARENT = getattr(
    _base, "_mc_extension_original_run_parent", _base._run_parent
)
_ORIGINAL_MONTE_CARLO_SEARCH = getattr(
    _base, "_mc_extension_original_search", _base.monte_carlo_vacancy_search
)
_ORIGINAL_DETERMINE_COUNTS = getattr(
    _base, "_mc_extension_original_determine_counts", _base.determine_vacancy_counts
)
_ORIGINAL_FINGERPRINT = getattr(
    _base, "_mc_extension_original_fingerprint", _base.vacancy_config_fingerprint
)


@dataclass(frozen=True)
class MonteCarloSearchCalculatorConfig:
    """Resolved calculator used only for Monte Carlo occupation energies."""

    backend: str
    model: str
    task: str
    device: str
    gpu_id: int
    explicitly_configured: bool = False


_ACTIVE_MC_CONFIG: MonteCarloSearchCalculatorConfig | None = None
_ACTIVE_MC_CALCULATOR: Any = None
_ACTIVE_EXPLICIT_COUNTS: tuple[int, ...] | None = None


def parse_explicit_vacancy_counts(section: dict[str, Any]) -> tuple[int, ...] | None:
    """Return validated explicit vacancy counts, or ``None`` for legacy behavior."""

    if "vacancy_counts" not in section:
        return None
    raw = section.get("vacancy_counts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("[vacancies].vacancy_counts must be a non-empty integer array")
    counts: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"[vacancies].vacancy_counts[{index}] must be a positive integer"
            )
        counts.append(int(value))
    if len(set(counts)) != len(counts):
        raise ValueError("[vacancies].vacancy_counts must not contain duplicates")
    return tuple(sorted(counts))


def parse_monte_carlo_search_calculator(
    section: dict[str, Any], cfg: _base.VacancyConfig
) -> MonteCarloSearchCalculatorConfig:
    """Resolve the optional MC-only calculator, inheriting final settings by default."""

    explicit_keys = {"mc_backend", "mc_model", "mc_task", "mc_device", "mc_gpu_id"}
    explicitly_configured = any(key in section for key in explicit_keys)
    backend, model, task = normalize_backend_config(
        backend=str(section.get("mc_backend", cfg.backend)),
        model=str(section.get("mc_model", cfg.model)),
        task=str(section.get("mc_task", cfg.task)),
        section_name="vacancies.mc_search",
    )
    device = str(section.get("mc_device", cfg.device)).strip().lower()
    gpu_id = section.get("mc_gpu_id", cfg.gpu_id)
    if device not in {"cpu", "cuda"}:
        raise ValueError("[vacancies].mc_device must be 'cpu' or 'cuda'")
    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ValueError("[vacancies].mc_gpu_id must be a non-negative integer")
    if cfg.device == "cuda" and device == "cuda" and int(gpu_id) != cfg.gpu_id:
        raise ValueError(
            "[vacancies] final and MC calculators currently must use the same gpu_id "
            "when both use CUDA in one vacancy process"
        )
    return MonteCarloSearchCalculatorConfig(
        backend=backend,
        model=model,
        task=task,
        device=device,
        gpu_id=int(gpu_id),
        explicitly_configured=explicitly_configured,
    )


def _explicit_count_result(
    scenarios: Sequence[dict[str, Any]],
    *,
    compensation_charge: int,
    available_sites: int,
) -> tuple[list[int], dict[str, Any]]:
    assert _ACTIVE_EXPLICIT_COUNTS is not None
    counts = list(_ACTIVE_EXPLICIT_COUNTS)
    if counts[-1] > available_sites:
        raise ValueError(
            "[vacancies].vacancy_counts requests more vacancies than available "
            f"vacancy-species sites: requested {counts[-1]}, available {available_sites}"
        )
    negative = [int(item["delta_Q"]) for item in scenarios if int(item["delta_Q"]) < 0]
    charge_based_max = max(
        (math.ceil(-charge / compensation_charge) for charge in negative), default=0
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


def determine_vacancy_counts(
    scenarios: Sequence[dict[str, Any]],
    *,
    compensation_charge: int,
    extra_vacancies: int,
    max_vacancies_cap: int,
    available_sites: int,
) -> tuple[list[int], dict[str, Any]]:
    """Honor explicit counts when requested; otherwise delegate unchanged."""

    if _ACTIVE_EXPLICIT_COUNTS is None:
        return _ORIGINAL_DETERMINE_COUNTS(
            scenarios,
            compensation_charge=compensation_charge,
            extra_vacancies=extra_vacancies,
            max_vacancies_cap=max_vacancies_cap,
            available_sites=available_sites,
        )
    return _explicit_count_result(
        scenarios,
        compensation_charge=compensation_charge,
        available_sites=available_sites,
    )


def vacancy_config_fingerprint(cfg: _base.VacancyConfig, *source_paths: Path) -> str:
    """Include opt-in MC calculator/count controls in cache invalidation."""

    base_fingerprint = _ORIGINAL_FINGERPRINT(cfg, *source_paths)
    mc_cfg = _ACTIVE_MC_CONFIG
    if (
        _ACTIVE_EXPLICIT_COUNTS is None
        and (mc_cfg is None or not mc_cfg.explicitly_configured)
    ):
        return base_fingerprint
    payload = {
        "base_fingerprint": base_fingerprint,
        "explicit_vacancy_counts": (
            list(_ACTIVE_EXPLICIT_COUNTS) if _ACTIVE_EXPLICIT_COUNTS is not None else None
        ),
        "mc_search_calculator": (
            asdict(mc_cfg) if mc_cfg is not None and mc_cfg.explicitly_configured else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def monte_carlo_vacancy_search(parent, **kwargs):
    """Route MC occupation energies through the MC-only calculator when active."""

    if _ACTIVE_MC_CALCULATOR is None:
        return _ORIGINAL_MONTE_CARLO_SEARCH(parent, **kwargs)
    forwarded = dict(kwargs)
    forwarded["energy_function"] = lambda structure: structure_energy_with_calculator(
        structure, _ACTIVE_MC_CALCULATOR
    )
    return _ORIGINAL_MONTE_CARLO_SEARCH(parent, **forwarded)


def _mc_provenance() -> dict[str, Any]:
    if _ACTIVE_MC_CONFIG is None:
        return {}
    return {
        "mc_search_backend": _ACTIVE_MC_CONFIG.backend,
        "mc_search_model": _ACTIVE_MC_CONFIG.model,
        "mc_search_task": _ACTIVE_MC_CONFIG.task,
        "mc_search_device": _ACTIVE_MC_CONFIG.device,
        "mc_search_gpu_id": _ACTIVE_MC_CONFIG.gpu_id,
    }


def _mixed_calculators(cfg: _base.VacancyConfig) -> bool:
    mc = _ACTIVE_MC_CONFIG
    if mc is None:
        return False
    return (mc.backend, mc.model, mc.task, mc.device, mc.gpu_id) != (
        cfg.backend,
        cfg.model,
        cfg.task,
        cfg.device,
        cfg.gpu_id,
    )


def _rewrite_mc_artifacts(
    parent: dict[str, Path | str],
    cfg: _base.VacancyConfig,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Make search and final energy provenance explicit in written artifacts."""

    if cfg.search_method != "monte-carlo" or _ACTIVE_MC_CONFIG is None:
        return rows, metadata

    provenance = _mc_provenance()
    mixed = _mixed_calculators(cfg)
    vacancy_root = Path(parent["candidate_dir"]) / _base.VACANCY_STAGE_DIR

    for row in rows:
        if int(row.get("n_vacancies", 0)) <= 0:
            continue
        if str(row.get("enumeration_mode", "")) != "monte_carlo":
            continue

        row.update(provenance)
        row["mc_search_energy_sp_total_eV"] = row.get("energy_sp_total_eV")
        row["energy_sp_backend"] = _ACTIVE_MC_CONFIG.backend
        row["energy_sp_model"] = _ACTIVE_MC_CONFIG.model
        row["energy_sp_task"] = _ACTIVE_MC_CONFIG.task
        row["energy_sp_device"] = _ACTIVE_MC_CONFIG.device
        row["final_backend"] = cfg.backend
        row["final_model"] = cfg.model
        row["final_task"] = cfg.task
        row["final_device"] = cfg.device
        row["topk_selection_energy_source"] = "monte_carlo_search_single_point"

        config_id = str(row.get("configuration_id", ""))
        n_vacancies = int(row["n_vacancies"])
        config_root = (
            vacancy_root / f"V_{cfg.vacancy_species}_{n_vacancies:02d}" / config_id
        )
        scan_meta_path = config_root / "01_scan" / "meta.json"
        scan_meta = _base._load_json(scan_meta_path)
        if scan_meta:
            scan_meta.update(
                {
                    "backend": _ACTIVE_MC_CONFIG.backend,
                    "model": _ACTIVE_MC_CONFIG.model,
                    "task": _ACTIVE_MC_CONFIG.task,
                    "device": _ACTIVE_MC_CONFIG.device,
                    "energy_role": "monte_carlo_search_single_point",
                    **provenance,
                    "final_backend": cfg.backend,
                    "final_model": cfg.model,
                    "final_task": cfg.task,
                    "final_device": cfg.device,
                }
            )
            _base._write_json(scan_meta_path, scan_meta)

        if mixed and row.get("energy_relaxed_total_eV") is not None:
            relax_meta_path = config_root / "02_relax" / "meta.json"
            relax_meta = _base._load_json(relax_meta_path)
            if relax_meta:
                search_initial = relax_meta.get("energy_initial_eV")
                relax_meta.update(
                    {
                        "mc_search_energy_initial_eV": search_initial,
                        "energy_initial_eV": None,
                        "energy_change_eV": None,
                        "energy_initial_comparable_to_relaxed": False,
                        "energy_change_note": (
                            "Initial search energy used a different calculator; "
                            "no cross-backend relaxation energy change is reported."
                        ),
                        **provenance,
                    }
                )
                _base._write_json(relax_meta_path, relax_meta)
                row["mc_search_energy_initial_eV"] = search_initial
                row["energy_initial_eV"] = None
                row["energy_change_eV"] = None
                row["energy_initial_comparable_to_relaxed"] = False
                row["energy_change_note"] = relax_meta["energy_change_note"]

    counts = sorted(
        {int(row["n_vacancies"]) for row in rows if int(row.get("n_vacancies", 0)) > 0}
    )
    for n_vacancies in counts:
        group_dir = vacancy_root / f"V_{cfg.vacancy_species}_{n_vacancies:02d}"
        group_rows = [row for row in rows if int(row.get("n_vacancies", -1)) == n_vacancies]
        group_rows.sort(
            key=lambda row: (float(row["energy_sp_total_eV"]), row["configuration_id"])
        )
        _base._write_csv(group_dir / "ranking_scan.csv", group_rows)

        relaxed_rows = [
            row for row in group_rows if row.get("energy_relaxed_total_eV") is not None
        ]
        relaxed_rows.sort(
            key=lambda row: (float(row["energy_relaxed_total_eV"]), row["configuration_id"])
        )
        _base._write_csv(group_dir / "ranking_relax.csv", relaxed_rows)

        summary_path = group_dir / "monte_carlo_summary.json"
        summary = _base._load_json(summary_path)
        if summary:
            summary.update(
                {
                    **provenance,
                    "topk_selection_energy_source": "monte_carlo_search_single_point",
                    "final_backend": cfg.backend,
                    "final_model": cfg.model,
                    "final_task": cfg.task,
                    "final_device": cfg.device,
                }
            )
            _base._write_json(summary_path, summary)

    resolved = metadata.setdefault("resolved_config", {})
    resolved["vacancy_counts"] = (
        list(_ACTIVE_EXPLICIT_COUNTS) if _ACTIVE_EXPLICIT_COUNTS is not None else None
    )
    resolved["mc_search_calculator"] = asdict(_ACTIVE_MC_CONFIG)
    metadata["mc_search_calculator"] = asdict(_ACTIVE_MC_CONFIG)
    metadata["explicit_vacancy_counts"] = (
        list(_ACTIVE_EXPLICIT_COUNTS) if _ACTIVE_EXPLICIT_COUNTS is not None else None
    )

    _base._write_json(vacancy_root / "vacancy_results.json", rows)
    _base._write_csv(vacancy_root / "vacancy_results.csv", rows)
    _base._write_json(vacancy_root / "meta.json", metadata)
    return rows, metadata


def _run_parent(
    parent: dict[str, Path | str], cfg: _base.VacancyConfig, calculator: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, metadata = _ORIGINAL_RUN_PARENT(parent, cfg, calculator)
    return _rewrite_mc_artifacts(parent, cfg, rows, metadata)


def run_vacancies(
    raw: dict[str, Any], root: Path, *, config_path: Path | None = None
) -> Path:
    """Run vacancies with optional MC-only calculator and explicit counts."""

    global _ACTIVE_MC_CONFIG, _ACTIVE_MC_CALCULATOR, _ACTIVE_EXPLICIT_COUNTS

    cfg = _base.parse_vacancy_config(raw, root)
    section = raw.get("vacancies") or {}
    explicit_counts = parse_explicit_vacancy_counts(section)
    mc_cfg = parse_monte_carlo_search_calculator(section, cfg)

    check_backend_dependency(cfg.backend, stage_name="Vacancies final calculator")
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

    mc_calculator = None
    if cfg.search_method == "monte-carlo":
        if (
            mc_cfg.backend == cfg.backend
            and mc_cfg.model == cfg.model
            and mc_cfg.task == cfg.task
            and mc_cfg.device == cfg.device
            and mc_cfg.gpu_id == cfg.gpu_id
        ):
            mc_calculator = calculator
        else:
            check_backend_dependency(mc_cfg.backend, stage_name="Vacancies Monte Carlo search")
            prepare_backend_runtime(
                backend=mc_cfg.backend,
                device=mc_cfg.device,
                gpu_id=mc_cfg.gpu_id,
                tf_threads=cfg.tf_threads,
                omp_threads=cfg.omp_threads,
            )
            mc_calculator = build_ase_calculator(
                backend=mc_cfg.backend,
                model=mc_cfg.model,
                task=mc_cfg.task,
                device=mc_cfg.device,
            )
            log.info(
                "Vacancy MC search calculator: %s / %s / %s (%s); "
                "final calculator: %s / %s / %s (%s)",
                mc_cfg.backend,
                mc_cfg.model,
                mc_cfg.task or "<none>",
                mc_cfg.device,
                cfg.backend,
                cfg.model,
                cfg.task or "<none>",
                cfg.device,
            )

    previous = (_ACTIVE_MC_CONFIG, _ACTIVE_MC_CALCULATOR, _ACTIVE_EXPLICIT_COUNTS)
    _ACTIVE_MC_CONFIG = mc_cfg
    _ACTIVE_MC_CALCULATOR = mc_calculator
    _ACTIVE_EXPLICIT_COUNTS = explicit_counts
    try:
        parent_root = cfg.parent_directory if cfg.parent_source == "directory" else cfg.outdir
        assert parent_root is not None
        parents = _base.discover_selected_parents(parent_root)
        all_rows: list[dict[str, Any]] = []
        for index, parent in enumerate(parents, start=1):
            log.info("Vacancy parent %d/%d: %s", index, len(parents), parent["parent_id"])
            rows, _ = _base._run_parent(parent, cfg, calculator)
            all_rows.extend(rows)

        csv_path = parent_root / "vacancies_database.csv"
        _base._write_csv(csv_path, all_rows)
        _base._write_json(parent_root / "vacancies_database.json", all_rows)
        if cfg.analysis.enabled:
            log.info("[6/7] Static-lattice vacancy thermodynamic analysis")
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
        log.info("[7/7] Summary writing complete: %s", csv_path)
        return csv_path
    finally:
        _ACTIVE_MC_CONFIG, _ACTIVE_MC_CALCULATOR, _ACTIVE_EXPLICIT_COUNTS = previous


def install_extensions() -> None:
    if getattr(_base, "_mc_search_extension_installed", False):
        return
    _base._mc_extension_original_run = _ORIGINAL_RUN
    _base._mc_extension_original_run_parent = _ORIGINAL_RUN_PARENT
    _base._mc_extension_original_search = _ORIGINAL_MONTE_CARLO_SEARCH
    _base._mc_extension_original_determine_counts = _ORIGINAL_DETERMINE_COUNTS
    _base._mc_extension_original_fingerprint = _ORIGINAL_FINGERPRINT
    _base.determine_vacancy_counts = determine_vacancy_counts
    _base.vacancy_config_fingerprint = vacancy_config_fingerprint
    _base.monte_carlo_vacancy_search = monte_carlo_vacancy_search
    _base._run_parent = _run_parent
    _base.run_vacancies = run_vacancies
    _base._mc_search_extension_installed = True


install_extensions()


__all__ = [
    "MonteCarloSearchCalculatorConfig",
    "determine_vacancy_counts",
    "install_extensions",
    "monte_carlo_vacancy_search",
    "parse_explicit_vacancy_counts",
    "parse_monte_carlo_search_calculator",
    "run_vacancies",
    "vacancy_config_fingerprint",
]

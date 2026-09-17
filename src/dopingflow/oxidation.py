from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pymatgen.core import Structure

log = logging.getLogger(__name__)


STRATEGY_METHODS: dict[str, tuple[str, ...]] = {
    "structural": ("bond-valence", "toss-bayesian"),
    "ml": ("toss-gnn", "chgnet", "bertos"),
    "dft": ("dft-electronic", "bader", "wannier", "eos"),
}
METHOD_STRATEGY = {
    method: strategy
    for strategy, methods in STRATEGY_METHODS.items()
    for method in methods
}
ALL_METHODS = tuple(METHOD_STRATEGY)

_METHOD_ALIASES = {
    "bond_valence": "bond-valence",
    "bondvalence": "bond-valence",
    "pymatgen-bv": "bond-valence",
    "pymatgen_bv": "bond-valence",
    "toss": "toss-bayesian",
    "toss_bayesian": "toss-bayesian",
    "toss-gnn": "toss-gnn",
    "toss_gnn": "toss-gnn",
    "chgnet": "chgnet",
    "bertos": "bertos",
    "dft": "dft-electronic",
    "dft_electronic": "dft-electronic",
    "bader": "bader",
    "wannier": "wannier",
    "eos": "eos",
}

# Process-local cache shared by optional ML adapters. It deliberately does not
# cache structures/results, only expensive model objects.
_PROCESS_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


class OptionalMethodUnavailable(RuntimeError):
    """Raised when a requested optional method cannot be imported or configured."""


@dataclass(frozen=True)
class OxidationConfig:
    root: Path
    source_root: Path
    output_dir: Path
    enabled: bool
    strategy: str
    methods: tuple[str, ...]
    include_vacancy_free: bool
    include_oxygen_vacancies: bool
    mapping_tolerance: float
    fail_fast: bool
    dft_followup_enabled: bool
    dft_followup_candidate_limit: int
    dft_followup_execute: bool
    dft_followup_methods: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructureTarget:
    target_id: str
    parent_id: str
    kind: str
    structure_path: Path
    n_vacancies: int
    vacancy_species: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def safe_id(self) -> str:
        text = self.target_id.replace("\\", "__").replace("/", "__")
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _normalize_method(value: str) -> str:
    key = str(value).strip().lower()
    key = _METHOD_ALIASES.get(key, key)
    if key not in METHOD_STRATEGY:
        raise ValueError(
            f"Unknown oxidation-state method '{value}'. Valid methods: {', '.join(ALL_METHODS)}"
        )
    return key


def _parse_method_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        raw = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError("[oxidation].methods must be an array or comma-separated string")
    methods: list[str] = []
    for item in raw:
        method = _normalize_method(item)
        if method not in methods:
            methods.append(method)
    return tuple(methods)


def parse_oxidation_config(
    raw: dict[str, Any],
    root: Path,
    *,
    strategy_override: str | None = None,
    methods_override: Sequence[str] | str | None = None,
) -> OxidationConfig:
    section = raw.get("oxidation", {}) or {}
    if not isinstance(section, dict):
        raise ValueError("[oxidation] must be a TOML table")

    enabled = bool(section.get("enabled", True))
    strategy = str(strategy_override or section.get("strategy", "structural")).strip().lower()
    if strategy not in {"structural", "ml", "dft", "combined"}:
        raise ValueError("[oxidation].strategy must be structural, ml, dft, or combined")

    if methods_override is not None:
        methods = _parse_method_list(methods_override)
    else:
        methods = _parse_method_list(section.get("methods"))
    if not methods:
        if strategy == "combined":
            raise ValueError(
                "[oxidation].methods is required for strategy='combined'; select methods explicitly"
            )
        methods = STRATEGY_METHODS[strategy]

    if strategy != "combined":
        wrong = [method for method in methods if METHOD_STRATEGY[method] != strategy]
        if wrong:
            raise ValueError(
                f"[oxidation].strategy='{strategy}' cannot select methods {wrong}; "
                "use strategy='combined' to mix method groups"
            )

    st = raw.get("structure", {}) or {}
    source_value = str(section.get("source_root", st.get("outdir", "random_structures"))).strip()
    source_path = Path(source_value).expanduser()
    source_root = (source_path if source_path.is_absolute() else root / source_path).resolve()

    output_value = section.get("output_dir", "06_oxidation")
    output_path = Path(str(output_value)).expanduser()
    output_dir = (
        output_path if output_path.is_absolute() else source_root / output_path
    ).resolve()

    mapping_tolerance = float(section.get("mapping_tolerance", 1.2))
    if mapping_tolerance <= 0:
        raise ValueError("[oxidation].mapping_tolerance must be > 0")

    followup = section.get("dft_followup", {}) or {}
    if not isinstance(followup, dict):
        raise ValueError("[oxidation.dft_followup] must be a TOML table")
    followup_limit = followup.get("candidate_limit", 10)
    if isinstance(followup_limit, bool) or not isinstance(followup_limit, int) or followup_limit <= 0:
        raise ValueError("[oxidation.dft_followup].candidate_limit must be a positive integer")
    followup_methods = _parse_method_list(
        followup.get("methods", ["dft-electronic", "bader", "eos"])
    )
    if any(METHOD_STRATEGY[method] != "dft" for method in followup_methods):
        raise ValueError("[oxidation.dft_followup].methods may contain only DFT methods")

    # Keep all nested method settings so adapters can evolve without the core
    # parser becoming coupled to optional third-party packages.
    settings = dict(section)

    return OxidationConfig(
        root=root.resolve(),
        source_root=source_root,
        output_dir=output_dir,
        enabled=enabled,
        strategy=strategy,
        methods=methods,
        include_vacancy_free=bool(section.get("include_vacancy_free", True)),
        include_oxygen_vacancies=bool(section.get("include_oxygen_vacancies", True)),
        mapping_tolerance=mapping_tolerance,
        fail_fast=bool(section.get("fail_fast", False)),
        dft_followup_enabled=bool(followup.get("enabled", False)),
        dft_followup_candidate_limit=int(followup_limit),
        dft_followup_execute=bool(followup.get("execute", False)),
        dft_followup_methods=followup_methods,
        settings=settings,
    )


def cached_model(key: tuple[Any, ...], loader: Callable[[], Any]) -> Any:
    """Load an optional model once per worker/process and reuse it."""
    if key not in _PROCESS_MODEL_CACHE:
        _PROCESS_MODEL_CACHE[key] = loader()
    return _PROCESS_MODEL_CACHE[key]


def clear_model_cache() -> None:
    """Clear process-local optional-model cache (mainly useful for tests)."""
    _PROCESS_MODEL_CACHE.clear()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _csv_write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def discover_oxidation_targets(cfg: OxidationConfig) -> tuple[list[StructureTarget], list[str]]:
    """Discover matched relaxed parent and O-vacancy structures from the existing workflow."""
    if not cfg.source_root.exists():
        raise FileNotFoundError(f"Oxidation source root does not exist: {cfg.source_root}")

    from dopingflow.vacancies import discover_selected_parents

    targets: list[StructureTarget] = []
    warnings: list[str] = []
    parents_by_id: dict[str, StructureTarget] = {}

    if cfg.include_vacancy_free:
        try:
            parents = discover_selected_parents(cfg.source_root)
        except (FileNotFoundError, RuntimeError) as exc:
            parents = []
            warnings.append(f"Could not discover vacancy-free selected parents: {exc}")
        for parent in parents:
            parent_id = str(parent["parent_id"])
            target = StructureTarget(
                target_id=parent_id,
                parent_id=parent_id,
                kind="vacancy-free",
                structure_path=Path(parent["relaxed_path"]).resolve(),
                n_vacancies=0,
                vacancy_species=None,
                metadata={
                    "composition": str(parent.get("composition", "")),
                    "candidate": str(parent.get("candidate", "")),
                    "source_candidate_dir": str(parent.get("candidate_dir", "")),
                },
            )
            parents_by_id[parent_id] = target
            targets.append(target)

    if cfg.include_oxygen_vacancies:
        vacancy_db = cfg.source_root / "vacancies_database.json"
        if not vacancy_db.exists():
            warnings.append(
                f"No {vacancy_db.name} found; no oxygen-vacancy targets were added"
            )
        else:
            rows = _json_load(vacancy_db)
            if not isinstance(rows, list):
                raise ValueError(f"Expected a list in {vacancy_db}")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                n_vac = int(row.get("n_vacancies") or 0)
                species = str(row.get("vacancy_species") or "")
                if n_vac <= 0 or species != "O":
                    continue
                relaxed_raw = str(row.get("relaxed_poscar_path") or "").strip()
                if not relaxed_raw:
                    continue
                relaxed = Path(relaxed_raw).expanduser()
                if not relaxed.is_absolute():
                    relaxed = (cfg.root / relaxed).resolve()
                if not relaxed.exists():
                    warnings.append(f"Vacancy result missing relaxed structure: {relaxed}")
                    continue
                parent_id = str(row.get("parent_id") or "")
                config_id = str(row.get("configuration_id") or relaxed.parent.parent.name)
                target = StructureTarget(
                    target_id=f"{parent_id}/V_O_{n_vac:02d}/{config_id}",
                    parent_id=parent_id,
                    kind="oxygen-vacancy",
                    structure_path=relaxed,
                    n_vacancies=n_vac,
                    vacancy_species="O",
                    metadata={
                        "configuration_id": config_id,
                        "search_method": row.get("search_method"),
                        "relaxation_backend": row.get("backend"),
                        "relaxation_model": row.get("model"),
                        "relaxation_task": row.get("task"),
                    },
                )
                targets.append(target)

    # Parent structures are required for valid parent-relative site changes. If
    # vacancy-only analysis was requested, discover parents internally without
    # adding them as output targets.
    if cfg.include_oxygen_vacancies:
        missing_parent_ids = {
            target.parent_id
            for target in targets
            if target.kind == "oxygen-vacancy" and target.parent_id not in parents_by_id
        }
        if missing_parent_ids:
            try:
                all_parents = discover_selected_parents(cfg.source_root)
            except (FileNotFoundError, RuntimeError):
                all_parents = []
            for parent in all_parents:
                parent_id = str(parent["parent_id"])
                if parent_id not in missing_parent_ids:
                    continue
                parents_by_id[parent_id] = StructureTarget(
                    target_id=parent_id,
                    parent_id=parent_id,
                    kind="vacancy-free",
                    structure_path=Path(parent["relaxed_path"]).resolve(),
                    n_vacancies=0,
                    vacancy_species=None,
                    metadata={"internal_parent_reference": True},
                )

    if not targets:
        raise RuntimeError("No structures were discovered for oxidation-state analysis")

    targets.sort(key=lambda target: (target.parent_id, target.n_vacancies, target.target_id))
    return targets, warnings


def base_method_result(
    *,
    method: str,
    target: StructureTarget,
    scope: str,
    status: str = "assigned",
    formal_oxidation_states: Sequence[dict[str, Any]] | None = None,
    bader_partial_charges: Sequence[dict[str, Any]] | None = None,
    orbital_populations: Sequence[dict[str, Any]] | None = None,
    magnetic_moments: Sequence[dict[str, Any]] | None = None,
    coordination: Sequence[dict[str, Any]] | None = None,
    method_scores: Sequence[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    limitations: Sequence[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "target_id": target.target_id,
        "parent_id": target.parent_id,
        "structure_kind": target.kind,
        "n_oxygen_vacancies": target.n_vacancies if target.vacancy_species == "O" else 0,
        "structure_path": str(target.structure_path),
        "method": method,
        "strategy": METHOD_STRATEGY[method],
        "prediction_scope": scope,
        "assignment_status": status,
        "formal_oxidation_states": list(formal_oxidation_states or []),
        "bader_partial_charges": list(bader_partial_charges or []),
        "orbital_populations": list(orbital_populations or []),
        "magnetic_moments": list(magnetic_moments or []),
        "coordination": list(coordination or []),
        "method_scores": list(method_scores or []),
        "provenance": dict(provenance or {}),
        "parent_relative_changes": [],
        "limitations": list(limitations or []),
        "error": error,
    }


def site_records(
    structure: Structure,
    values: Sequence[Any],
    *,
    value_key: str = "formal_oxidation_state",
    status: str = "assigned",
) -> list[dict[str, Any]]:
    if len(structure) != len(values):
        raise ValueError(
            f"Site-resolved result length {len(values)} does not match structure length {len(structure)}"
        )
    return [
        {
            "site_index": index,
            "element": site.specie.symbol,
            value_key: value,
            "assignment_status": status if value is not None else "missing",
        }
        for index, (site, value) in enumerate(zip(structure, values))
    ]


def unsupported_elements(structure: Structure, supported: Iterable[str]) -> list[str]:
    supported_set = {str(item) for item in supported}
    return sorted({site.specie.symbol for site in structure if site.specie.symbol not in supported_set})


def map_child_to_parent_sites(
    parent: Structure,
    child: Structure,
    *,
    tolerance: float,
) -> dict[int, int]:
    """Map surviving child atoms to same-element parent atoms without inventing vacancy sites."""
    if len(child) > len(parent):
        raise ValueError("Child structure contains more atoms than the parent")
    mapping: dict[int, int] = {}
    unused = set(range(len(parent)))
    for child_idx, child_site in enumerate(child):
        candidates = [
            idx for idx in unused if parent[idx].specie.symbol == child_site.specie.symbol
        ]
        if not candidates:
            raise ValueError(f"No parent site available for {child_site.specie.symbol}")
        distances = [
            child.lattice.get_distance_and_image(
                child_site.frac_coords,
                parent[idx].frac_coords,
            )[0]
            for idx in candidates
        ]
        best_pos = min(range(len(candidates)), key=lambda pos: distances[pos])
        if distances[best_pos] > tolerance:
            raise ValueError(
                f"Nearest same-element parent site is {distances[best_pos]:.3f} Å, "
                f"above mapping tolerance {tolerance:.3f} Å"
            )
        mapping[child_idx] = candidates[best_pos]
        unused.remove(candidates[best_pos])
    return mapping


def _method_settings(cfg: OxidationConfig, method: str) -> dict[str, Any]:
    key = method.replace("-", "_")
    value = cfg.settings.get(key, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"[oxidation.{key}] must be a TOML table")
    return value


def _method_runner(method: str) -> Callable[[StructureTarget, OxidationConfig, dict[str, Any]], dict[str, Any]]:
    if method in STRATEGY_METHODS["structural"]:
        from dopingflow.oxidation_structural import run_structural_method

        return lambda target, cfg, settings: run_structural_method(method, target, cfg, settings)
    if method in STRATEGY_METHODS["ml"]:
        from dopingflow.oxidation_ml import run_ml_method

        return lambda target, cfg, settings: run_ml_method(method, target, cfg, settings)
    if method in STRATEGY_METHODS["dft"]:
        from dopingflow.oxidation_dft import run_dft_method

        return lambda target, cfg, settings: run_dft_method(method, target, cfg, settings)
    raise ValueError(method)


def execute_method(
    method: str,
    target: StructureTarget,
    cfg: OxidationConfig,
    *,
    settings_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = dict(_method_settings(cfg, method))
    if settings_override:
        settings.update(settings_override)
    try:
        result = _method_runner(method)(target, cfg, settings)
        result.setdefault("target_id", target.target_id)
        result.setdefault("method", method)
        return result
    except OptionalMethodUnavailable as exc:
        return base_method_result(
            method=method,
            target=target,
            scope="composition-level" if method == "bertos" else "site-resolved",
            status="unavailable",
            provenance={"requested_settings": settings},
            limitations=["Optional method could not be loaded; other requested methods were allowed to continue."],
            error=str(exc),
        )
    except Exception as exc:
        if cfg.fail_fast:
            raise
        log.debug(
            "Oxidation method %s failed for %s",
            method,
            target.target_id,
            exc_info=True,
        )
        return base_method_result(
            method=method,
            target=target,
            scope="composition-level" if method == "bertos" else "site-resolved",
            status="failed",
            provenance={"requested_settings": settings},
            limitations=["This method failed; results from other requested methods remain valid independently."],
            error=f"{type(exc).__name__}: {exc}",
        )


def _formal_by_site(result: dict[str, Any]) -> dict[int, Any]:
    if result.get("prediction_scope") != "site-resolved":
        return {}
    out: dict[int, Any] = {}
    for record in result.get("formal_oxidation_states", []):
        if record.get("site_index") is not None and record.get("formal_oxidation_state") is not None:
            out[int(record["site_index"])] = record["formal_oxidation_state"]
    return out


def _compare_target_results(target_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    target_id = str(target_results[0]["target_id"]) if target_results else ""
    site_methods = [
        result
        for result in target_results
        if result.get("prediction_scope") == "site-resolved"
        and result.get("assignment_status") not in {"failed", "unavailable", "not-run"}
    ]
    composition_methods = [
        result["method"]
        for result in target_results
        if result.get("prediction_scope") == "composition-level"
    ]
    all_site_indices = sorted({idx for result in site_methods for idx in _formal_by_site(result)})
    per_site: list[dict[str, Any]] = []
    n_disagreement = 0
    n_unresolved = 0
    for site_idx in all_site_indices:
        assignments = {
            result["method"]: _formal_by_site(result).get(site_idx)
            for result in site_methods
        }
        present = [value for value in assignments.values() if value is not None]
        if not present:
            status = "unresolved"
            n_unresolved += 1
        elif len(present) != len(assignments):
            status = "missing-assignments"
            n_unresolved += 1
        elif len(set(present)) == 1:
            status = "agreement"
        else:
            status = "disagreement"
            n_disagreement += 1
        per_site.append(
            {
                "site_index": site_idx,
                "status": status,
                "assignments": assignments,
            }
        )
    missing_methods = [
        result["method"]
        for result in target_results
        if result.get("assignment_status") in {"failed", "unavailable", "not-run"}
    ]
    return {
        "target_id": target_id,
        "site_comparison": per_site,
        "n_disagreement_sites": n_disagreement,
        "n_unresolved_sites": n_unresolved,
        "composition_level_methods": composition_methods,
        "missing_or_failed_methods": missing_methods,
        "interpretation": (
            "Agreement/disagreement is reported descriptively only. Oxidation-state labels are not "
            "averaged and no majority vote is treated as physical truth."
        ),
    }


def _add_parent_relative_changes(
    results: list[dict[str, Any]],
    targets: Sequence[StructureTarget],
    cfg: OxidationConfig,
) -> None:
    target_by_id = {target.target_id: target for target in targets}
    parent_targets: dict[str, StructureTarget] = {}
    for target in targets:
        if target.kind == "vacancy-free":
            parent_targets[target.parent_id] = target

    # Parent-relative changes are provided only when both method results exist;
    # this keeps provenance explicit rather than silently adding work.
    by_key = {(result["target_id"], result["method"]): result for result in results}

    structures: dict[str, Structure] = {}
    for result in results:
        target = target_by_id.get(result["target_id"])
        if target is None or target.kind != "oxygen-vacancy":
            continue
        parent_target = parent_targets.get(target.parent_id)
        parent_result = by_key.get((target.parent_id, result["method"]))
        if parent_target is None or parent_result is None:
            result.setdefault("limitations", []).append(
                "Parent-relative changes unavailable because the matched parent was not analyzed with this method."
            )
            continue
        child_formal = _formal_by_site(result)
        parent_formal = _formal_by_site(parent_result)
        if not child_formal or not parent_formal:
            continue
        try:
            parent_structure = structures.setdefault(
                parent_target.target_id, Structure.from_file(parent_target.structure_path)
            )
            child_structure = structures.setdefault(
                target.target_id, Structure.from_file(target.structure_path)
            )
            mapping = map_child_to_parent_sites(
                parent_structure,
                child_structure,
                tolerance=cfg.mapping_tolerance,
            )
        except Exception as exc:
            result.setdefault("limitations", []).append(
                f"Parent-relative atom mapping failed: {type(exc).__name__}: {exc}"
            )
            continue
        changes = []
        for child_idx, parent_idx in mapping.items():
            child_value = child_formal.get(child_idx)
            parent_value = parent_formal.get(parent_idx)
            if child_value is None or parent_value is None:
                continue
            changes.append(
                {
                    "site_index": child_idx,
                    "parent_site_index": parent_idx,
                    "element": child_structure[child_idx].specie.symbol,
                    "formal_oxidation_state": child_value,
                    "parent_formal_oxidation_state": parent_value,
                    "delta_formal_oxidation_state": child_value - parent_value,
                }
            )
        result["parent_relative_changes"] = changes


def _flatten_site_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        base = {
            "target_id": result["target_id"],
            "parent_id": result["parent_id"],
            "structure_kind": result["structure_kind"],
            "method": result["method"],
            "strategy": result["strategy"],
            "prediction_scope": result["prediction_scope"],
            "assignment_status": result["assignment_status"],
        }
        if result["prediction_scope"] == "composition-level":
            for rec in result.get("formal_oxidation_states", []):
                rows.append({**base, **rec})
            if not result.get("formal_oxidation_states"):
                rows.append(base)
            continue

        merged: dict[int, dict[str, Any]] = {}
        descriptor_fields = {
            "formal_oxidation_states": "formal_oxidation_state",
            "bader_partial_charges": "bader_partial_charge",
            "orbital_populations": "orbital_populations",
            "magnetic_moments": "magnetic_moment",
            "coordination": "coordination_number",
        }
        for field_name, default_key in descriptor_fields.items():
            for rec in result.get(field_name, []):
                idx = rec.get("site_index")
                if idx is None:
                    continue
                idx = int(idx)
                merged.setdefault(idx, {"site_index": idx})
                for key, value in rec.items():
                    if key not in {"assignment_status"}:
                        merged[idx][key] = value
                if default_key not in merged[idx] and "value" in rec:
                    merged[idx][default_key] = rec["value"]
        for idx in sorted(merged):
            rows.append({**base, **merged[idx]})
        if not merged:
            rows.append(base)
    return rows



def _safe_output_parts(target_id: str) -> list[str]:
    """Convert a target ID into safe hierarchical output-directory components."""
    parts: list[str] = []
    for raw in str(target_id).replace("\\", "/").split("/"):
        raw = raw.strip()
        if not raw or raw in {".", ".."}:
            continue
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
        parts.append(safe or "target")
    return parts or ["target"]


def _target_output_dir(cfg: OxidationConfig, target: StructureTarget) -> Path:
    return cfg.output_dir / "structures" / Path(*_safe_output_parts(target.target_id))


def _write_per_structure_outputs(
    cfg: OxidationConfig,
    targets: Sequence[StructureTarget],
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write an auditable result package for every analyzed structure."""
    index_rows: list[dict[str, Any]] = []
    successful_statuses = {"assigned", "descriptors-only"}

    for target in targets:
        target_results = [r for r in results if r["target_id"] == target.target_id]
        target_dir = _target_output_dir(cfg, target)
        method_dir = target_dir / "methods"
        target_dir.mkdir(parents=True, exist_ok=True)
        method_dir.mkdir(parents=True, exist_ok=True)

        site_rows = _flatten_site_rows(target_results)
        method_statuses = {
            str(result["method"]): str(result.get("assignment_status", "unknown"))
            for result in target_results
        }
        unresolved = [
            method
            for method, status in method_statuses.items()
            if status not in successful_statuses
        ]
        successful = [
            method
            for method, status in method_statuses.items()
            if status in successful_statuses
        ]
        if unresolved and successful:
            overall_status = "partial"
        elif unresolved:
            overall_status = "unresolved"
        else:
            overall_status = "complete"

        _json_write(target_dir / "oxidation_results.json", target_results)
        _csv_write(target_dir / "oxidation_sites.csv", site_rows)
        for result in target_results:
            _json_write(method_dir / f"{result['method']}.json", result)

        summary = {
            "target_id": target.target_id,
            "parent_id": target.parent_id,
            "structure_kind": target.kind,
            "structure_path": str(target.structure_path),
            "n_oxygen_vacancies": (
                target.n_vacancies if target.vacancy_species == "O" else 0
            ),
            "metadata": target.metadata,
            "overall_status": overall_status,
            "method_statuses": method_statuses,
            "successful_methods": successful,
            "unresolved_methods": unresolved,
            "output_directory": str(target_dir),
        }
        _json_write(target_dir / "summary.json", summary)

        try:
            n_atoms = len(Structure.from_file(target.structure_path))
        except Exception:
            n_atoms = None
        index_rows.append(
            {
                **summary,
                "n_atoms": n_atoms,
                "n_methods": len(target_results),
                "n_successful_methods": len(successful),
                "n_unresolved_methods": len(unresolved),
            }
        )

    return index_rows


def _log_analysis_summary(
    results: Sequence[dict[str, Any]],
    *,
    n_targets: int,
    discovery_warnings: Sequence[str],
) -> None:
    """Report non-assignments once, at the end, without per-structure tracebacks."""
    unresolved = [
        result
        for result in results
        if result.get("assignment_status") not in {"assigned", "descriptors-only"}
    ]
    log.info(
        "Oxidation analysis completed: targets=%d method_results=%d unresolved=%d",
        n_targets,
        len(results),
        len(unresolved),
    )
    for message in discovery_warnings:
        log.warning("Oxidation discovery: %s", message)
    if not unresolved:
        return
    log.warning(
        "Oxidation analysis completed with %d structure/method case(s) without an assignment:",
        len(unresolved),
    )
    for result in unresolved:
        detail = result.get("error") or "; ".join(result.get("limitations", []))
        suffix = f" - {detail}" if detail else ""
        log.warning(
            "  %s [%s]: %s%s",
            result.get("target_id"),
            result.get("method"),
            result.get("assignment_status"),
            suffix,
        )


def run_oxidation(
    raw: dict[str, Any],
    root: Path,
    *,
    strategy_override: str | None = None,
    methods_override: Sequence[str] | str | None = None,
) -> Path:
    cfg = parse_oxidation_config(
        raw,
        root,
        strategy_override=strategy_override,
        methods_override=methods_override,
    )
    if not cfg.enabled:
        raise ValueError("[oxidation].enabled is false; enable it before running oxidation analysis")

    targets, discovery_warnings = discover_oxidation_targets(cfg)
    log.info(
        "Oxidation analysis: strategy=%s methods=%s targets=%d",
        cfg.strategy,
        ",".join(cfg.methods),
        len(targets),
    )

    results: list[dict[str, Any]] = []
    for target in targets:
        for method in cfg.methods:
            results.append(execute_method(method, target, cfg))

    _add_parent_relative_changes(results, targets, cfg)

    comparisons: list[dict[str, Any]] = []
    if cfg.strategy == "combined" or len(cfg.methods) > 1:
        for target in targets:
            target_results = [result for result in results if result["target_id"] == target.target_id]
            comparisons.append(_compare_target_results(target_results))

    followup_candidates: list[dict[str, Any]] = []
    if cfg.dft_followup_enabled:
        candidates = [
            comparison
            for comparison in comparisons
            if comparison["n_disagreement_sites"] > 0
            or comparison["n_unresolved_sites"] > 0
            or comparison["missing_or_failed_methods"]
        ]
        candidates = candidates[: cfg.dft_followup_candidate_limit]
        target_lookup = {target.target_id: target for target in targets}
        for comparison in candidates:
            target = target_lookup[comparison["target_id"]]
            entry = {
                "target_id": target.target_id,
                "reason": "conflicting, unresolved, or missing oxidation-state assignments",
                "candidate_limit": cfg.dft_followup_candidate_limit,
                "execute": cfg.dft_followup_execute,
                "methods": list(cfg.dft_followup_methods),
                "results": [],
            }
            if cfg.dft_followup_execute:
                for method in cfg.dft_followup_methods:
                    # Explicit override is the only path that can permit a new
                    # expensive calculation during follow-up.
                    entry["results"].append(
                        execute_method(
                            method,
                            target,
                            cfg,
                            settings_override={"execute": True, "followup": True},
                        )
                    )
            followup_candidates.append(entry)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = cfg.output_dir / "oxidation_results.json"
    _json_write(results_path, results)
    _json_write(cfg.output_dir / "oxidation_comparison.json", comparisons)
    _json_write(cfg.output_dir / "dft_followup_candidates.json", followup_candidates)
    _csv_write(cfg.output_dir / "oxidation_sites.csv", _flatten_site_rows(results))
    structure_index = _write_per_structure_outputs(cfg, targets, results)
    _csv_write(cfg.output_dir / "oxidation_structure_index.csv", structure_index)
    _json_write(cfg.output_dir / "oxidation_structure_index.json", structure_index)
    _json_write(
        cfg.output_dir / "meta.json",
        {
            "status": "complete",
            "strategy": cfg.strategy,
            "methods": list(cfg.methods),
            "source_root": str(cfg.source_root),
            "output_dir": str(cfg.output_dir),
            "n_targets": len(targets),
            "n_method_results": len(results),
            "target_ids": [target.target_id for target in targets],
            "per_structure_root": str(cfg.output_dir / "structures"),
            "structure_index_csv": str(cfg.output_dir / "oxidation_structure_index.csv"),
            "discovery_warnings": discovery_warnings,
            "dft_followup": {
                "enabled": cfg.dft_followup_enabled,
                "candidate_limit": cfg.dft_followup_candidate_limit,
                "execute": cfg.dft_followup_execute,
                "methods": list(cfg.dft_followup_methods),
            },
            "resolved_config": {
                **asdict(cfg),
                "root": str(cfg.root),
                "source_root": str(cfg.source_root),
                "output_dir": str(cfg.output_dir),
            },
            "interpretation": (
                "Each method is retained separately. Agreement is descriptive; no averaging or "
                "majority voting is used to establish oxidation-state truth."
            ),
        },
    )
    _log_analysis_summary(
        results,
        n_targets=len(targets),
        discovery_warnings=discovery_warnings,
    )
    return results_path


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def run_oxidation_from_toml(
    config_path: Path,
    *,
    strategy_override: str | None = None,
    methods_override: Sequence[str] | str | None = None,
) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_oxidation(
        raw,
        config_path.resolve().parent,
        strategy_override=strategy_override,
        methods_override=methods_override,
    )

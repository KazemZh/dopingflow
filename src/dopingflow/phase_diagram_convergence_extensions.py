"""Phase-diagram convergence guard and optional vacancy-hull extension.

The standard phase-diagram stage reads relaxed candidates from
``results_database.csv``.  This extension keeps its convergence guard and can
optionally add the lowest-energy oxygen-vacancy structure for every vacancy
count from ``vacancy_static_minima.csv`` to the same raw/corrected hull.

Enable the vacancy contribution with::

    [phase_diagram]
    include_vacancy_minima = true
    vacancy_results_directory = "vacancy-selected"

The usual ``phase_diagram_results.csv`` then contains the vacancy entries as
part of the evaluated chemical systems.  A compact vacancy-only table is also
written as ``vacancy_energy_above_hull.csv`` for plotting energy above hull
against the number of oxygen vacancies.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Structure

from dopingflow import phase_diagram as _base

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

log = logging.getLogger(__name__)

_BASE_CANDIDATE_ENTRIES_FROM_DATABASE = _base._candidate_entries_from_database

_ACTIVE_INCLUDE_VACANCY_MINIMA = False
_ACTIVE_VACANCY_ROOT: Path | None = None
_VACANCY_METADATA_BY_CANDIDATE_PATH: dict[str, dict[str, Any]] = {}

VACANCY_MINIMA_CSV = "vacancy_static_minima.csv"
VACANCY_HULL_CSV = "vacancy_energy_above_hull.csv"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_directory(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _vacancy_root_from_config(raw: dict[str, Any], root: Path) -> Path:
    phase = raw.get("phase_diagram", {}) or {}
    explicit = str(phase.get("vacancy_results_directory", "")).strip()
    if explicit:
        return _resolve_directory(root, explicit)

    vacancies = raw.get("vacancies", {}) or {}
    if str(vacancies.get("parent_source", "")).strip() == "directory":
        parent_directory = str(vacancies.get("parent_directory", "")).strip()
        if parent_directory:
            return _resolve_directory(root, parent_directory)

    return root.resolve()


def _vacancy_paths(
    vacancy_root: Path,
    row: dict[str, str],
) -> tuple[Path, Path, Path]:
    """Return candidate directory, relaxed POSCAR, and relaxation metadata."""
    n_vacancies = int(float(row["n_vacancies"]))

    stored_poscar = str(row.get("source_relaxed_poscar_path", "")).strip()
    if stored_poscar:
        stored_path = Path(stored_poscar).expanduser()
        if stored_path.is_file():
            candidate_dir = stored_path.parent.parent
            return candidate_dir, stored_path, stored_path.parent / "meta.json"

    parent_id = str(row.get("source_parent_id", "")).strip()
    if not parent_id:
        raise ValueError(
            "vacancy_static_minima.csv row is missing source_parent_id"
        )

    parent_parts = Path(parent_id).parts
    if not parent_parts:
        raise ValueError(f"Invalid vacancy source_parent_id: {parent_id!r}")

    candidate = parent_parts[-1]
    composition_directory = str(row.get("composition_directory", "")).strip()
    if not composition_directory:
        if len(parent_parts) < 2:
            raise ValueError(
                f"Cannot infer composition directory from source_parent_id={parent_id!r}"
            )
        composition_directory = parent_parts[-2]

    parent_dir = vacancy_root / composition_directory / candidate

    if n_vacancies == 0:
        candidate_dir = parent_dir
        poscar = candidate_dir / "02_relax" / "POSCAR"
        meta = candidate_dir / "02_relax" / "meta.json"
        return candidate_dir, poscar, meta

    vacancy_species = str(row.get("vacancy_species", "O")).strip() or "O"
    config_id = str(row.get("source_configuration_id", "")).strip()
    if not config_id:
        raise ValueError(
            f"Vacancy minimum n={n_vacancies} is missing source_configuration_id"
        )

    candidate_dir = (
        parent_dir
        / "05_vacancies"
        / f"V_{vacancy_species}_{n_vacancies:02d}"
        / config_id
    )
    poscar = candidate_dir / "02_relax" / "POSCAR"
    meta = candidate_dir / "02_relax" / "meta.json"
    return candidate_dir, poscar, meta


def _load_vacancy_minimum_entries(
    vacancy_root: Path,
) -> list[tuple[str, Path, PDEntry, dict[str, Any]]]:
    minima_path = vacancy_root / VACANCY_MINIMA_CSV
    if not minima_path.is_file():
        raise FileNotFoundError(
            f"Missing {minima_path}. Run vacancy thermodynamic analysis first."
        )

    output: list[tuple[str, Path, PDEntry, dict[str, Any]]] = []

    with minima_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = dict(raw_row)
            n_vacancies = int(float(row["n_vacancies"]))
            energy_source = str(row.get("energy_source", "")).strip().lower()
            if energy_source != "relaxed":
                raise ValueError(
                    "Vacancy energy-above-hull analysis requires relaxed minima, but "
                    f"{row.get('actual_composition_key')} n={n_vacancies} uses "
                    f"energy_source={energy_source!r}."
                )
            if not _as_bool(row.get("converged", False)):
                raise ValueError(
                    "Vacancy energy-above-hull analysis requires positively converged "
                    f"relaxations, but {row.get('actual_composition_key')} "
                    f"n={n_vacancies} is not converged."
                )

            energy_text = str(row.get("energy_relaxed_min_eV", "")).strip()
            if not energy_text:
                raise ValueError(
                    "vacancy_static_minima.csv is missing energy_relaxed_min_eV for "
                    f"{row.get('actual_composition_key')} n={n_vacancies}"
                )
            energy = float(energy_text)

            candidate_dir, poscar, meta_path = _vacancy_paths(vacancy_root, row)
            if not poscar.is_file():
                raise FileNotFoundError(
                    f"Relaxed vacancy structure not found: {poscar}"
                )
            if not meta_path.is_file():
                raise FileNotFoundError(
                    f"Vacancy relaxation metadata not found: {meta_path}"
                )

            structure = Structure.from_file(str(poscar))
            relax_meta = json.loads(meta_path.read_text(encoding="utf-8"))

            # Older vacancy relaxations used ``fmax_target`` while the phase-
            # diagram provenance checker expects ``fmax_target_eV_per_A``.
            fmax_target = relax_meta.get(
                "fmax_target_eV_per_A",
                relax_meta.get("fmax_target"),
            )

            attribute = {
                "entry_kind": "vacancy_minimum",
                "structure_path": str(poscar.resolve()),
                "metadata_path": str(meta_path.resolve()),
                "backend": relax_meta.get("backend", row.get("backend")),
                "model": relax_meta.get("model", row.get("model")),
                "task": relax_meta.get("task", row.get("task")),
                "backend_package": relax_meta.get("backend_package"),
                "backend_package_version": relax_meta.get(
                    "backend_package_version"
                ),
                "model_checkpoint_sha256": relax_meta.get(
                    "model_checkpoint_sha256"
                ),
                "optimizer": relax_meta.get("optimizer"),
                "fmax_target_eV_per_A": fmax_target,
                "max_steps": relax_meta.get("max_steps"),
                "converged": _as_bool(relax_meta.get("converged", True)),
                "relaxed_poscar_sha256": relax_meta.get(
                    "relaxed_poscar_sha256"
                ),
                "device": relax_meta.get("device"),
                "gpu_id": relax_meta.get("gpu_id"),
                "n_vacancies": n_vacancies,
                "actual_composition_key": row.get("actual_composition_key", ""),
                "vacancy_percent_of_parent_oxygen": row.get(
                    "vacancy_percent_of_parent_oxygen", ""
                ),
                "source_parent_id": row.get("source_parent_id", ""),
                "source_configuration_id": row.get(
                    "source_configuration_id", ""
                ),
            }

            name = (
                f"vacancy:{row.get('actual_composition_key', '')}:"
                f"n{n_vacancies}:"
                f"{row.get('source_configuration_id', 'parent')}"
            )
            entry = PDEntry(
                structure.composition,
                energy,
                name=name,
                attribute=attribute,
            )

            summary = {
                "actual_composition_key": row.get("actual_composition_key", ""),
                "composition_directory": row.get("composition_directory", ""),
                "n_vacancies": n_vacancies,
                "vacancy_percent_of_parent_oxygen": row.get(
                    "vacancy_percent_of_parent_oxygen", ""
                ),
                "source_parent_id": row.get("source_parent_id", ""),
                "source_configuration_id": row.get(
                    "source_configuration_id", ""
                ),
                "vacancy_structure_path": str(poscar.resolve()),
            }
            output.append((name, candidate_dir, entry, summary))

    return output


def _candidate_entries_from_database_converged(
    root: Path,
) -> list[tuple[str, Path, PDEntry]]:
    entries = _BASE_CANDIDATE_ENTRIES_FROM_DATABASE(root)
    accepted: list[tuple[str, Path, PDEntry]] = []
    rejected: list[str] = []

    for entry_name, candidate_dir, entry in entries:
        attribute = entry.attribute if isinstance(entry.attribute, dict) else {}
        if attribute.get("converged") is not True:
            rejected.append(entry_name)
            continue
        accepted.append((entry_name, candidate_dir, entry))

    if rejected:
        log.warning(
            "Excluded %d non-positively-converged candidate(s) from raw and corrected "
            "phase diagrams: %s. Rerun filtering/collect to remove stale selections "
            "from upstream database outputs.",
            len(rejected),
            rejected,
        )

    _VACANCY_METADATA_BY_CANDIDATE_PATH.clear()

    if _ACTIVE_INCLUDE_VACANCY_MINIMA:
        if _ACTIVE_VACANCY_ROOT is None:
            raise RuntimeError("Vacancy-hull analysis is active without a vacancy root")

        normal_paths = {str(candidate_dir.resolve()) for _, candidate_dir, _ in accepted}
        vacancy_entries = _load_vacancy_minimum_entries(_ACTIVE_VACANCY_ROOT)

        for entry_name, candidate_dir, entry, summary in vacancy_entries:
            candidate_key = str(candidate_dir.resolve())
            _VACANCY_METADATA_BY_CANDIDATE_PATH[candidate_key] = summary

            # The n=0 parent is normally already present in results_database.csv.
            # Reuse that entry instead of inserting a duplicate at the same energy.
            if summary["n_vacancies"] == 0 and candidate_key in normal_paths:
                continue

            accepted.append((entry_name, candidate_dir, entry))

        log.info(
            "Added vacancy minima from %s to the phase-diagram hull (%d minima).",
            _ACTIVE_VACANCY_ROOT,
            len(vacancy_entries),
        )

    return accepted


def _write_vacancy_hull_summary(phase_output: Path, root: Path) -> Path:
    if not _VACANCY_METADATA_BY_CANDIDATE_PATH:
        raise ValueError(
            "Vacancy minima were requested but no vacancy entries were mapped into "
            "the phase-diagram output."
        )

    with phase_output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[dict[str, Any]] = []
    for row in rows:
        candidate_path = str(row.get("candidate_path", "")).strip()
        if not candidate_path:
            continue
        key = str(Path(candidate_path).resolve())
        metadata = _VACANCY_METADATA_BY_CANDIDATE_PATH.get(key)
        if metadata is None:
            continue
        selected.append({**metadata, **row})

    if not selected:
        raise ValueError(
            "Vacancy minima were added to the hull, but no vacancy rows could be "
            "recovered from phase_diagram_results.csv."
        )

    selected.sort(
        key=lambda row: (
            str(row.get("actual_composition_key", "")),
            int(row.get("n_vacancies", 0)),
        )
    )

    preferred = [
        "actual_composition_key",
        "composition_directory",
        "n_vacancies",
        "vacancy_percent_of_parent_oxygen",
        "source_parent_id",
        "source_configuration_id",
        "chemical_system",
        "formula",
        "energy_above_hull_eV_per_atom",
        "energy_above_hull_raw_eV_per_atom",
        "energy_above_hull_corrected_eV_per_atom",
        "stable",
        "stable_raw",
        "stable_corrected",
        "decomposition",
        "decomposition_raw",
        "decomposition_corrected",
        "candidate_path",
        "vacancy_structure_path",
    ]
    present = {key for row in selected for key in row}
    fieldnames = [name for name in preferred if name in present]
    fieldnames.extend(sorted(present - set(fieldnames)))

    output = root / VACANCY_HULL_CSV
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    log.info("DONE vacancy energy-above-hull summary: %s", output)
    return output


def install_extensions() -> None:
    _base._candidate_entries_from_database = _candidate_entries_from_database_converged


def run_phase_diagram_from_toml(config_path: Path) -> Path:
    global _ACTIVE_INCLUDE_VACANCY_MINIMA, _ACTIVE_VACANCY_ROOT

    install_extensions()

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parent
    phase = raw.get("phase_diagram", {}) or {}
    include_vacancies = bool(phase.get("include_vacancy_minima", False))

    if not include_vacancies:
        _ACTIVE_INCLUDE_VACANCY_MINIMA = False
        _ACTIVE_VACANCY_ROOT = None
        _VACANCY_METADATA_BY_CANDIDATE_PATH.clear()
        return _base.run_phase_diagram_from_toml(config_path)

    vacancy_root = _vacancy_root_from_config(raw, root)
    if not vacancy_root.is_dir():
        raise FileNotFoundError(
            f"[phase_diagram].vacancy_results_directory is not a directory: "
            f"{vacancy_root}"
        )

    _ACTIVE_INCLUDE_VACANCY_MINIMA = True
    _ACTIVE_VACANCY_ROOT = vacancy_root

    # A cached phase-diagram file predating vacancy inclusion is not valid for
    # this analysis, so force a rebuild when vacancy minima are requested.
    raw_for_run = dict(raw)
    phase_for_run = dict(phase)
    phase_for_run["skip_if_done"] = False
    raw_for_run["phase_diagram"] = phase_for_run

    try:
        output = _base.run_phase_diagram(
            raw_for_run,
            root,
            config_path=config_path,
        )
        _write_vacancy_hull_summary(output, root)
        return output
    finally:
        _ACTIVE_INCLUDE_VACANCY_MINIMA = False
        _ACTIVE_VACANCY_ROOT = None


install_extensions()

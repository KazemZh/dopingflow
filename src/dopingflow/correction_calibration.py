"""Prepare same-backend calibration energies and fit an energy correction model."""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from dopingflow.calibration_expansion import (
    discover_phase_resolved_oxides,
    fetch_optimade_structure,
)
from dopingflow.correction_model_selection import (
    ModelSelectionConfig,
    ModelSelectionResult,
    select_correction_model_family,
)
from dopingflow.corrections import (
    CALIBRATION_EXPANSION_FILENAME,
    CANDIDATE_MODELS_DIRNAME,
    EXPERIMENTAL_SNAPSHOT_FILENAME,
    FIT_REPORT_FILENAME,
    KINGSBURY_DATASET,
    KINGSBURY_DATASET_SHA256,
    METADATA_FILENAME,
    MODEL_SELECTION_POLICY_VERSION,
    MODEL_SELECTION_FILENAME,
    CorrectionConfig,
    ExperimentalRecord,
    backend_signature_from_reference,
    content_hash,
    correction_activation_hash,
    feature_vector,
    fit_linear_correction_model,
    impute_missing_uncertainties,
    load_correction_model,
    load_experimental_dataset,
    model_directory,
    model_path,
    parse_correction_config,
    save_correction_model,
    validate_backend_compatibility,
    validate_reference_energy_provenance,
)
from dopingflow.refs import (
    REF_JSON,
    _file_sha256,
    _parse_ref_config,
    _per_formula_unit_energy,
    _read_poscar,
    _relax_structure_and_energy,
    _relaxation_signature,
    _write_poscar,
)

log = logging.getLogger(__name__)

CALCULATION_CACHE_FILENAME = "calibration_calculated_energies.json"
REJECTED_FILENAME = "calibration_rejected.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _as_bool(value: Any, *, default: bool = True) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value in calibration manifest: {value!r}")


def _optional_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    result = float(text)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite number in calibration manifest: {value!r}")
    return result


def _resolve_manifest_path(root: Path, manifest: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    # Paths in a manifest are relative to the manifest first.  This makes a
    # portable correction directory possible.  Fall back to project-relative
    # only when the manifest-relative target does not exist.
    local = (manifest.parent / path).resolve()
    project = (root / path).resolve()
    return local if local.exists() or not project.exists() else project


def load_calibration_manifest(config: CorrectionConfig, root: Path) -> list[dict[str, Any]]:
    path = config.calibration_manifest
    if not path.exists():
        if config.calibration_selection == "phase_resolved":
            return []
        raise FileNotFoundError(
            f"Energy correction is enabled but calibration manifest is missing: {path}. "
            "The manifest is separate from [references].oxides_ref."
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        manifest_fields = set(reader.fieldnames or [])
        if "formula" not in manifest_fields:
            raise ValueError(f"Calibration manifest {path} requires a formula column")
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            if not _as_bool(row.get("include"), default=True):
                continue
            formula = str(row.get("formula") or "").strip()
            if not formula:
                raise ValueError(f"{path}:{line_number}: formula is required")
            reduced_formula = Composition(formula).reduced_formula
            if reduced_formula in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate reduced formula {reduced_formula}; "
                    "one calculated polymorph per formula is required"
                )
            seen.add(reduced_formula)
            structure_path = _resolve_manifest_path(
                root,
                path,
                row.get("structure_path") or row.get("structure"),
            )
            energy = _optional_float(row.get("energy_total_eV"))
            if structure_path is None:
                raise ValueError(
                    f"{path}:{line_number}: structure_path is required even when an "
                    "energy is precomputed, so phase and oxygen environment can be verified"
                )
            if not structure_path.exists():
                raise FileNotFoundError(
                    f"{path}:{line_number}: calibration structure not found: {structure_path}"
                )
            structure = Structure.from_file(str(structure_path))
            structure_formula = structure.composition.reduced_formula
            if structure_formula != reduced_formula:
                raise ValueError(
                    f"{path}:{line_number}: manifest formula {reduced_formula} does not "
                    f"match structure composition {structure_formula}"
                )
            analyzer = SpacegroupAnalyzer(structure)
            source_space_group = analyzer.get_space_group_symbol()
            source_crystal_system = analyzer.get_crystal_system()
            declared_space_group = str(row.get("space_group") or "").strip()
            if declared_space_group and declared_space_group != source_space_group:
                raise ValueError(
                    f"{path}:{line_number}: declared space_group "
                    f"{declared_space_group!r} does not match structure-derived "
                    f"{source_space_group!r}"
                )
            records.append(
                {
                    "formula": formula,
                    "reduced_formula": reduced_formula,
                    "phase": str(row.get("phase") or "").strip(),
                    "structure_id": str(
                        row.get("structure_id") or row.get("likely_mpid") or ""
                    ).strip(),
                    "source_space_group": source_space_group,
                    "source_crystal_system": source_crystal_system,
                    "declared_space_group": declared_space_group,
                    "structure_path": structure_path,
                    "energy_total_eV": energy,
                    "e_above_hull_eV_per_atom": _optional_float(
                        row.get("e_above_hull_eV_per_atom")
                        or row.get("e_above_hull")
                    ),
                    "e_above_hull_provenance": str(
                        row.get("e_above_hull_provenance") or ""
                    ).strip(),
                    "e_above_hull_backend": str(
                        row.get("e_above_hull_backend") or ""
                    ).strip(),
                    "e_above_hull_model": str(
                        row.get("e_above_hull_model") or ""
                    ).strip(),
                    "e_above_hull_task": str(
                        row.get("e_above_hull_task") or ""
                    ).strip(),
                    "e_above_hull_backend_version": str(
                        row.get("e_above_hull_backend_version") or ""
                    ).strip(),
                    "e_above_hull_calculation_settings_hash": str(
                        row.get("e_above_hull_calculation_settings_hash") or ""
                    ).strip(),
                    "e_above_hull_identity_columns_present": all(
                        name in manifest_fields
                        for name in (
                            "e_above_hull_backend",
                            "e_above_hull_model",
                            "e_above_hull_task",
                            "e_above_hull_backend_version",
                            "e_above_hull_calculation_settings_hash",
                        )
                    ),
                    "oxide_type": str(row.get("oxide_type") or "").strip().lower(),
                    "backend": str(row.get("backend") or "").strip(),
                    "model": str(row.get("model") or "").strip(),
                    "task": str(row.get("task") or "").strip(),
                    "backend_version": str(row.get("backend_version") or "").strip(),
                    "calculation_settings": str(
                        row.get("calculation_settings") or ""
                    ).strip(),
                    "calculation_settings_hash": str(
                        row.get("calculation_settings_hash") or ""
                    ).strip(),
                    "converged": (
                        _as_bool(row.get("converged"), default=False)
                        if energy is not None
                        else None
                    ),
                    "line_number": line_number,
                }
            )
    if not records:
        raise ValueError(f"Calibration manifest has no included records: {path}")
    return records


def _generic_phase(value: str) -> bool:
    normalized = "".join(char for char in value.lower() if char.isalnum())
    return normalized in {
        "",
        "cr",
        "cryst",
        "crystal",
        "crystalline",
        "solid",
        "none",
        "unknown",
        "na",
    }


def _phase_matches(calculated: Mapping[str, Any], experimental: ExperimentalRecord) -> bool:
    structure_id = str(calculated.get("structure_id") or "").strip().lower()
    experimental_phase = str(experimental.phase or "").strip()
    calculated_phase = str(calculated.get("phase") or "").strip()
    if _generic_phase(experimental_phase):
        return False
    if _generic_phase(calculated_phase):
        return False
    exp_key = "".join(char for char in experimental_phase.lower() if char.isalnum())
    calc_key = "".join(char for char in calculated_phase.lower() if char.isalnum())
    phase_agrees = exp_key == calc_key or exp_key in calc_key or calc_key in exp_key
    if not phase_agrees:
        return False
    crystal_system_aliases = {
        "cubic": "cubic",
        "hex": "hexagonal",
        "hexagonal": "hexagonal",
        "orth": "orthorhombic",
        "orthorhombic": "orthorhombic",
        "monocl": "monoclinic",
        "monoclinic": "monoclinic",
        "tetrag": "tetragonal",
        "tetragonal": "tetragonal",
        "rhomb": "trigonal",
        "rhombohedral": "trigonal",
        "trigonal": "trigonal",
    }
    expected_crystal_system = crystal_system_aliases.get(exp_key)
    if expected_crystal_system is not None and str(
        calculated.get("source_crystal_system") or ""
    ).lower() != expected_crystal_system:
        return False
    try:
        SpaceGroup(experimental_phase)
    except ValueError:
        pass
    else:
        source_key = "".join(
            char
            for char in str(calculated.get("source_space_group") or "").lower()
            if char.isalnum()
        )
        if exp_key != source_key:
            return False
    # When the curated record provides a structure identifier, require the
    # manifest to identify that exact calculated polymorph as well.
    if experimental.likely_mpid:
        return structure_id == experimental.likely_mpid.strip().lower()
    return True


def _match_experimental_record(
    calculated: Mapping[str, Any],
    experimental: Sequence[ExperimentalRecord],
    *,
    allow_phase_mismatch: bool,
) -> tuple[ExperimentalRecord | None, str | None]:
    formula_matches = [
        record
        for record in experimental
        if record.reduced_formula == calculated["reduced_formula"]
    ]
    if not formula_matches:
        return None, "no_experimental_formula_match"
    phase_matches = [
        record for record in formula_matches if _phase_matches(calculated, record)
    ]
    if len(phase_matches) == 1:
        return phase_matches[0], None
    if len(phase_matches) > 1:
        return None, "ambiguous_experimental_phase_match"
    if allow_phase_mismatch and len(formula_matches) == 1:
        return formula_matches[0], "phase_mismatch_explicitly_allowed"
    return None, "phase_not_verified"


def _automatic_manifest_record(
    experimental: ExperimentalRecord,
    structure_path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    structure = Structure.from_file(str(structure_path))
    analyzer = SpacegroupAnalyzer(structure)
    return {
        "formula": experimental.formula,
        "reduced_formula": experimental.reduced_formula,
        "phase": experimental.phase,
        "structure_id": experimental.likely_mpid,
        "source_space_group": analyzer.get_space_group_symbol(),
        "source_crystal_system": analyzer.get_crystal_system(),
        "declared_space_group": "",
        "structure_path": structure_path,
        "energy_total_eV": None,
        "e_above_hull_eV_per_atom": None,
        "e_above_hull_provenance": "",
        "e_above_hull_backend": "",
        "e_above_hull_model": "",
        "e_above_hull_task": "",
        "e_above_hull_backend_version": "",
        "e_above_hull_calculation_settings_hash": "",
        "e_above_hull_identity_columns_present": False,
        "oxide_type": "oxide",
        "backend": "",
        "model": "",
        "task": "",
        "backend_version": "",
        "calculation_settings": "",
        "calculation_settings_hash": "",
        "converged": None,
        "line_number": None,
        "selection_source": "phase_resolved_optimade",
        "expansion_provenance": dict(provenance),
    }


def _materialize_phase_resolved_candidates(
    manifest_records: Sequence[Mapping[str, Any]],
    experimental: Sequence[ExperimentalRecord],
    config: CorrectionConfig,
    output_dir: Path,
    reference_data: Mapping[str, Any],
    relaxation_signature: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    discovery = discover_phase_resolved_oxides(
        experimental,
        config.target_elements,
        excluded_formula_tokens=config.exclude_polyanions,
    )
    manifest_by_formula: dict[str, list[Mapping[str, Any]]] = {}
    for record in manifest_records:
        manifest_by_formula.setdefault(str(record["reduced_formula"]), []).append(record)

    selected: list[dict[str, Any]] = []
    materializations: list[dict[str, Any]] = []
    acquisition_failures: list[dict[str, str]] = []
    structure_base_url = config.optimade_base_url
    if not structure_base_url.rstrip("/").endswith("/structures"):
        structure_base_url = f"{structure_base_url.rstrip('/')}/structures"

    for experimental_record in discovery.accepted_records:
        local_matches = [
            record
            for record in manifest_by_formula.get(
                experimental_record.reduced_formula,
                [],
            )
            if _phase_matches(record, experimental_record)
        ]
        if len(local_matches) > 1:
            raise ValueError(
                "Multiple manifest structures match phase-resolved record "
                f"{experimental_record.reduced_formula}/{experimental_record.likely_mpid}"
            )
        if local_matches:
            selected.append(dict(local_matches[0]))
            materializations.append(
                {
                    "formula": experimental_record.reduced_formula,
                    "likely_mpid": experimental_record.likely_mpid,
                    "source": "phase_verified_manifest",
                    "structure_path": str(local_matches[0]["structure_path"]),
                    "structure_sha256": _file_sha256(
                        Path(local_matches[0]["structure_path"])
                    ),
                }
            )
            continue

        if not config.auto_fetch_phase_structures:
            acquisition_failures.append(
                {
                    "formula": experimental_record.reduced_formula,
                    "likely_mpid": experimental_record.likely_mpid,
                    "reason": "missing_phase_verified_manifest_structure",
                }
            )
            continue
        try:
            fetched = fetch_optimade_structure(
                experimental_record.likely_mpid,
                experimental_record.formula,
                output_dir / "phase_structures",
                base_url=structure_base_url,
            )
        except Exception as exc:
            acquisition_failures.append(
                {
                    "formula": experimental_record.reduced_formula,
                    "likely_mpid": experimental_record.likely_mpid,
                    "reason": f"structure_acquisition_failed:{exc}",
                }
            )
            continue
        provenance = fetched.to_dict()
        provenance.pop("from_cache", None)
        automatic_record = _automatic_manifest_record(
            experimental_record,
            fetched.structure_path,
            provenance,
        )
        reference_entry = (reference_data.get("references", {}) or {}).get(
            experimental_record.reduced_formula
        )
        reused_reference = False
        if isinstance(reference_entry, Mapping):
            source_path = Path(str(reference_entry.get("source_poscar") or ""))
            relaxed_path = Path(str(reference_entry.get("relaxed_poscar") or ""))
            if (
                reference_entry.get("converged") is True
                and source_path.is_file()
                and relaxed_path.is_file()
                and reference_entry.get("E_total_eV") is not None
                and dict(reference_entry.get("relaxation_signature") or {})
                == dict(relaxation_signature)
            ):
                matcher = StructureMatcher(
                    ltol=0.1,
                    stol=0.2,
                    angle_tol=5.0,
                    primitive_cell=True,
                    scale=True,
                    attempt_supercell=True,
                )
                fetched_structure = Structure.from_file(str(fetched.structure_path))
                reference_source = Structure.from_file(str(source_path))
                if matcher.fit(fetched_structure, reference_source):
                    automatic_record = _automatic_manifest_record(
                        experimental_record,
                        relaxed_path,
                        {
                            **provenance,
                            "same_backend_reference_source_path": str(source_path),
                            "same_backend_reference_source_sha256": _file_sha256(
                                source_path
                            ),
                            "same_backend_reference_relaxed_sha256": _file_sha256(
                                relaxed_path
                            ),
                            "structure_matcher": {
                                "ltol": 0.1,
                                "stol": 0.2,
                                "angle_tol": 5.0,
                            },
                        },
                    )
                    automatic_record.update(
                        {
                            "energy_total_eV": float(
                                reference_entry["E_total_eV"]
                            ),
                            "backend": str(reference_data.get("backend") or ""),
                            "model": str(reference_data.get("model") or ""),
                            "task": str(reference_data.get("task") or ""),
                            "backend_version": str(
                                relaxation_signature.get(
                                    "backend_package_version"
                                )
                                or ""
                            ),
                            "calculation_settings": (
                                "reference_energies.json exact-structure reuse"
                            ),
                            "calculation_settings_hash": content_hash(
                                dict(relaxation_signature)
                            ),
                            "converged": True,
                            "selection_source": (
                                "phase_resolved_same_backend_reference_reuse"
                            ),
                        }
                    )
                    reused_reference = True
        selected.append(automatic_record)
        materializations.append(
            {
                "formula": experimental_record.reduced_formula,
                "likely_mpid": experimental_record.likely_mpid,
                "source": (
                    "materials_project_optimade_plus_same_backend_reference_reuse"
                    if reused_reference
                    else "materials_project_optimade"
                ),
                **dict(automatic_record.get("expansion_provenance") or provenance),
            }
        )

    snapshot = {
        "schema_version": 1,
        "policy": "all_strict_phase_resolved_ordinary_oxides_in_target_scope",
        "target_elements": list(config.target_elements),
        "discovery": dict(discovery.report),
        "materialized_count": len(selected),
        "materializations": materializations,
        "acquisition_failures": acquisition_failures,
    }
    if acquisition_failures:
        _write_json(output_dir / CALIBRATION_EXPANSION_FILENAME, snapshot)
        raise ValueError(
            "Could not materialize every phase-resolved calibration oxide. "
            "Provide exact phase-matched manifest structures or enable/fix OPTIMADE "
            f"acquisition. See {output_dir / CALIBRATION_EXPANSION_FILENAME}"
        )
    if not selected:
        _write_json(output_dir / CALIBRATION_EXPANSION_FILENAME, snapshot)
        raise ValueError(
            "No strict phase-resolved ordinary oxides were available for "
            f"target elements {config.target_elements}"
        )
    return selected, snapshot


def _elemental_reference_energies(
    reference_data: Mapping[str, Any],
    expected_relaxation_signature: Mapping[str, Any],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, entry in (reference_data.get("references", {}) or {}).items():
        if not isinstance(entry, Mapping):
            continue
        try:
            composition = Composition(str(name))
        except Exception:
            continue
        if len(composition.elements) != 1:
            continue
        if entry.get("converged") is not True:
            raise ValueError(
                f"Elemental reference {name} lacks positive convergence provenance"
            )
        entry_signature = entry.get("relaxation_signature")
        if entry_signature is None:
            raise ValueError(
                f"Elemental reference {name} lacks relaxation/backend provenance; "
                "rerun refs-build"
            )
        if dict(entry_signature) != dict(expected_relaxation_signature):
            raise ValueError(
                f"Elemental reference {name} has relaxation/backend provenance "
                "incompatible with the correction calibration"
            )
        symbol = composition.elements[0].symbol
        if "E_per_atom_eV" in entry:
            result[symbol] = float(entry["E_per_atom_eV"])
        elif "E_total_eV" in entry and entry.get("n_atoms"):
            result[symbol] = float(entry["E_total_eV"]) / float(entry["n_atoms"])
        elif "E_per_molecule_eV" in entry:
            result[symbol] = float(entry["E_per_molecule_eV"]) / float(
                composition.num_atoms
            )
    return result


def _cache_key(record: Mapping[str, Any], relaxation_signature: Mapping[str, Any]) -> str:
    return content_hash(
        {
            "calculation_cache_schema": 2,
            "formula": record["reduced_formula"],
            "source_sha256": _file_sha256(record["structure_path"]),
            "energy_total_eV": record["energy_total_eV"],
            "phase": record.get("phase"),
            "structure_id": record.get("structure_id"),
            "oxide_type": record.get("oxide_type"),
            "e_above_hull_eV_per_atom": record.get("e_above_hull_eV_per_atom"),
            "e_above_hull_provenance": record.get("e_above_hull_provenance"),
            "e_above_hull_backend": record.get("e_above_hull_backend"),
            "e_above_hull_model": record.get("e_above_hull_model"),
            "e_above_hull_task": record.get("e_above_hull_task"),
            "e_above_hull_backend_version": record.get(
                "e_above_hull_backend_version"
            ),
            "e_above_hull_calculation_settings_hash": record.get(
                "e_above_hull_calculation_settings_hash"
            ),
            "backend": record.get("backend"),
            "model": record.get("model"),
            "task": record.get("task"),
            "backend_version": record.get("backend_version"),
            "calculation_settings": record.get("calculation_settings"),
            "calculation_settings_hash": record.get("calculation_settings_hash"),
            "converged": record.get("converged"),
            "relaxation_signature": dict(relaxation_signature),
        }
    )


def _prepare_calculated_entry(
    record: Mapping[str, Any],
    *,
    ref_config: Any,
    relaxation_signature: Mapping[str, Any],
    output_dir: Path,
    cached: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _cache_key(record, relaxation_signature)
    cached_entry = cached.get(cache_key)
    if isinstance(cached_entry, Mapping):
        relaxed_path = Path(str(cached_entry.get("relaxed_structure_path", "")))
        cached_relaxed_hash = str(cached_entry.get("relaxed_structure_sha256") or "")
        if (
            relaxed_path.exists()
            and cached_relaxed_hash
            and _file_sha256(relaxed_path) == cached_relaxed_hash
        ):
            return dict(cached_entry)

    source_path: Path = record["structure_path"]
    structure = _read_poscar(source_path)
    source_space_group = SpacegroupAnalyzer(structure).get_space_group_symbol()
    energy_total = record["energy_total_eV"]
    if energy_total is None:
        relaxed, energy_total, n_steps, final_fmax, converged = _relax_structure_and_energy(
            structure,
            ref_config,
        )
        source_kind = "relaxed_by_dopingflow"
    else:
        if record.get("converged") is not True:
            raise ValueError(
                "A precomputed calibration energy requires converged=true "
                "geometry provenance"
            )
        expected_identity = {
            "backend": ref_config.backend,
            "model": ref_config.model,
            "task": ref_config.task,
        }
        declared_identity = {
            key: str(record.get(key) or "") for key in expected_identity
        }
        if declared_identity != expected_identity:
            raise ValueError(
                "A precomputed calibration energy requires backend/model/task fields "
                f"matching [references]; declared={declared_identity}, "
                f"expected={expected_identity}"
            )
        if not str(record.get("backend_version") or "").strip():
            raise ValueError(
                "A precomputed calibration energy requires backend_version provenance"
            )
        expected_backend_version = str(
            relaxation_signature.get("backend_package_version") or ""
        )
        if str(record.get("backend_version")) != expected_backend_version:
            raise ValueError(
                "A precomputed calibration energy backend_version does not match "
                f"the active backend package ({record.get('backend_version')!r} != "
                f"{expected_backend_version!r})"
            )
        if not str(record.get("calculation_settings") or "").strip():
            raise ValueError(
                "A precomputed calibration energy requires calculation_settings provenance"
            )
        expected_settings_hash = content_hash(dict(relaxation_signature))
        if str(record.get("calculation_settings_hash") or "") != expected_settings_hash:
            raise ValueError(
                "A precomputed calibration energy requires calculation_settings_hash "
                "matching the active [references] settings; expected "
                f"{expected_settings_hash}"
            )
        relaxed = structure
        n_steps = 0
        final_fmax = None
        converged = True
        source_kind = "precomputed_same_backend_energy"

    relaxed_space_group = SpacegroupAnalyzer(relaxed).get_space_group_symbol()
    if energy_total is not None and source_kind == "relaxed_by_dopingflow":
        if relaxed_space_group != source_space_group:
            raise ValueError(
                f"Calibration relaxation changed the detected space group for "
                f"{record['reduced_formula']}: {source_space_group} -> "
                f"{relaxed_space_group}; phase identity must be reviewed"
            )

    safe_formula = "".join(
        character if character.isalnum() or character in "_.-" else "-"
        for character in str(record["reduced_formula"])
    )
    relaxed_path = (
        output_dir
        / "relaxed_calibration"
        / f"{safe_formula}-{cache_key[:12]}.POSCAR"
    ).resolve()
    _write_poscar(relaxed, relaxed_path)
    energy_per_formula, reduced_composition, n_formula_units = _per_formula_unit_energy(
        relaxed,
        float(energy_total),
    )
    result = {
        "cache_key": cache_key,
        "formula": record["formula"],
        "reduced_formula": record["reduced_formula"],
        "source_structure_path": str(source_path),
        "source_sha256": _file_sha256(source_path),
        "relaxed_structure_path": str(relaxed_path),
        "relaxed_structure_sha256": _file_sha256(relaxed_path),
        "energy_source": source_kind,
        "energy_total_eV": float(energy_total),
        "energy_per_formula_eV": energy_per_formula,
        "n_formula_units": n_formula_units,
        "reduced_composition": reduced_composition,
        "oxide_type_manifest": str(record.get("oxide_type") or ""),
        "source_space_group": source_space_group,
        "relaxed_space_group": relaxed_space_group,
        "e_above_hull_eV_per_atom": record.get("e_above_hull_eV_per_atom"),
        "e_above_hull_provenance": record.get("e_above_hull_provenance"),
        "relaxation_signature": dict(relaxation_signature),
        "precomputed_backend_version": record.get("backend_version"),
        "precomputed_calculation_settings": record.get("calculation_settings"),
        "n_steps": n_steps,
        "final_fmax_eV_per_A": final_fmax,
        "converged": converged,
    }
    return result


def _calculated_formation_energy(
    calculated: Mapping[str, Any],
    elemental_energies: Mapping[str, float],
) -> float:
    composition = Composition(str(calculated["reduced_formula"]))
    missing = [
        element.symbol
        for element in composition.elements
        if element.symbol not in elemental_energies
    ]
    if missing:
        raise ValueError(
            "missing_same_backend_elemental_references:" + ",".join(sorted(missing))
        )
    elemental_total = sum(
        float(amount) * elemental_energies[element.symbol]
        for element, amount in composition.items()
    )
    return float(calculated["energy_per_formula_eV"]) - elemental_total


def _assign_same_backend_hulls(
    provisional: Sequence[Mapping[str, Any]],
    reference_data: Mapping[str, Any],
    elemental_energies: Mapping[str, float],
) -> None:
    """Compute deferred chemical-system hulls using same-backend ML energies."""

    phase_entries: list[tuple[frozenset[str], PDEntry]] = []
    deferred: dict[
        tuple[str, ...],
        list[tuple[dict[str, Any], PDEntry]],
    ] = {}
    for item in provisional:
        calculated = item["calculated"]
        composition = Composition(str(calculated["reduced_formula"]))
        entry = PDEntry(
            composition,
            float(item["calculated_formation"]),
            name=(
                f"calibration:{calculated['reduced_formula']}:"
                f"{calculated.get('cache_key', '')}"
            ),
        )
        symbols = frozenset(element.symbol for element in composition.elements)
        phase_entries.append((symbols, entry))
        if calculated.get("e_above_hull_eV_per_atom") is None:
            deferred.setdefault(tuple(sorted(symbols)), []).append(
                (calculated, entry)
            )

    for name, raw_entry in (reference_data.get("references", {}) or {}).items():
        if not isinstance(raw_entry, Mapping) or raw_entry.get("converged") is not True:
            continue
        reduced_composition = raw_entry.get("reduced_composition")
        energy_per_formula = raw_entry.get("E_per_formula_unit_eV")
        if not isinstance(reduced_composition, Mapping) or energy_per_formula is None:
            continue
        composition = Composition(dict(reduced_composition))
        if len(composition.elements) < 2:
            continue
        if any(
            element.symbol not in elemental_energies
            for element in composition.elements
        ):
            continue
        elemental_total = sum(
            float(amount) * elemental_energies[element.symbol]
            for element, amount in composition.items()
        )
        phase_entries.append(
            (
                frozenset(element.symbol for element in composition.elements),
                PDEntry(
                    composition,
                    float(energy_per_formula) - elemental_total,
                    name=f"reference:{name}",
                ),
            )
        )

    for system, candidates in deferred.items():
        system_set = frozenset(system)
        entries = [
            PDEntry(Composition(symbol), 0.0, name=f"element:{symbol}")
            for symbol in system
        ]
        entries.extend(
            entry
            for entry_symbols, entry in phase_entries
            if entry_symbols.issubset(system_set)
        )
        phase_diagram = PhaseDiagram(entries)
        for calculated, entry in candidates:
            _, e_above_hull = phase_diagram.get_decomp_and_e_above_hull(
                entry,
                allow_negative=True,
            )
            calculated["e_above_hull_eV_per_atom"] = max(
                0.0,
                float(e_above_hull),
            )
            calculated["e_above_hull_provenance"] = (
                "dopingflow same-backend chemical-system hull from phase-resolved "
                "calibration and reference_energies.json"
            )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _calibration_rows_for_terms(
    rows: Sequence[Mapping[str, Any]],
    terms: Sequence[str],
) -> list[dict[str, Any]]:
    featured: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        structure = Structure.from_file(str(row["calculated_structure_path"]))
        vector, matched_terms, detected_oxide_type = feature_vector(
            structure.composition.reduced_composition,
            terms,
            structure=structure,
            known_oxide_type=row.get("oxide_type") or None,
        )
        if not any(vector):
            raise ValueError(
                f"Calibration row {row['formula']} has no features in model {tuple(terms)}"
            )
        row["feature_vector"] = list(vector)
        row["matched_terms"] = list(matched_terms)
        row["oxide_type"] = detected_oxide_type
        featured.append(row)
    return featured


def run_corrections_fit(
    raw_cfg: Mapping[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> Path | None:
    """Fit or reuse the correction model configured in ``[energy_correction]``."""
    config = parse_correction_config(raw_cfg, root)
    if not config.enabled:
        log.info("SKIP energy correction fit: [energy_correction].enabled=false")
        return None

    reference_path = root / REF_JSON
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing same-backend reference energies: {reference_path}. Run refs-build first."
        )
    reference_data = _read_json(reference_path)
    validate_reference_energy_provenance(reference_data)
    backend_signature = backend_signature_from_reference(reference_data, root=root)
    if not backend_signature["backend"]:
        raise ValueError("reference_energies.json is missing backend provenance")

    ref_config = _parse_ref_config(dict(raw_cfg), root)
    relaxation_signature = _relaxation_signature(ref_config, root=root)
    reference_vs_config = {
        "backend": (backend_signature.get("backend"), ref_config.backend),
        "model": (backend_signature.get("model"), ref_config.model),
        "task": (backend_signature.get("task"), ref_config.task),
        "optimizer": (backend_signature.get("optimizer"), ref_config.optimizer),
        "fmax_eV_per_A": (backend_signature.get("fmax_eV_per_A"), ref_config.fmax),
        "max_steps": (backend_signature.get("max_steps"), ref_config.max_steps),
        "device": (backend_signature.get("device"), ref_config.device),
        "gpu_id": (
            backend_signature.get("gpu_id"),
            ref_config.gpu_id if ref_config.device == "cuda" else None,
        ),
        "backend_package_version": (
            backend_signature.get("backend_package_version"),
            relaxation_signature.get("backend_package_version"),
        ),
        "model_checkpoint_sha256": (
            backend_signature.get("model_checkpoint_sha256"),
            relaxation_signature.get("model_checkpoint_sha256"),
        ),
    }
    mismatches = {
        key: {"reference_cache": values[0], "current_config": values[1]}
        for key, values in reference_vs_config.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(
            "[references] no longer matches reference_energies.json; rerun "
            f"refs-build before fitting corrections. Differences: {mismatches}"
        )

    output_dir = model_directory(root, backend_signature)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = load_calibration_manifest(config, root)
    experimental = load_experimental_dataset(config)
    expansion_snapshot: dict[str, Any] | None = None
    if config.calibration_selection == "phase_resolved":
        manifest_records, expansion_snapshot = _materialize_phase_resolved_candidates(
            manifest_records,
            experimental,
            config,
            output_dir,
            reference_data,
            relaxation_signature,
        )
        _write_json(
            output_dir / CALIBRATION_EXPANSION_FILENAME,
            expansion_snapshot,
        )
    elemental_energies = _elemental_reference_energies(
        reference_data,
        relaxation_signature,
    )
    activation_input_hash = correction_activation_hash(config, root)

    cache_path = output_dir / CALCULATION_CACHE_FILENAME
    calculation_cache = _read_json(cache_path) if cache_path.exists() else {}
    updated_cache = dict(calculation_cache)
    prepared: list[tuple[dict[str, Any], ExperimentalRecord, str | None]] = []
    rejected: list[dict[str, Any]] = []

    for manifest_record in manifest_records:
        experimental_record, match_note = _match_experimental_record(
            manifest_record,
            experimental,
            allow_phase_mismatch=config.allow_phase_mismatch,
        )
        if experimental_record is None:
            rejected.append(
                {"formula": manifest_record["formula"], "reason": match_note}
            )
            continue
        if any(
            token in experimental_record.reduced_formula
            or token in experimental_record.formula
            for token in config.exclude_polyanions
        ):
            rejected.append(
                {
                    "formula": manifest_record["formula"],
                    "reason": "excluded_polyanion",
                }
            )
            continue
        relative_uncertainty = None
        if experimental_record.uncertainty_eV_per_atom is not None:
            denominator = abs(experimental_record.formation_enthalpy_eV_per_atom)
            relative_uncertainty = (
                math.inf
                if denominator == 0
                else experimental_record.uncertainty_eV_per_atom / denominator
            )
            if relative_uncertainty > config.max_relative_experimental_uncertainty:
                rejected.append(
                    {
                        "formula": manifest_record["formula"],
                        "reason": "experimental_relative_uncertainty_too_large",
                        "relative_uncertainty": relative_uncertainty,
                    }
                )
                continue
        e_above_hull = manifest_record.get("e_above_hull_eV_per_atom")
        defer_hull = (
            str(manifest_record.get("selection_source") or "").startswith(
                "phase_resolved_"
            )
            and e_above_hull is None
        )
        if (
            config.max_calculated_e_above_hull_eV_per_atom is not None
            and not defer_hull
        ):
            if e_above_hull is None:
                rejected.append(
                    {
                        "formula": manifest_record["formula"],
                        "reason": "missing_calculated_e_above_hull",
                    }
                )
                continue
            if not str(manifest_record.get("e_above_hull_provenance") or "").strip():
                rejected.append(
                    {
                        "formula": manifest_record["formula"],
                        "reason": "missing_same_backend_e_above_hull_provenance",
                    }
                )
                continue
            expected_hull_identity = {
                "backend": ref_config.backend,
                "model": ref_config.model,
                "task": ref_config.task,
                "backend_version": str(
                    relaxation_signature.get("backend_package_version") or ""
                ),
                "calculation_settings_hash": content_hash(
                    dict(relaxation_signature)
                ),
            }
            declared_hull_identity = {
                "backend": str(manifest_record.get("e_above_hull_backend") or ""),
                "model": str(manifest_record.get("e_above_hull_model") or ""),
                "task": str(manifest_record.get("e_above_hull_task") or ""),
                "backend_version": str(
                    manifest_record.get("e_above_hull_backend_version") or ""
                ),
                "calculation_settings_hash": str(
                    manifest_record.get(
                        "e_above_hull_calculation_settings_hash"
                    )
                    or ""
                ),
            }
            if (
                not manifest_record.get("e_above_hull_identity_columns_present")
                or declared_hull_identity != expected_hull_identity
            ):
                rejected.append(
                    {
                        "formula": manifest_record["formula"],
                        "reason": "calculated_e_above_hull_backend_mismatch",
                        "declared": declared_hull_identity,
                        "expected": expected_hull_identity,
                    }
                )
                continue
            if float(e_above_hull) < -1.0e-8:
                rejected.append(
                    {
                        "formula": manifest_record["formula"],
                        "reason": "negative_calculated_e_above_hull",
                        "e_above_hull_eV_per_atom": e_above_hull,
                    }
                )
                continue
            if float(e_above_hull) > config.max_calculated_e_above_hull_eV_per_atom:
                rejected.append(
                    {
                        "formula": manifest_record["formula"],
                        "reason": "calculated_phase_too_unstable",
                        "e_above_hull_eV_per_atom": e_above_hull,
                    }
                )
                continue
        calculated = _prepare_calculated_entry(
            manifest_record,
            ref_config=ref_config,
            relaxation_signature=relaxation_signature,
            output_dir=output_dir,
            cached=calculation_cache,
        )
        updated_cache[calculated["cache_key"]] = calculated
        if (
            calculated.get("energy_source") == "relaxed_by_dopingflow"
            and calculated.get("converged") is not True
        ):
            rejected.append(
                {
                    "formula": manifest_record["formula"],
                    "reason": "calibration_relaxation_not_converged",
                    "final_fmax_eV_per_A": calculated.get("final_fmax_eV_per_A"),
                }
            )
            continue
        prepared.append((calculated, experimental_record, match_note))

    if not prepared:
        _write_json(output_dir / REJECTED_FILENAME, rejected)
        raise ValueError("No calibration records passed formula, phase, and quality filters")

    provisional: list[dict[str, Any]] = []
    for calculated, experimental_record, match_note in prepared:
        try:
            calculated_formation = _calculated_formation_energy(
                calculated,
                elemental_energies,
            )
            structure = Structure.from_file(str(calculated["relaxed_structure_path"]))
            vector, matched_terms, detected_oxide_type = feature_vector(
                structure.composition.reduced_composition,
                config.correction_terms,
                structure=structure,
                known_oxide_type=calculated.get("oxide_type_manifest") or None,
            )
        except (ValueError, FileNotFoundError) as exc:
            rejected.append(
                {"formula": calculated["formula"], "reason": str(exc)}
            )
            continue
        if not any(vector):
            rejected.append(
                {
                    "formula": calculated["formula"],
                    "reason": "calibration_record_has_no_fitted_terms",
                }
            )
            continue

        provisional.append(
            {
                "calculated": calculated,
                "experimental": experimental_record,
                "match_note": match_note,
                "calculated_formation": calculated_formation,
                "vector": vector,
                "matched_terms": matched_terms,
                "detected_oxide_type": detected_oxide_type,
            }
        )

    _assign_same_backend_hulls(
        provisional,
        reference_data,
        elemental_energies,
    )
    hull_filtered: list[dict[str, Any]] = []
    for item in provisional:
        calculated = item["calculated"]
        e_above_hull = calculated.get("e_above_hull_eV_per_atom")
        if (
            config.max_calculated_e_above_hull_eV_per_atom is not None
            and e_above_hull is None
        ):
            rejected.append(
                {
                    "formula": calculated["formula"],
                    "reason": "missing_calculated_e_above_hull",
                }
            )
            continue
        if (
            config.max_calculated_e_above_hull_eV_per_atom is not None
            and float(e_above_hull)
            > config.max_calculated_e_above_hull_eV_per_atom
        ):
            rejected.append(
                {
                    "formula": calculated["formula"],
                    "reason": "calculated_phase_too_unstable",
                    "e_above_hull_eV_per_atom": e_above_hull,
                }
            )
            continue
        hull_filtered.append(item)
    provisional = hull_filtered

    _write_json(cache_path, updated_cache)
    _write_json(output_dir / REJECTED_FILENAME, rejected)
    if not provisional:
        raise ValueError(
            "No calibration records remain after correction-model applicability checks"
        )

    # Impute only from the population that actually enters the fit.  Records
    # rejected for missing elemental references or model inapplicability must
    # not influence the imputed uncertainty.
    imputed_records = impute_missing_uncertainties(
        [item["experimental"] for item in provisional]
    )
    uncertainty_filtered: list[tuple[dict[str, Any], ExperimentalRecord]] = []
    for item, experimental_record in zip(provisional, imputed_records, strict=True):
        uncertainty = experimental_record.uncertainty_eV_per_atom
        assert uncertainty is not None
        denominator = abs(experimental_record.formation_enthalpy_eV_per_atom)
        relative_uncertainty = (
            math.inf if denominator == 0 else uncertainty / denominator
        )
        if relative_uncertainty > config.max_relative_experimental_uncertainty:
            rejected.append(
                {
                    "formula": experimental_record.reduced_formula,
                    "reason": (
                        "imputed_experimental_relative_uncertainty_too_large"
                        if experimental_record.uncertainty_source == "imputed_mean"
                        else "experimental_relative_uncertainty_too_large"
                    ),
                    "relative_uncertainty": relative_uncertainty,
                }
            )
            continue
        uncertainty_filtered.append((item, experimental_record))
    _write_json(output_dir / REJECTED_FILENAME, rejected)
    if not uncertainty_filtered:
        raise ValueError(
            "No calibration records remain after experimental uncertainty filtering"
        )

    calibration_rows: list[dict[str, Any]] = []
    accepted_snapshot: list[dict[str, Any]] = []
    for item, experimental_record in uncertainty_filtered:
        calculated = item["calculated"]
        row = {
            "formula": experimental_record.reduced_formula,
            "feature_vector": list(item["vector"]),
            "matched_terms": list(item["matched_terms"]),
            "oxide_type": item["detected_oxide_type"],
            "experimental_formation_eV_per_formula": (
                experimental_record.formation_enthalpy_eV_per_formula
            ),
            "calculated_formation_eV_per_formula": item["calculated_formation"],
            "experimental_uncertainty_eV_per_formula": (
                experimental_record.uncertainty_eV_per_formula
            ),
            "uncertainty_source": experimental_record.uncertainty_source,
            "calculated_energy_total_eV": calculated["energy_total_eV"],
            "calculated_energy_per_formula_eV": calculated["energy_per_formula_eV"],
            "calculated_structure_path": calculated["relaxed_structure_path"],
            "calculated_structure_sha256": calculated["relaxed_structure_sha256"],
            "source_structure_sha256": calculated["source_sha256"],
            "energy_source": calculated["energy_source"],
            "source_space_group": calculated["source_space_group"],
            "relaxed_space_group": calculated["relaxed_space_group"],
            "e_above_hull_eV_per_atom": calculated["e_above_hull_eV_per_atom"],
            "e_above_hull_provenance": calculated["e_above_hull_provenance"],
            "relaxation_signature": calculated["relaxation_signature"],
            "precomputed_backend_version": calculated[
                "precomputed_backend_version"
            ],
            "precomputed_calculation_settings": calculated[
                "precomputed_calculation_settings"
            ],
            "phase_match_note": item["match_note"] or "matched",
        }
        calibration_rows.append(row)
        snapshot = experimental_record.to_dict()
        snapshot.update(
            {
                "feature_vector": list(item["vector"]),
                "matched_terms": list(item["matched_terms"]),
                "calculated_structure_path": calculated["relaxed_structure_path"],
                "calculated_structure_sha256": calculated[
                    "relaxed_structure_sha256"
                ],
                "calculated_energy_total_eV": calculated["energy_total_eV"],
                "calculated_formation_eV_per_formula": item[
                    "calculated_formation"
                ],
            }
        )
        accepted_snapshot.append(snapshot)

    manifest_snapshot = [
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in record.items()
        }
        for record in manifest_records
    ]
    model_selection: ModelSelectionResult | None = None
    selected_family = "manual"
    selected_terms = tuple(config.correction_terms)
    candidate_terms: dict[str, tuple[str, ...]] = {
        "manual": selected_terms,
    }
    selection_report: dict[str, Any] | None = None
    selection_run_hash = ""
    if config.model_family != "manual":
        selection_config = ModelSelectionConfig(
            min_m0_compounds=max(8, 1 + config.min_degrees_of_freedom),
            min_independent_cation_support=config.min_element_compounds,
            min_unique_oxygen_ratios=config.min_element_stoichiometries,
            max_condition_number=min(config.max_condition_number, 1.0e4),
            min_cv_rmse_improvement_eV_per_atom=(
                config.min_cv_improvement_eV_per_atom
            ),
            require_one_standard_error=config.require_cv_one_standard_error,
        )
        model_selection = select_correction_model_family(
            calibration_rows,
            raw_cfg,
            selection_config=selection_config,
            target_elements=config.m1_elements,
        )
        candidate_terms = {"m0": tuple(model_selection.m0_model.terms)}
        if model_selection.m1_model is not None:
            candidate_terms["m1"] = tuple(model_selection.m1_model.terms)

        if config.model_family == "m0":
            selected_family = "m0"
        elif config.model_family == "m1":
            if model_selection.m1_model is None:
                raise ValueError(
                    "M1 was requested, but the phase-resolved calibration data do "
                    "not satisfy the configured independent support and validation "
                    f"gates: {model_selection.report.get('m1_unavailable_reason')}"
                )
            selected_family = "m1"
        else:
            selected_family = model_selection.selected_family.lower()
        selected_terms = candidate_terms[selected_family]
        selection_report = {
            **dict(model_selection.report),
            "schema_version": 1,
            "requested_model_family": config.model_family,
            "published_model_family": selected_family,
            "forced_family": config.model_family in {"m0", "m1"},
        }
        selection_report["automatic_selected_family"] = str(
            model_selection.report["selected_family"]
        ).lower()
        selection_report["selected_family"] = selected_family
        selection_run_hash = content_hash(
            {
                "selection_policy_version": MODEL_SELECTION_POLICY_VERSION,
                "selection_report": selection_report,
                "activation_input_hash": activation_input_hash,
                "expansion_snapshot": expansion_snapshot,
                "calibration_observations": [
                    {
                        "formula": row["formula"],
                        "oxide_type": row["oxide_type"],
                        "experimental_formation_eV_per_formula": row[
                            "experimental_formation_eV_per_formula"
                        ],
                        "calculated_formation_eV_per_formula": row[
                            "calculated_formation_eV_per_formula"
                        ],
                        "experimental_uncertainty_eV_per_formula": row[
                            "experimental_uncertainty_eV_per_formula"
                        ],
                        "calculated_structure_sha256": row[
                            "calculated_structure_sha256"
                        ],
                    }
                    for row in calibration_rows
                ],
            }
        )
        selection_report["selection_run_hash"] = selection_run_hash

    calibration_rows = _calibration_rows_for_terms(
        calibration_rows,
        selected_terms,
    )
    for snapshot, row in zip(accepted_snapshot, calibration_rows, strict=True):
        snapshot["feature_vector"] = list(row["feature_vector"])
        snapshot["matched_terms"] = list(row["matched_terms"])
        if config.model_family != "manual":
            snapshot["selected_model_family"] = selected_family

    fit_hash_payload = {
            "backend_signature": backend_signature,
            "relaxation_signature": relaxation_signature,
            "correction_terms": selected_terms,
            "calibration_rows": calibration_rows,
            "accepted_experimental_snapshot": accepted_snapshot,
            "experimental_input_hash": content_hash(
                [record.to_dict() for record in experimental]
            ),
            "manifest_snapshot": manifest_snapshot,
            "activation_input_hash": activation_input_hash,
            "experimental_source": config.experimental_source,
            "filtering": {
                "max_relative_uncertainty": config.max_relative_experimental_uncertainty,
                "max_e_above_hull": config.max_calculated_e_above_hull_eV_per_atom,
                "allow_phase_mismatch": config.allow_phase_mismatch,
                "exclude_polyanions": config.exclude_polyanions,
                "min_degrees_of_freedom": config.min_degrees_of_freedom,
                "min_term_support": config.min_term_support,
                "max_condition_number": config.max_condition_number,
                "poor_fit_rmse_warning_eV_per_atom": (
                    config.poor_fit_rmse_warning_eV_per_atom
                ),
            },
        }
    if config.model_family != "manual":
        fit_hash_payload.update(
            {
                "model_family": selected_family,
                "selection_run_hash": selection_run_hash,
                "selection_report": selection_report,
                "expansion_snapshot_hash": (
                    content_hash(expansion_snapshot)
                    if expansion_snapshot is not None
                    else None
                ),
            }
        )
    fit_input_hash = content_hash(fit_hash_payload)
    output_model_path = model_path(root, backend_signature)
    if config.reuse_fitted and output_model_path.exists():
        existing = load_correction_model(output_model_path)
        validate_backend_compatibility(existing, backend_signature)
        required_sidecars = (
            output_dir / FIT_REPORT_FILENAME,
            output_dir / EXPERIMENTAL_SNAPSHOT_FILENAME,
            output_dir / METADATA_FILENAME,
            output_dir / CALCULATION_CACHE_FILENAME,
            output_dir / REJECTED_FILENAME,
        )
        if config.model_family != "manual":
            required_sidecars += (
                output_dir / MODEL_SELECTION_FILENAME,
                *tuple(
                    output_dir / CANDIDATE_MODELS_DIRNAME / f"{family}.json"
                    for family in candidate_terms
                ),
                *tuple(
                    output_dir
                    / CANDIDATE_MODELS_DIRNAME
                    / f"{family}_fit_report.json"
                    for family in candidate_terms
                ),
            )
            if expansion_snapshot is not None:
                required_sidecars += (
                    output_dir / CALIBRATION_EXPANSION_FILENAME,
                )
        if (
            existing.fit_input_hash == fit_input_hash
            and existing.activation_input_hash == activation_input_hash
            and (
                config.model_family == "manual"
                or existing.selection_run_hash == selection_run_hash
            )
            and all(path.exists() for path in required_sidecars)
        ):
            log.info("Correction model cache hit: %s", output_model_path)
            return output_model_path

    accepted_experimental_records = [record for _, record in uncertainty_filtered]
    dataset_names = sorted({record.dataset for record in accepted_experimental_records})
    dataset_version = (
        f"matminer-{_package_version('matminer')}:{KINGSBURY_DATASET}:"
        f"sha256={KINGSBURY_DATASET_SHA256}"
        if any(KINGSBURY_DATASET in name for name in dataset_names)
        else "custom-snapshot"
    )
    fitted_candidates: dict[str, tuple[Any, dict[str, Any]]] = {}
    for family, terms in candidate_terms.items():
        rows = _calibration_rows_for_terms(calibration_rows, terms)
        candidate_fit_hash = (
            fit_input_hash
            if family == selected_family
            else content_hash(
                {
                    "selected_fit_input_hash": fit_input_hash,
                    "candidate_family": family,
                    "correction_terms": terms,
                    "selection_run_hash": selection_run_hash,
                }
            )
        )
        candidate_model, candidate_report = fit_linear_correction_model(
            rows,
            correction_terms=terms,
            backend_signature=backend_signature,
            experimental_dataset="+".join(dataset_names),
            experimental_dataset_version=dataset_version,
            fit_input_hash=candidate_fit_hash,
            min_degrees_of_freedom=config.min_degrees_of_freedom,
            min_term_support=config.min_term_support,
            max_condition_number=config.max_condition_number,
            activation_input_hash=activation_input_hash,
            exclude_polyanions=config.exclude_polyanions,
        )
        if config.model_family != "manual":
            candidate_model = replace(
                candidate_model,
                model_family=family,
                selection_run_hash=selection_run_hash,
                target_elements=config.target_elements,
                selection_metadata={
                    "requested_model_family": config.model_family,
                    "selection_report": MODEL_SELECTION_FILENAME,
                    "selection_run_hash": selection_run_hash,
                    "calibration_expansion_snapshot": (
                        CALIBRATION_EXPANSION_FILENAME
                        if expansion_snapshot is not None
                        else None
                    ),
                    "calibration_expansion_snapshot_hash": (
                        content_hash(expansion_snapshot)
                        if expansion_snapshot is not None
                        else None
                    ),
                },
            )
        fitted_candidates[family] = (candidate_model, candidate_report)

    model, fit_report = fitted_candidates[selected_family]
    fit_report.update(
        {
            "candidate_count": len(manifest_records),
            "accepted_count": len(calibration_rows),
            "rejected_count": len(rejected),
            "rejected_records_file": REJECTED_FILENAME,
            "backend_signature": backend_signature,
            "experimental_uncertainty_statistics_eV_per_atom": {
                "minimum": min(
                    record.uncertainty_eV_per_atom or math.nan
                    for record in accepted_experimental_records
                ),
                "maximum": max(
                    record.uncertainty_eV_per_atom or math.nan
                    for record in accepted_experimental_records
                ),
                "mean": sum(
                    record.uncertainty_eV_per_atom or 0.0
                    for record in accepted_experimental_records
                )
                / len(accepted_experimental_records),
                "imputed_count": sum(
                    record.uncertainty_source == "imputed_mean"
                    for record in accepted_experimental_records
                ),
            },
        }
    )
    if selection_report is not None:
        selection_report["candidate_models"] = {
            family: {
                "fit_id": candidate_model.fit_id,
                "correction_terms": list(candidate_model.correction_terms),
                "fit_metrics": dict(candidate_model.fit_metrics),
                "model_file": f"{CANDIDATE_MODELS_DIRNAME}/{family}.json",
                "fit_report_file": (
                    f"{CANDIDATE_MODELS_DIRNAME}/{family}_fit_report.json"
                ),
            }
            for family, (candidate_model, _) in fitted_candidates.items()
        }
        selection_report_hash = content_hash(selection_report)
        for family, (candidate_model, candidate_report) in tuple(
            fitted_candidates.items()
        ):
            metadata = dict(candidate_model.selection_metadata or {})
            metadata["selection_report_hash"] = selection_report_hash
            fitted_candidates[family] = (
                replace(candidate_model, selection_metadata=metadata),
                candidate_report,
            )
        model, fit_report = fitted_candidates[selected_family]
        fit_report["model_selection"] = {
            "requested_model_family": config.model_family,
            "selected_model_family": selected_family,
            "selection_run_hash": selection_run_hash,
            "selection_report_file": MODEL_SELECTION_FILENAME,
        }
    validation_rmse = {
        key: value
        for key, value in model.fit_metrics.items()
        if key.endswith("rmse_eV_per_atom")
    }
    for symbol, values in fit_report["validation"][
        "leave_element_family_out"
    ].items():
        validation_rmse[f"leave_{symbol}_out_rmse_eV_per_atom"] = values[
            "rmse_eV_per_atom"
        ]
    worst_metric, worst_rmse = max(validation_rmse.items(), key=lambda item: item[1])
    if worst_rmse > config.poor_fit_rmse_warning_eV_per_atom:
        warning = (
            "Correction fit/validation RMSE exceeds the configured warning threshold "
            f"({worst_metric}): {worst_rmse:.6g} > "
            f"{config.poor_fit_rmse_warning_eV_per_atom:.6g} eV/atom"
        )
        log.warning(warning)
        fit_report["quality_warning"] = warning

    if selection_report is not None:
        candidate_dir = output_dir / CANDIDATE_MODELS_DIRNAME
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for family, (candidate_model, candidate_report) in fitted_candidates.items():
            candidate_report.update(
                {
                    "model_family": family,
                    "selection_run_hash": selection_run_hash,
                    "selected_for_application": family == selected_family,
                }
            )
            save_correction_model(candidate_model, candidate_dir / f"{family}.json")
            _write_json(
                candidate_dir / f"{family}_fit_report.json",
                candidate_report,
            )
        _write_json(output_dir / MODEL_SELECTION_FILENAME, selection_report)

    save_correction_model(model, output_model_path)
    _write_json(output_dir / FIT_REPORT_FILENAME, fit_report)
    _write_json(output_dir / EXPERIMENTAL_SNAPSHOT_FILENAME, accepted_snapshot)
    _write_json(
        output_dir / METADATA_FILENAME,
        {
            "fit_id": model.fit_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config_path": str(config_path.resolve()) if config_path else None,
            "backend_signature": backend_signature,
            "relaxation_signature": relaxation_signature,
            "fit_input_hash": fit_input_hash,
            "activation_input_hash": activation_input_hash,
            "model_family": model.model_family,
            "selection_run_hash": model.selection_run_hash,
            "target_elements": list(model.target_elements),
            "selection_metadata": dict(model.selection_metadata or {}),
            "applicability_signature": dict(model.applicability_signature or {}),
            "packages": {
                "dopingflow": _package_version("dopingflow"),
                "pymatgen": _package_version("pymatgen"),
                "matminer": _package_version("matminer"),
                "numpy": _package_version("numpy"),
            },
            "units": dict(model.units),
        },
    )
    log.info("Wrote fitted correction model: %s", output_model_path)
    return output_model_path


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def run_corrections_fit_from_toml(config_path: Path) -> Path | None:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_corrections_fit(
        raw,
        config_path.resolve().parent,
        config_path=config_path,
    )

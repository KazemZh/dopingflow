"""Runtime extensions for automatic correction-calibration expansion.

This module extends the base correction-calibration implementation with four
narrow behaviors:

1. failed structure acquisitions are recorded and skipped instead of aborting
   an otherwise usable calibration run;
2. automatic M1 calibration can gap-fill under-covered workflow cations with
   additional binary Kingsbury oxides carrying a curated ``likely_mpid``;
3. after normal quality filters, elements that remain under-covered trigger one
   second pass exposing every remaining eligible binary Kingsbury fallback; and
4. strict phase identity is verified against an immutable source structure,
   independently of the subsequently relaxed/reused same-backend energy
   geometry.  If the curated OPTIMADE structure is unavailable or incompatible
   with the Kingsbury phase label, the official Materials Project API may be
   used to find a unique phase-compatible structure of the same formula.

Materials Project data are used here only for structure/identity information.
All calibration energies are still evaluated with the active dopingflow ML
backend (or reused from an exact same-backend structure match).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from dopingflow import correction_calibration as _base
from dopingflow.calibration_expansion import (
    discover_phase_resolved_oxides,
    fetch_optimade_structure,
)
from dopingflow.calibration_gap_fill import (
    GapFillSelection,
    select_undercovered_binary_kingsbury_records,
)
from dopingflow.corrections import (
    CALIBRATION_EXPANSION_FILENAME,
    MODEL_SELECTION_FILENAME,
    CorrectionConfig,
    ExperimentalRecord,
    content_hash,
    parse_correction_config,
)
from dopingflow.phase_structure_fallback import (
    fetch_materials_project_phase_structure,
    phase_diagnostics,
    phase_matches_structure,
)
from dopingflow.refs import _file_sha256

log = logging.getLogger(__name__)

_BASE_MATCH_EXPERIMENTAL_RECORD = _base._match_experimental_record
_FORCED_GAP_FILL_ELEMENTS: set[str] = set()

_PHASE_REFERENCE_FIELDS = (
    "phase_match_policy",
    "experimental_likely_mpid",
    "phase_reference_material_id",
    "phase_reference_structure_path",
    "phase_reference_structure_sha256",
    "phase_reference_source",
    "phase_reference_verified",
    "phase_reference_diagnostics",
)


def _gap_fill_identity(record: ExperimentalRecord) -> tuple[str, str]:
    return (
        record.reduced_formula,
        str(record.likely_mpid or "").strip().lower(),
    )


def _annotate_phase_reference(
    record: dict[str, Any],
    experimental_record: ExperimentalRecord,
    *,
    structure_path: Path,
    material_id: str,
    source: str,
) -> dict[str, Any]:
    """Attach immutable source-phase identity independently of energy geometry."""

    structure = Structure.from_file(str(structure_path))
    diagnostics = phase_diagnostics(experimental_record.phase, structure)
    record.update(
        {
            "phase_match_policy": "verified_phase_reference_structure",
            "experimental_likely_mpid": str(experimental_record.likely_mpid or ""),
            "phase_reference_material_id": str(material_id),
            "phase_reference_structure_path": str(structure_path.resolve()),
            "phase_reference_structure_sha256": _file_sha256(structure_path),
            "phase_reference_source": source,
            "phase_reference_verified": bool(diagnostics["matches"]),
            "phase_reference_diagnostics": diagnostics,
            # structure_id identifies the geometry actually used as the immutable
            # phase reference.  For an MP formula-search fallback this may differ
            # from the historical Kingsbury likely_mpid.
            "structure_id": str(material_id),
        }
    )
    return record


def _copy_phase_reference_fields(
    source: Mapping[str, Any],
    destination: dict[str, Any],
) -> None:
    for key in _PHASE_REFERENCE_FIELDS:
        if key in source:
            destination[key] = source[key]
    if "phase_reference_material_id" in source:
        destination["structure_id"] = source["phase_reference_material_id"]


def _match_experimental_record_extended(
    calculated: Mapping[str, Any],
    experimental: Sequence[ExperimentalRecord],
    *,
    allow_phase_mismatch: bool,
) -> tuple[ExperimentalRecord | None, str | None]:
    """Match generic gap-fill and verified-source strict records safely."""

    policy = str(calculated.get("phase_match_policy") or "")
    if policy == "kingsbury_likely_mpid_gap_fill":
        formula_matches = [
            record
            for record in experimental
            if record.reduced_formula == calculated["reduced_formula"]
        ]
        structure_id = str(calculated.get("structure_id") or "").strip().lower()
        id_matches = [
            record
            for record in formula_matches
            if str(record.likely_mpid or "").strip().lower() == structure_id
        ]
        if len(id_matches) == 1:
            return id_matches[0], "phase_unverified_kingsbury_likely_mpid_gap_fill"
        if len(id_matches) > 1:
            return None, "ambiguous_kingsbury_likely_mpid_gap_fill"
        return None, "kingsbury_likely_mpid_gap_fill_not_verified"

    if policy == "verified_phase_reference_structure":
        if calculated.get("phase_reference_verified") is not True:
            return None, "phase_reference_structure_not_verified"
        formula_matches = [
            record
            for record in experimental
            if record.reduced_formula == calculated["reduced_formula"]
        ]
        experimental_id = str(
            calculated.get("experimental_likely_mpid") or ""
        ).strip().lower()
        if experimental_id:
            formula_matches = [
                record
                for record in formula_matches
                if str(record.likely_mpid or "").strip().lower() == experimental_id
            ]
        if len(formula_matches) == 1:
            phase_source = str(calculated.get("phase_reference_source") or "")
            note = (
                "phase_verified_materials_project_formula_fallback"
                if "formula_phase_unique" in phase_source
                else "phase_verified_from_immutable_source_structure"
            )
            return formula_matches[0], note
        if len(formula_matches) > 1:
            return None, "ambiguous_verified_phase_reference_record"
        return None, "verified_phase_reference_experimental_record_not_found"

    return _BASE_MATCH_EXPERIMENTAL_RECORD(
        calculated,
        experimental,
        allow_phase_mismatch=allow_phase_mismatch,
    )


def _mark_gap_fill_record(
    record: dict[str, Any],
    *,
    source: str,
    element_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record["phase_match_policy"] = "kingsbury_likely_mpid_gap_fill"
    record["selection_source"] = source
    provenance = dict(record.get("expansion_provenance") or {})
    provenance.update(
        {
            "gap_fill_policy": "undercovered_binary_kingsbury_likely_mpid",
            "phase_status": "generic_phase_label_likely_mpid_fallback_not_phase_verified",
        }
    )
    if element_requirements is not None:
        provenance["gap_fill_requirements"] = dict(element_requirements)
    record["expansion_provenance"] = provenance
    return record


def _reuse_same_backend_reference_if_matching(
    automatic_record: dict[str, Any],
    experimental_record: ExperimentalRecord,
    fetched_structure_path: Path,
    provenance: Mapping[str, Any],
    reference_data: Mapping[str, Any],
    relaxation_signature: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Reuse a same-backend energy without replacing immutable phase identity."""

    reference_entry = (reference_data.get("references", {}) or {}).get(
        experimental_record.reduced_formula
    )
    if not isinstance(reference_entry, Mapping):
        return automatic_record, False

    source_path = Path(str(reference_entry.get("source_poscar") or ""))
    relaxed_path = Path(str(reference_entry.get("relaxed_poscar") or ""))
    if not (
        reference_entry.get("converged") is True
        and source_path.is_file()
        and relaxed_path.is_file()
        and reference_entry.get("E_total_eV") is not None
        and dict(reference_entry.get("relaxation_signature") or {})
        == dict(relaxation_signature)
    ):
        return automatic_record, False

    matcher = StructureMatcher(
        ltol=0.1,
        stol=0.2,
        angle_tol=5.0,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
    )
    fetched_structure = Structure.from_file(str(fetched_structure_path))
    reference_source = Structure.from_file(str(source_path))
    if not matcher.fit(fetched_structure, reference_source):
        return automatic_record, False

    reused = _base._automatic_manifest_record(
        experimental_record,
        relaxed_path,
        {
            **dict(provenance),
            "same_backend_reference_source_path": str(source_path),
            "same_backend_reference_source_sha256": _file_sha256(source_path),
            "same_backend_reference_relaxed_sha256": _file_sha256(relaxed_path),
            "structure_matcher": {
                "ltol": 0.1,
                "stol": 0.2,
                "angle_tol": 5.0,
            },
        },
    )
    _copy_phase_reference_fields(automatic_record, reused)
    reused.update(
        {
            "energy_total_eV": float(reference_entry["E_total_eV"]),
            "backend": str(reference_data.get("backend") or ""),
            "model": str(reference_data.get("model") or ""),
            "task": str(reference_data.get("task") or ""),
            "backend_version": str(
                relaxation_signature.get("backend_package_version") or ""
            ),
            "calculation_settings": "reference_energies.json exact-structure reuse",
            "calculation_settings_hash": content_hash(dict(relaxation_signature)),
            "converged": True,
            "selection_source": "phase_resolved_same_backend_reference_reuse",
        }
    )
    return reused, True


def _gap_fill_selection(
    discovery_records: Sequence[ExperimentalRecord],
    experimental: Sequence[ExperimentalRecord],
    config: CorrectionConfig,
) -> GapFillSelection:
    if config.model_family not in {"auto", "m1"} or not config.m1_elements:
        return GapFillSelection(
            (),
            {
                "schema_version": 2,
                "policy": "undercovered_workflow_cations_binary_kingsbury_likely_mpid_gap_fill",
                "enabled": False,
                "reason": "gap filling is only needed for automatic/forced M1 model families",
                "selected_count": 0,
                "selected_formulas": [],
            },
        )
    result = select_undercovered_binary_kingsbury_records(
        experimental,
        discovery_records,
        config.m1_elements,
        min_compounds=config.min_element_compounds,
        min_stoichiometries=config.min_element_stoichiometries,
        force_elements=tuple(sorted(_FORCED_GAP_FILL_ELEMENTS)),
    )
    report = dict(result.report)
    report["enabled"] = True
    report["second_pass_forced_elements"] = sorted(_FORCED_GAP_FILL_ELEMENTS)
    return GapFillSelection(result.records, report)


def _materialize_phase_resolved_candidates_extended(
    manifest_records: Sequence[Mapping[str, Any]],
    experimental: Sequence[ExperimentalRecord],
    config: CorrectionConfig,
    output_dir: Path,
    reference_data: Mapping[str, Any],
    relaxation_signature: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize strict records plus targeted under-coverage gap-fill records."""

    discovery = discover_phase_resolved_oxides(
        experimental,
        config.target_elements,
        excluded_formula_tokens=config.exclude_polyanions,
    )
    gap_fill = _gap_fill_selection(discovery.accepted_records, experimental, config)
    gap_fill_ids = {_gap_fill_identity(record) for record in gap_fill.records}
    candidates = list(discovery.accepted_records) + list(gap_fill.records)

    manifest_by_formula: dict[str, list[Mapping[str, Any]]] = {}
    for record in manifest_records:
        manifest_by_formula.setdefault(str(record["reduced_formula"]), []).append(record)

    selected: list[dict[str, Any]] = []
    materializations: list[dict[str, Any]] = []
    acquisition_failures: list[dict[str, Any]] = []
    structure_base_url = config.optimade_base_url
    if not structure_base_url.rstrip("/").endswith("/structures"):
        structure_base_url = f"{structure_base_url.rstrip('/')}/structures"

    gap_requirements = {
        "min_element_compounds": config.min_element_compounds,
        "min_element_stoichiometries": config.min_element_stoichiometries,
    }

    for experimental_record in candidates:
        identity = _gap_fill_identity(experimental_record)
        is_gap_fill = identity in gap_fill_ids
        local_pool = manifest_by_formula.get(experimental_record.reduced_formula, [])
        if is_gap_fill:
            expected_id = str(experimental_record.likely_mpid or "").strip().lower()
            local_matches = [
                record
                for record in local_pool
                if str(record.get("structure_id") or "").strip().lower() == expected_id
            ]
        else:
            local_matches = [
                record
                for record in local_pool
                if _base._phase_matches(record, experimental_record)
            ]

        if len(local_matches) > 1:
            raise ValueError(
                "Multiple manifest structures match calibration record "
                f"{experimental_record.reduced_formula}/{experimental_record.likely_mpid}"
            )
        if local_matches:
            local_record = dict(local_matches[0])
            if is_gap_fill:
                _mark_gap_fill_record(
                    local_record,
                    source="phase_resolved_gap_fill_manifest_likely_mpid",
                    element_requirements=gap_requirements,
                )
                phase_verified = False
                phase_status = "generic_phase_gap_fill_not_verified"
            else:
                local_record = _annotate_phase_reference(
                    local_record,
                    experimental_record,
                    structure_path=Path(local_record["structure_path"]),
                    material_id=str(experimental_record.likely_mpid),
                    source="phase_verified_manifest",
                )
                phase_verified = local_record["phase_reference_verified"] is True
                phase_status = "verified" if phase_verified else "failed"
            selected.append(local_record)
            materializations.append(
                {
                    "formula": experimental_record.reduced_formula,
                    "likely_mpid": experimental_record.likely_mpid,
                    "source": (
                        "gap_fill_likely_mpid_manifest"
                        if is_gap_fill
                        else "phase_verified_manifest"
                    ),
                    "structure_path": str(local_record["structure_path"]),
                    "structure_sha256": _file_sha256(Path(local_record["structure_path"])),
                    "phase_verified": phase_verified,
                    "phase_verification_status": phase_status,
                    "phase_reference": {
                        key: local_record.get(key)
                        for key in _PHASE_REFERENCE_FIELDS
                        if key in local_record
                    },
                }
            )
            continue

        if not config.auto_fetch_phase_structures:
            acquisition_failures.append(
                {
                    "formula": experimental_record.reduced_formula,
                    "likely_mpid": experimental_record.likely_mpid,
                    "selection": "gap_fill" if is_gap_fill else "strict_phase_resolved",
                    "reason": (
                        "missing_likely_mpid_gap_fill_manifest_structure"
                        if is_gap_fill
                        else "missing_phase_verified_manifest_structure"
                    ),
                }
            )
            continue

        fetched_path: Path | None = None
        fetched_material_id = str(experimental_record.likely_mpid or "")
        fetched_source = ""
        provenance: dict[str, Any] = {}
        primary_problem: dict[str, Any] | None = None

        try:
            fetched = fetch_optimade_structure(
                experimental_record.likely_mpid,
                experimental_record.formula,
                output_dir / "phase_structures",
                base_url=structure_base_url,
            )
            provenance = fetched.to_dict()
            provenance.pop("from_cache", None)
            fetched_path = fetched.structure_path
            fetched_source = "materials_project_optimade"
            if not is_gap_fill:
                source_structure = Structure.from_file(str(fetched_path))
                if not phase_matches_structure(experimental_record.phase, source_structure):
                    primary_problem = {
                        "reason": "optimade_structure_phase_mismatch",
                        "diagnostics": phase_diagnostics(
                            experimental_record.phase,
                            source_structure,
                        ),
                    }
                    fetched_path = None
        except Exception as exc:
            primary_problem = {
                "reason": "optimade_structure_acquisition_failed",
                "error": str(exc),
            }

        # Strict records get a second, official MP-API route.  It first retries
        # the curated material ID and then performs a formula search, accepting
        # only one phase-compatible polymorph.  Generic gap-fill records remain
        # tied to their curated likely_mpid and are not phase-searched.
        fallback_problem: str | None = None
        if fetched_path is None and not is_gap_fill:
            try:
                fallback = fetch_materials_project_phase_structure(
                    experimental_record,
                    output_dir / "phase_structures",
                )
                fetched_path = fallback.structure_path
                fetched_material_id = fallback.material_id
                fetched_source = fallback.source
                provenance = fallback.to_dict()
                provenance.pop("from_cache", None)
                if primary_problem is not None:
                    provenance["primary_optimade_problem"] = primary_problem
            except Exception as exc:
                fallback_problem = str(exc)

        if fetched_path is None:
            acquisition_failures.append(
                {
                    "formula": experimental_record.reduced_formula,
                    "likely_mpid": experimental_record.likely_mpid,
                    "selection": "gap_fill" if is_gap_fill else "strict_phase_resolved",
                    "reason": (
                        "structure_acquisition_failed"
                        if is_gap_fill
                        else "phase_compatible_structure_acquisition_failed"
                    ),
                    "optimade": primary_problem,
                    "materials_project_api": fallback_problem,
                }
            )
            continue

        automatic_record = _base._automatic_manifest_record(
            experimental_record,
            fetched_path,
            provenance,
        )
        if not is_gap_fill:
            automatic_record = _annotate_phase_reference(
                automatic_record,
                experimental_record,
                structure_path=fetched_path,
                material_id=fetched_material_id,
                source=fetched_source,
            )
            if automatic_record["phase_reference_verified"] is not True:
                acquisition_failures.append(
                    {
                        "formula": experimental_record.reduced_formula,
                        "likely_mpid": experimental_record.likely_mpid,
                        "selection": "strict_phase_resolved",
                        "reason": "phase_reference_verification_failed_after_acquisition",
                        "phase_reference_material_id": fetched_material_id,
                        "phase_reference_diagnostics": automatic_record[
                            "phase_reference_diagnostics"
                        ],
                    }
                )
                continue

        automatic_record, reused_reference = _reuse_same_backend_reference_if_matching(
            automatic_record,
            experimental_record,
            fetched_path,
            provenance,
            reference_data,
            relaxation_signature,
        )

        if is_gap_fill:
            _mark_gap_fill_record(
                automatic_record,
                source=(
                    "phase_resolved_gap_fill_same_backend_reference_reuse"
                    if reused_reference
                    else "phase_resolved_gap_fill_optimade_likely_mpid"
                ),
                element_requirements=gap_requirements,
            )

        selected.append(automatic_record)
        phase_verified = (
            automatic_record.get("phase_reference_verified") is True
            if not is_gap_fill
            else False
        )
        materializations.append(
            {
                "formula": experimental_record.reduced_formula,
                "likely_mpid": experimental_record.likely_mpid,
                "actual_phase_reference_material_id": (
                    automatic_record.get("phase_reference_material_id")
                    if not is_gap_fill
                    else experimental_record.likely_mpid
                ),
                "source": (
                    "gap_fill_likely_mpid_optimade_plus_same_backend_reference_reuse"
                    if is_gap_fill and reused_reference
                    else (
                        "gap_fill_likely_mpid_optimade"
                        if is_gap_fill
                        else (
                            f"{fetched_source}_plus_same_backend_reference_reuse"
                            if reused_reference
                            else fetched_source
                        )
                    )
                ),
                "phase_verified": phase_verified,
                "phase_verification_status": (
                    "verified"
                    if phase_verified
                    else "generic_phase_gap_fill_not_verified"
                ),
                "phase_reference": {
                    key: automatic_record.get(key)
                    for key in _PHASE_REFERENCE_FIELDS
                    if key in automatic_record
                },
                **dict(automatic_record.get("expansion_provenance") or provenance),
            }
        )

    snapshot = {
        "schema_version": 3,
        "policy": (
            "strict_phase_reference_plus_mp_phase_fallback_plus_"
            "undercoverage_binary_kingsbury_gap_fill"
        ),
        "target_elements": list(config.target_elements),
        "discovery": dict(discovery.report),
        "gap_fill": dict(gap_fill.report),
        "candidate_count": len(candidates),
        "strict_candidate_count": len(discovery.accepted_records),
        "gap_fill_candidate_count": len(gap_fill.records),
        "materialized_count": len(selected),
        "materializations": materializations,
        "acquisition_failures": acquisition_failures,
        "scientific_note": (
            "Strict experimental phase identity is verified from an immutable source "
            "structure and is kept separate from the relaxed/reused ML energy geometry. "
            "If the curated OPTIMADE ID is unavailable or phase-incompatible, the official "
            "Materials Project API may supply a unique formula- and phase-compatible "
            "structure. Materials Project energies are never used in the correction fit. "
            "Generic-phase gap-fill records remain explicitly phase-unverified."
        ),
    }
    if acquisition_failures:
        _base._write_json(output_dir / CALIBRATION_EXPANSION_FILENAME, snapshot)
        log.warning(
            "Could not materialize/phase-verify %d of %d correction-calibration "
            "candidates. Continuing with %d successfully materialized compounds; "
            "failures remain auditable in %s",
            len(acquisition_failures),
            len(candidates),
            len(selected),
            output_dir / CALIBRATION_EXPANSION_FILENAME,
        )
    if not selected:
        _base._write_json(output_dir / CALIBRATION_EXPANSION_FILENAME, snapshot)
        raise ValueError(
            "No strict or gap-fill oxide calibration structures could be materialized "
            f"for target elements {config.target_elements}"
        )
    return selected, snapshot


def install_extensions() -> None:
    """Install the narrow calibration extensions into the base module."""

    _base._match_experimental_record = _match_experimental_record_extended
    _base._materialize_phase_resolved_candidates = (
        _materialize_phase_resolved_candidates_extended
    )


def _coverage_limited_elements(selection_report: Mapping[str, Any]) -> set[str]:
    """Return elements rejected specifically because independent coverage is lacking."""

    result: set[str] = set()
    excluded = selection_report.get("excluded_m1_elements", {}) or {}
    if not isinstance(excluded, Mapping):
        return result
    coverage_reasons = {
        "insufficient_independent_formula_ratio_support",
        "insufficient_unique_oxygen_ratios",
    }
    for element, reasons in excluded.items():
        if isinstance(reasons, str):
            reason_set = {reasons}
        elif isinstance(reasons, Sequence):
            reason_set = {str(reason) for reason in reasons}
        else:
            continue
        if reason_set & coverage_reasons:
            result.add(str(element))
    return result


def _read_selection_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run_corrections_fit(
    raw_cfg: Mapping[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> Path | None:
    """Run the base fit and retry once if final accepted coverage is insufficient."""

    global _FORCED_GAP_FILL_ELEMENTS
    config = parse_correction_config(raw_cfg, root)
    _FORCED_GAP_FILL_ELEMENTS = set()
    try:
        output = _base.run_corrections_fit(
            raw_cfg,
            root,
            config_path=config_path,
        )
        if output is None:
            return None
        if not (
            config.model_family == "auto"
            and config.calibration_selection == "phase_resolved"
            and config.m1_elements
        ):
            return output

        report_path = output.parent / MODEL_SELECTION_FILENAME
        first_report = _read_selection_report(report_path)
        if first_report is None:
            return output
        undercovered = _coverage_limited_elements(first_report)
        if not undercovered:
            return output

        _FORCED_GAP_FILL_ELEMENTS = set(undercovered)
        log.info(
            "Final M1 coverage is insufficient for %s after the first calibration pass. "
            "Searching all remaining eligible binary Kingsbury likely_mpid fallbacks "
            "for those elements and retrying once.",
            sorted(undercovered),
        )
        output = _base.run_corrections_fit(
            raw_cfg,
            root,
            config_path=config_path,
        )

        second_report = _read_selection_report(output.parent / MODEL_SELECTION_FILENAME)
        if second_report is not None:
            still_undercovered = _coverage_limited_elements(second_report)
            if still_undercovered:
                log.warning(
                    "M1 remains under-covered for %s after exhausting eligible binary "
                    "Kingsbury gap-fill candidates and phase-compatible MP structure "
                    "fallbacks. Those cation terms remain unavailable; the model selector "
                    "will retain the scientifically supported family.",
                    sorted(still_undercovered),
                )
        return output
    finally:
        _FORCED_GAP_FILL_ELEMENTS = set()


def run_corrections_fit_from_toml(config_path: Path) -> Path | None:
    raw = _base.tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_corrections_fit(
        raw,
        config_path.resolve().parent,
        config_path=config_path,
    )


install_extensions()

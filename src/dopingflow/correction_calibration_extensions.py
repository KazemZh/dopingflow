"""Runtime extensions for automatic correction-calibration expansion.

This module keeps the large calibration/fitting implementation in
``correction_calibration`` unchanged while extending two narrowly scoped
behaviors used by the CLI:

1. failed structure acquisitions are recorded and skipped instead of aborting
   an otherwise usable calibration run; and
2. automatic M1 calibration can gap-fill under-covered workflow cations with
   additional binary Kingsbury oxides carrying a curated ``likely_mpid``.

Gap-fill records with generic phase labels are explicitly tagged as
phase-unverified ``likely_mpid`` fallbacks.  They still pass through the same
ML-backend relaxation, phase-preservation, oxide classification, same-backend
hull, uncertainty, identifiability, and cross-validation gates as strict
phase-resolved records.

Importing this module installs the two extensions into
``dopingflow.correction_calibration`` before exposing its public fit entry
points.  This is intentionally narrow so existing callers and serialized model
formats remain unchanged.
"""

from __future__ import annotations

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
    CorrectionConfig,
    ExperimentalRecord,
    content_hash,
)
from dopingflow.refs import _file_sha256

log = logging.getLogger(__name__)

_BASE_MATCH_EXPERIMENTAL_RECORD = _base._match_experimental_record


def _gap_fill_identity(record: ExperimentalRecord) -> tuple[str, str]:
    return (
        record.reduced_formula,
        str(record.likely_mpid or "").strip().lower(),
    )


def _match_experimental_record_extended(
    calculated: Mapping[str, Any],
    experimental: Sequence[ExperimentalRecord],
    *,
    allow_phase_mismatch: bool,
) -> tuple[ExperimentalRecord | None, str | None]:
    """Permit only explicitly tagged Kingsbury likely-mpid gap-fill matches."""

    if str(calculated.get("phase_match_policy") or "") == "kingsbury_likely_mpid_gap_fill":
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
    """Reuse an already-relaxed same-backend reference only for an exact structure match."""

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
                "schema_version": 1,
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
    )
    report = dict(result.report)
    report["enabled"] = True
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
    acquisition_failures: list[dict[str, str]] = []
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
                record for record in local_pool if _base._phase_matches(record, experimental_record)
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
                    "phase_verified": not is_gap_fill,
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
                    "selection": "gap_fill" if is_gap_fill else "strict_phase_resolved",
                    "reason": f"structure_acquisition_failed:{exc}",
                }
            )
            continue

        provenance = fetched.to_dict()
        provenance.pop("from_cache", None)
        automatic_record = _base._automatic_manifest_record(
            experimental_record,
            fetched.structure_path,
            provenance,
        )
        automatic_record, reused_reference = _reuse_same_backend_reference_if_matching(
            automatic_record,
            experimental_record,
            fetched.structure_path,
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
        materializations.append(
            {
                "formula": experimental_record.reduced_formula,
                "likely_mpid": experimental_record.likely_mpid,
                "source": (
                    "gap_fill_likely_mpid_optimade_plus_same_backend_reference_reuse"
                    if is_gap_fill and reused_reference
                    else (
                        "gap_fill_likely_mpid_optimade"
                        if is_gap_fill
                        else (
                            "materials_project_optimade_plus_same_backend_reference_reuse"
                            if reused_reference
                            else "materials_project_optimade"
                        )
                    )
                ),
                "phase_verified": not is_gap_fill,
                **dict(automatic_record.get("expansion_provenance") or provenance),
            }
        )

    snapshot = {
        "schema_version": 2,
        "policy": "strict_phase_resolved_plus_undercoverage_binary_kingsbury_gap_fill",
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
            "Gap-fill records with generic experimental phase labels are not treated as "
            "phase verified. Their curated Kingsbury likely_mpid is used only to obtain a "
            "candidate geometry; all normal same-backend quality and validation gates remain active."
        ),
    }
    if acquisition_failures:
        _base._write_json(output_dir / CALIBRATION_EXPANSION_FILENAME, snapshot)
        log.warning(
            "Could not materialize %d of %d correction-calibration candidates. "
            "Continuing with %d successfully materialized compounds; failures remain "
            "auditable in %s",
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


install_extensions()

# Public API mirrors the base module after installing the extended behavior.
run_corrections_fit = _base.run_corrections_fit
run_corrections_fit_from_toml = _base.run_corrections_fit_from_toml

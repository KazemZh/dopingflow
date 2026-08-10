"""Targeted Kingsbury gap filling for under-covered correction chemistries.

The strict phase-resolved calibration expansion remains the preferred source of
calibration compounds.  This module adds a conservative fallback for automatic
M1 model selection: when a workflow cation lacks enough independent oxide
formulas/stoichiometries, search the full Kingsbury experimental dataset for
additional *binary* oxides of that cation whose phase label is generic but that
still carry a curated ``likely_mpid`` association.

These records are not silently promoted to phase-verified data.  Callers must
mark them as a ``likely_mpid`` phase fallback in provenance and retain the
normal same-backend relaxation, oxide-classification, hull, uncertainty, rank,
and cross-validation filters before they may influence a fitted model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pymatgen.core import Composition

from dopingflow.calibration_expansion import (
    NON_ORDINARY_BINARY_OXIDE_FORMULAS,
    is_generic_phase,
)
from dopingflow.corrections import KINGSBURY_DATASET, ExperimentalRecord

_MATERIAL_ID_RE = re.compile(r"^mp-[0-9]+$", re.IGNORECASE)


@dataclass(frozen=True)
class GapFillSelection:
    """Selected fallback records and an auditable coverage report."""

    records: tuple[ExperimentalRecord, ...]
    report: Mapping[str, Any]


def _composition(record: ExperimentalRecord) -> Composition:
    return Composition(record.formula)


def _contains_element(record: ExperimentalRecord, element: str) -> bool:
    return float(_composition(record).get(element, 0.0)) > 0.0


def _oxygen_ratio(record: ExperimentalRecord, element: str) -> float:
    composition = _composition(record)
    amount = float(composition.get(element, 0.0))
    if amount <= 0.0:
        raise ValueError(f"{record.formula} does not contain {element}")
    return round(float(composition.get("O", 0.0)) / amount, 12)


def _coverage(records: Sequence[ExperimentalRecord], element: str) -> dict[str, Any]:
    relevant = [record for record in records if _contains_element(record, element)]
    formulas = sorted({record.reduced_formula for record in relevant})
    ratios = sorted({_oxygen_ratio(record, element) for record in relevant})
    return {
        "unique_formulas": len(formulas),
        "formulas": formulas,
        "unique_oxygen_ratios": len(ratios),
        "oxygen_ratios": ratios,
    }


def _record_identity(record: ExperimentalRecord) -> tuple[str, str]:
    """Return the stable experimental/material identity used for strict-pool skips."""

    return (
        record.reduced_formula,
        str(record.likely_mpid or "").strip().lower(),
    )


def _is_binary_target_oxide(record: ExperimentalRecord, element: str) -> bool:
    try:
        composition = _composition(record)
    except Exception:
        return False
    symbols = {item.symbol for item in composition.elements}
    return symbols == {element, "O"}


def _is_kingsbury_record(record: ExperimentalRecord) -> bool:
    dataset = str(record.dataset or "")
    return dataset == KINGSBURY_DATASET or KINGSBURY_DATASET in dataset


def _candidate_sort_key(
    record: ExperimentalRecord,
    element: str,
    current_ratios: set[float],
) -> tuple[Any, ...]:
    ratio = _oxygen_ratio(record, element)
    uncertainty = record.uncertainty_eV_per_atom
    uncertainty_key = (
        float(uncertainty)
        if uncertainty is not None and math.isfinite(float(uncertainty))
        else math.inf
    )
    # Prefer a new independent O/cation stoichiometry, then a lower reported
    # experimental uncertainty, then deterministic chemistry/provenance keys.
    return (
        0 if ratio not in current_ratios else 1,
        uncertainty_key,
        ratio,
        record.reduced_formula,
        str(record.likely_mpid).lower(),
    )


def select_undercovered_binary_kingsbury_records(
    experimental: Sequence[ExperimentalRecord],
    strict_records: Sequence[ExperimentalRecord],
    target_elements: Sequence[str],
    *,
    min_compounds: int,
    min_stoichiometries: int,
    force_elements: Sequence[str] = (),
) -> GapFillSelection:
    """Select generic-phase binary Kingsbury oxides for under-covered cations.

    The function intentionally does not loosen the fitted-model gates.  It only
    supplies additional candidate structures to the existing calibration
    pipeline.  A fallback record must:

    * come from the Kingsbury dataset;
    * be a binary ``M-O`` compound for the target cation;
    * have a generic/missing phase label (otherwise the strict selector should
      already have considered it);
    * have a syntactically valid curated ``likely_mpid``;
    * not be one of the known peroxide/superoxide binary formulas; and
    * add a reduced formula not already present in the strict calibration pool.

    Normally candidates are chosen only until the configured formula and
    independent-stoichiometry targets are met.  ``force_elements`` is used by a
    second pass after the *final* quality filters: for those still-undercovered
    elements, every remaining eligible binary Kingsbury fallback is exposed so
    the fitting stage gets one last scientifically auditable chance to improve
    coverage.
    """

    if min_compounds < 1:
        raise ValueError("min_compounds must be >= 1")
    if min_stoichiometries < 1:
        raise ValueError("min_stoichiometries must be >= 1")

    normalized_targets = tuple(
        sorted({str(value).strip() for value in target_elements if str(value).strip()})
    )
    forced = {str(value).strip() for value in force_elements if str(value).strip()}
    outside_targets = sorted(forced - set(normalized_targets))
    if outside_targets:
        raise ValueError(
            "force_elements must be a subset of target_elements; outside target scope: "
            f"{outside_targets}"
        )

    selected: list[ExperimentalRecord] = []
    per_element: dict[str, dict[str, Any]] = {}
    strict_identities = {_record_identity(record) for record in strict_records}

    for element in normalized_targets:
        before = _coverage(strict_records, element)
        current_formulas = set(before["formulas"])
        current_ratios = set(float(value) for value in before["oxygen_ratios"])
        force_all_candidates = element in forced

        needs_gap_fill = (
            force_all_candidates
            or len(current_formulas) < min_compounds
            or len(current_ratios) < min_stoichiometries
        )
        candidates: list[ExperimentalRecord] = []
        rejected_counts: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

        if needs_gap_fill:
            seen_candidate_formulas: set[str] = set()
            for record in experimental:
                if not _is_kingsbury_record(record):
                    continue
                if not _is_binary_target_oxide(record, element):
                    continue
                # The strict records are the baseline, not failed gap-fill
                # candidates. Skip their exact material identities silently so
                # rejection accounting only describes *additional* records.
                if _record_identity(record) in strict_identities:
                    continue
                if record.reduced_formula in current_formulas:
                    reject("already_in_strict_pool")
                    continue
                if record.reduced_formula in seen_candidate_formulas:
                    reject("duplicate_reduced_formula")
                    continue
                if not is_generic_phase(record.phase):
                    # A non-generic phase with a usable mp-id should already have
                    # been discovered by the strict phase-resolved selector.  Do
                    # not create a second path that can disagree with it.
                    reject("non_generic_phase_belongs_to_strict_selector")
                    continue
                material_id = str(record.likely_mpid or "").strip()
                if not _MATERIAL_ID_RE.fullmatch(material_id):
                    reject("missing_or_invalid_likely_mpid")
                    continue
                if (
                    str(record.formula).replace(" ", "")
                    in NON_ORDINARY_BINARY_OXIDE_FORMULAS
                ):
                    reject("known_peroxide_or_superoxide_formula")
                    continue
                seen_candidate_formulas.add(record.reduced_formula)
                candidates.append(record)

        chosen: list[ExperimentalRecord] = []
        remaining = list(candidates)
        while remaining and (
            force_all_candidates
            or len(current_formulas) < min_compounds
            or len(current_ratios) < min_stoichiometries
        ):
            remaining.sort(
                key=lambda record: _candidate_sort_key(
                    record,
                    element,
                    current_ratios,
                )
            )
            record = remaining.pop(0)
            chosen.append(record)
            selected.append(record)
            current_formulas.add(record.reduced_formula)
            current_ratios.add(_oxygen_ratio(record, element))

        after = {
            "unique_formulas": len(current_formulas),
            "formulas": sorted(current_formulas),
            "unique_oxygen_ratios": len(current_ratios),
            "oxygen_ratios": sorted(current_ratios),
        }
        per_element[element] = {
            "undercovered_before_gap_fill": (
                before["unique_formulas"] < min_compounds
                or before["unique_oxygen_ratios"] < min_stoichiometries
            ),
            "forced_after_final_filter_undercoverage": force_all_candidates,
            "required_unique_formulas": int(min_compounds),
            "required_unique_oxygen_ratios": int(min_stoichiometries),
            "coverage_before": before,
            "eligible_gap_fill_candidates": [
                {
                    "formula": record.reduced_formula,
                    "phase": record.phase,
                    "likely_mpid": record.likely_mpid,
                    "oxygen_per_cation": _oxygen_ratio(record, element),
                    "uncertainty_eV_per_atom": record.uncertainty_eV_per_atom,
                }
                for record in sorted(
                    candidates,
                    key=lambda record: (
                        _oxygen_ratio(record, element),
                        record.reduced_formula,
                        str(record.likely_mpid).lower(),
                    ),
                )
            ],
            "selected_gap_fill_records": [
                {
                    "formula": record.reduced_formula,
                    "phase": record.phase,
                    "likely_mpid": record.likely_mpid,
                    "oxygen_per_cation": _oxygen_ratio(record, element),
                    "uncertainty_eV_per_atom": record.uncertainty_eV_per_atom,
                }
                for record in chosen
            ],
            "coverage_after_candidate_selection": after,
            "candidate_target_satisfied": (
                after["unique_formulas"] >= min_compounds
                and after["unique_oxygen_ratios"] >= min_stoichiometries
            ),
            "rejection_counts": dict(sorted(rejected_counts.items())),
            "note": (
                "Coverage here is pre-calculation. Same-backend relaxation, phase-preservation, "
                "ordinary-oxide classification, hull, uncertainty, rank, and CV filters still apply."
            ),
        }

    # A reduced formula must occur only once in model-family selection.  If the
    # same fallback formula were selected while filling two target elements,
    # retain it once; this can only arise for malformed/non-binary input, but
    # the deterministic guard keeps the API robust.
    unique_selected: list[ExperimentalRecord] = []
    seen: set[tuple[str, str]] = set()
    for record in selected:
        key = (record.reduced_formula, str(record.likely_mpid).lower())
        if key in seen:
            continue
        seen.add(key)
        unique_selected.append(record)

    report = {
        "schema_version": 2,
        "policy": "undercovered_workflow_cations_binary_kingsbury_likely_mpid_gap_fill",
        "phase_status": "generic_phase_label_likely_mpid_fallback_not_phase_verified",
        "target_elements": list(normalized_targets),
        "forced_elements": sorted(forced),
        "minimum_unique_formulas": int(min_compounds),
        "minimum_unique_oxygen_ratios": int(min_stoichiometries),
        "selected_count": len(unique_selected),
        "selected_formulas": [record.reduced_formula for record in unique_selected],
        "per_element": per_element,
    }
    return GapFillSelection(tuple(unique_selected), report)

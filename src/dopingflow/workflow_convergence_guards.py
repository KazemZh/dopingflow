"""Convergence guards shared by relax, filter, and formation stages.

Historically the relax worker returned ``status='ok'`` whenever the optimizer
completed without raising, even if it exhausted ``max_steps`` before reaching
the requested force threshold.  That allowed an unconverged structure to enter
``ranking_relax.csv`` and ``selected_candidates.txt``.  Formation-energy
correction provenance is intentionally stricter and exposed the inconsistency.

This compatibility layer makes the scientific semantics explicit:

* new relaxations that do not positively converge are labelled
  ``not_converged`` rather than ``ok``;
* filtering requires positive ``converged=true`` metadata, including when
  consuming older ranking files;
* formation defensively ignores unconverged candidates from stale selection
  files instead of aborting an otherwise valid composition folder.

The relaxed geometry/energy are retained on disk for diagnosis; they are simply
not admitted to thermodynamic ranking or correction application.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dopingflow import filtering as _filtering
from dopingflow import formation as _formation
from dopingflow import relax as _relax

log = logging.getLogger(__name__)

_BASE_RELAX_ONE_CANDIDATE = _relax._relax_one_candidate
_BASE_READ_RANKING_RELAX = _filtering._read_ranking_relax
_BASE_GET_CANDIDATE_POSCARS = _formation._get_candidate_poscars
_WARNED_FORMATION_PATHS: set[str] = set()


def _read_positive_convergence(meta_path: Path) -> tuple[bool, dict[str, Any] | None]:
    if not meta_path.is_file():
        return False, None
    try:
        value = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    if not isinstance(value, dict):
        return False, None
    return value.get("converged") is True, value


def _relax_one_candidate_convergence_aware(job: Any) -> dict[str, Any]:
    result = dict(_BASE_RELAX_ONE_CANDIDATE(job))
    if result.get("status") == "ok" and result.get("converged") is not True:
        result["status"] = "not_converged"
        final_fmax = result.get("final_fmax_eV_per_A")
        steps = result.get("optimizer_steps")
        result["error"] = (
            "optimizer_completed_without_positive_convergence"
            f"; final_fmax_eV_per_A={final_fmax}; optimizer_steps={steps}"
        )
    return result


def _read_ranking_relax_converged_only(path: Path) -> list[dict[str, Any]]:
    rows = _BASE_READ_RANKING_RELAX(path)
    kept: list[dict[str, Any]] = []
    rejected: list[str] = []
    folder = path.parent

    for row in rows:
        candidate = str(row.get("candidate") or "").strip()
        meta_path = folder / candidate / "02_relax" / "meta.json"
        converged, metadata = _read_positive_convergence(meta_path)
        if converged:
            kept.append(row)
            continue
        rejected.append(candidate or "<unknown>")
        if metadata is None:
            reason = "missing/unreadable relaxation metadata"
        else:
            reason = (
                "converged is not true"
                f" (final_fmax={metadata.get('final_fmax_eV_per_A')}, "
                f"target={metadata.get('fmax_target_eV_per_A')}, "
                f"steps={metadata.get('optimizer_steps')}, "
                f"max_steps={metadata.get('max_steps')})"
            )
        log.warning(
            "FILTER %s/%s: exclude unconverged relaxation: %s",
            folder.name,
            candidate,
            reason,
        )

    if rejected:
        log.info(
            "FILTER %s: excluded %d unconverged candidate(s): %s",
            folder.name,
            len(rejected),
            rejected,
        )
    if not kept:
        raise RuntimeError(
            f"No positively converged relaxation rows remain in {path}. "
            "Increase [relax].max_steps, adjust the optimizer/fmax if scientifically "
            "appropriate, and rerun the affected relaxations."
        )
    return kept


def _get_candidate_poscars_converged_only(folder: Path) -> list[Path]:
    paths = _BASE_GET_CANDIDATE_POSCARS(folder)
    kept: list[Path] = []
    skipped: list[str] = []

    for poscar in paths:
        meta_path = poscar.parent / "meta.json"
        converged, metadata = _read_positive_convergence(meta_path)
        if converged:
            kept.append(poscar)
            continue
        candidate = poscar.parents[1].name
        skipped.append(candidate)
        warning_key = str(meta_path.resolve())
        if warning_key not in _WARNED_FORMATION_PATHS:
            _WARNED_FORMATION_PATHS.add(warning_key)
            log.warning(
                "FORMATION %s/%s: skip unconverged relaxation "
                "(final_fmax=%s, target=%s, steps=%s, max_steps=%s)",
                folder.name,
                candidate,
                None if metadata is None else metadata.get("final_fmax_eV_per_A"),
                None if metadata is None else metadata.get("fmax_target_eV_per_A"),
                None if metadata is None else metadata.get("optimizer_steps"),
                None if metadata is None else metadata.get("max_steps"),
            )

    if skipped:
        log.info(
            "FORMATION %s: using %d positively converged candidate(s); skipped %d "
            "unconverged candidate(s): %s",
            folder.name,
            len(kept),
            len(skipped),
            skipped,
        )
    return kept


def install_extensions() -> None:
    _relax._relax_one_candidate = _relax_one_candidate_convergence_aware
    _filtering._read_ranking_relax = _read_ranking_relax_converged_only
    _formation._get_candidate_poscars = _get_candidate_poscars_converged_only


install_extensions()

"""Convergence guard for phase-diagram candidate entries.

The phase diagram must never use an energy from a relaxation that did not
positively reach the requested convergence criterion.  Older workflow outputs
could contain such candidates in ``results_database.csv`` because the relax
stage historically recorded an optimizer-completed calculation as ``status=ok``
even when ``converged=false``.

This extension filters those stale entries at the final consumer boundary so
both raw and corrected hulls are built from the same positively converged
candidate set.  Reference entries are unaffected.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pymatgen.analysis.phase_diagram import PDEntry

from dopingflow import phase_diagram as _base

log = logging.getLogger(__name__)

_BASE_CANDIDATE_ENTRIES_FROM_DATABASE = _base._candidate_entries_from_database


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

    return accepted


def install_extensions() -> None:
    _base._candidate_entries_from_database = _candidate_entries_from_database_converged


def run_phase_diagram_from_toml(config_path: Path) -> Path:
    install_extensions()
    return _base.run_phase_diagram_from_toml(config_path)


install_extensions()

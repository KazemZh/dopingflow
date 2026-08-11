from pathlib import Path

from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Composition

from dopingflow import phase_diagram_convergence_extensions as ext


def _entry(name: str, converged):
    return (
        name,
        Path(name),
        PDEntry(
            Composition("SnO2"),
            -10.0,
            name=name,
            attribute={"entry_kind": "candidate", "converged": converged},
        ),
    )


def test_phase_diagram_excludes_non_positive_convergence(monkeypatch, tmp_path):
    rows = [
        _entry("good", True),
        _entry("bad", False),
        _entry("legacy_missing", None),
    ]
    monkeypatch.setattr(ext, "_BASE_CANDIDATE_ENTRIES_FROM_DATABASE", lambda root: rows)

    result = ext._candidate_entries_from_database_converged(tmp_path)

    assert [name for name, _, _ in result] == ["good"]

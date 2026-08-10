import csv
import json

import pytest
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Composition

import dopingflow.phase_diagram as phase_module


def test_phase_diagram_honors_skip_if_done(tmp_path):
    output = tmp_path / "phase_diagram_results.csv"
    output.write_text("existing\n", encoding="utf-8")

    result = phase_module.run_phase_diagram(
        {"phase_diagram": {"skip_if_done": True}},
        tmp_path,
    )

    assert result == output
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_raw_phase_diagram_cache_hit_removes_obsolete_system_outputs(tmp_path):
    output = tmp_path / "phase_diagram_results.csv"
    output.write_text(
        "chemical_system,candidate\nO-Sn,candidate_001\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase_diagrams"
    output_dir.mkdir()
    current = output_dir / "phase_diagram_O-Sn.csv"
    obsolete = output_dir / "phase_diagram_O-Sb-Sn.csv"
    current.write_text("current\n", encoding="utf-8")
    obsolete.write_text("obsolete\n", encoding="utf-8")

    result = phase_module.run_phase_diagram(
        {"phase_diagram": {"skip_if_done": True}},
        tmp_path,
    )

    assert result == output
    assert current.exists()
    assert not obsolete.exists()


def test_phase_diagram_writes_one_file_per_exact_system_and_uses_threshold(
    tmp_path, monkeypatch
):
    reference_path = tmp_path / phase_module.REF_JSON
    reference_path.parent.mkdir(parents=True)
    reference_path.write_text(json.dumps({}), encoding="utf-8")

    references = [
        PDEntry(Composition("Sn"), -1.0, name="Sn"),
        PDEntry(Composition("O2"), -2.0, name="O2"),
    ]
    candidate_dir = tmp_path / "Sn10" / "candidate_001"
    candidate = PDEntry(Composition("SnO2"), -3.0, name="Sn10/candidate_001")
    obsolete_output = (
        tmp_path / "phase_diagrams" / "phase_diagram_O-Sb-Sn.csv"
    )
    obsolete_output.parent.mkdir(parents=True)
    obsolete_output.write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(phase_module, "_reference_entries_from_ref", lambda ref: references)
    monkeypatch.setattr(
        phase_module,
        "_candidate_entries_from_database",
        lambda root: [(candidate.name, candidate_dir, candidate)],
    )

    class FakePhaseDiagram:
        def __init__(self, entries):
            self.entries = entries

        def get_e_above_hull(self, entry):
            return 0.02

        def get_decomposition(self, composition):
            return {candidate: 1.0}

    monkeypatch.setattr(phase_module, "PhaseDiagram", FakePhaseDiagram)

    output = phase_module.run_phase_diagram(
        {
            "phase_diagram": {
                "skip_if_done": False,
                "stable_threshold_eV_per_atom": 0.05,
            }
        },
        tmp_path,
    )

    system_output = tmp_path / "phase_diagrams" / "phase_diagram_O-Sn.csv"
    assert output.exists()
    assert system_output.exists()
    assert not obsolete_output.exists()

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["chemical_system"] == "O-Sn"
    assert rows[0]["stable"] == "True"


def test_phase_diagram_rejects_negative_stability_threshold(tmp_path):
    with pytest.raises(ValueError, match="must be non-negative"):
        phase_module.run_phase_diagram(
            {"phase_diagram": {"stable_threshold_eV_per_atom": -0.01}},
            tmp_path,
        )

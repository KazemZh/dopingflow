from pathlib import Path
import sys

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.gui_config import CHOICES, DEFAULTS, STEP_KEYS
from gui.io_project import ProjectIndex


def test_vacancy_gui_defaults_are_one_flat_section():
    assert STEP_KEYS.count("vacancies") == 1
    section = DEFAULTS["vacancies"]
    assert all(not isinstance(value, dict) for value in section.values())
    assert CHOICES["vacancies.parent_source"] == ["selected_candidates", "directory"]
    assert CHOICES["vacancies.enumeration_mode"] == ["auto", "exact", "sample"]


def test_vacancy_toml_round_trip_stays_flat():
    dumped = toml.dumps({"vacancies": DEFAULTS["vacancies"]})
    assert "[vacancies]" in dumped
    assert "[vacancies." not in dumped
    loaded = toml.loads(dumped)
    assert loaded["vacancies"]["oxidation_state_values"] == [
        [3, 5], [5], [3], [5], [3, 4], [3, 4, 5], [2, 3, 4]
    ]


def test_project_index_detects_vacancy_results(tmp_path: Path):
    outdir = tmp_path / "random_structures"
    root = outdir / "Sb5" / "candidate_001" / "05_vacancies"
    group = root / "V_O_01"
    group.mkdir(parents=True)
    (root / "vacancy_results.csv").write_text("configuration_id\n", encoding="utf-8")
    (group / "ranking_scan.csv").write_text("configuration_id\n", encoding="utf-8")
    (outdir / "vacancies_database.csv").write_text("parent_id\n", encoding="utf-8")
    project = ProjectIndex(root=tmp_path, outdir=outdir)
    assert project.vacancy_database() == outdir / "vacancies_database.csv"
    assert project.vacancy_parents("Sb5") == ["candidate_001"]
    assert "V_O_01/ranking_scan.csv" in project.vacancy_rankings("Sb5", "candidate_001")

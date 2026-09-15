import csv
import json
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

import dopingflow.phase_diagram_convergence_extensions as ext


def _write_relaxed_structure(path: Path, formula: str, energy: float) -> None:
    structure = Structure(
        Lattice.cubic(5.0),
        ["Sn", "O", "O"] if formula == "SnO2" else ["Sn", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]][
            : 3 if formula == "SnO2" else 2
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Poscar(structure).write_file(path)
    (path.parent / "meta.json").write_text(
        json.dumps(
            {
                "energy_relaxed_total_eV": energy,
                "energy_relaxed_eV": energy,
                "backend": "mace",
                "model": "mh-1",
                "task": "matpes_r2scan",
                "optimizer": "bfgs",
                "fmax_target": 0.05,
                "fmax_target_eV_per_A": 0.05,
                "max_steps": 300,
                "converged": True,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )


def test_load_vacancy_minimum_entries_reconstructs_local_paths(tmp_path):
    parent = tmp_path / "Sb2p5" / "candidate_001"
    parent_poscar = parent / "02_relax" / "POSCAR"
    parent_reference_poscar = (
        parent
        / "05_vacancies"
        / "parent_reference"
        / "relaxed"
        / "POSCAR"
    )
    vacancy_poscar = (
        parent
        / "05_vacancies"
        / "V_O_01"
        / "config_0001"
        / "02_relax"
        / "POSCAR"
    )
    _write_relaxed_structure(parent_poscar, "SnO2", -10.2)
    _write_relaxed_structure(parent_reference_poscar, "SnO2", -10.0)
    _write_relaxed_structure(vacancy_poscar, "SnO", -8.0)
    (parent / "05_vacancies" / "parent_reference" / "source.json").write_text(
        json.dumps({"parent_relaxation_reused": False, "parent_converged": True}),
        encoding="utf-8",
    )

    minima = tmp_path / ext.VACANCY_MINIMA_CSV
    with minima.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "actual_composition_key",
                "composition_directory",
                "n_vacancies",
                "vacancy_species",
                "vacancy_percent_of_parent_oxygen",
                "source_parent_id",
                "source_configuration_id",
                "source_relaxed_poscar_path",
                "energy_source",
                "energy_relaxed_min_eV",
                "converged",
                "backend",
                "model",
                "task",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "actual_composition_key": "Sb2.5",
                "composition_directory": "Sb2p5",
                "n_vacancies": 0,
                "vacancy_species": "O",
                "vacancy_percent_of_parent_oxygen": 0.0,
                "source_parent_id": "Sb2p5/candidate_001",
                "source_configuration_id": "parent_reference",
                "source_relaxed_poscar_path": "/stale/remote/parent_reference/relaxed/POSCAR",
                "energy_source": "relaxed",
                "energy_relaxed_min_eV": -10.0,
                "converged": True,
                "backend": "mace",
                "model": "mh-1",
                "task": "matpes_r2scan",
            }
        )
        writer.writerow(
            {
                "actual_composition_key": "Sb2.5",
                "composition_directory": "Sb2p5",
                "n_vacancies": 1,
                "vacancy_species": "O",
                "vacancy_percent_of_parent_oxygen": 50.0,
                "source_parent_id": "Sb2p5/candidate_001",
                "source_configuration_id": "config_0001",
                "source_relaxed_poscar_path": "/stale/remote/vac/POSCAR",
                "energy_source": "relaxed",
                "energy_relaxed_min_eV": -8.0,
                "converged": True,
                "backend": "mace",
                "model": "mh-1",
                "task": "matpes_r2scan",
            }
        )

    entries = ext._load_vacancy_minimum_entries(tmp_path)

    assert len(entries) == 2
    assert entries[0][3]["n_vacancies"] == 0
    assert entries[0][1] == parent
    assert entries[0][2].attribute["structure_path"] == str(
        parent_reference_poscar.resolve()
    )
    assert entries[1][3]["n_vacancies"] == 1
    assert entries[1][1].name == "config_0001"
    assert entries[1][2].composition.reduced_formula == "SnO"
    assert entries[1][2].attribute["fmax_target_eV_per_A"] == 0.05


def test_write_vacancy_hull_summary_adds_vacancy_metadata(tmp_path):
    candidate_dir = tmp_path / "Sb2p5" / "candidate_001"
    candidate_dir.mkdir(parents=True)
    key = str(candidate_dir.resolve())
    ext._VACANCY_METADATA_BY_CANDIDATE_PATH.clear()
    ext._VACANCY_METADATA_BY_CANDIDATE_PATH[key] = {
        "actual_composition_key": "Sb2.5",
        "composition_directory": "Sb2p5",
        "n_vacancies": 0,
        "vacancy_percent_of_parent_oxygen": "0.0",
        "source_parent_id": "Sb2p5/candidate_001",
        "source_configuration_id": "parent_reference",
        "vacancy_structure_path": str(
            candidate_dir / "02_relax" / "POSCAR"
        ),
    }

    phase_output = tmp_path / "phase_diagram_results.csv"
    with phase_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "chemical_system",
                "candidate_path",
                "formula",
                "energy_above_hull_eV_per_atom",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "chemical_system": "O-Sn",
                "candidate_path": key,
                "formula": "SnO2",
                "energy_above_hull_eV_per_atom": 0.02,
            }
        )

    output = ext._write_vacancy_hull_summary(phase_output, tmp_path)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["actual_composition_key"] == "Sb2.5"
    assert rows[0]["n_vacancies"] == "0"
    assert rows[0]["energy_above_hull_eV_per_atom"] == "0.02"

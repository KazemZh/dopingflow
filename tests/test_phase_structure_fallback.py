from __future__ import annotations

from types import SimpleNamespace

from pymatgen.core import Lattice, Structure

from dopingflow.corrections import KINGSBURY_DATASET, ExperimentalRecord
from dopingflow.phase_structure_fallback import (
    expected_crystal_system,
    phase_diagnostics,
    phase_matches_structure,
    select_unique_phase_compatible_document,
)


def _record(formula: str, phase: str, material_id: str = "mp-1") -> ExperimentalRecord:
    structure = _structure_for_formula(formula, "cubic")
    n_atoms = float(structure.composition.reduced_composition.num_atoms)
    return ExperimentalRecord(
        formula=formula,
        reduced_formula=structure.composition.reduced_formula,
        formation_enthalpy_eV_per_atom=-1.0,
        formation_enthalpy_eV_per_formula=-n_atoms,
        uncertainty_eV_per_atom=0.01,
        uncertainty_eV_per_formula=0.01 * n_atoms,
        uncertainty_source="reported",
        phase=phase,
        temperature="298 K",
        source="synthetic",
        likely_mpid=material_id,
        doi="",
        reference_id="",
        notes="",
        dataset=KINGSBURY_DATASET,
        original_units="eV/atom",
    )


def _structure_for_formula(formula: str, crystal_system: str) -> Structure:
    from pymatgen.core import Composition

    composition = Composition(formula)
    species: list[str] = []
    for element, amount in composition.items():
        species.extend([element.symbol] * int(round(float(amount))))
    n = len(species)
    coords = [[i / n, i / n, i / n] for i in range(n)]
    if crystal_system == "cubic":
        lattice = Lattice.cubic(5.0)
    elif crystal_system == "tetragonal":
        lattice = Lattice.tetragonal(5.0, 7.0)
    elif crystal_system == "orthorhombic":
        lattice = Lattice.orthorhombic(5.0, 6.0, 7.0)
    else:
        raise ValueError(crystal_system)
    return Structure(lattice, species, coords)


def test_coarse_kingsbury_phase_aliases_map_to_crystal_systems():
    assert expected_crystal_system("cubic") == "cubic"
    assert expected_crystal_system("tetrag") == "tetragonal"
    assert expected_crystal_system("orth") == "orthorhombic"
    assert expected_crystal_system("hex") == "hexagonal"


def test_phase_verification_accepts_mno_like_cubic_source_and_rejects_wrong_system():
    cubic = _structure_for_formula("MnO", "cubic")
    tetragonal = _structure_for_formula("MnO", "tetragonal")

    assert phase_matches_structure("cubic", cubic) is True
    assert phase_matches_structure("cubic", tetragonal) is False

    diagnostics = phase_diagnostics("cubic", cubic)
    assert diagnostics["matches"] is True
    assert diagnostics["detected_crystal_system"] == "cubic"


def test_generic_phase_labels_are_not_promoted_to_verified_phase_identity():
    cubic = _structure_for_formula("TiO", "cubic")
    assert phase_matches_structure("solid", cubic) is False
    assert phase_matches_structure("cr", cubic) is False


def test_unique_formula_phase_candidate_is_selected_without_energy_ranking():
    record = _record("NbO", "cubic")
    cubic = SimpleNamespace(
        material_id="mp-cubic",
        structure=_structure_for_formula("NbO", "cubic"),
    )
    tetragonal = SimpleNamespace(
        material_id="mp-tetragonal",
        structure=_structure_for_formula("NbO", "tetragonal"),
    )

    selected, compatible_ids = select_unique_phase_compatible_document(
        [tetragonal, cubic],
        record,
    )

    assert selected is cubic
    assert compatible_ids == ("mp-cubic",)


def test_multiple_phase_compatible_candidates_are_left_ambiguous():
    # Use a simple rocksalt-like 1:1 composition so the synthetic structures
    # retain cubic symmetry under SpacegroupAnalyzer.
    record = _record("MnO", "cubic")
    first = SimpleNamespace(
        material_id="mp-1",
        structure=_structure_for_formula("MnO", "cubic"),
    )
    second = SimpleNamespace(
        material_id="mp-2",
        structure=_structure_for_formula("MnO", "cubic"),
    )

    selected, compatible_ids = select_unique_phase_compatible_document(
        [first, second],
        record,
    )

    assert selected is None
    assert compatible_ids == ("mp-1", "mp-2")

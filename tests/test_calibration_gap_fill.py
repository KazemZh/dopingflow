from __future__ import annotations

from pymatgen.core import Composition

from dopingflow.calibration_gap_fill import (
    select_undercovered_binary_kingsbury_records,
)
from dopingflow.corrections import KINGSBURY_DATASET, ExperimentalRecord


def _record(
    formula: str,
    *,
    phase: str,
    material_id: str,
    uncertainty: float | None = 0.01,
    dataset: str = KINGSBURY_DATASET,
) -> ExperimentalRecord:
    reduced = Composition(formula).reduced_formula
    n_atoms = float(Composition(reduced).num_atoms)
    return ExperimentalRecord(
        formula=formula,
        reduced_formula=reduced,
        formation_enthalpy_eV_per_atom=-1.0,
        formation_enthalpy_eV_per_formula=-n_atoms,
        uncertainty_eV_per_atom=uncertainty,
        uncertainty_eV_per_formula=(
            uncertainty * n_atoms if uncertainty is not None else None
        ),
        uncertainty_source="reported" if uncertainty is not None else "missing",
        phase=phase,
        temperature="298 K",
        source="synthetic",
        likely_mpid=material_id,
        doi="",
        reference_id="",
        notes="",
        dataset=dataset,
        original_units="eV/atom",
    )


def test_gap_fill_adds_generic_phase_binary_oxide_with_new_stoichiometry():
    strict = [
        _record("Mn2O3", phase="orthorhombic", material_id="mp-1"),
        _record("MnO2", phase="tetragonal", material_id="mp-2"),
    ]
    experimental = strict + [
        _record("MnO", phase="solid", material_id="mp-3"),
        _record("Mn3O4", phase="cr", material_id="mp-4", uncertainty=0.02),
    ]

    result = select_undercovered_binary_kingsbury_records(
        experimental,
        strict,
        ["Mn"],
        min_compounds=3,
        min_stoichiometries=3,
    )

    assert [record.reduced_formula for record in result.records] == ["MnO"]
    mn = result.report["per_element"]["Mn"]
    assert mn["undercovered_before_gap_fill"] is True
    assert mn["coverage_before"]["unique_formulas"] == 2
    assert mn["coverage_before"]["unique_oxygen_ratios"] == 2
    assert mn["coverage_after_candidate_selection"]["unique_formulas"] == 3
    assert mn["coverage_after_candidate_selection"]["unique_oxygen_ratios"] == 3
    assert mn["candidate_target_satisfied"] is True


def test_gap_fill_does_not_duplicate_strict_or_use_non_generic_phase_records():
    strict = [
        _record("Ti2O3", phase="rhombohedral", material_id="mp-10"),
        _record("TiO2", phase="tetragonal", material_id="mp-11"),
    ]
    experimental = strict + [
        _record("Ti2O3", phase="solid", material_id="mp-12"),
        _record("TiO", phase="cubic", material_id="mp-13"),
        _record("Ti3O5", phase="crystal", material_id="not-an-mpid"),
    ]

    result = select_undercovered_binary_kingsbury_records(
        experimental,
        strict,
        ["Ti"],
        min_compounds=3,
        min_stoichiometries=3,
    )

    assert result.records == ()
    ti = result.report["per_element"]["Ti"]
    assert ti["candidate_target_satisfied"] is False
    assert ti["rejection_counts"]["already_in_strict_pool"] == 1
    assert ti["rejection_counts"]["non_generic_phase_belongs_to_strict_selector"] == 1
    assert ti["rejection_counts"]["missing_or_invalid_likely_mpid"] == 1


def test_gap_fill_is_independent_of_reference_list_and_scopes_by_target_element():
    strict = [_record("In2O3", phase="cubic", material_id="mp-20")]
    experimental = strict + [
        _record("InO", phase="solid", material_id="mp-21"),
        _record("In2O", phase="crystalline", material_id="mp-22"),
        _record("SnO", phase="solid", material_id="mp-23"),
    ]

    result = select_undercovered_binary_kingsbury_records(
        experimental,
        strict,
        ["In"],
        min_compounds=3,
        min_stoichiometries=3,
    )

    formulas = {record.reduced_formula for record in result.records}
    assert formulas == {"InO", "In2O"}
    assert "SnO" not in formulas
    assert result.report["per_element"]["In"]["candidate_target_satisfied"] is True


def test_gap_fill_skips_element_when_strict_coverage_already_satisfies_targets():
    strict = [
        _record("Ce2O3", phase="hexagonal", material_id="mp-30"),
        _record("CeO2", phase="cubic", material_id="mp-31"),
        _record("CeO", phase="tetragonal", material_id="mp-32"),
    ]
    experimental = strict + [
        _record("Ce3O4", phase="solid", material_id="mp-33"),
    ]

    result = select_undercovered_binary_kingsbury_records(
        experimental,
        strict,
        ["Ce"],
        min_compounds=3,
        min_stoichiometries=3,
    )

    assert result.records == ()
    ce = result.report["per_element"]["Ce"]
    assert ce["undercovered_before_gap_fill"] is False
    assert ce["eligible_gap_fill_candidates"] == []

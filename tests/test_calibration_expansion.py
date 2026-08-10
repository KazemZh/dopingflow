from __future__ import annotations

import pytest

from dopingflow.calibration_expansion import (
    discover_phase_resolved_binary_oxides,
    discover_phase_resolved_oxides,
    fetch_optimade_structure,
    infer_scoped_elements,
)
from dopingflow.corrections import ExperimentalRecord


def _record(
    formula: str,
    *,
    phase: str = "cubic",
    material_id: str = "mp-1",
    value: float = -1.0,
) -> ExperimentalRecord:
    from pymatgen.core import Composition

    reduced = Composition(formula).reduced_formula
    n_atoms = float(Composition(reduced).num_atoms)
    return ExperimentalRecord(
        formula=formula,
        reduced_formula=reduced,
        formation_enthalpy_eV_per_atom=value,
        formation_enthalpy_eV_per_formula=value * n_atoms,
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
        dataset="synthetic",
        original_units="eV/atom",
    )


def test_infer_scope_from_enumerated_host_and_dopants():
    config = {
        "references": {"host": "SnO2"},
        "doping": {
            "mode": "enumerate",
            "host_species": "Sn",
            "dopants": ["Sb", "Ti", "Sb"],
            "must_include": ["Mn"],
        },
    }
    assert infer_scoped_elements(config) == ("Mn", "Sb", "Sn", "Ti")


def test_infer_scope_from_positive_explicit_compositions():
    config = {
        "references": {"host": "CeO2"},
        "doping": {
            "mode": "explicit",
            "compositions": [{"Sb": 2, "Mn": 3}, {"In": 0, "Nb": 1}],
        },
    }
    assert infer_scoped_elements(config) == ("Ce", "Mn", "Nb", "Sb")


def test_discovery_is_strict_deterministic_and_reports_coverage():
    mno = _record("MnO", material_id="mp-10")
    records = [
        _record("Mn2O3", phase="orth", material_id="mp-12"),
        _record("TiO", phase="cr", material_id="mp-13"),
        _record("SnO", phase="tetrag", material_id=""),
        _record("In2O3", phase="cubic", material_id="mp-15"),
        _record("MnTiO3", phase="trigon", material_id="mp-16"),
        mno,
        mno,
    ]
    result = discover_phase_resolved_binary_oxides(records, ["Ti", "Mn", "Sn"])
    reversed_result = discover_phase_resolved_binary_oxides(
        list(reversed(records)),
        ["Sn", "Mn", "Ti"],
    )

    assert [record.reduced_formula for record in result.records] == ["MnO", "Mn2O3"]
    assert reversed_result.accepted_records == result.accepted_records
    assert reversed_result.report == result.report
    assert result.report["accepted_formulas"] == ["MnO", "Mn2O3"]
    assert result.report["coverage_by_element"]["Mn"] == {
        "accepted_count": 2,
        "formulas": ["MnO", "Mn2O3"],
        "independent_oxygen_stoichiometry_count": 2,
        "oxygen_per_cation_ratios": [1.0, 1.5],
    }
    assert result.report["coverage_by_element"]["Ti"]["accepted_count"] == 0
    assert result.report["rejection_counts"] == {
        "duplicate_record": 1,
        "generic_or_missing_phase": 1,
        "missing_likely_mpid": 1,
        "outside_scoped_elements": 1,
        "too_many_non_oxygen_elements": 1,
    }


def test_all_oxide_discovery_includes_target_scope_ternaries():
    records = [
        _record("MnO", material_id="mp-10"),
        _record("MnTiO3", phase="trigonal", material_id="mp-16"),
        _record("MnAl2O4", phase="cubic", material_id="mp-17"),
    ]
    result = discover_phase_resolved_oxides(records, ["Mn", "Ti"])
    assert [record.reduced_formula for record in result.records] == [
        "MnO",
        "TiMnO3",
    ]
    assert result.report["rejection_counts"] == {"outside_scoped_elements": 1}


def test_discovery_excludes_formula_and_structure_identified_nonordinary_oxides():
    records = [
        _record("Na2O", material_id="mp-20"),
        _record("Na2O2", phase="hex", material_id="mp-21"),
        _record("NaO2", material_id="mp-22"),
        _record("MnO2", phase="tetrag", material_id="mp-23"),
    ]
    result = discover_phase_resolved_binary_oxides(
        records,
        ["Na", "Mn"],
        known_oxygen_environments={"mp-23": "superoxide"},
    )

    assert [record.formula for record in result.records] == ["Na2O"]
    assert result.report["rejection_counts"] == {
        "known_peroxide_or_superoxide_formula": 2,
        "non_ordinary_oxygen_environment": 1,
    }


def _optimade_payload(*, material_id: str = "mp-856", formula: str = "SnO2"):
    return {
        "data": {
            "id": material_id,
            "type": "structures",
            "attributes": {
                "chemical_formula_reduced": formula,
                "dimension_types": [1, 1, 1],
                "nperiodic_dimensions": 3,
                "lattice_vectors": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                "cartesian_site_positions": [
                    [0, 0, 0],
                    [1.25, 1.25, 1.25],
                    [3.75, 3.75, 3.75],
                ],
                "species_at_sites": ["Sn", "O", "O"],
            },
        }
    }


def test_optimade_fetch_validates_and_reuses_immutable_cache(tmp_path):
    calls: list[str] = []

    def transport(url: str):
        calls.append(url)
        return _optimade_payload()

    first = fetch_optimade_structure(
        "mp-856",
        "SnO2",
        tmp_path,
        transport=transport,
    )
    second = fetch_optimade_structure(
        "mp-856",
        "SnO2",
        tmp_path,
        transport=lambda _url: pytest.fail("cache reuse must not access the network"),
    )

    assert len(calls) == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.response_sha256 == first.response_sha256
    assert second.structure_sha256 == first.structure_sha256
    assert second.n_sites == 3
    assert first.response_json_path.exists()
    assert first.structure_path.exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_optimade_payload(material_id="mp-999"), "structure ID mismatch"),
        (_optimade_payload(formula="TiO2"), "declared formula does not match"),
    ],
)
def test_optimade_fetch_rejects_wrong_identity_without_caching(
    tmp_path,
    payload,
    message,
):
    with pytest.raises(ValueError, match=message):
        fetch_optimade_structure(
            "mp-856",
            "SnO2",
            tmp_path,
            transport=lambda _url: payload,
        )
    assert list(tmp_path.iterdir()) == []


def test_optimade_fetch_rejects_tampered_cached_poscar(tmp_path):
    cached = fetch_optimade_structure(
        "mp-856",
        "SnO2",
        tmp_path,
        transport=lambda _url: _optimade_payload(),
    )
    cached.structure_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="POSCAR hash mismatch"):
        fetch_optimade_structure(
            "mp-856",
            "SnO2",
            tmp_path,
            transport=lambda _url: pytest.fail("must validate cache first"),
        )

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from dopingflow.site_preference import (
    PreferenceTarget,
    _aggregate_pair_preferences,
    _cluster_distances,
    _nearest_pair_per_target,
    dopant_pair_records,
    dopant_triplet_records,
    parse_site_preference_config,
    run_site_preference,
    warren_cowley_records,
    symmetry_distinct_pair_orbits,
)


def _target(tmp_path: Path, structure: Structure, *, energy: float = -10.0) -> PreferenceTarget:
    path = tmp_path / "POSCAR"
    structure.to(filename=str(path), fmt="poscar")
    return PreferenceTarget(
        target_id="Sb5_Ti5/candidate_001",
        parent_id="Sb5_Ti5/candidate_001",
        kind="vacancy-free",
        structure_path=path,
        energy_eV=energy,
        energy_source="test",
    )


def test_parse_site_preference_defaults(tmp_path):
    raw = {
        "structure": {"outdir": "structures"},
        "doping": {"host_species": "Sn"},
        "scan": {"anion_species": ["O"]},
        "site_preference": {"enabled": True},
    }
    cfg = parse_site_preference_config(raw, tmp_path)
    assert cfg.host_species == "Sn"
    assert cfg.anion_species == ("O",)
    assert cfg.max_shells == 6
    assert cfg.source_root == (tmp_path / "structures").resolve()
    assert cfg.output_dir == (tmp_path / "structures" / "06_site_preference").resolve()
    assert not cfg.pair_scan.enabled
    assert not cfg.ordering_mc.enabled


@pytest.mark.parametrize(
    "section",
    [
        {"max_shells": 0},
        {"shell_tolerance_angstrom": 0},
        {"mapping_tolerance_angstrom": -1},
        {"ordering_mc": {"temperature_K": 0}},
        {"ordering_mc": {"steps": 100, "burn_in": 100}},
    ],
)
def test_bad_config_rejected(tmp_path, section):
    raw = {
        "doping": {"host_species": "Sn"},
        "site_preference": section,
    }
    with pytest.raises(ValueError):
        parse_site_preference_config(raw, tmp_path)


def test_cluster_distances_groups_relaxed_shell_splitting():
    centers = _cluster_distances(
        [3.00, 3.03, 3.08, 4.00, 4.05, 5.00],
        tolerance=0.10,
        max_shells=3,
    )
    assert len(centers) == 3
    assert centers[0] == pytest.approx((3.00 + 3.03 + 3.08) / 3)
    assert centers[1] == pytest.approx(4.025)
    assert centers[2] == pytest.approx(5.0)


def test_pair_records_use_periodic_distances_and_shells(tmp_path):
    structure = Structure(
        Lattice.cubic(4.0),
        ["Sb", "Ti", "Sn", "Sn", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
    )
    target = _target(tmp_path, structure)
    cfg = parse_site_preference_config(
        {
            "doping": {"host_species": "Sn"},
            "scan": {"anion_species": ["O"]},
            "site_preference": {"max_shells": 3, "shell_tolerance_angstrom": 0.05},
        },
        tmp_path,
    )
    rows = dopant_pair_records(target, structure, cfg)
    assert len(rows) == 1
    assert rows[0]["pair"] == "Sb-Ti"
    assert rows[0]["distance_angstrom"] == pytest.approx(2.0)
    assert rows[0]["shell"] == 1


def test_warren_cowley_detects_first_shell_association(tmp_path):
    structure = Structure(
        Lattice.cubic(4.0),
        ["Sb", "Ti", "Sn", "Sn", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
    )
    target = _target(tmp_path, structure)
    cfg = parse_site_preference_config(
        {
            "doping": {"host_species": "Sn"},
            "scan": {"anion_species": ["O"]},
            "site_preference": {"max_shells": 2, "shell_tolerance_angstrom": 0.05},
        },
        tmp_path,
    )
    rows = warren_cowley_records(target, structure, cfg)
    sb_ti = [row for row in rows if row["pair"] == "Sb-Ti" and row["shell"] == 1]
    assert sb_ti
    assert sb_ti[0]["warren_cowley_alpha"] < 0
    assert sb_ti[0]["interpretation"] == "association"


def test_preference_summary_uses_same_composition_energy_data():
    target_rows = [
        {
            "target_id": "A/c1",
            "composition": "A",
            "structure_kind": "vacancy-free",
            "n_oxygen_vacancies": 0,
            "energy_total_eV": -10.0,
            "delta_energy_within_group_eV": 0.0,
        },
        {
            "target_id": "A/c2",
            "composition": "A",
            "structure_kind": "vacancy-free",
            "n_oxygen_vacancies": 0,
            "energy_total_eV": -9.8,
            "delta_energy_within_group_eV": 0.2,
        },
    ]
    pair_rows = [
        {
            "target_id": "A/c1",
            "pair": "Sb-Ti",
            "distance_angstrom": 3.0,
            "shell": 1,
        },
        {
            "target_id": "A/c2",
            "pair": "Sb-Ti",
            "distance_angstrom": 5.0,
            "shell": 2,
        },
    ]
    nearest = _nearest_pair_per_target(pair_rows, target_rows)
    summary = _aggregate_pair_preferences(nearest)
    assert len(summary) == 1
    assert summary[0]["preferred_shell"] == 1
    close = [row for row in nearest if row["target_id"] == "A/c1"][0]
    assert close["delta_E_vs_farthest_eV"] == pytest.approx(-0.2)
    assert "proxy" in close["proxy_note"].lower()


def test_pair_scan_enumerates_symmetry_distinct_orbits():
    host = Structure(
        Lattice.cubic(4.0),
        ["Sn", "Sn", "Sn", "Sn", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
    )
    rows = symmetry_distinct_pair_orbits(
        host,
        [0, 1, 2, 3],
        "Sb",
        "Ti",
        max_shells=3,
        tolerance=0.05,
        symprec=1e-3,
        angle_tolerance=5.0,
    )
    assert rows
    assert all(row["degeneracy"] >= 1 for row in rows)
    assert all(row["orbit"] >= 1 for row in rows)
    assert all(row["shell"] >= 1 for row in rows)
    assert sum(row["degeneracy"] for row in rows) == 12
    # Labelled unlike dopants are retained as crystallographic assignments;
    # symmetry collapses only genuinely equivalent A/B placements.
    assert all(len(row["relative_cartesian_vector_angstrom"]) == 3 for row in rows)


def test_triplet_motif_classification(tmp_path):
    structure = Structure(
        Lattice.cubic(4.0),
        ["Sb", "Ti", "Nb", "Sn", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
    )
    target = _target(tmp_path, structure)
    cfg = parse_site_preference_config(
        {
            "doping": {"host_species": "Sn"},
            "scan": {"anion_species": ["O"]},
            "site_preference": {
                "max_shells": 2,
                "shell_tolerance_angstrom": 0.05,
                "motif_neighbor_shell_max": 1,
            },
        },
        tmp_path,
    )
    rows = dopant_triplet_records(target, structure, cfg)
    assert len(rows) == 1
    assert rows[0]["species_triplet"] == "Nb-Sb-Ti"
    assert rows[0]["motif"] == "connected_chain"
    assert rows[0]["neighbor_edge_count"] == 2


def test_disabled_stage_is_backward_compatible_without_configuration(tmp_path):
    assert run_site_preference({}, tmp_path) is None

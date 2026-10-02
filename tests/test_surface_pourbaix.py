from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from dopingflow.surface_pourbaix import (
    _select_dft_candidates,
    build_surface_pourbaix_grid,
    enumerate_surface_states,
    parse_surface_pourbaix_config,
    resolve_surface_pourbaix_source_summary,
    surface_state_delta_g_eV,
    surface_state_display_label,
)
from dopingflow.surface_pourbaix_thermo import KB_EV_K, LN10
from dopingflow.surface_pourbaix_sampling import (
    coverage_count_options,
    enumerate_binary_patterns,
    resolve_placement_side,
    symmetry_permutations,
)
from dopingflow.leaching import (
    enumerate_leaching_sites,
    parse_leaching_config,
    resolve_surface_summary,
)


def _slab() -> Structure:
    lattice = Lattice.from_parameters(6.0, 6.0, 24.0, 90, 90, 90)
    species = ["Sn", "Sn", "Sb", "O", "O", "O", "O"]
    frac = [
        [0.25, 0.25, 0.38],
        [0.75, 0.75, 0.52],
        [0.25, 0.75, 0.62],
        [0.25, 0.25, 0.35],
        [0.75, 0.75, 0.55],
        [0.25, 0.75, 0.67],
        [0.75, 0.25, 0.30],
    ]
    return Structure(lattice, species, frac)


def test_default_ph_window_is_minus_one_to_three(tmp_path) -> None:
    cfg = parse_surface_pourbaix_config(
        {"surface_pourbaix": {"enabled": True}}, tmp_path
    )
    assert cfg["pH_min"] == pytest.approx(-1.0)
    assert cfg["pH_max"] == pytest.approx(3.0)
    assert cfg["potential_scale"] == "SHE"


def test_rhe_che_is_ph_independent_for_fixed_rhe_potential() -> None:
    kwargs = dict(
        state_energy_eV=-11.0,
        clean_energy_eV=-10.0,
        delta_n_H=1,
        delta_n_O=0,
        h2_energy_eV=-6.0,
        h2o_energy_eV=-14.0,
        applied_potential_V=1.4,
        potential_scale="RHE",
        temperature_K=298.15,
    )
    low = surface_state_delta_g_eV(pH=-1.0, **kwargs)
    high = surface_state_delta_g_eV(pH=3.0, **kwargs)
    assert high == pytest.approx(low)


def test_she_che_has_expected_ph_slope() -> None:
    kwargs = dict(
        state_energy_eV=-11.0,
        clean_energy_eV=-10.0,
        delta_n_H=1,
        delta_n_O=0,
        h2_energy_eV=-6.0,
        h2o_energy_eV=-14.0,
        applied_potential_V=1.4,
        potential_scale="SHE",
        temperature_K=298.15,
    )
    low = surface_state_delta_g_eV(pH=-1.0, **kwargs)
    high = surface_state_delta_g_eV(pH=3.0, **kwargs)
    expected = 4.0 * KB_EV_K * 298.15 * LN10
    assert high - low == pytest.approx(expected)


def test_enumeration_includes_all_requested_surface_state_families(tmp_path) -> None:
    cfg = parse_surface_pourbaix_config(
        {
            "surface_pourbaix": {
                "enabled": True,
                "placement_side": "top",
                "surface_window_A": 4.0,
                "proton_coverages_pct": [50],
                "o_coverages_pct": [50],
                "oh_coverages_pct": [50],
                "h2o_coverages_pct": [50],
                "mixed_coverages_pct": [[50, 50]],
                "max_arrangements_per_stoichiometry": 2,
            }
        },
        tmp_path,
    )
    states = enumerate_surface_states(_slab(), cfg)
    families = {state["family"] for state in states}
    assert {"clean", "protonated", "O", "OH", "H2O", "mixed-O-OH"} <= families
    protonated = [state for state in states if state["family"] == "protonated"]
    assert protonated
    assert all(state["actual_coverage_pct"] > 0 for state in protonated)
    oh = [state for state in states if state["family"] == "OH"]
    assert oh and all(state["delta_n_H"] == state["delta_n_O"] for state in oh)
    water = [state for state in states if state["family"] == "H2O"]
    assert water and all(state["proton_electron_pairs"] == 0 for state in water)
    clean = next(state for state in states if state["family"] == "clean")
    assert clean["eligible_surface_oxygen_sites"] >= 1
    assert clean["eligible_surface_cation_sites"] >= 1


def test_grid_selects_lowest_free_energy_state() -> None:
    states = [
        dict(
            state_id="clean", family="clean", delta_n_H=0, delta_n_O=0,
            proton_electron_pairs=0, energy=-10.0,
        ),
        dict(
            state_id="H", family="protonated", delta_n_H=1, delta_n_O=0,
            proton_electron_pairs=1, energy=-14.0,
        ),
    ]
    grid, gaps = build_surface_pourbaix_grid(
        states,
        h2_energy_eV=-6.0,
        h2o_energy_eV=-14.0,
        potential_values=[0.0, 2.0],
        pH_values=[0.0],
        potential_scale="SHE",
        temperature_K=298.15,
        energy_key="energy",
        energy_level="test",
    )
    assert list(grid["stable_state_id"]) == ["H", "clean"]
    assert set(gaps["state_id"]) == {"clean", "H"}


def test_leaching_auto_prefers_surface_pourbaix_handoff(tmp_path) -> None:
    source_root = tmp_path / "structures-analysis"
    old_surface = source_root / "08_surfaces"
    old_surface.mkdir(parents=True)
    final_selected = old_surface / "surface_final_selected.csv"
    final_selected.write_text("surface_id\nold-clean-surface\n", encoding="utf-8")

    pourbaix_dir = source_root / "10_surface_pourbaix"
    pourbaix_dir.mkdir(parents=True)
    handoff = pourbaix_dir / "leaching_surface_states.csv"
    handoff.write_text("surface_id\nstable-electrochemical-surface\n", encoding="utf-8")

    config = {
        "surface": {
            "source_root": "structures-analysis",
            "outdir": "08_surfaces",
        },
        "surface_pourbaix": {
            "outdir": "10_surface_pourbaix",
        },
        "leaching": {
            "enabled": True,
            "source_mode": "auto",
        },
    }
    cfg = parse_leaching_config(config, tmp_path)
    assert resolve_surface_summary(config, cfg) == handoff.resolve()


def test_leaching_does_not_treat_surface_hydrogen_as_a_dopant(tmp_path) -> None:
    structure = _slab().copy()
    structure.append("H", [0.25, 0.75, 0.72])
    config = {
        "surface": {
            "host_species": "Sn",
            "anion_species": ["O"],
            "placement_side": "top",
            "cation_layer_tolerance_A": 2.0,
            "layers_per_zone": 1,
        },
        "leaching": {
            "enabled": True,
            "dopant_species": [],
            "zones": ["surface", "subsurface", "bulk"],
            "max_sites_per_surface_species": 20,
        },
    }
    cfg = parse_leaching_config(config, tmp_path)
    rows = enumerate_leaching_sites(
        {"host_species": "Sn", "dopant_species_json": "[]"},
        structure,
        cfg,
    )
    assert rows
    assert {row["dopant"] for row in rows} == {"Sb"}


def test_dft_candidate_selection_ignores_failed_ml_state_gap() -> None:
    records = [
        {
            "state_id": "clean",
            "family": "clean",
            "minimum_deltaG_above_stable_ml_eV": 0.0,
        },
        {
            "state_id": "good",
            "family": "OH",
            "minimum_deltaG_above_stable_ml_eV": 0.12,
        },
        {
            "state_id": "failed",
            "family": "O",
            "minimum_deltaG_above_stable_ml_eV": None,
        },
    ]
    cfg = {"dft": {"candidate_window_eV": 0.30, "max_states_per_surface": 10}}
    selected = _select_dft_candidates(records, cfg)
    assert [row["state_id"] for row in selected] == ["clean", "good"]


def test_surface_pourbaix_direct_surface_source_does_not_require_segregation(tmp_path) -> None:
    surface_dir = tmp_path / "structures-analysis" / "08_surfaces"
    surface_dir.mkdir(parents=True)
    direct = surface_dir / "surface_final_selected.csv"
    direct.write_text("surface_id\ndirect-surface\n", encoding="utf-8")

    config = {
        "surface": {
            "source_root": "structures-analysis",
            "outdir": "08_surfaces",
        },
        "surface_pourbaix": {
            "enabled": True,
            "source_mode": "surface",
        },
    }
    path, mode = resolve_surface_pourbaix_source_summary(config, tmp_path)
    assert path == direct.resolve()
    assert mode == "final-selected"


def test_surface_pourbaix_auto_prefers_direct_surface_over_existing_segregation(tmp_path) -> None:
    root = tmp_path / "structures-analysis"
    surface_dir = root / "08_surfaces"
    surface_dir.mkdir(parents=True)
    direct = surface_dir / "surface_final_selected.csv"
    direct.write_text("surface_id\ndirect-surface\n", encoding="utf-8")

    segregation_dir = root / "09_surface_segregation"
    segregation_dir.mkdir(parents=True)
    segregation = segregation_dir / "surface_segregation_summary.csv"
    segregation.write_text("surface_id\nsegregated-surface\n", encoding="utf-8")

    config = {
        "surface": {
            "source_root": "structures-analysis",
            "outdir": "08_surfaces",
        },
        "surface_segregation": {
            "source_root": "structures-analysis",
            "outdir": "09_surface_segregation",
        },
        "surface_pourbaix": {
            "enabled": True,
            "source_mode": "auto",
        },
    }

    auto_path, auto_mode = resolve_surface_pourbaix_source_summary(config, tmp_path)
    assert auto_path == direct.resolve()
    assert auto_mode == "final-selected"

    config["surface_pourbaix"]["source_mode"] = "segregation"
    seg_path, seg_mode = resolve_surface_pourbaix_source_summary(config, tmp_path)
    assert seg_path == segregation.resolve()
    assert seg_mode == "segregation"


def test_coverage_rounding_uses_actual_surface_site_count() -> None:
    assert coverage_count_options(12, 25.0) == [3]
    assert coverage_count_options(12, 50.0) == [6]
    # 25% of ten sites is exactly halfway between two and three occupied sites,
    # so both finite-cell realizations are intentionally retained.
    assert coverage_count_options(10, 25.0) == [2, 3]
    assert coverage_count_options(10, 75.0) == [7, 8]


def _dopant_bottom_slab() -> Structure:
    lattice = Lattice.from_parameters(6.0, 6.0, 24.0, 90, 90, 90)
    return Structure(
        lattice,
        ["Sn", "Sn", "Sb", "In", "O", "O"],
        [
            [0.25, 0.25, 0.30],
            [0.75, 0.75, 0.70],
            [0.25, 0.75, 0.32],
            [0.75, 0.25, 0.38],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
    )


def test_dopant_nearest_side_selects_side_closest_to_codopants() -> None:
    structure = _dopant_bottom_slab()
    side, info = resolve_placement_side(
        structure,
        {
            "placement_side": "dopant-nearest",
            "anion_species": ["O"],
            "host_species": "Sn",
            "side_target_species": ["Sb", "In"],
            "dopant_side_tie_tolerance_A": 0.10,
        },
    )
    assert side == "bottom"
    assert info["side_target_species"] == ["Sb", "In"]
    assert info["bottom_dopant_depth_A"] < info["top_dopant_depth_A"]


def _symmetric_square_slab() -> Structure:
    lattice = Lattice.from_parameters(4.0, 4.0, 20.0, 90, 90, 90)
    xy = [
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
    ]
    species = ["Sn"] * 4 + ["O"] * 4
    frac = [[x, y, 0.45] for x, y in xy] + [[x, y, 0.55] for x, y in xy]
    return Structure(lattice, species, frac)


def test_surface_site_symmetry_reduces_equivalent_single_occupations() -> None:
    structure = _symmetric_square_slab()
    sites = [(4, 1), (5, 1), (6, 1), (7, 1)]
    permutations = symmetry_permutations(
        structure,
        sites,
        enabled=True,
        symprec_A=0.05,
        mapping_tolerance_A=0.10,
    )
    assert len(permutations) > 1

    patterns, stats = enumerate_binary_patterns(
        structure,
        sites,
        1,
        {
            "symmetry_reduce": True,
            "symmetry_symprec_A": 0.05,
            "symmetry_angle_tolerance_deg": 5.0,
            "symmetry_mapping_tolerance_A": 0.10,
            "max_raw_configurations_per_stoichiometry": 1000,
            "max_arrangements_per_stoichiometry": 8,
            "anion_species": ["O"],
            "host_species": "Sn",
            "side_target_species": [],
        },
    )
    assert stats["raw_total"] == 4
    assert stats["symmetry_unique"] == 1
    assert len(patterns) == 1


def test_surface_state_display_labels_use_actual_coverage_not_arrangement_ids() -> None:
    assert surface_state_display_label(
        {
            "state_id": "protonated_H04_arr006",
            "family": "protonated",
            "actual_coverage_pct": 33.333333,
        }
    ) == "Protonated lattice O — 33.3%"
    assert surface_state_display_label(
        {
            "state_id": "O_02_arr004",
            "family": "O",
            "actual_coverage_pct": 20.0,
        }
    ) == "O* — 20%"
    assert surface_state_display_label(
        {
            "state_id": "mixed_O02_OH03_arr007",
            "family": "mixed-O-OH",
            "actual_o_coverage_pct": 20.0,
            "actual_oh_coverage_pct": 30.0,
        }
    ) == "Mixed O*/OH* — 20% O* + 30% OH*"
    assert surface_state_display_label(
        {"state_id": "clean", "family": "clean"}
    ) == "Clean"

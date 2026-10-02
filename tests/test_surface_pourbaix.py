from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from dopingflow.surface_pourbaix import (
    build_surface_pourbaix_grid,
    enumerate_surface_states,
    parse_surface_pourbaix_config,
    surface_state_delta_g_eV,
)
from dopingflow.surface_pourbaix_thermo import KB_EV_K, LN10
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
                "h_counts": [1],
                "adsorbate_counts": [1],
                "mixed_compositions": [[1, 1]],
                "max_arrangements_per_stoichiometry": 2,
            }
        },
        tmp_path,
    )
    states = enumerate_surface_states(_slab(), cfg)
    families = {state["family"] for state in states}
    assert {"clean", "protonated", "O", "OH", "H2O", "mixed-O-OH"} <= families
    by_family = {state["family"]: state for state in states}
    assert by_family["protonated"]["delta_n_H"] == 1
    assert by_family["OH"]["delta_n_H"] == 1
    assert by_family["OH"]["delta_n_O"] == 1
    assert by_family["H2O"]["proton_electron_pairs"] == 0


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

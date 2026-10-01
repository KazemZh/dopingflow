from __future__ import annotations

import math

import pytest
from pymatgen.core import Lattice, Structure

from dopingflow.surface_segregation import (
    K_B_EV_PER_K,
    SwapMCResult,
    classify_cation_sites,
    infer_dopant_species,
    parse_surface_segregation_config,
    run_surface_swap_mc,
    summarize_surface_occupancy,
)


def _simple_slab() -> Structure:
    lattice = Lattice.tetragonal(5.0, 20.0)
    species = ["Sn", "In", "Sn", "Sn", "O", "O", "O", "O"]
    frac = [
        [0.1, 0.1, 0.20],
        [0.6, 0.1, 0.30],
        [0.1, 0.6, 0.70],
        [0.6, 0.6, 0.80],
        [0.2, 0.2, 0.22],
        [0.7, 0.2, 0.32],
        [0.2, 0.7, 0.68],
        [0.7, 0.7, 0.78],
    ]
    return Structure(lattice, species, frac)


def test_surface_segregation_config_defaults(tmp_path) -> None:
    raw = {
        "structure": {"outdir": "random_structures"},
        "doping": {"host_species": "Sn"},
        "surface": {
            "enabled": True,
            "source_root": "random_structures",
            "outdir": "08_surfaces",
            "host_species": "Sn",
            "anion_species": ["O"],
            "miller_list": [[1, 1, 0]],
        },
        "surface_segregation": {"enabled": True},
    }

    cfg = parse_surface_segregation_config(raw, tmp_path)

    assert cfg.enabled is True
    assert cfg.temperature_K == pytest.approx(800.0)
    assert cfg.steps == 10000
    assert cfg.burn_in == 2000
    assert cfg.sample_interval == 20
    assert cfg.backend == "mace"
    assert cfg.host_species == "Sn"
    assert cfg.anion_species == ("O",)
    assert cfg.output_dir == (
        tmp_path / "random_structures" / "09_surface_segregation"
    ).resolve()


def test_cation_site_zones_are_defined_from_both_slab_sides() -> None:
    slab = _simple_slab()
    zones = classify_cation_sites(
        slab,
        host_species="Sn",
        dopant_species=["In"],
        cation_layer_tolerance_A=0.5,
        dopant_depth_layers=1,
    )

    assert zones == {
        0: "surface",
        1: "subsurface",
        2: "subsurface",
        3: "surface",
    }


def test_infer_dopants_excludes_host_and_anions() -> None:
    slab = _simple_slab()
    assert infer_dopant_species(
        slab,
        host_species="Sn",
        anion_species=["O"],
    ) == ("In",)


def test_swap_mc_preserves_coordinates_and_composition() -> None:
    slab = _simple_slab()
    zones = classify_cation_sites(
        slab,
        host_species="Sn",
        dopant_species=["In"],
        cation_layer_tolerance_A=0.5,
        dopant_depth_layers=1,
    )
    coords_before = [tuple(site.coords) for site in slab]
    composition_before = slab.composition.as_dict()

    def zero_energy(_: Structure) -> float:
        return 0.0

    result = run_surface_swap_mc(
        slab,
        site_zones=zones,
        host_species="Sn",
        dopant_species=["In"],
        energy_function=zero_energy,
        temperature_K=800.0,
        steps=200,
        burn_in=20,
        sample_interval=10,
        trace_interval=50,
        progress_interval=1000,
        seed=7,
    )

    assert result.attempted_moves == 200
    assert result.accepted_moves == 200
    assert result.n_samples == 18
    assert result.final_structure.composition.as_dict() == composition_before
    assert [tuple(site.coords) for site in result.final_structure] == coords_before
    assert sum(result.site_counts["In"].values()) == result.n_samples


def test_occupancy_derived_delta_g_uses_site_normalized_zone_density() -> None:
    slab = _simple_slab()
    zones = {
        0: "surface",
        1: "subsurface",
        2: "bulk",
        3: "bulk",
    }
    # Ten samples, one In atom. It is observed six times on the one surface
    # site and four times across the two bulk sites.
    result = SwapMCResult(
        final_structure=slab.copy(),
        best_structure=slab.copy(),
        start_energy_eV=0.0,
        final_energy_eV=0.0,
        best_energy_eV=0.0,
        attempted_moves=10,
        accepted_moves=5,
        attempted_by_dopant={"In": 10},
        accepted_by_dopant={"In": 5},
        n_samples=10,
        site_counts={"In": {0: 6, 2: 2, 3: 2}},
        zone_counts={"In": {"surface": 6, "bulk": 4}},
        trace=(),
        zone_trace=(),
    )

    _, zone_rows = summarize_surface_occupancy(
        slab,
        site_zones=zones,
        host_species="Sn",
        dopant_species=["In"],
        result=result,
        temperature_K=600.0,
    )

    rows = {row["zone"]: row for row in zone_rows}
    assert rows["surface"]["mean_site_occupancy"] == pytest.approx(0.6)
    assert rows["bulk"]["mean_site_occupancy"] == pytest.approx(0.2)

    expected = -K_B_EV_PER_K * 600.0 * math.log(0.6 / 0.2)
    assert rows["surface"]["delta_G_eff_vs_bulk_eV"] == pytest.approx(expected)
    assert rows["surface"]["delta_G_eff_vs_bulk_eV"] < 0
    assert rows["bulk"]["delta_G_eff_vs_bulk_eV"] == pytest.approx(0.0)


def test_site_pmf_sets_most_occupied_site_to_zero() -> None:
    slab = _simple_slab()
    zones = {
        0: "surface",
        1: "subsurface",
        2: "bulk",
        3: "bulk",
    }
    result = SwapMCResult(
        final_structure=slab.copy(),
        best_structure=slab.copy(),
        start_energy_eV=0.0,
        final_energy_eV=0.0,
        best_energy_eV=0.0,
        attempted_moves=10,
        accepted_moves=5,
        attempted_by_dopant={"In": 10},
        accepted_by_dopant={"In": 5},
        n_samples=10,
        site_counts={"In": {0: 7, 2: 3}},
        zone_counts={"In": {"surface": 7, "bulk": 3}},
        trace=(),
        zone_trace=(),
    )

    site_rows, _ = summarize_surface_occupancy(
        slab,
        site_zones=zones,
        host_species="Sn",
        dopant_species=["In"],
        result=result,
        temperature_K=800.0,
    )
    by_index = {row["site_index"]: row for row in site_rows}

    assert by_index[0]["site_pmf_eV_vs_most_occupied"] == pytest.approx(0.0)
    assert by_index[2]["site_pmf_eV_vs_most_occupied"] > 0
    assert by_index[1]["site_pmf_eV_vs_most_occupied"] is None

from __future__ import annotations

import json
import math

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from dopingflow.leaching import (
    KB_EV_K,
    electrochemical_metrics,
    enumerate_leaching_sites,
    leaching_delta_g_eV,
    parse_leaching_config,
    _load_completed_site_checkpoint,
    _site_calculation_fingerprint,
    _potential_scan_frame,
    _protonatable_oxygen_neighbors,
    _protonation_arrangements,
    protonation_delta_g_eV,
    protonation_adjusted_leaching_delta_g_eV,
    protonation_dissolution_thresholds,
    preview_leaching_sites,
    resolve_leaching_output_dir,
    resolve_surface_summary,
)


def test_leaching_inherits_enabled_surface_refine_calculator() -> None:
    cfg = parse_leaching_config(
        {
            "surface": {
                "refine": {
                    "enabled": True,
                    "backend": "mace",
                    "model": "mh-1",
                    "task": "matpes_r2scan",
                    "device": "cpu",
                }
            },
            "leaching": {"enabled": True},
        }
    )
    assert cfg["backend"] == "mace"
    assert cfg["model"] == "mh-1"
    assert cfg["task"] == "matpes_r2scan"
    assert cfg["zones"] == ["surface"]
    assert cfg["resume_completed"] is True
    assert cfg["protonation"]["enabled"] is False
    assert cfg["protonation"]["h_counts"] == [0, 1, 2, 3]


def test_dissolution_potential_zero_crossing_she() -> None:
    metrics = electrochemical_metrics(
        2.0,
        2,
        0.50,
        ion_activity=1.0,
        temperature_K=298.15,
        pH=0.0,
    )
    assert metrics["dissolution_potential_V_SHE"] == pytest.approx(1.50)
    dg = leaching_delta_g_eV(
        2.0,
        2,
        0.50,
        1.50,
        potential_scale="SHE",
        ion_activity=1.0,
        temperature_K=298.15,
        pH=0.0,
    )
    assert dg == pytest.approx(0.0, abs=1e-12)


def test_dilute_ion_activity_lowers_dissolution_threshold() -> None:
    one_molar = electrochemical_metrics(
        1.0,
        3,
        0.20,
        ion_activity=1.0,
        temperature_K=298.15,
        pH=0.0,
    )
    dilute = electrochemical_metrics(
        1.0,
        3,
        0.20,
        ion_activity=1e-6,
        temperature_K=298.15,
        pH=0.0,
    )
    expected = KB_EV_K * 298.15 * math.log(1e-6) / 3
    assert dilute["dissolution_potential_V_SHE"] < one_molar["dissolution_potential_V_SHE"]
    assert (
        dilute["dissolution_potential_V_SHE"]
        - one_molar["dissolution_potential_V_SHE"]
    ) == pytest.approx(expected)


def test_rhe_and_she_applied_potentials_are_consistent() -> None:
    pH = 2.0
    temperature = 298.15
    u_she = 1.40
    shift = KB_EV_K * temperature * math.log(10.0) * pH
    dg_she = leaching_delta_g_eV(
        1.3,
        3,
        0.10,
        u_she,
        potential_scale="SHE",
        ion_activity=1e-6,
        temperature_K=temperature,
        pH=pH,
    )
    dg_rhe = leaching_delta_g_eV(
        1.3,
        3,
        0.10,
        u_she + shift,
        potential_scale="RHE",
        ion_activity=1e-6,
        temperature_K=temperature,
        pH=pH,
    )
    assert dg_rhe == pytest.approx(dg_she)


def _surface_slab() -> Structure:
    lattice = Lattice.from_parameters(5.0, 5.0, 24.0, 90, 90, 90)
    species = ["Sn", "Sn", "Sn", "Sb", "Ti", "O", "O", "O", "O", "O"]
    frac = [
        [0.1, 0.1, 0.30],
        [0.6, 0.1, 0.40],
        [0.1, 0.6, 0.50],
        [0.6, 0.6, 0.72],
        [0.3, 0.3, 0.58],
        [0.2, 0.2, 0.32],
        [0.7, 0.2, 0.42],
        [0.2, 0.7, 0.52],
        [0.7, 0.7, 0.66],
        [0.5, 0.5, 0.75],
    ]
    return Structure(lattice, species, frac)


def test_site_enumeration_defaults_to_surface_zone() -> None:
    slab = _surface_slab()
    cfg = parse_leaching_config(
        {
            "surface": {
                "host_species": "Sn",
                "dopant_species": ["Sb", "Ti"],
                "placement_side": "top",
                "layers_per_zone": 1,
                "cation_layer_tolerance_A": 2.0,
            },
            "leaching": {"enabled": True, "dopant_species": ["Sb", "Ti"]},
        }
    )
    row = {
        "host_species": "Sn",
        "dopant_species_json": json.dumps(["Sb", "Ti"]),
        "target_zones_json": json.dumps({"Sb": "surface", "Ti": "subsurface"}),
    }
    sites = enumerate_leaching_sites(row, slab, cfg)
    assert sites
    assert {site["dopant"] for site in sites} == {"Sb"}
    assert all(site["detected_zone"] == "surface" for site in sites)
    assert all(site["initial_dopant_zone"] == "surface" for site in sites)
    assert all(site["surface_variant_declared_zone"] == "surface" for site in sites)
    assert all(site["initial_depth_from_selected_surface_A"] >= 0.0 for site in sites)




def test_completed_site_checkpoint_is_reused_and_stale_one_is_rejected(tmp_path) -> None:
    source_path = tmp_path / "surface.POSCAR"
    Poscar(_surface_slab()).write_file(str(source_path))
    cfg = parse_leaching_config(
        {
            "surface": {
                "fix_atoms": True,
                "fix_region": "middle",
                "fix_method": "layers",
                "fix_n_layers": 2,
                "fix_layer_tolerance_A": 0.6,
            },
            "leaching": {
                "enabled": True,
                "backend": "mace",
                "model": "mh-1",
                "task": "matpes_r2scan",
                "resume_completed": True,
                "relax_removed_surface": True,
                "optimizer": "bfgs",
                "fmax": 0.03,
                "max_steps": 500,
            },
        },
        tmp_path,
    )
    surface_cfg = {
        "fix_atoms": True,
        "fix_region": "middle",
        "fix_method": "layers",
        "fix_n_layers": 2,
        "fix_thickness_A": 4.0,
        "fix_layer_tolerance_A": 0.6,
    }
    base = {
        "surface_id": "Sb5/candidate_001/hkl_1_1_0/term_001/variant_001_Sb-surface",
        "dopant": "Sb",
        "site_index": 3,
    }
    fingerprint = _site_calculation_fingerprint(
        source_path, 3, "Sb", cfg, surface_cfg
    )
    relaxed_path = tmp_path / "POSCAR_relaxed"
    Poscar(_surface_slab()).write_file(str(relaxed_path))
    checkpoint = tmp_path / "leaching_result.json"
    checkpoint.write_text(
        json.dumps(
            {
                **base,
                "status": "ok",
                "backend": "mace",
                "model": "mh-1",
                "task": "matpes_r2scan",
                "removed_surface_energy_eV": -98.5,
                "removed_surface_converged": True,
                "removed_surface_final_fmax_eV_per_A": 0.02,
                "removed_surface_optimizer_steps": 42,
                "removed_surface_relaxed_structure_path": str(relaxed_path),
                "calculation_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )

    result, mode = _load_completed_site_checkpoint(
        checkpoint, base, cfg, source_path, surface_cfg
    )
    assert mode == "fingerprint"
    assert result is not None
    assert result["energy_eV"] == pytest.approx(-98.5)

    stale_cfg = dict(cfg)
    stale_cfg["fmax"] = 0.01
    result, mode = _load_completed_site_checkpoint(
        checkpoint, base, stale_cfg, source_path, surface_cfg
    )
    assert result is None
    assert mode == "fingerprint-mismatch"


def test_failed_site_checkpoint_is_not_reused(tmp_path) -> None:
    source_path = tmp_path / "surface.POSCAR"
    Poscar(_surface_slab()).write_file(str(source_path))
    cfg = parse_leaching_config(
        {"leaching": {"enabled": True, "resume_completed": True}},
        tmp_path,
    )
    base = {"surface_id": "x", "dopant": "Sb", "site_index": 3}
    checkpoint = tmp_path / "leaching_result.json"
    checkpoint.write_text(
        json.dumps(
            {
                **base,
                "status": "calculation-failed",
                "backend": cfg["backend"],
                "model": cfg["model"],
                "task": cfg["task"],
            }
        ),
        encoding="utf-8",
    )
    result, mode = _load_completed_site_checkpoint(
        checkpoint, base, cfg, source_path, {}
    )
    assert result is None
    assert mode == "not-complete"


def test_empty_potential_scan_has_stable_columns(tmp_path) -> None:
    scan = _potential_scan_frame([])
    assert scan.empty
    assert "applied_potential_V" in scan.columns
    assert "deltaG_leach_eV" in scan.columns

    path = tmp_path / "leaching_potential_scan.csv"
    scan.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert loaded.empty
    assert list(loaded.columns) == list(scan.columns)


def test_protonation_arrangements_target_neighboring_oxygen() -> None:
    slab = _surface_slab()
    cfg = parse_leaching_config(
        {
            "leaching": {
                "enabled": True,
                "anion_species": ["O"],
                "protonation": {
                    "enabled": True,
                    "h_counts": [0, 1, 2],
                    "neighbor_cutoff_A": 3.0,
                    "oh_bond_length_A": 0.98,
                    "max_arrangements_per_h_count": 5,
                },
            }
        }
    )
    neighbors = _protonatable_oxygen_neighbors(slab, 3, cfg)
    assert neighbors
    assert all(slab[item["oxygen_index"]].specie.symbol == "O" for item in neighbors)

    arrangements = _protonation_arrangements(slab, 3, cfg)
    assert arrangements
    assert {item["h_count"] for item in arrangements}.issuperset({1, 2})
    one_h = next(item for item in arrangements if item["h_count"] == 1)
    assert len(one_h["structure"]) == len(slab)
    assert sum(site.specie.symbol == "H" for site in one_h["structure"]) == 1
    assert "Sb" not in [
        site.specie.symbol
        for i, site in enumerate(one_h["structure"])
        if i == 3
    ]


def test_protonation_che_correction_and_threshold() -> None:
    # At pH 0 on the RHE scale, each consumed (H+ + e-) contributes +U.
    dg_prot = protonation_delta_g_eV(
        -2.0,
        2,
        1.50,
        potential_scale="RHE",
        temperature_K=298.15,
        pH=0.0,
    )
    assert dg_prot == pytest.approx(1.0)

    bare = leaching_delta_g_eV(
        4.0,
        5,
        0.30,
        1.50,
        potential_scale="RHE",
        ion_activity=1e-6,
        temperature_K=298.15,
        pH=0.0,
    )
    via_protonation_helper = protonation_adjusted_leaching_delta_g_eV(
        4.0,
        0,
        5,
        0.30,
        1.50,
        potential_scale="RHE",
        ion_activity=1e-6,
        temperature_K=298.15,
        pH=0.0,
    )
    assert via_protonation_helper == pytest.approx(bare)

    threshold = protonation_dissolution_thresholds(
        3.2,
        2,
        5,
        0.30,
        ion_activity=1e-6,
        temperature_K=298.15,
        pH=0.0,
    )
    assert threshold is not None
    dg_at_threshold = protonation_adjusted_leaching_delta_g_eV(
        3.2,
        2,
        5,
        0.30,
        threshold["dissolution_potential_V_SHE"],
        potential_scale="SHE",
        ion_activity=1e-6,
        temperature_K=298.15,
        pH=0.0,
    )
    assert dg_at_threshold == pytest.approx(0.0, abs=1e-12)


def test_relative_leaching_outdir_is_below_user_source_root(tmp_path) -> None:
    config = {
        "surface": {"source_root": "vacancy-selected/structures-analysis"},
        "leaching": {"enabled": True, "outdir": "09_leaching"},
    }
    cfg = parse_leaching_config(config, tmp_path)
    assert resolve_leaching_output_dir(config, cfg, tmp_path) == (
        tmp_path / "vacancy-selected" / "structures-analysis" / "09_leaching"
    ).resolve()


def test_auto_surface_source_prefers_final_selected(tmp_path) -> None:
    source_root = tmp_path / "structures-analysis"
    outdir = source_root / "08_surfaces"
    outdir.mkdir(parents=True)
    (outdir / "surface_screen_selected.csv").write_text("a\n1\n", encoding="utf-8")
    final = outdir / "surface_final_selected.csv"
    final.write_text("a\n2\n", encoding="utf-8")
    config = {
        "surface": {"source_root": "structures-analysis", "outdir": "08_surfaces"},
        "leaching": {"enabled": True},
    }
    cfg = parse_leaching_config(config, tmp_path)
    assert resolve_surface_summary(config, cfg) == final.resolve()


def test_preview_reads_surface_csv_and_exposes_site_provenance(tmp_path) -> None:
    source_root = tmp_path / "structures-analysis"
    outdir = source_root / "08_surfaces"
    variant = outdir / "targets" / "x" / "hkl_1_1_0" / "term_001" / "variant_001"
    variant.mkdir(parents=True)
    poscar = variant / "POSCAR_relaxed"
    Poscar(_surface_slab()).write_file(str(poscar))
    csv = outdir / "surface_final_selected.csv"
    pd.DataFrame(
        [
            {
                "target_id": "Sb5_Ti5/candidate_001",
                "parent_id": "Sb5_Ti5/candidate_001",
                "structure_kind": "vacancy-free",
                "n_oxygen_vacancies": 0,
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "termination_id": 1,
                "variant_id": 1,
                "variant_label": "Sb-surface__Ti-subsurface",
                "host_species": "Sn",
                "dopant_species_json": json.dumps(["Sb", "Ti"]),
                "target_zones_json": json.dumps({"Sb": "surface", "Ti": "subsurface"}),
                "refine_relaxed_structure_path": str(poscar),
                "refine_backend": "mace",
                "refine_model": "mh-1",
                "refine_task": "matpes_r2scan",
                "refine_energy_eV": -100.0,
            }
        ]
    ).to_csv(csv, index=False)

    config = {
        "surface": {
            "source_root": "structures-analysis",
            "outdir": "08_surfaces",
            "host_species": "Sn",
            "dopant_species": ["Sb", "Ti"],
            "placement_side": "top",
            "layers_per_zone": 1,
            "cation_layer_tolerance_A": 2.0,
            "refine": {
                "enabled": True,
                "backend": "mace",
                "model": "mh-1",
                "task": "matpes_r2scan",
            },
        },
        "leaching": {
            "enabled": True,
            "dopant_species": ["Sb", "Ti"],
            "zones": ["surface"],
        },
    }
    preview = preview_leaching_sites(config, tmp_path)
    assert len(preview) == 1
    row = preview.iloc[0]
    assert row["dopant"] == "Sb"
    assert row["target_id"] == "Sb5_Ti5/candidate_001"
    assert row["surface_source_stage"] == "refine"
    assert row["initial_dopant_zone"] == "surface"
    assert row["surface_variant_declared_zone"] == "surface"
    assert row["initial_site_index"] == row["site_index"]

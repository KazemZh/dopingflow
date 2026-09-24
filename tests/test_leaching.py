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
    preview_leaching_sites,
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


def test_auto_surface_source_prefers_final_selected(tmp_path) -> None:
    outdir = tmp_path / "08_surfaces"
    outdir.mkdir()
    (outdir / "surface_screen_selected.csv").write_text("a\n1\n", encoding="utf-8")
    final = outdir / "surface_final_selected.csv"
    final.write_text("a\n2\n", encoding="utf-8")
    config = {"surface": {"outdir": "08_surfaces"}, "leaching": {"enabled": True}}
    cfg = parse_leaching_config(config, tmp_path)
    assert resolve_surface_summary(config, cfg) == final.resolve()


def test_preview_reads_surface_csv_and_exposes_site_provenance(tmp_path) -> None:
    outdir = tmp_path / "08_surfaces"
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

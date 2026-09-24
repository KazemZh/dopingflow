from __future__ import annotations

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from dopingflow.surface_staged import (
    DEFAULT_MILLERS,
    _add_segregation_metrics,
    _parse_config,
    _rank,
    _screen_shortlist_with_segregation_references,
    _surface_energy,
    _topk,
    _variants,
)


def test_surface_defaults_use_low_index_sno2_facets_and_mace_r2scan_refine() -> None:
    cfg = _parse_config({"surface": {"enabled": True}})

    assert cfg["miller_list"] == DEFAULT_MILLERS
    assert cfg["source_root"] == "random_structures"
    assert cfg["include_vacancy_free"] is True
    assert cfg["include_oxygen_vacancies"] is False
    assert cfg["target_include"] == []
    assert cfg["screen"]["backend"] == "grace"
    assert cfg["screen"]["model"] == "GRACE-1L-OMAT"
    assert cfg["refine"]["backend"] == "mace"
    assert cfg["refine"]["model"] == "mh-1"
    assert cfg["refine"]["task"] == "matpes_r2scan"


def _co_doped_slab() -> Structure:
    lattice = Lattice.tetragonal(5.0, 25.0)
    species = [
        "Sn", "Sn", "Sn", "Sn", "Sn", "Sn", "Sb", "Ti",
        "O", "O", "O", "O", "O", "O", "O", "O",
    ]
    frac = [
        [0.10, 0.10, 0.18],
        [0.60, 0.10, 0.28],
        [0.10, 0.60, 0.38],
        [0.60, 0.60, 0.62],
        [0.10, 0.10, 0.72],
        [0.60, 0.10, 0.82],
        [0.25, 0.25, 0.48],
        [0.75, 0.75, 0.52],
        [0.20, 0.20, 0.20],
        [0.70, 0.20, 0.30],
        [0.20, 0.70, 0.40],
        [0.70, 0.70, 0.45],
        [0.20, 0.20, 0.55],
        [0.70, 0.20, 0.60],
        [0.20, 0.70, 0.70],
        [0.70, 0.70, 0.80],
    ]
    return Structure(lattice, species, frac)


def test_codopant_depth_variants_preserve_composition() -> None:
    slab = _co_doped_slab()
    cfg = {
        "dopant_variant_mode": "co-dopant-depth",
        "host_species": "Sn",
        "dopant_species": ["Sb", "Ti"],
        "anion_species": ["O"],
        "depth_zones": ["surface", "subsurface", "bulk"],
        "placement_side": "top",
        "cation_layer_tolerance_A": 3.0,
        "layers_per_zone": 1,
        "include_original_variant": True,
        "max_dopant_variants_per_termination": 18,
    }

    variants = _variants(slab, cfg)

    assert variants
    assert variants[0][0] == "original"
    assert len(variants) > 1
    original_formula = slab.composition.get_el_amt_dict()
    assert all(struct.composition.get_el_amt_dict() == original_formula for _, struct, _ in variants)
    assert any("Sb-" in label and "Ti-" in label for label, _, _ in variants[1:])


def test_surface_energy_requires_bulk_proportional_stoichiometry() -> None:
    bulk = Structure(
        Lattice.cubic(5.0),
        ["Sn", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    slab = Structure(
        Lattice.from_parameters(5.0, 5.0, 20.0, 90, 90, 90),
        ["Sn", "Sn", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.4],
            [0.5, 0.5, 0.6],
            [0.25, 0.25, 0.42],
            [0.75, 0.75, 0.45],
            [0.25, 0.75, 0.55],
            [0.75, 0.25, 0.58],
        ],
    )
    good = _surface_energy(bulk, slab, slab_energy=-18.0, bulk_energy=-10.0)
    assert good["surface_energy_status"] == "ok"
    assert good["surface_energy_J_m2"] > 0.0

    nonstoich = slab.copy()
    nonstoich.remove_sites([2])
    bad = _surface_energy(bulk, nonstoich, slab_energy=-17.0, bulk_energy=-10.0)
    assert bad["surface_energy_status"].startswith("not_computable_")


def test_ranking_excludes_non_computable_terminations() -> None:
    df = pd.DataFrame(
        [
            {
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "screen_surface_energy_status": "ok",
                "screen_surface_energy_J_m2": 1.2,
            },
            {
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 0,
                "miller_l": 0,
                "screen_surface_energy_status": "ok",
                "screen_surface_energy_J_m2": 0.8,
            },
            {
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 0,
                "miller_l": 1,
                "screen_surface_energy_status": "not_computable_not_proportional",
                "screen_surface_energy_J_m2": None,
            },
        ]
    )

    ranked = _rank(df, "screen")
    selected = _topk(ranked, "screen", 1)

    assert ranked["screen_rankable"].tolist() == [True, True, False]
    assert len(selected) == 1
    assert selected.iloc[0]["screen_surface_energy_J_m2"] == 0.8



def test_segregation_energy_uses_all_bulk_like_variant_as_reference() -> None:
    df = pd.DataFrame(
        [
            {
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "termination_id": 1,
                "variant_label": "Sb-bulk__Ti-bulk",
                "target_zones_json": '{"Sb": "bulk", "Ti": "bulk"}',
                "screen_energy_eV": -100.0,
            },
            {
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "termination_id": 1,
                "variant_label": "Sb-surface__Ti-bulk",
                "target_zones_json": '{"Sb": "surface", "Ti": "bulk"}',
                "screen_energy_eV": -100.4,
            },
            {
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "termination_id": 1,
                "variant_label": "Sb-surface__Ti-surface",
                "target_zones_json": '{"Sb": "surface", "Ti": "surface"}',
                "screen_energy_eV": -99.8,
            },
        ]
    )

    out = _add_segregation_metrics(df, "screen")

    assert set(out["screen_segregation_status"]) == {"ok"}
    assert (
        set(out["screen_segregation_reference_variant"])
        == {"Sb-bulk__Ti-bulk"}
    )
    values = dict(zip(out["variant_label"], out["screen_segregation_energy_eV"]))
    assert values["Sb-bulk__Ti-bulk"] == pytest.approx(0.0)
    assert values["Sb-surface__Ti-bulk"] == pytest.approx(-0.4)
    assert values["Sb-surface__Ti-surface"] == pytest.approx(0.2)



def test_screen_shortlist_keeps_bulk_like_segregation_reference() -> None:
    df = pd.DataFrame(
        [
            {
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "termination_id": 1,
                "variant_label": "Sb-surface__Ti-bulk",
                "target_zones_json": '{"Sb": "surface", "Ti": "bulk"}',
                "screen_energy_eV": -101.0,
                "screen_rankable": True,
                "screen_rank_overall": 1,
            },
            {
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "termination_id": 1,
                "variant_label": "Sb-bulk__Ti-bulk",
                "target_zones_json": '{"Sb": "bulk", "Ti": "bulk"}',
                "screen_energy_eV": -100.0,
                "screen_rankable": True,
                "screen_rank_overall": 8,
            },
            {
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 0,
                "miller_l": 0,
                "termination_id": 1,
                "variant_label": "Sb-surface__Ti-surface",
                "target_zones_json": '{"Sb": "surface", "Ti": "surface"}',
                "screen_energy_eV": -99.5,
                "screen_rankable": True,
                "screen_rank_overall": 2,
            },
        ]
    )

    selected = _screen_shortlist_with_segregation_references(df, top_k=1)

    assert set(selected["variant_label"]) == {
        "Sb-surface__Ti-bulk",
        "Sb-bulk__Ti-bulk",
    }
    reasons = dict(zip(selected["variant_label"], selected["screen_selection_reason"]))
    assert reasons["Sb-surface__Ti-bulk"] == "top_k"
    assert reasons["Sb-bulk__Ti-bulk"] == "segregation_reference"



def test_surface_requires_at_least_one_structure_kind() -> None:
    with pytest.raises(ValueError, match="include_vacancy_free"):
        _parse_config(
            {
                "surface": {
                    "enabled": True,
                    "include_vacancy_free": False,
                    "include_oxygen_vacancies": False,
                }
            }
        )


def test_surface_ranking_is_independent_for_each_source_target() -> None:
    df = pd.DataFrame(
        [
            {
                "target_id": "Sb5_Ti5/candidate_001",
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "screen_surface_energy_status": "ok",
                "screen_surface_energy_J_m2": 1.2,
            },
            {
                "target_id": "Sb5_Ti5/candidate_001/V_O_01/config_0001",
                "composition_tag": "Sb5_Ti5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "screen_surface_energy_status": "ok",
                "screen_surface_energy_J_m2": 0.7,
            },
        ]
    )

    ranked = _rank(df, "screen")

    assert ranked["screen_rank_overall"].tolist() == [1, 1]

from __future__ import annotations

import json

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from dopingflow.surface_staged import (
    DEFAULT_MILLERS,
    discover_surface_targets,
    _parse_config,
    _rank,
    _select_refinement_candidates,
    _surface_energy,
    _topk,
    _variants,
    resolve_surface_output_dir,
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
    assert cfg["refine"]["selection_mode"] == "global"
    assert cfg["refine"]["selection_top_k"] == 10
    assert cfg["refine"]["final_selection_mode"] == "all"
    assert cfg["refine"]["default_terminations_per_orientation"] == 3
    assert cfg["refine"]["default_variants_per_termination"] == 1


def test_surface_relative_outdir_is_below_user_source_root(tmp_path) -> None:
    config = {
        "surface": {
            "enabled": True,
            "source_root": "vacancy-selected/structures-analysis",
            "outdir": "08_surfaces",
        }
    }
    cfg = _parse_config(config)
    assert resolve_surface_output_dir(config, cfg, tmp_path) == (
        tmp_path / "vacancy-selected" / "structures-analysis" / "08_surfaces"
    ).resolve()


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



def _refinement_selection_frame() -> pd.DataFrame:
    rows = []
    energies = {
        (1, 0, 0): [0.50, 0.51, 0.52, 0.53],
        (1, 1, 0): [0.70, 0.71, 0.72],
        (1, 0, 1): [0.80, 0.81, 0.82],
        (0, 0, 1): [0.90, 0.91, 0.92],
    }
    overall = 1
    for hkl, values in energies.items():
        for within, gamma in enumerate(values, start=1):
            rows.append(
                {
                    "target_id": "In2p5_Sb2p5/candidate_001",
                    "composition_tag": "In2p5_Sb2p5",
                    "candidate": "candidate_001",
                    "miller_h": hkl[0],
                    "miller_k": hkl[1],
                    "miller_l": hkl[2],
                    "termination_id": within,
                    "variant_id": 1,
                    "screen_rankable": True,
                    "screen_rank_overall": overall,
                    "screen_rank_within_hkl": within,
                    "screen_surface_energy_J_m2": gamma,
                }
            )
            overall += 1
    return pd.DataFrame(rows)


def test_refinement_selection_global_can_be_dominated_by_one_orientation() -> None:
    df = _refinement_selection_frame()

    selected = _select_refinement_candidates(
        df,
        mode="global",
        top_k=3,
    )

    assert len(selected) == 3
    assert set(
        zip(
            selected["miller_h"],
            selected["miller_k"],
            selected["miller_l"],
        )
    ) == {(1, 0, 0)}
    assert set(selected["screen_selection_mode"]) == {"global"}
    assert set(selected["screen_selection_reason"]) == {"global_top_k"}


def test_refinement_selection_per_orientation_keeps_each_facet() -> None:
    df = _refinement_selection_frame()

    selected = _select_refinement_candidates(
        df,
        mode="per_orientation",
        top_k=2,
    )

    counts = selected.groupby(
        ["miller_h", "miller_k", "miller_l"]
    ).size().to_dict()
    assert counts == {
        (0, 0, 1): 2,
        (1, 0, 0): 2,
        (1, 0, 1): 2,
        (1, 1, 0): 2,
    }
    assert set(selected["screen_selection_mode"]) == {"per_orientation"}
    assert set(selected["screen_selection_reason"]) == {"orientation_top_k"}


def test_refinement_selection_balances_distinct_terminations() -> None:
    rows = []
    for hkl, base in [((1, 0, 0), 0.5), ((1, 1, 0), 0.8)]:
        overall = 1
        for term_id in (1, 2, 3):
            for variant_id in (1, 2):
                rows.append(
                    {
                        "target_id": "In2p5_Sb2p5/candidate_001",
                        "composition_tag": "In2p5_Sb2p5",
                        "candidate": "candidate_001",
                        "miller_h": hkl[0],
                        "miller_k": hkl[1],
                        "miller_l": hkl[2],
                        "termination_id": term_id,
                        "variant_id": variant_id,
                        "screen_rankable": True,
                        "screen_rank_overall": overall,
                        "screen_rank_within_hkl": overall,
                        "screen_surface_energy_J_m2": (
                            base + 0.10 * term_id + 0.01 * variant_id
                        ),
                    }
                )
                overall += 1
    df = pd.DataFrame(rows)

    selected = _select_refinement_candidates(
        df,
        mode="orientation_termination",
        top_k=99,
        default_terminations_per_orientation=2,
        default_variants_per_termination=1,
    )

    counts = selected.groupby(
        ["miller_h", "miller_k", "miller_l"]
    ).size().to_dict()
    assert counts == {(1, 0, 0): 2, (1, 1, 0): 2}
    for _, group in selected.groupby(["miller_h", "miller_k", "miller_l"]):
        assert group["termination_id"].nunique() == 2
        assert set(group["termination_id"]) == {1, 2}
        assert set(group["variant_id"]) == {1}


def test_refinement_selection_supports_per_orientation_limits() -> None:
    rows = []
    for hkl in ((1, 0, 0), (1, 1, 0)):
        for term_id in (1, 2, 3):
            for variant_id in (1, 2):
                rows.append(
                    {
                        "target_id": "In2p5_Sb2p5/candidate_001",
                        "composition_tag": "In2p5_Sb2p5",
                        "candidate": "candidate_001",
                        "miller_h": hkl[0],
                        "miller_k": hkl[1],
                        "miller_l": hkl[2],
                        "termination_id": term_id,
                        "variant_id": variant_id,
                        "screen_rankable": True,
                        "screen_rank_overall": 1,
                        "screen_rank_within_hkl": 1,
                        "screen_surface_energy_J_m2": (
                            term_id + 0.01 * variant_id
                        ),
                    }
                )
    df = pd.DataFrame(rows)

    selected = _select_refinement_candidates(
        df,
        mode="orientation_termination",
        top_k=99,
        orientation_limits={
            "1,0,0": {"terminations": 1, "variants_per_termination": 2},
            "1,1,0": {"terminations": 2, "variants_per_termination": 1},
        },
    )

    facet_100 = selected[
        (selected["miller_h"] == 1)
        & (selected["miller_k"] == 0)
        & (selected["miller_l"] == 0)
    ]
    facet_110 = selected[
        (selected["miller_h"] == 1)
        & (selected["miller_k"] == 1)
        & (selected["miller_l"] == 0)
    ]
    assert len(facet_100) == 2
    assert facet_100["termination_id"].nunique() == 1
    assert len(facet_110) == 2
    assert facet_110["termination_id"].nunique() == 2


def test_refinement_manual_override_edits_automatic_selection() -> None:
    df = pd.DataFrame(
        [
            {
                "target_id": "In2p5_Sb2p5/candidate_001",
                "composition_tag": "In2p5_Sb2p5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 0,
                "miller_l": 0,
                "termination_id": 1,
                "variant_id": 1,
                "screen_rankable": True,
                "screen_rank_overall": 1,
                "screen_rank_within_hkl": 1,
                "screen_surface_energy_J_m2": 0.50,
            },
            {
                "target_id": "In2p5_Sb2p5/candidate_001",
                "composition_tag": "In2p5_Sb2p5",
                "candidate": "candidate_001",
                "miller_h": 1,
                "miller_k": 1,
                "miller_l": 0,
                "termination_id": 2,
                "variant_id": 1,
                "screen_rankable": True,
                "screen_rank_overall": 2,
                "screen_rank_within_hkl": 1,
                "screen_surface_energy_J_m2": 0.70,
            },
        ]
    )

    automatic = _select_refinement_candidates(df, mode="global", top_k=1)
    auto_id = automatic.iloc[0]["surface_id"]
    other = _select_refinement_candidates(df, mode="global", top_k=2)
    other_id = other.iloc[1]["surface_id"]

    selected = _select_refinement_candidates(
        df,
        mode="global",
        top_k=1,
        manual_include_surface_ids=[other_id],
        manual_exclude_surface_ids=[auto_id],
    )

    assert selected["surface_id"].tolist() == [other_id]
    assert selected.iloc[0]["screen_selection_reason"] == "manual_include"


def test_refinement_selection_config_accepts_balanced_limits() -> None:
    cfg = _parse_config(
        {
            "surface": {
                "enabled": True,
                "refine": {
                    "selection_mode": "orientation_termination",
                    "default_terminations_per_orientation": 2,
                    "default_variants_per_termination": 2,
                    "orientation_limits": {
                        "1,0,0": {
                            "terminations": 4,
                            "variants_per_termination": 3,
                        }
                    },
                    "final_selection_mode": "all",
                },
            }
        }
    )

    assert cfg["refine"]["selection_mode"] == "orientation_termination"
    assert cfg["refine"]["orientation_limits"]["1,0,0"] == {
        "terminations": 4,
        "variants_per_termination": 3,
    }
    assert cfg["refine"]["final_selection_mode"] == "all"


def test_refinement_selection_config_accepts_per_orientation() -> None:
    cfg = _parse_config(
        {
            "surface": {
                "enabled": True,
                "screen": {"top_k_per_candidate": 10},
                "refine": {
                    "selection_mode": "per_orientation",
                    "selection_top_k": 5,
                },
            }
        }
    )

    assert cfg["refine"]["selection_mode"] == "per_orientation"
    assert cfg["refine"]["selection_top_k"] == 5


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



def test_surface_discovers_vacancy_free_and_oxygen_vacancy_targets(tmp_path) -> None:
    source = tmp_path / "vacancy-selected"
    comp = source / "Sb5_Ti5"
    candidate = comp / "candidate_001"
    (candidate / "01_scan").mkdir(parents=True)
    (candidate / "02_relax").mkdir(parents=True)
    (comp / "selected_candidates.txt").write_text("candidate_001\n", encoding="utf-8")

    parent = Structure(
        Lattice.cubic(5.0),
        ["Sn", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    Poscar(parent).write_file(str(candidate / "01_scan" / "POSCAR"))
    Poscar(parent).write_file(str(candidate / "02_relax" / "POSCAR"))

    vacancy_dir = source / "vacancy_structures" / "config_0001"
    vacancy_dir.mkdir(parents=True)
    vacancy = parent.copy()
    vacancy.remove_sites([2])
    vacancy_poscar = vacancy_dir / "POSCAR_relaxed"
    Poscar(vacancy).write_file(str(vacancy_poscar))

    (source / "vacancies_database.json").write_text(
        json.dumps(
            [
                {
                    "parent_id": "Sb5_Ti5/candidate_001",
                    "n_vacancies": 1,
                    "vacancy_species": "O",
                    "configuration_id": "config_0001",
                    "relaxed_poscar_path": str(vacancy_poscar),
                }
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "surface": {
            "enabled": True,
            "source_root": str(source),
            "include_vacancy_free": True,
            "include_oxygen_vacancies": True,
            "target_include": [],
        }
    }

    targets, warnings = discover_surface_targets(config, tmp_path)

    assert warnings == []
    assert [target.kind for target in targets] == ["vacancy-free", "oxygen-vacancy"]
    assert targets[0].target_id == "Sb5_Ti5/candidate_001"
    assert targets[1].target_id == (
        "Sb5_Ti5/candidate_001/V_O_01/config_0001"
    )

    config["surface"]["target_include"] = [
        "Sb5_Ti5/candidate_001/V_O_01/*"
    ]
    filtered, _ = discover_surface_targets(config, tmp_path)
    assert len(filtered) == 1
    assert filtered[0].kind == "oxygen-vacancy"

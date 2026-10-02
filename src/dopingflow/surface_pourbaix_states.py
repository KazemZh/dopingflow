from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from dopingflow.surface_staged import _evaluate
from dopingflow.surface_pourbaix_sampling import (
    enumerate_binary_patterns,
    enumerate_mixed_patterns,
    exposed_sites,
    mixed_realizable_coverages,
    realizable_coverages,
    resolve_placement_side,
    surface_normal,
)


def _inplane_unit(structure: Structure, normal: np.ndarray) -> np.ndarray:
    for raw in (structure.lattice.matrix[0], structure.lattice.matrix[1]):
        vec = np.asarray(raw, dtype=float)
        vec = vec - np.dot(vec, normal) * normal
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            return vec / norm
    raise ValueError("Could not construct an in-plane unit vector")


def _append(structure: Structure, symbol: str, coords: np.ndarray) -> None:
    structure.append(symbol, np.asarray(coords, dtype=float), coords_are_cartesian=True)


def _add_proton(
    structure: Structure, oxygen_index: int, sign: int, normal: np.ndarray, bond_A: float
) -> None:
    coords = np.asarray(structure[oxygen_index].coords) + float(sign) * normal * float(bond_A)
    _append(structure, "H", coords)


def _add_adsorbate(
    structure: Structure,
    cation_index: int,
    sign: int,
    family: str,
    normal: np.ndarray,
    inplane: np.ndarray,
    cfg: Mapping[str, Any],
) -> None:
    outward = float(sign) * normal
    o_coords = np.asarray(structure[cation_index].coords) + outward * float(cfg["adsorbate_height_A"])
    _append(structure, "O", o_coords)
    if family == "O":
        return
    if family == "OH":
        _append(structure, "H", o_coords + outward * float(cfg["oh_bond_length_A"]))
        return
    if family == "H2O":
        theta = math.radians(float(cfg["water_hoh_angle_deg"]))
        alpha = 0.5 * theta
        length = float(cfg["water_oh_bond_length_A"])
        v1 = math.cos(alpha) * outward + math.sin(alpha) * inplane
        v2 = math.cos(alpha) * outward - math.sin(alpha) * inplane
        _append(structure, "H", o_coords + length * v1)
        _append(structure, "H", o_coords + length * v2)
        return
    raise ValueError(f"Unknown adsorbate family {family}")


def _coverage_label(values: Sequence[float]) -> str:
    return ",".join(f"{float(value):g}" for value in values)


def _base_state_metadata(
    *,
    side_info: Mapping[str, Any],
    n_oxygen_sites: int,
    n_cation_sites: int,
) -> dict[str, Any]:
    return {
        "requested_placement_side": side_info.get("requested_placement_side"),
        "resolved_placement_side": side_info.get("resolved_placement_side"),
        "side_target_species": list(side_info.get("side_target_species", [])),
        "top_dopant_depth_A": side_info.get("top_dopant_depth_A"),
        "bottom_dopant_depth_A": side_info.get("bottom_dopant_depth_A"),
        "side_selection_reason": side_info.get("side_selection_reason"),
        "eligible_surface_oxygen_sites": int(n_oxygen_sites),
        "eligible_surface_cation_sites": int(n_cation_sites),
    }


def enumerate_surface_states(structure: Structure, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    normal = surface_normal(structure)
    inplane = _inplane_unit(structure, normal)
    resolved_side, side_info = resolve_placement_side(structure, cfg)

    anions = list(cfg["anion_species"])
    cations = sorted({
        site.specie.symbol for site in structure
        if site.specie.symbol not in set(anions) and site.specie.symbol != "H"
    })
    oxygen_sites = exposed_sites(
        structure,
        anions,
        normal=normal,
        placement_side=resolved_side,
        window_A=float(cfg["surface_window_A"]),
        limit=int(cfg.get("max_surface_oxygen_sites", 0)),
    )
    cation_sites = exposed_sites(
        structure,
        cations,
        normal=normal,
        placement_side=resolved_side,
        window_A=float(cfg["surface_window_A"]),
        limit=int(cfg.get("max_surface_cation_sites", 0)),
    )
    families = set(cfg["state_families"])
    common = _base_state_metadata(
        side_info=side_info,
        n_oxygen_sites=len(oxygen_sites),
        n_cation_sites=len(cation_sites),
    )
    states = [dict(
        state_id="clean",
        family="clean",
        delta_n_H=0,
        delta_n_O=0,
        proton_electron_pairs=0,
        arrangement_id=0,
        site_indices=[],
        site_sides=[],
        requested_coverage_pct=0.0,
        actual_coverage_pct=0.0,
        raw_arrangements_total=1,
        raw_arrangements_examined=1,
        symmetry_unique_arrangements=1,
        symmetry_operations=1,
        structure=structure.copy(),
        **common,
    )]

    if "protonated" in families and oxygen_sites:
        for coverage in realizable_coverages(
            len(oxygen_sites), cfg["proton_coverages_pct"]
        ):
            n_h = int(coverage["count"])
            patterns, stats = enumerate_binary_patterns(
                structure, oxygen_sites, n_h, cfg
            )
            for arr, pattern in enumerate(patterns, 1):
                candidate = structure.copy()
                selected = [
                    oxygen_sites[pos]
                    for pos, label in enumerate(pattern)
                    if int(label) != 0
                ]
                for idx, sign in selected:
                    _add_proton(
                        candidate, idx, sign, normal, cfg["oh_bond_length_A"]
                    )
                states.append(dict(
                    state_id=f"protonated_H{n_h:02d}_arr{arr:03d}",
                    family="protonated",
                    delta_n_H=n_h,
                    delta_n_O=0,
                    proton_electron_pairs=n_h,
                    arrangement_id=arr,
                    site_indices=[x[0] for x in selected],
                    site_sides=[x[1] for x in selected],
                    requested_coverage_pct=_coverage_label(
                        coverage["requested_coverages_pct"]
                    ),
                    actual_coverage_pct=float(coverage["actual_coverage_pct"]),
                    raw_arrangements_total=int(stats["raw_total"]),
                    raw_arrangements_examined=int(stats["raw_examined"]),
                    symmetry_unique_arrangements=int(stats["symmetry_unique"]),
                    symmetry_operations=int(stats.get("symmetry_operations", 1)),
                    structure=candidate,
                    **common,
                ))

    family_coverage_keys = {
        "O": "o_coverages_pct",
        "OH": "oh_coverages_pct",
        "H2O": "h2o_coverages_pct",
    }
    for family, h_per in (("O", 0), ("OH", 1), ("H2O", 2)):
        if family not in families or not cation_sites:
            continue
        for coverage in realizable_coverages(
            len(cation_sites), cfg[family_coverage_keys[family]]
        ):
            count = int(coverage["count"])
            patterns, stats = enumerate_binary_patterns(
                structure, cation_sites, count, cfg
            )
            for arr, pattern in enumerate(patterns, 1):
                candidate = structure.copy()
                selected = [
                    cation_sites[pos]
                    for pos, label in enumerate(pattern)
                    if int(label) != 0
                ]
                for idx, sign in selected:
                    _add_adsorbate(
                        candidate, idx, sign, family, normal, inplane, cfg
                    )
                d_h, d_o = count * h_per, count
                states.append(dict(
                    state_id=f"{family}_{count:02d}_arr{arr:03d}",
                    family=family,
                    delta_n_H=d_h,
                    delta_n_O=d_o,
                    proton_electron_pairs=d_h - 2 * d_o,
                    arrangement_id=arr,
                    site_indices=[x[0] for x in selected],
                    site_sides=[x[1] for x in selected],
                    requested_coverage_pct=_coverage_label(
                        coverage["requested_coverages_pct"]
                    ),
                    actual_coverage_pct=float(coverage["actual_coverage_pct"]),
                    raw_arrangements_total=int(stats["raw_total"]),
                    raw_arrangements_examined=int(stats["raw_examined"]),
                    symmetry_unique_arrangements=int(stats["symmetry_unique"]),
                    symmetry_operations=int(stats.get("symmetry_operations", 1)),
                    structure=candidate,
                    **common,
                ))

    if "mixed-O-OH" in families and cation_sites:
        for coverage in mixed_realizable_coverages(
            len(cation_sites), cfg["mixed_coverages_pct"]
        ):
            n_o, n_oh = int(coverage["n_o"]), int(coverage["n_oh"])
            patterns, stats = enumerate_mixed_patterns(
                structure, cation_sites, n_o, n_oh, cfg
            )
            for arr, pattern in enumerate(patterns, 1):
                candidate = structure.copy()
                selected_sites: list[tuple[int, int]] = []
                o_sites: list[int] = []
                oh_sites: list[int] = []
                for pos, label in enumerate(pattern):
                    if int(label) == 0:
                        continue
                    idx, sign = cation_sites[pos]
                    selected_sites.append((idx, sign))
                    family = "O" if int(label) == 1 else "OH"
                    _add_adsorbate(
                        candidate, idx, sign, family, normal, inplane, cfg
                    )
                    (o_sites if family == "O" else oh_sites).append(idx)
                total = n_o + n_oh
                states.append(dict(
                    state_id=f"mixed_O{n_o:02d}_OH{n_oh:02d}_arr{arr:03d}",
                    family="mixed-O-OH",
                    delta_n_H=n_oh,
                    delta_n_O=total,
                    proton_electron_pairs=n_oh - 2 * total,
                    arrangement_id=arr,
                    site_indices=[x[0] for x in selected_sites],
                    site_sides=[x[1] for x in selected_sites],
                    o_site_indices=o_sites,
                    oh_site_indices=oh_sites,
                    requested_o_coverage_pct=";".join(
                        f"{pair[0]:g}" for pair in coverage["requested_pairs_pct"]
                    ),
                    requested_oh_coverage_pct=";".join(
                        f"{pair[1]:g}" for pair in coverage["requested_pairs_pct"]
                    ),
                    actual_o_coverage_pct=float(
                        coverage["actual_o_coverage_pct"]
                    ),
                    actual_oh_coverage_pct=float(
                        coverage["actual_oh_coverage_pct"]
                    ),
                    actual_coverage_pct=float(
                        coverage["actual_o_coverage_pct"]
                        + coverage["actual_oh_coverage_pct"]
                    ),
                    raw_arrangements_total=int(stats["raw_total"]),
                    raw_arrangements_examined=int(stats["raw_examined"]),
                    symmetry_unique_arrangements=int(stats["symmetry_unique"]),
                    symmetry_operations=int(stats.get("symmetry_operations", 1)),
                    structure=candidate,
                    **common,
                ))
    return states


def _fingerprint(structure: Structure, calc: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    payload = dict(
        schema=1,
        lattice=np.round(structure.lattice.matrix, 10).tolist(),
        species=[site.specie.symbol for site in structure],
        cart_coords=np.round(structure.cart_coords, 10).tolist(),
        calculator={key: calc.get(key) for key in ("backend", "model", "task", "optimizer", "fmax", "max_steps", "relax")},
        state={key: state.get(key) for key in ("state_id", "family", "delta_n_H", "delta_n_O", "site_indices", "site_sides")},
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def screen_state(
    state: Mapping[str, Any], cfg: Mapping[str, Any], calculator: Any,
    fixed: Sequence[int], outdir: Path,
) -> dict[str, Any]:
    structure = state["structure"]
    state_dir = outdir / str(state["state_id"])
    state_dir.mkdir(parents=True, exist_ok=True)
    initial = state_dir / "POSCAR_initial"
    Poscar(structure).write_file(str(initial))
    fingerprint = _fingerprint(structure, cfg["screen"], state)
    manifest, result_path = state_dir / "state_manifest.json", state_dir / "result.json"
    if bool(cfg["resume_completed"]) and manifest.exists() and result_path.exists():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            relaxed = str(result.get("relaxed_structure_path") or "")
            if saved.get("fingerprint") == fingerprint and result.get("status") == "ok" and (not relaxed or Path(relaxed).exists()):
                return {**result, "checkpoint_reused": True, "structure_path": relaxed or str(initial)}
        except (OSError, json.JSONDecodeError):
            pass
    result = _evaluate(structure, fixed, cfg["screen"], calculator, state_dir)
    manifest.write_text(json.dumps({"fingerprint": fingerprint}, indent=2), encoding="utf-8")
    return {
        **result, "checkpoint_reused": False,
        "structure_path": str(result.get("relaxed_structure_path") or initial),
    }


def molecule_structure(kind: str, box_A: float) -> Structure:
    lattice = Lattice.cubic(float(box_A))
    center = np.array([box_A / 2.0] * 3, dtype=float)
    if kind == "H2":
        d = 0.74
        coords = [center + [0, 0, -d / 2], center + [0, 0, d / 2]]
        return Structure(lattice, ["H", "H"], coords, coords_are_cartesian=True)
    if kind == "H2O":
        bond, theta = 0.9572, math.radians(104.5)
        h1 = center + [bond * math.sin(theta / 2), 0, bond * math.cos(theta / 2)]
        h2 = center + [-bond * math.sin(theta / 2), 0, bond * math.cos(theta / 2)]
        return Structure(lattice, ["O", "H", "H"], [center, h1, h2], coords_are_cartesian=True)
    raise ValueError(kind)


def ml_reference_energy(kind: str, cfg: Mapping[str, Any], calculator: Any) -> tuple[float | None, str]:
    manual = cfg["manual_h2_energy_eV"] if kind == "H2" else cfg["manual_h2o_energy_eV"]
    correction = cfg["h2_free_energy_correction_eV"] if kind == "H2" else cfg["h2o_free_energy_correction_eV"]
    if manual is not None:
        return float(manual) + float(correction), "manual"
    if not bool(cfg["compute_references"]):
        return None, "disabled"
    calc_cfg = dict(cfg["screen"])
    calc_cfg["relax"] = bool(cfg["reference_relax"])
    result = _evaluate(
        molecule_structure(kind, cfg["reference_box_A"]), [], calc_cfg, calculator,
        Path(cfg["output_dir"]) / "references" / kind,
    )
    if result.get("status") != "ok" or result.get("energy_eV") is None:
        return None, "calculation-failed"
    return float(result["energy_eV"]) + float(correction), "ml-calculation"

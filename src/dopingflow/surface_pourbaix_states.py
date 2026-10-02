from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from dopingflow.surface_staged import _evaluate


def surface_normal(structure: Structure) -> np.ndarray:
    normal = np.cross(structure.lattice.matrix[0], structure.lattice.matrix[1])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("Surface cell vectors are degenerate")
    return np.asarray(normal, dtype=float) / norm


def exposed_sites(
    structure: Structure,
    species: Sequence[str],
    *,
    normal: np.ndarray,
    placement_side: str,
    window_A: float,
    limit: int,
) -> list[tuple[int, int]]:
    allowed = set(species)
    rows = [
        (i, float(np.dot(site.coords, normal)))
        for i, site in enumerate(structure)
        if site.specie.symbol in allowed
    ]
    if not rows:
        return []
    lo, hi = min(x[1] for x in rows), max(x[1] for x in rows)
    selected: list[tuple[int, int, float]] = []
    if placement_side in {"top", "both"}:
        selected.extend((i, +1, hi - p) for i, p in rows if hi - p <= window_A + 1e-12)
    if placement_side in {"bottom", "both"}:
        selected.extend((i, -1, p - lo) for i, p in rows if p - lo <= window_A + 1e-12)
    selected.sort(key=lambda item: (item[2], item[0], -item[1]))
    result, seen = [], set()
    for idx, sign, _ in selected:
        key = (idx, sign)
        if key not in seen:
            result.append(key)
            seen.add(key)
        if len(result) >= int(limit):
            break
    return result


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


def _combos(items: Sequence[Any], n: int, cap: int) -> list[tuple[Any, ...]]:
    if n <= 0 or n > len(items):
        return []
    result = []
    for combo in combinations(items, n):
        result.append(combo)
        if len(result) >= int(cap):
            break
    return result


def enumerate_surface_states(structure: Structure, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    normal = surface_normal(structure)
    inplane = _inplane_unit(structure, normal)
    anions = list(cfg["anion_species"])
    cations = sorted({
        site.specie.symbol for site in structure
        if site.specie.symbol not in set(anions) and site.specie.symbol != "H"
    })
    oxygen_sites = exposed_sites(
        structure, anions, normal=normal, placement_side=str(cfg["placement_side"]),
        window_A=float(cfg["surface_window_A"]), limit=int(cfg["max_surface_oxygen_sites"]),
    )
    cation_sites = exposed_sites(
        structure, cations, normal=normal, placement_side=str(cfg["placement_side"]),
        window_A=float(cfg["surface_window_A"]), limit=int(cfg["max_surface_cation_sites"]),
    )
    cap = int(cfg["max_arrangements_per_stoichiometry"])
    families = set(cfg["state_families"])
    states = [dict(
        state_id="clean", family="clean", delta_n_H=0, delta_n_O=0,
        proton_electron_pairs=0, arrangement_id=0, site_indices=[], site_sides=[],
        structure=structure.copy(),
    )]

    if "protonated" in families:
        for n_h in cfg["h_counts"]:
            for arr, combo in enumerate(_combos(oxygen_sites, int(n_h), cap), 1):
                candidate = structure.copy()
                for idx, sign in combo:
                    _add_proton(candidate, idx, sign, normal, cfg["oh_bond_length_A"])
                states.append(dict(
                    state_id=f"protonated_H{int(n_h):02d}_arr{arr:03d}", family="protonated",
                    delta_n_H=int(n_h), delta_n_O=0, proton_electron_pairs=int(n_h),
                    arrangement_id=arr, site_indices=[x[0] for x in combo],
                    site_sides=[x[1] for x in combo], structure=candidate,
                ))

    for family, h_per in (("O", 0), ("OH", 1), ("H2O", 2)):
        if family not in families:
            continue
        for count in cfg["adsorbate_counts"]:
            for arr, combo in enumerate(_combos(cation_sites, int(count), cap), 1):
                candidate = structure.copy()
                for idx, sign in combo:
                    _add_adsorbate(candidate, idx, sign, family, normal, inplane, cfg)
                d_h, d_o = int(count) * h_per, int(count)
                states.append(dict(
                    state_id=f"{family}_{int(count):02d}_arr{arr:03d}", family=family,
                    delta_n_H=d_h, delta_n_O=d_o, proton_electron_pairs=d_h - 2 * d_o,
                    arrangement_id=arr, site_indices=[x[0] for x in combo],
                    site_sides=[x[1] for x in combo], structure=candidate,
                ))

    if "mixed-O-OH" in families:
        for n_o, n_oh in cfg["mixed_compositions"]:
            total, arr = int(n_o) + int(n_oh), 0
            for selected in combinations(cation_sites, total):
                for oh_positions in combinations(range(total), int(n_oh)):
                    arr += 1
                    candidate = structure.copy()
                    oh_set = set(oh_positions)
                    for pos, (idx, sign) in enumerate(selected):
                        _add_adsorbate(
                            candidate, idx, sign, "OH" if pos in oh_set else "O",
                            normal, inplane, cfg,
                        )
                    states.append(dict(
                        state_id=f"mixed_O{int(n_o):02d}_OH{int(n_oh):02d}_arr{arr:03d}",
                        family="mixed-O-OH", delta_n_H=int(n_oh), delta_n_O=total,
                        proton_electron_pairs=int(n_oh) - 2 * total,
                        arrangement_id=arr, site_indices=[x[0] for x in selected],
                        site_sides=[x[1] for x in selected], structure=candidate,
                    ))
                    if arr >= cap:
                        break
                if arr >= cap:
                    break
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

from __future__ import annotations

import itertools
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from ase.constraints import FixAtoms
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar

from dopingflow.ml_backends import (
    build_ase_calculator,
    check_backend_dependency,
    normalize_backend_config,
    prepare_backend_runtime,
)
from dopingflow.surface import (
    _compute_bulk_equivalent_factor,
    _compute_final_fmax_from_forces,
    _compute_surface_energy,
    _estimate_slab_and_vacuum_thickness_A,
    _get_optimizer_class,
    _resolve_bulk_structure_path,
    _select_candidates,
    _select_fixed_atom_indices,
    _sort_structure_for_poscar,
    _validate_database_columns,
    _write_poscar_with_selective_dynamics,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_MILLERS = [(1, 1, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]
_VALID_ZONES = {"surface", "subsurface", "bulk"}


def run_surface_scan_from_toml(config_path: Path) -> Path | None:
    with open(config_path, "rb") as handle:
        return run_surface_scan(tomllib.load(handle))


def run_surface_refine_from_toml(config_path: Path) -> Path | None:
    with open(config_path, "rb") as handle:
        return run_surface_refine(tomllib.load(handle))


def run_surface_workflow_from_toml(config_path: Path) -> Path | None:
    with open(config_path, "rb") as handle:
        config = tomllib.load(handle)
    out = run_surface_scan(config)
    if out is None:
        return None
    cfg = _parse_config(config)
    return run_surface_refine(config) if cfg["refine"]["enabled"] else out


def _calculator_cfg(
    raw: Mapping[str, Any],
    section: str,
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    data = dict(defaults)
    data.update(dict(raw or {}))
    backend, model, task = normalize_backend_config(
        backend=str(data["backend"]),
        model=str(data["model"]),
        task=str(data["task"]),
        section_name=section,
    )
    data.update(backend=backend, model=model, task=task)
    data["device"] = str(data["device"]).lower()
    data["optimizer"] = str(data["optimizer"]).lower()
    if data["device"] not in {"cpu", "cuda"}:
        raise ValueError(f"[{section}].device must be cpu or cuda")
    if data["optimizer"] not in {"bfgs", "lbfgs", "fire", "mdmin", "quasinewton"}:
        raise ValueError(f"[{section}].optimizer is not supported")
    for key in ("gpu_id", "tf_threads", "omp_threads", "max_steps", "top_k_per_candidate"):
        data[key] = int(data[key])
    data["fmax"] = float(data["fmax"])
    if data["gpu_id"] < 0 or data["tf_threads"] <= 0 or data["omp_threads"] <= 0:
        raise ValueError(f"[{section}] invalid runtime settings")
    if data["max_steps"] <= 0 or data["top_k_per_candidate"] <= 0 or data["fmax"] <= 0:
        raise ValueError(f"[{section}] invalid relaxation/ranking settings")
    return data


def _parse_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    surface = dict(config.get("surface", {}) or {})
    defaults = {
        "enabled": False,
        "source_summary": "results_database.csv",
        "composition_tag": None,
        "composition_tags": [],
        "selection_mode": "top_n",
        "candidate_id": 1,
        "candidate_ids": [],
        "rank_start": 1,
        "rank_end": 10,
        "top_n": 3,
        "formation_energy_min": -1e9,
        "formation_energy_max": 1e9,
        "bandgap_min": -1e9,
        "bandgap_max": 1e9,
        "max_candidates": 20,
        "orientation_mode": "explicit",
        "miller_list": [list(x) for x in DEFAULT_MILLERS],
        "max_miller": 1,
        "max_orientations": 8,
        "min_slab_size": 12.0,
        "min_vacuum_size": 15.0,
        "center_slab": True,
        "in_unit_planes": False,
        "lll_reduce": False,
        "primitive": False,
        "reorient_lattice": True,
        "orthogonal_c": True,
        "termination_mode": "all",
        "max_terminations_per_orientation": 12,
        "symmetrize_slabs": False,
        "outdir": "08_surfaces",
        "max_total_surfaces": 1000,
        "write_cif": False,
        "fix_atoms": False,
        "fix_region": "middle",
        "fix_method": "layers",
        "fix_n_layers": 2,
        "fix_thickness_A": 4.0,
        "fix_layer_tolerance_A": 0.6,
        "dopant_variant_mode": "co-dopant-depth",
        "host_species": "",
        "dopant_species": [],
        "anion_species": ["O"],
        "depth_zones": ["surface", "subsurface", "bulk"],
        "placement_side": "top",
        "cation_layer_tolerance_A": 0.8,
        "layers_per_zone": 1,
        "include_original_variant": True,
        "max_dopant_variants_per_termination": 18,
        "screen_summary_csv": "surface_screen_summary.csv",
        "screen_selected_csv": "surface_screen_selected.csv",
        "refine_summary_csv": "surface_refine_summary.csv",
        "refine_selected_csv": "surface_final_selected.csv",
    }
    for key, value in defaults.items():
        surface.setdefault(key, value)
    surface["poscar_order"] = (config.get("generate", {}) or {}).get("poscar_order")

    # Backward-compatible flat keys are mapped to the screen calculator.
    screen_defaults = {
        "enabled": True,
        "backend": surface.get("surface_backend", "grace"),
        "model": surface.get("surface_model", "GRACE-1L-OMAT"),
        "task": surface.get("surface_task", ""),
        "device": surface.get("surface_device", "cpu"),
        "gpu_id": surface.get("surface_gpu_id", 0),
        "tf_threads": surface.get("surface_tf_threads", 1),
        "omp_threads": surface.get("surface_omp_threads", 1),
        "relax": surface.get("relax_surface", True),
        "optimizer": surface.get("surface_optimizer", "bfgs"),
        "fmax": surface.get("surface_fmax", 0.05),
        "max_steps": surface.get("surface_max_steps", 300),
        "top_k_per_candidate": 10,
    }
    refine_defaults = {
        "enabled": False,
        "backend": "mace",
        "model": "mh-1",
        "task": "matpes_r2scan",
        "device": "cpu",
        "gpu_id": 0,
        "tf_threads": 1,
        "omp_threads": 1,
        "relax": True,
        "optimizer": "bfgs",
        "fmax": 0.03,
        "max_steps": 500,
        "top_k_per_candidate": 5,
    }
    surface["screen"] = _calculator_cfg(
        surface.get("screen", {}), "surface.screen", screen_defaults
    )
    surface["refine"] = _calculator_cfg(
        surface.get("refine", {}), "surface.refine", refine_defaults
    )

    if str(surface["orientation_mode"]).lower() not in {"explicit", "automatic"}:
        raise ValueError("[surface].orientation_mode must be explicit or automatic")
    if str(surface["termination_mode"]).lower() not in {"all", "first"}:
        raise ValueError("[surface].termination_mode must be all or first")
    if str(surface["dopant_variant_mode"]).lower() not in {"none", "co-dopant-depth"}:
        raise ValueError("[surface].dopant_variant_mode must be none or co-dopant-depth")
    if str(surface["placement_side"]).lower() not in {"top", "bottom", "both"}:
        raise ValueError("[surface].placement_side must be top, bottom, or both")
    zones = [str(x).lower() for x in surface["depth_zones"]]
    if not zones or any(x not in _VALID_ZONES for x in zones):
        raise ValueError("[surface].depth_zones may contain surface, subsurface, bulk")
    surface["depth_zones"] = zones
    surface["miller_list"] = [
        tuple(int(x) for x in miller) for miller in surface["miller_list"]
    ]
    return surface


def parse_surface_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Public validation/normalization helper used by the CLI and GUI."""
    return _parse_config(config)


def preview_surface_candidates(
    config: Mapping[str, Any],
    project_root: Path | str = Path("."),
) -> pd.DataFrame:
    """Return the bulk database rows selected by the surface configuration."""
    cfg = _parse_config(config)
    root = Path(project_root).expanduser().resolve()
    summary = Path(str(cfg["source_summary"])).expanduser()
    if not summary.is_absolute():
        summary = (root / summary).resolve()
    if not summary.exists():
        raise FileNotFoundError(f"[surface] Summary file not found: {summary}")
    database = pd.read_csv(summary)
    _validate_database_columns(database)
    return _select_candidates(database, cfg).copy()


def _millers(
    structure: Structure,
    cfg: Mapping[str, Any],
) -> List[Tuple[int, int, int]]:
    if str(cfg["orientation_mode"]).lower() == "explicit":
        values = list(cfg["miller_list"])
    else:
        values = [
            tuple(int(x) for x in hkl)
            for hkl in get_symmetrically_distinct_miller_indices(
                structure, int(cfg["max_miller"])
            )
        ]
        values = sorted(values, key=lambda hkl: (sum(abs(v) for v in hkl), hkl))
    return values[: int(cfg["max_orientations"])]


def _slabs(
    structure: Structure,
    hkl: Tuple[int, int, int],
    cfg: Mapping[str, Any],
) -> List[Structure]:
    generator = SlabGenerator(
        structure,
        hkl,
        float(cfg["min_slab_size"]),
        float(cfg["min_vacuum_size"]),
        center_slab=bool(cfg["center_slab"]),
        in_unit_planes=bool(cfg["in_unit_planes"]),
        lll_reduce=bool(cfg["lll_reduce"]),
        primitive=bool(cfg["primitive"]),
        reorient_lattice=bool(cfg["reorient_lattice"]),
    )
    slabs = list(generator.get_slabs(symmetrize=bool(cfg["symmetrize_slabs"])))
    if cfg["orthogonal_c"]:
        slabs = [slab.get_orthogonal_c_slab() for slab in slabs]
    if str(cfg["termination_mode"]).lower() == "first":
        slabs = slabs[:1]
    return slabs[: int(cfg["max_terminations_per_orientation"])]


def _host_dopants(
    structure: Structure,
    cfg: Mapping[str, Any],
) -> Tuple[str, List[str]]:
    counts: Dict[str, int] = {}
    for site in structure:
        symbol = site.specie.symbol
        counts[symbol] = counts.get(symbol, 0) + 1
    anions = {str(x) for x in cfg["anion_species"]}
    cations = {element: n for element, n in counts.items() if element not in anions}
    if not cations:
        raise ValueError("[surface] No cation species were found for dopant-depth scanning")
    host = str(cfg.get("host_species", "")).strip() or max(cations, key=cations.get)
    dopants = [str(x) for x in cfg["dopant_species"]] or sorted(
        element for element in cations if element != host
    )
    return host, dopants


def _cation_layers(
    structure: Structure,
    species: Sequence[str],
    tolerance_A: float,
) -> List[List[int]]:
    species_set = set(species)
    rows = sorted(
        [
            (i, float(site.coords[2]))
            for i, site in enumerate(structure)
            if site.specie.symbol in species_set
        ],
        key=lambda x: x[1],
    )
    if not rows:
        return []
    layers: List[List[int]] = []
    current, z0 = [rows[0][0]], rows[0][1]
    for idx, z in rows[1:]:
        if abs(z - z0) <= tolerance_A:
            current.append(idx)
        else:
            layers.append(current)
            current, z0 = [idx], z
    layers.append(current)
    return layers


def _zone_host_sites(
    structure: Structure,
    host: str,
    dopants: Sequence[str],
    cfg: Mapping[str, Any],
) -> Dict[str, List[int]]:
    layers = _cation_layers(
        structure,
        [host, *dopants],
        float(cfg["cation_layer_tolerance_A"]),
    )
    if not layers:
        return {zone: [] for zone in _VALID_ZONES}

    n = max(1, int(cfg["layers_per_zone"]))
    side = str(cfg["placement_side"]).lower()
    if side == "top":
        surface_layers = layers[-n:]
        subsurface_layers = layers[
            max(0, len(layers) - 2 * n) : max(0, len(layers) - n)
        ]
    elif side == "bottom":
        surface_layers = layers[:n]
        subsurface_layers = layers[n : 2 * n]
    else:
        surface_layers = layers[:n] + layers[-n:]
        subsurface_layers = layers[n : 2 * n] + layers[
            max(0, len(layers) - 2 * n) : max(0, len(layers) - n)
        ]

    center = 0.5 * (len(layers) - 1)
    bulk_layers = [
        layers[i]
        for i in sorted(
            range(len(layers)),
            key=lambda i: abs(i - center),
        )[:n]
    ]

    def host_only(group: Sequence[Sequence[int]]) -> List[int]:
        return [
            i
            for layer in group
            for i in layer
            if structure[i].specie.symbol == host
        ]

    return {
        "surface": host_only(surface_layers),
        "subsurface": host_only(subsurface_layers),
        "bulk": host_only(bulk_layers),
    }


def _move_dopant(
    structure: Structure,
    dopant: str,
    host: str,
    targets: Sequence[int],
) -> Tuple[Structure, Dict[str, Any]] | None:
    dopant_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == dopant
    ]
    targets = [i for i in targets if structure[i].specie.symbol == host]
    if not dopant_indices or not targets:
        return None

    distance, source, target = min(
        (float(structure.get_distance(i, j)), i, j)
        for i in dopant_indices
        for j in targets
    )
    moved = structure.copy()
    moved.replace(source, host)
    moved.replace(target, dopant)
    return moved, {
        "dopant": dopant,
        "from_index": source,
        "to_index": target,
        "swap_distance_A": distance,
    }


def _variants(
    slab: Structure,
    cfg: Mapping[str, Any],
) -> List[Tuple[str, Structure, Dict[str, Any]]]:
    if str(cfg["dopant_variant_mode"]).lower() == "none":
        return [("original", slab.copy(), {"target_zones": {}, "moves": []})]

    host, dopants = _host_dopants(slab, cfg)
    if not dopants:
        return [
            (
                "original",
                slab.copy(),
                {
                    "host_species": host,
                    "dopant_species": [],
                    "target_zones": {},
                    "moves": [],
                },
            )
        ]

    variants: List[Tuple[str, Structure, Dict[str, Any]]] = []
    seen: set[Tuple[str, ...]] = set()

    if cfg["include_original_variant"]:
        key = tuple(site.specie.symbol for site in slab)
        seen.add(key)
        variants.append(
            (
                "original",
                slab.copy(),
                {
                    "host_species": host,
                    "dopant_species": dopants,
                    "target_zones": {},
                    "moves": [],
                },
            )
        )

    for combo in itertools.product(cfg["depth_zones"], repeat=len(dopants)):
        current, moves, valid = slab.copy(), [], True
        for dopant, zone in zip(dopants, combo):
            sites = _zone_host_sites(current, host, dopants, cfg)
            moved = _move_dopant(current, dopant, host, sites[zone])
            if moved is None:
                valid = False
                break
            current, move = moved
            move["target_zone"] = zone
            moves.append(move)

        if not valid:
            continue

        key = tuple(site.specie.symbol for site in current)
        if key in seen:
            continue
        seen.add(key)

        label = "__".join(
            f"{dopant}-{zone}" for dopant, zone in zip(dopants, combo)
        )
        variants.append(
            (
                label,
                current,
                {
                    "host_species": host,
                    "dopant_species": dopants,
                    "target_zones": dict(zip(dopants, combo)),
                    "moves": moves,
                },
            )
        )
        if len(variants) >= int(cfg["max_dopant_variants_per_termination"]):
            break

    return variants


def _prepare_calculator(cfg: Mapping[str, Any], stage: str):
    check_backend_dependency(str(cfg["backend"]), stage_name=stage)
    prepare_backend_runtime(
        backend=str(cfg["backend"]),
        device=str(cfg["device"]),
        gpu_id=int(cfg["gpu_id"]),
        tf_threads=int(cfg["tf_threads"]),
        omp_threads=int(cfg["omp_threads"]),
    )
    return build_ase_calculator(
        backend=str(cfg["backend"]),
        model=str(cfg["model"]),
        task=str(cfg["task"]),
        device=str(cfg["device"]),
    )


def _evaluate(
    structure: Structure,
    fixed: Sequence[int],
    cfg: Mapping[str, Any],
    calculator: Any,
    outdir: Path,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator
    if fixed:
        atoms.set_constraint(FixAtoms(indices=list(fixed)))

    t0 = time.time()
    steps, converged = 0, True
    try:
        if cfg["relax"]:
            optimizer = _get_optimizer_class(str(cfg["optimizer"]))(
                atoms,
                logfile=str(outdir / "relax.log"),
                trajectory=str(outdir / "relax.traj"),
            )
            converged = bool(
                optimizer.run(
                    fmax=float(cfg["fmax"]),
                    steps=int(cfg["max_steps"]),
                )
            )
            steps = int(getattr(optimizer, "nsteps", 0))

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
        final_fmax = _compute_final_fmax_from_forces(forces, list(fixed))
        final_structure = AseAtomsAdaptor.get_structure(atoms)

        final_path = outdir / "POSCAR_relaxed"
        if cfg["relax"]:
            Poscar(final_structure).write_file(str(final_path))

        result = {
            "status": "ok",
            "energy_eV": energy,
            "converged": bool(converged or final_fmax <= float(cfg["fmax"])),
            "final_fmax_eV_per_A": final_fmax,
            "optimizer_steps": steps,
            "backend": cfg["backend"],
            "model": cfg["model"],
            "task": cfg["task"],
            "relaxed_structure_path": str(final_path) if cfg["relax"] else "",
            "walltime_s": time.time() - t0,
        }
    except Exception as exc:
        result = {
            "status": "fail",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    (outdir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def _bulk_energy(
    structure: Structure,
    cfg: Mapping[str, Any],
    calculator: Any,
    outdir: Path,
) -> float | None:
    single_point_cfg = dict(cfg)
    single_point_cfg["relax"] = False
    result = _evaluate(
        structure,
        [],
        single_point_cfg,
        calculator,
        outdir,
    )
    return float(result["energy_eV"]) if result.get("status") == "ok" else None


def _surface_energy(
    bulk: Structure,
    slab: Structure,
    slab_energy: float | None,
    bulk_energy: float | None,
) -> Dict[str, Any]:
    if slab_energy is None:
        return {"surface_energy_status": "missing_slab_energy"}
    if bulk_energy is None:
        return {"surface_energy_status": "missing_bulk_energy"}

    n_bulk, status, ratios = _compute_bulk_equivalent_factor(bulk, slab)
    if status != "ok" or n_bulk is None:
        return {
            "surface_energy_status": f"not_computable_{status}",
            "species_ratios_json": json.dumps(ratios, sort_keys=True),
        }

    area = float(
        np.linalg.norm(
            np.cross(
                slab.lattice.matrix[0],
                slab.lattice.matrix[1],
            )
        )
    )
    gamma_eV_A2, gamma_J_m2 = _compute_surface_energy(
        float(slab_energy),
        float(bulk_energy),
        float(n_bulk),
        area,
    )
    return {
        "surface_energy_status": "ok",
        "surface_energy_eV_A2": gamma_eV_A2,
        "surface_energy_J_m2": gamma_J_m2,
        "n_bulk_equiv": n_bulk,
        "species_ratios_json": json.dumps(ratios, sort_keys=True),
    }


def _base_record(
    row: pd.Series,
    bulk_path: Path,
    hkl: Tuple[int, int, int],
    term_id: int,
    variant_id: int,
    label: str,
    meta: Mapping[str, Any],
    variant_dir: Path,
    structure: Structure,
) -> Dict[str, Any]:
    thickness, vacuum, c_length = _estimate_slab_and_vacuum_thickness_A(structure)
    area = float(
        np.linalg.norm(
            np.cross(
                structure.lattice.matrix[0],
                structure.lattice.matrix[1],
            )
        )
    )
    return {
        "composition_tag": str(row["composition_tag"]),
        "candidate": str(row["candidate"]),
        "candidate_path": str(row["candidate_path"]),
        "bulk_structure_path": str(bulk_path),
        "E_form_norm": row.get("E_form_norm"),
        "bandgap_eV": row.get("bandgap_eV"),
        "miller_h": hkl[0],
        "miller_k": hkl[1],
        "miller_l": hkl[2],
        "termination_id": term_id,
        "variant_id": variant_id,
        "variant_label": label,
        "host_species": meta.get("host_species"),
        "dopant_species_json": json.dumps(meta.get("dopant_species", [])),
        "target_zones_json": json.dumps(
            meta.get("target_zones", {}),
            sort_keys=True,
        ),
        "dopant_moves_json": json.dumps(
            meta.get("moves", []),
            sort_keys=True,
        ),
        "n_atoms": len(structure),
        "surface_area_A2": area,
        "slab_thickness_A_est": thickness,
        "vacuum_thickness_A_est": vacuum,
        "c_length_A": c_length,
        "surface_variant_dir": str(variant_dir),
        "generated_structure_path": str(variant_dir / "POSCAR"),
    }


def _run_candidate(
    row: pd.Series,
    cfg: Mapping[str, Any],
    calc_cfg: Mapping[str, Any],
    calculator: Any,
    outdir: Path,
) -> List[Dict[str, Any]]:
    bulk_path = _resolve_bulk_structure_path(Path(str(row["candidate_path"])))
    bulk = Structure.from_file(bulk_path)
    candidate_dir = outdir / str(row["composition_tag"]) / str(row["candidate"])

    bulk_energy = _bulk_energy(
        bulk,
        calc_cfg,
        calculator,
        candidate_dir / "bulk_reference" / "screen",
    )

    records: List[Dict[str, Any]] = []
    for hkl in _millers(bulk, cfg):
        for term_id, slab in enumerate(_slabs(bulk, hkl, cfg), start=1):
            slab = _sort_structure_for_poscar(
                slab,
                preferred_order=cfg.get("poscar_order"),
            )
            for variant_id, (label, variant, meta) in enumerate(
                _variants(slab, cfg),
                start=1,
            ):
                variant = _sort_structure_for_poscar(
                    variant,
                    preferred_order=cfg.get("poscar_order"),
                )
                fixed = _select_fixed_atom_indices(variant, dict(cfg))
                variant_dir = (
                    candidate_dir
                    / f"hkl_{hkl[0]}_{hkl[1]}_{hkl[2]}"
                    / f"term_{term_id:03d}"
                    / f"variant_{variant_id:03d}_{label.replace('/', '-')}"
                )
                variant_dir.mkdir(parents=True, exist_ok=True)

                if fixed:
                    _write_poscar_with_selective_dynamics(
                        variant,
                        fixed,
                        variant_dir / "POSCAR",
                    )
                else:
                    Poscar(variant).write_file(str(variant_dir / "POSCAR"))

                if cfg["write_cif"]:
                    variant.to(
                        fmt="cif",
                        filename=str(variant_dir / "slab.cif"),
                    )

                result = _evaluate(
                    variant,
                    fixed,
                    calc_cfg,
                    calculator,
                    variant_dir / "screen",
                )
                final = variant
                if result.get("status") == "ok" and result.get(
                    "relaxed_structure_path"
                ):
                    final = Structure.from_file(result["relaxed_structure_path"])

                rec = _base_record(
                    row,
                    bulk_path,
                    hkl,
                    term_id,
                    variant_id,
                    label,
                    meta,
                    variant_dir,
                    final,
                )
                rec.update(
                    {
                        "screen_status": result.get("status"),
                        "screen_backend": calc_cfg["backend"],
                        "screen_model": calc_cfg["model"],
                        "screen_task": calc_cfg["task"],
                        "screen_energy_eV": result.get("energy_eV"),
                        "screen_energy_eV_atom": (
                            float(result["energy_eV"]) / len(final)
                            if result.get("energy_eV") is not None
                            else None
                        ),
                        "screen_converged": result.get("converged"),
                        "screen_final_fmax_eV_per_A": result.get(
                            "final_fmax_eV_per_A"
                        ),
                        "screen_optimizer_steps": result.get("optimizer_steps"),
                        "screen_relaxed_structure_path": result.get(
                            "relaxed_structure_path"
                        ),
                        "screen_bulk_reference_eV": bulk_energy,
                    }
                )
                rec.update(
                    {
                        f"screen_{key}": value
                        for key, value in _surface_energy(
                            bulk,
                            final,
                            result.get("energy_eV"),
                            bulk_energy,
                        ).items()
                    }
                )
                (variant_dir / "meta.json").write_text(
                    json.dumps(rec, indent=2, default=str),
                    encoding="utf-8",
                )
                records.append(rec)

    return records


def _add_segregation_metrics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add same-termination co-dopant segregation energies.

    The reference is the generated variant in which every explicitly moved
    dopant species targets a bulk-like cation layer. Because compared rows have
    identical composition, orientation, termination, atom count, and calculator,
    the difference in total energy is a well-defined segregation descriptor:

        E_seg = E_variant - E_all_bulk_like

    Negative values therefore mean that the requested surface/subsurface
    placement is preferred over the corresponding all-bulk-like placement.
    """
    out = df.copy()
    energy_col = f"{prefix}_energy_eV"
    out[f"{prefix}_segregation_reference_variant"] = ""
    out[f"{prefix}_segregation_reference_energy_eV"] = np.nan
    out[f"{prefix}_segregation_energy_eV"] = np.nan
    out[f"{prefix}_segregation_status"] = "missing_bulk_like_reference"

    if energy_col not in out.columns or "target_zones_json" not in out.columns:
        return out

    group_cols = [
        "composition_tag",
        "candidate",
        "miller_h",
        "miller_k",
        "miller_l",
        "termination_id",
    ]
    for _, group in out.groupby(group_cols, sort=False):
        reference_rows = []
        for idx, row in group.iterrows():
            try:
                zones = json.loads(str(row.get("target_zones_json", "{}") or "{}"))
            except json.JSONDecodeError:
                zones = {}
            if zones and all(str(zone).lower() == "bulk" for zone in zones.values()):
                energy = pd.to_numeric(
                    pd.Series([row.get(energy_col)]), errors="coerce"
                ).iloc[0]
                if pd.notna(energy):
                    reference_rows.append((idx, float(energy)))

        if not reference_rows:
            continue

        # There should normally be one all-bulk-like variant. If representative
        # generation ever produces more than one, use the lowest-energy one and
        # record the exact variant chosen.
        ref_idx, ref_energy = min(reference_rows, key=lambda item: item[1])
        ref_variant = str(out.loc[ref_idx].get("variant_label", ""))
        energies = pd.to_numeric(out.loc[group.index, energy_col], errors="coerce")
        valid = energies.notna()

        out.loc[group.index, f"{prefix}_segregation_reference_variant"] = ref_variant
        out.loc[group.index, f"{prefix}_segregation_reference_energy_eV"] = ref_energy
        out.loc[group.index[valid], f"{prefix}_segregation_energy_eV"] = (
            energies.loc[valid] - ref_energy
        )
        out.loc[group.index[valid], f"{prefix}_segregation_status"] = "ok"
        out.loc[group.index[~valid], f"{prefix}_segregation_status"] = "missing_energy"

    return out


def _rank(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    gamma = f"{prefix}_surface_energy_J_m2"
    status = f"{prefix}_surface_energy_status"
    out[gamma] = pd.to_numeric(out.get(gamma), errors="coerce")
    out[f"{prefix}_rankable"] = out[status].eq("ok") & out[gamma].notna()
    out[f"{prefix}_rank_overall"] = pd.NA
    out[f"{prefix}_rank_within_hkl"] = pd.NA

    for _, group in out.groupby(["composition_tag", "candidate"], sort=False):
        good = group[group[f"{prefix}_rankable"]].sort_values(gamma)
        out.loc[good.index, f"{prefix}_rank_overall"] = range(
            1,
            len(good) + 1,
        )

    for _, group in out.groupby(
        [
            "composition_tag",
            "candidate",
            "miller_h",
            "miller_k",
            "miller_l",
        ],
        sort=False,
    ):
        good = group[group[f"{prefix}_rankable"]].sort_values(gamma)
        out.loc[good.index, f"{prefix}_rank_within_hkl"] = range(
            1,
            len(good) + 1,
        )

    return out


def _topk(
    df: pd.DataFrame,
    prefix: str,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(["composition_tag", "candidate"], sort=False):
        good = group[group[f"{prefix}_rankable"].fillna(False)].copy()
        if not good.empty:
            rows.append(
                good.sort_values(f"{prefix}_rank_overall").head(top_k)
            )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else df.head(0).copy()
    )


def run_surface_scan(config: Mapping[str, Any]) -> Path | None:
    cfg = _parse_config(config)
    if not cfg["enabled"]:
        print("[surface] Stage disabled. Skipping.")
        return None

    summary_path = Path(cfg["source_summary"])
    if not summary_path.exists():
        raise FileNotFoundError(
            f"[surface] Summary file not found: {summary_path}"
        )

    database = pd.read_csv(summary_path)
    _validate_database_columns(database)
    selected = _select_candidates(database, cfg)
    if selected.empty:
        raise RuntimeError("[surface] No bulk candidates selected")

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    calculator = _prepare_calculator(cfg["screen"], "Surface screen")

    records: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        records.extend(
            _run_candidate(
                row,
                cfg,
                cfg["screen"],
                calculator,
                outdir,
            )
        )
        if len(records) > int(cfg["max_total_surfaces"]):
            raise RuntimeError(
                f"[surface] Exceeded max_total_surfaces={cfg['max_total_surfaces']}"
            )

    dataframe = _add_segregation_metrics(pd.DataFrame(records), "screen")
    dataframe = _rank(dataframe, "screen")
    summary = outdir / str(cfg["screen_summary_csv"])
    selected_path = outdir / str(cfg["screen_selected_csv"])
    dataframe.to_csv(summary, index=False)
    _topk(
        dataframe,
        "screen",
        int(cfg["screen"]["top_k_per_candidate"]),
    ).to_csv(selected_path, index=False)

    print(f"[surface] Screened {len(dataframe)} slab variants -> {summary}")
    return summary


def run_surface_refine(config: Mapping[str, Any]) -> Path | None:
    cfg = _parse_config(config)
    if not cfg["enabled"] or not cfg["refine"]["enabled"]:
        print("[surface] Refinement disabled. Skipping.")
        return None

    outdir = Path(cfg["outdir"])
    selected_path = outdir / str(cfg["screen_selected_csv"])
    if not selected_path.exists():
        raise FileNotFoundError(
            f"[surface] Run surface-scan first; missing {selected_path}"
        )

    selected = pd.read_csv(selected_path)
    calculator = _prepare_calculator(cfg["refine"], "Surface refine")
    bulk_cache: Dict[Tuple[str, str], float | None] = {}
    records: List[Dict[str, Any]] = []

    for _, row in selected.iterrows():
        rec = row.to_dict()
        key = (str(row["composition_tag"]), str(row["candidate"]))
        bulk_path = Path(str(row["bulk_structure_path"]))
        bulk = Structure.from_file(bulk_path)

        if key not in bulk_cache:
            candidate_dir = Path(str(row["surface_variant_dir"])).parents[2]
            bulk_cache[key] = _bulk_energy(
                bulk,
                cfg["refine"],
                calculator,
                candidate_dir / "bulk_reference" / "refine",
            )

        source = str(row.get("screen_relaxed_structure_path") or "")
        source_path = (
            Path(source)
            if source and Path(source).exists()
            else Path(str(row["generated_structure_path"]))
        )
        structure = Structure.from_file(source_path)
        fixed = _select_fixed_atom_indices(structure, dict(cfg))

        result = _evaluate(
            structure,
            fixed,
            cfg["refine"],
            calculator,
            Path(str(row["surface_variant_dir"])) / "refine",
        )
        final = structure
        if result.get("status") == "ok" and result.get(
            "relaxed_structure_path"
        ):
            final = Structure.from_file(result["relaxed_structure_path"])

        rec.update(
            {
                "refine_status": result.get("status"),
                "refine_backend": cfg["refine"]["backend"],
                "refine_model": cfg["refine"]["model"],
                "refine_task": cfg["refine"]["task"],
                "refine_energy_eV": result.get("energy_eV"),
                "refine_energy_eV_atom": (
                    float(result["energy_eV"]) / len(final)
                    if result.get("energy_eV") is not None
                    else None
                ),
                "refine_converged": result.get("converged"),
                "refine_final_fmax_eV_per_A": result.get(
                    "final_fmax_eV_per_A"
                ),
                "refine_optimizer_steps": result.get("optimizer_steps"),
                "refine_relaxed_structure_path": result.get(
                    "relaxed_structure_path"
                ),
                "refine_bulk_reference_eV": bulk_cache[key],
            }
        )
        rec.update(
            {
                f"refine_{name}": value
                for name, value in _surface_energy(
                    bulk,
                    final,
                    result.get("energy_eV"),
                    bulk_cache[key],
                ).items()
            }
        )
        records.append(rec)

    dataframe = _add_segregation_metrics(pd.DataFrame(records), "refine")
    dataframe = _rank(dataframe, "refine")
    summary = outdir / str(cfg["refine_summary_csv"])
    final_path = outdir / str(cfg["refine_selected_csv"])
    dataframe.to_csv(summary, index=False)
    _topk(
        dataframe,
        "refine",
        int(cfg["refine"]["top_k_per_candidate"]),
    ).to_csv(final_path, index=False)

    print(f"[surface] Refined {len(dataframe)} slab variants -> {summary}")
    return summary


__all__ = [
    "DEFAULT_MILLERS",
    "parse_surface_config",
    "preview_surface_candidates",
    "run_surface_scan",
    "run_surface_scan_from_toml",
    "run_surface_refine",
    "run_surface_refine_from_toml",
    "run_surface_workflow_from_toml",
]

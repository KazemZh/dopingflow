from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE, LBFGS, MDMin, QuasiNewton
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar

from dopingflow.ml_backends import (
    build_ase_calculator,
    check_backend_dependency,
    normalize_backend_config,
    prepare_backend_runtime,
    set_default_runtime_env,
)

set_default_runtime_env(tf_threads=1, omp_threads=1)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

EV_PER_A2_TO_J_PER_M2 = 16.02176634

def run_surface_from_toml(config_path: Path) -> None:
    """
    Load TOML config and run the surface generation stage.
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    run_surface(config)


def run_surface(config: Dict[str, Any]) -> None:
    """
    Main entry point for surface generation (+ optional surface relaxation).
    """
    surface_cfg = _parse_surface_config(config)

    if not surface_cfg["enabled"]:
        print("[surface] Stage disabled. Skipping.")
        return

    if surface_cfg["relax_surface"]:
        check_backend_dependency(surface_cfg["surface_backend"], stage_name="Surface relax")

    summary_path = Path(surface_cfg["source_summary"])
    if not summary_path.exists():
        raise FileNotFoundError(f"[surface] Summary file not found: {summary_path}")

    print(f"[surface] Reading candidate database: {summary_path}")
    df = pd.read_csv(summary_path)

    _validate_database_columns(df)

    df_sel = _select_candidates(df, surface_cfg)

    if df_sel.empty:
        raise RuntimeError("[surface] No candidates selected for surface generation.")

    print(f"[surface] Selected {len(df_sel)} candidate row(s).")

    outdir = Path(surface_cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    total_surfaces = 0

    for _, row in df_sel.iterrows():
        records = _process_candidate(row, surface_cfg, outdir)
        all_records.extend(records)
        total_surfaces += len(records)

        if total_surfaces > surface_cfg["max_total_surfaces"]:
            raise RuntimeError(
                f"[surface] Exceeded max_total_surfaces={surface_cfg['max_total_surfaces']}"
            )

    summary_csv = outdir / surface_cfg["summary_csv"]
    pd.DataFrame(all_records).to_csv(summary_csv, index=False)

    print(f"[surface] Done. Generated {total_surfaces} surfaces.")
    print(f"[surface] Summary saved to: {summary_csv}")


def _parse_surface_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    surface = cfg.get("surface", {}) or {}

    defaults = {
        "enabled": False,
        "source_summary": "results_database.csv",
        "composition_tag": None,
        "composition_tags": [],
        "selection_mode": "filters",   # id | ids | rank_range | top_n | filters
        "candidate_id": 1,
        "candidate_ids": [],
        "rank_start": 1,
        "rank_end": 10,
        "top_n": 5,
        "formation_energy_min": -1.0e9,
        "formation_energy_max": 1.0e9,
        "bandgap_min": -1.0e9,
        "bandgap_max": 1.0e9,
        "orientation_mode": "explicit",   # explicit | automatic
        "miller_list": [[1, 0, 0]],
        "max_miller": 1,
        "max_orientations": 6,
        "min_slab_size": 12.0,
        "min_vacuum_size": 15.0,
        "center_slab": True,
        "in_unit_planes": False,
        "lll_reduce": False,
        "primitive": False,
        "reorient_lattice": True,
        "orthogonal_c": True,
        "termination_mode": "all",   # all | first
        "max_terminations_per_orientation": 20,
        "outdir": "generated_surfaces",
        "write_poscar": True,
        "write_cif": False,
        "write_metadata_json": True,
        "summary_csv": "surface_summary.csv",
        "max_candidates": 20,
        "max_total_surfaces": 200,
        # fixing options
        "fix_atoms": False,
        "fix_region": "bottom",          # bottom | middle
        "fix_method": "layers",          # layers | thickness
        "fix_n_layers": 2,
        "fix_thickness_A": 4.0,
        "fix_layer_tolerance_A": 0.6,
        # surface relaxation options
        "relax_surface": False,
        "surface_backend": "m3gnet",
        "surface_model": "default",
        "surface_task": "",
        "surface_optimizer": "bfgs",
        "surface_device": "cpu",
        "surface_gpu_id": 0,
        "surface_tf_threads": 1,
        "surface_omp_threads": 1,
        "surface_fmax": 0.05,
        "surface_max_steps": 300,
        "surface_relaxed_filename": "POSCAR_relaxed",
        "surface_relax_log_filename": "surface_relax.log",
        "surface_relax_traj_filename": "surface_relax.traj",
        "surface_relax_meta_filename": "surface_relax.json",
    }

    for k, v in defaults.items():
        surface.setdefault(k, v)

    generate_cfg = cfg.get("generate", {}) or {}
    surface["poscar_order"] = generate_cfg.get("poscar_order", None)

    # normalize backend config exactly like relax.py style
    backend, model, task = normalize_backend_config(
        backend=str(surface["surface_backend"]).strip().lower(),
        model=str(surface["surface_model"]).strip(),
        task=str(surface["surface_task"]).strip(),
        section_name="surface",
    )
    surface["surface_backend"] = backend
    surface["surface_model"] = model
    surface["surface_task"] = task

    # validations
    if str(surface["selection_mode"]).lower() not in {"id", "ids", "rank_range", "top_n", "filters"}:
        raise ValueError('[surface].selection_mode must be one of: "id", "ids", "rank_range", "top_n", "filters"')

    if str(surface["orientation_mode"]).lower() not in {"explicit", "automatic"}:
        raise ValueError('[surface].orientation_mode must be either "explicit" or "automatic"')

    if str(surface["termination_mode"]).lower() not in {"all", "first"}:
        raise ValueError('[surface].termination_mode must be either "all" or "first"')

    if float(surface["min_slab_size"]) <= 0:
        raise ValueError("[surface].min_slab_size must be > 0")
    if float(surface["min_vacuum_size"]) <= 0:
        raise ValueError("[surface].min_vacuum_size must be > 0")
    if int(surface["max_miller"]) < 1:
        raise ValueError("[surface].max_miller must be >= 1")
    if int(surface["max_orientations"]) <= 0:
        raise ValueError("[surface].max_orientations must be > 0")
    if int(surface["max_terminations_per_orientation"]) <= 0:
        raise ValueError("[surface].max_terminations_per_orientation must be > 0")
    if int(surface["max_candidates"]) <= 0:
        raise ValueError("[surface].max_candidates must be > 0")
    if int(surface["max_total_surfaces"]) <= 0:
        raise ValueError("[surface].max_total_surfaces must be > 0")

    if str(surface["fix_region"]).lower() not in {"bottom", "middle"}:
        raise ValueError('[surface].fix_region must be either "bottom" or "middle"')
    if str(surface["fix_method"]).lower() not in {"layers", "thickness"}:
        raise ValueError('[surface].fix_method must be either "layers" or "thickness"')
    if int(surface["fix_n_layers"]) < 0:
        raise ValueError("[surface].fix_n_layers must be >= 0")
    if float(surface["fix_thickness_A"]) < 0:
        raise ValueError("[surface].fix_thickness_A must be >= 0")
    if float(surface["fix_layer_tolerance_A"]) <= 0:
        raise ValueError("[surface].fix_layer_tolerance_A must be > 0")

    if float(surface["surface_fmax"]) <= 0:
        raise ValueError("[surface].surface_fmax must be > 0")
    if int(surface["surface_max_steps"]) <= 0:
        raise ValueError("[surface].surface_max_steps must be > 0")
    if int(surface["surface_tf_threads"]) <= 0:
        raise ValueError("[surface].surface_tf_threads must be > 0")
    if int(surface["surface_omp_threads"]) <= 0:
        raise ValueError("[surface].surface_omp_threads must be > 0")
    if str(surface["surface_device"]).lower() not in {"cpu", "cuda"}:
        raise ValueError('[surface].surface_device must be either "cpu" or "cuda"')
    if int(surface["surface_gpu_id"]) < 0:
        raise ValueError("[surface].surface_gpu_id must be >= 0")
    if str(surface["surface_optimizer"]).lower() not in {"bfgs", "lbfgs", "fire", "mdmin", "quasinewton"}:
        raise ValueError(
            '[surface].surface_optimizer must be one of: "bfgs", "lbfgs", "fire", "mdmin", "quasinewton"'
        )

    return surface


def _validate_database_columns(df: pd.DataFrame) -> None:
    required = ["candidate", "candidate_path", "E_form_norm", "bandgap_eV", "composition_tag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[surface] Missing required columns in results database: {missing}")


def _candidate_label_from_numeric_id(candidate_id: int) -> str:
    return f"candidate_{int(candidate_id):03d}"


def _sort_structure_for_poscar(
    structure: Structure,
    preferred_order: List[str] | None = None,
) -> Structure:
    """
    Return a structure sorted for clean POSCAR writing.

    preferred_order example:
        ["Sb", "Ti", "Sn", "O"]

    Species not listed are placed afterward in alphabetical order.
    Within each species, sites are sorted by z, then y, then x.
    """
    if preferred_order is None:
        return structure.get_sorted_structure()

    order_map = {el: i for i, el in enumerate(preferred_order)}

    def sort_key(site):
        sp = site.specie.symbol
        return (
            order_map.get(sp, 10_000),
            sp,
            float(site.frac_coords[2]),
            float(site.frac_coords[1]),
            float(site.frac_coords[0]),
        )

    sorted_sites = sorted(structure.sites, key=sort_key)
    return Structure.from_sites(sorted_sites)


def _select_candidates(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    # 1) composition filter first
    selected_tags = []

    if cfg.get("composition_tag"):
        selected_tags.append(str(cfg["composition_tag"]))

    if cfg.get("composition_tags"):
        selected_tags.extend(str(x) for x in cfg["composition_tags"])

    if selected_tags:
        selected_tags = list(dict.fromkeys(selected_tags))
        df = df[df["composition_tag"].isin(selected_tags)]

    if df.empty:
        return df

    if "rank_relax_filtered" in df.columns:
        df["rank_relax_filtered"] = pd.to_numeric(df["rank_relax_filtered"], errors="coerce")
    if "E_form_norm" in df.columns:
        df["E_form_norm"] = pd.to_numeric(df["E_form_norm"], errors="coerce")
    if "bandgap_eV" in df.columns:
        df["bandgap_eV"] = pd.to_numeric(df["bandgap_eV"], errors="coerce")

    # 2) apply selection_mode inside chosen composition(s)
    mode = str(cfg.get("selection_mode", "filters")).lower()

    if mode == "id":
        label = _candidate_label_from_numeric_id(cfg["candidate_id"])
        df = df[df["candidate"] == label]

    elif mode == "ids":
        labels = {_candidate_label_from_numeric_id(x) for x in cfg.get("candidate_ids", [])}
        df = df[df["candidate"].isin(labels)]

    elif mode == "filters":
        form_min = cfg.get("formation_energy_min", -1.0e9)
        form_max = cfg.get("formation_energy_max", 1.0e9)
        bg_min = cfg.get("bandgap_min", -1.0e9)
        bg_max = cfg.get("bandgap_max", 1.0e9)

        df = df[
            (df["E_form_norm"] >= form_min)
            & (df["E_form_norm"] <= form_max)
            & (df["bandgap_eV"] >= bg_min)
            & (df["bandgap_eV"] <= bg_max)
        ]

        sort_cols = []
        ascending = []

        if "rank_relax_filtered" in df.columns:
            sort_cols.append("rank_relax_filtered")
            ascending.append(True)
        elif "E_form_norm" in df.columns:
            sort_cols.append("E_form_norm")
            ascending.append(True)

        if sort_cols:
            df = df.sort_values(sort_cols, ascending=ascending)

    elif mode == "top_n":
        if "rank_relax_filtered" in df.columns:
            df = df.sort_values("rank_relax_filtered", ascending=True)
        elif "E_form_norm" in df.columns:
            df = df.sort_values("E_form_norm", ascending=True)
        df = df.head(int(cfg["top_n"]))

    elif mode == "rank_range":
        if "rank_relax_filtered" in df.columns:
            df = df.sort_values("rank_relax_filtered", ascending=True)
        elif "E_form_norm" in df.columns:
            df = df.sort_values("E_form_norm", ascending=True)

        start = max(int(cfg["rank_start"]) - 1, 0)
        end = int(cfg["rank_end"])
        df = df.iloc[start:end]

    else:
        raise ValueError(f"[surface] Unknown selection_mode: {mode}")

    return df.head(int(cfg["max_candidates"]))


def _resolve_bulk_structure_path(candidate_path: Path) -> Path:
    """
    Try common locations for the relaxed structure to be used for slab generation.
    """
    candidates = [
        candidate_path / "02_relax" / "POSCAR",
        candidate_path / "02_relax" / "CONTCAR",
        candidate_path / "POSCAR",
        candidate_path / "CONTCAR",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "[surface] Could not locate a bulk structure file for candidate path: "
        f"{candidate_path}\nTried:\n" + "\n".join(str(p) for p in candidates)
    )


def _group_atoms_into_layers_by_z(
    structure: Structure,
    tolerance_A: float = 0.6,
) -> List[List[int]]:
    """
    Group atom indices into z-layers using Cartesian z positions.
    Returns layers ordered from bottom to top.
    """
    z_with_idx = sorted(
        [(i, float(site.coords[2])) for i, site in enumerate(structure.sites)],
        key=lambda x: x[1],
    )

    layers: List[List[int]] = []
    current_layer: List[int] = []
    current_z_ref = None

    for idx, z in z_with_idx:
        if current_z_ref is None:
            current_layer = [idx]
            current_z_ref = z
        elif abs(z - current_z_ref) <= tolerance_A:
            current_layer.append(idx)
        else:
            layers.append(current_layer)
            current_layer = [idx]
            current_z_ref = z

    if current_layer:
        layers.append(current_layer)

    return layers


def _select_fixed_atom_indices(
    structure: Structure,
    cfg: Dict[str, Any],
) -> List[int]:
    """
    Return site indices to fix according to cfg.
    """
    if not cfg.get("fix_atoms", False):
        return []

    region = str(cfg["fix_region"]).lower()
    method = str(cfg["fix_method"]).lower()

    if method == "layers":
        layers = _group_atoms_into_layers_by_z(
            structure,
            tolerance_A=float(cfg["fix_layer_tolerance_A"]),
        )

        n_layers = int(cfg["fix_n_layers"])
        if n_layers <= 0 or not layers:
            return []

        if region == "bottom":
            chosen_layers = layers[:n_layers]
        elif region == "middle":
            n_total = len(layers)
            start = max((n_total - n_layers) // 2, 0)
            end = min(start + n_layers, n_total)
            chosen_layers = layers[start:end]
        else:
            raise ValueError(f"[surface] Unknown fix_region: {region}")

        return sorted(idx for layer in chosen_layers for idx in layer)

    if method == "thickness":
        z_values = [float(site.coords[2]) for site in structure.sites]
        if not z_values:
            return []

        z_min = min(z_values)
        z_max = max(z_values)
        thickness = float(cfg["fix_thickness_A"])

        if thickness <= 0:
            return []

        if region == "bottom":
            cutoff = z_min + thickness
            return [i for i, site in enumerate(structure.sites) if float(site.coords[2]) <= cutoff]

        if region == "middle":
            z_center = 0.5 * (z_min + z_max)
            half = 0.5 * thickness
            return [
                i for i, site in enumerate(structure.sites)
                if abs(float(site.coords[2]) - z_center) <= half
            ]

        raise ValueError(f"[surface] Unknown fix_region: {region}")

    raise ValueError(f"[surface] Unknown fix_method: {method}")


def _write_poscar_with_selective_dynamics(
    structure: Structure,
    fixed_indices: List[int],
    out_path: Path,
) -> None:
    """
    Write one POSCAR file.
    Fixed atoms -> F F F
    Free atoms  -> T T T
    """
    fixed_set = set(fixed_indices)

    selective_dynamics = []
    for i in range(len(structure)):
        if i in fixed_set:
            selective_dynamics.append([False, False, False])  # F F F
        else:
            selective_dynamics.append([True, True, True])     # T T T

    poscar = Poscar(
        structure,
        selective_dynamics=selective_dynamics,
    )
    poscar.write_file(str(out_path))


def _estimate_slab_and_vacuum_thickness_A(structure: Structure) -> Tuple[float, float, float]:
    """
    Estimate slab thickness and vacuum thickness along Cartesian z.
    Works best when c is perpendicular to the slab plane.
    Returns:
        slab_thickness_A, vacuum_thickness_A, c_length_A
    """
    if len(structure) == 0:
        c_len = float(structure.lattice.c)
        return 0.0, c_len, c_len

    z_values = [float(site.coords[2]) for site in structure.sites]
    z_min = min(z_values)
    z_max = max(z_values)
    slab_thickness = z_max - z_min
    c_len = float(structure.lattice.c)
    vacuum_thickness = max(c_len - slab_thickness, 0.0)

    return slab_thickness, vacuum_thickness, c_len

def _species_counts(struct: Structure) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for site in struct.sites:
        el = site.species_string
        counts[el] = counts.get(el, 0) + 1
    return counts


def _compute_bulk_equivalent_factor(
    bulk_struct: Structure,
    slab_struct: Structure,
    rtol: float = 1e-6,
) -> tuple[float | None, str, Dict[str, float]]:
    """
    Determine whether the slab composition is proportional to the parent bulk composition.

    Returns:
        (n_bulk_equiv, status, ratios_by_species)

    status:
        - "ok"
        - "missing_species"
        - "extra_species"
        - "not_proportional"
        - "invalid_bulk"
    """
    bulk_counts = _species_counts(bulk_struct)
    slab_counts = _species_counts(slab_struct)

    if not bulk_counts:
        return None, "invalid_bulk", {}

    bulk_species = set(bulk_counts.keys())
    slab_species = set(slab_counts.keys())

    if slab_species != bulk_species:
        if not slab_species.issubset(bulk_species):
            return None, "extra_species", {}
        return None, "missing_species", {}

    ratios: Dict[str, float] = {}
    for sp in sorted(bulk_species):
        b = bulk_counts[sp]
        s = slab_counts[sp]
        if b <= 0:
            return None, "invalid_bulk", {}
        ratios[sp] = float(s) / float(b)

    vals = list(ratios.values())
    ref = vals[0]
    if not all(abs(v - ref) <= max(rtol * max(abs(ref), 1.0), 1e-12) for v in vals[1:]):
        return None, "not_proportional", ratios

    return ref, "ok", ratios


def _compute_surface_energy(
    slab_energy_eV: float,
    bulk_energy_eV: float,
    n_bulk_equiv: float,
    surface_area_A2: float,
) -> tuple[float, float]:
    """
    Compute surface energy for a stoichiometric symmetric slab.

    Returns:
        (surface_energy_eV_A2, surface_energy_J_m2)
    """
    if surface_area_A2 <= 0:
        raise ValueError("surface_area_A2 must be > 0")

    gamma_eV_A2 = (float(slab_energy_eV) - float(n_bulk_equiv) * float(bulk_energy_eV)) / (2.0 * float(surface_area_A2))
    gamma_J_m2 = gamma_eV_A2 * EV_PER_A2_TO_J_PER_M2
    return gamma_eV_A2, gamma_J_m2

def _get_optimizer_class(name: str):
    name = str(name).strip().lower()
    mapping = {
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "fire": FIRE,
        "mdmin": MDMin,
        "quasinewton": QuasiNewton,
    }
    if name not in mapping:
        raise ValueError(
            '[surface] surface_optimizer must be one of: "bfgs", "lbfgs", "fire", "mdmin", "quasinewton"'
        )
    return mapping[name]


def _compute_final_fmax_from_forces(
    forces: np.ndarray,
    fixed_indices: List[int],
) -> float:
    """
    Compute max force norm on non-fixed atoms.
    """
    if forces.size == 0:
        return 0.0

    mask = np.ones(len(forces), dtype=bool)
    if fixed_indices:
        mask[np.array(fixed_indices, dtype=int)] = False

    mobile_forces = forces[mask]
    if len(mobile_forces) == 0:
        return 0.0

    norms = np.linalg.norm(mobile_forces, axis=1)
    return float(np.max(norms))


def _relax_surface_structure(
    structure: Structure,
    fixed_indices: List[int],
    cfg: Dict[str, Any],
    term_dir: Path,
) -> Tuple[Structure, Dict[str, Any]]:
    """
    Relax a slab structure with the same backend strategy as relax.py,
    but with ASE FixAtoms constraints for selected slab atoms.
    """
    t0 = time.time()

    backend = cfg["surface_backend"]
    model = cfg["surface_model"]
    task = cfg["surface_task"]
    device = str(cfg["surface_device"]).lower()
    gpu_id = int(cfg["surface_gpu_id"])
    tf_threads = int(cfg["surface_tf_threads"])
    omp_threads = int(cfg["surface_omp_threads"])
    optimizer_name = str(cfg["surface_optimizer"]).lower()
    fmax = float(cfg["surface_fmax"])
    max_steps = int(cfg["surface_max_steps"])

    prepare_backend_runtime(
        backend=backend,
        device=device,
        gpu_id=gpu_id,
        tf_threads=tf_threads,
        omp_threads=omp_threads,
    )

    calculator = build_ase_calculator(
        backend=backend,
        model=model,
        task=task,
        device=device,
    )

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator

    if fixed_indices:
        atoms.set_constraint(FixAtoms(indices=fixed_indices))

    optimizer_cls = _get_optimizer_class(optimizer_name)

    traj_path = term_dir / str(cfg["surface_relax_traj_filename"])
    log_path = term_dir / str(cfg["surface_relax_log_filename"])

    dyn = optimizer_cls(
        atoms,
        trajectory=str(traj_path),
        logfile=str(log_path),
    )

    try:
        converged = bool(dyn.run(fmax=fmax, steps=max_steps))
    except TypeError:
        dyn.run(fmax=fmax, steps=max_steps)
        converged = False

    energy_relaxed_eV = float(atoms.get_potential_energy())
    forces = np.array(atoms.get_forces(apply_constraint=False), dtype=float)
    final_fmax = _compute_final_fmax_from_forces(forces, fixed_indices=fixed_indices)
    relaxed_structure = AseAtomsAdaptor.get_structure(atoms)

    meta = {
        "status": "ok",
        "method": f"{backend} + ASE optimizer",
        "backend": backend,
        "model": model,
        "task": task,
        "optimizer": optimizer_name,
        "device": device,
        "gpu_id": gpu_id if device == "cuda" else None,
        "fmax_target_eV_per_A": fmax,
        "max_steps": max_steps,
        "optimizer_steps": int(getattr(dyn, "nsteps", 0)),
        "final_fmax_eV_per_A": final_fmax,
        "converged": bool(converged or final_fmax <= fmax),
        "walltime_s": float(time.time() - t0),
        "energy_relaxed_eV": energy_relaxed_eV,
        "n_fixed_atoms": len(fixed_indices),
        "fixed_atom_indices": fixed_indices,
        "traj_path": str(traj_path),
        "log_path": str(log_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return relaxed_structure, meta


def _write_surface_relax_failure_meta(
    term_dir: Path,
    cfg: Dict[str, Any],
    fixed_indices: List[int],
    exc: Exception,
) -> Dict[str, Any]:
    meta = {
        "status": "fail",
        "method": f"{cfg['surface_backend']} + ASE optimizer",
        "backend": cfg["surface_backend"],
        "model": cfg["surface_model"],
        "task": cfg["surface_task"],
        "optimizer": cfg["surface_optimizer"],
        "device": cfg["surface_device"],
        "gpu_id": cfg["surface_gpu_id"] if str(cfg["surface_device"]).lower() == "cuda" else None,
        "n_fixed_atoms": len(fixed_indices),
        "fixed_atom_indices": fixed_indices,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    relax_meta_path = term_dir / str(cfg["surface_relax_meta_filename"])
    with open(relax_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def _process_candidate(row, cfg: Dict[str, Any], outdir: Path) -> List[Dict[str, Any]]:
    candidate_label = str(row["candidate"])
    candidate_path = Path(row["candidate_path"])
    composition_tag = str(row["composition_tag"])

    structure_path = _resolve_bulk_structure_path(candidate_path)
    structure = Structure.from_file(structure_path)

    millers = _build_miller_list(cfg)

    candidate_dir = outdir / composition_tag / candidate_label
    candidate_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    print(
        f"[surface] Candidate {candidate_label} | composition={composition_tag} "
        f"| structure={structure_path}"
    )

    for hkl in millers:
        slabs = _generate_slabs(structure, hkl, cfg)

        if not slabs:
            print(f"[surface]   hkl={hkl}: no slabs generated")
            continue

        hkl_dir = candidate_dir / f"hkl_{hkl[0]}_{hkl[1]}_{hkl[2]}"
        hkl_dir.mkdir(parents=True, exist_ok=True)

        for i, slab in enumerate(slabs, start=1):
            term_dir = hkl_dir / f"term_{i:03d}"
            term_dir.mkdir(parents=True, exist_ok=True)

            poscar_path = term_dir / "POSCAR"
            cif_path = term_dir / "slab.cif"
            meta_path = term_dir / "meta.json"

            slab_sorted = _sort_structure_for_poscar(
                slab,
                preferred_order=cfg.get("poscar_order"),
            )

            fixed_indices = _select_fixed_atom_indices(slab_sorted, cfg)

            # write original generated slab
            if cfg["write_poscar"]:
                if cfg.get("fix_atoms", False):
                    _write_poscar_with_selective_dynamics(
                        structure=slab_sorted,
                        fixed_indices=fixed_indices,
                        out_path=poscar_path,
                    )
                else:
                    slab_sorted.to(fmt="poscar", filename=str(poscar_path))

            if cfg["write_cif"]:
                slab_sorted.to(fmt="cif", filename=str(cif_path))

            surface_area = float(slab.surface_area) if hasattr(slab, "surface_area") else math.nan
            slab_thickness_A, vacuum_thickness_A, c_length_A = _estimate_slab_and_vacuum_thickness_A(slab_sorted)

            # optional relaxation
            relaxed_path = term_dir / str(cfg["surface_relaxed_filename"])
            relax_meta_path = term_dir / str(cfg["surface_relax_meta_filename"])

            relax_success = False
            relax_energy = None
            relax_final_fmax = None
            relax_steps = None
            relax_walltime = None

            if cfg.get("relax_surface", False):
                try:
                    relaxed_structure, relax_meta = _relax_surface_structure(
                        structure=slab_sorted,
                        fixed_indices=fixed_indices,
                        cfg=cfg,
                        term_dir=term_dir,
                    )

                    relaxed_sorted = _sort_structure_for_poscar(
                        relaxed_structure,
                        preferred_order=cfg.get("poscar_order"),
                    )

                    if cfg.get("fix_atoms", False):
                        _write_poscar_with_selective_dynamics(
                            structure=relaxed_sorted,
                            fixed_indices=fixed_indices,
                            out_path=relaxed_path,
                        )
                    else:
                        relaxed_sorted.to(fmt="poscar", filename=str(relaxed_path))

                    with open(relax_meta_path, "w", encoding="utf-8") as f:
                        json.dump(relax_meta, f, indent=2)

                    relax_success = bool(relax_meta.get("status") == "ok")
                    relax_energy = relax_meta.get("energy_relaxed_eV")
                    relax_final_fmax = relax_meta.get("final_fmax_eV_per_A")
                    relax_steps = relax_meta.get("optimizer_steps")
                    relax_walltime = relax_meta.get("walltime_s")

                except Exception as exc:
                    _write_surface_relax_failure_meta(
                        term_dir=term_dir,
                        cfg=cfg,
                        fixed_indices=fixed_indices,
                        exc=exc,
                    )
                    relax_success = False

            # ---------------------------------------
            # surface energy
            # ---------------------------------------
            surface_energy_status = "not_computed"
            surface_energy_eV_A2 = None
            surface_energy_J_m2 = None
            surface_energy_n_bulk_equiv = None
            surface_energy_formula = None
            surface_energy_species_ratios = {}
            bulk_energy_eV = None

            # Parent bulk relaxed energy from results database
            if "E_relaxed_eV" in row and pd.notna(row["E_relaxed_eV"]):
                bulk_energy_eV = float(row["E_relaxed_eV"])

            slab_energy_for_surface = None
            slab_structure_for_surface = slab_sorted

            # Surface energy requires a slab energy
            if cfg.get("relax_surface", False) and relax_success and relax_energy is not None:
                slab_energy_for_surface = float(relax_energy)
                if 'relaxed_sorted' in locals():
                    slab_structure_for_surface = relaxed_sorted
            else:
                surface_energy_status = "missing_slab_energy"

            if slab_energy_for_surface is not None:
                if bulk_energy_eV is None:
                    surface_energy_status = "missing_bulk_energy"
                else:
                    n_bulk_equiv, comp_status, ratios = _compute_bulk_equivalent_factor(
                        bulk_struct=structure,
                        slab_struct=slab_structure_for_surface,
                    )
                    surface_energy_species_ratios = ratios

                    if comp_status == "ok" and n_bulk_equiv is not None:
                        try:
                            gamma_eV_A2, gamma_J_m2 = _compute_surface_energy(
                                slab_energy_eV=slab_energy_for_surface,
                                bulk_energy_eV=bulk_energy_eV,
                                n_bulk_equiv=n_bulk_equiv,
                                surface_area_A2=surface_area,
                            )
                            surface_energy_status = "ok"
                            surface_energy_eV_A2 = gamma_eV_A2
                            surface_energy_J_m2 = gamma_J_m2
                            surface_energy_n_bulk_equiv = n_bulk_equiv
                            surface_energy_formula = "(E_slab - n*E_bulk) / (2A)"
                        except Exception as exc:
                            surface_energy_status = f"failed: {repr(exc)}"
                    else:
                        surface_energy_status = f"not_computable_{comp_status}"

            meta = {
                "composition_tag": composition_tag,
                "candidate": candidate_label,
                "candidate_path": str(candidate_path),
                "bulk_structure_path": str(structure_path),
                "formation_energy_norm": float(row["E_form_norm"]) if pd.notna(row["E_form_norm"]) else None,
                "bandgap_eV": float(row["bandgap_eV"]) if pd.notna(row["bandgap_eV"]) else None,
                "miller_index": list(hkl),
                "termination_id": i,
                "n_atoms": len(slab_sorted),
                "surface_area_A2": surface_area,
                "min_slab_size_input": float(cfg["min_slab_size"]),
                "min_vacuum_size_input": float(cfg["min_vacuum_size"]),
                "center_slab": bool(cfg["center_slab"]),
                "primitive": bool(cfg["primitive"]),
                "lll_reduce": bool(cfg["lll_reduce"]),
                "in_unit_planes": bool(cfg["in_unit_planes"]),
                "reorient_lattice": bool(cfg["reorient_lattice"]),
                "orthogonal_c": bool(cfg["orthogonal_c"]),
                "slab_thickness_A_est": slab_thickness_A,
                "vacuum_thickness_A_est": vacuum_thickness_A,
                "c_length_A": c_length_A,
                "poscar_path": str(poscar_path) if cfg["write_poscar"] else None,
                "cif_path": str(cif_path) if cfg["write_cif"] else None,
                "poscar_order": cfg.get("poscar_order"),
                "fix_atoms": bool(cfg.get("fix_atoms", False)),
                "fix_region": cfg.get("fix_region"),
                "fix_method": cfg.get("fix_method"),
                "fix_n_layers": int(cfg.get("fix_n_layers", 0)),
                "fix_thickness_A": float(cfg.get("fix_thickness_A", 0.0)),
                "fix_layer_tolerance_A": float(cfg.get("fix_layer_tolerance_A", 0.0)),
                "n_fixed_atoms": len(fixed_indices),
                "fixed_atom_indices": fixed_indices,
                "relax_surface": bool(cfg.get("relax_surface", False)),
                "surface_backend": cfg.get("surface_backend"),
                "surface_model": cfg.get("surface_model"),
                "surface_task": cfg.get("surface_task"),
                "surface_optimizer": cfg.get("surface_optimizer"),
                "surface_device": cfg.get("surface_device"),
                "surface_gpu_id": cfg.get("surface_gpu_id"),
                "surface_tf_threads": cfg.get("surface_tf_threads"),
                "surface_omp_threads": cfg.get("surface_omp_threads"),
                "surface_fmax": cfg.get("surface_fmax"),
                "surface_max_steps": cfg.get("surface_max_steps"),
                "surface_relaxed_path": str(relaxed_path) if cfg.get("relax_surface", False) else None,
                "surface_relax_meta_path": str(relax_meta_path) if cfg.get("relax_surface", False) else None,
                "surface_relax_success": relax_success,
                "surface_relaxed_energy_eV": relax_energy,
                "surface_relax_final_fmax_eV_per_A": relax_final_fmax,
                "surface_relax_optimizer_steps": relax_steps,
                "surface_relax_walltime_s": relax_walltime,
                "surface_energy_status": surface_energy_status,
                "surface_energy_eV_A2": surface_energy_eV_A2,
                "surface_energy_J_m2": surface_energy_J_m2,
                "surface_energy_reference_bulk_eV": bulk_energy_eV,
                "surface_energy_n_bulk_equiv": surface_energy_n_bulk_equiv,
                "surface_energy_area_A2": surface_area,
                "surface_energy_formula": surface_energy_formula,
                "surface_energy_species_ratios": surface_energy_species_ratios,                
            }

            if cfg["write_metadata_json"]:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

            records.append(
                {
                    "composition_tag": composition_tag,
                    "candidate": candidate_label,
                    "candidate_path": str(candidate_path),
                    "bulk_structure_path": str(structure_path),
                    "E_form_norm": row["E_form_norm"],
                    "bandgap_eV": row["bandgap_eV"],
                    "miller_h": hkl[0],
                    "miller_k": hkl[1],
                    "miller_l": hkl[2],
                    "termination_id": i,
                    "n_atoms": len(slab_sorted),
                    "surface_area_A2": surface_area,
                    "slab_thickness_A_est": slab_thickness_A,
                    "vacuum_thickness_A_est": vacuum_thickness_A,
                    "c_length_A": c_length_A,
                    "fix_atoms": bool(cfg.get("fix_atoms", False)),
                    "fix_region": cfg.get("fix_region"),
                    "fix_method": cfg.get("fix_method"),
                    "n_fixed_atoms": len(fixed_indices),
                    "relax_surface": bool(cfg.get("relax_surface", False)),
                    "surface_backend": cfg.get("surface_backend"),
                    "surface_model": cfg.get("surface_model"),
                    "surface_task": cfg.get("surface_task"),
                    "surface_optimizer": cfg.get("surface_optimizer"),
                    "surface_device": cfg.get("surface_device"),
                    "surface_gpu_id": cfg.get("surface_gpu_id"),
                    "surface_fmax": cfg.get("surface_fmax"),
                    "surface_max_steps": cfg.get("surface_max_steps"),
                    "surface_relax_success": relax_success,
                    "surface_relaxed_energy_eV": relax_energy,
                    "surface_relax_final_fmax_eV_per_A": relax_final_fmax,
                    "surface_relax_optimizer_steps": relax_steps,
                    "surface_relax_walltime_s": relax_walltime,
                    "poscar_path": str(poscar_path) if cfg["write_poscar"] else "",
                    "surface_relaxed_path": str(relaxed_path) if cfg.get("relax_surface", False) else "",
                    "cif_path": str(cif_path) if cfg["write_cif"] else "",
                    "meta_path": str(meta_path) if cfg["write_metadata_json"] else "",
                    "surface_relax_meta_path": str(relax_meta_path) if cfg.get("relax_surface", False) else "",
                    "surface_energy_status": surface_energy_status,
                    "surface_energy_eV_A2": surface_energy_eV_A2,
                    "surface_energy_J_m2": surface_energy_J_m2,
                    "surface_energy_reference_bulk_eV": bulk_energy_eV,
                    "surface_energy_n_bulk_equiv": surface_energy_n_bulk_equiv,                    
                }
            )

    return records


def _build_miller_list(cfg: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    if str(cfg["orientation_mode"]).lower() == "explicit":
        return [tuple(int(x) for x in m) for m in cfg["miller_list"]]

    millers: List[Tuple[int, int, int]] = []
    max_m = int(cfg["max_miller"])

    for h in range(0, max_m + 1):
        for k in range(0, max_m + 1):
            for l in range(0, max_m + 1):
                if (h, k, l) != (0, 0, 0):
                    millers.append((h, k, l))

    return millers[: int(cfg["max_orientations"])]


def _generate_slabs(
    structure: Structure,
    hkl: Tuple[int, int, int],
    cfg: Dict[str, Any],
):
    gen = SlabGenerator(
        initial_structure=structure,
        miller_index=hkl,
        min_slab_size=cfg["min_slab_size"],
        min_vacuum_size=cfg["min_vacuum_size"],
        center_slab=cfg["center_slab"],
        in_unit_planes=cfg["in_unit_planes"],
        lll_reduce=cfg["lll_reduce"],
        primitive=cfg["primitive"],
        reorient_lattice=cfg["reorient_lattice"],
    )

    slabs = gen.get_slabs()

    # enforce c ⟂ (a, b)
    if cfg.get("orthogonal_c", True):
        slabs = [slab.get_orthogonal_c_slab() for slab in slabs]

    mode = str(cfg["termination_mode"]).lower()
    if mode == "first":
        slabs = slabs[:1]
    elif mode != "all":
        raise ValueError(f"[surface] Unknown termination_mode: {cfg['termination_mode']}")

    return slabs[: int(cfg["max_terminations_per_orientation"])]
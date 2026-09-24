from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from dopingflow.ml_backends import normalize_backend_config
from dopingflow.surface import _select_fixed_atom_indices
from dopingflow.surface_staged import (
    _evaluate,
    _prepare_calculator,
    parse_surface_config,
    resolve_surface_output_dir,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

KB_EV_K = 8.617333262145e-5
LN10 = math.log(10.0)
_VALID_ZONES = {"surface", "subsurface", "bulk"}
_POTENTIAL_SCAN_COLUMNS = [
    "surface_id",
    "target_id",
    "dopant",
    "site_index",
    "detected_zone",
    "initial_dopant_zone",
    "initial_depth_from_selected_surface_A",
    "applied_potential_V",
    "potential_scale",
    "deltaG_leach_eV",
    "leaching_thermodynamically_favorable",
]


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    raise ValueError("Expected an array or comma-separated string")


def _map(value: Any, cast) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Expected a TOML inline table / mapping")
    return {str(k): cast(v) for k, v in value.items()}


def _calculator_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    surface = dict(config.get("surface", {}) or {})
    refine = dict(surface.get("refine", {}) or {})
    screen = dict(surface.get("screen", {}) or {})
    if bool(refine.get("enabled", False)):
        defaults = dict(
            backend="mace", model="mh-1", task="matpes_r2scan", device="cpu",
            gpu_id=0, tf_threads=1, omp_threads=1, optimizer="bfgs",
            fmax=0.03, max_steps=500,
        )
        defaults.update(refine)
        return defaults
    defaults = dict(
        backend="grace", model="GRACE-1L-OMAT", task="", device="cpu",
        gpu_id=0, tf_threads=1, omp_threads=1, optimizer="bfgs",
        fmax=0.05, max_steps=300,
    )
    defaults.update(screen)
    return defaults


def parse_leaching_config(
    config: Mapping[str, Any], project_root: Path | str = Path(".")
) -> dict[str, Any]:
    raw = dict(config.get("leaching", {}) or {})
    surface = dict(config.get("surface", {}) or {})
    structure = dict(config.get("structure", {}) or {})
    oxidation = dict(config.get("oxidation", {}) or {})
    conductivity = dict(config.get("conductivity", {}) or {})
    source_root_default = (
        str(raw.get("source_root", "")).strip()
        or str(surface.get("source_root", "")).strip()
        or str(conductivity.get("source_root", "")).strip()
        or str(oxidation.get("source_root", "")).strip()
        or str(structure.get("outdir", "random_structures")).strip()
    )
    calc = _calculator_defaults(config)
    defaults = dict(
        enabled=False,
        source_root=source_root_default,
        source_summary="",
        source_mode="auto",
        surface_include=[],
        dopant_species=[],
        zones=["surface"],
        placement_side=surface.get("placement_side", "top"),
        cation_layer_tolerance_A=surface.get("cation_layer_tolerance_A", 0.8),
        layers_per_zone=surface.get("layers_per_zone", 1),
        anion_species=surface.get("anion_species", ["O"]),
        max_sites_per_surface_species=12,
        oxygen_neighbor_cutoff_A=2.8,
        inherit_surface_fixed_atoms=True,
        reuse_surface_energy=True,
        relax_parent_if_recomputed=True,
        relax_removed_surface=True,
        resume_completed=True,
        outdir="09_leaching",
        summary_csv="leaching_summary.csv",
        aggregate_csv="leaching_surface_summary.csv",
        potential_scan_csv="leaching_potential_scan.csv",
        summary_json="leaching_results.json",
        preview_csv="leaching_preview.csv",
        reference_energies_file="reference_structures/reference_energies.json",
        metals_dir="reference_structures/metals",
        manual_metal_chemical_potentials_eV={},
        compute_missing_metal_references=True,
        relax_metal_reference=False,
        redox_reference_file="",
        oxidation_states={},
        standard_reduction_potentials_V_SHE={},
        aqueous_species={},
        ion_activities={},
        default_ion_activity=1e-6,
        temperature_K=298.15,
        pH=0.0,
        potential_scale="RHE",
        potentials_V=[1.23, 1.50, 1.70],
        **{k: calc[k] for k in (
            "backend", "model", "task", "device", "gpu_id", "tf_threads",
            "omp_threads", "optimizer", "fmax", "max_steps"
        )},
    )
    for key, value in defaults.items():
        raw.setdefault(key, value)

    raw["surface_include"] = _list(raw["surface_include"])
    raw["dopant_species"] = _list(raw["dopant_species"])
    raw["anion_species"] = _list(raw["anion_species"]) or ["O"]
    raw["zones"] = [x.lower() for x in _list(raw["zones"])]
    if not raw["zones"] or any(x not in _VALID_ZONES for x in raw["zones"]):
        raise ValueError("[leaching].zones may contain surface, subsurface, bulk")

    raw["source_mode"] = str(raw["source_mode"]).lower()
    valid_sources = {
        "auto", "final-selected", "screen-selected", "refine-summary", "screen-summary"
    }
    if raw["source_mode"] not in valid_sources:
        raise ValueError(f"[leaching].source_mode must be one of {sorted(valid_sources)}")
    raw["placement_side"] = str(raw["placement_side"]).lower()
    if raw["placement_side"] not in {"top", "bottom", "both"}:
        raise ValueError("[leaching].placement_side must be top, bottom, or both")

    backend, model, task = normalize_backend_config(
        backend=str(raw["backend"]).lower(), model=str(raw["model"]),
        task=str(raw["task"]), section_name="leaching",
    )
    raw.update(backend=backend, model=model, task=task)
    raw["device"] = str(raw["device"]).lower()
    raw["optimizer"] = str(raw["optimizer"]).lower()
    if raw["device"] not in {"cpu", "cuda"}:
        raise ValueError("[leaching].device must be cpu or cuda")
    if raw["optimizer"] not in {"bfgs", "lbfgs", "fire", "mdmin", "quasinewton"}:
        raise ValueError("[leaching].optimizer is not supported")

    for key in ("gpu_id", "tf_threads", "omp_threads", "max_steps", "layers_per_zone", "max_sites_per_surface_species"):
        raw[key] = int(raw[key])
    for key in ("fmax", "cation_layer_tolerance_A", "oxygen_neighbor_cutoff_A", "temperature_K", "pH", "default_ion_activity"):
        raw[key] = float(raw[key])
    if raw["gpu_id"] < 0 or raw["tf_threads"] <= 0 or raw["omp_threads"] <= 0:
        raise ValueError("[leaching] invalid runtime settings")
    if raw["max_steps"] <= 0 or raw["fmax"] <= 0 or raw["layers_per_zone"] <= 0:
        raise ValueError("[leaching] relaxation/layer settings must be positive")
    if raw["max_sites_per_surface_species"] <= 0 or raw["temperature_K"] <= 0:
        raise ValueError("[leaching] site cap and temperature must be positive")
    if raw["default_ion_activity"] <= 0:
        raise ValueError("[leaching].default_ion_activity must be positive")

    raw["potential_scale"] = str(raw["potential_scale"]).upper()
    if raw["potential_scale"] not in {"SHE", "RHE"}:
        raise ValueError("[leaching].potential_scale must be SHE or RHE")
    raw["potentials_V"] = [float(x) for x in (raw.get("potentials_V") or [])]
    raw["manual_metal_chemical_potentials_eV"] = _map(raw.get("manual_metal_chemical_potentials_eV"), float)
    raw["oxidation_states"] = _map(raw.get("oxidation_states"), int)
    raw["standard_reduction_potentials_V_SHE"] = _map(raw.get("standard_reduction_potentials_V_SHE"), float)
    raw["aqueous_species"] = _map(raw.get("aqueous_species"), str)
    raw["ion_activities"] = _map(raw.get("ion_activities"), float)
    if any(v <= 0 for v in raw["ion_activities"].values()):
        raise ValueError("[leaching].ion_activities must be positive")
    raw["project_root"] = Path(project_root).expanduser().resolve()
    return raw


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def resolve_leaching_output_dir(
    config: Mapping[str, Any],
    cfg: Mapping[str, Any] | None = None,
    project_root: Path | str = Path("."),
) -> Path:
    """Resolve leaching output below the user-selected source/parent root."""
    parsed = dict(cfg or parse_leaching_config(config, project_root))
    out = Path(str(parsed["outdir"])).expanduser()
    if out.is_absolute():
        return out.resolve()

    source = Path(str(parsed["source_root"])).expanduser()
    root = Path(project_root).expanduser().resolve()
    source_root = source.resolve() if source.is_absolute() else (root / source).resolve()
    return (source_root / out).resolve()


def resolve_surface_summary(config: Mapping[str, Any], cfg: Mapping[str, Any]) -> Path:
    root = Path(cfg["project_root"])
    source_root = _path(root, str(cfg["source_root"]))

    if str(cfg.get("source_summary", "")).strip():
        value = Path(str(cfg["source_summary"])).expanduser()
        p = value.resolve() if value.is_absolute() else (source_root / value).resolve()
        if not p.exists():
            raise FileNotFoundError(f"[leaching] Surface summary not found: {p}")
        return p

    surface_cfg = parse_surface_config(config)
    out = resolve_surface_output_dir(config, surface_cfg, root)
    choices = {
        "final-selected": out / str(surface_cfg.get("refine_selected_csv", "surface_final_selected.csv")),
        "screen-selected": out / str(surface_cfg.get("screen_selected_csv", "surface_screen_selected.csv")),
        "refine-summary": out / str(surface_cfg.get("refine_summary_csv", "surface_refine_summary.csv")),
        "screen-summary": out / str(surface_cfg.get("screen_summary_csv", "surface_screen_summary.csv")),
    }
    if cfg["source_mode"] != "auto":
        p = choices[str(cfg["source_mode"])]
        if not p.exists():
            raise FileNotFoundError(f"[leaching] Requested surface source does not exist: {p}")
        return p
    for name in ("final-selected", "screen-selected", "refine-summary", "screen-summary"):
        if choices[name].exists():
            return choices[name]
    raise FileNotFoundError("[leaching] Run `dopingflow surface` first or set source_summary")

def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.replace("/", "__"))


def _surface_id(row: Mapping[str, Any]) -> str:
    target = str(row.get("target_id") or f"{row.get('composition_tag','')}/{row.get('candidate','')}").strip("/")
    hkl = tuple(int(float(row.get(k, 0) or 0)) for k in ("miller_h", "miller_k", "miller_l"))
    term = int(float(row.get("termination_id", 0) or 0))
    variant = int(float(row.get("variant_id", 0) or 0))
    return f"{target}/hkl_{hkl[0]}_{hkl[1]}_{hkl[2]}/term_{term:03d}/variant_{variant:03d}_{row.get('variant_label','variant')}"


def _selected(row: Mapping[str, Any], patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    sid, target = _surface_id(row), str(row.get("target_id", ""))
    return any(fnmatch.fnmatchcase(value, pat) for pat in patterns for value in (sid, target, _safe(sid)))


def _structure_path(row: Mapping[str, Any], root: Path) -> tuple[Path, str]:
    for stage, key in (("refine", "refine_relaxed_structure_path"), ("screen", "screen_relaxed_structure_path"), ("generated", "generated_structure_path")):
        value = str(row.get(key) or "").strip()
        if value and value.lower() != "nan":
            p = _path(root, value)
            if p.exists():
                return p, stage
    raise FileNotFoundError(f"[leaching] No slab structure for {_surface_id(row)}")


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, type(default)):
        return value
    try:
        data = json.loads(str(value or ""))
        return data if isinstance(data, type(default)) else default
    except (TypeError, json.JSONDecodeError):
        return default


def _cation_layers(structure: Structure, species: Iterable[str], tol: float) -> list[list[int]]:
    allowed = set(species)
    rows = sorted(
        ((i, float(s.coords[2])) for i, s in enumerate(structure) if s.specie.symbol in allowed),
        key=lambda item: item[1],
    )
    if not rows:
        return []
    layers, current, z0 = [], [rows[0][0]], rows[0][1]
    for idx, z in rows[1:]:
        if abs(z - z0) <= tol:
            current.append(idx)
        else:
            layers.append(current); current, z0 = [idx], z
    layers.append(current)
    return layers


def _zones(structure: Structure, cations: Sequence[str], cfg: Mapping[str, Any]) -> dict[str, list[int]]:
    layers = _cation_layers(structure, cations, float(cfg["cation_layer_tolerance_A"]))
    if not layers:
        return {z: [] for z in _VALID_ZONES}
    n, side = max(1, int(cfg["layers_per_zone"])), str(cfg["placement_side"])
    if side == "top":
        surf, sub = layers[-n:], layers[max(0, len(layers)-2*n):max(0, len(layers)-n)]
    elif side == "bottom":
        surf, sub = layers[:n], layers[n:2*n]
    else:
        surf = layers[:n] + layers[-n:]
        sub = layers[n:2*n] + layers[max(0, len(layers)-2*n):max(0, len(layers)-n)]
    center = 0.5 * (len(layers) - 1)
    bulk = [layers[i] for i in sorted(range(len(layers)), key=lambda i: abs(i-center))[:n]]
    flatten = lambda groups: sorted({i for layer in groups for i in layer})
    return {"surface": flatten(surf), "subsurface": flatten(sub), "bulk": flatten(bulk)}


def _dopants(row: Mapping[str, Any], structure: Structure, cfg: Mapping[str, Any]) -> list[str]:
    wanted = list(cfg["dopant_species"]) or [str(x) for x in _json(row.get("dopant_species_json"), [])]
    present = {s.specie.symbol for s in structure}
    if wanted:
        return [x for x in wanted if x in present]
    host, anions = str(row.get("host_species") or ""), set(cfg["anion_species"])
    return sorted(x for x in present if x not in anions and x != host)


def enumerate_leaching_sites(
    row: Mapping[str, Any], structure: Structure, cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    dopants = _dopants(row, structure, cfg)
    anions = set(cfg["anion_species"])
    cations = sorted({s.specie.symbol for s in structure if s.specie.symbol not in anions})
    zones = _zones(structure, cations, cfg)
    allowed = {i for zone, ids in zones.items() if zone in set(cfg["zones"]) for i in ids}
    detected: dict[int, str] = {}
    for zone in ("surface", "subsurface", "bulk"):
        for idx in zones[zone]:
            detected.setdefault(idx, zone)
    declared = _json(row.get("target_zones_json"), {})
    records: list[dict[str, Any]] = []
    for dopant in dopants:
        ids = [i for i, s in enumerate(structure) if s.specie.symbol == dopant and i in allowed]
        ids.sort(key=lambda i: float(structure[i].coords[2]), reverse=cfg["placement_side"] != "bottom")
        for ordinal, idx in enumerate(ids[: int(cfg["max_sites_per_surface_species"])], 1):
            site = structure[idx]
            od = [float(structure.get_distance(idx, j)) for j, s in enumerate(structure) if s.specie.symbol in anions]
            cutoff = float(cfg["oxygen_neighbor_cutoff_A"])
            cation_z = [float(s.coords[2]) for s in structure if s.specie.symbol in cations]
            z = float(site.coords[2])
            if cfg["placement_side"] == "top":
                depth = max(cation_z) - z
            elif cfg["placement_side"] == "bottom":
                depth = z - min(cation_z)
            else:
                depth = min(max(cation_z) - z, z - min(cation_z))
            initial_zone = detected.get(idx, "")
            declared_zone = str(declared.get(dopant, ""))
            records.append(dict(
                dopant=dopant,
                site_index=idx,
                site_ordinal=ordinal,
                detected_zone=initial_zone,
                declared_target_zone=declared_zone,
                initial_dopant_zone=initial_zone,
                initial_dopant_zone_source="relaxed-surface-geometry",
                surface_variant_declared_zone=declared_zone,
                initial_site_index=idx,
                initial_cart_x_A=float(site.coords[0]),
                initial_cart_y_A=float(site.coords[1]),
                initial_cart_z_A=z,
                initial_frac_x=float(site.frac_coords[0]),
                initial_frac_y=float(site.frac_coords[1]),
                initial_frac_z=float(site.frac_coords[2]),
                initial_depth_from_selected_surface_A=float(max(depth, 0.0)),
                site_cart_x_A=float(site.coords[0]), site_cart_y_A=float(site.coords[1]), site_cart_z_A=z,
                site_frac_x=float(site.frac_coords[0]), site_frac_y=float(site.frac_coords[1]), site_frac_z=float(site.frac_coords[2]),
                oxygen_coordination_within_cutoff=sum(d <= cutoff for d in od),
                nearest_oxygen_distance_A=min(od) if od else None,
            ))
    return records


def _base(row: Mapping[str, Any], path: Path, stage: str, site: Mapping[str, Any]) -> dict[str, Any]:
    return dict(
        surface_id=_surface_id(row), target_id=str(row.get("target_id", "")), parent_id=str(row.get("parent_id", "")),
        structure_kind=str(row.get("structure_kind", "")), n_oxygen_vacancies=int(float(row.get("n_oxygen_vacancies", 0) or 0)),
        composition_tag=str(row.get("composition_tag", "")), candidate=str(row.get("candidate", "")),
        miller_h=int(float(row.get("miller_h", 0) or 0)), miller_k=int(float(row.get("miller_k", 0) or 0)), miller_l=int(float(row.get("miller_l", 0) or 0)),
        termination_id=int(float(row.get("termination_id", 0) or 0)), variant_id=int(float(row.get("variant_id", 0) or 0)),
        variant_label=str(row.get("variant_label", "")), surface_structure_path=str(path), surface_source_stage=stage,
        **dict(site),
    )


def preview_leaching_sites(config: Mapping[str, Any], project_root: Path | str = Path(".")) -> pd.DataFrame:
    cfg = parse_leaching_config(config, project_root)
    summary = resolve_surface_summary(config, cfg)
    root, records = Path(cfg["project_root"]), []
    for _, series in pd.read_csv(summary).iterrows():
        row = series.to_dict()
        if not _selected(row, cfg["surface_include"]):
            continue
        path, stage = _structure_path(row, root)
        structure = Structure.from_file(path)
        records.extend(_base(row, path, stage, site) for site in enumerate_leaching_sites(row, structure, cfg))
    frame = pd.DataFrame(records)
    frame.attrs["source_summary"] = str(summary)
    return frame


def _calculator_identity(cfg: Mapping[str, Any]) -> dict[str, str]:
    return {k: str(cfg[k]) for k in ("backend", "model", "task")}


def _same_calculator(row: Mapping[str, Any], stage: str, cfg: Mapping[str, Any]) -> bool:
    return stage in {"screen", "refine"} and all(
        str(row.get(f"{stage}_{key}", "")) == str(cfg[key]) for key in ("backend", "model", "task")
    )


def _stage_energy(row: Mapping[str, Any], stage: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(f"{stage}_energy_eV")]), errors="coerce").iloc[0]
    return float(value) if stage in {"screen", "refine"} and pd.notna(value) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _site_calculation_fingerprint(
    source_path: Path,
    site_index: int,
    dopant: str,
    cfg: Mapping[str, Any],
    surface_cfg: Mapping[str, Any],
) -> str:
    """Fingerprint the expensive parent/removal calculation inputs."""
    constraint_keys = (
        "fix_atoms", "fix_region", "fix_method", "fix_n_layers",
        "fix_thickness_A", "fix_layer_tolerance_A",
    )
    payload = {
        "schema_version": 1,
        "source_sha256": _sha256(source_path),
        "site_index": int(site_index),
        "dopant": str(dopant),
        "calculator": _calculator_identity(cfg),
        "relax_removed_surface": bool(cfg["relax_removed_surface"]),
        "optimizer": str(cfg["optimizer"]),
        "fmax": float(cfg["fmax"]),
        "max_steps": int(cfg["max_steps"]),
        "inherit_surface_fixed_atoms": bool(cfg["inherit_surface_fixed_atoms"]),
        "surface_constraints": {
            key: surface_cfg.get(key) for key in constraint_keys
        },
        "reuse_surface_energy": bool(cfg["reuse_surface_energy"]),
        "relax_parent_if_recomputed": bool(cfg["relax_parent_if_recomputed"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_completed_site_checkpoint(
    checkpoint_path: Path,
    base: Mapping[str, Any],
    cfg: Mapping[str, Any],
    source_path: Path,
    surface_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Return a reusable structural result from a completed per-site checkpoint."""
    if not bool(cfg.get("resume_completed", True)) or not checkpoint_path.exists():
        return None, "disabled-or-missing"
    try:
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable"
    if saved.get("status") != "ok":
        return None, "not-complete"

    try:
        energy = float(saved["removed_surface_energy_eV"])
    except (KeyError, TypeError, ValueError):
        return None, "missing-energy"

    for key in ("surface_id", "dopant"):
        if str(saved.get(key, "")) != str(base.get(key, "")):
            return None, f"{key}-mismatch"
    try:
        if int(saved.get("site_index")) != int(base.get("site_index")):
            return None, "site-index-mismatch"
    except (TypeError, ValueError):
        return None, "site-index-invalid"

    if any(str(saved.get(k, "")) != str(cfg[k]) for k in ("backend", "model", "task")):
        return None, "calculator-mismatch"

    saved_source = str(saved.get("surface_structure_path") or "").strip()
    current_source = str(base.get("surface_structure_path") or "").strip()
    if saved_source and current_source:
        try:
            if _path(Path(cfg["project_root"]), saved_source) != _path(
                Path(cfg["project_root"]), current_source
            ):
                return None, "surface-structure-mismatch"
        except Exception:
            if saved_source != current_source:
                return None, "surface-structure-mismatch"

    current_fp = _site_calculation_fingerprint(
        source_path,
        int(base["site_index"]),
        str(base["dopant"]),
        cfg,
        surface_cfg,
    )
    saved_fp = str(saved.get("calculation_fingerprint", "")).strip()
    if saved_fp and saved_fp != current_fp:
        return None, "fingerprint-mismatch"

    # Checkpoints written before restart support did not contain a fingerprint.
    # They are accepted only when the stable site/calculator identity matches.
    if bool(cfg["relax_removed_surface"]):
        relaxed = str(saved.get("removed_surface_relaxed_structure_path") or "").strip()
        if relaxed and not Path(relaxed).exists():
            return None, "relaxed-structure-missing"

    return {
        "status": "ok",
        "energy_eV": energy,
        "converged": saved.get("removed_surface_converged"),
        "final_fmax_eV_per_A": saved.get("removed_surface_final_fmax_eV_per_A"),
        "optimizer_steps": saved.get("removed_surface_optimizer_steps"),
        "relaxed_structure_path": saved.get("removed_surface_relaxed_structure_path", ""),
    }, ("fingerprint" if saved_fp else "legacy-identity")


def _global_metal_reference(element: str, cfg: Mapping[str, Any]) -> tuple[float | None, str]:
    path = _path(Path(cfg["project_root"]), str(cfg["reference_energies_file"]))
    if not path.exists():
        return None, "global-reference-file-missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "global-reference-file-unreadable"
    if any(str(data.get(k, "")) != str(cfg[k]) for k in ("backend", "model", "task")):
        return None, "global-reference-calculator-mismatch"
    entry = (data.get("references", {}) or {}).get(element, {})
    if entry.get("type") != "metal":
        return None, "global-metal-reference-missing"
    try:
        return float(entry["E_per_atom_eV"]), "global-reference-cache"
    except (KeyError, TypeError, ValueError):
        return None, "global-metal-reference-invalid"


def _metal_reference(
    element: str, cfg: Mapping[str, Any], calculator: Any, outdir: Path,
    run_cache: dict[str, tuple[float | None, str]],
) -> tuple[float | None, str]:
    if element in run_cache:
        return run_cache[element]
    manual = cfg["manual_metal_chemical_potentials_eV"]
    if element in manual:
        run_cache[element] = (float(manual[element]), "manual")
        return run_cache[element]
    energy, source = _global_metal_reference(element, cfg)
    if energy is not None or not cfg["compute_missing_metal_references"]:
        run_cache[element] = (energy, source)
        return run_cache[element]

    source_path = _path(Path(cfg["project_root"]), str(cfg["metals_dir"])) / f"{element}.POSCAR"
    if not source_path.exists():
        run_cache[element] = (None, f"metal-poscar-missing:{source_path}")
        return run_cache[element]
    cache_path = outdir / "references" / "metal_reference_energies.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        cache = {}
    entry = (cache.get("elements", {}) or {}).get(element, {})
    identity, source_hash = _calculator_identity(cfg), _sha256(source_path)
    if entry.get("calculator") == identity and entry.get("source_sha256") == source_hash and bool(entry.get("relax")) == bool(cfg["relax_metal_reference"]):
        try:
            run_cache[element] = (float(entry["E_per_atom_eV"]), "leaching-reference-cache")
            return run_cache[element]
        except (KeyError, TypeError, ValueError):
            pass
    structure = Structure.from_file(source_path)
    calc_cfg = dict(cfg); calc_cfg["relax"] = bool(cfg["relax_metal_reference"])
    result = _evaluate(structure, [], calc_cfg, calculator, outdir / "references" / element)
    if result.get("status") != "ok" or result.get("energy_eV") is None:
        run_cache[element] = (None, "metal-reference-calculation-failed")
        return run_cache[element]
    value = float(result["energy_eV"]) / len(structure)
    cache.setdefault("schema_version", 1)
    cache.setdefault("elements", {})[element] = dict(
        calculator=identity, source_path=str(source_path), source_sha256=source_hash,
        relax=bool(cfg["relax_metal_reference"]), E_total_eV=float(result["energy_eV"]),
        n_atoms=len(structure), E_per_atom_eV=value, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    run_cache[element] = (value, "leaching-reference-calculation")
    return run_cache[element]


def load_redox_references(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    value = str(cfg.get("redox_reference_file", "")).strip()
    if value:
        path = _path(Path(cfg["project_root"]), value)
        if not path.exists():
            raise FileNotFoundError(f"[leaching] Redox reference file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for element, entry in (data.get("elements", data) if isinstance(data, Mapping) else {}).items():
            if isinstance(entry, Mapping):
                refs[str(element)] = dict(entry)
    for el, val in cfg["oxidation_states"].items(): refs.setdefault(el, {})["oxidation_state"] = int(val)
    for el, val in cfg["standard_reduction_potentials_V_SHE"].items(): refs.setdefault(el, {})["standard_reduction_potential_V_SHE"] = float(val)
    for el, val in cfg["aqueous_species"].items(): refs.setdefault(el, {})["aqueous_species"] = str(val)
    return refs


def electrochemical_metrics(
    extraction_energy_eV: float, oxidation_state: int, standard_reduction_potential_V_SHE: float,
    *, ion_activity: float, temperature_K: float, pH: float,
) -> dict[str, float]:
    n = int(oxidation_state)
    if n <= 0 or ion_activity <= 0:
        raise ValueError("oxidation_state and ion_activity must be positive")
    thermal = KB_EV_K * float(temperature_K)
    activity_shift = thermal * math.log(float(ion_activity)) / n
    u_she = float(standard_reduction_potential_V_SHE) + float(extraction_energy_eV) / n + activity_shift
    rhe_shift = thermal * LN10 * float(pH)
    return dict(
        nernst_activity_shift_V=activity_shift, rhe_she_shift_V=rhe_shift,
        dissolution_potential_V_SHE=u_she, dissolution_potential_V_RHE=u_she + rhe_shift,
    )


def leaching_delta_g_eV(
    extraction_energy_eV: float, oxidation_state: int, standard_reduction_potential_V_SHE: float,
    applied_potential_V: float, *, potential_scale: str, ion_activity: float,
    temperature_K: float, pH: float,
) -> float:
    n, thermal = int(oxidation_state), KB_EV_K * float(temperature_K)
    if n <= 0 or ion_activity <= 0:
        raise ValueError("oxidation_state and ion_activity must be positive")
    u_she = float(applied_potential_V)
    if str(potential_scale).upper() == "RHE":
        u_she -= thermal * LN10 * float(pH)
    elif str(potential_scale).upper() != "SHE":
        raise ValueError("potential_scale must be SHE or RHE")
    return float(extraction_energy_eV) + n * (float(standard_reduction_potential_V_SHE) - u_she) + thermal * math.log(float(ion_activity))


def _parent(
    row: Mapping[str, Any], structure: Structure, stage: str, cfg: Mapping[str, Any],
    calculator: Any, outdir: Path, surface_cfg: Mapping[str, Any],
) -> tuple[float | None, Structure, str, dict[str, Any] | None]:
    if cfg["reuse_surface_energy"] and _same_calculator(row, stage, cfg):
        energy = _stage_energy(row, stage)
        if energy is not None:
            return energy, structure, f"reused-{stage}-surface-energy", None
    fixed = _select_fixed_atom_indices(structure, dict(surface_cfg)) if cfg["inherit_surface_fixed_atoms"] else []
    calc_cfg = dict(cfg); calc_cfg["relax"] = bool(cfg["relax_parent_if_recomputed"])
    result = _evaluate(structure, fixed, calc_cfg, calculator, outdir / "parent_recompute")
    if result.get("status") != "ok" or result.get("energy_eV") is None:
        return None, structure, "parent-calculation-failed", result
    final = structure
    path = str(result.get("relaxed_structure_path") or "")
    if path and Path(path).exists(): final = Structure.from_file(path)
    return float(result["energy_eV"]), final, "recomputed", result


def _potential_scan_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a stable-schema potential-scan table, even when no rows exist."""
    return pd.DataFrame(list(rows), columns=_POTENTIAL_SCAN_COLUMNS)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    cols = ["surface_id", "target_id", "composition_tag", "candidate", "miller_h", "miller_k", "miller_l", "termination_id", "variant_id", "variant_label", "dopant", "initial_dopant_zone"]
    rows = []
    for keys, group in df.groupby(cols, dropna=False, sort=False):
        rec = dict(zip(cols, keys))
        numeric = lambda col: pd.to_numeric(group[col], errors="coerce") if col in group else pd.Series(np.nan, index=group.index)
        ext, she, rhe = numeric("extraction_energy_eV"), numeric("dissolution_potential_V_SHE"), numeric("dissolution_potential_V_RHE")
        rec.update(
            n_leaching_sites=len(group), n_completed_sites=int(group["status"].eq("ok").sum()),
            minimum_extraction_energy_eV=float(ext.min()) if ext.notna().any() else np.nan,
            minimum_dissolution_potential_V_SHE=float(she.min()) if she.notna().any() else np.nan,
            minimum_dissolution_potential_V_RHE=float(rhe.min()) if rhe.notna().any() else np.nan,
        )
        if ext.notna().any():
            vulnerable = group.loc[ext.idxmin()]
            rec.update(
                most_vulnerable_site_index=int(vulnerable["site_index"]),
                most_vulnerable_detected_zone=str(vulnerable["detected_zone"]),
                most_vulnerable_initial_depth_A=float(vulnerable["initial_depth_from_selected_surface_A"]),
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def run_leaching(
    config: Mapping[str, Any], project_root: Path | str = Path("."), *, dry_run: bool = False
) -> Path | None:
    cfg = parse_leaching_config(config, project_root)
    if not cfg["enabled"]:
        print("[leaching] Stage disabled. Skipping."); return None
    source = resolve_surface_summary(config, cfg)
    preview = preview_leaching_sites(config, project_root)
    outdir = resolve_leaching_output_dir(config, cfg, project_root); outdir.mkdir(parents=True, exist_ok=True)
    preview_path = outdir / str(cfg["preview_csv"]); preview.to_csv(preview_path, index=False)
    if dry_run:
        print(f"[leaching] Dry run: {len(preview)} site(s) -> {preview_path}"); return preview_path
    if preview.empty:
        raise RuntimeError("[leaching] No dopant sites matched the selected surfaces/zones")

    source_df = pd.read_csv(source)
    rows = {_surface_id(r.to_dict()): r.to_dict() for _, r in source_df.iterrows()}
    calculator = _prepare_calculator(cfg, "Dopant leaching")
    surface_cfg = parse_surface_config(config)
    redox = load_redox_references(cfg)
    metal_cache: dict[str, tuple[float | None, str]] = {}
    parent_cache: dict[str, tuple[float | None, Structure, str]] = {}
    records, potential_rows = [], []
    n_reused, n_calculated = 0, 0

    for _, preview_row in preview.iterrows():
        base, sid = preview_row.to_dict(), str(preview_row["surface_id"])
        row = rows[sid]
        source_path = _path(Path(cfg["project_root"]), str(base["surface_structure_path"]))
        surface_dir = outdir / "surfaces" / _safe(sid)
        if sid not in parent_cache:
            energy, final, energy_source, result = _parent(
                row, Structure.from_file(source_path), str(base["surface_source_stage"]),
                cfg, calculator, surface_dir, surface_cfg,
            )
            parent_cache[sid] = (energy, final, energy_source)
            if result is not None:
                surface_dir.mkdir(parents=True, exist_ok=True)
                (surface_dir / "parent_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        parent_energy, parent_structure, parent_energy_source = parent_cache[sid]
        idx, dopant = int(base["site_index"]), str(base["dopant"])
        rec = dict(base)
        if idx >= len(parent_structure) or parent_structure[idx].specie.symbol != dopant:
            rec.update(status="site-mapping-failed", parent_energy_eV=parent_energy, parent_energy_source=parent_energy_source)
            records.append(rec); continue

        site_dir = surface_dir / f"{dopant}_site_{idx:04d}"; site_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = site_dir / "leaching_result.json"
        fingerprint = _site_calculation_fingerprint(source_path, idx, dopant, cfg, surface_cfg)
        result, checkpoint_mode = _load_completed_site_checkpoint(
            checkpoint_path, base, cfg, source_path, surface_cfg
        )
        if result is not None:
            n_reused += 1
            print(
                f"[leaching] RESUME {sid} {dopant} site {idx}: "
                f"reusing completed checkpoint ({checkpoint_mode})"
            )
        else:
            removed = parent_structure.copy(); removed.remove_sites([idx])
            Poscar(removed).write_file(str(site_dir / "POSCAR_removed_unrelaxed"))
            fixed = _select_fixed_atom_indices(removed, dict(surface_cfg)) if cfg["inherit_surface_fixed_atoms"] else []
            calc_cfg = dict(cfg); calc_cfg["relax"] = bool(cfg["relax_removed_surface"])
            result = _evaluate(removed, fixed, calc_cfg, calculator, site_dir / "relax")
            n_calculated += 1

        rec.update(
            status="ok" if result.get("status") == "ok" else "calculation-failed",
            backend=cfg["backend"], model=cfg["model"], task=cfg["task"],
            parent_energy_eV=parent_energy, parent_energy_source=parent_energy_source,
            removed_surface_energy_eV=result.get("energy_eV"), removed_surface_converged=result.get("converged"),
            removed_surface_final_fmax_eV_per_A=result.get("final_fmax_eV_per_A"), removed_surface_optimizer_steps=result.get("optimizer_steps"),
            removed_surface_relaxed_structure_path=result.get("relaxed_structure_path"), site_output_dir=str(site_dir),
            calculation_fingerprint=fingerprint,
            calculation_reused_from_checkpoint=result is not None and checkpoint_mode in {"fingerprint", "legacy-identity"},
            checkpoint_compatibility=checkpoint_mode,
        )
        # Persist the expensive structural result immediately. If the process
        # stops during reference/electrochemical post-processing, this site can
        # still be resumed without repeating its relaxation.
        checkpoint_path.write_text(
            json.dumps(rec, indent=2, default=str),
            encoding="utf-8",
        )

        mu, mu_source = _metal_reference(dopant, cfg, calculator, outdir, metal_cache)
        rec.update(metal_reference_eV_atom=mu, metal_reference_source=mu_source, extraction_status="not-computable")
        extraction = None
        if parent_energy is not None and result.get("status") == "ok" and result.get("energy_eV") is not None and mu is not None:
            extraction = float(result["energy_eV"]) + float(mu) - float(parent_energy)
            rec.update(extraction_energy_eV=extraction, extraction_status="ok")

        ref = redox.get(dopant, {})
        n, e0 = ref.get("oxidation_state"), ref.get("standard_reduction_potential_V_SHE")
        activity = float(cfg["ion_activities"].get(dopant, cfg["default_ion_activity"]))
        rec.update(
            aqueous_species=str(ref.get("aqueous_species", f"{dopant} ion")), oxidation_state=n,
            standard_reduction_potential_V_SHE=e0, redox_reference_source=str(ref.get("source", "user-config" if (n is not None or e0 is not None) else "")),
            ion_activity=activity, temperature_K=cfg["temperature_K"], pH=cfg["pH"], electrochemical_status="missing-redox-reference",
        )
        if extraction is not None and n is not None and e0 is not None:
            rec.update(electrochemical_metrics(extraction, int(n), float(e0), ion_activity=activity, temperature_K=cfg["temperature_K"], pH=cfg["pH"]))
            rec["electrochemical_status"] = "ok"
            for potential in cfg["potentials_V"]:
                dg = leaching_delta_g_eV(extraction, int(n), float(e0), potential, potential_scale=cfg["potential_scale"], ion_activity=activity, temperature_K=cfg["temperature_K"], pH=cfg["pH"])
                potential_rows.append(dict(
                    surface_id=sid, target_id=rec["target_id"], dopant=dopant, site_index=idx,
                    detected_zone=rec["detected_zone"], initial_dopant_zone=rec["initial_dopant_zone"],
                    initial_depth_from_selected_surface_A=rec["initial_depth_from_selected_surface_A"], applied_potential_V=potential,
                    potential_scale=cfg["potential_scale"], deltaG_leach_eV=dg,
                    leaching_thermodynamically_favorable=bool(dg < 0),
                ))
        (site_dir / "leaching_result.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        records.append(rec)

    results = pd.DataFrame(records)
    for col, rank in (("extraction_energy_eV", "extraction_vulnerability_rank"), ("dissolution_potential_V_SHE", "dissolution_vulnerability_rank")):
        if col in results:
            results[col] = pd.to_numeric(results[col], errors="coerce")
            results[rank] = results.groupby(["target_id", "dopant"], dropna=False)[col].rank(method="min", ascending=True)
            results[f"{rank}_within_initial_zone"] = results.groupby(
                ["target_id", "dopant", "initial_dopant_zone"], dropna=False
            )[col].rank(method="min", ascending=True)

    summary = outdir / str(cfg["summary_csv"])
    results.to_csv(summary, index=False)
    _aggregate(results).to_csv(outdir / str(cfg["aggregate_csv"]), index=False)
    _potential_scan_frame(potential_rows).to_csv(
        outdir / str(cfg["potential_scan_csv"]),
        index=False,
    )
    payload = dict(
        schema_version=1, source_summary=str(source), calculator=_calculator_identity(cfg),
        thermodynamic_model="metal-referenced extraction + user-supplied M^z+/M redox reference",
        potential_scale=cfg["potential_scale"], temperature_K=cfg["temperature_K"], pH=cfg["pH"],
        potentials_V=cfg["potentials_V"],
        resume_completed=bool(cfg["resume_completed"]),
        n_reused_checkpoints=n_reused,
        n_calculated_this_run=n_calculated,
        notes=[
            "Extraction energy: E(slab-M)+mu_M(metal)-E(slab+M).",
            "Electrochemical values are omitted unless oxidation state and standard reduction potential are supplied.",
            "Simple-ion model excludes explicit solvent, charged slabs, surface hydroxylation, aqueous complex speciation, kinetic barriers, and multi-atom pathways.",
        ], results=records,
    )
    (outdir / str(cfg["summary_json"])).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(
        f"[leaching] Analyzed {len(results)} dopant-removal site(s) "
        f"({n_reused} reused, {n_calculated} calculated this run) -> {summary}"
    )
    return summary


def run_leaching_from_toml(config_path: Path, *, dry_run: bool = False) -> Path | None:
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, "rb") as handle:
        config = tomllib.load(handle)
    return run_leaching(config, config_path.parent, dry_run=dry_run)


__all__ = [
    "KB_EV_K", "parse_leaching_config", "resolve_surface_summary", "enumerate_leaching_sites",
    "preview_leaching_sites", "resolve_leaching_output_dir", "load_redox_references", "electrochemical_metrics",
    "leaching_delta_g_eV", "run_leaching", "run_leaching_from_toml",
]

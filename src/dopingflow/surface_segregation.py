"""Finite-temperature surface segregation Monte Carlo for selected terminations."""

from __future__ import annotations

import csv
import fnmatch
import json
import logging
import math
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from dopingflow.ml_backends import normalize_backend_config
from dopingflow.surface_staged import (
    _cation_layers,
    _classify_cation_layer,
    ensure_surface_ids,
    parse_surface_config,
    resolve_surface_output_dir,
)

log = logging.getLogger(__name__)

K_B_EV_PER_K = 8.617333262145e-5
EV_TO_KJ_MOL = 96.4853321233
_ZONES = ("surface", "subsurface", "bulk")


@dataclass(frozen=True)
class SurfaceSegregationTarget:
    surface_id: str
    target_id: str
    structure_path: Path
    source_stage: str
    miller: tuple[int, int, int]
    termination_id: int
    termination_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def safe_id(self) -> str:
        text = self.surface_id.replace("\\", "__").replace("/", "__")
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


@dataclass(frozen=True)
class SurfaceSegregationConfig:
    root: Path
    source_root: Path
    source_summary: Path
    source_mode: str
    output_dir: Path
    enabled: bool
    surface_include: tuple[str, ...]
    max_surfaces: int
    host_species: str
    anion_species: tuple[str, ...]
    dopant_species: tuple[str, ...]
    cation_layer_tolerance_A: float
    dopant_depth_layers: int
    temperature_K: float
    steps: int
    burn_in: int
    sample_interval: int
    trace_interval: int
    progress_interval: int
    seed: int
    backend: str
    model: str
    task: str
    device: str
    gpu_id: int
    tf_threads: int
    omp_threads: int
    parallel_surfaces: int
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SwapMCResult:
    final_structure: Structure
    best_structure: Structure
    start_energy_eV: float
    final_energy_eV: float
    best_energy_eV: float
    attempted_moves: int
    accepted_moves: int
    attempted_by_dopant: dict[str, int]
    accepted_by_dopant: dict[str, int]
    n_samples: int
    site_counts: dict[str, dict[int, int]]
    zone_counts: dict[str, dict[str, int]]
    trace: tuple[dict[str, Any], ...]
    zone_trace: tuple[dict[str, Any], ...]


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(x).strip() for x in value]
    else:
        raise ValueError("Expected a string or array")
    return tuple(dict.fromkeys(x for x in items if x))


def _positive_int(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _surface_source_root(raw: Mapping[str, Any], root: Path) -> Path:
    section = dict(raw.get("surface_segregation", {}) or {})
    surface = dict(raw.get("surface", {}) or {})
    structure = dict(raw.get("structure", {}) or {})
    source = (
        str(section.get("source_root", "")).strip()
        or str(surface.get("source_root", "")).strip()
        or str(structure.get("outdir", "random_structures")).strip()
    )
    return _resolve_path(source, root)


def _surface_table_candidates(
    raw: Mapping[str, Any],
    root: Path,
) -> dict[str, Path]:
    surface_cfg = parse_surface_config(raw)
    surface_root = resolve_surface_output_dir(raw, surface_cfg, root)
    return {
        "final-selected": surface_root / str(surface_cfg["refine_selected_csv"]),
        "refine-summary": surface_root / str(surface_cfg["refine_summary_csv"]),
        "screen-selected": surface_root / str(surface_cfg["screen_selected_csv"]),
        "screen-summary": surface_root / str(surface_cfg["screen_summary_csv"]),
    }


def resolve_surface_segregation_source_summary(
    raw: Mapping[str, Any],
    root: Path,
) -> tuple[Path, str]:
    section = dict(raw.get("surface_segregation", {}) or {})
    source_root = _surface_source_root(raw, root)
    explicit = str(section.get("source_summary", "")).strip()
    if explicit:
        return _resolve_path(explicit, source_root), "explicit"

    mode = str(section.get("source_mode", "auto")).strip().lower()
    candidates = _surface_table_candidates(raw, root)
    if mode == "auto":
        for name in (
            "final-selected",
            "refine-summary",
            "screen-selected",
            "screen-summary",
        ):
            path = candidates[name]
            if path.exists():
                return path, name
        return candidates["final-selected"], "final-selected"

    if mode not in candidates:
        raise ValueError(
            "[surface_segregation].source_mode must be one of: "
            "auto, final-selected, refine-summary, screen-selected, screen-summary"
        )
    return candidates[mode], mode


def resolve_surface_segregation_output_dir(
    raw: Mapping[str, Any],
    root: Path,
) -> Path:
    section = dict(raw.get("surface_segregation", {}) or {})
    source_root = _surface_source_root(raw, root)
    value = str(section.get("outdir", "09_surface_segregation"))
    return _resolve_path(value, source_root)


def parse_surface_segregation_config(
    raw: Mapping[str, Any],
    root: Path,
) -> SurfaceSegregationConfig:
    section = dict(raw.get("surface_segregation", {}) or {})
    surface = dict(raw.get("surface", {}) or {})
    doping = dict(raw.get("doping", {}) or {})
    scan = dict(raw.get("scan", {}) or {})

    source_summary, source_mode = resolve_surface_segregation_source_summary(raw, root)
    source_root = _surface_source_root(raw, root)
    output_dir = resolve_surface_segregation_output_dir(raw, root)

    host_species = str(
        section.get(
            "host_species",
            surface.get("host_species", doping.get("host_species", "Sn")),
        )
    ).strip()
    if not host_species:
        raise ValueError("[surface_segregation].host_species is required")

    anion_species = _string_list(
        section.get(
            "anion_species",
            surface.get("anion_species", scan.get("anion_species", ["O"])),
        )
    )
    if not anion_species:
        raise ValueError("[surface_segregation].anion_species must be non-empty")

    backend = str(section.get("backend", "mace")).strip().lower()
    model = str(section.get("model", "small")).strip()
    task = str(section.get("task", "")).strip()
    backend, model, task = normalize_backend_config(
        backend=backend,
        model=model,
        task=task,
        section_name="surface_segregation",
    )

    device = str(section.get("device", "cpu")).strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("[surface_segregation].device must be cpu or cuda")

    temperature = float(section.get("temperature_K", 800.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("[surface_segregation].temperature_K must be > 0")

    steps = _positive_int(section, "steps", 10000)
    burn_in = int(section.get("burn_in", 2000))
    if burn_in < 0 or burn_in >= steps:
        raise ValueError(
            "[surface_segregation].burn_in must be >= 0 and smaller than steps"
        )

    sample_interval = _positive_int(section, "sample_interval", 20)
    trace_interval = _positive_int(section, "trace_interval", max(20, sample_interval))
    progress_interval = _positive_int(section, "progress_interval", 500)
    parallel_surfaces = _positive_int(section, "parallel_surfaces", 1)
    if parallel_surfaces > 1 and device != "cpu":
        raise ValueError(
            "[surface_segregation].parallel_surfaces > 1 is supported only on CPU"
        )

    layer_tolerance = float(
        section.get(
            "cation_layer_tolerance_A",
            surface.get("cation_layer_tolerance_A", 0.8),
        )
    )
    if not math.isfinite(layer_tolerance) or layer_tolerance <= 0:
        raise ValueError(
            "[surface_segregation].cation_layer_tolerance_A must be > 0"
        )

    depth_layers = _positive_int(
        section,
        "dopant_depth_layers",
        int(surface.get("dopant_depth_layers", 1)),
    )

    return SurfaceSegregationConfig(
        root=root.resolve(),
        source_root=source_root,
        source_summary=source_summary,
        source_mode=source_mode,
        output_dir=output_dir,
        enabled=bool(section.get("enabled", False)),
        surface_include=_string_list(section.get("surface_include")),
        max_surfaces=_positive_int(section, "max_surfaces", 10),
        host_species=host_species,
        anion_species=anion_species,
        dopant_species=_string_list(section.get("dopant_species")),
        cation_layer_tolerance_A=layer_tolerance,
        dopant_depth_layers=depth_layers,
        temperature_K=temperature,
        steps=steps,
        burn_in=burn_in,
        sample_interval=sample_interval,
        trace_interval=trace_interval,
        progress_interval=progress_interval,
        seed=int(section.get("seed", 42)),
        backend=backend,
        model=model,
        task=task,
        device=device,
        gpu_id=max(0, int(section.get("gpu_id", 0))),
        tf_threads=_positive_int(section, "tf_threads", 1),
        omp_threads=_positive_int(section, "omp_threads", 1),
        parallel_surfaces=parallel_surfaces,
        settings=dict(section),
    )


def _is_present(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"nan", "none"}


def _structure_path_from_row(row: Mapping[str, Any]) -> tuple[Path, str]:
    for column, stage in (
        ("refine_relaxed_structure_path", "refine"),
        ("screen_relaxed_structure_path", "screen"),
        ("generated_structure_path", "generated"),
    ):
        value = row.get(column)
        if _is_present(value):
            return Path(str(value)).expanduser().resolve(), stage
    raise FileNotFoundError(
        f"No usable surface structure path for {row.get('surface_id', 'unknown surface')}"
    )


def _matches_surface(row: Mapping[str, Any], selectors: Sequence[str]) -> bool:
    if not selectors:
        return True
    values = [
        str(row.get("surface_id", "")),
        str(row.get("target_id", "")),
        str(row.get("termination_label", "")),
    ]
    hkl = (
        f"({int(float(row.get('miller_h', 0) or 0))}"
        f"{int(float(row.get('miller_k', 0) or 0))}"
        f"{int(float(row.get('miller_l', 0) or 0))})"
    )
    values.append(hkl)
    for selector in selectors:
        token = str(selector).strip()
        if not token:
            continue
        if any(fnmatch.fnmatchcase(value, token) for value in values):
            return True
    return False


def discover_surface_segregation_targets(
    cfg: SurfaceSegregationConfig,
) -> list[SurfaceSegregationTarget]:
    if not cfg.source_summary.exists():
        raise FileNotFoundError(
            f"Surface segregation source table not found: {cfg.source_summary}"
        )
    frame = pd.read_csv(cfg.source_summary)
    if frame.empty:
        return []
    frame = ensure_surface_ids(frame)

    if "termination_label" not in frame.columns:
        raise RuntimeError(
            "Surface segregation requires a natural-termination surface summary. "
            "Rerun the current surface-scan workflow first."
        )

    targets: list[SurfaceSegregationTarget] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        mapping = row.to_dict()
        if not _matches_surface(mapping, cfg.surface_include):
            continue
        surface_id = str(mapping["surface_id"])
        if surface_id in seen:
            continue
        try:
            path, stage = _structure_path_from_row(mapping)
        except FileNotFoundError:
            continue
        if not path.exists():
            continue

        hkl = tuple(
            int(float(mapping.get(key, 0) or 0))
            for key in ("miller_h", "miller_k", "miller_l")
        )
        targets.append(
            SurfaceSegregationTarget(
                surface_id=surface_id,
                target_id=str(mapping.get("target_id", "")),
                structure_path=path,
                source_stage=stage,
                miller=hkl,
                termination_id=int(float(mapping.get("termination_id", 0) or 0)),
                termination_label=str(mapping.get("termination_label", "")),
                metadata=mapping,
            )
        )
        seen.add(surface_id)
        if len(targets) >= cfg.max_surfaces:
            break
    return targets


def preview_surface_segregation_targets(
    raw: Mapping[str, Any],
    root: Path,
) -> pd.DataFrame:
    cfg = parse_surface_segregation_config(raw, root)
    rows = []
    for target in discover_surface_segregation_targets(cfg):
        rows.append(
            {
                "surface_id": target.surface_id,
                "target_id": target.target_id,
                "miller": f"({target.miller[0]}{target.miller[1]}{target.miller[2]})",
                "termination_id": target.termination_id,
                "termination_label": target.termination_label,
                "source_stage": target.source_stage,
                "structure_path": str(target.structure_path),
            }
        )
    return pd.DataFrame(rows)


def infer_dopant_species(
    structure: Structure,
    *,
    host_species: str,
    anion_species: Sequence[str],
    requested: Sequence[str] = (),
) -> tuple[str, ...]:
    if requested:
        return tuple(dict.fromkeys(str(x) for x in requested))
    anions = set(anion_species)
    species = sorted(
        {
            site.specie.symbol
            for site in structure
            if site.specie.symbol != host_species
            and site.specie.symbol not in anions
        }
    )
    return tuple(species)


def classify_cation_sites(
    structure: Structure,
    *,
    host_species: str,
    dopant_species: Sequence[str],
    cation_layer_tolerance_A: float,
    dopant_depth_layers: int,
) -> dict[int, str]:
    cation_species = [host_species, *dopant_species]
    layers = _cation_layers(
        structure,
        cation_species,
        float(cation_layer_tolerance_A),
    )
    zones: dict[int, str] = {}
    for layer_index, layer in enumerate(layers):
        zone = _classify_cation_layer(
            layer_index,
            len(layers),
            int(dopant_depth_layers),
        )
        for site_index in layer:
            zones[int(site_index)] = zone
    return zones


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def run_surface_swap_mc(
    structure: Structure,
    *,
    site_zones: Mapping[int, str],
    host_species: str,
    dopant_species: Sequence[str],
    energy_function: Callable[[Structure], float],
    temperature_K: float,
    steps: int,
    burn_in: int,
    sample_interval: int,
    trace_interval: int,
    progress_interval: int,
    seed: int,
    progress_label: str = "",
) -> SwapMCResult:
    """Run fixed-composition host↔dopant Metropolis swaps on one surface.

    Coordinates and lattice are fixed during sampling. Only cation identities
    change. This isolates configurational segregation statistics and makes long
    MC trajectories practical with ML force-field single-point energies.
    """
    dopants = tuple(dict.fromkeys(str(x) for x in dopant_species))
    if not dopants:
        raise ValueError("At least one dopant species is required for segregation MC")
    if temperature_K <= 0 or steps <= 0 or sample_interval <= 0:
        raise ValueError("temperature, steps, and sample_interval must be > 0")
    if burn_in < 0 or burn_in >= steps:
        raise ValueError("burn_in must be >= 0 and smaller than steps")

    movable = sorted(
        index
        for index in site_zones
        if structure[index].specie.symbol in {host_species, *dopants}
    )
    if not movable:
        raise RuntimeError("No movable cation sites were identified")

    initial_counts = Counter(structure[index].specie.symbol for index in movable)
    if initial_counts[host_species] <= 0:
        raise RuntimeError("Segregation MC requires at least one host-cation site")
    for dopant in dopants:
        if initial_counts[dopant] <= 0:
            raise RuntimeError(
                f"Segregation MC requested dopant {dopant} but none is present"
            )

    beta = 1.0 / (K_B_EV_PER_K * float(temperature_K))
    rng = random.Random(int(seed))
    current = structure.copy()
    current_energy = float(energy_function(current))
    start_energy = current_energy
    best = current.copy()
    best_energy = current_energy

    attempted = 0
    accepted = 0
    attempted_by_dopant: Counter[str] = Counter()
    accepted_by_dopant: Counter[str] = Counter()
    site_counts: dict[str, Counter[int]] = {
        dopant: Counter() for dopant in dopants
    }
    zone_counts: dict[str, Counter[str]] = {
        dopant: Counter() for dopant in dopants
    }
    trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "energy_eV": current_energy,
            "best_energy_eV": best_energy,
            "accepted_moves": 0,
            "attempted_moves": 0,
            "acceptance_fraction": 0.0,
        }
    ]
    zone_trace: list[dict[str, Any]] = []
    n_samples = 0

    for step in range(1, int(steps) + 1):
        host_sites = [
            index
            for index in movable
            if current[index].specie.symbol == host_species
        ]
        dopant_sites = [
            index
            for index in movable
            if current[index].specie.symbol in dopants
        ]
        if not host_sites or not dopant_sites:
            break

        dopant_site = rng.choice(dopant_sites)
        host_site = rng.choice(host_sites)
        moving_dopant = current[dopant_site].specie.symbol

        proposal = current.copy()
        proposal[dopant_site] = host_species
        proposal[host_site] = moving_dopant
        proposal_energy = float(energy_function(proposal))
        delta = proposal_energy - current_energy
        accept = delta <= 0.0 or rng.random() < math.exp(-beta * delta)

        attempted += 1
        attempted_by_dopant[moving_dopant] += 1
        if accept:
            current = proposal
            current_energy = proposal_energy
            accepted += 1
            accepted_by_dopant[moving_dopant] += 1
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy

        is_sample = (
            step > burn_in
            and (step - burn_in) % sample_interval == 0
        )
        if is_sample:
            n_samples += 1
            for dopant in dopants:
                zone_snapshot = Counter()
                for site_index in movable:
                    if current[site_index].specie.symbol != dopant:
                        continue
                    site_counts[dopant][site_index] += 1
                    zone = str(site_zones[site_index])
                    zone_counts[dopant][zone] += 1
                    zone_snapshot[zone] += 1
                for zone in _ZONES:
                    zone_trace.append(
                        {
                            "step": step,
                            "sample": n_samples,
                            "dopant": dopant,
                            "zone": zone,
                            "count": int(zone_snapshot[zone]),
                        }
                    )

        if (
            step == 1
            or step % trace_interval == 0
            or is_sample
            or step == steps
        ):
            trace.append(
                {
                    "step": step,
                    "energy_eV": current_energy,
                    "best_energy_eV": best_energy,
                    "accepted_moves": accepted,
                    "attempted_moves": attempted,
                    "acceptance_fraction": (
                        accepted / attempted if attempted else 0.0
                    ),
                }
            )

        if progress_interval and (
            step % progress_interval == 0 or step == steps
        ):
            prefix = f"{progress_label}: " if progress_label else ""
            print(
                f"[surface-segregation] {prefix}step {step}/{steps}; "
                f"E={current_energy:.6f} eV; best={best_energy:.6f} eV; "
                f"accept={accepted}/{attempted} "
                f"({accepted / attempted if attempted else 0.0:.3f})",
                flush=True,
            )

    if n_samples <= 0:
        raise RuntimeError(
            "Segregation MC produced no production samples; adjust burn_in, "
            "steps, or sample_interval"
        )

    return SwapMCResult(
        final_structure=current,
        best_structure=best,
        start_energy_eV=start_energy,
        final_energy_eV=current_energy,
        best_energy_eV=best_energy,
        attempted_moves=attempted,
        accepted_moves=accepted,
        attempted_by_dopant=dict(attempted_by_dopant),
        accepted_by_dopant=dict(accepted_by_dopant),
        n_samples=n_samples,
        site_counts={
            dopant: dict(counts) for dopant, counts in site_counts.items()
        },
        zone_counts={
            dopant: dict(counts) for dopant, counts in zone_counts.items()
        },
        trace=tuple(trace),
        zone_trace=tuple(zone_trace),
    )


def summarize_surface_occupancy(
    structure: Structure,
    *,
    site_zones: Mapping[int, str],
    host_species: str,
    dopant_species: Sequence[str],
    result: SwapMCResult,
    temperature_K: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return site occupancies/PMFs and zone occupancies/effective ΔGseg."""
    dopants = tuple(dopant_species)
    cation_indices = sorted(site_zones)
    n_cation_sites = len(cation_indices)
    zone_site_counts = Counter(site_zones.values())

    site_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []

    for dopant in dopants:
        n_dopant = sum(
            structure[index].specie.symbol == dopant
            for index in cation_indices
        )
        if n_dopant <= 0:
            continue

        probabilities = {
            index: (
                result.site_counts.get(dopant, {}).get(index, 0)
                / result.n_samples
            )
            for index in cation_indices
        }
        p_max = max(probabilities.values()) if probabilities else 0.0

        for index in cation_indices:
            probability = float(probabilities[index])
            pmf = None
            if probability > 0 and p_max > 0:
                pmf = -K_B_EV_PER_K * temperature_K * math.log(
                    probability / p_max
                )
            site = structure[index]
            site_rows.append(
                {
                    "dopant": dopant,
                    "site_index": int(index),
                    "zone": str(site_zones[index]),
                    "occupancy_probability": probability,
                    "visit_count": int(
                        result.site_counts.get(dopant, {}).get(index, 0)
                    ),
                    "site_pmf_eV_vs_most_occupied": pmf,
                    "site_pmf_kJ_mol_vs_most_occupied": (
                        pmf * EV_TO_KJ_MOL if pmf is not None else None
                    ),
                    "x_A": float(site.coords[0]),
                    "y_A": float(site.coords[1]),
                    "z_A": float(site.coords[2]),
                    "initial_species": site.specie.symbol,
                    "temperature_K": float(temperature_K),
                    "n_samples": int(result.n_samples),
                }
            )

        mean_site_occupancy: dict[str, float] = {}
        for zone in _ZONES:
            n_zone_sites = int(zone_site_counts.get(zone, 0))
            observations = int(
                result.zone_counts.get(dopant, {}).get(zone, 0)
            )
            mean_site_occupancy[zone] = (
                observations / (result.n_samples * n_zone_sites)
                if n_zone_sites > 0
                else 0.0
            )

        bulk_density = mean_site_occupancy.get("bulk", 0.0)
        random_occupancy = n_dopant / n_cation_sites if n_cation_sites else 0.0

        for zone in _ZONES:
            n_zone_sites = int(zone_site_counts.get(zone, 0))
            observations = int(
                result.zone_counts.get(dopant, {}).get(zone, 0)
            )
            density = mean_site_occupancy[zone]
            fraction = (
                observations / (result.n_samples * n_dopant)
                if n_dopant > 0
                else None
            )
            delta_g = None
            if zone == "bulk" and bulk_density > 0:
                delta_g = 0.0
            elif density > 0 and bulk_density > 0:
                delta_g = -K_B_EV_PER_K * temperature_K * math.log(
                    density / bulk_density
                )

            zone_rows.append(
                {
                    "dopant": dopant,
                    "zone": zone,
                    "n_zone_sites": n_zone_sites,
                    "n_dopant_atoms": int(n_dopant),
                    "observation_count": observations,
                    "fraction_of_dopant_observations": fraction,
                    "mean_site_occupancy": density,
                    "random_site_occupancy": random_occupancy,
                    "enrichment_factor_vs_random": (
                        density / random_occupancy
                        if random_occupancy > 0
                        else None
                    ),
                    "delta_G_eff_vs_bulk_eV": delta_g,
                    "delta_G_eff_vs_bulk_kJ_mol": (
                        delta_g * EV_TO_KJ_MOL
                        if delta_g is not None
                        else None
                    ),
                    "temperature_K": float(temperature_K),
                    "n_samples": int(result.n_samples),
                    "free_energy_note": (
                        "Occupancy-derived effective segregation free energy: "
                        "-kBT ln(rho_zone/rho_bulk), where rho is mean dopant "
                        "occupancy per cation site. This is a finite-temperature "
                        "PMF/effective preference, not a zero-temperature static "
                        "segregation energy."
                    ),
                }
            )

    return site_rows, zone_rows


def _build_calculator(cfg: SurfaceSegregationConfig) -> Any:
    from dopingflow.ml_backends import (
        build_ase_calculator,
        check_backend_dependency,
        prepare_backend_runtime,
    )

    prepare_backend_runtime(
        backend=cfg.backend,
        device=cfg.device,
        gpu_id=cfg.gpu_id,
        tf_threads=cfg.tf_threads,
        omp_threads=cfg.omp_threads,
    )
    check_backend_dependency(
        cfg.backend,
        stage_name="Surface segregation Monte Carlo",
    )
    return build_ase_calculator(
        backend=cfg.backend,
        model=cfg.model,
        task=cfg.task,
        device=cfg.device,
    )


def _surface_site_rows(
    structure: Structure,
    site_zones: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows = []
    for index in sorted(site_zones):
        site = structure[index]
        rows.append(
            {
                "site_index": int(index),
                "initial_species": site.specie.symbol,
                "zone": str(site_zones[index]),
                "x_A": float(site.coords[0]),
                "y_A": float(site.coords[1]),
                "z_A": float(site.coords[2]),
            }
        )
    return rows


def _run_target(
    cfg: SurfaceSegregationConfig,
    target: SurfaceSegregationTarget,
    target_index: int,
    calculator: Any,
) -> dict[str, Any]:
    from dopingflow.ml_relaxation import structure_energy_with_calculator

    structure = Structure.from_file(target.structure_path)
    dopants = infer_dopant_species(
        structure,
        host_species=cfg.host_species,
        anion_species=cfg.anion_species,
        requested=cfg.dopant_species,
    )
    if not dopants:
        raise RuntimeError(
            f"{target.surface_id}: no dopant species could be inferred"
        )

    site_zones = classify_cation_sites(
        structure,
        host_species=cfg.host_species,
        dopant_species=dopants,
        cation_layer_tolerance_A=cfg.cation_layer_tolerance_A,
        dopant_depth_layers=cfg.dopant_depth_layers,
    )
    if not site_zones:
        raise RuntimeError(
            f"{target.surface_id}: no cation layers could be classified"
        )

    target_dir = cfg.output_dir / "surfaces" / target.safe_id
    target_dir.mkdir(parents=True, exist_ok=True)
    Poscar(structure).write_file(target_dir / "POSCAR_start")
    _write_csv(target_dir / "cation_site_zones.csv", _surface_site_rows(structure, site_zones))

    def energy_function(candidate: Structure) -> float:
        return float(
            structure_energy_with_calculator(candidate, calculator)
        )

    result = run_surface_swap_mc(
        structure,
        site_zones=site_zones,
        host_species=cfg.host_species,
        dopant_species=dopants,
        energy_function=energy_function,
        temperature_K=cfg.temperature_K,
        steps=cfg.steps,
        burn_in=cfg.burn_in,
        sample_interval=cfg.sample_interval,
        trace_interval=cfg.trace_interval,
        progress_interval=cfg.progress_interval,
        seed=cfg.seed + target_index,
        progress_label=target.surface_id,
    )

    Poscar(result.best_structure).write_file(target_dir / "POSCAR_best_mc")
    Poscar(result.final_structure).write_file(target_dir / "POSCAR_final_mc")

    site_rows, zone_rows = summarize_surface_occupancy(
        structure,
        site_zones=site_zones,
        host_species=cfg.host_species,
        dopant_species=dopants,
        result=result,
        temperature_K=cfg.temperature_K,
    )
    _write_csv(target_dir / "site_occupancy.csv", site_rows)
    _write_csv(target_dir / "zone_occupancy.csv", zone_rows)
    _write_csv(target_dir / "mc_trace.csv", result.trace)
    _write_csv(target_dir / "zone_trace.csv", result.zone_trace)

    swap_rows = []
    for dopant in dopants:
        attempted = int(result.attempted_by_dopant.get(dopant, 0))
        accepted = int(result.accepted_by_dopant.get(dopant, 0))
        swap_rows.append(
            {
                "dopant": dopant,
                "attempted_moves": attempted,
                "accepted_moves": accepted,
                "acceptance_fraction": accepted / attempted if attempted else 0.0,
            }
        )
    _write_csv(target_dir / "swap_statistics.csv", swap_rows)

    summary = {
        "surface_id": target.surface_id,
        "target_id": target.target_id,
        "miller_h": target.miller[0],
        "miller_k": target.miller[1],
        "miller_l": target.miller[2],
        "termination_id": target.termination_id,
        "termination_label": target.termination_label,
        "source_structure_path": str(target.structure_path),
        "source_stage": target.source_stage,
        "temperature_K": cfg.temperature_K,
        "host_species": cfg.host_species,
        "dopant_species": list(dopants),
        "n_cation_sites": len(site_zones),
        "n_surface_sites": sum(zone == "surface" for zone in site_zones.values()),
        "n_subsurface_sites": sum(zone == "subsurface" for zone in site_zones.values()),
        "n_bulk_sites": sum(zone == "bulk" for zone in site_zones.values()),
        "steps_requested": cfg.steps,
        "burn_in": cfg.burn_in,
        "sample_interval": cfg.sample_interval,
        "n_samples": result.n_samples,
        "attempted_moves": result.attempted_moves,
        "accepted_moves": result.accepted_moves,
        "acceptance_fraction": (
            result.accepted_moves / result.attempted_moves
            if result.attempted_moves
            else 0.0
        ),
        "start_energy_eV": result.start_energy_eV,
        "final_energy_eV": result.final_energy_eV,
        "best_energy_eV": result.best_energy_eV,
        "backend": cfg.backend,
        "model": cfg.model,
        "task": cfg.task,
        "device": cfg.device,
        "seed": cfg.seed + target_index,
        "sampling_mode": "fixed-geometry host-dopant swap Metropolis MC",
        "site_occupancy_csv": str(target_dir / "site_occupancy.csv"),
        "zone_occupancy_csv": str(target_dir / "zone_occupancy.csv"),
        "trace_csv": str(target_dir / "mc_trace.csv"),
        "zone_trace_csv": str(target_dir / "zone_trace.csv"),
        "swap_statistics_csv": str(target_dir / "swap_statistics.csv"),
        "best_structure_path": str(target_dir / "POSCAR_best_mc"),
        "final_structure_path": str(target_dir / "POSCAR_final_mc"),
    }
    _write_json(target_dir / "summary.json", summary)
    return summary


_SURFACE_SEG_WORKER_CFG: SurfaceSegregationConfig | None = None
_SURFACE_SEG_WORKER_CALCULATOR: Any = None


def _init_worker(cfg: SurfaceSegregationConfig) -> None:
    global _SURFACE_SEG_WORKER_CFG, _SURFACE_SEG_WORKER_CALCULATOR
    _SURFACE_SEG_WORKER_CFG = cfg
    _SURFACE_SEG_WORKER_CALCULATOR = _build_calculator(cfg)


def _worker(
    payload: tuple[int, SurfaceSegregationTarget],
) -> tuple[int, dict[str, Any]]:
    if _SURFACE_SEG_WORKER_CFG is None or _SURFACE_SEG_WORKER_CALCULATOR is None:
        raise RuntimeError("Surface-segregation worker was not initialized")
    index, target = payload
    return (
        index,
        _run_target(
            _SURFACE_SEG_WORKER_CFG,
            target,
            index,
            _SURFACE_SEG_WORKER_CALCULATOR,
        ),
    )


def run_surface_segregation(
    raw: Mapping[str, Any],
    root: Path,
    *,
    dry_run: bool = False,
) -> Path | None:
    section = dict(raw.get("surface_segregation", {}) or {})
    if not bool(section.get("enabled", False)):
        return None

    cfg = parse_surface_segregation_config(raw, root)
    targets = discover_surface_segregation_targets(cfg)
    if not targets:
        raise RuntimeError("Surface segregation selected no usable surfaces")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plan = [
        {
            "surface_id": target.surface_id,
            "target_id": target.target_id,
            "miller": list(target.miller),
            "termination_id": target.termination_id,
            "termination_label": target.termination_label,
            "structure_path": str(target.structure_path),
            "temperature_K": cfg.temperature_K,
            "steps": cfg.steps,
            "burn_in": cfg.burn_in,
            "sample_interval": cfg.sample_interval,
            "backend": cfg.backend,
            "model": cfg.model,
            "task": cfg.task,
            "device": cfg.device,
        }
        for target in targets
    ]
    _write_json(cfg.output_dir / "mc_plan.json", plan)
    if dry_run:
        return cfg.output_dir / "mc_plan.json"

    worker_count = min(cfg.parallel_surfaces, len(targets))
    if cfg.device != "cpu":
        worker_count = 1

    summaries_by_index: dict[int, dict[str, Any]] = {}
    if worker_count <= 1:
        calculator = _build_calculator(cfg)
        for index, target in enumerate(targets):
            summaries_by_index[index] = _run_target(
                cfg,
                target,
                index,
                calculator,
            )
    else:
        log.info(
            "Surface segregation MC: %d surfaces, %d CPU workers, "
            "%d OpenMP thread(s) per worker",
            len(targets),
            worker_count,
            cfg.omp_threads,
        )
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_init_worker,
            initargs=(cfg,),
        ) as executor:
            futures = {
                executor.submit(_worker, (index, target)): (index, target)
                for index, target in enumerate(targets)
            }
            for future in as_completed(futures):
                index, target = futures[future]
                try:
                    result_index, summary = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Surface segregation MC failed for {target.surface_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                summaries_by_index[result_index] = summary

    summaries = [
        summaries_by_index[index]
        for index in range(len(targets))
        if index in summaries_by_index
    ]
    summary_csv = cfg.output_dir / "surface_segregation_summary.csv"
    _write_csv(summary_csv, summaries)
    _write_json(cfg.output_dir / "surface_segregation_summary.json", summaries)
    _write_json(
        cfg.output_dir / "config_resolved.json",
        {
            **asdict(cfg),
            "root": str(cfg.root),
            "source_root": str(cfg.source_root),
            "source_summary": str(cfg.source_summary),
            "output_dir": str(cfg.output_dir),
        },
    )
    return summary_csv


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def run_surface_segregation_from_toml(
    config_path: Path,
    *,
    dry_run: bool = False,
) -> Path | None:
    config_path = Path(config_path).expanduser().resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_surface_segregation(
        raw,
        config_path.parent,
        dry_run=dry_run,
    )


__all__ = [
    "K_B_EV_PER_K",
    "SurfaceSegregationConfig",
    "SurfaceSegregationTarget",
    "SwapMCResult",
    "classify_cation_sites",
    "discover_surface_segregation_targets",
    "infer_dopant_species",
    "parse_surface_segregation_config",
    "preview_surface_segregation_targets",
    "resolve_surface_segregation_output_dir",
    "resolve_surface_segregation_source_summary",
    "run_surface_segregation",
    "run_surface_segregation_from_toml",
    "run_surface_swap_mc",
    "summarize_surface_occupancy",
]

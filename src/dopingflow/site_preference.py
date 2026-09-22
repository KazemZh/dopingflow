from __future__ import annotations

import csv
import fnmatch
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairScanConfig:
    enabled: bool = False
    execute: bool = False
    source_target: str = ""
    pairs: tuple[tuple[str, str], ...] = ()
    max_shells: int = 6
    symprec: float = 1e-3
    angle_tolerance: float = 5.0
    backend: str = "mace"
    model: str = "small"
    task: str = ""
    device: str = "cpu"
    gpu_id: int = 0
    tf_threads: int = 1
    omp_threads: int = 1
    relax: bool = True
    optimizer: str = "bfgs"
    fmax: float = 0.05
    max_steps: int = 300
    relax_mode: str = "atoms"
    cell_filter: str = "frechet"


@dataclass(frozen=True)
class OrderingMCConfig:
    enabled: bool = False
    execute: bool = False
    target_include: tuple[str, ...] = ()
    max_targets: int = 5
    temperature_K: float = 800.0
    steps: int = 10000
    burn_in: int = 2000
    sample_interval: int = 20
    seed: int = 42
    backend: str = "mace"
    model: str = "small"
    task: str = ""
    device: str = "cpu"
    gpu_id: int = 0
    tf_threads: int = 1
    omp_threads: int = 1
    relax_best: bool = True
    optimizer: str = "bfgs"
    fmax: float = 0.05
    max_steps: int = 300
    relax_mode: str = "atoms"
    cell_filter: str = "frechet"


@dataclass(frozen=True)
class SitePreferenceConfig:
    root: Path
    source_root: Path
    output_dir: Path
    enabled: bool
    host_species: str
    anion_species: tuple[str, ...]
    include_vacancy_free: bool
    include_oxygen_vacancies: bool
    target_include: tuple[str, ...]
    max_shells: int
    shell_tolerance_angstrom: float
    mapping_tolerance_angstrom: float
    motif_neighbor_shell_max: int
    pair_scan: PairScanConfig
    ordering_mc: OrderingMCConfig
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreferenceTarget:
    target_id: str
    parent_id: str
    kind: str
    structure_path: Path
    energy_eV: float | None
    energy_source: str | None
    n_vacancies: int = 0
    vacancy_species: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def safe_id(self) -> str:
        text = self.target_id.replace("\\", "__").replace("/", "__")
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)

    @property
    def composition_label(self) -> str:
        return self.parent_id.split("/", 1)[0]


def _parse_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        raise ValueError("Expected a string or array")
    return tuple(dict.fromkeys(item for item in items if item))


def _parse_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    pairs: list[tuple[str, str]] = []
    if isinstance(value, str):
        raw_items: Iterable[Any] = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raise ValueError("[site_preference.pair_scan].pairs must be an array")

    for item in raw_items:
        if isinstance(item, str):
            token = item.replace("–", "-").replace("—", "-").strip()
            parts = [part.strip() for part in token.split("-") if part.strip()]
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            parts = [str(item[0]).strip(), str(item[1]).strip()]
        else:
            raise ValueError(
                "Each [site_preference.pair_scan].pairs entry must be 'Sb-Ti' or ['Sb','Ti']"
            )
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid pair specification: {item!r}")
        pair = (parts[0], parts[1])
        if pair not in pairs and (pair[1], pair[0]) not in pairs:
            pairs.append(pair)
    return tuple(pairs)


def _positive_int(section: dict[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def _backend_settings(section: dict[str, Any], *, default_model: str = "small") -> dict[str, Any]:
    backend = str(section.get("backend", "mace")).strip().lower()
    model = str(section.get("model", default_model)).strip()
    task = str(section.get("task", "")).strip()
    device = str(section.get("device", "cpu")).strip().lower()
    gpu_id = int(section.get("gpu_id", 0))
    tf_threads = _positive_int(section, "tf_threads", 1)
    omp_threads = _positive_int(section, "omp_threads", 1)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if gpu_id < 0:
        raise ValueError("gpu_id must be >= 0")
    from dopingflow.ml_backends import normalize_backend_config

    backend, model, task = normalize_backend_config(
        backend=backend,
        model=model,
        task=task,
        section_name="site_preference",
    )
    return {
        "backend": backend,
        "model": model,
        "task": task,
        "device": device,
        "gpu_id": gpu_id,
        "tf_threads": tf_threads,
        "omp_threads": omp_threads,
    }


def parse_site_preference_config(raw: dict[str, Any], root: Path) -> SitePreferenceConfig:
    section = raw.get("site_preference", {}) or {}
    if not isinstance(section, dict):
        raise ValueError("[site_preference] must be a TOML table")

    structure = raw.get("structure", {}) or {}
    doping = raw.get("doping", {}) or {}
    scan = raw.get("scan", {}) or {}

    source_value = str(section.get("source_root", structure.get("outdir", "random_structures")))
    source_path = Path(source_value).expanduser()
    source_root = source_path if source_path.is_absolute() else (root / source_path)
    source_root = source_root.resolve()

    output_value = str(section.get("output_dir", "06_site_preference"))
    output_path = Path(output_value).expanduser()
    output_dir = output_path if output_path.is_absolute() else (source_root / output_path)
    output_dir = output_dir.resolve()

    host_species = str(section.get("host_species", doping.get("host_species", ""))).strip()
    if not host_species:
        raise ValueError(
            "[site_preference].host_species is required, or set [doping].host_species"
        )

    anion_raw = section.get("anion_species", scan.get("anion_species", ["O"]))
    anion_species = _parse_string_list(anion_raw)
    if not anion_species:
        raise ValueError("[site_preference].anion_species must be non-empty")
    if host_species in anion_species:
        raise ValueError("host_species cannot also be an anion species")

    max_shells = _positive_int(section, "max_shells", 6)
    shell_tol = float(section.get("shell_tolerance_angstrom", 0.12))
    mapping_tol = float(section.get("mapping_tolerance_angstrom", 1.5))
    motif_neighbor_shell_max = _positive_int(section, "motif_neighbor_shell_max", 1)
    if motif_neighbor_shell_max > max_shells:
        raise ValueError(
            "[site_preference].motif_neighbor_shell_max cannot exceed max_shells"
        )
    if not math.isfinite(shell_tol) or shell_tol <= 0:
        raise ValueError("[site_preference].shell_tolerance_angstrom must be > 0")
    if not math.isfinite(mapping_tol) or mapping_tol <= 0:
        raise ValueError("[site_preference].mapping_tolerance_angstrom must be > 0")

    pair_section = section.get("pair_scan", {}) or {}
    if not isinstance(pair_section, dict):
        raise ValueError("[site_preference.pair_scan] must be a TOML table")
    pair_backend = _backend_settings(pair_section)
    pair_scan = PairScanConfig(
        enabled=bool(pair_section.get("enabled", False)),
        execute=bool(pair_section.get("execute", False)),
        source_target=str(pair_section.get("source_target", "")).strip(),
        pairs=_parse_pairs(pair_section.get("pairs")),
        max_shells=_positive_int(pair_section, "max_shells", max_shells),
        symprec=float(pair_section.get("symprec", scan.get("symprec", 1e-3))),
        angle_tolerance=float(pair_section.get("angle_tolerance", 5.0)),
        relax=bool(pair_section.get("relax", True)),
        optimizer=str(pair_section.get("optimizer", "bfgs")).strip().lower(),
        fmax=float(pair_section.get("fmax", 0.05)),
        max_steps=_positive_int(pair_section, "max_steps", 300),
        relax_mode=str(pair_section.get("relax_mode", "atoms")).strip().lower(),
        cell_filter=str(pair_section.get("cell_filter", "frechet")).strip().lower(),
        **pair_backend,
    )
    if pair_scan.fmax <= 0:
        raise ValueError("[site_preference.pair_scan].fmax must be > 0")
    if not math.isfinite(pair_scan.symprec) or pair_scan.symprec <= 0:
        raise ValueError("[site_preference.pair_scan].symprec must be > 0")
    if not math.isfinite(pair_scan.angle_tolerance) or pair_scan.angle_tolerance <= 0:
        raise ValueError("[site_preference.pair_scan].angle_tolerance must be > 0")

    mc_section = section.get("ordering_mc", {}) or {}
    if not isinstance(mc_section, dict):
        raise ValueError("[site_preference.ordering_mc] must be a TOML table")
    mc_backend = _backend_settings(mc_section)
    ordering_mc = OrderingMCConfig(
        enabled=bool(mc_section.get("enabled", False)),
        execute=bool(mc_section.get("execute", False)),
        target_include=_parse_string_list(mc_section.get("target_include")),
        max_targets=_positive_int(mc_section, "max_targets", 5),
        temperature_K=float(mc_section.get("temperature_K", 800.0)),
        steps=_positive_int(mc_section, "steps", 10000),
        burn_in=max(0, int(mc_section.get("burn_in", 2000))),
        sample_interval=_positive_int(mc_section, "sample_interval", 20),
        seed=int(mc_section.get("seed", 42)),
        relax_best=bool(mc_section.get("relax_best", True)),
        optimizer=str(mc_section.get("optimizer", "bfgs")).strip().lower(),
        fmax=float(mc_section.get("fmax", 0.05)),
        max_steps=_positive_int(mc_section, "max_steps", 300),
        relax_mode=str(mc_section.get("relax_mode", "atoms")).strip().lower(),
        cell_filter=str(mc_section.get("cell_filter", "frechet")).strip().lower(),
        **mc_backend,
    )
    if not math.isfinite(ordering_mc.temperature_K) or ordering_mc.temperature_K <= 0:
        raise ValueError("[site_preference.ordering_mc].temperature_K must be > 0")
    if ordering_mc.burn_in >= ordering_mc.steps:
        raise ValueError("[site_preference.ordering_mc].burn_in must be smaller than steps")
    if ordering_mc.fmax <= 0:
        raise ValueError("[site_preference.ordering_mc].fmax must be > 0")

    return SitePreferenceConfig(
        root=root.resolve(),
        source_root=source_root,
        output_dir=output_dir,
        enabled=bool(section.get("enabled", True)),
        host_species=host_species,
        anion_species=anion_species,
        include_vacancy_free=bool(section.get("include_vacancy_free", True)),
        include_oxygen_vacancies=bool(section.get("include_oxygen_vacancies", True)),
        target_include=_parse_string_list(section.get("target_include")),
        max_shells=max_shells,
        shell_tolerance_angstrom=shell_tol,
        mapping_tolerance_angstrom=mapping_tol,
        motif_neighbor_shell_max=motif_neighbor_shell_max,
        pair_scan=pair_scan,
        ordering_mc=ordering_mc,
        settings=dict(section),
    )


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _target_matches(target: PreferenceTarget, selectors: Sequence[str]) -> bool:
    if not selectors:
        return True
    for selector in selectors:
        selector = str(selector).strip().replace("\\", "/")
        if not selector:
            continue
        if fnmatch.fnmatchcase(target.target_id, selector):
            return True
        safe_selector = selector.replace("/", "__")
        if fnmatch.fnmatchcase(target.safe_id, safe_selector):
            return True
    return False


def _first_float(mapping: dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return None


def discover_site_preference_targets(
    cfg: SitePreferenceConfig,
) -> tuple[list[PreferenceTarget], dict[str, PreferenceTarget], list[str]]:
    from dopingflow.vacancies import discover_selected_parents

    warnings: list[str] = []
    if not cfg.source_root.exists():
        raise FileNotFoundError(f"Site-preference source root does not exist: {cfg.source_root}")

    parents_raw = discover_selected_parents(cfg.source_root)
    parent_map: dict[str, PreferenceTarget] = {}
    targets: list[PreferenceTarget] = []

    for parent in parents_raw:
        parent_id = str(parent["parent_id"])
        candidate_dir = Path(parent["candidate_dir"]).resolve()
        meta_path = candidate_dir / "02_relax" / "meta.json"
        meta = _load_json(meta_path, {}) or {}
        energy = _first_float(meta, ("energy_relaxed_eV", "energy_relaxed_total_eV"))
        target = PreferenceTarget(
            target_id=parent_id,
            parent_id=parent_id,
            kind="vacancy-free",
            structure_path=Path(parent["relaxed_path"]).resolve(),
            energy_eV=energy,
            energy_source=str(meta_path) if energy is not None else None,
            n_vacancies=0,
            vacancy_species=None,
            metadata={
                "composition": str(parent.get("composition", "")),
                "candidate": str(parent.get("candidate", "")),
                "candidate_dir": str(candidate_dir),
                "symmetry_path": str(Path(parent["symmetry_path"]).resolve()),
            },
        )
        parent_map[parent_id] = target
        if cfg.include_vacancy_free:
            targets.append(target)

    if cfg.include_oxygen_vacancies:
        vacancy_db = cfg.source_root / "vacancies_database.json"
        rows = _load_json(vacancy_db, [])
        if not rows:
            warnings.append(f"No usable {vacancy_db.name} found; vacancy targets were skipped")
        elif not isinstance(rows, list):
            raise ValueError(f"Expected a list in {vacancy_db}")
        else:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                n_vacancies = int(row.get("n_vacancies") or 0)
                species = str(row.get("vacancy_species") or "")
                if n_vacancies <= 0 or species != "O":
                    continue
                parent_id = str(row.get("parent_id") or "").strip()
                if parent_id not in parent_map:
                    warnings.append(
                        f"Vacancy row references unknown selected parent {parent_id!r}; skipped"
                    )
                    continue
                raw_path = str(row.get("relaxed_poscar_path") or "").strip()
                if not raw_path:
                    continue
                relaxed = _resolve_path(raw_path, cfg.root)
                if not relaxed.exists():
                    warnings.append(f"Missing vacancy relaxed structure: {relaxed}")
                    continue
                config_id = str(row.get("configuration_id") or relaxed.parent.parent.name)
                energy = _first_float(
                    row,
                    (
                        "energy_relaxed_total_eV",
                        "energy_relaxed_eV",
                        "energy_sp_total_eV",
                    ),
                )
                targets.append(
                    PreferenceTarget(
                        target_id=f"{parent_id}/V_O_{n_vacancies:02d}/{config_id}",
                        parent_id=parent_id,
                        kind="oxygen-vacancy",
                        structure_path=relaxed,
                        energy_eV=energy,
                        energy_source=str(vacancy_db) if energy is not None else None,
                        n_vacancies=n_vacancies,
                        vacancy_species="O",
                        metadata={
                            "configuration_id": config_id,
                            "search_method": row.get("search_method"),
                            "backend": row.get("backend"),
                            "model": row.get("model"),
                            "task": row.get("task"),
                        },
                    )
                )

    targets = [target for target in targets if _target_matches(target, cfg.target_include)]
    if not targets:
        raise RuntimeError("No structures matched the site-preference analysis selection")
    targets.sort(key=lambda item: (item.composition_label, item.n_vacancies, item.target_id))
    return targets, parent_map, warnings


def _cluster_distances(distances: Iterable[float], tolerance: float, max_shells: int) -> list[float]:
    values = sorted(float(value) for value in distances if value > 1e-8 and math.isfinite(value))
    if not values:
        return []
    clusters: list[list[float]] = []
    for value in values:
        if not clusters:
            clusters.append([value])
            continue
        center = sum(clusters[-1]) / len(clusters[-1])
        if abs(value - center) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters[:max_shells]]


def _indices_by_role(
    structure: Structure, host_species: str, anion_species: Sequence[str]
) -> tuple[list[int], list[int], list[int]]:
    anions = set(anion_species)
    cation_indices: list[int] = []
    anion_indices: list[int] = []
    dopant_indices: list[int] = []
    for index, site in enumerate(structure):
        element = site.species_string
        if element in anions:
            anion_indices.append(index)
        else:
            cation_indices.append(index)
            if element != host_species:
                dopant_indices.append(index)
    return cation_indices, anion_indices, dopant_indices


def cation_shell_centers(
    structure: Structure,
    host_species: str,
    anion_species: Sequence[str],
    *,
    max_shells: int,
    tolerance: float,
) -> list[float]:
    cations, _, _ = _indices_by_role(structure, host_species, anion_species)
    return _cluster_distances(
        (structure.get_distance(i, j) for i, j in combinations(cations, 2)),
        tolerance,
        max_shells,
    )


def cation_anion_shell_centers(
    structure: Structure,
    host_species: str,
    anion_species: Sequence[str],
    *,
    max_shells: int,
    tolerance: float,
) -> list[float]:
    cations, anions, _ = _indices_by_role(structure, host_species, anion_species)
    return _cluster_distances(
        (structure.get_distance(i, j) for i in cations for j in anions),
        tolerance,
        max_shells,
    )


def _shell_index(distance: float, centers: Sequence[float]) -> int | None:
    if not centers:
        return None
    return min(range(len(centers)), key=lambda index: abs(distance - centers[index])) + 1


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def dopant_pair_records(
    target: PreferenceTarget,
    structure: Structure,
    cfg: SitePreferenceConfig,
) -> list[dict[str, Any]]:
    _, _, dopants = _indices_by_role(structure, cfg.host_species, cfg.anion_species)
    centers = cation_shell_centers(
        structure,
        cfg.host_species,
        cfg.anion_species,
        max_shells=cfg.max_shells,
        tolerance=cfg.shell_tolerance_angstrom,
    )
    records: list[dict[str, Any]] = []
    for i, j in combinations(dopants, 2):
        a = structure[i].species_string
        b = structure[j].species_string
        pa, pb = _canonical_pair(a, b)
        distance = float(structure.get_distance(i, j))
        records.append(
            {
                "target_id": target.target_id,
                "parent_id": target.parent_id,
                "composition": target.composition_label,
                "structure_kind": target.kind,
                "n_oxygen_vacancies": target.n_vacancies if target.vacancy_species == "O" else 0,
                "pair": f"{pa}-{pb}",
                "element_a": pa,
                "element_b": pb,
                "site_i": int(i),
                "site_j": int(j),
                "distance_angstrom": distance,
                "shell": _shell_index(distance, centers),
                "shell_center_angstrom": (
                    centers[_shell_index(distance, centers) - 1]
                    if _shell_index(distance, centers) is not None
                    else None
                ),
                "energy_total_eV": target.energy_eV,
            }
        )
    return records


def dopant_triplet_records(
    target: PreferenceTarget,
    structure: Structure,
    cfg: SitePreferenceConfig,
) -> list[dict[str, Any]]:
    """Describe three-dopant motifs using the host-cation coordination shells."""
    _, _, dopants = _indices_by_role(structure, cfg.host_species, cfg.anion_species)
    if len(dopants) < 3:
        return []
    centers = cation_shell_centers(
        structure,
        cfg.host_species,
        cfg.anion_species,
        max_shells=cfg.max_shells,
        tolerance=cfg.shell_tolerance_angstrom,
    )
    records: list[dict[str, Any]] = []
    for i, j, k in combinations(dopants, 3):
        indices = (i, j, k)
        species = tuple(structure[index].species_string for index in indices)
        pair_data: list[dict[str, Any]] = []
        for left, right in ((i, j), (i, k), (j, k)):
            distance = float(structure.get_distance(left, right))
            shell = _shell_index(distance, centers)
            pair_data.append(
                {
                    "pair": "-".join(
                        _canonical_pair(
                            structure[left].species_string,
                            structure[right].species_string,
                        )
                    ),
                    "site_i": int(left),
                    "site_j": int(right),
                    "distance_angstrom": distance,
                    "shell": shell,
                }
            )

        neighbor_edges = sum(
            item["shell"] is not None
            and int(item["shell"]) <= cfg.motif_neighbor_shell_max
            for item in pair_data
        )
        if neighbor_edges == 3:
            motif = "compact_triangle"
        elif neighbor_edges == 2:
            motif = "connected_chain"
        elif neighbor_edges == 1:
            motif = "isolated_pair_plus_third"
        else:
            motif = "dispersed"

        distances = [float(item["distance_angstrom"]) for item in pair_data]
        shell_signature = sorted(
            int(item["shell"]) if item["shell"] is not None else 999
            for item in pair_data
        )
        records.append(
            {
                "target_id": target.target_id,
                "parent_id": target.parent_id,
                "composition": target.composition_label,
                "structure_kind": target.kind,
                "n_oxygen_vacancies": target.n_vacancies,
                "species_triplet": "-".join(sorted(species)),
                "site_indices": [int(i), int(j), int(k)],
                "motif": motif,
                "neighbor_shell_max": cfg.motif_neighbor_shell_max,
                "neighbor_edge_count": neighbor_edges,
                "shell_signature": shell_signature,
                "pair_data": pair_data,
                "minimum_pair_distance_angstrom": min(distances),
                "mean_pair_distance_angstrom": sum(distances) / 3.0,
                "maximum_pair_distance_angstrom": max(distances),
                "triangle_perimeter_angstrom": sum(distances),
                "energy_total_eV": target.energy_eV,
            }
        )
    return records


def _triplet_target_presence(
    triplet_records: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    meta = {row["target_id"]: row for row in target_rows}
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in triplet_records:
        counts[(row["target_id"], row["species_triplet"], row["motif"])] += 1

    output: list[dict[str, Any]] = []
    for (target_id, species_triplet, motif), count in sorted(counts.items()):
        target = meta[target_id]
        output.append(
            {
                "target_id": target_id,
                "composition": target["composition"],
                "structure_kind": target["structure_kind"],
                "n_oxygen_vacancies": target["n_oxygen_vacancies"],
                "species_triplet": species_triplet,
                "motif": motif,
                "n_instances": count,
                "energy_total_eV": target["energy_total_eV"],
                "delta_energy_within_group_eV": target[
                    "delta_energy_within_group_eV"
                ],
            }
        )
    return output


def _aggregate_triplet_preferences(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["composition"]),
                str(row["structure_kind"]),
                int(row["n_oxygen_vacancies"]),
                str(row["species_triplet"]),
                str(row["motif"]),
            )
        ].append(row)

    motif_stats: list[dict[str, Any]] = []
    for key, group in grouped.items():
        composition, kind, n_vac, species_triplet, motif = key
        energies = [
            float(row["delta_energy_within_group_eV"])
            for row in group
            if row.get("delta_energy_within_group_eV") is not None
        ]
        motif_stats.append(
            {
                "composition": composition,
                "structure_kind": kind,
                "n_oxygen_vacancies": n_vac,
                "species_triplet": species_triplet,
                "motif": motif,
                "n_targets": len(group),
                "n_instances": sum(int(row["n_instances"]) for row in group),
                "median_delta_energy_within_group_eV": (
                    median(energies) if energies else None
                ),
                "mean_delta_energy_within_group_eV": (
                    sum(energies) / len(energies) if energies else None
                ),
            }
        )

    by_triplet: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in motif_stats:
        by_triplet[
            (
                row["composition"],
                row["structure_kind"],
                row["n_oxygen_vacancies"],
                row["species_triplet"],
            )
        ].append(row)

    preferred: list[dict[str, Any]] = []
    for key, group in by_triplet.items():
        energetic = [
            row for row in group
            if row["median_delta_energy_within_group_eV"] is not None
        ]
        if energetic:
            best = min(
                energetic,
                key=lambda row: (
                    float(row["median_delta_energy_within_group_eV"]),
                    float(row["mean_delta_energy_within_group_eV"]),
                ),
            )
            basis = "lowest median same-composition configuration energy"
        else:
            best = max(
                group,
                key=lambda row: (int(row["n_targets"]), int(row["n_instances"])),
            )
            basis = "most frequently observed motif; no comparable energies available"
        composition, kind, n_vac, species_triplet = key
        preferred.append(
            {
                "composition": composition,
                "structure_kind": kind,
                "n_oxygen_vacancies": n_vac,
                "species_triplet": species_triplet,
                "preferred_motif": best["motif"],
                "n_targets": best["n_targets"],
                "n_instances": best["n_instances"],
                "median_delta_energy_within_group_eV": best[
                    "median_delta_energy_within_group_eV"
                ],
                "preference_basis": basis,
                "interpretation_note": (
                    "Motifs are classified by the number of pair edges lying within "
                    "motif_neighbor_shell_max; the energy remains a full-configuration energy."
                ),
            }
        )
    preferred.sort(
        key=lambda row: (
            row["composition"],
            row["n_oxygen_vacancies"],
            row["species_triplet"],
        )
    )
    return preferred


def _map_surviving_species(
    parent: Structure,
    child: Structure,
    species: str,
    tolerance: float,
) -> tuple[dict[int, int], list[int]]:
    parent_indices = [i for i, site in enumerate(parent) if site.species_string == species]
    child_indices = [i for i, site in enumerate(child) if site.species_string == species]
    unused = set(parent_indices)
    mapping: dict[int, int] = {}

    for child_index in child_indices:
        candidates = list(unused)
        if not candidates:
            break
        distances = [
            child.lattice.get_distance_and_image(
                child[child_index].frac_coords,
                parent[parent_index].frac_coords,
            )[0]
            for parent_index in candidates
        ]
        best_position = min(range(len(candidates)), key=lambda pos: distances[pos])
        if distances[best_position] > tolerance:
            raise ValueError(
                f"Could not map {species} site: nearest parent site is "
                f"{distances[best_position]:.3f} Å > tolerance {tolerance:.3f} Å"
            )
        parent_index = candidates[best_position]
        mapping[child_index] = parent_index
        unused.remove(parent_index)
    return mapping, sorted(unused)


def dopant_vacancy_records(
    target: PreferenceTarget,
    structure: Structure,
    parent_structure: Structure,
    cfg: SitePreferenceConfig,
) -> list[dict[str, Any]]:
    if target.kind != "oxygen-vacancy":
        return []

    _, missing_oxygen = _map_surviving_species(
        parent_structure,
        structure,
        "O",
        cfg.mapping_tolerance_angstrom,
    )
    if target.n_vacancies and len(missing_oxygen) != target.n_vacancies:
        log.warning(
            "%s: mapped %d O vacancies but metadata reports %d",
            target.target_id,
            len(missing_oxygen),
            target.n_vacancies,
        )

    _, _, dopants = _indices_by_role(structure, cfg.host_species, cfg.anion_species)
    centers = cation_anion_shell_centers(
        parent_structure,
        cfg.host_species,
        cfg.anion_species,
        max_shells=cfg.max_shells,
        tolerance=cfg.shell_tolerance_angstrom,
    )
    records: list[dict[str, Any]] = []
    for dopant_index in dopants:
        element = structure[dopant_index].species_string
        for vacancy_number, parent_oxygen_index in enumerate(missing_oxygen, start=1):
            vacancy_frac = parent_structure[parent_oxygen_index].frac_coords
            distance = float(
                structure.lattice.get_distance_and_image(
                    structure[dopant_index].frac_coords, vacancy_frac
                )[0]
            )
            shell = _shell_index(distance, centers)
            records.append(
                {
                    "target_id": target.target_id,
                    "parent_id": target.parent_id,
                    "composition": target.composition_label,
                    "structure_kind": target.kind,
                    "n_oxygen_vacancies": target.n_vacancies,
                    "pair": f"{element}-V_O",
                    "element": element,
                    "dopant_site_index": int(dopant_index),
                    "vacancy_number": vacancy_number,
                    "parent_oxygen_site_index": int(parent_oxygen_index),
                    "distance_angstrom": distance,
                    "shell": shell,
                    "shell_center_angstrom": centers[shell - 1] if shell is not None else None,
                    "energy_total_eV": target.energy_eV,
                }
            )
    return records


def warren_cowley_records(
    target: PreferenceTarget,
    structure: Structure,
    cfg: SitePreferenceConfig,
) -> list[dict[str, Any]]:
    cations, _, dopants = _indices_by_role(structure, cfg.host_species, cfg.anion_species)
    centers = cation_shell_centers(
        structure,
        cfg.host_species,
        cfg.anion_species,
        max_shells=cfg.max_shells,
        tolerance=cfg.shell_tolerance_angstrom,
    )
    dopant_species = sorted({structure[index].species_string for index in dopants})
    if not dopant_species or not centers:
        return []

    adjacency: dict[int, dict[int, list[int]]] = {
        shell: {index: [] for index in cations}
        for shell in range(1, len(centers) + 1)
    }
    for i, j in combinations(cations, 2):
        distance = float(structure.get_distance(i, j))
        shell = _shell_index(distance, centers)
        if shell is None:
            continue
        if abs(distance - centers[shell - 1]) > cfg.shell_tolerance_angstrom * 1.75:
            continue
        adjacency[shell][i].append(j)
        adjacency[shell][j].append(i)

    n_cations = len(cations)
    counts = Counter(structure[index].species_string for index in cations)
    records: list[dict[str, Any]] = []
    for a_pos, element_a in enumerate(dopant_species):
        for element_b in dopant_species[a_pos:]:
            count_a = counts[element_a]
            count_b = counts[element_b]
            if count_a <= 0 or count_b <= 0:
                continue
            if element_a == element_b:
                if n_cations <= 1 or count_b <= 1:
                    continue
                concentration_b = (count_b - 1) / (n_cations - 1)
            else:
                if n_cations <= 1:
                    continue
                concentration_b = count_b / (n_cations - 1)
            if concentration_b <= 0:
                continue

            a_sites = [index for index in cations if structure[index].species_string == element_a]
            for shell, shell_center in enumerate(centers, start=1):
                probabilities: list[float] = []
                for index in a_sites:
                    neighbors = adjacency[shell][index]
                    if not neighbors:
                        continue
                    b_neighbors = sum(
                        structure[neighbor].species_string == element_b
                        for neighbor in neighbors
                    )
                    probabilities.append(b_neighbors / len(neighbors))
                if not probabilities:
                    continue
                probability = sum(probabilities) / len(probabilities)
                alpha = 1.0 - probability / concentration_b
                records.append(
                    {
                        "target_id": target.target_id,
                        "parent_id": target.parent_id,
                        "composition": target.composition_label,
                        "structure_kind": target.kind,
                        "n_oxygen_vacancies": target.n_vacancies,
                        "pair": f"{element_a}-{element_b}",
                        "element_a": element_a,
                        "element_b": element_b,
                        "shell": shell,
                        "shell_center_angstrom": shell_center,
                        "conditional_probability": probability,
                        "random_probability": concentration_b,
                        "warren_cowley_alpha": alpha,
                        "interpretation": (
                            "association" if alpha < -0.05
                            else "avoidance" if alpha > 0.05
                            else "approximately-random"
                        ),
                        "energy_total_eV": target.energy_eV,
                    }
                )
    return records


def _group_energy_baselines(targets: Sequence[PreferenceTarget]) -> dict[tuple[str, str, int], float]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for target in targets:
        if target.energy_eV is not None:
            grouped[(target.composition_label, target.kind, target.n_vacancies)].append(
                target.energy_eV
            )
    return {key: min(values) for key, values in grouped.items() if values}


def _target_rows(
    targets: Sequence[PreferenceTarget],
    pair_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    pair_count = Counter(row["target_id"] for row in pair_records)
    baselines = _group_energy_baselines(targets)
    rows: list[dict[str, Any]] = []
    for target in targets:
        baseline = baselines.get(
            (target.composition_label, target.kind, target.n_vacancies)
        )
        delta = (
            target.energy_eV - baseline
            if target.energy_eV is not None and baseline is not None
            else None
        )
        structure = Structure.from_file(target.structure_path)
        rows.append(
            {
                "target_id": target.target_id,
                "parent_id": target.parent_id,
                "composition": target.composition_label,
                "structure_kind": target.kind,
                "n_oxygen_vacancies": target.n_vacancies,
                "formula": structure.composition.reduced_formula,
                "n_atoms": len(structure),
                "energy_total_eV": target.energy_eV,
                "delta_energy_within_group_eV": delta,
                "energy_source": target.energy_source,
                "structure_path": str(target.structure_path),
                "n_dopant_pairs": pair_count[target.target_id],
            }
        )
    return rows


def _nearest_pair_per_target(
    pair_records: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_meta = {row["target_id"]: row for row in target_rows}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_records:
        grouped[(row["target_id"], row["pair"])].append(row)

    output: list[dict[str, Any]] = []
    for (target_id, pair), records in grouped.items():
        nearest = min(records, key=lambda row: float(row["distance_angstrom"]))
        meta = target_meta[target_id]
        output.append(
            {
                "target_id": target_id,
                "composition": meta["composition"],
                "structure_kind": meta["structure_kind"],
                "n_oxygen_vacancies": meta["n_oxygen_vacancies"],
                "pair": pair,
                "nearest_distance_angstrom": nearest["distance_angstrom"],
                "nearest_shell": nearest["shell"],
                "energy_total_eV": meta["energy_total_eV"],
                "delta_energy_within_group_eV": meta["delta_energy_within_group_eV"],
            }
        )

    farthest_reference: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in output:
        key = (
            row["composition"],
            row["structure_kind"],
            int(row["n_oxygen_vacancies"]),
            row["pair"],
        )
        current = farthest_reference.get(key)
        if current is None or float(row["nearest_distance_angstrom"]) > float(
            current["nearest_distance_angstrom"]
        ):
            farthest_reference[key] = row
    for row in output:
        key = (
            row["composition"],
            row["structure_kind"],
            int(row["n_oxygen_vacancies"]),
            row["pair"],
        )
        ref = farthest_reference[key]
        if row["energy_total_eV"] is not None and ref["energy_total_eV"] is not None:
            row["delta_E_vs_farthest_eV"] = float(row["energy_total_eV"]) - float(
                ref["energy_total_eV"]
            )
        else:
            row["delta_E_vs_farthest_eV"] = None
        row["farthest_reference_target"] = ref["target_id"]
        row["proxy_note"] = (
            "Configuration-energy proxy only; other dopant arrangements may also differ."
        )
    return output


def _nearest_vacancy_pair_per_target(
    vacancy_records: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reduce each dopant–vacancy pair type to its nearest vacancy in each target."""
    target_meta = {row["target_id"]: row for row in target_rows}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in vacancy_records:
        grouped[(row["target_id"], row["pair"])].append(row)

    output: list[dict[str, Any]] = []
    for (target_id, pair), records in grouped.items():
        nearest = min(records, key=lambda row: float(row["distance_angstrom"]))
        meta = target_meta[target_id]
        output.append(
            {
                "target_id": target_id,
                "composition": meta["composition"],
                "structure_kind": meta["structure_kind"],
                "n_oxygen_vacancies": meta["n_oxygen_vacancies"],
                "pair": pair,
                "nearest_distance_angstrom": nearest["distance_angstrom"],
                "nearest_shell": nearest["shell"],
                "energy_total_eV": meta["energy_total_eV"],
                "delta_energy_within_group_eV": meta["delta_energy_within_group_eV"],
            }
        )

    farthest_reference: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in output:
        key = (
            row["composition"],
            row["structure_kind"],
            int(row["n_oxygen_vacancies"]),
            row["pair"],
        )
        current = farthest_reference.get(key)
        if current is None or float(row["nearest_distance_angstrom"]) > float(
            current["nearest_distance_angstrom"]
        ):
            farthest_reference[key] = row

    for row in output:
        key = (
            row["composition"],
            row["structure_kind"],
            int(row["n_oxygen_vacancies"]),
            row["pair"],
        )
        reference = farthest_reference[key]
        if row["energy_total_eV"] is not None and reference["energy_total_eV"] is not None:
            row["delta_E_vs_farthest_eV"] = float(row["energy_total_eV"]) - float(
                reference["energy_total_eV"]
            )
        else:
            row["delta_E_vs_farthest_eV"] = None
        row["farthest_reference_target"] = reference["target_id"]
        row["proxy_note"] = (
            "Configuration-energy proxy: vacancy position and other local arrangements can also differ."
        )
    return output


def _aggregate_pair_preferences(nearest_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in nearest_rows:
        shell = row.get("nearest_shell")
        if shell is None:
            continue
        key = (
            str(row["composition"]),
            str(row["structure_kind"]),
            int(row["n_oxygen_vacancies"]),
            str(row["pair"]),
            int(shell),
        )
        grouped[key].append(row)

    shell_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        composition, kind, n_vac, pair, shell = key
        energies = [
            float(row["delta_energy_within_group_eV"])
            for row in rows
            if row.get("delta_energy_within_group_eV") is not None
        ]
        proxies = [
            float(row["delta_E_vs_farthest_eV"])
            for row in rows
            if row.get("delta_E_vs_farthest_eV") is not None
        ]
        distances = [float(row["nearest_distance_angstrom"]) for row in rows]
        shell_rows.append(
            {
                "composition": composition,
                "structure_kind": kind,
                "n_oxygen_vacancies": n_vac,
                "pair": pair,
                "shell": shell,
                "n_targets": len(rows),
                "mean_nearest_distance_angstrom": sum(distances) / len(distances),
                "minimum_delta_energy_within_group_eV": min(energies) if energies else None,
                "mean_delta_energy_within_group_eV": (
                    sum(energies) / len(energies) if energies else None
                ),
                "median_delta_energy_within_group_eV": median(energies) if energies else None,
                "mean_delta_E_vs_farthest_eV": (
                    sum(proxies) / len(proxies) if proxies else None
                ),
            }
        )

    by_pair: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in shell_rows:
        by_pair[
            (
                row["composition"],
                row["structure_kind"],
                row["n_oxygen_vacancies"],
                row["pair"],
            )
        ].append(row)

    preferred: list[dict[str, Any]] = []
    for key, rows in by_pair.items():
        with_energy = [
            row for row in rows if row["median_delta_energy_within_group_eV"] is not None
        ]
        if with_energy:
            best = min(
                with_energy,
                key=lambda row: (
                    float(row["median_delta_energy_within_group_eV"]),
                    float(row["mean_delta_energy_within_group_eV"]),
                    int(row["shell"]),
                ),
            )
            basis = "lowest median configuration energy"
        else:
            best = max(rows, key=lambda row: (int(row["n_targets"]), -int(row["shell"])))
            basis = "most frequently observed shell; no comparable energies available"
        composition, kind, n_vac, pair = key
        preferred.append(
            {
                "composition": composition,
                "structure_kind": kind,
                "n_oxygen_vacancies": n_vac,
                "pair": pair,
                "preferred_shell": best["shell"],
                "preferred_shell_mean_distance_angstrom": best[
                    "mean_nearest_distance_angstrom"
                ],
                "n_targets_in_preferred_shell": best["n_targets"],
                "preference_basis": basis,
                "median_delta_energy_within_group_eV": best[
                    "median_delta_energy_within_group_eV"
                ],
                "mean_delta_E_vs_farthest_eV": best["mean_delta_E_vs_farthest_eV"],
                "interpretation_note": (
                    "This is an observed configuration preference. A controlled pair scan "
                    "is the stricter test of an isolated pair interaction."
                ),
            }
        )
    preferred.sort(
        key=lambda row: (
            row["composition"],
            row["n_oxygen_vacancies"],
            row["pair"],
        )
    )
    return preferred


def _auto_pair_types(
    targets: Sequence[PreferenceTarget],
    cfg: SitePreferenceConfig,
) -> tuple[tuple[str, str], ...]:
    species: set[str] = set()
    for target in targets:
        if target.kind != "vacancy-free":
            continue
        structure = Structure.from_file(target.structure_path)
        _, _, dopants = _indices_by_role(structure, cfg.host_species, cfg.anion_species)
        species.update(structure[index].species_string for index in dopants)
    ordered = sorted(species)
    return tuple((a, b) for pos, a in enumerate(ordered) for b in ordered[pos:])


def _select_pair_scan_source(
    targets: Sequence[PreferenceTarget],
    parent_map: dict[str, PreferenceTarget],
    selector: str,
) -> PreferenceTarget:
    parents = list(parent_map.values())
    if selector:
        matched = [target for target in parents if _target_matches(target, (selector,))]
        if not matched:
            raise ValueError(
                f"[site_preference.pair_scan].source_target matched no selected parent: {selector}"
            )
        parents = matched
    with_energy = [target for target in parents if target.energy_eV is not None]
    if with_energy:
        return min(with_energy, key=lambda target: float(target.energy_eV))
    if parents:
        return sorted(parents, key=lambda target: target.target_id)[0]
    raise RuntimeError("No vacancy-free selected parent is available as pair-scan source")


def _build_calculator(settings: PairScanConfig | OrderingMCConfig, stage_name: str) -> Any:
    from dopingflow.ml_backends import (
        build_ase_calculator,
        check_backend_dependency,
        prepare_backend_runtime,
    )

    check_backend_dependency(settings.backend, stage_name=stage_name)
    prepare_backend_runtime(
        backend=settings.backend,
        device=settings.device,
        gpu_id=settings.gpu_id,
        tf_threads=settings.tf_threads,
        omp_threads=settings.omp_threads,
    )
    return build_ase_calculator(
        backend=settings.backend,
        model=settings.model,
        task=settings.task,
        device=settings.device,
    )


def _host_only_parent(
    structure: Structure, host_species: str, anion_species: Sequence[str]
) -> tuple[Structure, list[int]]:
    parent = structure.copy()
    cations, _, _ = _indices_by_role(parent, host_species, anion_species)
    for index in cations:
        parent[index] = host_species
    return parent, cations


def symmetry_distinct_pair_orbits(
    structure: Structure,
    cation_indices: Sequence[int],
    element_a: str,
    element_b: str,
    *,
    max_shells: int,
    tolerance: float,
    symprec: float,
    angle_tolerance: float,
) -> list[dict[str, Any]]:
    """Enumerate symmetry-distinct placements for one dopant pair on the cation sublattice.

    Unlike dopants are treated as labelled species, so A-at-i/B-at-j and A-at-j/B-at-i
    collapse only when a host-lattice symmetry operation actually maps one assignment
    onto the other. This prevents a radial-distance-only scan from hiding inequivalent
    crystallographic orientations.
    """
    from dopingflow.utils.symmetry import (
        build_sublattice_symmetry_permutations,
        canonical_occupancy_key,
    )

    indices = list(cation_indices)
    centers = _cluster_distances(
        (structure.get_distance(i, j) for i, j in combinations(indices, 2)),
        tolerance,
        max_shells,
    )
    if not centers:
        return []

    permutations = build_sublattice_symmetry_permutations(
        structure,
        indices,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    n_sites = len(indices)
    same_species = element_a == element_b
    assignments: Iterable[tuple[int, int]]
    if same_species:
        assignments = combinations(range(n_sites), 2)
    else:
        assignments = (
            (i, j)
            for i in range(n_sites)
            for j in range(n_sites)
            if i != j
        )

    orbits: dict[bytes, dict[str, Any]] = {}
    for pos_i, pos_j in assignments:
        labels = [0] * n_sites
        labels[pos_i] = 1
        labels[pos_j] = 1 if same_species else 2
        key = canonical_occupancy_key(labels, permutations)
        if key in orbits:
            orbits[key]["degeneracy"] += 1
            continue

        site_i = indices[pos_i]
        site_j = indices[pos_j]
        distance, image = structure.lattice.get_distance_and_image(
            structure[site_i].frac_coords,
            structure[site_j].frac_coords,
        )
        shell = _shell_index(float(distance), centers)
        if shell is None or shell > max_shells:
            continue
        delta_frac = (
            structure[site_j].frac_coords
            + image
            - structure[site_i].frac_coords
        )
        orbits[key] = {
            "canonical_key": key.hex(),
            "site_i": int(site_i),
            "site_j": int(site_j),
            "shell": int(shell),
            "shell_center_angstrom": float(centers[shell - 1]),
            "initial_distance_angstrom": float(distance),
            "relative_fractional_vector": [float(value) for value in delta_frac],
            "relative_cartesian_vector_angstrom": [
                float(value)
                for value in structure.lattice.get_cartesian_coords(delta_frac)
            ],
            "degeneracy": 1,
            "symmetry_operation_count": len(permutations),
        }

    rows = sorted(
        orbits.values(),
        key=lambda row: (
            int(row["shell"]),
            float(row["initial_distance_angstrom"]),
            row["canonical_key"],
        ),
    )
    per_shell_count: dict[int, int] = defaultdict(int)
    for row in rows:
        shell = int(row["shell"])
        per_shell_count[shell] += 1
        row["orbit"] = per_shell_count[shell]
    return rows


def run_pair_scan(
    cfg: SitePreferenceConfig,
    targets: Sequence[PreferenceTarget],
    parent_map: dict[str, PreferenceTarget],
) -> list[dict[str, Any]]:
    settings = cfg.pair_scan
    if not settings.enabled:
        return []

    source = _select_pair_scan_source(targets, parent_map, settings.source_target)
    symmetry_path = Path(str(source.metadata.get("symmetry_path") or source.structure_path))
    source_structure = Structure.from_file(symmetry_path)
    parent, cations = _host_only_parent(
        source_structure, cfg.host_species, cfg.anion_species
    )

    pairs = settings.pairs or _auto_pair_types(targets, cfg)
    if not pairs:
        raise RuntimeError(
            "Pair scan could not infer dopant pairs; set [site_preference.pair_scan].pairs"
        )

    calculator = _build_calculator(settings, "Site-preference pair scan") if settings.execute else None
    from dopingflow.ml_relaxation import (
        relax_structure_with_calculator,
        structure_energy_with_calculator,
    )

    root = cfg.output_dir / "pair_scan"
    rows: list[dict[str, Any]] = []
    for element_a, element_b in pairs:
        pair_name = f"{element_a}-{element_b}"
        orbits = symmetry_distinct_pair_orbits(
            parent,
            cations,
            element_a,
            element_b,
            max_shells=settings.max_shells,
            tolerance=cfg.shell_tolerance_angstrom,
            symprec=settings.symprec,
            angle_tolerance=settings.angle_tolerance,
        )
        if not orbits:
            log.warning("No pair orbits found for %s", pair_name)
            continue
        for orbit_record in orbits:
            shell = int(orbit_record["shell"])
            orbit = int(orbit_record["orbit"])
            i = int(orbit_record["site_i"])
            j = int(orbit_record["site_j"])
            initial_distance = float(orbit_record["initial_distance_angstrom"])
            structure = parent.copy()
            structure[i] = element_a
            structure[j] = element_b
            run_dir = (
                root
                / pair_name
                / f"shell_{shell:02d}"
                / f"orbit_{orbit:02d}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            Poscar(structure).write_file(run_dir / "POSCAR_initial")

            row: dict[str, Any] = {
                "pair": pair_name,
                "element_a": element_a,
                "element_b": element_b,
                **orbit_record,
                "source_target": source.target_id,
                "source_structure": str(symmetry_path),
                "backend": settings.backend,
                "model": settings.model,
                "task": settings.task,
                "executed": settings.execute,
                "relaxed": False,
                "energy_total_eV": None,
                "final_distance_angstrom": initial_distance,
                "converged": None,
            }
            if settings.execute:
                if settings.relax:
                    relaxed, energy, nsteps, final_force, converged = (
                        relax_structure_with_calculator(
                            structure,
                            calculator=calculator,
                            optimizer_name=settings.optimizer,
                            fmax=settings.fmax,
                            max_steps=settings.max_steps,
                            relax_mode=settings.relax_mode,
                            cell_filter=settings.cell_filter,
                        )
                    )
                    Poscar(relaxed).write_file(run_dir / "POSCAR_relaxed")
                    row.update(
                        {
                            "relaxed": True,
                            "energy_total_eV": float(energy),
                            "final_distance_angstrom": float(relaxed.get_distance(i, j)),
                            "optimizer_steps": int(nsteps),
                            "final_fmax_eV_per_A": float(final_force),
                            "converged": bool(converged),
                        }
                    )
                else:
                    energy = structure_energy_with_calculator(structure, calculator)
                    row["energy_total_eV"] = float(energy)
                _write_json(run_dir / "result.json", row)
            rows.append(row)

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair"]].append(row)
    for pair_rows in by_pair.values():
        evaluated = [row for row in pair_rows if row["energy_total_eV"] is not None]
        if not evaluated:
            continue
        farthest_shell = max(int(row["shell"]) for row in evaluated)
        farthest_candidates = [
            row for row in evaluated if int(row["shell"]) == farthest_shell
        ]
        farthest = min(
            farthest_candidates,
            key=lambda row: float(row["energy_total_eV"]),
        )
        far_energy = float(farthest["energy_total_eV"])
        for row in pair_rows:
            if row["energy_total_eV"] is not None:
                row["delta_E_vs_farthest_eV"] = float(row["energy_total_eV"]) - far_energy
                row["farthest_reference_shell"] = farthest["shell"]
                row["farthest_reference_orbit"] = farthest["orbit"]
            else:
                row["delta_E_vs_farthest_eV"] = None
                row["farthest_reference_shell"] = None
                row["farthest_reference_orbit"] = None

    _write_csv(root / "pair_scan.csv", rows)
    _write_json(root / "pair_scan.json", rows)
    return rows


def _mc_selected_targets(
    cfg: SitePreferenceConfig,
    targets: Sequence[PreferenceTarget],
) -> list[PreferenceTarget]:
    candidates = [
        target for target in targets
        if target.kind == "vacancy-free"
        and _target_matches(target, cfg.ordering_mc.target_include)
    ]
    by_composition: dict[str, list[PreferenceTarget]] = defaultdict(list)
    for target in candidates:
        by_composition[target.composition_label].append(target)
    selected: list[PreferenceTarget] = []
    for composition in sorted(by_composition):
        group = by_composition[composition]
        with_energy = [target for target in group if target.energy_eV is not None]
        if with_energy:
            selected.append(min(with_energy, key=lambda target: float(target.energy_eV)))
        else:
            selected.append(sorted(group, key=lambda target: target.target_id)[0])
    return selected[: cfg.ordering_mc.max_targets]


def _mc_sro_snapshot(
    structure: Structure,
    target: PreferenceTarget,
    cfg: SitePreferenceConfig,
) -> list[dict[str, Any]]:
    return warren_cowley_records(target, structure, cfg)


def run_ordering_mc(
    cfg: SitePreferenceConfig,
    targets: Sequence[PreferenceTarget],
) -> list[dict[str, Any]]:
    settings = cfg.ordering_mc
    if not settings.enabled:
        return []

    selected = _mc_selected_targets(cfg, targets)
    if not selected:
        raise RuntimeError("Ordering MC selected no vacancy-free structures")

    plan = [
        {
            "target_id": target.target_id,
            "structure_path": str(target.structure_path),
            "temperature_K": settings.temperature_K,
            "steps": settings.steps,
            "execute": settings.execute,
        }
        for target in selected
    ]
    root = cfg.output_dir / "ordering_mc"
    _write_json(root / "mc_plan.json", plan)
    if not settings.execute:
        return plan

    calculator = _build_calculator(settings, "Site-preference ordering MC")
    from dopingflow.ml_relaxation import (
        relax_structure_with_calculator,
        structure_energy_with_calculator,
    )

    k_b_eV_per_K = 8.617333262145e-5
    beta = 1.0 / (k_b_eV_per_K * settings.temperature_K)
    summaries: list[dict[str, Any]] = []

    for target_index, target in enumerate(selected):
        rng = random.Random(settings.seed + target_index)
        symmetry_source = Path(
            str(target.metadata.get("symmetry_path") or target.structure_path)
        )
        current = Structure.from_file(symmetry_source)
        cations, _, _ = _indices_by_role(current, cfg.host_species, cfg.anion_species)
        movable = [
            index for index in cations
            if current[index].species_string == cfg.host_species
            or current[index].species_string not in cfg.anion_species
        ]
        distinct_species = {current[index].species_string for index in movable}
        if len(distinct_species) < 2:
            log.warning("Skipping MC for %s: only one cation species", target.target_id)
            continue

        current_energy = float(structure_energy_with_calculator(current, calculator))
        start_energy = current_energy
        best = current.copy()
        best_energy = current_energy
        accepted = 0
        attempted = 0
        samples = 0
        sro_accumulator: dict[tuple[str, int], list[float]] = defaultdict(list)
        trace: list[dict[str, Any]] = []

        for step in range(1, settings.steps + 1):
            different = False
            for _ in range(50):
                i, j = rng.sample(movable, 2)
                if current[i].species_string != current[j].species_string:
                    different = True
                    break
            if not different:
                break

            proposal = current.copy()
            species_i = proposal[i].species_string
            species_j = proposal[j].species_string
            proposal[i] = species_j
            proposal[j] = species_i
            proposal_energy = float(structure_energy_with_calculator(proposal, calculator))
            delta = proposal_energy - current_energy
            accept = delta <= 0 or rng.random() < math.exp(-beta * delta)
            attempted += 1
            if accept:
                current = proposal
                current_energy = proposal_energy
                accepted += 1
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy

            if step > settings.burn_in and (
                (step - settings.burn_in) % settings.sample_interval == 0
            ):
                samples += 1
                pseudo_target = PreferenceTarget(
                    target_id=target.target_id,
                    parent_id=target.parent_id,
                    kind=target.kind,
                    structure_path=target.structure_path,
                    energy_eV=current_energy,
                    energy_source="ordering_mc",
                )
                for row in _mc_sro_snapshot(current, pseudo_target, cfg):
                    sro_accumulator[(row["pair"], int(row["shell"]))].append(
                        float(row["warren_cowley_alpha"])
                    )
            if step == 1 or step % max(1, settings.sample_interval * 10) == 0:
                trace.append(
                    {
                        "step": step,
                        "energy_eV": current_energy,
                        "best_energy_eV": best_energy,
                        "accepted": accepted,
                        "attempted": attempted,
                    }
                )

        target_dir = root / target.safe_id
        target_dir.mkdir(parents=True, exist_ok=True)
        Poscar(best).write_file(target_dir / "POSCAR_best_mc")
        best_relaxed_energy: float | None = None
        if settings.relax_best:
            relaxed, best_relaxed_energy, nsteps, final_force, converged = (
                relax_structure_with_calculator(
                    best,
                    calculator=calculator,
                    optimizer_name=settings.optimizer,
                    fmax=settings.fmax,
                    max_steps=settings.max_steps,
                    relax_mode=settings.relax_mode,
                    cell_filter=settings.cell_filter,
                )
            )
            Poscar(relaxed).write_file(target_dir / "POSCAR_best_relaxed")
            relaxation = {
                "energy_relaxed_eV": float(best_relaxed_energy),
                "optimizer_steps": int(nsteps),
                "final_fmax_eV_per_A": float(final_force),
                "converged": bool(converged),
            }
        else:
            relaxation = None

        sro_rows = [
            {
                "target_id": target.target_id,
                "pair": pair,
                "shell": shell,
                "temperature_K": settings.temperature_K,
                "n_samples": len(values),
                "mean_warren_cowley_alpha": sum(values) / len(values),
                "median_warren_cowley_alpha": median(values),
            }
            for (pair, shell), values in sorted(sro_accumulator.items())
            if values
        ]
        _write_csv(target_dir / "sro_temperature_average.csv", sro_rows)
        _write_json(target_dir / "trace.json", trace)
        summary = {
            "target_id": target.target_id,
            "temperature_K": settings.temperature_K,
            "steps_requested": settings.steps,
            "steps_attempted": attempted,
            "accepted_moves": accepted,
            "acceptance_fraction": accepted / attempted if attempted else 0.0,
            "samples": samples,
            "source_relaxed_energy_eV": target.energy_eV,
            "mc_sampling_geometry": str(symmetry_source),
            "mc_start_energy_eV": start_energy,
            "best_mc_energy_eV": best_energy,
            "relaxation": relaxation,
            "sro_temperature_average": sro_rows,
            "backend": settings.backend,
            "model": settings.model,
            "task": settings.task,
        }
        _write_json(target_dir / "summary.json", summary)
        summaries.append(summary)

    _write_json(root / "ordering_mc_summary.json", summaries)
    return summaries


def run_site_preference(
    raw: dict[str, Any],
    root: Path,
) -> Path | None:
    cfg = parse_site_preference_config(raw, root)
    if not cfg.enabled:
        return None

    targets, parent_map, warnings = discover_site_preference_targets(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    pair_records: list[dict[str, Any]] = []
    vacancy_records: list[dict[str, Any]] = []
    sro_records: list[dict[str, Any]] = []
    triplet_records: list[dict[str, Any]] = []
    analysis_errors: list[dict[str, str]] = []

    for target in targets:
        try:
            structure = Structure.from_file(target.structure_path)
            pair_records.extend(dopant_pair_records(target, structure, cfg))
            triplet_records.extend(dopant_triplet_records(target, structure, cfg))
            sro_records.extend(warren_cowley_records(target, structure, cfg))
            if target.kind == "oxygen-vacancy":
                parent_target = parent_map[target.parent_id]
                parent_structure = Structure.from_file(parent_target.structure_path)
                vacancy_records.extend(
                    dopant_vacancy_records(
                        target,
                        structure,
                        parent_structure,
                        cfg,
                    )
                )
        except Exception as exc:
            log.exception("Site-preference analysis failed for %s", target.target_id)
            analysis_errors.append(
                {"target_id": target.target_id, "error": f"{type(exc).__name__}: {exc}"}
            )

    target_rows = _target_rows(targets, pair_records)
    nearest_rows = _nearest_pair_per_target(pair_records, target_rows)
    preference_rows = _aggregate_pair_preferences(nearest_rows)
    nearest_vacancy_rows = _nearest_vacancy_pair_per_target(vacancy_records, target_rows)
    vacancy_preference_rows = _aggregate_pair_preferences(nearest_vacancy_rows)
    for row in vacancy_preference_rows:
        row["interpretation_note"] = (
            "Observed dopant–oxygen-vacancy shell preference from same-composition vacancy "
            "configurations; vacancy placement and other local ordering can also affect the energy."
        )
    triplet_target_rows = _triplet_target_presence(triplet_records, target_rows)
    triplet_preference_rows = _aggregate_triplet_preferences(triplet_target_rows)

    _write_csv(cfg.output_dir / "site_preference_targets.csv", target_rows)
    _write_csv(cfg.output_dir / "dopant_pairs.csv", pair_records)
    _write_csv(cfg.output_dir / "dopant_triplets.csv", triplet_records)
    _write_csv(cfg.output_dir / "triplet_target_motifs.csv", triplet_target_rows)
    _write_csv(cfg.output_dir / "triplet_motif_summary.csv", triplet_preference_rows)
    _write_csv(cfg.output_dir / "dopant_vacancy_pairs.csv", vacancy_records)
    _write_csv(
        cfg.output_dir / "nearest_dopant_vacancy_by_target.csv",
        nearest_vacancy_rows,
    )
    _write_csv(
        cfg.output_dir / "dopant_vacancy_preference_summary.csv",
        vacancy_preference_rows,
    )
    _write_csv(cfg.output_dir / "warren_cowley_sro.csv", sro_records)
    _write_csv(cfg.output_dir / "nearest_pair_by_target.csv", nearest_rows)
    _write_csv(cfg.output_dir / "pair_preference_summary.csv", preference_rows)

    pair_scan_rows = run_pair_scan(cfg, targets, parent_map)
    mc_rows = run_ordering_mc(cfg, targets)

    payload = {
        "stage": "site_preference",
        "host_species": cfg.host_species,
        "anion_species": list(cfg.anion_species),
        "n_targets": len(targets),
        "n_dopant_pair_records": len(pair_records),
        "n_dopant_vacancy_records": len(vacancy_records),
        "n_dopant_vacancy_preference_rows": len(vacancy_preference_rows),
        "n_sro_records": len(sro_records),
        "n_triplet_records": len(triplet_records),
        "n_triplet_preference_rows": len(triplet_preference_rows),
        "n_pair_preference_rows": len(preference_rows),
        "pair_scan_enabled": cfg.pair_scan.enabled,
        "pair_scan_executed": cfg.pair_scan.execute,
        "n_pair_scan_rows": len(pair_scan_rows),
        "ordering_mc_enabled": cfg.ordering_mc.enabled,
        "ordering_mc_executed": cfg.ordering_mc.execute,
        "n_ordering_mc_targets": len(mc_rows),
        "warnings": warnings,
        "analysis_errors": analysis_errors,
        "outputs": {
            "targets": str(cfg.output_dir / "site_preference_targets.csv"),
            "dopant_pairs": str(cfg.output_dir / "dopant_pairs.csv"),
            "dopant_triplets": str(cfg.output_dir / "dopant_triplets.csv"),
            "triplet_target_motifs": str(cfg.output_dir / "triplet_target_motifs.csv"),
            "triplet_motif_summary": str(cfg.output_dir / "triplet_motif_summary.csv"),
            "dopant_vacancy_pairs": str(cfg.output_dir / "dopant_vacancy_pairs.csv"),
            "nearest_dopant_vacancy_by_target": str(
                cfg.output_dir / "nearest_dopant_vacancy_by_target.csv"
            ),
            "dopant_vacancy_preference_summary": str(
                cfg.output_dir / "dopant_vacancy_preference_summary.csv"
            ),
            "warren_cowley_sro": str(cfg.output_dir / "warren_cowley_sro.csv"),
            "nearest_pair_by_target": str(cfg.output_dir / "nearest_pair_by_target.csv"),
            "pair_preference_summary": str(cfg.output_dir / "pair_preference_summary.csv"),
            "pair_scan": str(cfg.output_dir / "pair_scan" / "pair_scan.csv"),
            "ordering_mc": str(cfg.output_dir / "ordering_mc" / "ordering_mc_summary.json"),
        },
        "interpretation": {
            "warren_cowley_alpha": {
                "negative": "association relative to random occupancy",
                "zero": "approximately random occupancy",
                "positive": "avoidance relative to random occupancy",
            },
            "configuration_energy": (
                "Pair preferences from existing structures correlate nearest pair shell with "
                "same-composition configuration energies; they are not isolated pair-binding energies."
            ),
            "triplet_motifs": (
                "Three-dopant motifs are classified from host-cation neighbor-shell connectivity "
                "as compact triangles, connected chains, isolated pairs plus a third dopant, or dispersed."
            ),
            "pair_scan": (
                "The controlled pair scan replaces all cations by the host except the selected "
                "dopant pair and compares all symmetry-distinct pair orientations in the requested shells."
            ),
            "ordering_mc": (
                "Ordering MC preserves composition and swaps cation identities on the pre-relaxation "
                "symmetry geometry so the starting ordering does not receive a relaxation bias; only "
                "the best sampled occupation is optionally relaxed afterwards."
            ),
        },
        "config": {
            **asdict(cfg),
            "root": str(cfg.root),
            "source_root": str(cfg.source_root),
            "output_dir": str(cfg.output_dir),
        },
    }
    output = cfg.output_dir / "site_preference_summary.json"
    _write_json(output, payload)
    return output


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def run_site_preference_from_toml(config_path: Path) -> Path | None:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_site_preference(raw, config_path.resolve().parent)

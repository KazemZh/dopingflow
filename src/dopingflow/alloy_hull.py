"""Restricted one-dimensional alloy convex hull for a single dopant path.

The stage constructs the lower convex envelope along a fixed composition line,
for example Sn_(1-x)Sb_xO2.  It is distinct from the full multicomponent phase
 diagram, which may decompose a candidate into phases off that composition line.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pymatgen.core import Composition, Structure

log = logging.getLogger(__name__)

REF_JSON = Path("reference_structures/reference_energies.json")
OUT_DB = "results_database.csv"
OUT_RESULTS = "alloy_hull_results.csv"
OUT_VERTICES = "alloy_hull_vertices.csv"
OUT_SUMMARY = "alloy_hull_summary.json"
RELAX_META = Path("02_relax") / "meta.json"
RELAX_POSCAR = Path("02_relax") / "POSCAR"

X_TOL = 1e-10
ENERGY_TOL = 1e-8


@dataclass(frozen=True)
class AlloyHullConfig:
    host_species: str
    anion_species: str
    dopant: str | None
    endpoint_reference: str | None


@dataclass(frozen=True)
class CandidatePoint:
    candidate: str
    composition_tag: str
    candidate_path: Path
    formula: str
    x: float
    energy_total_eV: float
    n_cations: float
    energy_per_cation_eV: float

    @property
    def label(self) -> str:
        return f"{self.composition_tag}/{self.candidate}"


@dataclass(frozen=True)
class HullPoint:
    x: float
    energy_per_cation_eV: float
    label: str
    source: str
    candidate_path: str = ""
    candidate: str = ""
    composition_tag: str = ""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse_config(raw_cfg: dict[str, Any]) -> AlloyHullConfig:
    doping = raw_cfg.get("doping", {}) or {}
    scan = raw_cfg.get("scan", {}) or {}
    alloy = raw_cfg.get("alloy_hull", {}) or {}

    host_species = str(doping.get("host_species", "")).strip()
    if not host_species:
        raise ValueError("[doping].host_species is required for alloy-hull")

    anions = [str(x) for x in (scan.get("anion_species", ["O"]) or [])]
    if len(anions) != 1:
        raise ValueError("alloy-hull currently supports exactly one anion species")

    dopant_raw = str(alloy.get("dopant", "auto")).strip()
    dopant = None if dopant_raw.lower() in {"", "auto"} else dopant_raw

    endpoint_raw = str(alloy.get("endpoint_reference", "auto")).strip()
    if endpoint_raw.lower() in {"", "auto"}:
        endpoint_reference: str | None = "auto"
    elif endpoint_raw.lower() in {"none", "off", "false"}:
        endpoint_reference = None
    else:
        endpoint_reference = endpoint_raw

    return AlloyHullConfig(
        host_species=host_species,
        anion_species=anions[0],
        dopant=dopant,
        endpoint_reference=endpoint_reference,
    )


def _read_relaxed_energy(candidate_dir: Path) -> float:
    meta_path = candidate_dir / RELAX_META
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing relaxed-energy metadata: {meta_path}")
    data = _load_json(meta_path)
    if "energy_relaxed_eV" not in data:
        raise KeyError(f"{meta_path} missing energy_relaxed_eV")
    return float(data["energy_relaxed_eV"])


def _candidate_elements(structure: Structure, host: str, anion: str) -> List[str]:
    elements = sorted({str(element) for element in structure.composition.as_dict()})
    return [element for element in elements if element not in {host, anion}]


def _read_candidates(
    root: Path,
    *,
    host: str,
    anion: str,
    requested_dopant: str | None,
) -> tuple[str, List[CandidatePoint]]:
    database_path = root / OUT_DB
    if not database_path.exists():
        raise FileNotFoundError(f"Missing {OUT_DB}. Run collect first.")

    raw_records: List[tuple[dict[str, str], Structure, Path]] = []
    discovered_dopants: set[str] = set()

    with database_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate_path_text = (row.get("candidate_path") or "").strip()
            candidate = (row.get("candidate") or "").strip()
            if not candidate_path_text or not candidate:
                continue

            candidate_path = Path(candidate_path_text)
            poscar_path = candidate_path / RELAX_POSCAR
            if not poscar_path.exists():
                log.warning("Skipping %s: missing %s", candidate_path, RELAX_POSCAR)
                continue

            structure = Structure.from_file(str(poscar_path))
            dopants_here = _candidate_elements(structure, host, anion)
            if not dopants_here:
                continue
            if len(dopants_here) != 1:
                raise ValueError(
                    f"Candidate {candidate_path} contains multiple dopants {dopants_here}. "
                    "Use a separate one-dimensional alloy-hull path for each dopant."
                )

            discovered_dopants.add(dopants_here[0])
            raw_records.append((row, structure, candidate_path))

    if not raw_records:
        raise ValueError("No doped candidates were found in results_database.csv")

    if requested_dopant is None:
        if len(discovered_dopants) != 1:
            raise ValueError(
                "[alloy_hull].dopant is required because multiple dopants were found: "
                + ", ".join(sorted(discovered_dopants))
            )
        dopant = next(iter(discovered_dopants))
    else:
        dopant = requested_dopant

    candidates: List[CandidatePoint] = []
    for row, structure, candidate_path in raw_records:
        dopants_here = _candidate_elements(structure, host, anion)
        if dopants_here != [dopant]:
            continue

        composition = structure.composition.as_dict()
        n_host = float(composition.get(host, 0.0))
        n_dopant = float(composition.get(dopant, 0.0))
        n_cations = n_host + n_dopant
        if n_host < -X_TOL or n_dopant <= 0.0 or n_cations <= 0.0:
            continue

        candidates.append(
            CandidatePoint(
                candidate=(row.get("candidate") or candidate_path.name).strip(),
                composition_tag=(row.get("composition_tag") or candidate_path.parent.name).strip(),
                candidate_path=candidate_path,
                formula=structure.composition.reduced_formula,
                x=n_dopant / n_cations,
                energy_total_eV=_read_relaxed_energy(candidate_path),
                n_cations=n_cations,
                energy_per_cation_eV=_read_relaxed_energy(candidate_path) / n_cations,
            )
        )

    if not candidates:
        raise ValueError(f"No candidates with dopant {dopant!r} were found.")

    return dopant, candidates


def _host_point(ref: dict[str, Any], host: str) -> tuple[HullPoint, float]:
    host_ref = ref.get("host", {}) or {}
    formula = str(host_ref.get("name", "")).strip()
    if not formula:
        raise KeyError("reference_energies.json missing host.name")
    if "E_unit_total_eV" not in host_ref or "n_atoms_unit" not in host_ref:
        raise KeyError("reference_energies.json missing host energy information")

    composition = Composition(formula).reduced_composition.as_dict()
    n_host_fu = float(composition.get(host, 0.0))
    if n_host_fu <= 0.0:
        raise ValueError(f"Host formula {formula} does not contain host species {host}")

    atoms_per_fu = float(sum(composition.values()))
    n_fu = float(host_ref["n_atoms_unit"]) / atoms_per_fu
    energy_per_fu = float(host_ref["E_unit_total_eV"]) / n_fu
    energy_per_cation = energy_per_fu / n_host_fu
    anion_per_host = sum(v for element, v in composition.items() if element != host) / n_host_fu

    return (
        HullPoint(
            x=0.0,
            energy_per_cation_eV=energy_per_cation,
            label=formula,
            source="host_reference",
        ),
        anion_per_host,
    )


def _composition_from_reference(name: str, entry: Dict[str, Any]) -> Dict[str, float]:
    raw = entry.get("reduced_composition")
    if isinstance(raw, dict):
        return {str(element): float(amount) for element, amount in raw.items()}
    return {
        str(element): float(amount)
        for element, amount in Composition(name).reduced_composition.as_dict().items()
    }


def _endpoint_point(
    ref: dict[str, Any],
    *,
    dopant: str,
    anion: str,
    host_anion_per_cation: float,
    requested_reference: str | None,
) -> HullPoint | None:
    if requested_reference is None:
        return None

    references = ref.get("references", {}) or {}

    if requested_reference == "auto":
        names: Iterable[str] = sorted(references)
    else:
        if requested_reference not in references:
            raise KeyError(
                f"[alloy_hull].endpoint_reference={requested_reference!r} was not found "
                "in reference_energies.json"
            )
        names = [requested_reference]

    matches: List[HullPoint] = []
    for name in names:
        entry = references.get(name)
        if not isinstance(entry, dict) or entry.get("type") != "oxide":
            continue
        if "E_per_formula_unit_eV" not in entry:
            continue

        composition = _composition_from_reference(str(name), entry)
        elements = set(composition)
        if elements != {dopant, anion}:
            continue

        n_dopant = float(composition[dopant])
        if n_dopant <= 0.0:
            continue
        anion_per_dopant = float(composition[anion]) / n_dopant
        if abs(anion_per_dopant - host_anion_per_cation) > X_TOL:
            continue

        matches.append(
            HullPoint(
                x=1.0,
                energy_per_cation_eV=float(entry["E_per_formula_unit_eV"]) / n_dopant,
                label=str(name),
                source="endpoint_reference",
            )
        )

    if requested_reference != "auto" and not matches:
        raise ValueError(
            f"Endpoint reference {requested_reference!r} is not a binary {dopant}-{anion} "
            f"oxide with the same {anion}/cation ratio as the host line."
        )

    if not matches:
        log.warning(
            "No stoichiometrically matched endpoint reference was found. "
            "The 1D hull will be limited to the sampled composition range."
        )
        return None

    return min(matches, key=lambda point: point.energy_per_cation_eV)


def _candidate_hull_points(candidates: Iterable[CandidatePoint]) -> List[HullPoint]:
    return [
        HullPoint(
            x=candidate.x,
            energy_per_cation_eV=candidate.energy_per_cation_eV,
            label=candidate.label,
            source="candidate",
            candidate_path=str(candidate.candidate_path),
            candidate=candidate.candidate,
            composition_tag=candidate.composition_tag,
        )
        for candidate in candidates
    ]


def _minimum_point_at_each_composition(points: Iterable[HullPoint]) -> List[HullPoint]:
    minima: Dict[float, HullPoint] = {}
    for point in points:
        key = round(point.x, 12)
        current = minima.get(key)
        if current is None or point.energy_per_cation_eV < current.energy_per_cation_eV - ENERGY_TOL:
            minima[key] = point
        elif current is not None and abs(point.energy_per_cation_eV - current.energy_per_cation_eV) <= ENERGY_TOL:
            # Prefer a candidate over a virtual endpoint only for exact numerical ties.
            if current.source != "candidate" and point.source == "candidate":
                minima[key] = point
    return sorted(minima.values(), key=lambda point: point.x)


def _lower_convex_hull(points: List[HullPoint]) -> List[HullPoint]:
    if len(points) < 2:
        raise ValueError("At least two distinct compositions are required for a 1D hull")

    hull: List[HullPoint] = []
    for point in points:
        while len(hull) >= 2:
            left, middle = hull[-2], hull[-1]
            slope_left = (
                (middle.energy_per_cation_eV - left.energy_per_cation_eV)
                / (middle.x - left.x)
            )
            slope_right = (
                (point.energy_per_cation_eV - middle.energy_per_cation_eV)
                / (point.x - middle.x)
            )
            if slope_right <= slope_left + ENERGY_TOL:
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def _hull_energy_and_segment(
    x: float,
    hull_vertices: List[HullPoint],
) -> tuple[float, HullPoint, HullPoint]:
    for vertex in hull_vertices:
        if abs(x - vertex.x) <= X_TOL:
            return vertex.energy_per_cation_eV, vertex, vertex

    for left, right in zip(hull_vertices[:-1], hull_vertices[1:]):
        if left.x < x < right.x:
            fraction = (x - left.x) / (right.x - left.x)
            energy = (
                (1.0 - fraction) * left.energy_per_cation_eV
                + fraction * right.energy_per_cation_eV
            )
            return energy, left, right

    raise ValueError(f"Composition x={x:.12g} lies outside the constructed alloy hull")


def _decomposition_string(x: float, left: HullPoint, right: HullPoint) -> str:
    if left == right:
        return f"1 {left.label}"
    fraction_right = (x - left.x) / (right.x - left.x)
    fraction_left = 1.0 - fraction_right
    return f"{fraction_left:.6g} {left.label} + {fraction_right:.6g} {right.label}"


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_database(root: Path, results: List[Dict[str, Any]]) -> None:
    db_path = root / OUT_DB
    result_by_path = {str(row["candidate_path"]): row for row in results}

    with db_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        database_rows = list(reader)

    extra_columns = [
        "energy_per_cation_eV",
        "hull_energy_1d_eV_per_cation",
        "energy_above_1d_hull_eV_per_cation",
        "on_1d_hull",
        "left_hull_vertex_1d",
        "right_hull_vertex_1d",
        "decomposition_1d",
    ]
    final_fields = fieldnames + [column for column in extra_columns if column not in fieldnames]

    for row in database_rows:
        result = result_by_path.get(str(row.get("candidate_path") or ""))
        if result is None:
            continue
        for column in extra_columns:
            row[column] = result[column]

    temporary = db_path.with_suffix(db_path.suffix + ".tmp")
    _write_csv(temporary, database_rows, final_fields)
    os.replace(temporary, db_path)


def run_alloy_hull(
    raw_cfg: dict[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> Path:
    """Construct and write a restricted 1D alloy convex hull.

    The hull uses direct relaxed total energies normalized per cation. It is
    reference-independent for a fixed stoichiometric alloy line because adding
    any linear reference term in x leaves the convex envelope unchanged.
    """
    cfg = _parse_config(raw_cfg)
    reference_path = root / REF_JSON
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference JSON: {reference_path}")

    ref = _load_json(reference_path)
    dopant, candidates = _read_candidates(
        root,
        host=cfg.host_species,
        anion=cfg.anion_species,
        requested_dopant=cfg.dopant,
    )

    host, host_anion_per_cation = _host_point(ref, cfg.host_species)
    endpoint = _endpoint_point(
        ref,
        dopant=dopant,
        anion=cfg.anion_species,
        host_anion_per_cation=host_anion_per_cation,
        requested_reference=cfg.endpoint_reference,
    )

    all_points = [host, *_candidate_hull_points(candidates)]
    if endpoint is not None:
        all_points.append(endpoint)

    composition_minima = _minimum_point_at_each_composition(all_points)
    hull_vertices = _lower_convex_hull(composition_minima)

    result_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        hull_energy, left, right = _hull_energy_and_segment(candidate.x, hull_vertices)
        energy_above = max(0.0, candidate.energy_per_cation_eV - hull_energy)
        result_rows.append(
            {
                "candidate": candidate.candidate,
                "composition_tag": candidate.composition_tag,
                "candidate_path": str(candidate.candidate_path),
                "formula": candidate.formula,
                "dopant": dopant,
                "x_dopant": candidate.x,
                "energy_total_eV": candidate.energy_total_eV,
                "n_cations": candidate.n_cations,
                "energy_per_cation_eV": candidate.energy_per_cation_eV,
                "hull_energy_1d_eV_per_cation": hull_energy,
                "energy_above_1d_hull_eV_per_cation": energy_above,
                "on_1d_hull": energy_above <= ENERGY_TOL,
                "left_hull_vertex_1d": left.label,
                "right_hull_vertex_1d": right.label,
                "decomposition_1d": _decomposition_string(candidate.x, left, right),
            }
        )

    result_rows.sort(key=lambda row: (float(row["x_dopant"]), float(row["energy_total_eV"])))
    vertices_rows = [
        {
            "x_dopant": point.x,
            "x_dopant_percent": 100.0 * point.x,
            "energy_per_cation_eV": point.energy_per_cation_eV,
            "label": point.label,
            "source": point.source,
            "candidate": point.candidate,
            "composition_tag": point.composition_tag,
            "candidate_path": point.candidate_path,
        }
        for point in hull_vertices
    ]

    _write_csv(
        root / OUT_RESULTS,
        result_rows,
        [
            "candidate",
            "composition_tag",
            "candidate_path",
            "formula",
            "dopant",
            "x_dopant",
            "energy_total_eV",
            "n_cations",
            "energy_per_cation_eV",
            "hull_energy_1d_eV_per_cation",
            "energy_above_1d_hull_eV_per_cation",
            "on_1d_hull",
            "left_hull_vertex_1d",
            "right_hull_vertex_1d",
            "decomposition_1d",
        ],
    )
    _write_csv(
        root / OUT_VERTICES,
        vertices_rows,
        [
            "x_dopant",
            "x_dopant_percent",
            "energy_per_cation_eV",
            "label",
            "source",
            "candidate",
            "composition_tag",
            "candidate_path",
        ],
    )
    _update_database(root, result_rows)

    summary = {
        "stage": "alloy_hull",
        "host_species": cfg.host_species,
        "dopant": dopant,
        "anion_species": cfg.anion_species,
        "endpoint_reference_requested": cfg.endpoint_reference,
        "endpoint_reference_used": endpoint.label if endpoint is not None else None,
        "sampled_x_min": min(point.x for point in composition_minima),
        "sampled_x_max": max(point.x for point in composition_minima),
        "n_candidates": len(candidates),
        "n_composition_minima": len(composition_minima),
        "n_hull_vertices": len(hull_vertices),
        "hull_vertices": vertices_rows,
        "outputs": {
            "results": OUT_RESULTS,
            "vertices": OUT_VERTICES,
            "database": OUT_DB,
        },
    }
    (root / OUT_SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("1D alloy hull: dopant=%s, candidates=%d, vertices=%d", dopant, len(candidates), len(hull_vertices))
    log.info("DONE alloy-hull: wrote %s and %s", OUT_RESULTS, OUT_VERTICES)
    return root / OUT_RESULTS


def run_alloy_hull_from_toml(config_path: Path) -> Path:
    raw_cfg = _load_raw_toml(config_path)
    return run_alloy_hull(raw_cfg, config_path.resolve().parent, config_path=config_path)

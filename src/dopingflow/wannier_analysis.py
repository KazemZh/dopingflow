from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pymatgen.core import Structure


_SPREAD_RE = re.compile(
    r"^\s*WF centre and spread\s+(\d+)\s+\(.*\)\s+([-+0-9.eE]+)\s*$"
)


def parse_wannier_spreads(path: Path) -> dict[int, float]:
    """Return final Wannier spreads keyed by zero-based WF index.

    Wannier90 prints the same WF repeatedly during localization.  Assignment in
    this loop deliberately overwrites earlier iterations, leaving the final
    occurrence from the converged/final summary block.
    """
    spreads: dict[int, float] = {}
    if not path.is_file():
        return spreads
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _SPREAD_RE.match(line)
            if not match:
                continue
            try:
                spreads[int(match.group(1)) - 1] = float(match.group(2))
            except ValueError:
                continue
    return spreads


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def analyze_wannier_centres(
    structure: Structure,
    centres: Sequence[dict[str, Any]],
    spreads: dict[int, float] | None = None,
    *,
    atom_center_cutoff_angstrom: float = 0.45,
    bond_center_cutoff_angstrom: float = 1.35,
    bond_distance_balance_angstrom: float = 0.30,
    delocalized_spread_threshold_ang2: float = 3.0,
    electrons_per_wf: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Classify Wannier centers geometrically using periodic minimum distances.

    The classification is intentionally a descriptor, not an oxidation-state
    assignment.  The thresholds are exposed to the user because their useful
    values are chemistry and localization dependent.
    """
    if len(structure) == 0:
        raise ValueError("Wannier-center analysis requires a non-empty structure")
    if atom_center_cutoff_angstrom <= 0:
        raise ValueError("atom_center_cutoff_angstrom must be positive")
    if bond_center_cutoff_angstrom <= 0:
        raise ValueError("bond_center_cutoff_angstrom must be positive")
    if atom_center_cutoff_angstrom > bond_center_cutoff_angstrom:
        raise ValueError("atom_center_cutoff_angstrom cannot exceed bond_center_cutoff_angstrom")
    if bond_distance_balance_angstrom < 0:
        raise ValueError("bond_distance_balance_angstrom must be >= 0")
    if delocalized_spread_threshold_ang2 <= 0:
        raise ValueError("delocalized_spread_threshold_ang2 must be positive")
    if electrons_per_wf is not None and electrons_per_wf <= 0:
        raise ValueError("electrons_per_wf must be positive when supplied")

    spreads = dict(spreads or {})
    enriched: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    geometric_counts: Counter[str] = Counter()

    site_summary: list[dict[str, Any]] = [
        {
            "site_index": index,
            "element": site.specie.symbol,
            "nearest_wf_count": 0,
            "nearest_wf_electron_equivalent": 0.0 if electrons_per_wf is not None else None,
            "atom_centered_wf_count": 0,
            "bond_centered_wf_participation": 0,
            "ambiguous_wf_participation": 0,
            "delocalized_wf_nearest_count": 0,
        }
        for index, site in enumerate(structure)
    ]

    for raw in centres:
        record = dict(raw)
        center_index = int(record.get("center_index", len(enriched)))
        cart = record.get("cartesian_angstrom")
        if not isinstance(cart, (list, tuple)) or len(cart) != 3:
            raise ValueError(f"Wannier center {center_index} has invalid Cartesian coordinates")
        cart_values = [float(value) for value in cart]
        frac = structure.lattice.get_fractional_coords(cart_values)
        distances = structure.lattice.get_all_distances([frac], structure.frac_coords)[0]
        order = sorted(range(len(structure)), key=lambda idx: float(distances[idx]))
        nearest = int(order[0])
        second = int(order[1]) if len(order) > 1 else None
        d1 = float(distances[nearest])
        d2 = float(distances[second]) if second is not None else None
        within = [
            int(idx)
            for idx in order
            if float(distances[idx]) <= bond_center_cutoff_angstrom
        ]

        if d1 <= atom_center_cutoff_angstrom:
            geometric = "atom-centered"
            associated = [nearest]
        elif (
            second is not None
            and d2 is not None
            and d2 <= bond_center_cutoff_angstrom
            and abs(d2 - d1) <= bond_distance_balance_angstrom
        ):
            geometric = "bond-centered"
            associated = [nearest, second]
        else:
            geometric = "multicenter/ambiguous"
            associated = within or [nearest]

        spread = spreads.get(center_index)
        delocalized = bool(
            spread is not None and spread >= delocalized_spread_threshold_ang2
        )
        classification = "anomalous/delocalized" if delocalized else geometric
        classification_counts[classification] += 1
        geometric_counts[geometric] += 1

        record.update(
            {
                "center_index": center_index,
                "wannier_number": center_index + 1,
                "cartesian_angstrom": cart_values,
                "spread_ang2": float(spread) if spread is not None else None,
                "classification": classification,
                "geometric_classification": geometric,
                "is_delocalized_spread_outlier": delocalized,
                "nearest_site_index": nearest,
                "nearest_element": structure[nearest].specie.symbol,
                "nearest_distance_angstrom": d1,
                "second_nearest_site_index": second,
                "second_nearest_element": (
                    structure[second].specie.symbol if second is not None else None
                ),
                "second_nearest_distance_angstrom": d2,
                "associated_site_indices": associated,
                "associated_elements": [structure[idx].specie.symbol for idx in associated],
                "n_atoms_within_bond_cutoff": len(within),
                "electron_equivalent": electrons_per_wf,
            }
        )
        enriched.append(record)

        site_summary[nearest]["nearest_wf_count"] += 1
        if electrons_per_wf is not None:
            site_summary[nearest]["nearest_wf_electron_equivalent"] += electrons_per_wf
        if geometric == "atom-centered":
            site_summary[nearest]["atom_centered_wf_count"] += 1
        elif geometric == "bond-centered":
            for idx in associated:
                site_summary[idx]["bond_centered_wf_participation"] += 1
        else:
            for idx in associated:
                site_summary[idx]["ambiguous_wf_participation"] += 1
        if delocalized:
            site_summary[nearest]["delocalized_wf_nearest_count"] += 1

    spread_values = [
        float(record["spread_ang2"])
        for record in enriched
        if record.get("spread_ang2") is not None
    ]
    max_record = max(
        (record for record in enriched if record.get("spread_ang2") is not None),
        key=lambda record: float(record["spread_ang2"]),
        default=None,
    )
    summary = {
        "n_wannier_centres": len(enriched),
        "n_spread_records": len(spread_values),
        "median_spread_ang2": _median(spread_values),
        "max_spread_ang2": (
            float(max_record["spread_ang2"]) if max_record is not None else None
        ),
        "max_spread_center_index": (
            int(max_record["center_index"]) if max_record is not None else None
        ),
        "classification_counts": dict(sorted(classification_counts.items())),
        "geometric_classification_counts": dict(sorted(geometric_counts.items())),
        "n_delocalized_spread_outliers": int(
            classification_counts.get("anomalous/delocalized", 0)
        ),
        "electrons_per_wf": electrons_per_wf,
        "represented_electrons": (
            float(electrons_per_wf * len(enriched))
            if electrons_per_wf is not None
            else None
        ),
        "thresholds": {
            "atom_center_cutoff_angstrom": atom_center_cutoff_angstrom,
            "bond_center_cutoff_angstrom": bond_center_cutoff_angstrom,
            "bond_distance_balance_angstrom": bond_distance_balance_angstrom,
            "delocalized_spread_threshold_ang2": delocalized_spread_threshold_ang2,
        },
        "interpretation": (
            "Center/site associations are periodic geometric descriptors. Nearest-center electron "
            "equivalents are not Bader charges, atomic populations, or formal oxidation states."
        ),
    }
    return enriched, site_summary, summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def write_wannier_analysis_files(
    workdir: Path,
    centres: Sequence[dict[str, Any]],
    site_summary: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    centres_csv = workdir / "wannier_centres_analysis.csv"
    sites_csv = workdir / "wannier_site_summary.csv"
    analysis_json = workdir / "wannier_analysis.json"
    _write_csv(centres_csv, centres)
    _write_csv(sites_csv, site_summary)
    analysis_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "centres": list(centres),
                "site_summary": list(site_summary),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "centres_analysis_csv": str(centres_csv),
        "site_summary_csv": str(sites_csv),
        "analysis_json": str(analysis_json),
    }

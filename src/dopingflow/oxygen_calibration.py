"""Backend-specific oxygen-reference calibration from experimental oxide enthalpies.

The vacancy workflow needs an oxygen reference on the same energy scale as the
selected ML backend.  This module derives that reference from *real binary
oxides already calculated by refs-build* and experimental 298 K formation
enthalpies.  No universal O2 correction is hard-coded.

Two scopes are supported:

``global``
    Use every eligible ordinary binary oxide in the reference inventory.

``chemistry-specific``
    Use only eligible oxides whose cation is present in the vacancy parent
    chemistry (host plus actually present dopants).

The fitted quantity is an effective per-oxygen reference
``mu_O_reference_eV``.  For each oxide M_x O_y,

    mu_O,i = (E_MxOy - x E_M - DeltaHf_exp) / y

and the reported reference is the arithmetic mean of the eligible per-O
values.  Individual values and residuals are retained so a poor or
chemistry-dependent fit is visible rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from pymatgen.analysis.structure_analyzer import oxide_type as classify_oxide_type
from pymatgen.core import Composition, Structure

from dopingflow.corrections import (
    ExperimentalRecord,
    load_custom_experimental_dataset,
    load_kingsbury_dataset,
    merge_experimental_datasets,
)

CALIBRATION_SCOPES = {"global", "chemistry-specific"}
EXPERIMENTAL_SOURCES = {"kingsbury", "kingsbury+custom", "custom"}


@dataclass(frozen=True)
class OxygenCalibrationRequest:
    reference_file: Path
    scope: str
    target_elements: tuple[str, ...]
    experimental_source: str = "kingsbury"
    experimental_data: Path | None = None
    dataset_cache_dir: Path | None = None
    min_references: int = 2
    include_host_oxide: bool = True


def _resolve_reference_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _normal_binary_oxide(structure: Structure) -> bool:
    composition = structure.composition
    symbols = {element.symbol for element in composition.elements}
    if "O" not in symbols or len(symbols - {"O"}) != 1:
        return False
    try:
        kind = str(classify_oxide_type(structure)).strip().lower()
    except Exception:
        return False
    return kind in {"oxide", "normal"}


def _reduced_binary_formula(composition: Composition) -> tuple[str, str, float, float] | None:
    reduced = composition.reduced_composition
    amounts = reduced.get_el_amt_dict()
    non_oxygen = [symbol for symbol in amounts if symbol != "O"]
    if len(non_oxygen) != 1 or "O" not in amounts:
        return None
    cation = non_oxygen[0]
    x = float(amounts[cation])
    y = float(amounts["O"])
    if x <= 0 or y <= 0:
        return None
    return reduced.reduced_formula, cation, x, y


def _energy_per_reduced_formula_from_structure(
    structure: Structure, total_energy_eV: float
) -> float:
    _, factor = structure.composition.get_reduced_composition_and_factor()
    factor = float(factor)
    if factor <= 0:
        raise ValueError("Invalid reduced-composition factor")
    return float(total_energy_eV) / factor


def _experimental_records(
    source: str,
    *,
    custom_path: Path | None,
    dataset_cache_dir: Path | None,
) -> list[ExperimentalRecord]:
    source = str(source).strip().lower()
    if source not in EXPERIMENTAL_SOURCES:
        raise ValueError(
            "oxygen calibration experimental source must be one of: "
            "kingsbury, kingsbury+custom, custom"
        )
    curated: list[ExperimentalRecord] = []
    custom: list[ExperimentalRecord] = []
    if source in {"kingsbury", "kingsbury+custom"}:
        curated = load_kingsbury_dataset(dataset_cache_dir)
    if source in {"custom", "kingsbury+custom"}:
        if custom_path is None:
            raise ValueError(
                f"oxygen calibration experimental_source={source!r} requires "
                "oxygen_calibration_experimental_data"
            )
        custom = load_custom_experimental_dataset(custom_path)
    if source == "kingsbury+custom":
        return merge_experimental_datasets(curated, custom)
    return custom if source == "custom" else curated


def _select_experimental_record(
    records: Sequence[ExperimentalRecord], reduced_formula: str
) -> tuple[ExperimentalRecord | None, int]:
    matches = [record for record in records if record.reduced_formula == reduced_formula]
    if not matches:
        return None, 0

    def rank(record: ExperimentalRecord) -> tuple[float, str, str]:
        uncertainty = record.uncertainty_eV_per_formula
        return (
            float(uncertainty) if uncertainty is not None else math.inf,
            str(record.source),
            str(record.reference_id),
        )

    return min(matches, key=rank), len(matches)


def _reference_identity(data: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(data.get("backend", "")).strip(),
        str(data.get("model", "")).strip(),
        str(data.get("task", "")).strip(),
    )


def _validate_backend_identity(
    data: dict[str, Any], *, backend: str, model: str, task: str
) -> None:
    stored = _reference_identity(data)
    expected = (str(backend).strip(), str(model).strip(), str(task).strip())
    if stored != expected:
        raise ValueError(
            "Oxygen-calibration reference energies use a different calculator: "
            f"reference={stored}, vacancy={expected}. Rerun refs-build with the "
            "vacancy backend/model/task before calibrating oxygen."
        )


def _candidate_from_entry(
    *,
    name: str,
    entry: dict[str, Any],
    root: Path,
    source_kind: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if source_kind == "reference" and str(entry.get("type", "")).lower() != "oxide":
        return None, "not an oxide reference"

    relaxed_path = _resolve_reference_path(
        root,
        entry.get("relaxed_poscar") or entry.get("relaxed_unit_poscar"),
    )
    if relaxed_path is None or not relaxed_path.exists():
        return None, "relaxed oxide structure is unavailable"
    try:
        structure = Structure.from_file(str(relaxed_path))
    except Exception as exc:
        return None, f"cannot read relaxed oxide structure: {exc}"
    if not _normal_binary_oxide(structure):
        return None, "not an ordinary binary oxide"

    formula_info = _reduced_binary_formula(structure.composition)
    if formula_info is None:
        return None, "not a binary M-O composition"
    reduced_formula, cation, x, y = formula_info

    if source_kind == "reference":
        energy = entry.get("E_per_formula_unit_eV")
        if energy is None:
            total = entry.get("E_total_eV")
            if total is None:
                return None, "oxide energy is unavailable"
            energy = _energy_per_reduced_formula_from_structure(structure, float(total))
    else:
        total = entry.get("E_unit_total_eV")
        if total is None:
            return None, "host unit-cell energy is unavailable"
        energy = _energy_per_reduced_formula_from_structure(structure, float(total))

    return (
        {
            "name": name,
            "reduced_formula": reduced_formula,
            "cation": cation,
            "x_cation": x,
            "y_oxygen": y,
            "oxide_energy_eV_per_formula": float(energy),
            "relaxed_structure": str(relaxed_path),
            "source_kind": source_kind,
        },
        None,
    )


def _metal_energy(reference_data: dict[str, Any], element: str) -> float | None:
    entry = (reference_data.get("references", {}) or {}).get(element)
    if not isinstance(entry, dict) or str(entry.get("type", "")).lower() != "metal":
        return None
    if entry.get("E_per_formula_unit_eV") is not None:
        return float(entry["E_per_formula_unit_eV"])
    if entry.get("E_per_atom_eV") is not None:
        return float(entry["E_per_atom_eV"])
    return None


def _collect_candidates(
    data: dict[str, Any], *, root: Path, include_host: bool
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen_formulas: set[str] = set()

    for name, raw_entry in sorted((data.get("references", {}) or {}).items()):
        if not isinstance(raw_entry, dict) or str(raw_entry.get("type", "")).lower() != "oxide":
            continue
        candidate, reason = _candidate_from_entry(
            name=str(name), entry=raw_entry, root=root, source_kind="reference"
        )
        if candidate is None:
            excluded.append({"name": str(name), "reason": str(reason)})
            continue
        if candidate["reduced_formula"] in seen_formulas:
            excluded.append({"name": str(name), "reason": "duplicate reduced oxide formula"})
            continue
        seen_formulas.add(candidate["reduced_formula"])
        candidates.append(candidate)

    host = data.get("host", {}) or {}
    if include_host and isinstance(host, dict) and host.get("name"):
        name = str(host.get("name"))
        candidate, reason = _candidate_from_entry(
            name=name, entry=host, root=root, source_kind="host"
        )
        if candidate is None:
            excluded.append({"name": name, "reason": str(reason)})
        elif candidate["reduced_formula"] not in seen_formulas:
            candidates.append(candidate)
            seen_formulas.add(candidate["reduced_formula"])

    return candidates, excluded


def fit_oxygen_reference(
    request: OxygenCalibrationRequest,
    *,
    backend: str,
    model: str,
    task: str,
) -> dict[str, Any]:
    """Fit an effective per-O reference from refs-build and experiment."""

    scope = str(request.scope).strip().lower().replace("_", "-")
    if scope not in CALIBRATION_SCOPES:
        raise ValueError(
            "oxygen calibration scope must be 'global' or 'chemistry-specific'"
        )
    if request.min_references < 1:
        raise ValueError("oxygen_calibration_min_references must be >= 1")
    if not request.reference_file.exists():
        raise FileNotFoundError(
            f"Oxygen-calibration reference file does not exist: {request.reference_file}"
        )

    data = json.loads(request.reference_file.read_text(encoding="utf-8"))
    _validate_backend_identity(data, backend=backend, model=model, task=task)
    root = request.reference_file.resolve().parents[1]
    records = _experimental_records(
        request.experimental_source,
        custom_path=request.experimental_data,
        dataset_cache_dir=request.dataset_cache_dir,
    )
    candidates, excluded = _collect_candidates(
        data, root=root, include_host=request.include_host_oxide
    )

    target = {str(symbol) for symbol in request.target_elements if str(symbol).strip()}
    if scope == "chemistry-specific" and not target:
        raise ValueError(
            "chemistry-specific oxygen calibration requires at least one target cation"
        )

    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        if scope == "chemistry-specific" and candidate["cation"] not in target:
            excluded.append(
                {
                    "name": candidate["name"],
                    "reason": "cation outside target chemistry",
                }
            )
            continue
        metal_energy = _metal_energy(data, candidate["cation"])
        if metal_energy is None:
            excluded.append(
                {
                    "name": candidate["name"],
                    "reason": f"missing bulk-metal reference {candidate['cation']}",
                }
            )
            continue
        experimental, n_matches = _select_experimental_record(
            records, candidate["reduced_formula"]
        )
        if experimental is None:
            excluded.append(
                {
                    "name": candidate["name"],
                    "reason": "no experimental 298 K formation enthalpy",
                }
            )
            continue

        x = float(candidate["x_cation"])
        y = float(candidate["y_oxygen"])
        delta_h = float(experimental.formation_enthalpy_eV_per_formula)
        mu_o = (
            float(candidate["oxide_energy_eV_per_formula"])
            - x * float(metal_energy)
            - delta_h
        ) / y
        accepted.append(
            {
                **candidate,
                "metal_energy_eV_per_atom": float(metal_energy),
                "experimental_delta_hf_298_eV_per_formula": delta_h,
                "experimental_uncertainty_eV_per_formula": (
                    float(experimental.uncertainty_eV_per_formula)
                    if experimental.uncertainty_eV_per_formula is not None
                    else None
                ),
                "experimental_phase": experimental.phase,
                "experimental_source": experimental.source,
                "experimental_dataset": experimental.dataset,
                "experimental_reference_id": experimental.reference_id,
                "experimental_doi": experimental.doi,
                "experimental_match_count": int(n_matches),
                "mu_O_inferred_eV_per_O": float(mu_o),
            }
        )

    if len(accepted) < request.min_references:
        names = ", ".join(item["name"] for item in accepted) or "none"
        raise ValueError(
            f"{scope} oxygen calibration found {len(accepted)} eligible oxide(s) "
            f"({names}), but oxygen_calibration_min_references="
            f"{request.min_references}. Add real oxide/metal references with "
            "experimental data, lower the minimum deliberately, or choose a raw/"
            "explicit oxygen-reference mode."
        )

    mu_values = np.array(
        [float(item["mu_O_inferred_eV_per_O"]) for item in accepted], dtype=float
    )
    mu_fit = float(np.mean(mu_values))
    deviations = mu_values - mu_fit
    formation_residuals: list[float] = []
    for item, deviation in zip(accepted, deviations):
        residual = -float(item["y_oxygen"]) * float(deviation)
        item["mu_O_deviation_from_fit_eV_per_O"] = float(deviation)
        item["formation_enthalpy_residual_eV_per_formula"] = residual
        formation_residuals.append(residual)

    formation_array = np.asarray(formation_residuals, dtype=float)
    return {
        "schema_version": 1,
        "method": "mean_per_oxygen_from_binary_oxide_formation_enthalpies",
        "scope": scope,
        "target_elements": sorted(target),
        "backend": str(backend),
        "model": str(model),
        "task": str(task),
        "reference_file": str(request.reference_file),
        "experimental_source_mode": request.experimental_source,
        "experimental_data": (
            str(request.experimental_data) if request.experimental_data is not None else None
        ),
        "dataset_cache_dir": (
            str(request.dataset_cache_dir) if request.dataset_cache_dir is not None else None
        ),
        "include_host_oxide": bool(request.include_host_oxide),
        "minimum_references": int(request.min_references),
        "n_references": len(accepted),
        "mu_O_reference_eV": mu_fit,
        "mean_absolute_mu_spread_eV_per_O": float(np.mean(np.abs(deviations))),
        "rmse_mu_spread_eV_per_O": float(np.sqrt(np.mean(deviations**2))),
        "sample_std_mu_eV_per_O": (
            float(np.std(mu_values, ddof=1)) if len(mu_values) > 1 else 0.0
        ),
        "formation_enthalpy_mae_eV_per_formula": float(
            np.mean(np.abs(formation_array))
        ),
        "formation_enthalpy_rmse_eV_per_formula": float(
            np.sqrt(np.mean(formation_array**2))
        ),
        "references_used": accepted,
        "excluded_references": excluded,
    }


def chemistry_elements_from_minimum(row: dict[str, Any]) -> tuple[str, ...]:
    """Return host plus actually present dopant cations for one vacancy composition."""

    elements: set[str] = set()
    host = str(row.get("host_species", "")).strip()
    if host and host != "O":
        elements.add(host)
    counts = row.get("dopant_counts_json") or {}
    if isinstance(counts, str):
        try:
            counts = json.loads(counts)
        except json.JSONDecodeError:
            counts = {}
    if isinstance(counts, dict):
        for element, count in counts.items():
            try:
                amount = int(count)
            except (TypeError, ValueError):
                continue
            symbol = str(element).strip()
            if amount > 0 and symbol and symbol != "O":
                elements.add(symbol)
    return tuple(sorted(elements))


def write_oxygen_calibration_report(path: Path, calibrations: Iterable[dict[str, Any]]) -> Path:
    payload = {
        "schema_version": 1,
        "calibrations": list(calibrations),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "CALIBRATION_SCOPES",
    "EXPERIMENTAL_SOURCES",
    "OxygenCalibrationRequest",
    "chemistry_elements_from_minimum",
    "fit_oxygen_reference",
    "write_oxygen_calibration_report",
]

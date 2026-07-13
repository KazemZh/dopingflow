"""Global relative-energy post-processing for dopingflow result databases."""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SOURCE_TO_RELATIVE = {
    "E_form_eV_per_cation__": "E_form_rel_eV_per_cation__",
    "E_mix_eV_per_cation__": "E_mix_rel_eV_per_cation__",
}


def relative_energy_enabled(raw_cfg: dict[str, Any]) -> bool:
    """Return the flat ``[formation].relative_enabled`` setting."""
    formation = raw_cfg.get("formation", {}) or {}
    return bool(formation.get("relative_enabled", False))


def _as_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        return None if text == "" else float(text)
    except (TypeError, ValueError):
        return None


def _endpoint_x_from_config(raw_cfg: dict[str, Any]) -> float | None:
    """Read optional [formation].endpoint_x; None represents automatic mode."""
    formation = raw_cfg.get("formation", {}) or {}
    raw_value = formation.get("endpoint_x", "auto")

    if raw_value is None or str(raw_value).strip().lower() in {"", "auto"}:
        return None

    endpoint_x = float(raw_value)
    if not 0.0 < endpoint_x <= 1.0:
        raise ValueError("[formation].endpoint_x must be in (0, 1] or 'auto'")
    return endpoint_x


def _resolve_endpoint_x(rows: list[dict[str, str]], configured_x: float | None) -> float:
    x_values = sorted(
        {
            float(x)
            for row in rows
            if (x := _as_float(row.get("x_dopant"))) is not None and x > 0.0
        }
    )
    if not x_values:
        raise ValueError("Cannot calculate relative energies: no positive x_dopant values found.")

    if configured_x is None:
        return x_values[-1]

    if not any(abs(x - configured_x) <= 1e-8 for x in x_values):
        available = ", ".join(f"{x:.8g}" for x in x_values)
        raise ValueError(
            f"No candidate exists at [formation].endpoint_x={configured_x:.8g}. "
            f"Available actual dopant fractions: {available}"
        )
    return configured_x


def populate_relative_energy_columns(
    csv_path: Path,
    raw_cfg: dict[str, Any],
) -> Path:
    """Populate missing legacy relative-energy columns in one CSV.

    Formation now writes oxide-endmember tie-line values directly. Existing
    relative columns are therefore authoritative and are never overwritten by
    this compatibility postprocessor. For older databases that contain only
    absolute reference-specific columns, the fallback calculation is:

        E_rel(x) = E(x) - (x / X) E_min(X)

    ``X`` is ``[formation].endpoint_x`` when supplied, otherwise the largest
    actual dopant fraction in the completed result database.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Result database not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        log.warning("Relative energies skipped: %s has no rows", csv_path)
        return csv_path
    if "x_dopant" not in fieldnames:
        raise KeyError(f"{csv_path} is missing required column 'x_dopant'")

    endpoint_x = _resolve_endpoint_x(rows, _endpoint_x_from_config(raw_cfg))
    new_columns: list[str] = []

    for source_prefix, relative_prefix in _SOURCE_TO_RELATIVE.items():
        source_columns = sorted(
            column for column in fieldnames if column.startswith(source_prefix)
        )

        for source_column in source_columns:
            label = source_column.removeprefix(source_prefix)
            relative_column = f"{relative_prefix}{label}"
            if relative_column in fieldnames:
                log.info("Preserving formation-stage relative column %s", relative_column)
                continue
            endpoint_energies = [
                energy
                for row in rows
                if (x := _as_float(row.get("x_dopant"))) is not None
                and abs(x - endpoint_x) <= 1e-8
                and (energy := _as_float(row.get(source_column))) is not None
            ]
            if not endpoint_energies:
                log.warning(
                    "Relative column %s skipped: no valid endpoint energy at X=%.8g",
                    relative_column,
                    endpoint_x,
                )
                continue

            endpoint_energy = min(endpoint_energies)
            for row in rows:
                x = _as_float(row.get("x_dopant"))
                energy = _as_float(row.get(source_column))
                row[relative_column] = (
                    ""
                    if x is None or energy is None
                    else f"{energy - (x / endpoint_x) * endpoint_energy:.8f}"
                )

            if relative_column not in fieldnames and relative_column not in new_columns:
                new_columns.append(relative_column)

            log.info(
                "Relative %s: X=%.8g; endpoint minimum for %s = %.8f eV/cation",
                "formation" if source_prefix.startswith("E_form") else "mixing",
                endpoint_x,
                label,
                endpoint_energy,
            )

    if not new_columns and not any(
        column.startswith("E_form_rel_eV_per_cation__")
        or column.startswith("E_mix_rel_eV_per_cation__")
        for column in fieldnames
    ):
        log.warning("Relative energies skipped: no reference-specific per-cation energy columns found.")
        return csv_path

    final_fieldnames = fieldnames + new_columns
    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, csv_path)

    log.info("Populated relative-energy columns in %s using X=%.8g", csv_path, endpoint_x)
    return csv_path

# src/dopingflow/collect.py
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

OUT_CSV = "results_database.csv"
META_COMP = "metadata.json"
SELECTED_TXT = "selected_candidates.txt"
RANK_RELAX_FILTERED = "ranking_relax_filtered.csv"
RANK_SCAN = "ranking_scan.csv"
BANDGAP_SUMMARY = "bandgap_alignn_summary.csv"
FORMATION_CSV = "formation_energies.csv"
RELAX_META = Path("02_relax") / "meta.json"
FORMATION_META = Path("04_formation") / "meta.json"

_DYNAMIC_METRICS = (
    "E_form_eV_total",
    "E_form_eV_per_atom",
    "E_form_eV_per_cation",
    "E_form_eV_per_dopant",
    "E_mix_eV_total",
    "E_mix_eV_per_atom",
    "E_mix_eV_per_cation",
    "E_mix_eV_per_dopant",
    "E_form_rel_eV_per_cation",
    "E_mix_rel_eV_per_cation",
    "n_O2_out",
    "mixing_reaction_reference",
)
_RELATIVE_METRICS = (
    "E_form_rel_eV_per_cation",
    "E_mix_rel_eV_per_cation",
)
_RELATIVE_SOURCES = {
    "E_form_rel_eV_per_cation": "E_form_eV_per_cation",
    "E_mix_rel_eV_per_cation": "E_mix_eV_per_cation",
}


@dataclass(frozen=True)
class DBConfig:
    outdir: Path
    skip_if_done: bool
    relative_enabled: bool
    relative_endpoint_x: float | None


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read JSON %s: %s", path, exc)
        return None


def safe_get(data: Optional[dict], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _to_int(value: Any) -> Optional[int]:
    try:
        text = str(value).strip()
        return None if text == "" else int(float(text))
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        text = str(value).strip()
        return None if text == "" else float(text)
    except Exception:
        return None


def _parse_relative_endpoint(raw: dict[str, Any]) -> float | None:
    formation = raw.get("formation", {}) or {}
    relative = formation.get("relative", {}) or {}
    endpoint_raw = relative.get("endpoint_x", "auto")

    if endpoint_raw is None or str(endpoint_raw).strip().lower() in {"", "auto"}:
        return None

    endpoint_x = float(endpoint_raw)
    if not 0.0 < endpoint_x <= 1.0:
        raise ValueError(
            "[formation.relative].endpoint_x must be in (0, 1] or 'auto'"
        )
    return endpoint_x


def _parse_db_config(raw: dict[str, Any], root: Path) -> DBConfig:
    structure = raw.get("structure", {}) or {}
    database = raw.get("database", {}) or {}
    formation = raw.get("formation", {}) or {}
    relative = formation.get("relative", {}) or {}

    return DBConfig(
        outdir=(root / str(structure.get("outdir", "random_structures"))).resolve(),
        skip_if_done=bool(database.get("skip_if_done", True)),
        relative_enabled=bool(relative.get("enabled", False)),
        relative_endpoint_x=_parse_relative_endpoint(raw),
    )


def read_selected_txt(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def read_filtered_table(path: Path) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return output

    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate = (row.get("candidate") or "").strip()
            if candidate:
                output[candidate] = {
                    "rank_relax_filtered": _to_int(row.get("rank_filtered")),
                    "E_relaxed_eV_filtered": _to_float(row.get("energy_relaxed_eV")),
                    "delta_e_eV": _to_float(row.get("delta_e_eV")),
                    "filter_mode": (row.get("filter_mode") or "").strip(),
                }
    return output


def read_scan_ranking(path: Path) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return output

    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate = (
                row.get("candidate")
                or row.get("candidate_id")
                or row.get("name")
                or row.get("folder")
                or ""
            ).strip()
            if not candidate:
                continue

            output[candidate] = {
                "rank_scan": _to_int(row.get("rank") or row.get("rank_scan")),
                "E_scan_eV": _to_float(
                    row.get("E_eV")
                    or row.get("energy_eV")
                    or row.get("E_scan_eV")
                    or row.get("energy_sp_eV")
                    or row.get("energy")
                ),
            }
    return output


def read_bandgap_summary(path: Path) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return output

    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate = (
                row.get("candidate") or row.get("candidate_id") or row.get("name") or ""
            ).strip()
            if not candidate:
                continue

            bandgap = None
            for key in (
                "bandgap_eV_ALIGNN_MBJ",
                "bandgap_eV",
                "bandgap",
                "pred_bandgap",
                "pred_bandgap_eV",
            ):
                bandgap = _to_float(row.get(key))
                if bandgap is not None:
                    break
            output[candidate] = {"bandgap_eV": bandgap}
    return output


def _normalise_reference_result(result: Dict[str, Any]) -> Dict[str, Any]:
    mixing = result.get("mixing", {}) or {}
    relative = result.get("relative", {}) or {}
    return {
        "E_form_eV_total": result.get("E_form_eV_total"),
        "E_form_eV_per_atom": result.get("E_form_eV_per_atom"),
        "E_form_eV_per_cation": result.get("E_form_eV_per_cation"),
        "E_form_eV_per_dopant": result.get("E_form_eV_per_dopant"),
        "E_mix_eV_total": mixing.get("E_mix_eV_total"),
        "E_mix_eV_per_atom": mixing.get("E_mix_eV_per_atom"),
        "E_mix_eV_per_cation": mixing.get("E_mix_eV_per_cation"),
        "E_mix_eV_per_dopant": mixing.get("E_mix_eV_per_dopant"),
        "E_form_rel_eV_per_cation": relative.get("E_form_rel_eV_per_cation"),
        "E_mix_rel_eV_per_cation": relative.get("E_mix_rel_eV_per_cation"),
        "n_O2_out": mixing.get("n_O2_out"),
        "mixing_reaction_reference": mixing.get("reaction_reference", ""),
    }


def read_formation_meta(path: Path) -> Dict[str, Any]:
    data = read_json(path) or {}
    results: Dict[str, Dict[str, Any]] = {}

    stored = data.get("reference_results", {})
    if isinstance(stored, dict):
        for label, result in stored.items():
            if isinstance(result, dict):
                results[str(label)] = _normalise_reference_result(result)

    if not results and data:
        label = str(data.get("primary_reference_label") or "legacy")
        results[label] = _normalise_reference_result(data)

    return {
        "reference_mode": data.get("reference_mode", ""),
        "reference_results": results,
        "dopant_counts_dict": data.get("dopant_counts"),
        "n_atoms_supercell": data.get("n_atoms_supercell"),
        "x_dopant": data.get("x_dopant"),
    }


def _parse_dynamic_reference_columns(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for key, value in row.items():
        if "__" not in key:
            continue

        metric, label = key.split("__", 1)
        if metric not in _DYNAMIC_METRICS:
            continue

        result = results.setdefault(label, {})
        result[metric] = (
            str(value or "")
            if metric == "mixing_reaction_reference"
            else _to_float(value)
        )
    return results


def read_formation_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return output

    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate = (row.get("candidate") or "").strip()
            if not candidate:
                continue

            results = _parse_dynamic_reference_columns(row)
            if not results and row.get("E_form_eV_total") not in {None, ""}:
                label = str(row.get("primary_reference_label") or "legacy")
                results[label] = {
                    "E_form_eV_total": _to_float(row.get("E_form_eV_total")),
                    "E_form_eV_per_atom": _to_float(row.get("E_form_eV_per_atom")),
                    "E_form_eV_per_cation": _to_float(row.get("E_form_eV_per_cation")),
                    "E_form_eV_per_dopant": _to_float(row.get("E_form_per_dopant")),
                    "E_mix_eV_total": _to_float(row.get("E_mix_eV_total")),
                    "E_mix_eV_per_atom": _to_float(row.get("E_mix_eV_per_atom")),
                    "E_mix_eV_per_cation": _to_float(row.get("E_mix_eV_per_cation")),
                    "E_mix_eV_per_dopant": _to_float(row.get("E_mix_eV_per_dopant")),
                    "E_form_rel_eV_per_cation": _to_float(row.get("E_form_rel_eV_per_cation")),
                    "E_mix_rel_eV_per_cation": _to_float(row.get("E_mix_rel_eV_per_cation")),
                    "n_O2_out": _to_float(row.get("n_O2_out")),
                    "mixing_reaction_reference": str(
                        row.get("mixing_reaction_reference") or ""
                    ),
                }

            output[candidate] = {
                "reference_mode": (row.get("reference_mode") or "").strip(),
                "reference_results": results,
                "n_dopant_atoms": _to_int(row.get("n_dopant_atoms")),
                "dopant_counts": (row.get("dopant_counts") or "").strip(),
                "x_dopant": _to_float(row.get("x_dopant")),
            }
    return output


def _dynamic_column_name(metric: str, label: str) -> str:
    return f"{metric}__{label}"


def _populate_dynamic_columns(
    row: Dict[str, Any],
    results: Dict[str, Dict[str, Any]],
) -> None:
    for label, values in results.items():
        for metric in _DYNAMIC_METRICS:
            value = values.get(metric)
            # Relative values are calculated globally after all compositions
            # have been collected. Do not create empty columns beforehand.
            if metric in _RELATIVE_METRICS and value is None:
                continue
            row[_dynamic_column_name(metric, label)] = value


def _actual_endpoint_x(rows: List[Dict[str, Any]], requested_x: float | None) -> float:
    x_values = sorted(
        {
            float(row["x_dopant"])
            for row in rows
            if _to_float(row.get("x_dopant")) is not None
            and float(row["x_dopant"]) > 0.0
        }
    )
    if not x_values:
        raise ValueError("Relative energy requested, but no positive x_dopant values were collected.")

    if requested_x is None:
        return x_values[-1]

    matches = [x for x in x_values if abs(x - requested_x) <= 1e-8]
    if not matches:
        available = ", ".join(f"{x:.8g}" for x in x_values)
        raise ValueError(
            f"No collected candidate has endpoint_x={requested_x:.8g}. "
            f"Available actual dopant fractions: {available}"
        )
    return matches[0]


def _reference_labels(rows: List[Dict[str, Any]], source_metric: str) -> List[str]:
    prefix = f"{source_metric}__"
    return sorted(
        {
            column[len(prefix):]
            for row in rows
            for column in row
            if column.startswith(prefix)
        }
    )


def _calculate_relative_columns(
    rows: List[Dict[str, Any]],
    *,
    endpoint_x: float | None,
) -> set[str]:
    """
    Populate relative energies after ALL composition folders have been collected.

    This belongs in collection rather than a per-folder formation stage because
    the definition requires one global endpoint X for the entire workflow.
    """
    X = _actual_endpoint_x(rows, endpoint_x)
    added_columns: set[str] = set()

    for relative_metric, source_metric in _RELATIVE_SOURCES.items():
        for label in _reference_labels(rows, source_metric):
            source_column = _dynamic_column_name(source_metric, label)
            relative_column = _dynamic_column_name(relative_metric, label)

            endpoint_values = [
                float(row[source_column])
                for row in rows
                if _to_float(row.get("x_dopant")) is not None
                and abs(float(row["x_dopant"]) - X) <= 1e-8
                and _to_float(row.get(source_column)) is not None
            ]
            if not endpoint_values:
                log.warning(
                    "Cannot calculate %s: no valid endpoint energy for reference %s at X=%.8g",
                    relative_metric,
                    label,
                    X,
                )
                continue

            endpoint_energy = min(endpoint_values)
            for row in rows:
                x = _to_float(row.get("x_dopant"))
                energy = _to_float(row.get(source_column))
                row[relative_column] = (
                    None if x is None or energy is None
                    else energy - (x / X) * endpoint_energy
                )

            added_columns.add(relative_column)
            log.info(
                "Relative %s for %s: X=%.8g, E_min(X)=%.8f eV/cation",
                relative_metric,
                label,
                X,
                endpoint_energy,
            )

    return added_columns


def run_collect(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> Path:
    """Collect selected candidates into one wide, reference-aware database."""
    cfg = _parse_db_config(raw_cfg, root)
    out_csv = (root / OUT_CSV).resolve()

    if cfg.skip_if_done and out_csv.exists():
        log.info("SKIP %s already exists: %s", OUT_CSV, out_csv)
        log.info("Set [database].skip_if_done=false to overwrite.")
        return out_csv
    if not cfg.outdir.exists():
        raise FileNotFoundError(f"[structure].outdir not found: {cfg.outdir}")

    base_header = [
        "composition_tag",
        "requested_index",
        "requested_pct_json",
        "effective_pct_json",
        "rounded_counts_json",
        "host_species",
        "n_host",
        "supercell_json",
        "candidate",
        "candidate_path",
        "rank_relax_filtered",
        "E_relaxed_eV_filtered",
        "delta_e_eV",
        "filter_mode",
        "rank_scan",
        "E_scan_eV",
        "E_relaxed_eV",
        "bandgap_eV",
        "reference_mode",
        "x_dopant",
        "n_dopant_atoms",
        "dopant_counts_json",
        "dopant_counts",
        "reference_labels_json",
    ]
    rows_out: List[Dict[str, Any]] = []
    dynamic_header: set[str] = set()

    for folder in sorted(path for path in cfg.outdir.iterdir() if path.is_dir()):
        selected = read_selected_txt(folder / SELECTED_TXT)
        filtered = read_filtered_table(folder / RANK_RELAX_FILTERED)
        candidates = selected or sorted(filtered)
        if not candidates:
            log.warning(
                "Skip %s: no %s and no %s",
                folder.name,
                SELECTED_TXT,
                RANK_RELAX_FILTERED,
            )
            continue

        composition_meta = read_json(folder / META_COMP) or {}
        scans = read_scan_ranking(folder / RANK_SCAN)
        bandgaps = read_bandgap_summary(folder / BANDGAP_SUMMARY)
        formation_csv = read_formation_csv(folder / FORMATION_CSV)

        for candidate in candidates:
            candidate_dir = folder / candidate
            relax_meta = read_json(candidate_dir / RELAX_META) or {}
            fmeta = read_formation_meta(candidate_dir / FORMATION_META)
            fcsv = formation_csv.get(candidate, {})

            reference_results = (
                fmeta.get("reference_results")
                or fcsv.get("reference_results")
                or {}
            )
            if not isinstance(reference_results, dict):
                reference_results = {}

            dopant_counts_dict = fmeta.get("dopant_counts_dict")
            if not isinstance(dopant_counts_dict, dict):
                dopant_counts_dict = None

            n_dopants = (
                int(sum(int(value) for value in dopant_counts_dict.values()))
                if dopant_counts_dict is not None
                else fcsv.get("n_dopant_atoms")
            )
            x_dopant = fmeta.get("x_dopant")
            if x_dopant is None:
                x_dopant = fcsv.get("x_dopant")

            row: Dict[str, Any] = {
                "composition_tag": folder.name,
                "requested_index": safe_get(composition_meta, "requested_index"),
                "requested_pct_json": (
                    json.dumps(safe_get(composition_meta, "requested_pct"))
                    if safe_get(composition_meta, "requested_pct") is not None
                    else ""
                ),
                "effective_pct_json": (
                    json.dumps(safe_get(composition_meta, "effective_pct"))
                    if safe_get(composition_meta, "effective_pct") is not None
                    else ""
                ),
                "rounded_counts_json": (
                    json.dumps(safe_get(composition_meta, "rounded_counts"))
                    if safe_get(composition_meta, "rounded_counts") is not None
                    else ""
                ),
                "host_species": safe_get(composition_meta, "host_species", default=""),
                "n_host": safe_get(composition_meta, "n_host"),
                "supercell_json": (
                    json.dumps(safe_get(composition_meta, "supercell"))
                    if safe_get(composition_meta, "supercell") is not None
                    else ""
                ),
                "candidate": candidate,
                "candidate_path": str(candidate_dir.resolve()),
                "rank_relax_filtered": filtered.get(candidate, {}).get("rank_relax_filtered"),
                "E_relaxed_eV_filtered": filtered.get(candidate, {}).get("E_relaxed_eV_filtered"),
                "delta_e_eV": filtered.get(candidate, {}).get("delta_e_eV"),
                "filter_mode": filtered.get(candidate, {}).get("filter_mode", ""),
                "rank_scan": scans.get(candidate, {}).get("rank_scan"),
                "E_scan_eV": scans.get(candidate, {}).get("E_scan_eV"),
                "E_relaxed_eV": relax_meta.get("energy_relaxed_eV"),
                "bandgap_eV": bandgaps.get(candidate, {}).get("bandgap_eV"),
                "reference_mode": fmeta.get("reference_mode") or fcsv.get("reference_mode") or "",
                "x_dopant": _to_float(x_dopant),
                "n_dopant_atoms": n_dopants,
                "dopant_counts_json": (
                    json.dumps(dopant_counts_dict)
                    if dopant_counts_dict is not None
                    else ""
                ),
                "dopant_counts": fcsv.get("dopant_counts", ""),
                "reference_labels_json": json.dumps(sorted(reference_results)),
            }
            _populate_dynamic_columns(row, reference_results)
            dynamic_header.update(key for key in row if "__" in key)
            rows_out.append(row)

    if cfg.relative_enabled:
        dynamic_header.update(
            _calculate_relative_columns(
                rows_out,
                endpoint_x=cfg.relative_endpoint_x,
            )
        )

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_header + sorted(dynamic_header))
        writer.writeheader()
        writer.writerows(rows_out)

    log.info("DONE Step 07 collect: wrote %d rows to %s", len(rows_out), out_csv)
    return out_csv


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_collect_from_toml(config_path: Path) -> Path:
    raw = _load_raw_toml(config_path)
    return run_collect(raw, config_path.resolve().parent, config_path=config_path)

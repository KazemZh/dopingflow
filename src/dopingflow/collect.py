# src/dopingflow/collect.py
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dopingflow.corrections import (
    content_hash,
    load_active_correction_model,
    parse_correction_config,
)
from dopingflow.formation import RELAX_POSCAR as FORMATION_RELAX_POSCAR
from dopingflow.formation import _formation_correction_input_hash

log = logging.getLogger(__name__)

OUT_CSV = "results_database.csv"

# composition-level files
META_COMP = "metadata.json"
SELECTED_TXT = "selected_candidates.txt"
RANK_RELAX_FILTERED = "ranking_relax_filtered.csv"
RANK_SCAN = "ranking_scan.csv"
BANDGAP_SUMMARY = "bandgap_alignn_summary.csv"
FORMATION_CSV = "formation_energies.csv"

# candidate-level
RELAX_META = Path("02_relax") / "meta.json"
FORMATION_META = Path("04_formation") / "meta.json"


@dataclass(frozen=True)
class DBConfig:
    outdir: Path
    skip_if_done: bool


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_get(d: Optional[dict], *keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_int(x: Any) -> Optional[int]:
    try:
        text = str(x).strip()
        if text == "":
            return None
        return int(float(text))
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        text = str(x).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def _parse_db_config(raw: dict[str, Any], root: Path) -> DBConfig:
    st = raw.get("structure", {}) or {}
    db = raw.get("database", {}) or {}

    outdir_name = str(st.get("outdir", "random_structures"))
    outdir = (root / outdir_name).resolve()

    skip_if_done = bool(db.get("skip_if_done", True))
    return DBConfig(outdir=outdir, skip_if_done=skip_if_done)


def read_selected_txt(path: Path) -> List[str]:
    if not path.exists():
        return []
    names: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def read_filtered_table(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate = (row.get("candidate") or "").strip()
            if not candidate:
                continue

            out[candidate] = {
                "rank_relax_filtered": _to_int(row.get("rank_filtered")),
                "E_relaxed_eV_filtered": _to_float(row.get("energy_relaxed_eV")),
                "delta_e_eV": _to_float(row.get("delta_e_eV")),
                "filter_mode": (row.get("filter_mode") or "").strip(),
            }

    return out


def read_scan_ranking(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate = (
                row.get("candidate")
                or row.get("candidate_id")
                or row.get("name")
                or row.get("folder")
                or ""
            ).strip()
            if not candidate:
                continue

            rank = _to_int(row.get("rank") or row.get("rank_scan"))
            energy = _to_float(
                row.get("E_eV")
                or row.get("energy_eV")
                or row.get("E_scan_eV")
                or row.get("energy_sp_eV")
                or row.get("energy")
            )

            out[candidate] = {"rank_scan": rank, "E_scan_eV": energy}
    return out


def read_bandgap_summary(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate = (row.get("candidate") or row.get("candidate_id") or row.get("name") or "").strip()
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
                if key in row and row[key] is not None and str(row[key]).strip() != "":
                    bandgap = _to_float(row[key])
                    if bandgap is not None:
                        break

            out[candidate] = {"bandgap_eV": bandgap}
    return out


def read_formation_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    formation_energies.csv written by formation.py.

    This remains a fallback. The preferred source is candidate_*/04_formation/meta.json,
    because it contains the full ``reference_results`` block for multi-oxide runs.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate = (row.get("candidate") or "").strip()
            if not candidate:
                continue

            parsed: Dict[str, Any] = {
                "reference_mode": (row.get("reference_mode") or "").strip(),
                "E_form_eV_total": _to_float(
                    row.get("E_form_eV_total")
                    or row.get("E_form_total")
                    or row.get("E_form_total_eV")
                ),
                "E_form_norm": _to_float(
                    row.get("E_form_per_dopant")
                    or row.get("E_form_per_host")
                    or row.get("E_form_norm")
                    or row.get("E_form_total")
                ),
                "n_dopant_atoms": _to_int(row.get("n_dopant_atoms")),
                "dopant_counts": (row.get("dopant_counts") or "").strip(),
                # legacy/primary mixing-energy fallback columns
                "x_dopant": _to_float(row.get("x_dopant")),
                "E_mix_eV_total": _to_float(row.get("E_mix_eV_total")),
                "E_mix_eV_per_atom": _to_float(row.get("E_mix_eV_per_atom")),
                "E_mix_eV_per_cation": _to_float(row.get("E_mix_eV_per_cation")),
                "E_mix_eV_per_dopant": _to_float(row.get("E_mix_eV_per_dopant")),
                "n_O2_out": _to_float(row.get("n_O2_out")),
                "mixing_reaction_reference": (row.get("mixing_reaction_reference") or "").strip(),
            }

            # Preserve wide columns already written by the new formation.py, for example:
            # E_form_eV_per_cation__Sb2O5, E_mix_rel_eV_per_cation__SbO2, ...
            for key, value in row.items():
                if key in parsed or key in {
                    "candidate",
                    "E_doped_eV",
                    "n_dopant_atoms",
                    "dopant_counts",
                    "x_dopant",
                    "reference_mode",
                }:
                    continue
                if "__" in key or key.startswith(("E_form_", "E_mix_", "n_O2_out", "mixing_reaction_reference")):
                    parsed[key] = _to_float(value) if key.startswith(("E_", "n_O2")) else (value or "")

            out[candidate] = parsed
    return out


def _flatten_reference_results(reference_results: Any) -> Dict[str, Any]:
    """
    Flatten formation.py's multi-reference output without removing the old primary columns.

    Input shape:
      reference_results = {
        "Sb2O3": {
          "E_form_eV_per_cation": ...,
          "mixing": {"E_mix_eV_per_cation": ...},
          "relative": {"E_mix_rel_eV_per_cation": ...}
        },
        "Sb2O5": {...}
      }

    Output shape:
      E_form_eV_per_cation__Sb2O3
      E_mix_eV_per_cation__Sb2O3
      E_mix_rel_eV_per_cation__Sb2O3
      ...
    """
    values: Dict[str, Any] = {}
    if not isinstance(reference_results, dict):
        return values

    for label, result in sorted(reference_results.items()):
        if not isinstance(result, dict):
            continue
        label = str(label)

        # Formation energies and useful endpoint metadata
        for key in (
            "E_form_eV_total",
            "E_form_eV_per_atom",
            "E_form_eV_per_cation",
            "E_form_eV_per_dopant",
            "oxide_endpoint_eV_per_cation",
            "oxide_endpoint_correction_eV_per_cation",
        ):
            if key in result:
                values[f"{key}__{label}"] = result.get(key)

        for key in (
            "formation_energy_raw_eV_total",
            "energy_correction_eV_total",
            "correction_uncertainty_eV_total",
            "correction_uncertainty_eV_per_atom",
            "correction_uncertainty_eV_per_cation",
            "correction_uncertainty_eV_per_dopant",
            "E_form_corrected_eV_total",
            "E_form_corrected_eV_per_atom",
            "E_form_corrected_eV_per_cation",
            "E_form_corrected_eV_per_dopant",
        ):
            if key in result:
                values[f"{key}__{label}"] = result.get(key)
        correction = result.get("energy_correction", {}) or {}
        if isinstance(correction, dict) and correction:
            for output_key, source_key in (
                ("correction_applied", "applied"),
                ("correction_reason", "reason"),
                ("correction_method", "method"),
                ("correction_fit_id", "fit_id"),
                ("correction_model_family", "model_family"),
                ("correction_selection_run_hash", "selection_run_hash"),
                ("correction_experimental_dataset", "experimental_dataset"),
                (
                    "correction_experimental_dataset_version",
                    "experimental_dataset_version",
                ),
                ("correction_backend", "backend"),
                ("correction_model", "model"),
                ("correction_task", "task"),
                ("correction_feature_vector_json", "feature_vector"),
                (
                    "correction_applicability_signature_json",
                    "applicability_signature",
                ),
            ):
                values[f"{output_key}__{label}"] = correction.get(source_key)

        mixing = result.get("mixing", {}) or {}
        if isinstance(mixing, dict):
            for key in (
                "E_mix_eV_total",
                "E_mix_eV_per_atom",
                "E_mix_eV_per_cation",
                "E_mix_eV_per_dopant",
                "n_O2_out",
            ):
                if key in mixing:
                    values[f"{key}__{label}"] = mixing.get(key)
            for key in (
                "E_mix_raw_eV_total",
                "energy_correction_eV_total",
                "correction_uncertainty_eV_total",
                "E_mix_corrected_eV_total",
                "E_mix_corrected_eV_per_atom",
                "E_mix_corrected_eV_per_cation",
                "E_mix_corrected_eV_per_dopant",
            ):
                if key in mixing:
                    values[f"mixing_{key}__{label}"] = mixing.get(key)
            if "reaction_reference" in mixing:
                values[f"mixing_reaction_reference__{label}"] = mixing.get("reaction_reference", "")

        relative = result.get("relative", {}) or {}
        if isinstance(relative, dict):
            for key in (
                "endpoint_x",
                "E_endpoint_eV_per_cation",
                "E_form_endpoint_eV_per_cation",
                "E_mix_endpoint_eV_per_cation",
                "E_form_rel_eV_per_cation",
                "E_mix_rel_eV_per_cation",
            ):
                if key in relative:
                    output_key = "relative_endpoint_x" if key == "endpoint_x" else key
                    values[f"{output_key}__{label}"] = relative.get(key)
            if "reference" in relative:
                values[f"relative_reference__{label}"] = relative.get("reference", "")

        endpoint_by_dopant = result.get("oxide_endpoint_eV_per_cation_by_dopant", {}) or {}
        if isinstance(endpoint_by_dopant, dict) and endpoint_by_dopant:
            values[f"oxide_endpoint_by_dopant_json__{label}"] = json.dumps(
                endpoint_by_dopant, sort_keys=True
            )

        oxide_references = result.get("oxide_references", {}) or {}
        if isinstance(oxide_references, dict) and oxide_references:
            values[f"oxide_references_json__{label}"] = json.dumps(oxide_references, sort_keys=True)

    return values


def read_formation_meta(path: Path) -> Dict[str, Any]:
    """
    candidate_*/04_formation/meta.json written by formation.py.

    Backward compatibility:
      - keeps the old primary/top-level fields
      - additionally exposes wide multi-oxide fields from reference_results
    """
    data = read_json(path) or {}

    mixing = data.get("mixing", None)
    if not isinstance(mixing, dict):
        mixing = {}

    reference_results = data.get("reference_results", None)
    wide = _flatten_reference_results(reference_results)

    return {
        "reference_mode": data.get("reference_mode"),
        "E_form_eV_total": data.get("E_form_eV_total"),
        "E_form_norm": safe_get(data, "reported", "value", default=None),
        "E_form_norm_unit": safe_get(data, "reported", "unit", default=""),
        "dopant_counts_dict": data.get("dopant_counts", None),
        "n_atoms_supercell": data.get("n_atoms_supercell", None),
        "x_dopant": data.get("x_dopant", None),
        "primary_reference_label": data.get("primary_reference_label", ""),
        # primary/legacy mixing
        "E_mix_eV_total": mixing.get("E_mix_eV_total", None),
        "E_mix_eV_per_atom": mixing.get("E_mix_eV_per_atom", None),
        "E_mix_eV_per_cation": mixing.get("E_mix_eV_per_cation", None),
        "E_mix_eV_per_dopant": mixing.get("E_mix_eV_per_dopant", None),
        "n_O2_out": mixing.get("n_O2_out", None),
        "mixing_reaction_reference": mixing.get("reaction_reference", ""),
        # full multi-oxide/wide output
        "wide_reference_results": wide,
        "energy_correction": data.get("energy_correction", None),
        "E_form_corrected_eV_total": data.get("E_form_corrected_eV_total", None),
        "energy_correction_eV_total": data.get("energy_correction_eV_total", None),
        "correction_uncertainty_eV_total": data.get(
            "correction_uncertainty_eV_total", None
        ),
        "E_form_corrected_norm": safe_get(
            data, "reported_corrected", "value", default=None
        ),
        "E_form_corrected_norm_uncertainty": safe_get(
            data, "reported_corrected", "uncertainty", default=None
        ),
        "E_form_corrected_norm_unit": safe_get(
            data, "reported_corrected", "unit", default=""
        ),
    }


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


_ENERGY_CORRECTION_COLUMN_PREFIXES = (
    "formation_energy_raw_",
    "formation_correction_",
    "energy_correction_",
    "correction_",
    "E_form_corrected_",
    "E_mix_raw_",
    "E_mix_corrected_",
    "mixing_E_mix_raw_",
    "mixing_energy_correction_",
    "mixing_correction_",
    "mixing_E_mix_corrected_",
    "experimental_dataset",
)


def _is_energy_correction_column(name: str) -> bool:
    return str(name).startswith(_ENERGY_CORRECTION_COLUMN_PREFIXES)


def _database_has_correction_columns(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
    except (OSError, csv.Error):
        return False
    return any(_is_energy_correction_column(field) for field in fieldnames)


def run_collect(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> Path:
    """
    Step 07: Collect results into ONE flat CSV database (results_database.csv),
    only for filtered/selected candidates.

    This collector preserves the historical primary columns and adds wide columns
    for every oxide-reference scenario found in formation.py's ``reference_results``.
    """
    cfg = _parse_db_config(raw_cfg, root)
    correction_config = parse_correction_config(raw_cfg, root)
    correction_model = None
    reference_input_hash = ""
    if correction_config.enabled:
        reference_path = root / "reference_structures" / "reference_energies.json"
        reference_data = read_json(reference_path)
        if reference_data is None:
            raise FileNotFoundError(
                "Energy correction is enabled but reference_energies.json is missing "
                "or invalid; run refs-build and corrections-fit first"
            )
        correction_model = load_active_correction_model(
            raw_cfg,
            root,
            reference_data,
        )
        reference_input_hash = content_hash(reference_data)

    out_csv = (root / OUT_CSV).resolve()
    if cfg.skip_if_done and out_csv.exists():
        if correction_config.enabled:
            log.info(
                "REBUILD %s: energy correction is enabled and formation inputs "
                "may have changed",
                OUT_CSV,
            )
        elif _database_has_correction_columns(out_csv):
            log.info(
                "REBUILD %s: energy correction is disabled but the existing "
                "database contains corrected columns",
                OUT_CSV,
            )
        else:
            log.info("SKIP %s already exists: %s", OUT_CSV, out_csv)
            log.info("Set [database].skip_if_done=false to overwrite.")
            return out_csv

    if not cfg.outdir.exists():
        raise FileNotFoundError(f"[structure].outdir not found: {cfg.outdir}")

    folders = sorted(path for path in cfg.outdir.iterdir() if path.is_dir())
    log.info("Step 07 collect: %d composition folders in: %s", len(folders), cfg.outdir)

    base_header = [
        # composition-level
        "composition_tag",
        "requested_index",
        "requested_pct_json",
        "effective_pct_json",
        "rounded_counts_json",
        "host_species",
        "n_host",
        "supercell_json",
        # candidate-level
        "candidate",
        "candidate_path",
        # relax filtered
        "rank_relax_filtered",
        "E_relaxed_eV_filtered",
        "delta_e_eV",
        "filter_mode",
        # scan
        "rank_scan",
        "E_scan_eV",
        # relax meta
        "E_relaxed_eV",
        # bandgap
        "bandgap_eV",
        # formation: primary/backward-compatible columns
        "reference_mode",
        "primary_reference_label",
        "E_form_eV_total",
        "E_form_norm",
        "E_form_norm_unit",
        "n_dopant_atoms",
        "dopant_counts_json",
        # mixing: primary/backward-compatible columns
        "x_dopant",
        "E_mix_eV_total",
        "E_mix_eV_per_atom",
        "E_mix_eV_per_cation",
        "E_mix_eV_per_dopant",
        "n_O2_out",
        "mixing_reaction_reference",
        # legacy string
        "dopant_counts",
    ]

    rows_out: List[Dict[str, Any]] = []
    dynamic_fields: set[str] = set()

    for folder in folders:
        comp_tag = folder.name

        selected = read_selected_txt(folder / SELECTED_TXT)
        filtered_map = read_filtered_table(folder / RANK_RELAX_FILTERED)

        if selected:
            candidate_names = selected
        elif filtered_map:
            candidate_names = sorted(filtered_map.keys())
        else:
            log.warning("Skip %s: no %s and no %s", comp_tag, SELECTED_TXT, RANK_RELAX_FILTERED)
            continue

        comp_meta = read_json(folder / META_COMP) or {}
        scan_map = read_scan_ranking(folder / RANK_SCAN)
        bg_map = read_bandgap_summary(folder / BANDGAP_SUMMARY)
        form_csv_map = read_formation_csv(folder / FORMATION_CSV)

        requested_pct = safe_get(comp_meta, "requested_pct", default=None)
        effective_pct = safe_get(comp_meta, "effective_pct", default=None)
        rounded_counts = safe_get(comp_meta, "rounded_counts", default=None)
        supercell = safe_get(comp_meta, "supercell", default=None)

        for candidate in candidate_names:
            cand_dir = folder / candidate
            relax_meta = read_json(cand_dir / RELAX_META) or {}

            fmeta = read_formation_meta(cand_dir / FORMATION_META)
            fcsv = form_csv_map.get(candidate, {}) if isinstance(form_csv_map, dict) else {}

            reference_mode = fmeta.get("reference_mode") or fcsv.get("reference_mode") or ""

            E_form_total = fmeta.get("E_form_eV_total")
            E_form_norm = fmeta.get("E_form_norm")
            E_form_unit = fmeta.get("E_form_norm_unit") or ""

            if E_form_total is None:
                E_form_total = fcsv.get("E_form_eV_total")
            if E_form_norm is None:
                E_form_norm = fcsv.get("E_form_norm")

            dop_counts_dict = fmeta.get("dopant_counts_dict")
            dop_counts_json = json.dumps(dop_counts_dict, sort_keys=True) if isinstance(dop_counts_dict, dict) else ""
            dop_counts_legacy = fcsv.get("dopant_counts", "") if isinstance(fcsv, dict) else ""

            n_dopant_atoms = None
            if isinstance(dop_counts_dict, dict):
                try:
                    n_dopant_atoms = int(sum(int(value) for value in dop_counts_dict.values()))
                except Exception:
                    n_dopant_atoms = None
            if n_dopant_atoms is None and isinstance(fcsv, dict):
                n_dopant_atoms = _to_int(fcsv.get("n_dopant_atoms"))

            x_dopant = fmeta.get("x_dopant")
            E_mix_total = fmeta.get("E_mix_eV_total")
            E_mix_atom = fmeta.get("E_mix_eV_per_atom")
            E_mix_cation = fmeta.get("E_mix_eV_per_cation")
            E_mix_dopant = fmeta.get("E_mix_eV_per_dopant")
            n_O2_out = fmeta.get("n_O2_out")
            mixing_reaction = fmeta.get("mixing_reaction_reference") or ""

            if x_dopant is None:
                x_dopant = fcsv.get("x_dopant")
            if E_mix_total is None:
                E_mix_total = fcsv.get("E_mix_eV_total")
            if E_mix_atom is None:
                E_mix_atom = fcsv.get("E_mix_eV_per_atom")
            if E_mix_cation is None:
                E_mix_cation = fcsv.get("E_mix_eV_per_cation")
            if E_mix_dopant is None:
                E_mix_dopant = fcsv.get("E_mix_eV_per_dopant")
            if n_O2_out is None:
                n_O2_out = fcsv.get("n_O2_out")
            if not mixing_reaction:
                mixing_reaction = fcsv.get("mixing_reaction_reference", "")

            row: Dict[str, Any] = {
                "composition_tag": comp_tag,
                "requested_index": safe_get(comp_meta, "requested_index", default=None),
                "requested_pct_json": json.dumps(requested_pct, sort_keys=True) if requested_pct is not None else "",
                "effective_pct_json": json.dumps(effective_pct, sort_keys=True) if effective_pct is not None else "",
                "rounded_counts_json": json.dumps(rounded_counts, sort_keys=True) if rounded_counts is not None else "",
                "host_species": safe_get(comp_meta, "host_species", default=""),
                "n_host": safe_get(comp_meta, "n_host", default=None),
                "supercell_json": json.dumps(supercell, sort_keys=True) if supercell is not None else "",
                "candidate": candidate,
                "candidate_path": str(cand_dir.resolve()),
                "rank_relax_filtered": filtered_map.get(candidate, {}).get("rank_relax_filtered", None),
                "E_relaxed_eV_filtered": filtered_map.get(candidate, {}).get("E_relaxed_eV_filtered", None),
                "delta_e_eV": filtered_map.get(candidate, {}).get("delta_e_eV", None),
                "filter_mode": filtered_map.get(candidate, {}).get("filter_mode", ""),
                "rank_scan": scan_map.get(candidate, {}).get("rank_scan", None),
                "E_scan_eV": scan_map.get(candidate, {}).get("E_scan_eV", None),
                "E_relaxed_eV": relax_meta.get("energy_relaxed_eV", None),
                "bandgap_eV": bg_map.get(candidate, {}).get("bandgap_eV", None),
                "reference_mode": reference_mode,
                "primary_reference_label": fmeta.get("primary_reference_label") or "",
                "E_form_eV_total": _to_float(E_form_total),
                "E_form_norm": _to_float(E_form_norm),
                "E_form_norm_unit": str(E_form_unit or ""),
                "n_dopant_atoms": n_dopant_atoms,
                "dopant_counts_json": dop_counts_json,
                "x_dopant": _to_float(x_dopant),
                "E_mix_eV_total": _to_float(E_mix_total),
                "E_mix_eV_per_atom": _to_float(E_mix_atom),
                "E_mix_eV_per_cation": _to_float(E_mix_cation),
                "E_mix_eV_per_dopant": _to_float(E_mix_dopant),
                "n_O2_out": _to_float(n_O2_out),
                "mixing_reaction_reference": str(mixing_reaction or ""),
                "dopant_counts": dop_counts_legacy,
            }

            correction_metadata = fmeta.get("energy_correction")
            if correction_config.enabled:
                if not isinstance(correction_metadata, dict) or not correction_metadata:
                    raise ValueError(
                        f"Correction-enabled collection requires corrected formation "
                        f"metadata for {cand_dir}; rerun the formation stage"
                    )
                assert correction_model is not None
                if correction_metadata.get("fit_id") != correction_model.fit_id:
                    raise ValueError(
                        f"Formation correction fit for {cand_dir} is stale; rerun "
                        "the formation stage before collect"
                    )
                try:
                    expected_formation_hash = _formation_correction_input_hash(
                        cand_dir / FORMATION_RELAX_POSCAR,
                        correction_model,
                        dict(raw_cfg.get("formation", {}) or {}),
                        reference_input_hash,
                    )
                except OSError as exc:
                    raise ValueError(
                        f"Cannot validate correction inputs for {cand_dir}; rerun "
                        "the formation stage"
                    ) from exc
                if (
                    correction_metadata.get("formation_input_hash")
                    != expected_formation_hash
                ):
                    raise ValueError(
                        f"Formation correction inputs for {cand_dir} changed; rerun "
                        "the formation stage before collect"
                    )
                primary_correction_values = {
                    "formation_energy_raw_eV_total": _to_float(E_form_total),
                    "energy_correction_eV_total": _to_float(
                        fmeta.get("energy_correction_eV_total")
                    ),
                    "correction_uncertainty_eV_total": _to_float(
                        fmeta.get("correction_uncertainty_eV_total")
                    ),
                    "E_form_corrected_eV_total": _to_float(
                        fmeta.get("E_form_corrected_eV_total")
                    ),
                    "E_form_corrected_norm": _to_float(
                        fmeta.get("E_form_corrected_norm")
                    ),
                    "E_form_corrected_norm_uncertainty": _to_float(
                        fmeta.get("E_form_corrected_norm_uncertainty")
                    ),
                    "E_form_corrected_norm_unit": str(
                        fmeta.get("E_form_corrected_norm_unit") or ""
                    ),
                    "correction_applied": correction_metadata.get("applied", False),
                    "correction_reason": correction_metadata.get("reason", ""),
                    "correction_method": correction_metadata.get("method", ""),
                    "correction_fit_id": correction_metadata.get("fit_id", ""),
                    "correction_model_family": correction_metadata.get(
                        "model_family", ""
                    ),
                    "correction_selection_run_hash": correction_metadata.get(
                        "selection_run_hash", ""
                    ),
                    "correction_parameter_set": correction_metadata.get(
                        "parameter_set", ""
                    ),
                    "experimental_dataset": correction_metadata.get(
                        "experimental_dataset", ""
                    ),
                    "experimental_dataset_version": correction_metadata.get(
                        "experimental_dataset_version", ""
                    ),
                    "correction_activation_input_hash": correction_metadata.get(
                        "activation_input_hash", ""
                    ),
                    "formation_correction_input_hash": correction_metadata.get(
                        "formation_input_hash", ""
                    ),
                    "candidate_energy_provenance_mode": safe_get(
                        correction_metadata,
                        "candidate_energy_provenance",
                        "mode",
                        default="",
                    ),
                    "candidate_energy_provenance_assumptions_json": safe_get(
                        correction_metadata,
                        "candidate_energy_provenance",
                        "assumptions",
                        default=[],
                    ),
                    "candidate_energy_execution_differences_json": safe_get(
                        correction_metadata,
                        "candidate_energy_provenance",
                        "execution_differences",
                        default=[],
                    ),
                    "candidate_energy_current_poscar_sha256": safe_get(
                        correction_metadata,
                        "candidate_energy_provenance",
                        "current_relaxed_poscar_sha256",
                        default="",
                    ),
                    "candidate_energy_original_poscar_sha256": safe_get(
                        correction_metadata,
                        "candidate_energy_provenance",
                        "original_relaxed_poscar_sha256",
                        default="",
                    ),
                    "correction_applicability_signature_json": (
                        correction_metadata.get("applicability_signature", {})
                    ),
                    "correction_backend": safe_get(
                        correction_metadata, "backend_signature", "backend", default=""
                    ),
                    "correction_model": safe_get(
                        correction_metadata, "backend_signature", "model", default=""
                    ),
                    "correction_task": safe_get(
                        correction_metadata, "backend_signature", "task", default=""
                    ),
                    "correction_backend_package": safe_get(
                        correction_metadata,
                        "backend_signature",
                        "backend_package",
                        default="",
                    ),
                    "correction_backend_package_version": safe_get(
                        correction_metadata,
                        "backend_signature",
                        "backend_package_version",
                        default="",
                    ),
                    "correction_model_checkpoint_sha256": safe_get(
                        correction_metadata,
                        "backend_signature",
                        "model_checkpoint_sha256",
                        default="",
                    ),
                }
                row.update(primary_correction_values)
                dynamic_fields.update(primary_correction_values)

            # Add multi-oxide wide columns from meta.json first.
            wide = fmeta.get("wide_reference_results", {})
            if isinstance(wide, dict):
                for key, value in wide.items():
                    if (
                        not correction_config.enabled
                        and _is_energy_correction_column(key)
                    ):
                        continue
                    row[key] = value
                    dynamic_fields.add(key)

            # Add wide fallback columns from formation_energies.csv, but do not overwrite meta.
            if isinstance(fcsv, dict):
                for key, value in fcsv.items():
                    if key in row or key in {
                        "reference_mode",
                        "E_form_eV_total",
                        "E_form_norm",
                        "n_dopant_atoms",
                        "dopant_counts",
                        "x_dopant",
                        "E_mix_eV_total",
                        "E_mix_eV_per_atom",
                        "E_mix_eV_per_cation",
                        "E_mix_eV_per_dopant",
                        "n_O2_out",
                        "mixing_reaction_reference",
                    }:
                        continue
                    if "__" in key:
                        if (
                            not correction_config.enabled
                            and _is_energy_correction_column(key)
                        ):
                            continue
                        row[key] = value
                        dynamic_fields.add(key)

            rows_out.append(row)

    # Stable ordering: historical columns first, then all wide multi-reference columns.
    header = base_header + sorted(field for field in dynamic_fields if field not in base_header)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow({key: _format_csv_value(row.get(key, "")) for key in header})

    log.info("DONE Step 07 collect: wrote %d rows to %s", len(rows_out), out_csv)
    return out_csv


# TOML wrapper
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_collect_from_toml(config_path: Path) -> Path:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    return run_collect(raw, root, config_path=config_path)

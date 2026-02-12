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

# composition-level files
META_COMP = "metadata.json"
SELECTED_TXT = "selected_candidates.txt"
RANK_RELAX_FILTERED = "ranking_relax_filtered.csv"  # fallback
RANK_SCAN = "ranking_scan.csv"
BANDGAP_SUMMARY = "bandgap_alignn_summary.csv"
FORMATION_CSV = "formation_energies.csv"

# candidate-level
RELAX_META = Path("02_relax") / "meta.json"


@dataclass(frozen=True)
class DBConfig:
    outdir: Path
    skip_if_done: bool


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(d: Optional[dict], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).strip())
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
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            names.append(ln)
    return names


def read_filtered_table(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse ranking_relax_filtered.csv (fallback) into:
      candidate -> {"rank_relax_filtered": int, "E_relaxed_eV_filtered": float}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cand = (row.get("candidate") or row.get("candidate_id") or row.get("name") or "").strip()
            if not cand:
                continue

            E_rel = (
                row.get("E_relaxed_eV")
                or row.get("energy_relaxed_eV")
                or row.get("E_eV")
                or row.get("energy")
            )
            rank_rel = row.get("rank_relax") or row.get("rank")

            out[cand] = {
                "rank_relax_filtered": _to_int(rank_rel),
                "E_relaxed_eV_filtered": _to_float(E_rel),
            }

    return out


def read_scan_ranking(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cand = (
                row.get("candidate")
                or row.get("candidate_id")
                or row.get("name")
                or row.get("folder")
                or ""
            ).strip()
            if not cand:
                continue

            rank = _to_int(row.get("rank") or row.get("rank_scan"))
            E = _to_float(row.get("E_eV") or row.get("energy_eV") or row.get("E_scan_eV") or row.get("energy"))

            out[cand] = {"rank_scan": rank, "E_scan_eV": E}
    return out


def read_bandgap_summary(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cand = (row.get("candidate") or row.get("candidate_id") or row.get("name") or "").strip()
            if not cand:
                continue

            bg = None
            for key in ("bandgap_eV_ALIGNN_MBJ", "bandgap_eV", "bandgap", "pred_bandgap", "pred_bandgap_eV"):
                if key in row and row[key] is not None and str(row[key]).strip() != "":
                    bg = _to_float(row[key])
                    if bg is not None:
                        break

            out[cand] = {"bandgap_eV": bg}
    return out


def read_formation_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cand = (row.get("candidate") or "").strip()
            if not cand:
                continue

            out[cand] = {
                "E_form_eV_total": _to_float(row.get("E_form_eV_total")),
                "E_form_norm": _to_float(
                    row.get("E_form_per_dopant")
                    or row.get("E_form_per_host")
                    or row.get("E_form_norm")
                    or row.get("E_form_total")
                ),
                "n_dopant_atoms": _to_int(row.get("n_dopant_atoms")),
                "dopant_counts": (row.get("dopant_counts") or "").strip(),
            }
    return out


def run_collect(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> Path:
    """
    Step 07: Collect results into ONE flat CSV database (results_database.csv),
    ONLY for the filtered/selected candidates (Step 04 output).

    Selection priority:
      1) selected_candidates.txt
      2) ranking_relax_filtered.csv
    """
    cfg = _parse_db_config(raw_cfg, root)

    out_csv = (root / OUT_CSV).resolve()
    if cfg.skip_if_done and out_csv.exists():
        log.info("SKIP %s already exists: %s", OUT_CSV, out_csv)
        log.info("Set [database].skip_if_done=false to overwrite.")
        return out_csv

    if not cfg.outdir.exists():
        raise FileNotFoundError(f"[structure].outdir not found: {cfg.outdir}")

    folders = sorted([p for p in cfg.outdir.iterdir() if p.is_dir()])
    log.info("Step 07 collect: %d composition folders in: %s", len(folders), cfg.outdir)

    header = [
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
        # relax filtered (from ranking_relax_filtered.csv if available)
        "rank_relax_filtered",
        "E_relaxed_eV_filtered",
        # scan
        "rank_scan",
        "E_scan_eV",
        # relax meta (step 03)
        "E_relaxed_eV",
        # bandgap
        "bandgap_eV",
        # formation
        "E_form_eV_total",
        "E_form_norm",
        "n_dopant_atoms",
        "dopant_counts",
    ]

    rows_out: List[Dict[str, Any]] = []

    for folder in folders:
        comp_tag = folder.name

        selected = read_selected_txt(folder / SELECTED_TXT)
        filtered_map = read_filtered_table(folder / RANK_RELAX_FILTERED)

        # Strict selection policy (never include unfiltered candidates)
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
        form_map = read_formation_csv(folder / FORMATION_CSV)

        requested_pct = safe_get(comp_meta, "requested_pct", default=None)
        effective_pct = safe_get(comp_meta, "effective_pct", default=None)
        rounded_counts = safe_get(comp_meta, "rounded_counts", default=None)
        supercell = safe_get(comp_meta, "supercell", default=None)

        for cand in candidate_names:
            cand_dir = folder / cand
            relax_meta = read_json(cand_dir / RELAX_META) or {}

            row = {
                "composition_tag": comp_tag,
                "requested_index": safe_get(comp_meta, "requested_index", default=None),
                "requested_pct_json": json.dumps(requested_pct) if requested_pct is not None else "",
                "effective_pct_json": json.dumps(effective_pct) if effective_pct is not None else "",
                "rounded_counts_json": json.dumps(rounded_counts) if rounded_counts is not None else "",
                "host_species": safe_get(comp_meta, "host_species", default=""),
                "n_host": safe_get(comp_meta, "n_host", default=None),
                "supercell_json": json.dumps(supercell) if supercell is not None else "",
                "candidate": cand,
                "candidate_path": str(cand_dir.resolve()),
                # filtered relax info (if file exists)
                "rank_relax_filtered": filtered_map.get(cand, {}).get("rank_relax_filtered", None),
                "E_relaxed_eV_filtered": filtered_map.get(cand, {}).get("E_relaxed_eV_filtered", None),
                # scan
                "rank_scan": scan_map.get(cand, {}).get("rank_scan", None),
                "E_scan_eV": scan_map.get(cand, {}).get("E_scan_eV", None),
                # relax meta
                "E_relaxed_eV": relax_meta.get("energy_relaxed_eV", None),
                # bandgap
                "bandgap_eV": bg_map.get(cand, {}).get("bandgap_eV", None),
                # formation
                "E_form_eV_total": form_map.get(cand, {}).get("E_form_eV_total", None),
                "E_form_norm": form_map.get(cand, {}).get("E_form_norm", None),
                "n_dopant_atoms": form_map.get(cand, {}).get("n_dopant_atoms", None),
                "dopant_counts": form_map.get(cand, {}).get("dopant_counts", ""),
            }

            rows_out.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    log.info("DONE Step 07 collect: wrote %d rows to %s", len(rows_out), out_csv)
    return out_csv


# TOML wrapper
try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_collect_from_toml(config_path: Path) -> Path:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    return run_collect(raw, root, config_path=config_path)

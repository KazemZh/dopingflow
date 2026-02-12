# src/dopingflow/filtering.py
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# -----------------------------
# Fixed outputs (do not expose to users)
# -----------------------------
RANKING_IN = "ranking_relax.csv"
RANKING_OUT = "ranking_relax_filtered.csv"
SELECTED_LIST = "selected_candidates.txt"


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class FilterConfig:
    outdir: Path
    mode: str                 # "window" or "topn"
    window_meV: float         # used if mode="window"
    max_candidates: int       # used if mode="topn"
    skip_if_done: bool


def _parse_filter_config(raw: dict[str, Any], root: Path) -> FilterConfig:
    st = raw.get("structure", {}) or {}
    flt = raw.get("filter", {}) or {}

    outdir_name = str(st.get("outdir", "random_structures"))
    outdir = (root / outdir_name).resolve()

    mode = str(flt.get("mode", "window")).strip().lower()
    window_meV = float(flt.get("window_meV", 50.0))
    max_candidates = int(flt.get("max_candidates", 12))
    skip_if_done = bool(flt.get("skip_if_done", True))

    if mode not in {"window", "topn"}:
        raise ValueError("[filter].mode must be 'window' or 'topn'")
    if window_meV <= 0:
        raise ValueError("[filter].window_meV must be > 0")
    if max_candidates <= 0:
        raise ValueError("[filter].max_candidates must be > 0")

    return FilterConfig(
        outdir=outdir,
        mode=mode,
        window_meV=window_meV,
        max_candidates=max_candidates,
        skip_if_done=skip_if_done,
    )


# -----------------------------
# I/O helpers
# -----------------------------
def _read_ranking_relax(path: Path) -> List[Dict[str, Any]]:
    """
    Read ranking_relax.csv and keep only rows with status==ok.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            status = (r.get("status") or "").strip()
            if status != "ok":
                continue

            cand = (r.get("candidate") or "").strip()
            if not cand:
                continue

            try:
                e_rel = float(r["energy_relaxed_eV"])
            except Exception:
                continue

            rows.append(
                {
                    "candidate": cand,
                    "energy_relaxed_eV": e_rel,
                    "rank_relax": r.get("rank_relax", ""),
                    "rank_sp": r.get("rank_sp", ""),
                    "energy_sp_eV": r.get("energy_sp_eV", ""),
                    "signature": r.get("signature", ""),
                }
            )

    if not rows:
        raise RuntimeError(f"No OK rows found in {path}. Did Step 03 finish successfully?")
    return rows


def _write_outputs(folder: Path, kept: List[Dict[str, Any]], emin: float, mode_desc: str) -> None:
    out_csv = folder / RANKING_OUT
    out_list = folder / SELECTED_LIST

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank_filtered",
                "candidate",
                "energy_relaxed_eV",
                "delta_e_eV",
                "rank_relax",
                "rank_sp",
                "energy_sp_eV",
                "signature",
                "filter_mode",
            ]
        )
        for i, r in enumerate(kept, start=1):
            dE = r["energy_relaxed_eV"] - emin
            w.writerow(
                [
                    i,
                    r["candidate"],
                    f"{r['energy_relaxed_eV']:.10f}",
                    f"{dE:.10f}",
                    r.get("rank_relax", ""),
                    r.get("rank_sp", ""),
                    r.get("energy_sp_eV", ""),
                    r.get("signature", ""),
                    mode_desc,
                ]
            )

    out_list.write_text("\n".join([r["candidate"] for r in kept]) + "\n", encoding="utf-8")


# -----------------------------
# Core filtering
# -----------------------------
def _filter_rows(
    rows: List[Dict[str, Any]],
    cfg: FilterConfig,
    window_meV_override: Optional[float],
    topn_override: Optional[int],
) -> Tuple[List[Dict[str, Any]], float, str]:
    rows_sorted = sorted(rows, key=lambda r: r["energy_relaxed_eV"])
    emin = float(rows_sorted[0]["energy_relaxed_eV"])

    # overrides force the mode
    mode = cfg.mode
    if topn_override is not None:
        mode = "topn"
    if window_meV_override is not None:
        mode = "window"

    if mode == "window":
        window_meV = float(window_meV_override) if window_meV_override is not None else cfg.window_meV
        window_eV = window_meV / 1000.0
        kept = [r for r in rows_sorted if (r["energy_relaxed_eV"] - emin) <= window_eV]
        mode_desc = f"window_{window_meV:g}meV"
        return kept, emin, mode_desc

    # mode == "topn"
    topn = int(topn_override) if topn_override is not None else cfg.max_candidates
    kept = rows_sorted[: min(topn, len(rows_sorted))]
    mode_desc = f"topn_{topn}"
    return kept, emin, mode_desc


def _process_folder(
    folder: Path,
    cfg: FilterConfig,
    *,
    window_meV_override: Optional[float],
    topn_override: Optional[int],
    force: bool,
) -> None:
    ranking_path = folder / RANKING_IN
    if not ranking_path.exists():
        log.info("SKIP %s: missing %s (run Step 03 first)", folder.name, RANKING_IN)
        return

    out_csv = folder / RANKING_OUT
    out_list = folder / SELECTED_LIST

    if cfg.skip_if_done and (not force) and out_csv.exists() and out_list.exists():
        log.info("SKIP %s: outputs exist (%s, %s)", folder.name, RANKING_OUT, SELECTED_LIST)
        return

    rows = _read_ranking_relax(ranking_path)
    kept, emin, mode_desc = _filter_rows(rows, cfg, window_meV_override, topn_override)

    _write_outputs(folder, kept, emin, mode_desc)

    log.info("OK   %s: Emin=%.10f eV | kept %d/%d | %s", folder.name, emin, len(kept), len(rows), mode_desc)
    log.info("     wrote: %s, %s", out_csv, out_list)


# -----------------------------
# Public API
# -----------------------------
def run_filtering(
    raw_cfg: dict[str, Any],
    root: Path,
    *,
    only: Optional[str] = None,
    force: bool = False,
    window_meV: Optional[float] = None,
    topn: Optional[int] = None,
) -> None:
    """
    Step 04: filter relaxed candidates after Step 03.

    For each composition folder in [structure].outdir:
      reads: ranking_relax.csv
      writes: ranking_relax_filtered.csv, selected_candidates.txt

    Filtering:
      - [filter].mode="window": keep candidates within window_meV above Emin
      - [filter].mode="topn": keep lowest-energy max_candidates
    """
    cfg = _parse_filter_config(raw_cfg, root)

    if not cfg.outdir.exists():
        raise FileNotFoundError(f"Output directory not found: {cfg.outdir} (did you run Step 01?)")

    folders = sorted([p for p in cfg.outdir.iterdir() if p.is_dir()])

    log.info("Step 04 filter: %d structure folders in: %s", len(folders), cfg.outdir)
    log.info("Input : %s", RANKING_IN)
    log.info("Output: %s, %s", RANKING_OUT, SELECTED_LIST)

    for i, folder in enumerate(folders, start=1):
        if only and folder.name != only:
            continue

        log.info("RUN (%d/%d) %s", i, len(folders), folder.name)
        _process_folder(folder, cfg, window_meV_override=window_meV, topn_override=topn, force=force)

    log.info("DONE Step 04 filter for all structure folders.")


# TOML wrapper (like your other steps)
try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_filtering_from_toml(
    config_path: Path,
    *,
    only: Optional[str] = None,
    force: bool = False,
    window_meV: Optional[float] = None,
    topn: Optional[int] = None,
) -> None:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    run_filtering(raw, root, only=only, force=force, window_meV=window_meV, topn=topn)

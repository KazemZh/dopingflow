# src/dopingflow/relax.py
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Keep TF noise low in *this* process (workers also set it again).
# ----------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")


# -----------------------------
# Config model
# -----------------------------
@dataclass(frozen=True)
class RelaxConfig:
    fmax: float
    n_workers: int
    tf_threads: int
    omp_threads: int
    order: List[str]         # from [generate].poscar_order
    outdir: Path             # from [structure].outdir
    skip_if_done: bool       # folder-level skip if ranking_relax exists
    skip_candidate_if_done: bool  # candidate-level skip if 02_relax/meta exists


def _parse_relax_config(raw: dict[str, Any], root: Path) -> RelaxConfig:
    st = raw.get("structure", {}) or {}
    gen = raw.get("generate", {}) or {}
    rel = raw.get("relax", {}) or {}
    dop = raw.get("doping", {}) or {}

    outdir_name = str(st.get("outdir", "random_structures"))
    outdir = (root / outdir_name).resolve()

    order = [str(x) for x in (gen.get("poscar_order", []) or [])]

    fmax = float(rel.get("fmax", 0.05))
    n_workers = int(rel.get("n_workers", 6))
    tf_threads = int(rel.get("tf_threads", 1))
    omp_threads = int(rel.get("omp_threads", 1))
    skip_if_done = bool(rel.get("skip_if_done", True))
    skip_candidate_if_done = bool(rel.get("skip_candidate_if_done", True))

    if not order:
        raise ValueError("[generate].poscar_order must be defined and non-empty")
    if fmax <= 0:
        raise ValueError("[relax].fmax must be > 0")
    if n_workers <= 0:
        raise ValueError("[relax].n_workers must be > 0")
    if tf_threads <= 0:
        raise ValueError("[relax].tf_threads must be > 0")
    if omp_threads <= 0:
        raise ValueError("[relax].omp_threads must be > 0")

    # host_species is not strictly required for relaxation, but helps catch mis-configs
    _ = str(dop.get("host_species", "")).strip()

    return RelaxConfig(
        fmax=fmax,
        n_workers=n_workers,
        tf_threads=tf_threads,
        omp_threads=omp_threads,
        order=order,
        outdir=outdir,
        skip_if_done=skip_if_done,
        skip_candidate_if_done=skip_candidate_if_done,
    )


# -----------------------------
# Helpers
# -----------------------------
def _reorder_sites(struct: Structure, order: List[str]) -> Structure:
    order_index = {el: i for i, el in enumerate(order)}

    def key(site):
        el = site.species_string
        return (
            order_index.get(el, 999),
            el,
            site.frac_coords[2],
            site.frac_coords[1],
            site.frac_coords[0],
        )

    sites_sorted = sorted(struct.sites, key=key)
    return Structure(struct.lattice, [s.species for s in sites_sorted], [s.frac_coords for s in sites_sorted])


def _safe_write_poscar(struct: Structure, path: Path, order: List[str]) -> None:
    s2 = _reorder_sites(struct, order)
    Poscar(s2).write_file(str(path), vasp4_compatible=False)


def _species_counts(struct: Structure) -> Dict[str, int]:
    return dict(Counter(site.species_string for site in struct))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _trajectory_last_energy(result: Dict[str, Any]) -> float:
    traj = result.get("trajectory", None)
    if traj is None:
        raise KeyError("Relaxer result has no 'trajectory'.")
    if hasattr(traj, "energies") and len(traj.energies) > 0:
        return float(traj.energies[-1])
    raise KeyError("trajectory has no energies or it is empty.")


# -----------------------------
# Worker
# -----------------------------
_RELAXER = None


def _init_worker(tf_threads: int, omp_threads: int) -> None:
    """
    Runs once per worker process (spawned).
    Must set env vars and silence TF warnings *inside* the worker.
    """
    import warnings
    import logging as _logging

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(tf_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(tf_threads)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")
    warnings.filterwarnings("ignore", message=".*casting an input of type complex64.*")
    _logging.getLogger("tensorflow").setLevel(_logging.ERROR)

    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        try:
            tf.autograph.set_verbosity(0)
        except Exception:
            pass
    except Exception:
        pass

    global _RELAXER
    from m3gnet.models import Relaxer

    _RELAXER = Relaxer()


def _relax_one_candidate(job: Tuple[str, float, int, int, List[str], bool]) -> Dict[str, Any]:
    """
    job:
      (candidate_dir_str, fmax, tf_threads, omp_threads, order, skip_candidate_if_done)
    """
    cand_path = Path(job[0])
    fmax = float(job[1])
    tf_threads = int(job[2])
    omp_threads = int(job[3])
    order = job[4]
    skip_candidate_if_done = bool(job[5])

    scan_dir = cand_path / "01_scan"
    poscar_in = scan_dir / "POSCAR"
    meta_in = scan_dir / "meta.json"

    out_dir = cand_path / "02_relax"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_out = out_dir / "meta.json"
    if skip_candidate_if_done and meta_out.exists() and (out_dir / "POSCAR").exists():
        # Candidate already done
        meta_prev = _load_json(meta_out) or {}
        return {
            "candidate": cand_path.name,
            "status": "skip",
            "energy_relaxed_eV": meta_prev.get("energy_relaxed_eV", ""),
            "rank_sp": (meta_prev.get("source_scan") or {}).get("rank_sp", ""),
            "energy_sp_eV": (meta_prev.get("source_scan") or {}).get("energy_sp_eV", ""),
            "signature": (meta_prev.get("source_scan") or {}).get("signature", ""),
            "walltime_s": meta_prev.get("walltime_s", ""),
            "note": "already exists",
        }

    meta_scan = _load_json(meta_in) or {}
    if not poscar_in.exists():
        return {"candidate": cand_path.name, "status": "skip", "error": f"missing {poscar_in}"}

    # Ensure worker init happened
    global _RELAXER
    if _RELAXER is None:
        _init_worker(tf_threads=tf_threads, omp_threads=omp_threads)

    t0 = time.time()
    try:
        s0 = Structure.from_file(str(poscar_in))

        res = _RELAXER.relax(s0, fmax=fmax, verbose=False)
        s_rel = res["final_structure"]
        E_rel = _trajectory_last_energy(res)

        _safe_write_poscar(s_rel, out_dir / "POSCAR", order=order)

        meta_relax = {
            "stage": "02_relax",
            "method": "m3gnet Relaxer (pretrained), parallel over candidates",
            "fmax_target_eV_per_A": float(fmax),
            "walltime_s": float(time.time() - t0),
            "energy_relaxed_eV": float(E_rel),
            "species_counts_total": _species_counts(s_rel),
            "source_scan": {
                "rank_sp": meta_scan.get("rank_sp"),
                "energy_sp_eV": meta_scan.get("energy_sp_eV"),
                "signature": meta_scan.get("signature"),
                "eval_order_index": meta_scan.get("eval_order_index"),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_out.write_text(json.dumps(meta_relax, indent=2), encoding="utf-8")

        return {
            "candidate": cand_path.name,
            "status": "ok",
            "energy_relaxed_eV": float(E_rel),
            "rank_sp": meta_scan.get("rank_sp"),
            "energy_sp_eV": meta_scan.get("energy_sp_eV"),
            "signature": meta_scan.get("signature"),
            "walltime_s": meta_relax["walltime_s"],
        }

    except Exception as e:
        meta_fail = {
            "stage": "02_relax",
            "method": "m3gnet Relaxer (pretrained), parallel over candidates",
            "status": "fail",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_out.write_text(json.dumps(meta_fail, indent=2), encoding="utf-8")

        return {
            "candidate": cand_path.name,
            "status": "fail",
            "error": repr(e),
            "rank_sp": meta_scan.get("rank_sp"),
            "energy_sp_eV": meta_scan.get("energy_sp_eV"),
            "signature": meta_scan.get("signature"),
        }


def _write_ranking_csv(folder: Path, rows: List[Dict[str, Any]]) -> Path:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_rows_sorted = sorted(ok_rows, key=lambda r: float(r["energy_relaxed_eV"]))

    for i, r in enumerate(ok_rows_sorted, start=1):
        r["rank_relax"] = i

    ranking_path = folder / "ranking_relax.csv"
    with open(ranking_path, "w", encoding="utf-8") as f:
        f.write("candidate,rank_relax,energy_relaxed_eV,rank_sp,energy_sp_eV,signature,status,walltime_s,error\n")

        # OK rows first (ranked)
        for r in ok_rows_sorted:
            f.write(
                f"{r.get('candidate','')},{r.get('rank_relax','')},{r.get('energy_relaxed_eV','')},"
                f"{r.get('rank_sp','')},{r.get('energy_sp_eV','')},{r.get('signature','')},"
                f"{r.get('status','')},{r.get('walltime_s','')},\n"
            )

        # non-ok rows
        for r in rows:
            if r.get("status") == "ok":
                continue
            f.write(
                f"{r.get('candidate','')},,{r.get('energy_relaxed_eV','')},"
                f"{r.get('rank_sp','')},{r.get('energy_sp_eV','')},{r.get('signature','')},"
                f"{r.get('status','')},{r.get('walltime_s','')},{r.get('error','')}\n"
            )

    return ranking_path


def _run_folder(folder: Path, cfg: RelaxConfig) -> None:
    candidates = sorted([p for p in folder.glob("candidate_*") if p.is_dir()])

    if not candidates:
        log.info("SKIP %s: no candidate_* folders found", folder.name)
        return

    if cfg.skip_if_done and (folder / "ranking_relax.csv").exists():
        log.info("SKIP %s: ranking_relax.csv already exists", folder.name)
        return

    log.info("%s: found %d candidates", folder.name, len(candidates))
    log.info("Relax settings: n_workers=%d fmax=%s tf_threads=%d omp_threads=%d",
             cfg.n_workers, cfg.fmax, cfg.tf_threads, cfg.omp_threads)

    # spawn is safest with TF
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ctx = mp.get_context("spawn")

    jobs = [
        (str(c), cfg.fmax, cfg.tf_threads, cfg.omp_threads, cfg.order, cfg.skip_candidate_if_done)
        for c in candidates
    ]

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=cfg.n_workers, mp_context=ctx,
                             initializer=_init_worker, initargs=(cfg.tf_threads, cfg.omp_threads)) as ex:
        futures = [ex.submit(_relax_one_candidate, j) for j in jobs]

        for fut in as_completed(futures):
            r = fut.result()
            rows.append(r)

            if r.get("status") == "ok":
                log.info("OK   %s/%s  E_rel=%.6f eV  wall=%.1fs",
                         folder.name, r["candidate"], r["energy_relaxed_eV"], float(r.get("walltime_s", 0.0)))
            elif r.get("status") == "skip":
                log.info("SKIP %s/%s  %s", folder.name, r.get("candidate", "?"), r.get("note", ""))
            else:
                log.warning("%s %s/%s  %s",
                            str(r.get("status", "?")).upper(),
                            folder.name, r.get("candidate", "?"),
                            r.get("error", ""))

    ranking_path = _write_ranking_csv(folder, rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_fail = sum(1 for r in rows if r.get("status") == "fail")
    n_skip = sum(1 for r in rows if r.get("status") == "skip")

    log.info("%s: finished in %.1fs | OK=%d FAIL=%d SKIP=%d",
             folder.name, time.time() - t0, n_ok, n_fail, n_skip)
    log.info("%s: wrote %s", folder.name, ranking_path)


# -----------------------------
# Public API
# -----------------------------
def run_relax(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> None:
    """
    Step 03: Relax candidates produced by Step 02.
      For each structure folder in [structure].outdir:
        - relax candidate_*/01_scan/POSCAR in parallel
        - write candidate_*/02_relax/POSCAR and meta.json
        - write ranking_relax.csv per structure folder
    """
    cfg = _parse_relax_config(raw_cfg, root)

    if not cfg.outdir.exists():
        raise FileNotFoundError(f"Output directory not found: {cfg.outdir} (did you run step 01?)")

    folders = sorted([p for p in cfg.outdir.iterdir() if p.is_dir()])

    log.info("Step 03 relax: %d structure folders in: %s", len(folders), cfg.outdir)
    log.info("NOTE: main-directory POSCAR is ignored; only subfolders are processed.")

    for i, folder in enumerate(folders, start=1):
        log.info("RUN (%d/%d) %s", i, len(folders), folder.name)
        _run_folder(folder, cfg)

    log.info("DONE Step 03 relax for all structure folders.")


# TOML wrapper (like your other steps)
try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_relax_from_toml(config_path: Path) -> None:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    run_relax(raw, root, config_path=config_path)

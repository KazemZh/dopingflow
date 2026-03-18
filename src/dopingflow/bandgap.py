# src/dopingflow/bandgap.py
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dopingflow.hardware import resolve_torch_device

log = logging.getLogger(__name__)

# DGL backend must be set before importing DGL/ALIGNN in this process
os.environ.setdefault("DGLBACKEND", "pytorch")

# ---- Fixed I/O names (do not expose to users) ----
RELAX_POSCAR_REL = "02_relax/POSCAR"
CAND_LIST_NAME = "selected_candidates.txt"
OUT_CSV_NAME = "bandgap_alignn_summary.csv"
BAND_META_REL = "03_band/meta.json"


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class BandgapConfig:
    outdir: Path
    skip_if_done: bool
    cutoff: float
    max_neighbors: int
    n_workers: int
    device: str
    gpu_id: int
    batch_size: int


def _parse_bandgap_config(raw: dict[str, Any], root: Path) -> BandgapConfig:
    st = raw.get("structure", {}) or {}
    bg = raw.get("bandgap", {}) or {}

    outdir_name = str(st.get("outdir", "random_structures"))
    outdir = (root / outdir_name).resolve()

    skip_if_done = bool(bg.get("skip_if_done", True))
    cutoff = float(bg.get("cutoff", 8.0))
    max_neighbors = int(bg.get("max_neighbors", 12))
    n_workers = int(bg.get("n_workers", 1))
    device = str(bg.get("device", "cpu")).lower()
    gpu_id = int(bg.get("gpu_id", 0))
    batch_size = int(bg.get("batch_size", 32))

    if cutoff <= 0:
        raise ValueError("[bandgap].cutoff must be > 0")
    if max_neighbors <= 0:
        raise ValueError("[bandgap].max_neighbors must be > 0")
    if n_workers <= 0:
        raise ValueError("[bandgap].n_workers must be >= 1")
    if device not in {"cpu", "cuda"}:
        raise ValueError('[bandgap].device must be either "cpu" or "cuda"')
    if gpu_id < 0:
        raise ValueError("[bandgap].gpu_id must be >= 0")
    if batch_size <= 0:
        raise ValueError("[bandgap].batch_size must be >= 1")

    return BandgapConfig(
        outdir=outdir,
        skip_if_done=skip_if_done,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        n_workers=n_workers,
        device=device,
        gpu_id=gpu_id,
        batch_size=batch_size,
    )


# -----------------------------
# ALIGNN helpers
# -----------------------------
def _poscar_to_jarvis_atoms(poscar_path: Path):
    from jarvis.core.atoms import Atoms
    from pymatgen.core import Structure

    s = Structure.from_file(str(poscar_path))
    return Atoms(
        lattice_mat=s.lattice.matrix,
        coords=s.frac_coords,
        elements=[str(sp) for sp in s.species],
        cartesian=False,
    )


def _find_model_dir(root: Path) -> Path:
    if not root.exists():
        raise RuntimeError(f"Model root folder not found: {root}")

    cfgs = list(root.rglob("config.json"))
    if not cfgs:
        raise RuntimeError(f"Could not find config.json under {root}")

    for cfg in cfgs:
        md = cfg.parent
        ckpts = list(md.glob("checkpoint_*.pt")) + list(md.glob("*.pt"))
        if ckpts:
            return md

    return cfgs[0].parent


def _load_local_alignn_model(model_dir: Path, device_str: str):
    import torch
    from alignn.config import ALIGNNConfig
    from alignn.models.alignn import ALIGNN

    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"config.json not found in {model_dir}")

    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_cfg = cfg_dict.get("model", cfg_dict)
    if not isinstance(model_cfg, dict):
        raise RuntimeError("config.json has unexpected format: no 'model' dict found.")

    model_cfg.setdefault("name", "alignn")
    cfg = ALIGNNConfig(**model_cfg)

    ckpts = sorted(model_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        ckpts = sorted(model_dir.glob("*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoint_*.pt or *.pt found in {model_dir}")
    ckpt = ckpts[-1]

    device = torch.device(device_str)

    log.info("ALIGNN config : %s", cfg_path)
    log.info("ALIGNN ckpt   : %s", ckpt)
    log.info("ALIGNN device : %s", device)

    model = ALIGNN(cfg)
    state = torch.load(str(ckpt), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model, cfg_path, ckpt, device


def _build_alignn_inputs(jarvis_atoms, cutoff: float, max_neighbors: int):
    import torch
    from alignn.graphs import Graph

    out_graph = Graph.atom_dgl_multigraph(
        jarvis_atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )

    if isinstance(out_graph, (list, tuple)) and len(out_graph) == 3:
        g, lg, lat = out_graph
    elif isinstance(out_graph, (list, tuple)) and len(out_graph) == 2:
        g, lg = out_graph
        lat = torch.tensor(jarvis_atoms.lattice_mat, dtype=torch.float32).unsqueeze(0)
    else:
        raise RuntimeError(f"Unexpected graph return type/length: {type(out_graph)}")

    return g, lg, lat


def _predict_bandgap(model, jarvis_atoms, cutoff: float, max_neighbors: int) -> float:
    import torch

    g, lg, lat = _build_alignn_inputs(jarvis_atoms, cutoff=cutoff, max_neighbors=max_neighbors)

    device = next(model.parameters()).device
    g = g.to(device)
    lg = lg.to(device)
    lat = lat.to(device)

    with torch.no_grad():
        pred = model((g, lg, lat))

    return float(torch.as_tensor(pred).detach().cpu().reshape(-1)[0])


def _predict_bandgap_batch(model, items):
    import dgl
    import torch

    device = next(model.parameters()).device

    gs, lgs, lats = zip(*items)
    bg = dgl.batch(gs).to(device)
    blg = dgl.batch(lgs).to(device)
    blat = torch.cat(lats, dim=0).to(device)

    with torch.no_grad():
        pred = model((bg, blg, blat))

    return torch.as_tensor(pred).detach().cpu().reshape(-1).tolist()


def _load_selected_candidates(path: Path) -> List[str]:
    out: List[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def _write_band_meta(candidate_dir: Path, bandgap_eV: float, extra: Dict[str, Any]) -> None:
    meta_path = candidate_dir / BAND_META_REL
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "stage": "03_band",
        "property": "bandgap",
        "bandgap_eV_ALIGNN_MBJ": float(bandgap_eV),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# -----------------------------
# Worker for CPU parallel mode
# -----------------------------
def _cpu_predict_one(
    poscar_path_str: str,
    cutoff: float,
    max_neighbors: int,
    model_dir_str: str,
) -> Dict[str, Any]:
    poscar_path = Path(poscar_path_str)
    cand_dir = poscar_path.parents[1]
    cand_name = cand_dir.name

    try:
        jat = _poscar_to_jarvis_atoms(poscar_path)
        t0 = time.time()

        model, cfg_path, ckpt_path, _device = _load_local_alignn_model(Path(model_dir_str), "cpu")
        bg = _predict_bandgap(model, jat, cutoff=cutoff, max_neighbors=max_neighbors)
        wall_s = time.time() - t0

        return {
            "candidate": cand_name,
            "bandgap": float(bg),
            "status": "ok",
            "extra": {
                "status": "ok",
                "model_key": "jv_mbj_bandgap_alignn (local)",
                "model_dir": str(model_dir_str),
                "config_json": str(cfg_path),
                "checkpoint": str(ckpt_path),
                "input_poscar": str(poscar_path),
                "n_atoms": int(len(jat.elements)),
                "composition": dict(jat.composition.to_dict()),
                "graph_params": {
                    "cutoff": float(cutoff),
                    "max_neighbors": int(max_neighbors),
                },
                "walltime_s": float(wall_s),
            },
        }
    except Exception as e:
        return {
            "candidate": cand_name,
            "bandgap": float("nan"),
            "status": "fail",
            "extra": {
                "status": "fail",
                "error": repr(e),
                "input_poscar": str(poscar_path),
                "graph_params": {
                    "cutoff": float(cutoff),
                    "max_neighbors": int(max_neighbors),
                },
                "model_key": "jv_mbj_bandgap_alignn (local)",
                "model_dir": str(model_dir_str),
            },
        }


# -----------------------------
# Per-folder processing
# -----------------------------
def _get_relaxed_poscars(folder: Path) -> List[Path]:
    cand_list = folder / CAND_LIST_NAME
    if cand_list.exists():
        cand_names = _load_selected_candidates(cand_list)
        poscars = [folder / c / RELAX_POSCAR_REL for c in cand_names]
        poscars = [p for p in poscars if p.exists()]
        log.info("SELECT %s: using %d relaxed POSCARs from %s", folder.name, len(poscars), CAND_LIST_NAME)
        return poscars

    poscars = sorted(folder.glob(f"candidate_*/{RELAX_POSCAR_REL}"))
    log.info("SELECT %s: using glob: %d relaxed POSCARs", folder.name, len(poscars))
    return poscars


def _write_summary_csv(out_csv: Path, rows: List[Tuple[str, float]]) -> None:
    rows_sorted = sorted(rows, key=lambda x: (math.isnan(x[1]), x[1]))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "bandgap_eV_ALIGNN_MBJ"])
        for cand, bg in rows_sorted:
            w.writerow([cand, f"{bg:.6f}"])


def _run_folder(folder: Path, cfg: BandgapConfig, model, cfg_path: Path | None, ckpt_path: Path | None, model_dir: Path) -> None:
    out_csv = folder / OUT_CSV_NAME

    if cfg.skip_if_done and out_csv.exists():
        log.info("SKIP %s: %s already exists", folder.name, OUT_CSV_NAME)
        return

    poscars = _get_relaxed_poscars(folder)
    if not poscars:
        log.info("SKIP %s: no relaxed POSCARs found (run Step 03; Step 04 optional)", folder.name)
        return

    rows: List[Tuple[str, float]] = []
    todo_poscars: List[Path] = []

    for poscar_path in poscars:
        cand_dir = poscar_path.parents[1]
        cand_name = cand_dir.name
        meta_path = cand_dir / BAND_META_REL

        if cfg.skip_if_done and meta_path.exists():
            try:
                prev = json.loads(meta_path.read_text(encoding="utf-8"))
                bg_prev = float(prev.get("bandgap_eV_ALIGNN_MBJ", float("nan")))
                rows.append((cand_name, bg_prev))
                log.info("%s/%s: SKIP bandgap (meta exists) bg=%.4f eV", folder.name, cand_name, bg_prev)
                continue
            except Exception:
                log.info("%s/%s: meta exists but unreadable -> recompute", folder.name, cand_name)

        todo_poscars.append(poscar_path)

    if not todo_poscars:
        _write_summary_csv(out_csv, rows)
        log.info("OK   %s: wrote %s and candidate_*/03_band/meta.json", folder.name, OUT_CSV_NAME)
        return

    # -------------------------
    # CPU parallel mode
    # -------------------------
    if cfg.device == "cpu" and cfg.n_workers > 1 and len(todo_poscars) > 1:
        log.info("%s: running CPU parallel bandgap with %d workers", folder.name, cfg.n_workers)

        futures = []
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
            for poscar_path in todo_poscars:
                futures.append(
                    ex.submit(
                        _cpu_predict_one,
                        str(poscar_path),
                        cfg.cutoff,
                        cfg.max_neighbors,
                        str(model_dir),
                    )
                )

            for fut in as_completed(futures):
                res = fut.result()
                cand_name = res["candidate"]
                bg = float(res["bandgap"])
                status = res["status"]
                extra = res["extra"]

                cand_dir = folder / cand_name

                if status == "ok":
                    log.info("%s/%s: bandgap = %.4f eV", folder.name, cand_name, bg)
                else:
                    log.warning("%s/%s: FAIL bandgap: %s", folder.name, cand_name, extra.get("error", "unknown"))

                _write_band_meta(cand_dir, bg, extra)
                rows.append((cand_name, bg))

    # -------------------------
    # CUDA batched mode
    # -------------------------
    elif cfg.device == "cuda":
        log.info("%s: running CUDA batched bandgap (batch_size=%d)", folder.name, cfg.batch_size)

        batch_items: List[Tuple[Any, Any, Any]] = []
        batch_meta: List[Tuple[Path, Any]] = []

        for poscar_path in todo_poscars:
            try:
                jat = _poscar_to_jarvis_atoms(poscar_path)
                g, lg, lat = _build_alignn_inputs(jat, cutoff=cfg.cutoff, max_neighbors=cfg.max_neighbors)
                batch_items.append((g, lg, lat))
                batch_meta.append((poscar_path, jat))
            except Exception as e:
                cand_dir = poscar_path.parents[1]
                cand_name = cand_dir.name
                log.warning("%s/%s: FAIL bandgap input build: %s", folder.name, cand_name, repr(e))
                _write_band_meta(
                    cand_dir,
                    float("nan"),
                    {
                        "status": "fail",
                        "error": repr(e),
                        "input_poscar": str(poscar_path),
                        "graph_params": {
                            "cutoff": float(cfg.cutoff),
                            "max_neighbors": int(cfg.max_neighbors),
                        },
                        "model_key": "jv_mbj_bandgap_alignn (local)",
                        "model_dir": str(model_dir),
                        "config_json": str(cfg_path) if cfg_path else "",
                        "checkpoint": str(ckpt_path) if ckpt_path else "",
                    },
                )
                rows.append((cand_name, float("nan")))
                continue

            if len(batch_items) == cfg.batch_size:
                try:
                    preds = _predict_bandgap_batch(model, batch_items)

                    for (poscar_path_i, jat_i), bg in zip(batch_meta, preds):
                        cand_dir = poscar_path_i.parents[1]
                        cand_name = cand_dir.name

                        log.info("%s/%s: bandgap = %.4f eV", folder.name, cand_name, bg)

                        extra = {
                            "status": "ok",
                            "model_key": "jv_mbj_bandgap_alignn (local)",
                            "model_dir": str(model_dir),
                            "config_json": str(cfg_path) if cfg_path else "",
                            "checkpoint": str(ckpt_path) if ckpt_path else "",
                            "input_poscar": str(poscar_path_i),
                            "n_atoms": int(len(jat_i.elements)),
                            "composition": dict(jat_i.composition.to_dict()),
                            "graph_params": {
                                "cutoff": float(cfg.cutoff),
                                "max_neighbors": int(cfg.max_neighbors),
                            },
                        }
                        _write_band_meta(cand_dir, float(bg), extra)
                        rows.append((cand_name, float(bg)))

                except Exception as e:
                    for poscar_path_i, _jat_i in batch_meta:
                        cand_dir = poscar_path_i.parents[1]
                        cand_name = cand_dir.name
                        log.warning("%s/%s: FAIL bandgap batch: %s", folder.name, cand_name, repr(e))
                        _write_band_meta(
                            cand_dir,
                            float("nan"),
                            {
                                "status": "fail",
                                "error": repr(e),
                                "input_poscar": str(poscar_path_i),
                                "graph_params": {
                                    "cutoff": float(cfg.cutoff),
                                    "max_neighbors": int(cfg.max_neighbors),
                                },
                                "model_key": "jv_mbj_bandgap_alignn (local)",
                                "model_dir": str(model_dir),
                                "config_json": str(cfg_path) if cfg_path else "",
                                "checkpoint": str(ckpt_path) if ckpt_path else "",
                            },
                        )
                        rows.append((cand_name, float("nan")))

                batch_items = []
                batch_meta = []

        # Flush remaining batch
        if batch_items:
            try:
                preds = _predict_bandgap_batch(model, batch_items)

                for (poscar_path_i, jat_i), bg in zip(batch_meta, preds):
                    cand_dir = poscar_path_i.parents[1]
                    cand_name = cand_dir.name

                    log.info("%s/%s: bandgap = %.4f eV", folder.name, cand_name, bg)

                    extra = {
                        "status": "ok",
                        "model_key": "jv_mbj_bandgap_alignn (local)",
                        "model_dir": str(model_dir),
                        "config_json": str(cfg_path) if cfg_path else "",
                        "checkpoint": str(ckpt_path) if ckpt_path else "",
                        "input_poscar": str(poscar_path_i),
                        "n_atoms": int(len(jat_i.elements)),
                        "composition": dict(jat_i.composition.to_dict()),
                        "graph_params": {
                            "cutoff": float(cfg.cutoff),
                            "max_neighbors": int(cfg.max_neighbors),
                        },
                    }
                    _write_band_meta(cand_dir, float(bg), extra)
                    rows.append((cand_name, float(bg)))

            except Exception as e:
                for poscar_path_i, _jat_i in batch_meta:
                    cand_dir = poscar_path_i.parents[1]
                    cand_name = cand_dir.name
                    log.warning("%s/%s: FAIL bandgap batch: %s", folder.name, cand_name, repr(e))
                    _write_band_meta(
                        cand_dir,
                        float("nan"),
                        {
                            "status": "fail",
                            "error": repr(e),
                            "input_poscar": str(poscar_path_i),
                            "graph_params": {
                                "cutoff": float(cfg.cutoff),
                                "max_neighbors": int(cfg.max_neighbors),
                            },
                            "model_key": "jv_mbj_bandgap_alignn (local)",
                            "model_dir": str(model_dir),
                            "config_json": str(cfg_path) if cfg_path else "",
                            "checkpoint": str(ckpt_path) if ckpt_path else "",
                        },
                    )
                    rows.append((cand_name, float("nan")))

    # -------------------------
    # CPU serial mode
    # -------------------------
    else:
        for poscar_path in todo_poscars:
            cand_dir = poscar_path.parents[1]
            cand_name = cand_dir.name

            try:
                jat = _poscar_to_jarvis_atoms(poscar_path)

                t0 = time.time()
                bg = _predict_bandgap(model, jat, cutoff=cfg.cutoff, max_neighbors=cfg.max_neighbors)
                wall_s = time.time() - t0

                log.info("%s/%s: bandgap = %.4f eV", folder.name, cand_name, bg)

                extra = {
                    "status": "ok",
                    "model_key": "jv_mbj_bandgap_alignn (local)",
                    "model_dir": str(model_dir),
                    "config_json": str(cfg_path) if cfg_path else "",
                    "checkpoint": str(ckpt_path) if ckpt_path else "",
                    "input_poscar": str(poscar_path),
                    "n_atoms": int(len(jat.elements)),
                    "composition": dict(jat.composition.to_dict()),
                    "graph_params": {
                        "cutoff": float(cfg.cutoff),
                        "max_neighbors": int(cfg.max_neighbors),
                    },
                    "walltime_s": float(wall_s),
                }
                _write_band_meta(cand_dir, bg, extra)
                rows.append((cand_name, bg))

            except Exception as e:
                log.warning("%s/%s: FAIL bandgap: %s", folder.name, cand_name, repr(e))
                _write_band_meta(
                    cand_dir,
                    float("nan"),
                    {
                        "status": "fail",
                        "error": repr(e),
                        "input_poscar": str(poscar_path),
                        "graph_params": {
                            "cutoff": float(cfg.cutoff),
                            "max_neighbors": int(cfg.max_neighbors),
                        },
                        "model_key": "jv_mbj_bandgap_alignn (local)",
                        "model_dir": str(model_dir),
                        "config_json": str(cfg_path) if cfg_path else "",
                        "checkpoint": str(ckpt_path) if ckpt_path else "",
                    },
                )
                rows.append((cand_name, float("nan")))

    _write_summary_csv(out_csv, rows)
    log.info("OK   %s: wrote %s and candidate_*/03_band/meta.json", folder.name, OUT_CSV_NAME)


# -----------------------------
# Public API
# -----------------------------
def run_bandgap(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> None:
    """
    Step 05: Predict bandgap (ALIGNN local model) for relaxed candidates.

    Selection:
      1) If selected_candidates.txt exists in a composition folder -> use it
      2) Else fallback to candidate_*/02_relax/POSCAR

    Outputs per composition folder:
      - bandgap_alignn_summary.csv
      - candidate_*/03_band/meta.json
    """
    cfg = _parse_bandgap_config(raw_cfg, root)
    device_str = resolve_torch_device(cfg.device, cfg.gpu_id)

    model_root_env = os.environ.get("ALIGNN_MODEL_DIR", "").strip()
    if not model_root_env:
        raise RuntimeError("Missing env var ALIGNN_MODEL_DIR (points to your ALIGNN model folder).")

    model_root = Path(model_root_env)
    model_dir = _find_model_dir(model_root)

    # Load model once only for CPU serial / CUDA mode
    model = None
    cfg_path = None
    ckpt_path = None
    device = cfg.device

    if cfg.device == "cuda" or cfg.n_workers == 1:
        model, cfg_path, ckpt_path, device = _load_local_alignn_model(model_dir, device_str)

    if not cfg.outdir.exists():
        raise FileNotFoundError(f"Output directory not found: {cfg.outdir} (did you run Step 01?)")

    folders = sorted([p for p in cfg.outdir.iterdir() if p.is_dir()])

    log.info("Step 05 bandgap: scanning %d structure folders in: %s", len(folders), cfg.outdir)
    log.info("Bandgap device: %s", cfg.device)
    log.info("Bandgap CPU workers: %d", cfg.n_workers)
    log.info("Bandgap batch size: %d", cfg.batch_size)
    log.info("Output per folder: %s", OUT_CSV_NAME)
    log.info("NOTE: main-directory files are ignored; only subfolders are processed.")

    for i, folder in enumerate(folders, start=1):
        log.info("RUN (%d/%d) %s", i, len(folders), folder.name)
        _run_folder(folder, cfg, model, cfg_path, ckpt_path, model_dir)

    log.info("DONE Step 05 bandgap for all structure folders.")


# TOML wrapper
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_bandgap_from_toml(config_path: Path) -> None:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    run_bandgap(raw, root, config_path=config_path)
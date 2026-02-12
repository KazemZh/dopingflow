# src/dopingflow/bandgap.py
from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _parse_bandgap_config(raw: dict[str, Any], root: Path) -> BandgapConfig:
    st = raw.get("structure", {}) or {}
    bg = raw.get("bandgap", {}) or {}

    outdir_name = str(st.get("outdir", "random_structures"))
    outdir = (root / outdir_name).resolve()

    skip_if_done = bool(bg.get("skip_if_done", True))
    cutoff = float(bg.get("cutoff", 8.0))
    max_neighbors = int(bg.get("max_neighbors", 12))

    if cutoff <= 0:
        raise ValueError("[bandgap].cutoff must be > 0")
    if max_neighbors <= 0:
        raise ValueError("[bandgap].max_neighbors must be > 0")

    return BandgapConfig(
        outdir=outdir,
        skip_if_done=skip_if_done,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )


# -----------------------------
# ALIGNN helpers (ported)
# -----------------------------
def _poscar_to_jarvis_atoms(poscar_path: Path):
    from pymatgen.core import Structure
    from jarvis.core.atoms import Atoms

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


def _load_local_alignn_model(model_dir: Path):
    import torch
    from alignn.models.alignn import ALIGNN
    from alignn.config import ALIGNNConfig

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

    log.info("ALIGNN config: %s", cfg_path)
    log.info("ALIGNN ckpt  : %s", ckpt)

    model = ALIGNN(cfg)
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg_path, ckpt


def _predict_bandgap(model, jarvis_atoms, cutoff: float, max_neighbors: int) -> float:
    import torch
    from alignn.graphs import Graph

    out_graph = Graph.atom_dgl_multigraph(jarvis_atoms, cutoff=cutoff, max_neighbors=max_neighbors)

    if isinstance(out_graph, (list, tuple)) and len(out_graph) == 3:
        g, lg, lat = out_graph
    elif isinstance(out_graph, (list, tuple)) and len(out_graph) == 2:
        g, lg = out_graph
        lat = torch.tensor(jarvis_atoms.lattice_mat, dtype=torch.float32).unsqueeze(0)
    else:
        raise RuntimeError(f"Unexpected graph return type/length: {type(out_graph)}")

    device = next(model.parameters()).device
    g = g.to(device)
    lg = lg.to(device)
    lat = lat.to(device)

    with torch.no_grad():
        pred = model((g, lg, lat))

    return float(pred.detach().cpu().numpy().reshape(-1)[0])


def _load_selected_candidates(path: Path) -> List[str]:
    out: List[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def _write_band_meta(candidate_dir: Path, bandgap_eV: float, extra: Dict[str, Any]) -> None:
    out_dir = candidate_dir / "03_band"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "stage": "03_band",
        "property": "bandgap",
        "bandgap_eV_ALIGNN_MBJ": float(bandgap_eV),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


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


def _run_folder(folder: Path, cfg: BandgapConfig, model, cfg_path: Path, ckpt_path: Path, model_dir: Path) -> None:
    out_csv = folder / OUT_CSV_NAME

    if cfg.skip_if_done and out_csv.exists():
        log.info("SKIP %s: %s already exists", folder.name, OUT_CSV_NAME)
        return

    poscars = _get_relaxed_poscars(folder)
    if not poscars:
        log.info("SKIP %s: no relaxed POSCARs found (run Step 03; Step 04 optional)", folder.name)
        return

    rows: List[Tuple[str, float]] = []

    for poscar_path in poscars:
        cand_dir = poscar_path.parents[1]  # candidate_XXX
        cand_name = cand_dir.name

        jat = _poscar_to_jarvis_atoms(poscar_path)

        t0 = time.time()
        bg = _predict_bandgap(model, jat, cutoff=cfg.cutoff, max_neighbors=cfg.max_neighbors)
        wall_s = time.time() - t0

        log.info("%s/%s: bandgap = %.4f eV", folder.name, cand_name, bg)

        extra = {
            "model_key": "jv_mbj_bandgap_alignn (local)",
            "model_dir": str(model_dir),
            "config_json": str(cfg_path),
            "checkpoint": str(ckpt_path),
            "input_poscar": str(poscar_path),
            "n_atoms": int(len(jat.elements)),
            "composition": dict(jat.composition.to_dict()),
            "graph_params": {"cutoff": float(cfg.cutoff), "max_neighbors": int(cfg.max_neighbors)},
            "walltime_s": float(wall_s),
        }
        _write_band_meta(cand_dir, bg, extra)
        rows.append((cand_name, bg))

    rows_sorted = sorted(rows, key=lambda x: x[1])
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "bandgap_eV_ALIGNN_MBJ"])
        for cand, bg in rows_sorted:
            w.writerow([cand, f"{bg:.6f}"])

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

    model_root_env = os.environ.get("ALIGNN_MODEL_DIR", "").strip()
    if not model_root_env:
        raise RuntimeError("Missing env var ALIGNN_MODEL_DIR (points to your ALIGNN model folder).")

    model_root = Path(model_root_env)
    model_dir = _find_model_dir(model_root)
    model, cfg_path, ckpt_path = _load_local_alignn_model(model_dir)

    if not cfg.outdir.exists():
        raise FileNotFoundError(f"Output directory not found: {cfg.outdir} (did you run Step 01?)")

    folders = sorted([p for p in cfg.outdir.iterdir() if p.is_dir()])

    log.info("Step 05 bandgap: scanning %d structure folders in: %s", len(folders), cfg.outdir)
    log.info("Output per folder: %s", OUT_CSV_NAME)
    log.info("NOTE: main-directory files are ignored; only subfolders are processed.")

    for i, folder in enumerate(folders, start=1):
        log.info("RUN (%d/%d) %s", i, len(folders), folder.name)
        _run_folder(folder, cfg, model, cfg_path, ckpt_path, model_dir)

    log.info("DONE Step 05 bandgap for all structure folders.")


# TOML wrapper (like your other steps)
try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_bandgap_from_toml(config_path: Path) -> None:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    run_bandgap(raw, root, config_path=config_path)

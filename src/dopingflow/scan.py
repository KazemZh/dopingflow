from __future__ import annotations

import heapq
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

log = logging.getLogger(__name__)

# -----------------------------
# TF/M3GNet noise suppression
# (must be set before importing TF/m3gnet in this process)
# -----------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")


# -----------------------------
# Config model
# -----------------------------
@dataclass(frozen=True)
class ScanConfig:
    poscar_in: str
    topk: int
    symprec: float
    max_enum: int
    nproc: int
    chunksize: int
    order: List[str]  # from [generate].poscar_order
    anion_species: List[str]
    host_species: str
    max_unique: int
    skip_if_done: bool


def _parse_scan_config(raw: dict[str, Any]) -> ScanConfig:
    scan = raw.get("scan", {}) or {}
    gen = raw.get("generate", {}) or {}
    dop = raw.get("doping", {}) or {}

    poscar_in = str(scan.get("poscar_in", "POSCAR"))
    topk = int(scan.get("topk", 15))
    symprec = float(scan.get("symprec", 1e-3))
    max_enum = int(scan.get("max_enum", 50_000_000))
    nproc = int(scan.get("nproc", 12))
    chunksize = int(scan.get("chunksize", 50))
    anion_species = [str(x) for x in (scan.get("anion_species", ["O"]) or [])]
    host_species = str(dop.get("host_species", "")).strip()
    max_unique = int(scan.get("max_unique", 200_000))
    skip_if_done = bool(scan.get("skip_if_done", True))

    # Order comes from [generate] ONLY (single source of truth)
    order = [str(x) for x in (gen.get("poscar_order", []) or [])]

    if topk <= 0:
        raise ValueError("[scan].topk must be > 0")
    if symprec <= 0:
        raise ValueError("[scan].symprec must be > 0")
    if max_enum <= 0:
        raise ValueError("[scan].max_enum must be > 0")
    if nproc <= 0:
        raise ValueError("[scan].nproc must be > 0")
    if chunksize <= 0:
        raise ValueError("[scan].chunksize must be > 0")
    if not order:
        raise ValueError("[generate].poscar_order must be defined and non-empty")
    if not host_species:
        raise ValueError("[doping].host_species is required")
    if not anion_species:
        raise ValueError("[scan].anion_species must be non-empty")
    if max_unique <= 0:
        raise ValueError("[scan].max_unique must be > 0")

    return ScanConfig(
        poscar_in=poscar_in,
        topk=topk,
        symprec=symprec,
        max_enum=max_enum,
        nproc=nproc,
        chunksize=chunksize,
        order=order,
        anion_species=anion_species,
        host_species=host_species,
        max_unique=max_unique,
        skip_if_done=skip_if_done,
    )


def _get_outdir(raw: dict[str, Any], root: Path) -> Path:
    st = raw.get("structure", {}) or {}
    outdir_name = str(st.get("outdir", "random_structures"))
    return (root / outdir_name).resolve()


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
    return Structure(
        struct.lattice,
        [s.species for s in sites_sorted],
        [s.frac_coords for s in sites_sorted],
    )


def _infer_enumeration_sublattice(
    struct: Structure,
    host: str,
    anions: List[str],
) -> Tuple[List[int], Dict[str, int], int]:
    """
    Enumerated sublattice = all sites whose species_string NOT in anions.
    Infer dopant counts from the input POSCAR in each structure folder.
    """
    sub_idx: List[int] = []
    counts = Counter()

    for i, site in enumerate(struct):
        el = site.species_string
        if el in anions:
            continue
        sub_idx.append(i)
        counts[el] += 1

    if host not in counts:
        raise ValueError(f"Host '{host}' not found on enumerated sublattice. Found: {dict(counts)}")

    dopant_counts = {el: c for el, c in counts.items() if el != host}
    host_count = counts[host]
    return sub_idx, dopant_counts, host_count


def _estimate_num_configs(n_sites: int, dopant_counts: Dict[str, int]) -> int:
    remaining = n_sites
    total = 1
    for _, cnt in dopant_counts.items():
        total *= math.comb(remaining, cnt)
        remaining -= cnt
    return total


def _make_parent_structure(base: Structure, sublattice_indices: List[int], host: str) -> Structure:
    parent = base.copy()
    for idx in sublattice_indices:
        parent[idx] = host
    return parent


def _build_symmetry_permutations(parent: Structure, sublattice_indices: List[int], symprec: float) -> List[np.ndarray]:
    sga = SpacegroupAnalyzer(parent, symprec=symprec)
    ops = sga.get_symmetry_operations(cartesian=False)

    frac = np.array([parent[i].frac_coords for i in sublattice_indices], dtype=float)
    N = frac.shape[0]

    def match_index(coord):
        c = coord % 1.0
        d = frac - c
        d -= np.round(d)
        dist2 = np.sum(d * d, axis=1)
        j = int(np.argmin(dist2))
        if dist2[j] > (symprec * 10) ** 2:
            raise RuntimeError(f"Failed to match symmetry-mapped site (min dist^2={dist2[j]}).")
        return j

    perms = []
    for op in ops:
        R = np.array(op.rotation_matrix, dtype=float)
        t = np.array(op.translation_vector, dtype=float)

        perm = np.empty(N, dtype=np.int32)
        for i in range(N):
            r_new = R.dot(frac[i]) + t
            perm[i] = match_index(r_new)
        perms.append(perm)

    # unique perms
    uniq, seen = [], set()
    for p in perms:
        b = p.tobytes()
        if b not in seen:
            seen.add(b)
            uniq.append(p)
    return uniq


def _canonical_key(labels: np.ndarray, perms: List[np.ndarray]) -> bytes:
    best = None
    for perm in perms:
        img = labels[perm]
        b = img.tobytes()
        if best is None or b < best:
            best = b
    assert best is not None
    return best


def _enumerate_label_configs(N: int, dopant_label_counts: Dict[int, int]):
    """
    Supports up to 3 dopants total.
    labels are int array length N:
      0 -> host
      1.. -> dopants
    """
    items = list(dopant_label_counts.items())
    allpos = list(range(N))

    if len(items) == 0:
        yield np.zeros(N, dtype=np.int8)
        return

    if len(items) == 1:
        (l1, c1) = items[0]
        for A in combinations(allpos, c1):
            labels = np.zeros(N, dtype=np.int8)
            labels[list(A)] = l1
            yield labels
        return

    if len(items) == 2:
        (l1, c1), (l2, c2) = items
        for A in combinations(allpos, c1):
            rem1 = [p for p in allpos if p not in A]
            for B in combinations(rem1, c2):
                labels = np.zeros(N, dtype=np.int8)
                labels[list(A)] = l1
                labels[list(B)] = l2
                yield labels
        return

    if len(items) == 3:
        (l1, c1), (l2, c2), (l3, c3) = items
        for A in combinations(allpos, c1):
            rem1 = [p for p in allpos if p not in A]
            for B in combinations(rem1, c2):
                rem2 = [p for p in rem1 if p not in B]
                for C in combinations(rem2, c3):
                    labels = np.zeros(N, dtype=np.int8)
                    labels[list(A)] = l1
                    labels[list(B)] = l2
                    labels[list(C)] = l3
                    yield labels
        return

    raise ValueError("This enumerator supports up to 3 dopants total.")


def _dopant_signature_from_labels(labels: np.ndarray, label_to_el: Dict[int, str]) -> str:
    parts = []
    for pos, lab in enumerate(labels):
        lab = int(lab)
        if lab == 0:
            continue
        parts.append(f"{label_to_el[lab]}{pos}")
    return "_".join(parts) if parts else "pristine"


# -----------------------------
# Multiprocessing worker (top-level)
# -----------------------------
_MODEL = None


def _init_worker():
    # --- suppress TF warnings inside each worker process ---
    import os
    import logging
    import warnings

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")
    warnings.filterwarnings("ignore", message=".*casting an input of type complex64.*")

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        try:
            tf.autograph.set_verbosity(0)
        except Exception:
            pass
    except Exception:
        pass
    
    global _MODEL
    from m3gnet.models import M3GNet

    _MODEL = M3GNet.load()


def _energy_worker_with_labels(args):
    """
    args: (base_dict, sublattice_indices, labels, label_to_el)
    returns: (E, labels)
    """
    global _MODEL
    base_dict, sublattice_indices, labels, label_to_el = args
    base_local = Structure.from_dict(base_dict)

    s = base_local.copy()
    for pos, site_index in enumerate(sublattice_indices):
        s[site_index] = label_to_el[int(labels[pos])]

    E = float(_MODEL.predict_structure(s))
    return (E, labels)


# -----------------------------
# Core: scan one structure folder
# -----------------------------
def _scan_one_folder(struct_dir: Path, cfg: ScanConfig) -> None:
    poscar_path = struct_dir / cfg.poscar_in
    if not poscar_path.exists():
        raise FileNotFoundError(f"Missing {cfg.poscar_in} in {struct_dir}")

    base = Structure.from_file(str(poscar_path))

    sub_idx, dopant_counts, host_count = _infer_enumeration_sublattice(
        base, host=cfg.host_species, anions=cfg.anion_species
    )

    N = len(sub_idx)
    raw_ncfg = _estimate_num_configs(N, dopant_counts)

    log.info("CWD: %s", struct_dir)
    log.info("Enumerated sublattice size: %d (host=%s, anions=%s)", N, cfg.host_species, cfg.anion_species)
    log.info("Dopant counts inferred on sublattice: %s", dopant_counts)
    log.info("Estimated raw configurations: %d", raw_ncfg)

    if raw_ncfg > cfg.max_enum:
        raise RuntimeError(
            f"Too many raw configs ({raw_ncfg}) > max_enum ({cfg.max_enum}). "
            "Reduce dopant counts / use sampling, or increase max_enum carefully."
        )

    if host_count + sum(dopant_counts.values()) != N:
        raise RuntimeError("Counts inconsistent with sublattice size.")

    parent = _make_parent_structure(base, sub_idx, host=cfg.host_species)
    perms = _build_symmetry_permutations(parent, sub_idx, symprec=cfg.symprec)
    log.info("Symmetry operations (unique permutations on sublattice): %d", len(perms))

    # Label mapping (stable)
    dopants_sorted = sorted(dopant_counts.items())
    label_to_el: Dict[int, str] = {0: cfg.host_species}
    dopant_label_counts: Dict[int, int] = {}
    for lab, (el, cnt) in enumerate(dopants_sorted, start=1):
        label_to_el[lab] = el
        dopant_label_counts[lab] = int(cnt)

    log.info("Label map: %s", label_to_el)
    log.info("Label counts: %s", dopant_label_counts)

    # Generate symmetry-unique configs
    seen = set()
    unique_labels: List[np.ndarray] = []
    checked_raw = 0

    for labels in _enumerate_label_configs(N, dopant_label_counts):
        checked_raw += 1
        key = _canonical_key(labels, perms)
        if key in seen:
            continue
        seen.add(key)
        unique_labels.append(labels.copy())

        if len(unique_labels) >= cfg.max_unique:
            raise RuntimeError(
                f"Unique(sym) configs reached max_unique={cfg.max_unique}. "
                "This composition is too large for full enumeration."
            )

    log.info("Done generating: raw=%d, unique(sym)=%d", checked_raw, len(unique_labels))

    # Parallel energy evaluation
    base_dict = base.as_dict()
    jobs = [(base_dict, sub_idx, lab, label_to_el) for lab in unique_labels]

    ctx = get_context("spawn")  # safest with TF
    best = []  # heap: (-E, idx, labels)

    t0 = time.time()
    with ctx.Pool(processes=cfg.nproc, initializer=_init_worker) as pool:
        for done, (E, labels) in enumerate(
            pool.imap_unordered(_energy_worker_with_labels, jobs, chunksize=cfg.chunksize),
            start=1,
        ):
            item = (-E, done, labels.copy())
            if len(best) < cfg.topk:
                heapq.heappush(best, item)
            else:
                worst_E = -best[0][0]
                if E < worst_E:
                    heapq.heapreplace(best, item)

            if done % 2000 == 0 or done == len(jobs):
                current = sorted([-x[0] for x in best])
                log.info("%d/%d evaluated | best energies: %s", done, len(jobs), current)

    log.info("Evaluation walltime: %.1f s", time.time() - t0)

    # Write outputs
    best_sorted = sorted([(-E, idx, lab) for (E, idx, lab) in best], key=lambda x: x[0])

    ranking_rows = []
    for rank, (E, eval_idx, labels) in enumerate(best_sorted, start=1):
        s = base.copy()
        for pos, site_index in enumerate(sub_idx):
            s[site_index] = label_to_el[int(labels[pos])]

        s2 = _reorder_sites(s, cfg.order)
        sig = _dopant_signature_from_labels(labels, label_to_el)

        cand_dir = struct_dir / f"candidate_{rank:03d}" / "01_scan"
        cand_dir.mkdir(parents=True, exist_ok=True)

        Poscar(s2).write_file(cand_dir / "POSCAR", vasp4_compatible=False)

        meta = {
            "stage": "01_scan",
            "method": "M3GNet-singlepoint (sym-unique enumeration)",
            "rank_sp": rank,
            "energy_sp_eV": float(E),
            "signature": sig,
            "eval_order_index": int(eval_idx),
            "config": {
                "poscar_in": cfg.poscar_in,
                "topk": cfg.topk,
                "symprec": cfg.symprec,
                "max_enum": cfg.max_enum,
                "nproc": cfg.nproc,
                "chunksize": cfg.chunksize,
                "order": cfg.order,
                "anion_species": cfg.anion_species,
                "host_species": cfg.host_species,
                "max_unique": cfg.max_unique,
            },
            "counts": {
                "sublattice_sites": int(N),
                "host_count_on_sublattice": int(host_count),
                "dopant_counts_on_sublattice": {k: int(v) for k, v in dopant_counts.items()},
                "raw_configs_checked": int(checked_raw),
                "unique_sym_configs": int(len(unique_labels)),
                "n_sym_perms": int(len(perms)),
            },
            "labels": [int(x) for x in labels.tolist()],
            "label_to_el": {str(k): v for k, v in label_to_el.items()},
        }

        (cand_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        ranking_rows.append(
            {
                "candidate": f"candidate_{rank:03d}",
                "rank_sp": rank,
                "energy_sp_eV": float(E),
                "signature": sig,
            }
        )

        log.info("SAVE %s | rank %d | E=%.6f eV | %s", cand_dir / "POSCAR", rank, E, sig)

    # CSV
    (struct_dir / "ranking_scan.csv").write_text(
        "candidate,rank_sp,energy_sp_eV,signature\n"
        + "\n".join(
            f"{r['candidate']},{r['rank_sp']},{r['energy_sp_eV']:.10f},{r['signature']}"
            for r in ranking_rows
        )
        + "\n",
        encoding="utf-8",
    )

    # Summary
    with open(struct_dir / "scan_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"CWD: {struct_dir}\n")
        f.write(f"POSCAR_IN: {cfg.poscar_in}\n")
        f.write(f"TOPK: {cfg.topk}\n")
        f.write(f"SYMPREC: {cfg.symprec}\n")
        f.write(f"NPROC: {cfg.nproc}\n")
        f.write(f"CHUNKSIZE: {cfg.chunksize}\n")
        f.write(f"Host: {cfg.host_species}\n")
        f.write(f"Anions: {cfg.anion_species}\n")
        f.write(f"Sublattice sites: {N}\n")
        f.write(f"Dopant counts: {dopant_counts}\n")
        f.write(f"Raw checked: {checked_raw}\n")
        f.write(f"Unique(sym): {len(unique_labels)}\n")
        f.write("Best energies (eV):\n")
        for r in ranking_rows:
            f.write(f"  {r['candidate']}: {r['energy_sp_eV']:.10f} | {r['signature']}\n")

    log.info("DONE %s | wrote candidate_*/01_scan, ranking_scan.csv, scan_summary.txt", struct_dir)


# -----------------------------
# Public API
# -----------------------------
def run_scan(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> None:
    """
    Step 02: For each structure folder in [structure].outdir:
      - enumerate symmetry-unique dopant permutations on the cation sublattice
      - evaluate single-point energy using M3GNet in parallel
      - keep top-k lowest energies
    """
    cfg = _parse_scan_config(raw_cfg)
    outdir = _get_outdir(raw_cfg, root)

    if not outdir.exists():
        raise FileNotFoundError(f"Output directory not found: {outdir} (did you run step 01?)")

    subdirs = sorted([p for p in outdir.iterdir() if p.is_dir()])

    log.info("Step 02 scan: %d structure folders in: %s", len(subdirs), outdir)
    log.info("NOTE: only subfolders are processed (main-directory POSCAR is ignored).")

    for i, sdir in enumerate(subdirs, start=1):
        poscar_path = sdir / cfg.poscar_in
        if not poscar_path.exists():
            log.info("SKIP (%d/%d) %s: missing %s", i, len(subdirs), sdir.name, cfg.poscar_in)
            continue

        if cfg.skip_if_done and (sdir / "ranking_scan.csv").exists():
            log.info("SKIP (%d/%d) %s: ranking_scan.csv already exists", i, len(subdirs), sdir.name)
            continue

        log.info("RUN (%d/%d) %s", i, len(subdirs), sdir.name)
        _scan_one_folder(sdir, cfg)

    log.info("DONE Step 02 scan for all structure folders.")


# TOML wrapper (like your other steps)
try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_scan_from_toml(config_path: Path) -> None:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    run_scan(raw, root, config_path=config_path)

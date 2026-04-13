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

from dopingflow.ml_backends import (
    check_backend_dependency,
    normalize_backend_config,
    prepare_backend_runtime,
    set_default_runtime_env,
    build_ase_calculator,
)
from dopingflow.ml_relaxation import structure_energy_with_calculator

log = logging.getLogger(__name__)

# conservative defaults in this process
set_default_runtime_env(tf_threads=1, omp_threads=1)


# -----------------------------
# Config model
# -----------------------------
@dataclass(frozen=True)
class ScanConfig:
    poscar_in: str
    topk: int
    symprec: float
    max_enum: int
    n_workers: int
    chunksize: int
    order: List[str]
    anion_species: List[str]
    host_species: str
    max_unique: int
    skip_if_done: bool

    device: str
    gpu_id: int

    backend: str
    model: str
    task: str

    mode: str
    sample_budget: int
    sample_batch_size: int
    sample_patience: int
    sample_seed: int
    sample_max_saved: int


def _parse_scan_config(raw: dict[str, Any]) -> ScanConfig:
    scan = raw.get("scan", {}) or {}
    gen = raw.get("generate", {}) or {}
    dop = raw.get("doping", {}) or {}

    poscar_in = str(scan.get("poscar_in", "POSCAR"))
    topk = int(scan.get("topk", 15))
    symprec = float(scan.get("symprec", 1e-3))
    max_enum = int(scan.get("max_enum", 300_000))
    n_workers = int(scan.get("n_workers", 12))
    chunksize = int(scan.get("chunksize", 50))
    anion_species = [str(x) for x in (scan.get("anion_species", ["O"]) or [])]
    host_species = str(dop.get("host_species", "")).strip()
    max_unique = int(scan.get("max_unique", 100_000))
    skip_if_done = bool(scan.get("skip_if_done", True))

    backend = str(scan.get("backend", "m3gnet")).strip().lower()
    model = str(scan.get("model", "default")).strip()
    task = str(scan.get("task", "")).strip()

    device = str(scan.get("device", "cpu")).strip().lower()
    gpu_id = int(scan.get("gpu_id", 0))

    order = [str(x) for x in (gen.get("poscar_order", []) or [])]

    mode = str(scan.get("mode", "auto")).strip().lower()
    sample_budget = int(scan.get("sample_budget", 20_000))
    sample_batch_size = int(scan.get("sample_batch_size", 256))
    sample_patience = int(scan.get("sample_patience", 4_000))
    sample_seed = int(scan.get("sample_seed", 42))
    sample_max_saved = int(scan.get("sample_max_saved", 50_000))

    if topk <= 0:
        raise ValueError("[scan].topk must be > 0")
    if symprec <= 0:
        raise ValueError("[scan].symprec must be > 0")
    if max_enum <= 0:
        raise ValueError("[scan].max_enum must be > 0")
    if n_workers <= 0:
        raise ValueError("[scan].n_workers must be > 0")
    if chunksize <= 0:
        raise ValueError("[scan].chunksize must be > 0")
    if not host_species:
        raise ValueError("[doping].host_species is required")
    if not anion_species:
        raise ValueError("[scan].anion_species must be non-empty")
    if max_unique <= 0:
        raise ValueError("[scan].max_unique must be > 0")
    if device not in {"cpu", "cuda"}:
        raise ValueError('[scan].device must be either "cpu" or "cuda"')
    if gpu_id < 0:
        raise ValueError("[scan].gpu_id must be >= 0")
    if mode not in {"auto", "exact", "sample"}:
        raise ValueError("[scan].mode must be one of: auto, exact, sample")
    if sample_budget <= 0:
        raise ValueError("[scan].sample_budget must be > 0")
    if sample_batch_size <= 0:
        raise ValueError("[scan].sample_batch_size must be > 0")
    if sample_patience <= 0:
        raise ValueError("[scan].sample_patience must be > 0")
    if sample_max_saved <= 0:
        raise ValueError("[scan].sample_max_saved must be > 0")

    backend, model, task = normalize_backend_config(
        backend=backend,
        model=model,
        task=task,
        section_name="scan",
    )

    return ScanConfig(
        poscar_in=poscar_in,
        topk=topk,
        symprec=symprec,
        max_enum=max_enum,
        n_workers=n_workers,
        chunksize=chunksize,
        order=order,
        anion_species=anion_species,
        host_species=host_species,
        max_unique=max_unique,
        skip_if_done=skip_if_done,
        device=device,
        gpu_id=gpu_id,
        backend=backend,
        model=model,
        task=task,
        mode=mode,
        sample_budget=sample_budget,
        sample_batch_size=sample_batch_size,
        sample_patience=sample_patience,
        sample_seed=sample_seed,
        sample_max_saved=sample_max_saved,
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
    sub_idx: List[int] = []
    counts = Counter()

    for i, site in enumerate(struct):
        el = site.species_string
        if el in anions:
            continue
        sub_idx.append(i)
        counts[el] += 1

    if host not in counts:
        raise ValueError(
            f"Host '{host}' not found on enumerated sublattice. Found: {dict(counts)}"
        )

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


def _build_symmetry_permutations(
    parent: Structure, sublattice_indices: List[int], symprec: float
) -> List[np.ndarray]:
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
            raise RuntimeError(
                f"Failed to match symmetry-mapped site (min dist^2={dist2[j]})."
            )
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
            Aset = set(A)
            rem1 = [p for p in allpos if p not in Aset]
            for B in combinations(rem1, c2):
                labels = np.zeros(N, dtype=np.int8)
                labels[list(A)] = l1
                labels[list(B)] = l2
                yield labels
        return

    if len(items) == 3:
        (l1, c1), (l2, c2), (l3, c3) = items
        for A in combinations(allpos, c1):
            Aset = set(A)
            rem1 = [p for p in allpos if p not in Aset]
            for B in combinations(rem1, c2):
                Bset = set(B)
                rem2 = [p for p in rem1 if p not in Bset]
                for C in combinations(rem2, c3):
                    labels = np.zeros(N, dtype=np.int8)
                    labels[list(A)] = l1
                    labels[list(B)] = l2
                    labels[list(C)] = l3
                    yield labels
        return

    raise ValueError("Exact enumerator supports up to 3 dopants total. Use mode='sample' for more.")


def _random_labels(
    N: int,
    dopant_label_counts: Dict[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.zeros(N, dtype=np.int8)
    available = np.arange(N)

    for lab, cnt in sorted(dopant_label_counts.items()):
        chosen = rng.choice(available, size=cnt, replace=False)
        labels[chosen] = lab
        keep_mask = ~np.isin(available, chosen)
        available = available[keep_mask]

    return labels


def _dopant_signature_from_labels(labels: np.ndarray, label_to_el: Dict[int, str]) -> str:
    parts = []
    for pos, lab in enumerate(labels):
        lab = int(lab)
        if lab == 0:
            continue
        parts.append(f"{label_to_el[lab]}{pos}")
    return "_".join(parts) if parts else "pristine"


def _choose_scan_mode(raw_ncfg: int, cfg: ScanConfig, n_dopants: int) -> str:
    if cfg.mode == "exact":
        if n_dopants > 3:
            raise RuntimeError(
                "mode='exact' requested, but exact enumerator supports only up to 3 dopants."
            )
        return "exact"

    if cfg.mode == "sample":
        return "sample"

    if n_dopants > 3:
        return "sample"

    if raw_ncfg <= cfg.max_enum:
        return "exact"

    return "sample"


def _update_best_heap(best, E: float, labels: np.ndarray, idx: int, topk: int) -> bool:
    item = (-E, idx, labels.copy())

    if len(best) < topk:
        heapq.heappush(best, item)
        return True

    worst_E = -best[0][0]
    if E < worst_E:
        heapq.heapreplace(best, item)
        return True

    return False


# -----------------------------
# Backend state
# -----------------------------
_CALCULATOR = None
_CALCULATOR_BACKEND = None
_CALCULATOR_MODEL = None
_CALCULATOR_TASK = None
_CALCULATOR_DEVICE = None


def _init_calculator(
    *,
    backend: str,
    model: str,
    task: str,
    device: str,
    gpu_id: int,
    tf_threads: int = 1,
    omp_threads: int = 1,
) -> None:
    global _CALCULATOR, _CALCULATOR_BACKEND, _CALCULATOR_MODEL, _CALCULATOR_TASK, _CALCULATOR_DEVICE

    prepare_backend_runtime(
        backend=backend,
        device=device,
        gpu_id=gpu_id,
        tf_threads=tf_threads,
        omp_threads=omp_threads,
    )

    _CALCULATOR = build_ase_calculator(
        backend=backend,
        model=model,
        task=task,
        device=device,
    )
    _CALCULATOR_BACKEND = backend
    _CALCULATOR_MODEL = model
    _CALCULATOR_TASK = task
    _CALCULATOR_DEVICE = device


def _ensure_calculator(
    *,
    backend: str,
    model: str,
    task: str,
    device: str,
    gpu_id: int,
    tf_threads: int = 1,
    omp_threads: int = 1,
) -> None:
    if (
        _CALCULATOR is not None
        and _CALCULATOR_BACKEND == backend
        and _CALCULATOR_MODEL == model
        and _CALCULATOR_TASK == task
        and _CALCULATOR_DEVICE == device
    ):
        return

    _init_calculator(
        backend=backend,
        model=model,
        task=task,
        device=device,
        gpu_id=gpu_id,
        tf_threads=tf_threads,
        omp_threads=omp_threads,
    )


def _worker_initializer(
    backend: str,
    model: str,
    task: str,
    device: str,
    gpu_id: int,
):
    _init_calculator(
        backend=backend,
        model=model,
        task=task,
        device=device,
        gpu_id=gpu_id,
        tf_threads=1,
        omp_threads=1,
    )


def _energy_worker_with_labels(args):
    base_dict, sublattice_indices, labels, label_to_el = args
    base_local = Structure.from_dict(base_dict)

    s = base_local.copy()
    for pos, site_index in enumerate(sublattice_indices):
        s[site_index] = label_to_el[int(labels[pos])]

    E = structure_energy_with_calculator(s, _CALCULATOR)
    return (E, labels)


def _energy_jobs_serial(jobs):
    for job in jobs:
        yield _energy_worker_with_labels(job)


# -----------------------------
# Exact mode
# -----------------------------
def _evaluate_exact_configs(
    *,
    base: Structure,
    sub_idx: List[int],
    unique_labels: List[np.ndarray],
    label_to_el: Dict[int, str],
    cfg: ScanConfig,
) -> Tuple[List[Tuple[float, int, np.ndarray]], Dict[str, int]]:
    base_dict = base.as_dict()
    jobs = [(base_dict, sub_idx, lab, label_to_el) for lab in unique_labels]

    best = []
    t0 = time.time()
    effective_n_workers = 1 if cfg.device == "cuda" else cfg.n_workers

    if effective_n_workers == 1:
        _ensure_calculator(
            backend=cfg.backend,
            model=cfg.model,
            task=cfg.task,
            device=cfg.device,
            gpu_id=cfg.gpu_id,
            tf_threads=1,
            omp_threads=1,
        )

        for done, (E, labels) in enumerate(_energy_jobs_serial(jobs), start=1):
            _update_best_heap(best, E, labels, done, cfg.topk)

            if done % 2000 == 0 or done == len(jobs):
                current = sorted([-x[0] for x in best])
                log.info("%d/%d evaluated | best energies: %s", done, len(jobs), current)

    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=effective_n_workers,
            initializer=_worker_initializer,
            initargs=(cfg.backend, cfg.model, cfg.task, cfg.device, cfg.gpu_id),
        ) as pool:
            for done, (E, labels) in enumerate(
                pool.imap_unordered(_energy_worker_with_labels, jobs, chunksize=cfg.chunksize),
                start=1,
            ):
                _update_best_heap(best, E, labels, done, cfg.topk)

                if done % 2000 == 0 or done == len(jobs):
                    current = sorted([-x[0] for x in best])
                    log.info("%d/%d evaluated | best energies: %s", done, len(jobs), current)

    log.info("Exact evaluation walltime: %.1f s", time.time() - t0)

    best_sorted = sorted([(-E, idx, lab) for (E, idx, lab) in best], key=lambda x: x[0])
    stats = {
        "attempted_samples": 0,
        "unique_sym_samples": len(unique_labels),
        "evaluated_samples": len(unique_labels),
        "stopped_by_patience": 0,
    }
    return best_sorted, stats


# -----------------------------
# Sampling mode
# -----------------------------
def _sample_unique_configs_and_rank(
    *,
    base: Structure,
    sub_idx: List[int],
    perms: List[np.ndarray],
    label_to_el: Dict[int, str],
    dopant_label_counts: Dict[int, int],
    cfg: ScanConfig,
) -> Tuple[List[Tuple[float, int, np.ndarray]], Dict[str, int]]:
    rng = np.random.default_rng(cfg.sample_seed)
    base_dict = base.as_dict()

    seen = set()
    best = []

    attempted = 0
    no_improve_counter = 0
    eval_counter = 0
    t0 = time.time()

    effective_n_workers = 1 if cfg.device == "cuda" else cfg.n_workers

    if effective_n_workers == 1:
        _ensure_calculator(
            backend=cfg.backend,
            model=cfg.model,
            task=cfg.task,
            device=cfg.device,
            gpu_id=cfg.gpu_id,
            tf_threads=1,
            omp_threads=1,
        )

        while attempted < cfg.sample_budget and no_improve_counter < cfg.sample_patience:
            batch_labels = []
            batch_keys = set()

            while len(batch_labels) < cfg.sample_batch_size and attempted < cfg.sample_budget:
                attempted += 1
                labels = _random_labels(len(sub_idx), dopant_label_counts, rng)
                key = _canonical_key(labels, perms)

                if key in seen or key in batch_keys:
                    continue

                seen.add(key)
                batch_keys.add(key)
                batch_labels.append(labels)

                if len(seen) >= cfg.sample_max_saved:
                    log.warning(
                        "Reached sample_max_saved=%d canonical keys; stopping sampling.",
                        cfg.sample_max_saved,
                    )
                    attempted = cfg.sample_budget
                    break

            if not batch_labels:
                break

            jobs = [(base_dict, sub_idx, lab, label_to_el) for lab in batch_labels]

            improved_in_batch = False
            for E, labels in _energy_jobs_serial(jobs):
                eval_counter += 1
                improved = _update_best_heap(best, E, labels, eval_counter, cfg.topk)
                if improved:
                    improved_in_batch = True

            if improved_in_batch:
                no_improve_counter = 0
            else:
                no_improve_counter += len(batch_labels)

            if eval_counter % 500 == 0 or attempted >= cfg.sample_budget:
                current = sorted([-x[0] for x in best])
                log.info(
                    "sampled attempts=%d | unique=%d | evaluated=%d | best=%s | no_improve=%d",
                    attempted,
                    len(seen),
                    eval_counter,
                    current,
                    no_improve_counter,
                )

    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=effective_n_workers,
            initializer=_worker_initializer,
            initargs=(cfg.backend, cfg.model, cfg.task, cfg.device, cfg.gpu_id),
        ) as pool:
            while attempted < cfg.sample_budget and no_improve_counter < cfg.sample_patience:
                batch_labels = []
                batch_keys = set()

                while len(batch_labels) < cfg.sample_batch_size and attempted < cfg.sample_budget:
                    attempted += 1
                    labels = _random_labels(len(sub_idx), dopant_label_counts, rng)
                    key = _canonical_key(labels, perms)

                    if key in seen or key in batch_keys:
                        continue

                    seen.add(key)
                    batch_keys.add(key)
                    batch_labels.append(labels)

                    if len(seen) >= cfg.sample_max_saved:
                        log.warning(
                            "Reached sample_max_saved=%d canonical keys; stopping sampling.",
                            cfg.sample_max_saved,
                        )
                        attempted = cfg.sample_budget
                        break

                if not batch_labels:
                    break

                jobs = [(base_dict, sub_idx, lab, label_to_el) for lab in batch_labels]

                improved_in_batch = False
                for E, labels in pool.imap_unordered(
                    _energy_worker_with_labels,
                    jobs,
                    chunksize=cfg.chunksize,
                ):
                    eval_counter += 1
                    improved = _update_best_heap(best, E, labels, eval_counter, cfg.topk)
                    if improved:
                        improved_in_batch = True

                if improved_in_batch:
                    no_improve_counter = 0
                else:
                    no_improve_counter += len(batch_labels)

                if eval_counter % 500 == 0 or attempted >= cfg.sample_budget:
                    current = sorted([-x[0] for x in best])
                    log.info(
                        "sampled attempts=%d | unique=%d | evaluated=%d | best=%s | no_improve=%d",
                        attempted,
                        len(seen),
                        eval_counter,
                        current,
                        no_improve_counter,
                    )

    log.info("Sampling evaluation walltime: %.1f s", time.time() - t0)

    best_sorted = sorted([(-E, idx, lab) for (E, idx, lab) in best], key=lambda x: x[0])
    stats = {
        "attempted_samples": attempted,
        "unique_sym_samples": len(seen),
        "evaluated_samples": eval_counter,
        "stopped_by_patience": int(no_improve_counter >= cfg.sample_patience),
    }
    return best_sorted, stats


# -----------------------------
# Exact unique generation
# -----------------------------
def _enumerate_unique_configs_exact(
    *,
    N: int,
    perms: List[np.ndarray],
    dopant_label_counts: Dict[int, int],
    cfg: ScanConfig,
) -> Tuple[List[np.ndarray], int]:
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

    log.info("Done exact generation: raw=%d, unique(sym)=%d", checked_raw, len(unique_labels))
    return unique_labels, checked_raw


# -----------------------------
# Core
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
    log.info(
        "Enumerated sublattice size: %d (host=%s, anions=%s)",
        N,
        cfg.host_species,
        cfg.anion_species,
    )
    log.info("Dopant counts inferred on sublattice: %s", dopant_counts)
    log.info("Estimated raw configurations: %d", raw_ncfg)

    if host_count + sum(dopant_counts.values()) != N:
        raise RuntimeError("Counts inconsistent with sublattice size.")

    parent = _make_parent_structure(base, sub_idx, host=cfg.host_species)
    perms = _build_symmetry_permutations(parent, sub_idx, symprec=cfg.symprec)
    log.info("Symmetry operations (unique permutations on sublattice): %d", len(perms))

    dopants_sorted = sorted(dopant_counts.items())
    label_to_el: Dict[int, str] = {0: cfg.host_species}
    dopant_label_counts: Dict[int, int] = {}
    for lab, (el, cnt) in enumerate(dopants_sorted, start=1):
        label_to_el[lab] = el
        dopant_label_counts[lab] = int(cnt)

    log.info("Label map: %s", label_to_el)
    log.info("Label counts: %s", dopant_label_counts)

    selected_mode = _choose_scan_mode(raw_ncfg, cfg, n_dopants=len(dopant_label_counts))
    log.info("Selected scan mode: %s", selected_mode)

    if selected_mode == "exact":
        if raw_ncfg > cfg.max_enum:
            raise RuntimeError(
                f"Exact mode selected but raw configs ({raw_ncfg}) > max_enum ({cfg.max_enum})."
            )

        unique_labels, checked_raw = _enumerate_unique_configs_exact(
            N=N,
            perms=perms,
            dopant_label_counts=dopant_label_counts,
            cfg=cfg,
        )

        best_sorted, stats = _evaluate_exact_configs(
            base=base,
            sub_idx=sub_idx,
            unique_labels=unique_labels,
            label_to_el=label_to_el,
            cfg=cfg,
        )

        unique_count = len(unique_labels)

    elif selected_mode == "sample":
        best_sorted, stats = _sample_unique_configs_and_rank(
            base=base,
            sub_idx=sub_idx,
            perms=perms,
            label_to_el=label_to_el,
            dopant_label_counts=dopant_label_counts,
            cfg=cfg,
        )

        checked_raw = stats["attempted_samples"]
        unique_count = stats["unique_sym_samples"]

    else:
        raise RuntimeError(f"Unknown scan mode: {selected_mode}")

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
            "method": f"{cfg.backend}-singlepoint ({selected_mode})",
            "backend": cfg.backend,
            "model": cfg.model,
            "task": cfg.task,
            "rank_sp": rank,
            "energy_sp_eV": float(E),
            "signature": sig,
            "eval_order_index": int(eval_idx),
            "config": {
                "backend": cfg.backend,
                "model": cfg.model,
                "task": cfg.task,
                "poscar_in": cfg.poscar_in,
                "topk": cfg.topk,
                "symprec": cfg.symprec,
                "max_enum": cfg.max_enum,
                "n_workers": cfg.n_workers,
                "chunksize": cfg.chunksize,
                "order": cfg.order,
                "anion_species": cfg.anion_species,
                "host_species": cfg.host_species,
                "max_unique": cfg.max_unique,
                "device": cfg.device,
                "gpu_id": cfg.gpu_id,
                "mode": cfg.mode,
                "selected_mode": selected_mode,
                "sample_budget": cfg.sample_budget,
                "sample_batch_size": cfg.sample_batch_size,
                "sample_patience": cfg.sample_patience,
                "sample_seed": cfg.sample_seed,
                "sample_max_saved": cfg.sample_max_saved,
            },
            "counts": {
                "sublattice_sites": int(N),
                "host_count_on_sublattice": int(host_count),
                "dopant_counts_on_sublattice": {k: int(v) for k, v in dopant_counts.items()},
                "raw_configs_checked": int(checked_raw),
                "unique_sym_configs": int(unique_count),
                "n_sym_perms": int(len(perms)),
                **stats,
            },
            "labels": [int(x) for x in labels.tolist()],
            "label_to_el": {str(k): v for k, v in label_to_el.items()},
        }

        (cand_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        ranking_rows.append(
            {
                "backend": cfg.backend,
                "model": cfg.model,
                "task": cfg.task,
                "candidate": f"candidate_{rank:03d}",
                "rank_sp": rank,
                "energy_sp_eV": float(E),
                "signature": sig,
            }
        )

        log.info("SAVE %s | rank %d | E=%.6f eV | %s", cand_dir / "POSCAR", rank, E, sig)

    (struct_dir / "ranking_scan.csv").write_text(
        "candidate,rank_sp,energy_sp_eV,signature,backend,model,task\n"
        + "\n".join(
            f"{r['candidate']},{r['rank_sp']},{r['energy_sp_eV']:.10f},{r['signature']},{r['backend']},{r['model']},{r['task']}"
            for r in ranking_rows
        )
        + "\n",
        encoding="utf-8",
    )

    with open(struct_dir / "scan_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"CWD: {struct_dir}\n")
        f.write(f"POSCAR_IN: {cfg.poscar_in}\n")
        f.write(f"TOPK: {cfg.topk}\n")
        f.write(f"SYMPREC: {cfg.symprec}\n")
        f.write(f"n_workers: {cfg.n_workers}\n")
        f.write(f"CHUNKSIZE: {cfg.chunksize}\n")
        f.write(f"device: {cfg.device}\n")
        f.write(f"gpu_id: {cfg.gpu_id}\n")
        f.write(f"backend: {cfg.backend}\n")
        f.write(f"model: {cfg.model}\n")
        f.write(f"task: {cfg.task}\n")
        f.write(f"Host: {cfg.host_species}\n")
        f.write(f"Anions: {cfg.anion_species}\n")
        f.write(f"Mode requested: {cfg.mode}\n")
        f.write(f"Mode selected: {selected_mode}\n")
        f.write(f"Sublattice sites: {N}\n")
        f.write(f"Dopant counts: {dopant_counts}\n")
        f.write(f"Estimated raw configs: {raw_ncfg}\n")
        f.write(f"Raw checked: {checked_raw}\n")
        f.write(f"Unique(sym): {unique_count}\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write("Best energies (eV):\n")
        for r in ranking_rows:
            f.write(f"  {r['candidate']}: {r['energy_sp_eV']:.10f} | {r['signature']}\n")

    log.info("DONE %s | wrote candidate_*/01_scan, ranking_scan.csv, scan_summary.txt", struct_dir)


# -----------------------------
# Public API
# -----------------------------
def run_scan(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> None:
    """
    Step 02:
      - enumerate / sample symmetry-unique dopant arrangements
      - evaluate single-point energies using selected ML backend via ASE calculator
      - keep top-k lowest energies
    """
    cfg = _parse_scan_config(raw_cfg)
    check_backend_dependency(cfg.backend, stage_name="Scan")

    outdir = _get_outdir(raw_cfg, root)

    if not outdir.exists():
        raise FileNotFoundError(f"Output directory not found: {outdir} (did you run step 01?)")

    subdirs = sorted([p for p in outdir.iterdir() if p.is_dir()])

    log.info("Step 02 scan: %d structure folders in: %s", len(subdirs), outdir)
    log.info(
        "Scan backend: backend=%s model=%s task=%s device=%s gpu_id=%d",
        cfg.backend,
        cfg.model,
        cfg.task,
        cfg.device,
        cfg.gpu_id,
    )
    if cfg.device == "cuda":
        log.info("CUDA mode: forcing effective_n_workers=1 for safe GPU usage.")
    log.info("NOTE: only subfolders are processed (main-directory POSCAR is ignored).")

    for i, sdir in enumerate(subdirs, start=1):
        poscar_path = sdir / cfg.poscar_in
        if not poscar_path.exists():
            log.info("SKIP (%d/%d) %s: missing %s", i, len(subdirs), sdir.name, cfg.poscar_in)
            continue

        if cfg.skip_if_done and (sdir / "ranking_scan.csv").exists():
            log.info("SKIP (%d/%d) %s: ranking_scan.csv already exists", i, len(subdirs), sdir.name)
            continue

        if cfg.device == "cpu" and _CALCULATOR is None:
            log.info(
                "Loading %s calculator in main process: model=%s device=%s",
                cfg.backend.upper(),
                cfg.model,
                cfg.device,
            )
            _ensure_calculator(
                backend=cfg.backend,
                model=cfg.model,
                task=cfg.task,
                device=cfg.device,
                gpu_id=cfg.gpu_id,
                tf_threads=1,
                omp_threads=1,
            )

        log.info("RUN (%d/%d) %s", i, len(subdirs), sdir.name)
        _scan_one_folder(sdir, cfg)

    log.info("DONE Step 02 scan for all structure folders.")


# TOML wrapper
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_scan_from_toml(config_path: Path) -> None:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    run_scan(raw, root, config_path=config_path)
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

log = logging.getLogger(__name__)


# -----------------------------
# Config for this step
# -----------------------------
@dataclass(frozen=True)
class GenerateConfig:
    # structure
    base_poscar: str
    supercell: Tuple[int, int, int]
    outdir: str

    # generate
    poscar_order: List[str]
    seed_base: int
    clean_outdir: bool  # new: keep default True to match old behavior

    # doping
    mode: str  # "explicit" or "enumerate"
    host_species: str

    # explicit mode
    compositions: List[Dict[str, float]]

    # enumerate mode
    dopants: List[str]
    must_include: List[str]
    max_dopants_total: int
    allowed_totals: List[float]
    levels: List[float]


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse_generate_config(raw: dict[str, Any]) -> GenerateConfig:
    st = raw.get("structure", {}) or {}
    gen = raw.get("generate", {}) or {}
    dop = raw.get("doping", {}) or {}

    base_poscar = str(st.get("base_poscar", "POSCAR")).strip()
    supercell = tuple(int(x) for x in st.get("supercell", [1, 1, 1]))
    if len(supercell) != 3:
        raise ValueError("[structure].supercell must have 3 integers")
    outdir = str(st.get("outdir", "random_structures")).strip()

    poscar_order = list(gen.get("poscar_order", []))
    seed_base = int(gen.get("seed_base", 0))
    clean_outdir = bool(gen.get("clean_outdir", True))  # default=True matches old script

    mode = str(dop.get("mode", "explicit")).lower().strip()
    if mode not in {"explicit", "enumerate"}:
        raise ValueError("[doping].mode must be 'explicit' or 'enumerate'")

    host_species = str(dop.get("host_species", "")).strip()
    if not host_species:
        raise ValueError("[doping].host_species is required")

    compositions = dop.get("compositions", []) or []
    dopants = dop.get("dopants", []) or []
    must_include = dop.get("must_include", []) or []
    max_dopants_total = int(dop.get("max_dopants_total", 2))
    allowed_totals = dop.get("allowed_totals", []) or []
    levels = dop.get("levels", []) or []

    # normalize types
    compositions = [dict((str(k), float(v)) for k, v in c.items()) for c in compositions]
    dopants = [str(x) for x in dopants]
    must_include = [str(x) for x in must_include]
    allowed_totals = [float(x) for x in allowed_totals]
    levels = [float(x) for x in levels]

    return GenerateConfig(
        base_poscar=base_poscar,
        supercell=supercell,
        outdir=outdir,
        poscar_order=poscar_order,
        seed_base=seed_base,
        clean_outdir=clean_outdir,
        mode=mode,
        host_species=host_species,
        compositions=compositions,
        dopants=dopants,
        must_include=must_include,
        max_dopants_total=max_dopants_total,
        allowed_totals=allowed_totals,
        levels=levels,
    )


# -----------------------------
# Core helpers (kept from your script)
# -----------------------------
def validate_composition_minimal(requested_pct: Dict[str, float]) -> None:
    if not requested_pct:
        raise ValueError("Composition dict is empty.")
    for k, v in requested_pct.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Invalid element key: {k!r}")
        if v < 0:
            raise ValueError(f"Negative percent for {k}: {v}")


def normalize_to_counts_and_effective(
    n_host: int, requested_pct: Dict[str, float]
) -> tuple[Dict[str, int], Dict[str, float], List[str], float, float]:
    """
    Convert requested dopant percentages (relative to host sites) into integer dopant counts
    by rounding to the nearest integer number of substitutions.

    Parameters
    ----------
    n_host : int
        Number of host sites available for substitution.
    requested_pct : dict[str, float]
        Dopant element -> requested percent (relative to host sites).

    Returns
    -------
    counts : dict[str, int]
        Dopant element -> integer dopant count after rounding.
    effective_pct : dict[str, float]
        Dopant element -> effective percent after rounding.
    warnings : list[str]
        Human-readable warnings about rounding deviations.
    requested_total_pct : float
        Sum of requested dopant percentages.
    effective_total_pct : float
        Sum of effective dopant percentages after rounding.

    Raises
    ------
    ValueError
        If requested total exceeds 100% or rounded dopant atoms exceed host sites.
    """

    warnings: List[str] = []

    requested_total = float(sum(requested_pct.values()))
    if requested_total > 100.0 + 1e-9:
        raise ValueError(f"Requested total dopant % exceeds 100%: {requested_total}")

    counts: Dict[str, int] = {}
    effective: Dict[str, float] = {}

    for el, pct in requested_pct.items():
        raw = pct * n_host / 100.0
        c = int(round(raw))
        counts[el] = c

        eff = (100.0 * c / n_host) if n_host > 0 else 0.0
        effective[el] = eff

        if abs(eff - pct) > 1e-9:
            warnings.append(f"{el}: requested {pct:.6g}% -> rounded to {c} atoms -> effective {eff:.6g}%")

    eff_total = float(sum(effective.values()))
    if abs(eff_total - requested_total) > 1e-9:
        warnings.append(
            f"Total: requested {requested_total:.6g}% -> effective {eff_total:.6g}% after rounding"
        )

    # sanity: total dopant atoms cannot exceed host sites
    if sum(counts.values()) > n_host:
        raise ValueError(
            f"Rounded dopant atoms ({sum(counts.values())}) exceed host sites ({n_host}). "
            "Reduce requested doping or increase supercell."
        )

    return counts, effective, warnings, requested_total, eff_total


def composition_tag(effective_pct: Dict[str, float], must_first: List[str] | None = None) -> str:
    """
    Tag used as folder name. Example: Sb5_Ba5  (numbers are integers if near-int).
    """
    must_first = must_first or []

    def fmt(x: float) -> str:
        # prefer integer-like formatting when close
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.4g}".replace(".", "p")

    items = [(k, effective_pct[k]) for k in effective_pct.keys()]
    # stable ordering: must_include first, then alphabetical
    items.sort(key=lambda kv: (0 if kv[0] in must_first else 1, kv[0]))

    parts = []
    for el, pct in items:
        if pct <= 0:
            continue
        parts.append(f"{el}{fmt(pct)}")
    return "_".join(parts)


def stable_seed_from_tag(tag: str, seed_base: int) -> int:
    h = hashlib.sha256((str(seed_base) + "::" + tag).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def build_structure_from_counts(
    pristine: Structure,
    host_species: str,
    dopant_counts: Dict[str, int],
    seed: int,
) -> Structure:
    """
    Randomly substitute host_species sites with dopants according to dopant_counts.
    Deterministic given seed + tag.
    """
    import random

    s = pristine.copy()
    rng = random.Random(seed)

    host_indices = [i for i, site in enumerate(s) if site.species_string == host_species]
    rng.shuffle(host_indices)

    cursor = 0
    for dopant, n in dopant_counts.items():
        for _ in range(n):
            idx = host_indices[cursor]
            cursor += 1
            s[idx] = dopant
    return s


def reorder_structure_by_species(s: Structure, order: List[str]) -> Structure:
    """
    Reorder sites so POSCAR species order matches user preference.
    If order is empty, keep pymatgen default ordering.
    """
    if not order:
        return s

    # group sites by requested species order, then append remaining species
    order = [x for x in order if x]
    species_in_s = [sp.symbol for sp in s.composition.elements]
    remaining = [x for x in species_in_s if x not in order]
    final = order + sorted(remaining)

    # pymatgen supports sorting by species string order
    return s.get_sorted_structure(key=lambda site: final.index(site.species_string) if site.species_string in final else 10**9)


def enumerate_compositions(
    dopants: List[str],
    must_include: List[str],
    max_dopants_total: int,
    allowed_totals: List[float],
    levels: List[float],
) -> List[Dict[str, float]]:
    """
    Generate composition dictionaries according to rules.
    (Same intent as your original script.)
    """
    from itertools import combinations, product

    dopants = list(dict.fromkeys(dopants))
    must_include = list(dict.fromkeys(must_include))

    if not dopants:
        raise ValueError("enumerate mode requires [doping].dopants")
    if not levels:
        raise ValueError("enumerate mode requires [doping].levels (e.g., [5,10,15])")
    if not allowed_totals:
        raise ValueError("enumerate mode requires [doping].allowed_totals (e.g., [5,10,15])")

    comps: List[Dict[str, float]] = []

    # choose dopant sets of size 1..max_dopants_total including must_include
    for k in range(1, max_dopants_total + 1):
        for subset in combinations(dopants, k):
            subset = list(subset)
            if any(m not in subset for m in must_include):
                continue

            # assign levels to each dopant
            for lv in product(levels, repeat=k):
                comp = {el: float(p) for el, p in zip(subset, lv)}
                total = sum(comp.values())
                if any(abs(total - t) < 1e-9 for t in allowed_totals):
                    comps.append(comp)

    # stable unique
    uniq = {}
    for c in comps:
        key = tuple(sorted(c.items()))
        uniq[key] = c
    return list(uniq.values())


# -----------------------------
# Public API (module entry)
# -----------------------------
def run_generate(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> Path:
    """
    Generate one random doped POSCAR per composition.

    Output:
      <outdir>/<tag>/POSCAR
      <outdir>/<tag>/metadata.json

    Returns:
      outdir path
    """
    cfg = _parse_generate_config(raw_cfg)

    base_poscar_path = (root / cfg.base_poscar).resolve()
    if not base_poscar_path.exists():
        raise FileNotFoundError(
            f"Base POSCAR not found: {base_poscar_path}\n"
            f"Put '{cfg.base_poscar}' in the workflow root, or change [structure].base_poscar."
        )

    outdir = (root / cfg.outdir).resolve()
    if cfg.clean_outdir and outdir.exists():
        log.info("Cleaning existing outdir: %s", outdir)
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pristine = Structure.from_file(str(base_poscar_path))
    pristine.make_supercell(cfg.supercell)

    host_indices = [i for i, site in enumerate(pristine) if site.species_string == cfg.host_species]
    n_host = len(host_indices)
    log.info(
        "Supercell %s: host='%s' sites=%d, total sites=%d",
        cfg.supercell,
        cfg.host_species,
        n_host,
        len(pristine),
    )
    if n_host == 0:
        raise RuntimeError(
            f"No sites with species '{cfg.host_species}' found after applying supercell {cfg.supercell}."
        )

    if cfg.mode == "explicit":
        requested_comps = cfg.compositions
        must_first_for_tag: List[str] = []
    else:
        requested_comps = enumerate_compositions(
            dopants=cfg.dopants,
            must_include=cfg.must_include,
            max_dopants_total=cfg.max_dopants_total,
            allowed_totals=cfg.allowed_totals,
            levels=cfg.levels,
        )
        must_first_for_tag = cfg.must_include

    log.info("Mode=%s. Compositions to generate: %d", cfg.mode, len(requested_comps))

    # avoid in-run collisions (different requested -> same effective tag)
    tag_counts: Dict[str, int] = {}

    for idx, requested_comp in enumerate(requested_comps, start=1):
        validate_composition_minimal(requested_comp)

        dopant_counts, effective_pct, warnings, total_req, total_eff = normalize_to_counts_and_effective(
            n_host=n_host,
            requested_pct=requested_comp,
        )

        base_tag = composition_tag(effective_pct, must_first=must_first_for_tag) or "pristine"
        tag_counts[base_tag] = tag_counts.get(base_tag, 0) + 1
        tag = base_tag if tag_counts[base_tag] == 1 else f"{base_tag}__v{tag_counts[base_tag]}"

        seed = stable_seed_from_tag(tag, cfg.seed_base)

        if warnings:
            log.warning("DOPING ROUNDING WARNING (requested #%d): %s", idx, requested_comp)
            for w in warnings:
                log.warning("  - %s", w)
            log.warning("  -> Using EFFECTIVE composition tag: %s", tag)

        s = build_structure_from_counts(
            pristine=pristine,
            host_species=cfg.host_species,
            dopant_counts=dopant_counts,
            seed=seed,
        )
        s2 = reorder_structure_by_species(s, cfg.poscar_order)

        comp_dir = outdir / tag
        comp_dir.mkdir(exist_ok=True)

        Poscar(s2).write_file(str(comp_dir / "POSCAR"), vasp4_compatible=False)

        meta = {
            "composition_tag_effective": tag,
            "composition_tag_effective_base": base_tag,
            "requested_index": idx,
            "seed": seed,
            "supercell": list(cfg.supercell),
            "poscar_order": cfg.poscar_order,
            "host_species": cfg.host_species,
            "n_host": n_host,
            "requested_pct": requested_comp,
            "rounded_counts": dopant_counts,
            "effective_pct": effective_pct,
            "requested_total_pct": total_req,
            "effective_total_pct": total_eff,
            "rounding_warnings": warnings,
            "input_file": str(config_path.name) if config_path else "input.toml",
        }
        (comp_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Done. Wrote structures to: %s", outdir)
    return outdir


# Optional convenience wrapper (useful for quick manual runs)
def run_generate_from_toml(config_path: Path) -> Path:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    return run_generate(raw, root, config_path=config_path)

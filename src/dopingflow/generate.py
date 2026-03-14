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
    outdir: str

    # generate
    poscar_order: List[str]
    seed_base: int
    clean_outdir: bool  # keep default True to match old behavior

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
# NEW: strict host supercell loader
# -----------------------------
def _load_relaxed_host_supercell(root: Path) -> tuple[Structure, Path, dict[str, Any]]:
    """
    Strict loader:
      - requires reference_structures/reference_energies.json
      - loads host.relaxed_supercell_poscar from that JSON
    Returns: (Structure, poscar_path, json_data)
    """
    ref_json = (root / "reference_structures" / "reference_energies.json").resolve()
    if not ref_json.exists():
        raise FileNotFoundError(
            "Missing reference energies file:\n"
            f"  {ref_json}\n\n"
            "Run: dopingflow refs-build -c input.toml"
        )

    data = json.loads(ref_json.read_text(encoding="utf-8"))
    try:
        rel_path = data["host"]["relaxed_supercell_poscar"]
    except Exception as e:
        raise KeyError(
            "reference_energies.json does not contain host.relaxed_supercell_poscar.\n"
            "Your refs-build output is not compatible with this version."
        ) from e

    poscar_path = (root / rel_path).resolve()
    if not poscar_path.exists():
        raise FileNotFoundError(
            "Relaxed host supercell POSCAR not found:\n"
            f"  {poscar_path}\n\n"
            "Run refs-build again, or check reference_structures/relaxed outputs."
        )

    s = Structure.from_file(str(poscar_path))
    return s, poscar_path, data


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

    if sum(counts.values()) > n_host:
        raise ValueError(
            f"Rounded dopant atoms ({sum(counts.values())}) exceed host sites ({n_host}). "
            "Reduce requested doping or increase supercell."
        )

    return counts, effective, warnings, requested_total, eff_total


def composition_tag(effective_pct: Dict[str, float], must_first: List[str] | None = None) -> str:
    must_first = must_first or []

    def fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.4g}".replace(".", "p")

    items = [(k, effective_pct[k]) for k in effective_pct.keys()]
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
    if not order:
        return s

    order = [x for x in order if x]
    species_in_s = [sp.symbol for sp in s.composition.elements]
    remaining = [x for x in species_in_s if x not in order]
    final = order + sorted(remaining)

    return s.get_sorted_structure(
        key=lambda site: final.index(site.species_string) if site.species_string in final else 10**9
    )


def enumerate_compositions(
    dopants: List[str],
    must_include: List[str],
    max_dopants_total: int,
    allowed_totals: List[float],
    levels: List[float],
) -> List[Dict[str, float]]:
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

    for k in range(1, max_dopants_total + 1):
        for subset in combinations(dopants, k):
            subset = list(subset)
            if any(m not in subset for m in must_include):
                continue

            for lv in product(levels, repeat=k):
                comp = {el: float(p) for el, p in zip(subset, lv)}
                total = sum(comp.values())
                if any(abs(total - t) < 1e-9 for t in allowed_totals):
                    comps.append(comp)

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

    Requires:
      - refs-build completed
      - reference_structures/reference_energies.json exists
      - relaxed host supercell POSCAR exists

    Output:
      <outdir>/<tag>/POSCAR
      <outdir>/<tag>/metadata.json
    """
    cfg = _parse_generate_config(raw_cfg)

    outdir = (root / cfg.outdir).resolve()
    if cfg.clean_outdir and outdir.exists():
        log.info("Cleaning existing outdir: %s", outdir)
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pristine, host_supercell_path, ref_data = _load_relaxed_host_supercell(root)

    host_indices = [i for i, site in enumerate(pristine) if site.species_string == cfg.host_species]
    n_host = len(host_indices)
    log.info(
        "Using relaxed host supercell: %s",
        host_supercell_path,
    )
    log.info("Host='%s' sites=%d, total sites=%d", cfg.host_species, n_host, len(pristine))

    if n_host == 0:
        raise RuntimeError(
            f"No sites with species '{cfg.host_species}' found in the relaxed host supercell. "
            "Check [doping].host_species."
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
            "refs_json": str((root / "reference_structures" / "reference_energies.json").resolve()),
            "host_supercell_poscar": str(host_supercell_path),
            "refs_reference_mode": str(ref_data.get("reference_mode", "")),
        }
        (comp_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Done. Wrote structures to: %s", outdir)
    return outdir


def run_generate_from_toml(config_path: Path) -> Path:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    return run_generate(raw, root, config_path=config_path)
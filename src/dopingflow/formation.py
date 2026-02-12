# src/dopingflow/formation.py
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)

# fixed I/O names (do not expose to users)
REF_JSON = Path("reference_structures/reference_energies.json")
CAND_LIST = "selected_candidates.txt"
RELAX_META = "02_relax/meta.json"
RELAX_POSCAR = "02_relax/POSCAR"
OUT_CSV = "formation_energies.csv"
OUT_META_REL = "04_formation/meta.json"


@dataclass(frozen=True)
class FormationConfig:
    outdir: Path
    host_species: str
    anion_species: List[str]
    skip_if_done: bool
    normalize: str  # "total" | "per_dopant" | "per_host"


def _parse_formation_config(raw: dict[str, Any], root: Path) -> FormationConfig:
    st = raw.get("structure", {}) or {}
    dop = raw.get("doping", {}) or {}
    scan = raw.get("scan", {}) or {}
    form = raw.get("formation", {}) or {}

    outdir_name = str(st.get("outdir", "random_structures"))
    outdir = (root / outdir_name).resolve()

    host_species = str(dop.get("host_species", "")).strip()
    if not host_species:
        raise ValueError("[doping].host_species is required")

    anion_species = [str(x) for x in (scan.get("anion_species", ["O"]) or [])]
    if not anion_species:
        raise ValueError("[scan].anion_species must be non-empty")

    skip_if_done = bool(form.get("skip_if_done", True))
    normalize = str(form.get("normalize", "per_dopant")).strip().lower()
    if normalize not in {"total", "per_dopant", "per_host"}:
        raise ValueError("[formation].normalize must be one of: total, per_dopant, per_host")

    return FormationConfig(
        outdir=outdir,
        host_species=host_species,
        anion_species=anion_species,
        skip_if_done=skip_if_done,
        normalize=normalize,
    )


def _load_ref_json(root: Path) -> dict[str, Any]:
    p = (root / REF_JSON).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Missing reference file: {p}\n"
            "Run Step 00 first: dopingflow refs-build -c input.toml"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _read_selected_candidates(path: Path) -> List[str]:
    out: List[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def _count_species_from_poscar(poscar_path: Path) -> Dict[str, int]:
    from pymatgen.core import Structure

    s = Structure.from_file(str(poscar_path))
    counts: Dict[str, int] = {}
    for site in s:
        el = site.species_string
        counts[el] = counts.get(el, 0) + 1
    return counts


def _get_candidate_poscars(folder: Path) -> List[Path]:
    cand_list = folder / CAND_LIST
    if cand_list.exists():
        names = _read_selected_candidates(cand_list)
        poscars = [folder / n / RELAX_POSCAR for n in names]
        poscars = [p for p in poscars if p.exists()]
        log.info("SELECT %s: using %d candidates from %s", folder.name, len(poscars), CAND_LIST)
        return poscars

    poscars = sorted(folder.glob(f"candidate_*/{RELAX_POSCAR}"))
    log.info("SELECT %s: using glob: %d candidates", folder.name, len(poscars))
    return poscars


def _load_relax_energy(meta_path: Path) -> float:
    d = json.loads(meta_path.read_text(encoding="utf-8"))
    if "energy_relaxed_eV" not in d:
        raise KeyError(f"{meta_path} missing 'energy_relaxed_eV'")
    return float(d["energy_relaxed_eV"])


def _compute_substitution_dopant_counts(
    counts_doped: Dict[str, int],
    host: str,
    anions: List[str],
) -> Dict[str, int]:
    """
    Substitution model on host sites:
    dopants are all species that are NOT host and NOT anions.
    """
    dopants: Dict[str, int] = {}
    for el, n in counts_doped.items():
        if el == host:
            continue
        if el in anions:
            continue
        dopants[el] = int(n)
    return dopants


def _write_candidate_meta(candidate_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir = candidate_dir / "04_formation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_formation(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> None:
    """
    Step 06: Compute formation energies for relaxed (and optionally filtered) candidates.

    Reads:
      - E_doped from candidate_*/02_relax/meta.json (energy_relaxed_eV)
      - mu, E_pristine from reference_structures/reference_energies.json

    Writes per composition folder:
      - formation_energies.csv
      - candidate_*/04_formation/meta.json
    """
    cfg = _parse_formation_config(raw_cfg, root)
    ref = _load_ref_json(root)

    E_pristine = float(ref["pristine"]["E_pristine_eV"])
    mu = {k: float(v) for k, v in (ref.get("mu_eV_per_atom", {}) or {}).items()}
    mu_host = mu.get(cfg.host_species, None)
    if mu_host is None:
        raise KeyError(
            f"Reference mu missing for host_species='{cfg.host_species}' in {REF_JSON}."
        )

    if not cfg.outdir.exists():
        raise FileNotFoundError(f"Output directory not found: {cfg.outdir} (did you run Step 01?)")

    folders = sorted([p for p in cfg.outdir.iterdir() if p.is_dir()])

    log.info("Step 06 formation: scanning %d folders in: %s", len(folders), cfg.outdir)
    log.info("Using reference file: %s", (root / REF_JSON).resolve())
    log.info("Output per folder: %s", OUT_CSV)

    for i, folder in enumerate(folders, start=1):
        out_csv = folder / OUT_CSV

        if cfg.skip_if_done and out_csv.exists():
            log.info("SKIP (%d/%d) %s: %s exists", i, len(folders), folder.name, OUT_CSV)
            continue

        poscars = _get_candidate_poscars(folder)
        if not poscars:
            log.info("SKIP (%d/%d) %s: no relaxed candidates found (run Step 03)", i, len(folders), folder.name)
            continue

        rows: List[Tuple[str, float, float, float, int, str]] = []

        for poscar in poscars:
            cand_dir = poscar.parents[1]  # candidate_XXX
            meta_path = cand_dir / RELAX_META
            if not meta_path.exists():
                log.warning("%s/%s: missing %s -> skip", folder.name, cand_dir.name, RELAX_META)
                continue

            E_doped = _load_relax_energy(meta_path)
            counts = _count_species_from_poscar(poscar)
            dop_counts = _compute_substitution_dopant_counts(counts, cfg.host_species, cfg.anion_species)
            n_dop_total = sum(dop_counts.values())

            # formation energy correction term
            corr = 0.0
            missing: List[str] = []
            for d, n in dop_counts.items():
                if d not in mu:
                    missing.append(d)
                    continue
                corr += float(n) * (mu_host - mu[d])

            if missing:
                log.warning("%s/%s: missing mu for %s -> skip", folder.name, cand_dir.name, missing)
                continue

            E_form_total = float(E_doped - E_pristine + corr)

            # normalization mode
            if cfg.normalize == "total":
                E_report = E_form_total
                norm_tag = "total_eV"
            elif cfg.normalize == "per_host":
                # NOTE: this is the *supercell total atoms* (as in your original script).
                # If you later want exact host-sublattice size, we can store it in refs JSON.
                n_atoms_supercell = int(ref["pristine"]["n_atoms_supercell"])
                E_report = E_form_total / float(n_atoms_supercell)
                norm_tag = "eV_per_supercell_atom"
            else:
                # per dopant atom
                if n_dop_total <= 0:
                    E_report = E_form_total
                    norm_tag = "total_eV"
                else:
                    E_report = E_form_total / float(n_dop_total)
                    norm_tag = "eV_per_dopant_atom"

            payload: Dict[str, Any] = {
                "stage": "04_formation",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "definition": ref.get("definition", ""),
                "host_species": cfg.host_species,
                "anion_species": cfg.anion_species,
                "E_doped_eV": float(E_doped),
                "E_pristine_eV": float(E_pristine),
                "mu_eV_per_atom": {cfg.host_species: float(mu_host), **{k: float(mu[k]) for k in dop_counts.keys()}},
                "dopant_counts": dop_counts,
                "E_form_eV_total": float(E_form_total),
                "reported": {"value": float(E_report), "unit": norm_tag},
            }
            _write_candidate_meta(cand_dir, payload)

            dop_str = ";".join([f"{k}:{v}" for k, v in sorted(dop_counts.items())]) if dop_counts else ""
            rows.append((cand_dir.name, E_doped, E_form_total, E_report, n_dop_total, dop_str))

        if not rows:
            log.info("SKIP %s: no valid candidates", folder.name)
            continue

        # Sort by total formation energy
        rows.sort(key=lambda x: x[2])

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "candidate",
                    "E_doped_eV",
                    "E_form_eV_total",
                    f"E_form_{cfg.normalize}",
                    "n_dopant_atoms",
                    "dopant_counts",
                ]
            )
            for cand, E_d, Eft, Erf, nd, dops in rows:
                w.writerow([cand, f"{E_d:.8f}", f"{Eft:.8f}", f"{Erf:.8f}", nd, dops])

        log.info("OK   %s: wrote %s (rows=%d)", folder.name, OUT_CSV, len(rows))

    log.info("DONE Step 06 formation.")


# TOML wrapper
try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_formation_from_toml(config_path: Path) -> None:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    run_formation(raw, root, config_path=config_path)

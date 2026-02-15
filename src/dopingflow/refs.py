from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Any

_RELAXER = None

log = logging.getLogger(__name__)

# Fixed paths (project convention)
REF_DIR = "reference_structures"
REF_JSON = Path("reference_structures/reference_energies.json")
MP_CACHE_DIR = Path("reference_structures/mp_cache")


# -----------------------------
# Config model for this step
# -----------------------------
@dataclass(frozen=True)
class RefConfig:
    supercell: tuple[int, int, int]
    host_species: str

    # preferred location for pristine reference:
    pristine_poscar: str  # path relative to workflow root

    # reference setup
    source: str                 # "local" or "mp"
    bulk_dir: Path              # absolute
    mp_ids: Dict[str, str]      # element -> mp-id
    fmax: float

    # caching
    skip_if_done: bool


def _silence_tensorflow_noise() -> None:
    """
    Must run before importing tensorflow/m3gnet in this process.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")
    warnings.filterwarnings("ignore", message=".*casting an input of type complex64.*")

    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    try:
        import tensorflow as tf  # noqa: F401

        tf.get_logger().setLevel("ERROR")
        try:
            tf.autograph.set_verbosity(0)
        except Exception:
            pass
    except Exception:
        # If TF isn't installed in this environment, ignore.
        pass


def _collect_dopants_from_input(raw: dict[str, Any]) -> Set[str]:
    """
    Collect all dopant elements that may appear, based on [doping] input.
    Works for mode="explicit" and mode="enumerate".
    """
    dop = raw.get("doping", {})
    mode = str(dop.get("mode", "explicit")).lower().strip()

    dopants: Set[str] = set()

    if mode == "explicit":
        comps = dop.get("compositions", []) or []
        for c in comps:
            if isinstance(c, dict):
                for el in c.keys():
                    dopants.add(str(el))
    else:
        for el in (dop.get("dopants", []) or []):
            dopants.add(str(el))
        for el in (dop.get("must_include", []) or []):
            dopants.add(str(el))

    dopants.discard("")
    return dopants


def _parse_ref_config(raw: dict[str, Any], root: Path) -> RefConfig:
    st = raw.get("structure", {}) or {}
    dop = raw.get("doping", {}) or {}
    refs = raw.get("references", {}) or {}

    sc = tuple(int(x) for x in st.get("supercell", [1, 1, 1]))
    if len(sc) != 3:
        raise ValueError("[structure].supercell must have 3 integers")

    host_species = str(dop.get("host_species", "")).strip()
    if not host_species:
        raise ValueError("[doping].host_species is required")

    # IMPORTANT: fix the mismatch from your original script.
    # Prefer [references].pristine_poscar; fallback to [structure].base_poscar for backward compatibility.
    pristine_poscar = str(refs.get("pristine_poscar", st.get("base_poscar", "POSCAR"))).strip()
    if not pristine_poscar:
        raise ValueError("pristine_poscar must be non-empty (set [references].pristine_poscar)")

    source = str(refs.get("source", "local")).strip().lower()
    if source not in {"local", "mp"}:
        raise ValueError("[references].source must be 'local' or 'mp'")

    bulk_dir_rel = Path(str(refs.get("bulk_dir", "reference_structures")))
    bulk_dir = (root / bulk_dir_rel).resolve()

    mp_ids = refs.get("mp_ids", {}) or {}
    mp_ids = {str(k): str(v) for k, v in mp_ids.items()}

    fmax = float(refs.get("fmax", 0.02))
    if fmax <= 0:
        raise ValueError("[references].fmax must be > 0")

    skip_if_done = bool(refs.get("skip_if_done", True))

    return RefConfig(
        supercell=sc,
        host_species=host_species,
        pristine_poscar=pristine_poscar,
        source=source,
        bulk_dir=bulk_dir,
        mp_ids=mp_ids,
        fmax=fmax,
        skip_if_done=skip_if_done,
    )


def _ensure_dirs(root: Path) -> None:
    (root / REF_DIR).mkdir(parents=True, exist_ok=True)
    (root / MP_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def _print_missing_local_list(bulk_dir: Path, elements: List[str]) -> None:
    missing = []
    for el in elements:
        p = bulk_dir / f"{el}.POSCAR"
        if not p.exists():
            missing.append(p)
    if missing:
        log.warning("Missing local bulk reference POSCARs:")
        for p in missing:
            log.warning("  - %s", p)
        log.warning("Create them and rerun, or switch to [references].source='mp' with mp_ids + MP_API_KEY.")


def _get_structure_from_local(bulk_dir: Path, element: str):
    from pymatgen.core import Structure

    path = bulk_dir / f"{element}.POSCAR"
    if not path.exists():
        raise FileNotFoundError(f"Missing local bulk reference: {path}")
    return Structure.from_file(str(path)), path


def _get_structure_from_mp(element: str, mpid: str):
    """
    Download a structure from Materials Project via mp-api.
    Requires: export MP_API_KEY="..."
    """
    from mp_api.client import MPRester

    api_key = os.environ.get("MP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MP_API_KEY env var not set, required when [references].source='mp'")

    with MPRester(api_key) as mpr:
        return mpr.get_structure_by_material_id(mpid)


def _relax_and_energy(struct, fmax: float) -> float:
    """
    Relax a structure with M3GNet Relaxer and return the final total energy (eV).
    """
    global _RELAXER
    _silence_tensorflow_noise()
    if _RELAXER is None:
        from m3gnet.models import Relaxer
        _RELAXER = Relaxer()

    res = _RELAXER.relax(struct, fmax=fmax, verbose=False)

    traj = res.get("trajectory", None)
    if traj is None or not hasattr(traj, "energies") or len(traj.energies) == 0:
        raise RuntimeError("Relaxer output has no trajectory energies.")
    return float(traj.energies[-1])


def run_refs_build(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> Path:
    """
    Build/cache reference energies needed for formation energy calculations.

    Writes: references/reference_energies.json
    Returns: path to written JSON
    """
    cfg = _parse_ref_config(raw_cfg, root)
    out_json = root / REF_JSON

    if cfg.skip_if_done and out_json.exists():
        log.info("Step refs.build: SKIP (exists): %s", out_json)
        log.info("Set [references].skip_if_done=false to force recomputation.")
        return out_json

    _ensure_dirs(root)

    dopants = sorted(_collect_dopants_from_input(raw_cfg))
    needed_elements: List[str] = [cfg.host_species] + dopants

    from pymatgen.core import Structure

    pristine_unit_path = (root / cfg.pristine_poscar).resolve()
    if not pristine_unit_path.exists():
        raise FileNotFoundError(
            "Pristine unit-cell POSCAR not found:\n"
            f"  {pristine_unit_path}\n\n"
            "Set it via [references].pristine_poscar in input.toml."
        )

    # 1) pristine supercell energy
    pristine_unit = Structure.from_file(str(pristine_unit_path))
    pristine_super = pristine_unit.copy()
    pristine_super.make_supercell(cfg.supercell)

    log.info("Computing E_pristine for supercell %s (fmax=%s)", cfg.supercell, cfg.fmax)
    log.info("Pristine unit-cell: %s", pristine_unit_path)
    t0 = time.time()
    E_pristine = _relax_and_energy(pristine_super, fmax=cfg.fmax)
    t_pristine = time.time() - t0
    log.info("REF pristine_supercell: E=%.6f eV (wall=%.1fs)", E_pristine, t_pristine)

    # 2) bulk chemical potentials per atom
    mu: Dict[str, float] = {}
    bulk_meta: Dict[str, dict] = {}

    if cfg.source == "local":
        _print_missing_local_list(cfg.bulk_dir, needed_elements)

    for el in needed_elements:
        t1 = time.time()

        if cfg.source == "local":
            s_bulk, src_path = _get_structure_from_local(cfg.bulk_dir, el)
            src = str(src_path)
        else:
            if el not in cfg.mp_ids:
                raise ValueError(
                    f"[references].source='mp' but no mp_ids entry for element '{el}'. "
                    "Add it under [references].mp_ids."
                )
            mpid = cfg.mp_ids[el]
            s_bulk = _get_structure_from_mp(el, mpid)

            cache_path = root / MP_CACHE_DIR / f"{el}_{mpid}.cif"
            try:
                s_bulk.to(fmt="cif", filename=str(cache_path))
                src = str(cache_path)
            except Exception:
                src = f"mp:{mpid}"

        E_bulk = _relax_and_energy(s_bulk, fmax=cfg.fmax)
        n = len(s_bulk)
        mu_el = E_bulk / float(n)

        mu[el] = float(mu_el)
        bulk_meta[el] = {
            "source": cfg.source,
            "source_path_or_id": src,
            "n_atoms_bulk": int(n),
            "E_bulk_eV": float(E_bulk),
            "mu_eV_per_atom": float(mu_el),
            "fmax": float(cfg.fmax),
            "walltime_s": float(time.time() - t1),
        }

        log.info("REF %s: E_bulk=%.6f eV, n=%d, mu=%.8f eV/atom", el, E_bulk, n, mu_el)

    ref = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "definition": "E_form = E_doped - E_pristine + sum_dopants n_dop * (mu_host - mu_dop)",
        "host_species": cfg.host_species,
        "supercell": list(cfg.supercell),

        "pristine": {
            "pristine_poscar_unitcell": str(pristine_unit_path),
            "n_atoms_unitcell": int(len(pristine_unit)),
            "n_atoms_supercell": int(len(pristine_super)),
            "E_pristine_eV": float(E_pristine),
            "fmax": float(cfg.fmax),
            "walltime_s": float(t_pristine),
        },

        "dopants_detected_from_input": dopants,
        "mu_eV_per_atom": mu,
        "bulk_details": bulk_meta,
    }

    if config_path is not None:
        ref["config_path"] = str(config_path.resolve())

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    log.info("Wrote reference energies: %s", out_json)
    return out_json

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_refs_build_from_toml(config_path: Path) -> Path:
    """
    Convenience wrapper used by the CLI:
    reads input.toml -> raw dict, sets root=config parent, calls run_refs_build.
    """
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    return run_refs_build(raw, root, config_path=config_path)

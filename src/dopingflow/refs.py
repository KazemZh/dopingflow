from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List

from dopingflow.ml_backends import (
    build_ase_calculator,
    check_backend_dependency,
    normalize_backend_config,
    prepare_backend_runtime,
    set_default_runtime_env,
)
from dopingflow.ml_relaxation import relax_structure_with_calculator

log = logging.getLogger(__name__)

set_default_runtime_env(tf_threads=1, omp_threads=1)

# Project convention
REF_DIR = Path("reference_structures")
REF_JSON = REF_DIR / "reference_energies.json"
RELAXED_DIR = REF_DIR / "relaxed"
RELAXED_REFS_DIR = RELAXED_DIR / "refs"

_CALCULATOR = None
_CALCULATOR_BACKEND = None
_CALCULATOR_MODEL = None
_CALCULATOR_TASK = None
_CALCULATOR_DEVICE = None


# -----------------------------
# Config model
# -----------------------------
@dataclass(frozen=True)
class RefConfig:
    reference_mode: str
    skip_if_done: bool

    fmax: float
    max_steps: int
    tf_threads: int
    omp_threads: int

    device: str
    gpu_id: int
    backend: str
    model: str
    task: str
    optimizer: str

    host: str
    host_dir: Path
    supercell: tuple[int, int, int]

    metal_ref: List[str]
    metals_dir: Path

    oxides_ref: List[str]
    oxides_dir: Path

    gas_ref: str
    gas_dir: Path
    oxygen_mode: str
    muO_shift_ev: float


# -----------------------------
# Utilities
# -----------------------------
def _ensure_dirs(root: Path) -> None:
    (root / REF_DIR).mkdir(parents=True, exist_ok=True)
    (root / RELAXED_DIR).mkdir(parents=True, exist_ok=True)
    (root / RELAXED_REFS_DIR).mkdir(parents=True, exist_ok=True)


def _file_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"POSCAR not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backend_package(backend: str) -> str:
    return {
        "mace": "mace-torch",
        "uma": "fairchem-core",
        "m3gnet": "m3gnet",
        "grace": "tensorpotential",
    }.get(str(backend).lower(), "unknown")


def _backend_package_version(backend: str) -> str:
    package = _backend_package(backend)
    if package == "unknown":
        return "unknown"
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _model_checkpoint_sha256(model: str, root: Path | None = None) -> str | None:
    path = Path(str(model)).expanduser()
    if not path.is_absolute() and root is not None:
        path = root / path
    return _file_sha256(path) if path.is_file() else None


def _relaxation_signature(
    cfg: RefConfig,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Settings that must match before a relaxed structure can be reused."""
    signature = {
        "schema_version": 2,
        "backend": cfg.backend,
        "model": cfg.model,
        "task": cfg.task,
        "backend_package": _backend_package(cfg.backend),
        "backend_package_version": _backend_package_version(cfg.backend),
        "optimizer": cfg.optimizer,
        "fmax": cfg.fmax,
        "max_steps": cfg.max_steps,
        "device": cfg.device,
        "gpu_id": cfg.gpu_id if cfg.device == "cuda" else None,
        "tf_threads": cfg.tf_threads,
        "omp_threads": cfg.omp_threads,
    }
    checkpoint_hash = _model_checkpoint_sha256(cfg.model, root)
    if checkpoint_hash is not None:
        signature["model_checkpoint_sha256"] = checkpoint_hash
    return signature


def _cache_entry_matches(
    entry: Any,
    *,
    source_path: Path,
    relaxed_path: Path,
    signature: dict[str, Any],
    ref_type: str | None = None,
    relaxed_hash_key: str = "relaxed_sha256",
) -> bool:
    if not isinstance(entry, dict) or not relaxed_path.exists():
        return False
    if ref_type is not None and entry.get("type") != ref_type:
        return False
    return (
        entry.get("source_sha256") == _file_sha256(source_path)
        and entry.get("relaxation_signature") == signature
        and entry.get(relaxed_hash_key) == _file_sha256(relaxed_path)
    )


def _parse_ref_config(raw: dict[str, Any], root: Path) -> RefConfig:
    refs = raw.get("references", {}) or {}

    reference_mode = str(refs.get("reference_mode", "metal")).strip().lower()
    if reference_mode not in {"metal", "oxide"}:
        raise ValueError("[references].reference_mode must be 'metal' or 'oxide'")

    skip_if_done = bool(refs.get("skip_if_done", True))

    fmax = float(refs.get("fmax", 0.02))
    max_steps = int(refs.get("max_steps", 300))
    tf_threads = int(refs.get("tf_threads", 1))
    omp_threads = int(refs.get("omp_threads", 1))

    if fmax <= 0:
        raise ValueError("[references].fmax must be > 0")
    if max_steps <= 0:
        raise ValueError("[references].max_steps must be > 0")
    if tf_threads <= 0:
        raise ValueError("[references].tf_threads must be > 0")
    if omp_threads <= 0:
        raise ValueError("[references].omp_threads must be > 0")

    device = str(refs.get("device", "cpu")).strip().lower()
    gpu_id = int(refs.get("gpu_id", 0))
    if device not in {"cpu", "cuda"}:
        raise ValueError('[references].device must be either "cpu" or "cuda"')
    if gpu_id < 0:
        raise ValueError("[references].gpu_id must be >= 0")

    backend = str(refs.get("backend", "m3gnet")).strip().lower()
    model = str(refs.get("model", "default")).strip()
    task = str(refs.get("task", "")).strip()
    optimizer = str(refs.get("optimizer", "bfgs")).strip().lower()

    if optimizer not in {"bfgs", "lbfgs", "fire", "mdmin", "quasinewton"}:
        raise ValueError(
            '[references].optimizer must be one of: "bfgs", "lbfgs", "fire", "mdmin", "quasinewton"'
        )

    backend, model, task = normalize_backend_config(
        backend=backend,
        model=model,
        task=task,
        section_name="references",
    )

    host = str(refs.get("host", "")).strip()
    if not host:
        raise ValueError("[references].host is required (e.g. 'SnO2')")

    host_dir = Path(str(refs.get("host_dir", "reference_structures/oxides")))
    host_dir = (root / host_dir).resolve()

    sc = refs.get("supercell", None)
    if sc is None:
        raise ValueError("[references].supercell is required (e.g. [5,2,1])")
    supercell = tuple(int(x) for x in sc)
    if len(supercell) != 3:
        raise ValueError("[references].supercell must have 3 integers")

    metal_ref = [str(x) for x in (refs.get("metal_ref", []) or [])]
    metals_dir = Path(str(refs.get("metals_dir", "reference_structures/metals")))
    metals_dir = (root / metals_dir).resolve()

    oxides_ref = [str(x) for x in (refs.get("oxides_ref", []) or [])]
    oxides_dir = Path(str(refs.get("oxides_dir", "reference_structures/oxides")))
    oxides_dir = (root / oxides_dir).resolve()

    gas_ref = str(refs.get("gas_ref", "O2")).strip()
    gas_dir = Path(str(refs.get("gas_dir", "reference_structures/gas")))
    gas_dir = (root / gas_dir).resolve()

    oxygen_mode = str(refs.get("oxygen_mode", "O-rich")).strip()
    if oxygen_mode not in {"O-rich", "O-poor"}:
        raise ValueError("[references].oxygen_mode must be 'O-rich' or 'O-poor'")

    muO_shift_ev = float(refs.get("muO_shift_ev", 0.0))

    if reference_mode == "metal":
        if not metal_ref:
            raise ValueError("reference_mode='metal' but [references].metal_ref is empty")
    else:
        if not oxides_ref:
            raise ValueError("reference_mode='oxide' but [references].oxides_ref is empty")
        if not gas_ref:
            raise ValueError("reference_mode='oxide' but [references].gas_ref is empty")

    return RefConfig(
        reference_mode=reference_mode,
        skip_if_done=skip_if_done,
        fmax=fmax,
        max_steps=max_steps,
        tf_threads=tf_threads,
        omp_threads=omp_threads,
        device=device,
        gpu_id=gpu_id,
        backend=backend,
        model=model,
        task=task,
        optimizer=optimizer,
        host=host,
        host_dir=host_dir,
        supercell=supercell,
        metal_ref=metal_ref,
        metals_dir=metals_dir,
        oxides_ref=oxides_ref,
        oxides_dir=oxides_dir,
        gas_ref=gas_ref,
        gas_dir=gas_dir,
        oxygen_mode=oxygen_mode,
        muO_shift_ev=muO_shift_ev,
    )


def _read_poscar(path: Path):
    from pymatgen.core import Structure

    if not path.exists():
        raise FileNotFoundError(f"POSCAR not found: {path}")
    return Structure.from_file(str(path))


def _write_poscar(struct, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    struct.to(fmt="poscar", filename=str(path))


def _ensure_calculator(cfg: RefConfig) -> None:
    global _CALCULATOR, _CALCULATOR_BACKEND, _CALCULATOR_MODEL, _CALCULATOR_TASK, _CALCULATOR_DEVICE

    if (
        _CALCULATOR is not None
        and _CALCULATOR_BACKEND == cfg.backend
        and _CALCULATOR_MODEL == cfg.model
        and _CALCULATOR_TASK == cfg.task
        and _CALCULATOR_DEVICE == cfg.device
    ):
        return

    prepare_backend_runtime(
        backend=cfg.backend,
        device=cfg.device,
        gpu_id=cfg.gpu_id,
        tf_threads=cfg.tf_threads,
        omp_threads=cfg.omp_threads,
    )

    _CALCULATOR = build_ase_calculator(
        backend=cfg.backend,
        model=cfg.model,
        task=cfg.task,
        device=cfg.device,
    )
    _CALCULATOR_BACKEND = cfg.backend
    _CALCULATOR_MODEL = cfg.model
    _CALCULATOR_TASK = cfg.task
    _CALCULATOR_DEVICE = cfg.device


def _relax_structure_and_energy(struct, cfg: RefConfig):
    """
    Unified structural relaxation using the same ASE calculator/optimizer route as relax.py.
    Returns:
      (relaxed_structure, final_energy_eV, n_steps, final_fmax, converged)
    """
    _ensure_calculator(cfg)
    return relax_structure_with_calculator(
        struct,
        calculator=_CALCULATOR,
        optimizer_name=cfg.optimizer,
        fmax=cfg.fmax,
        max_steps=cfg.max_steps,
    )


def _per_formula_unit_energy(struct, E_total: float) -> tuple[float, dict[str, float], float]:
    comp = struct.composition
    red = comp.reduced_composition
    red_dict = {str(el): float(amt) for el, amt in red.get_el_amt_dict().items()}

    el0 = next(iter(red_dict.keys()))
    amt0_total = float(comp.get_el_amt_dict()[el0])
    amt0_red = float(red_dict[el0])
    if amt0_red <= 0:
        raise ValueError("Invalid reduced composition.")
    n_fu = amt0_total / amt0_red
    E_fu = E_total / float(n_fu)
    return float(E_fu), red_dict, float(n_fu)


def _per_molecule_energy_O2(struct, E_total: float) -> float:
    comp = struct.composition.get_el_amt_dict()
    nO = float(comp.get("O", 0.0))
    if nO <= 0 or abs(nO / 2 - round(nO / 2)) > 1e-6:
        raise ValueError("O2 POSCAR must contain an even number of O atoms.")
    n_mol = nO / 2.0
    return float(E_total / n_mol)


# -----------------------------
# Main entry
# -----------------------------
def run_refs_build(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> Path:
    """
    Build/cache relaxed reference energies needed for formation energy calculations.

    Outputs:
      - reference_structures/reference_energies.json
      - reference_structures/relaxed/host_unit_relaxed.POSCAR
      - reference_structures/relaxed/host_supercell_<a>x<b>x<c>_relaxed.POSCAR
      - reference_structures/relaxed/refs/<name>_relaxed.POSCAR
    """
    cfg = _parse_ref_config(raw_cfg, root)
    out_json = root / REF_JSON
    _ensure_dirs(root)

    cached: dict[str, Any] = {}
    if cfg.skip_if_done and out_json.exists():
        try:
            cached = json.loads(out_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Ignoring unreadable reference cache: %s", out_json)

    signature = _relaxation_signature(cfg, root=root)

    host_path = (cfg.host_dir / f"{cfg.host}.POSCAR").resolve()
    log.info("Host POSCAR: %s", host_path)

    host_unit_relaxed_path = (root / RELAXED_DIR / "host_unit_relaxed.POSCAR").resolve()
    sc_tag = f"{cfg.supercell[0]}x{cfg.supercell[1]}x{cfg.supercell[2]}"
    host_super_relaxed_path = (
        root / RELAXED_DIR / f"host_supercell_{sc_tag}_relaxed.POSCAR"
    ).resolve()
    cached_host = cached.get("host", {})
    reuse_host = (
        cached_host.get("name") == cfg.host
        and cached_host.get("supercell") == list(cfg.supercell)
        and _cache_entry_matches(
            cached_host,
            source_path=host_path,
            relaxed_path=host_unit_relaxed_path,
            signature=signature,
            relaxed_hash_key="relaxed_unit_sha256",
        )
        and host_super_relaxed_path.exists()
        and cached_host.get("relaxed_supercell_sha256")
        == _file_sha256(host_super_relaxed_path)
    )

    if reuse_host:
        host_data = dict(cached_host)
        log.info("REF cache hit: host %s", cfg.host)
    else:
        check_backend_dependency(cfg.backend, stage_name="References")
        host_unit = _read_poscar(host_path)
        t0 = time.time()
        (
            host_unit_relaxed,
            E_host_unit,
            nsteps_unit,
            fmax_unit,
            conv_unit,
        ) = _relax_structure_and_energy(host_unit, cfg)
        t_unit = time.time() - t0
        _write_poscar(host_unit_relaxed, host_unit_relaxed_path)

        host_super = host_unit_relaxed.copy()
        host_super.make_supercell(cfg.supercell)
        t1 = time.time()
        (
            host_super_relaxed,
            E_host_super,
            nsteps_super,
            fmax_super,
            conv_super,
        ) = _relax_structure_and_energy(host_super, cfg)
        t_super = time.time() - t1
        _write_poscar(host_super_relaxed, host_super_relaxed_path)

        host_data = {
            "name": cfg.host,
            "source_poscar": str(host_path),
            "source_sha256": _file_sha256(host_path),
            "relaxation_signature": signature,
            "supercell": list(cfg.supercell),
            "relaxed_unit_poscar": str(host_unit_relaxed_path),
            "relaxed_supercell_poscar": str(host_super_relaxed_path),
            "relaxed_unit_sha256": _file_sha256(host_unit_relaxed_path),
            "relaxed_supercell_sha256": _file_sha256(host_super_relaxed_path),
            "n_atoms_unit": int(len(host_unit_relaxed)),
            "n_atoms_supercell": int(len(host_super_relaxed)),
            "E_unit_total_eV": float(E_host_unit),
            "E_supercell_total_eV": float(E_host_super),
            "E_unit_per_atom_eV": float(E_host_unit) / float(len(host_unit_relaxed)),
            "E_supercell_per_atom_eV": float(E_host_super) / float(len(host_super_relaxed)),
            "unit_optimizer_steps": int(nsteps_unit),
            "supercell_optimizer_steps": int(nsteps_super),
            "unit_final_fmax_eV_per_A": float(fmax_unit),
            "supercell_final_fmax_eV_per_A": float(fmax_super),
            "unit_converged": bool(conv_unit),
            "supercell_converged": bool(conv_super),
            "unit_walltime_s": float(t_unit),
            "supercell_walltime_s": float(t_super),
        }
        log.info("HOST relaxed and cached: %s", cfg.host)

    # --- 3) Relax reference structures ---
    references: Dict[str, dict] = {}
    cached_references = cached.get("references", {}) or {}

    def relax_ref(name: str, poscar_path: Path, ref_type: str) -> None:
        out_poscar = (root / RELAXED_REFS_DIR / f"{name}_relaxed.POSCAR").resolve()
        cached_entry = cached_references.get(name)
        if _cache_entry_matches(
            cached_entry,
            source_path=poscar_path,
            relaxed_path=out_poscar,
            signature=signature,
            ref_type=ref_type,
        ):
            references[name] = dict(cached_entry)
            log.info("REF cache hit: %s (%s)", name, ref_type)
            return

        check_backend_dependency(cfg.backend, stage_name="References")
        s = _read_poscar(poscar_path)
        t = time.time()
        s_relaxed, E, nsteps, fmax_final, converged = _relax_structure_and_energy(s, cfg)
        wall = time.time() - t

        _write_poscar(s_relaxed, out_poscar)

        entry: dict[str, Any] = {
            "type": ref_type,
            "source_poscar": str(poscar_path),
            "source_sha256": _file_sha256(poscar_path),
            "relaxation_signature": signature,
            "relaxed_poscar": str(out_poscar),
            "relaxed_sha256": _file_sha256(out_poscar),
            "n_atoms": int(len(s_relaxed)),
            "E_total_eV": float(E),
            "E_per_atom_eV": float(E) / float(len(s_relaxed)),
            "fmax_target_eV_per_A": float(cfg.fmax),
            "final_fmax_eV_per_A": float(fmax_final),
            "max_steps": int(cfg.max_steps),
            "optimizer": cfg.optimizer,
            "n_steps": int(nsteps),
            "converged": bool(converged),
            "walltime_s": float(wall),
            "backend": cfg.backend,
            "model": cfg.model,
            "task": cfg.task,
            "device": cfg.device,
            "gpu_id": cfg.gpu_id if cfg.device == "cuda" else None,
        }

        try:
            E_fu, red_dict, n_fu = _per_formula_unit_energy(s_relaxed, E)
            entry["E_per_formula_unit_eV"] = float(E_fu)
            entry["reduced_composition"] = red_dict
            entry["n_formula_units"] = float(n_fu)
        except Exception:
            pass

        if name == cfg.gas_ref:
            try:
                entry["E_per_molecule_eV"] = float(_per_molecule_energy_O2(s_relaxed, E))
            except Exception:
                pass

        references[name] = entry
        log.info(
            "REF %s (%s): E=%.6f eV, converged=%s, saved=%s",
            name,
            ref_type,
            E,
            converged,
            out_poscar,
        )

    # Both lists feed the phase diagram regardless of the formation-energy
    # reference mode. The mode still controls formation-energy semantics.
    for el in cfg.metal_ref:
        p = (cfg.metals_dir / f"{el}.POSCAR").resolve()
        relax_ref(el, p, ref_type="metal")

    for ox in cfg.oxides_ref:
        p = (cfg.oxides_dir / f"{ox}.POSCAR").resolve()
        relax_ref(ox, p, ref_type="oxide")

    # Oxygen closes oxide-containing phase diagrams. Preserve the old metal
    # mode behavior when no oxide references were requested.
    if cfg.reference_mode == "oxide" or cfg.oxides_ref:
        p_g = (cfg.gas_dir / f"{cfg.gas_ref}.POSCAR").resolve()
        relax_ref(cfg.gas_ref, p_g, ref_type="gas")

    # --- 4) Write JSON cache ---
    out: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_mode": cfg.reference_mode,
        "skip_if_done": cfg.skip_if_done,
        "backend": cfg.backend,
        "model": cfg.model,
        "task": cfg.task,
        "backend_package": signature["backend_package"],
        "backend_package_version": signature["backend_package_version"],
        "optimizer": cfg.optimizer,
        "device": cfg.device,
        "gpu_id": cfg.gpu_id if cfg.device == "cuda" else None,
        "fmax": cfg.fmax,
        "max_steps": cfg.max_steps,
        "tf_threads": cfg.tf_threads,
        "omp_threads": cfg.omp_threads,
        "supercell": list(cfg.supercell),
        "host": host_data,
        "references": references,
        "reference_inventory": {
            "metal_ref": cfg.metal_ref,
            "oxides_ref": cfg.oxides_ref,
            "gas_ref": (
                cfg.gas_ref
                if (cfg.reference_mode == "oxide" or cfg.oxides_ref)
                else None
            ),
        },
    }
    if "model_checkpoint_sha256" in signature:
        out["model_checkpoint_sha256"] = signature["model_checkpoint_sha256"]

    if cfg.reference_mode == "oxide":
        out["oxide_mode"] = {
            "oxides_ref": cfg.oxides_ref,
            "oxides_dir": str(cfg.oxides_dir),
            "gas_ref": cfg.gas_ref,
            "gas_dir": str(cfg.gas_dir),
            "oxygen_mode": cfg.oxygen_mode,
            "muO_shift_ev": cfg.muO_shift_ev,
        }
    else:
        out["metal_mode"] = {
            "metal_ref": cfg.metal_ref,
            "metals_dir": str(cfg.metals_dir),
        }

    if config_path is not None:
        out["config_path"] = str(config_path.resolve())

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote reference energies: %s", out_json)
    return out_json


# --- TOML loader wrapper used by CLI ---
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_refs_build_from_toml(config_path: Path) -> Path:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    return run_refs_build(raw, root, config_path=config_path)

"""Shared, provenance-checked GPAW single points for oxidation and transport.

Only identical electronic settings are interchangeable. A denser k mesh is not
silently substituted: it can change charge densities and occupations.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from dopingflow.oxidation import OptionalMethodUnavailable

DEFAULTS = {
    "code": "gpaw",
    "mode": "pw",
    "xc": "PBE",
    "ecut_eV": 500.0,
    "kpts": [1, 1, 1],
    "gamma": True,
    "smearing_eV": 0.05,
    "convergence_density": 1e-5,
    "maxiter": 333,
    "charge": 0.0,
    "spinpol": "auto",
    "initial_magmoms": {},
    "nbands": None,
}


def electronic_settings(settings):
    from dopingflow.oxidation_dft import _parse_initial_magmoms, _parse_kpts

    result = {key: settings.get(key, default) for key, default in DEFAULTS.items()}
    result["kpts"] = list(_parse_kpts(settings))
    result["initial_magmoms"] = _parse_initial_magmoms(settings)
    result["spinpol"] = str(result["spinpol"]).lower()
    for key in ("ecut_eV", "smearing_eV", "convergence_density", "charge"):
        result[key] = float(result[key])
        if not np.isfinite(result[key]):
            raise ValueError(f"{key} must be finite")
    return result


def structure_identity(path):
    structure = Structure.from_file(path)
    return {
        "species": [str(site.specie) for site in structure],
        "cell": np.round(structure.lattice.matrix, 10).tolist(),
        "positions": np.round(structure.cart_coords, 10).tolist(),
        "initial_magmoms": structure.site_properties.get("magmom"),
    }


def calculation_key(target, settings):
    try:
        gpaw_version = version("gpaw")
    except PackageNotFoundError:
        gpaw_version = None
    payload = {
        "schema": 1,
        "gpaw_version": gpaw_version,
        "setup_path": os.environ.get("GPAW_SETUP_PATH", ""),
        "structure": structure_identity(target.structure_path),
        "settings": electronic_settings(settings),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _stamp(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _valid(path, manifest, identity, wavefunctions):
    try:
        record = json.loads(manifest.read_text())
        return (
            record["identity"] == identity
            and record["file"] == _stamp(path)
            and path.stat().st_size > 0
            and (not wavefunctions or record["wavefunctions"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _record(path, manifest, identity, wavefunctions):
    temp = manifest.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {"identity": identity, "file": _stamp(path), "wavefunctions": wavefunctions}, indent=2
        )
    )
    temp.replace(manifest)


@contextmanager
def _lock(path):
    # HPC/Linux processes serialize identical single points. A crash releases flock.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _legacy_compatible(path, target, settings):
    """Adopt old results only with settings metadata AND actual geometry checks."""
    from dopingflow.oxidation_dft import _gpaw_restart

    # A mismatching modern manifest must NEVER fall back to old stage metadata:
    # the local GPW may have been replaced by a cache hit with different settings.
    if path.with_suffix(".provenance.json").exists():
        return False
    try:
        metadata = json.loads((path.parent / "gpaw_run_metadata.json").read_text())
        if electronic_settings(metadata["settings"]) != electronic_settings(settings):
            return False
        if settings.get("save_wavefunctions", False) and not metadata["settings"].get(
            "save_wavefunctions"
        ):
            return False
        atoms, _ = _gpaw_restart(path)
        expected = Structure.from_file(target.structure_path)
        return (
            atoms.get_chemical_symbols() == [s.specie.symbol for s in expected]
            and np.allclose(atoms.cell.array, expected.lattice.matrix, atol=1e-8, rtol=0)
            and np.allclose(atoms.positions, expected.cart_coords, atol=1e-8, rtol=0)
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OptionalMethodUnavailable):
        return False


def ensure_gpaw(target, cfg, settings):
    """Return (stage-local GPW, reused). No execution unless explicitly enabled.

    Cache copies survive stage cleanup. Stage copies keep all existing oxidation
    postprocessors compatible; their sidecars are bound to the exact GPW stamp.
    """
    from dopingflow.oxidation_dft import _run_gpaw_single_point, _workdir

    try:
        from gpaw.mpi import world
    except ImportError:
        world = None
    if world is not None and world.size != 1:
        raise OptionalMethodUnavailable(
            "Shared DFT orchestration must run in one process (not mpiexec); multi-rank execution is not yet supported"
        )
    key, identity = calculation_key(target, settings)
    root = Path(settings.get("cache_root", cfg.source_root / "dft_cache")).expanduser()
    if not root.is_absolute():
        root = cfg.root / root
    cache = root / key
    local = _workdir(target, cfg, settings) / str(settings.get("gpw_file", "oxidation.gpw"))
    manifest = local.with_suffix(".provenance.json")
    shared = cache / "result.gpw"
    shared_manifest = cache / "provenance.json"
    wavefunctions = bool(settings.get("save_wavefunctions", False))
    reuse = bool(settings.get("reuse_existing", True))
    with _lock(root / f"{key}.lock"), _lock(local.with_suffix(".lock")):
        if reuse and _valid(local, manifest, identity, wavefunctions):
            # Register valid stage outputs for the other analysis, even if cache was removed.
            if not _valid(shared, shared_manifest, identity, wavefunctions):
                cache.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local, shared)
                actual = json.loads(manifest.read_text())["wavefunctions"]
                _record(shared, shared_manifest, identity, actual)
            return local, True
        if reuse and _valid(shared, shared_manifest, identity, wavefunctions):
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(shared, local)
            actual = json.loads(shared_manifest.read_text())["wavefunctions"]
            _record(local, manifest, identity, actual)
            return local, True
        # Legacy oxidation results can be reused before their first cache registration.
        candidates = [local]
        for directory in settings.get("reuse_workdirs", []):
            candidates.append(Path(directory) / str(settings.get("gpw_file", "oxidation.gpw")))
        if reuse:
            for candidate in candidates:
                if candidate.is_file() and _legacy_compatible(candidate, target, settings):
                    local.parent.mkdir(parents=True, exist_ok=True)
                    if candidate.resolve() != local.resolve():
                        shutil.copy2(candidate, local)
                    actual = json.loads((candidate.parent / "gpaw_run_metadata.json").read_text())[
                        "settings"
                    ].get("save_wavefunctions", False)
                    _record(local, manifest, identity, actual)
                    cache.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local, shared)
                    _record(shared, shared_manifest, identity, actual)
                    return local, True
        if not settings.get("execute", False):
            raise OptionalMethodUnavailable(
                f"No compatible, verified GPAW result for {local}. Enable execute=true to calculate; "
                "unverified files, changed geometry/settings, or missing wavefunctions are not reused."
            )
        # Invalidate before running so interrupted replacement cannot retain stale provenance.
        manifest.unlink(missing_ok=True)
        local = _run_gpaw_single_point(target, cfg, settings)
        _record(local, manifest, identity, wavefunctions)
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, shared)
        _record(shared, shared_manifest, identity, wavefunctions)
        return local, False


def derived_current(path, gpw):
    """Bind derived analysis to both its own bytes' stamp and the source GPW."""
    try:
        payload = json.loads(path.with_suffix(path.suffix + ".source.json").read_text())
        return payload == {"gpw": _stamp(gpw), "artifact": _stamp(path)}
    except (OSError, ValueError):
        return False


def record_derived(path, gpw):
    path.with_suffix(path.suffix + ".source.json").write_text(
        json.dumps({"gpw": _stamp(gpw), "artifact": _stamp(path)})
    )

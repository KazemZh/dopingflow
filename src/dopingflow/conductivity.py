"""Band-transport screening on selected relaxed parent and vacancy structures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dopingflow.oxidation import (
    OptionalMethodUnavailable,
    _csv_write,
    _json_write,
    discover_oxidation_targets,
    parse_oxidation_config,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _finite_list(section, name, default, positive=False):
    values = section.get(name, default)
    if not isinstance(values, list) or not values:
        raise ValueError(f"[conductivity].{name} must be a nonempty array")
    result = [float(v) for v in values]
    if not all(np.isfinite(v) and (not positive or v > 0) for v in result):
        raise ValueError(f"Invalid [conductivity].{name}")
    return result


def parse_config(raw, root):
    section = dict(raw.get("conductivity", {}) or {})
    section.setdefault("enabled", False)  # Never append expensive work to old run-all inputs.
    section["temperatures_K"] = _finite_list(section, "temperatures_K", [300.0], True)
    section["excess_electrons_cm3"] = _finite_list(section, "excess_electrons_cm3", [0.0])
    if section.get("relaxation_time_fs") is not None:
        tau = float(section["relaxation_time_fs"])
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("relaxation_time_fs must be finite and positive")
        section["relaxation_time_fs"] = tau
    for key, default, minimum in (("interpolation_factor", 5, 2), ("dos_points", 4000, 100)):
        value = section.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        section[key] = value
    if not isinstance(section.get("target_include", []), list):
        raise TypeError("target_include must be an array")
    selection = {
        key: section[key]
        for key in (
            "source_root",
            "include_vacancy_free",
            "include_oxygen_vacancies",
            "target_include",
        )
        if key in section
    }
    oxidation_section = raw.get("oxidation", {}) or {}
    if "source_root" not in selection and oxidation_section.get("source_root"):
        selection["source_root"] = oxidation_section["source_root"]
    selection.update(
        {"strategy": "structural", "output_dir": section.get("output_dir", "07_conductivity")}
    )
    cfg = parse_oxidation_config(
        {"structure": raw.get("structure", {}), "oxidation": selection}, root
    )
    # Inherit physical settings to maximize bidirectional reuse; transport mesh is an
    # explicit override because Gamma-only oxidation data cannot resolve velocities.
    dft = dict((raw.get("oxidation", {}) or {}).get("dft_electronic", {}) or {})
    ox = oxidation_section
    if "dft-auto" in ox.get("methods", []) or (
        ox.get("strategy") == "dft" and not ox.get("methods")
    ):
        from dopingflow.dft_cache import DEFAULTS

        auto = ox.get("dft_auto", {}) or {}
        dft.update(
            {
                k: v
                for k, v in auto.items()
                if k in DEFAULTS or k in {"save_wavefunctions", "gpw_file", "cache_root"}
            }
        )
    dft.update(section.get("dft", {}) or {})
    dft.setdefault("kpts", [4, 4, 4])
    dft.setdefault("save_wavefunctions", True)
    dft["execute"] = bool((section.get("dft", {}) or {}).get("execute", False))
    dft["output_root"] = (section.get("dft", {}) or {}).get("output_root", "dft_conductivity")
    # Do not inherit an oxidation stage's explicit directory.
    dft.pop("workdir", None)
    if (section.get("dft", {}) or {}).get("workdir"):
        dft["workdir"] = section["dft"]["workdir"]
    from dopingflow.oxidation_dft import _parse_kpts

    if any(n < 2 for n in _parse_kpts(dft)):
        raise ValueError(
            "3D conductivity needs kpts >= 2 in every direction; converge the mesh (Gamma-only is insufficient)"
        )
    if str(dft.get("code", "gpaw")).lower() != "gpaw":
        raise ValueError("Conductivity currently supports the GPAW backend only")
    if section.get("transport_regime", "band") != "band":
        raise ValueError(
            "Only transport_regime='band' is implemented; localized carriers need a hopping model"
        )
    section["dft"] = dft
    return cfg, section


def discover_targets(cfg):
    """Discover targets with exactly the same rules used by oxidation analysis."""
    return discover_oxidation_targets(cfg)


def select_targets(cfg, section=None):
    """Return the discovered targets.

    Conductivity intentionally shares oxidation's source_root, vacancy toggles,
    and target_include semantics so users do not need to learn a second
    structure-selection interface.
    """
    return discover_targets(cfg)


def gpaw_bands(path):
    """Explicitly preserve both collinear spin channels (upstream reader drops one)."""
    from BoltzTraP2.units import Angstrom, eV

    from dopingflow.oxidation_dft import _gpaw_restart

    atoms, calc = _gpaw_restart(path)
    nk = len(calc.get_ibz_k_points())
    if nk < 2:
        raise ValueError("At least two irreducible k points are required; converge a 3D mesh")
    nspin = calc.get_number_of_spins()
    if nspin not in (1, 2):
        raise ValueError("Only nonmagnetic and collinear-spin GPAW calculations are supported")
    energies = (
        np.concatenate(
            [
                np.array([calc.get_eigenvalues(kpt=k, spin=s) for k in range(nk)]).T
                for s in range(nspin)
            ]
        )
        * eV
    )
    # Use final site moments to avoid imposing nonmagnetic crystal symmetries.
    magmom = np.asarray(calc.get_magnetic_moments()) if nspin == 2 else None
    lattice = np.array(atoms.cell).T * Angstrom
    return SimpleNamespace(
        atoms=atoms,
        kpoints=np.asarray(calc.get_ibz_k_points()),
        ebands=energies,
        mommat=None,
        magmom=magmom,
        get_lattvec=lambda: lattice,
        nelect=float(calc.get_number_of_electrons()),
        dosweight=2.0 if nspin == 1 else 1.0,
        fermi=float(calc.get_fermi_level()) * eV,
    )


def integrate_transport(data, section):
    from BoltzTraP2 import bandlib, fite, sphere
    from BoltzTraP2.units import BOLTZMANN, eV

    lattice = data.get_lattvec()
    equiv = sphere.get_equivalences(
        data.atoms, data.magmom, section["interpolation_factor"] * len(data.kpoints)
    )
    coeffs = fite.fitde3D(data, equiv)
    bands, velocities, _ = fite.getBTPbands(equiv, coeffs, lattice, curvature=False)
    energy, dos, vvdos, _ = bandlib.BTPDOS(bands, velocities, npts=section["dos_points"])
    volume_cm3 = float(data.atoms.get_volume()) * 1e-24
    volume_au = abs(float(np.linalg.det(lattice)))
    rows = []
    for temperature in section["temperatures_K"]:
        margin = 10 * BOLTZMANN * temperature
        if energy[-1] - energy[0] <= 2 * margin:
            raise ValueError(
                "Insufficient band-energy range for this temperature; include more bands"
            )
        for excess in section["excess_electrons_cm3"]:
            electrons = data.nelect + excess * volume_cm3
            if electrons <= 0:
                raise ValueError("Requested excess carrier concentration leaves no electrons")
            mu = bandlib.solve_for_mu(
                energy, dos, electrons, temperature, dosweight=data.dosweight, refine=True
            )
            if not energy[0] + margin < mu < energy[-1] - margin:
                raise ValueError(
                    "Chemical potential is too close to sampled band limits; increase nbands / energy range"
                )
            Tr, mur = np.array([temperature]), np.array([mu])
            counts, L0, L1, L2, _ = bandlib.fermiintegrals(
                energy, dos, vvdos, mur, Tr, dosweight=data.dosweight
            )
            if abs(-counts[0, 0] - electrons) > 1e-4:
                raise ValueError("Carrier-count integration did not converge; increase dos_points")
            sigma, _, _, _ = bandlib.calc_Onsager_coefficients(L0, L1, L2, mur, Tr, volume_au)
            tensor = np.asarray(sigma[0, 0])
            if not np.isfinite(tensor).all():
                raise ValueError("Nonfinite transport tensor")
            row = {
                "temperature_K": temperature,
                "excess_electrons_cm3": excess,
                "chemical_potential_relative_to_dft_fermi_eV": float((mu - data.fermi) / eV),
                "sigma_over_tau_S_per_m_per_s": tensor.tolist(),
                "sigma_over_tau_trace_average_S_per_m_per_s": float(np.trace(tensor) / 3),
            }
            if section.get("relaxation_time_fs") is not None:
                tau = section["relaxation_time_fs"] * 1e-15
                row.update(
                    {
                        "assumed_relaxation_time_fs": section["relaxation_time_fs"],
                        "conditional_sigma_S_per_m": (tensor * tau).tolist(),
                        "conditional_sigma_trace_average_S_per_cm": float(
                            np.trace(tensor) * tau / 300
                        ),
                    }
                )
            rows.append(row)
    return rows


def run_conductivity(raw, root, *, dry_run=False):
    if not (raw.get("conductivity", {}) or {}).get("enabled", False):
        return None
    cfg, section = parse_config(raw, root)
    targets, warnings = select_targets(cfg, section)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    selection = [
        {
            "target_id": t.target_id,
            "structure_path": str(t.structure_path),
            "kind": t.kind,
            **t.metadata,
        }
        for t in targets
    ]
    _json_write(cfg.output_dir / "selected_structures.json", selection)
    results, csv_rows = [], []
    for target in targets:
        record = {
            "target_id": target.target_id,
            "kind": target.kind,
            "structure_path": str(target.structure_path),
            "status": "selected",
        }
        if not dry_run:
            try:
                # Check optional transport dependency before launching expensive DFT.
                try:
                    import BoltzTraP2  # noqa: F401
                except ImportError as exc:
                    raise OptionalMethodUnavailable(
                        "Install dopingflow[conductivity] and GPAW to run transport"
                    ) from exc
                from dopingflow.dft_cache import ensure_gpaw
                from dopingflow.oxidation_dft import _workdir

                settings = dict(section["dft"])
                oxidation = (raw.get("oxidation", {}) or {}).get("dft_electronic", {}) or {}
                ox = raw.get("oxidation", {}) or {}
                if "dft-auto" in ox.get("methods", []) or (
                    ox.get("strategy") == "dft" and not ox.get("methods")
                ):
                    oxidation = {**oxidation, **(ox.get("dft_auto", {}) or {})}
                if oxidation.get("output_root") == "gpaw_oxidation":
                    oxidation = {**oxidation, "output_root": "dft_oxidation"}
                settings["reuse_workdirs"] = [str(_workdir(target, cfg, oxidation))]
                gpw, reused = ensure_gpaw(target, cfg, settings)
                record.update({"gpw_file": str(gpw), "dft_reused": reused})
                # This is explicitly a band-like hypothesis; do not invent localization
                # classification from a DOS or silently switch to a hopping model.
                if section.get("transport_regime", "band") != "band":
                    raise OptionalMethodUnavailable(
                        "Localized/hopping transport requires a separate validated hopping-rate model"
                    )
                rows = integrate_transport(gpaw_bands(gpw), section)
                record.update(
                    {
                        "status": "calculated",
                        "transport_assumption": "band-like, constant relaxation time",
                        "rows": rows,
                    }
                )
                for row in rows:
                    csv_rows.append({"target_id": target.target_id, **row})
            except Exception as exc:
                record.update(
                    {
                        "status": "unavailable"
                        if isinstance(exc, OptionalMethodUnavailable)
                        else "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if section.get("fail_fast", False):
                    _json_write(
                        cfg.output_dir / "conductivity_results.json",
                        {"results": results + [record]},
                    )
                    raise
        results.append(record)
        # Hash target IDs to avoid safe_id collisions across discovered targets.
        key = (
            target.safe_id[:100] + "-" + hashlib.sha256(target.target_id.encode()).hexdigest()[:12]
        )
        _json_write(cfg.output_dir / "structures" / key / "conductivity.json", record)
    output = cfg.output_dir / "conductivity_results.json"
    _json_write(
        output,
        {
            "schema_version": 1,
            "dry_run": dry_run,
            "selection": selection,
            "warnings": warnings,
            "results": results,
            "settings": section,
            "limitations": [
                "Target discovery uses the same source_root, vacancy toggles, and target_include rules as oxidation-state analysis.",
                "Band-like transport is a hypothesis requiring localization checks; polaron hopping and AMSET scattering are not calculated by this stage.",
                "sigma/tau is not absolute conductivity. Any sigma uses the explicitly assumed relaxation time.",
                "Positive excess_electrons_cm3 adds electrons to the explicit structure; negative removes them. Zero preserves its DFT electron count. This is not a defect-ionization or mobile-carrier prediction.",
                "Converge k mesh, interpolation, empty bands and DOS grid. Periodic vacancies do not model random-defect scattering or grain boundaries.",
            ],
        },
    )
    _csv_write(cfg.output_dir / "conductivity.csv", csv_rows)
    return output


def run_conductivity_from_toml(config_path, *, dry_run=False):
    path = Path(config_path).resolve()
    return run_conductivity(tomllib.loads(path.read_text()), path.parent, dry_run=dry_run)

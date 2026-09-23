"""Band-transport screening on selected relaxed parent and vacancy structures."""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dopingflow.oxidation import (
    OptionalMethodUnavailable,
    StructureTarget,
    _csv_write,
    _json_write,
    _target_output_dir,
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

    comparison = section.get("comparison", {}) or {}
    if not isinstance(comparison, dict):
        raise TypeError("[conductivity.comparison] must be a TOML table")
    comparison = dict(comparison)
    comparison["enabled"] = bool(comparison.get("enabled", False))
    comparison["reference_target"] = str(
        comparison.get("reference_target", "")
    ).strip()
    comparison["reference_structure_path"] = str(
        comparison.get("reference_structure_path", "")
    ).strip()
    comparison["reference_source_root"] = str(
        comparison.get("reference_source_root", "")
    ).strip()
    comparison["reference_label"] = str(
        comparison.get("reference_label", "Reference")
    ).strip() or "Reference"
    comparison["reference_composition"] = str(
        comparison.get("reference_composition", "")
    ).strip()
    # Backward compatibility for inputs written by the ATO-specific GUI.
    if "reference_sb_percent" in comparison:
        legacy_sb = float(comparison["reference_sb_percent"])
        if not np.isfinite(legacy_sb) or legacy_sb <= 0:
            raise ValueError(
                "[conductivity.comparison].reference_sb_percent must be finite and positive"
            )
        comparison["reference_sb_percent"] = legacy_sb
        if not comparison["reference_composition"]:
            comparison["reference_composition"] = f"{legacy_sb:g}% Sb"
    comparison["basis"] = str(
        comparison.get("basis", "reference-benchmark")
    ).strip().lower()
    legacy_basis_map = {
        "ato-5pct-sb-benchmark": "reference-benchmark",
        "fixed-sb": "fixed-composition",
    }
    comparison["basis"] = legacy_basis_map.get(
        comparison["basis"], comparison["basis"]
    )
    allowed_bases = {
        "reference-benchmark",
        "same-total-dopant",
        "fixed-composition",
        "custom",
    }
    if comparison["basis"] not in allowed_bases:
        raise ValueError(
            "[conductivity.comparison].basis must be reference-benchmark, "
            "same-total-dopant, fixed-composition, or custom"
        )
    if comparison["enabled"] and not (
        comparison["reference_target"]
        or comparison["reference_structure_path"]
        or comparison["reference_source_root"]
    ):
        raise ValueError(
            "[conductivity.comparison] needs a reference target, source, or structure path "
            "when comparison is enabled"
        )
    section["comparison"] = comparison
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


def _solve_mu_for_count(bandlib, energy, dos, electrons, temperature, dosweight):
    """Solve the finite-T chemical potential against the actual DOS electron count.

    BoltzTraP2.solve_for_mu(refine=True) uses a bounded minimization whose default
    energy tolerance can leave an electron-count residual larger than DopingFlow's
    validation threshold for steep DOS features. Increasing DOS bins does not fix
    that optimizer tolerance. Start from the upstream estimate, then refine the
    electron-count equation with a bracketed root solve when needed.
    """
    mu = bandlib.solve_for_mu(
        energy, dos, electrons, temperature, dosweight=dosweight, refine=True
    )

    def count_residual(mu_value):
        integrated = -float(
            bandlib.calc_N(
                energy, dos, float(mu_value), temperature, dosweight=dosweight
            )
        )
        return integrated - electrons

    residual = count_residual(mu)
    tolerance = max(1.0e-6, 1.0e-8 * abs(electrons))
    if abs(residual) <= tolerance:
        return float(mu), float(residual), tolerance

    low = float(energy[0])
    high = float(energy[-1])
    low_residual = count_residual(low)
    high_residual = count_residual(high)
    if low_residual == 0.0:
        return low, 0.0, tolerance
    if high_residual == 0.0:
        return high, 0.0, tolerance
    if low_residual * high_residual > 0.0:
        raise ValueError(
            "Requested carrier count is not bracketed by the sampled band-energy range; "
            "increase nbands / energy range rather than dos_points. "
            f"Electron-count residuals at the DOS limits are {low_residual:.3e} and "
            f"{high_residual:.3e} electrons."
        )

    from scipy.optimize import brentq

    mu = brentq(
        count_residual,
        low,
        high,
        xtol=1.0e-13,
        rtol=max(4.0 * np.finfo(float).eps, 1.0e-14),
        maxiter=200,
    )
    residual = count_residual(mu)
    if abs(residual) > tolerance:
        raise ValueError(
            "Carrier-count root solve did not converge to the requested electron count: "
            f"residual={residual:.3e} electrons (tolerance={tolerance:.3e})."
        )
    return float(mu), float(residual), tolerance

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
            mu, mu_count_residual, count_tolerance = _solve_mu_for_count(
                bandlib,
                energy,
                dos,
                electrons,
                temperature,
                data.dosweight,
            )
            if not energy[0] + margin < mu < energy[-1] - margin:
                raise ValueError(
                    "Chemical potential is too close to sampled band limits; increase nbands / energy range"
                )
            Tr, mur = np.array([temperature]), np.array([mu])
            counts, L0, L1, L2, _ = bandlib.fermiintegrals(
                energy, dos, vvdos, mur, Tr, dosweight=data.dosweight
            )
            integrated_electrons = float(-counts[0, 0])
            carrier_count_residual = integrated_electrons - electrons
            if abs(carrier_count_residual) > count_tolerance:
                raise ValueError(
                    "Carrier-count integration is inconsistent after chemical-potential "
                    "root refinement: "
                    f"residual={carrier_count_residual:.3e} electrons "
                    f"(tolerance={count_tolerance:.3e})."
                )
            sigma, _, _, _ = bandlib.calc_Onsager_coefficients(L0, L1, L2, mur, Tr, volume_au)
            tensor = np.asarray(sigma[0, 0])
            if not np.isfinite(tensor).all():
                raise ValueError("Nonfinite transport tensor")
            # Human-readable transport unit used by the GUI and comparison tables.
            # 1 (S m^-1 s^-1) = 1e-17 (S cm^-1 fs^-1).
            tensor_S_per_cm_per_fs = tensor * 1.0e-17
            trace_average_S_per_cm_per_fs = float(
                np.trace(tensor_S_per_cm_per_fs) / 3
            )
            row = {
                "temperature_K": temperature,
                "excess_electrons_cm3": excess,
                "chemical_potential_relative_to_dft_fermi_eV": float((mu - data.fermi) / eV),
                "target_electron_count": float(electrons),
                "integrated_electron_count": integrated_electrons,
                "carrier_count_residual_electrons": carrier_count_residual,
                "chemical_potential_solver_residual_electrons": mu_count_residual,
                "sigma_over_tau_S_per_cm_per_fs": tensor_S_per_cm_per_fs.tolist(),
                "sigma_over_tau_trace_average_S_per_cm_per_fs": trace_average_S_per_cm_per_fs,
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




def _reference_matches(target_id, selector):
    """Match an exact/safe target ID or shell-style wildcard selector."""
    target_id = str(target_id).replace("\\", "/")
    selector = str(selector).strip().replace("\\", "/")
    if not selector:
        return False
    safe_id = target_id.replace("/", "__")
    safe_selector = selector.replace("/", "__")
    return (
        fnmatch.fnmatchcase(target_id, selector)
        or fnmatch.fnmatchcase(safe_id, safe_selector)
    )


def _resolved_path(value, root):
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _reference_target_id_from_structure(path):
    path = Path(path).resolve()
    candidate_dir = path.parent.parent if path.parent.name == "02_relax" else path.parent
    if candidate_dir.parent != candidate_dir:
        return f"{candidate_dir.parent.name}/{candidate_dir.name}"
    return "reference/reference"


def _candidate_structure_from_path(path):
    """Resolve a file/directory candidate hint to the relaxed structure file."""
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        return None
    for candidate in (
        path / "02_relax" / "POSCAR",
        path / "02_relax" / "CONTCAR",
        path / "POSCAR",
        path / "CONTCAR",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_reference_structure_hint(value, root):
    """Accept a POSCAR/CIF, candidate directory, or simple candidate glob.

    A common GUI input is a path ending in candidate_003/*. Treat this as the
    candidate directory and prefer 02_relax/POSCAR rather than requiring the
    user to know the exact relaxed-file path.
    """
    text = str(value or "").strip()
    if not text:
        return None

    expanded = os.path.expanduser(text)
    if expanded.endswith("/*"):
        base = _resolved_path(expanded[:-2], root)
        resolved = _candidate_structure_from_path(base)
        if resolved is not None:
            return resolved

    direct = _resolved_path(expanded, root)
    resolved = _candidate_structure_from_path(direct)
    if resolved is not None:
        return resolved

    if any(char in expanded for char in "*?[]"):
        pattern = expanded
        if not Path(pattern).is_absolute():
            pattern = str(Path(root) / pattern)
        candidates = []
        for match in sorted(glob.glob(pattern)):
            resolved = _candidate_structure_from_path(match)
            if resolved is not None and resolved not in candidates:
                candidates.append(resolved)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                "Reference path pattern matched multiple relaxed structures: "
                + ", ".join(str(path) for path in candidates[:8])
            )
    return None

def _reference_cfg(raw, root, cfg, comparison, *, source_root=None):
    source_value = source_root
    if source_value is None:
        source_value = comparison.get("reference_source_root") or str(cfg.source_root)
    source_root = _resolved_path(source_value, root)
    selector = str(comparison.get("reference_target", "")).strip()
    selection = {
        "source_root": str(source_root),
        "strategy": "structural",
        "include_vacancy_free": True,
        "include_oxygen_vacancies": False,
        "target_include": [selector] if selector else [],
        # DFT work directories remain controlled by conductivity.dft.output_root.
        "output_dir": str(cfg.output_dir),
    }
    return parse_oxidation_config(
        {"structure": raw.get("structure", {}), "oxidation": selection},
        Path(root),
    )


def _automatic_reference_roots(raw, root, cfg, comparison):
    """Roots searched when the user does not explicitly set reference_source_root."""
    explicit_root = str(comparison.get("reference_source_root", "")).strip()
    if explicit_root:
        return [_resolved_path(explicit_root, root)]

    values = []
    structure = raw.get("structure", {}) or {}
    if structure.get("outdir"):
        values.append(_resolved_path(structure["outdir"], root))
    values.append(cfg.source_root)

    roots = []
    seen = set()
    for value in values:
        resolved = Path(value).resolve()
        key = str(resolved)
        if key not in seen:
            roots.append(resolved)
            seen.add(key)
    return roots


def discover_reference_candidates(raw, root, cfg, comparison):
    """Return vacancy-free candidates for the persistent conductivity reference.

    When no explicit reference root is supplied, search the project's normal
    structure output first and the conductivity source root second. The reference
    can be any user-selected material/structure; 5% Sb ATO is only one use case.
    """
    explicit = str(comparison.get("reference_structure_path", "")).strip()
    source_hint = str(comparison.get("reference_source_root", "")).strip()
    target_hint = str(comparison.get("reference_target", "")).strip()
    structure_path = _resolve_reference_structure_hint(explicit, root) if explicit else None
    resolved_from_target_hint = False

    # Be forgiving when a candidate directory (or candidate_003/*) was pasted
    # into either the source-root or target-selector field.
    if structure_path is None and source_hint:
        structure_path = _resolve_reference_structure_hint(source_hint, root)
    if structure_path is None and target_hint:
        structure_path = _resolve_reference_structure_hint(target_hint, root)
        resolved_from_target_hint = structure_path is not None

    if explicit and structure_path is None:
        raise FileNotFoundError(
            "Reference structure path could not be resolved. Provide a POSCAR/CIF, "
            "a candidate directory containing 02_relax/POSCAR, or a pattern resolving "
            f"to one candidate: {explicit}"
        )

    if structure_path is not None:
        target_id = target_hint
        if resolved_from_target_hint or not target_id or any(char in target_id for char in "*?[]"):
            target_id = _reference_target_id_from_structure(structure_path)
        explicit_comparison = dict(comparison)
        explicit_comparison["reference_source_root"] = ""
        return _reference_cfg(raw, root, cfg, explicit_comparison), [
            StructureTarget(
                target_id=target_id,
                parent_id=target_id,
                kind="vacancy-free",
                structure_path=structure_path,
                n_vacancies=0,
                vacancy_species=None,
                metadata={
                    "reference_label": comparison.get("reference_label", "Reference"),
                    "reference_composition": comparison.get("reference_composition", ""),
                    "explicit_reference_structure": True,
                },
            )
        ]

    searched = []
    matches = []
    first_cfg = None
    errors = []
    for candidate_root in _automatic_reference_roots(raw, root, cfg, comparison):
        searched.append(str(candidate_root))
        try:
            ref_cfg = _reference_cfg(
                raw, root, cfg, comparison, source_root=candidate_root
            )
            if first_cfg is None:
                first_cfg = ref_cfg
            candidates, _ = discover_oxidation_targets(ref_cfg)
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(f"{candidate_root}: {exc}")
            continue
        for target in candidates:
            if target.kind != "vacancy-free" or int(target.n_vacancies or 0) != 0:
                continue
            key = (target.target_id, str(target.structure_path.resolve()))
            if key not in {(t.target_id, str(t.structure_path.resolve())) for _, t in matches}:
                matches.append((ref_cfg, target))

    if not matches:
        details = "; ".join(errors)
        message = (
            "Conductivity reference matched no vacancy-free structure. "
            f"Searched reference roots: {', '.join(searched) or '(none)'}. "
            "Set an exact reference_source_root/reference_target or provide "
            "reference_structure_path."
        )
        if details:
            message += f" Discovery details: {details}"
        raise OptionalMethodUnavailable(message)

    # One unique target is required. If the same target exists in more than one
    # structure tree, the resolved structure path keeps those alternatives distinct.
    if len(matches) != 1:
        names = ", ".join(
            f"{target.target_id} @ {ref_cfg.source_root}"
            for ref_cfg, target in matches[:8]
        )
        suffix = "" if len(matches) <= 8 else ", ..."
        raise ValueError(
            "Conductivity reference selector must resolve to exactly one vacancy-free "
            f"structure across the searched roots; matched {len(matches)}: {names}{suffix}"
        )
    ref_cfg, target = matches[0]
    return ref_cfg, [target]


def _reference_store_path(cfg, comparison):
    label = str(comparison.get("reference_label", "Reference"))
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label).strip("_")
    return cfg.output_dir / "references" / (slug or "Reference") / "reference.json"


def _transport_settings_fingerprint(section):
    """Fingerprint method/settings shared by every candidate in one comparison set."""
    from dopingflow.dft_cache import electronic_settings

    try:
        gpaw_version = version("gpaw")
    except PackageNotFoundError:
        gpaw_version = None
    payload = {
        "schema": 1,
        "gpaw_version": gpaw_version,
        "setup_path": os.environ.get("GPAW_SETUP_PATH", ""),
        "dft_settings": electronic_settings(section["dft"]),
        "temperatures_K": [float(v) for v in section["temperatures_K"]],
        "excess_electrons_cm3": [
            float(v) for v in section["excess_electrons_cm3"]
        ],
        "interpolation_factor": int(section["interpolation_factor"]),
        "dos_points": int(section["dos_points"]),
        "transport_regime": section.get("transport_regime", "band"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest, payload


def _reference_transport_fingerprint(target, section):
    """Fingerprint reference geometry plus the common comparison settings."""
    from dopingflow.dft_cache import calculation_key

    dft_key, dft_identity = calculation_key(target, section["dft"])
    settings_fingerprint, settings_payload = _transport_settings_fingerprint(section)
    payload = {
        "schema": 3,
        "target_id": target.target_id,
        "structure_path": str(target.structure_path.resolve()),
        "dft_key": dft_key,
        "dft_identity": dft_identity,
        "transport_settings_fingerprint": settings_fingerprint,
        "transport_settings": settings_payload,
        "reference_composition": str(
            section.get("comparison", {}).get("reference_composition", "")
        ),
        "legacy_reference_sb_percent": section.get("comparison", {}).get(
            "reference_sb_percent"
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest, payload


def _load_persistent_reference(path, fingerprint, fingerprint_payload=None):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("schema_version") != 3:
        return None
    if payload.get("record", {}).get("status") != "calculated":
        return None
    if payload.get("fingerprint") == fingerprint:
        return payload

    # Reference label/composition are metadata, not physics. Older reference
    # fingerprints included ATO-specific composition fields, so accept a cached
    # record when the actual structure, DFT identity, and transport settings match.
    if fingerprint_payload is not None:
        old = payload.get("fingerprint_payload", {}) or {}
        physical_keys = (
            "target_id",
            "structure_path",
            "dft_key",
            "transport_settings_fingerprint",
        )
        if all(old.get(key) == fingerprint_payload.get(key) for key in physical_keys):
            return payload
    return None


def prepare_persistent_reference(raw, root, cfg, section, *, dry_run=False):
    """Load or calculate the reusable user-selected conductivity reference."""
    comparison = section.get("comparison", {})
    if not comparison.get("enabled", False):
        return None, []

    warnings = []
    ref_cfg, candidates = discover_reference_candidates(
        raw, root, cfg, comparison
    )
    if not candidates:
        raise OptionalMethodUnavailable(
            "Conductivity reference matched no vacancy-free structure."
        )
    if len(candidates) != 1:
        names = ", ".join(target.target_id for target in candidates[:8])
        suffix = "" if len(candidates) <= 8 else ", ..."
        raise ValueError(
            "Conductivity reference selector must resolve to exactly one vacancy-free "
            f"structure; matched {len(candidates)}: {names}{suffix}"
        )
    target = candidates[0]
    fingerprint, fingerprint_payload = _reference_transport_fingerprint(
        target, section
    )
    store = _reference_store_path(cfg, comparison)
    cached = _load_persistent_reference(
        store, fingerprint, fingerprint_payload
    )
    if cached is not None:
        record = dict(cached["record"])
        record["persistent_reference_reused"] = True
        record["reference_store"] = str(store)
        record["reference_label"] = comparison.get("reference_label", "Reference")
        record["reference_composition"] = comparison.get("reference_composition", "")
        if comparison.get("reference_sb_percent") is not None:
            record["reference_sb_percent"] = comparison.get("reference_sb_percent")
        record["comparison_basis"] = comparison.get(
            "basis", "reference-benchmark"
        )
        record["reference_fingerprint"] = fingerprint
        record["transport_settings_fingerprint"] = fingerprint_payload[
            "transport_settings_fingerprint"
        ]
        return record, warnings

    record = {
        "target_id": target.target_id,
        "kind": target.kind,
        "n_oxygen_vacancies": 0,
        "structure_path": str(target.structure_path),
        "reference_label": comparison.get("reference_label", "Reference"),
        "reference_composition": comparison.get("reference_composition", ""),
        "comparison_basis": comparison.get("basis", "reference-benchmark"),
        "status": "selected" if dry_run else "pending",
        "persistent_reference_reused": False,
        "reference_store": str(store),
        "reference_fingerprint": fingerprint,
        "transport_settings_fingerprint": fingerprint_payload[
            "transport_settings_fingerprint"
        ],
    }
    if comparison.get("reference_sb_percent") is not None:
        record["reference_sb_percent"] = comparison.get("reference_sb_percent")
    if dry_run:
        return record, warnings

    try:
        import BoltzTraP2  # noqa: F401
    except ImportError as exc:
        raise OptionalMethodUnavailable(
            "Install dopingflow[conductivity] and GPAW to calculate the conductivity reference"
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
    settings["reuse_workdirs"] = [str(_workdir(target, ref_cfg, oxidation))]

    gpw, reused = ensure_gpaw(target, ref_cfg, settings)
    rows = integrate_transport(gpaw_bands(gpw), section)
    record.update(
        {
            "status": "calculated",
            "gpw_file": str(gpw),
            "dft_reused": reused,
            "transport_assumption": "band-like, constant relaxation time",
            "rows": rows,
        }
    )
    payload = {
        "schema_version": 3,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "record": record,
    }
    _json_write(store, payload)
    return record, warnings


def collect_compatible_transport_results(output_dir, settings_fingerprint, current_results=()):
    """Collect prior per-target results compatible with the current comparison settings."""
    records = {}
    root = Path(output_dir) / "structures"
    if root.exists():
        for path in root.rglob("conductivity.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if (
                record.get("status") == "calculated"
                and record.get("transport_settings_fingerprint") == settings_fingerprint
                and record.get("target_id")
            ):
                records[str(record["target_id"])] = record
    # Current in-memory results take precedence over older copies.
    for record in current_results:
        if (
            record.get("status") == "calculated"
            and record.get("transport_settings_fingerprint") == settings_fingerprint
            and record.get("target_id")
        ):
            records[str(record["target_id"])] = record
    return list(records.values())


def build_reference_comparison(results, comparison, reference=None):
    """Compare every calculated target with one persistent user-selected reference.

    The same reference is used across the screened set, including vacancy-containing
    structures. Temperature and rigid-band excess-electron concentration must match.
    """
    if not comparison.get("enabled", False):
        return [], []
    if not reference or reference.get("status") != "calculated":
        return [], ["Reference conductivity is not available yet."]

    reference_rows = {
        (
            float(row["temperature_K"]),
            float(row["excess_electrons_cm3"]),
        ): row
        for row in reference.get("rows", [])
        if row.get("sigma_over_tau_trace_average_S_per_cm_per_fs") is not None
    }
    label = str(comparison.get("reference_label", "Reference"))
    basis = str(
        comparison.get("basis", "reference-benchmark")
    )
    reference_vacancies = int(reference.get("n_oxygen_vacancies", 0) or 0)

    rows = []
    warnings = []
    for result in results:
        if result.get("status") != "calculated":
            continue
        for row in result.get("rows", []):
            key = (
                float(row["temperature_K"]),
                float(row["excess_electrons_cm3"]),
            )
            reference_row = reference_rows.get(key)
            if reference_row is None:
                continue
            value = float(row["sigma_over_tau_trace_average_S_per_cm_per_fs"])
            reference_value = float(
                reference_row["sigma_over_tau_trace_average_S_per_cm_per_fs"]
            )
            if not np.isfinite(value) or not np.isfinite(reference_value):
                continue
            if abs(reference_value) <= np.finfo(float).tiny:
                warnings.append(
                    f"Reference sigma/tau is zero for {key}; ratio cannot be calculated."
                )
                continue
            ratio = value / reference_value
            rows.append(
                {
                    "target_id": result["target_id"],
                    "reference_target_id": reference["target_id"],
                    "reference_label": label,
                    "reference_composition": comparison.get(
                        "reference_composition", ""
                    ),
                    "reference_sb_percent": comparison.get("reference_sb_percent"),
                    "comparison_basis": basis,
                    "structure_kind": result.get("kind"),
                    "n_oxygen_vacancies": int(
                        result.get("n_oxygen_vacancies", 0) or 0
                    ),
                    "reference_n_oxygen_vacancies": reference_vacancies,
                    "temperature_K": key[0],
                    "excess_electrons_cm3": key[1],
                    "sigma_over_tau_trace_average_S_per_cm_per_fs": value,
                    "reference_sigma_over_tau_trace_average_S_per_cm_per_fs": reference_value,
                    "relative_to_reference": ratio,
                    "percent_change_vs_reference": 100.0 * (ratio - 1.0),
                }
            )
    return rows, warnings

def rebuild_reference_comparison(raw, root):
    """Calculate/reuse only the selected reference and rebuild comparison tables.

    Existing target conductivity.json files are read from disk; no screened target
    GPAW or BoltzTraP2 calculation is rerun by this operation.
    """
    if not (raw.get("conductivity", {}) or {}).get("enabled", False):
        raise ValueError("Conductivity is disabled; enable [conductivity] first")

    cfg, section = parse_config(raw, root)
    comparison = section.get("comparison", {})
    if not comparison.get("enabled", False):
        raise ValueError(
            "Reference comparison is disabled; enable [conductivity.comparison] first"
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    reference_summary_path = cfg.output_dir / "conductivity_reference.json"
    reference_summary_path.unlink(missing_ok=True)

    try:
        reference_record, reference_warnings = prepare_persistent_reference(
            raw, root, cfg, section, dry_run=False
        )
        warnings.extend(reference_warnings)
    except Exception as exc:
        message = (
            "Conductivity reference unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        reference_record = {
            "reference_label": comparison.get("reference_label", "Reference"),
            "reference_composition": comparison.get("reference_composition", ""),
            "comparison_basis": comparison.get("basis", "reference-benchmark"),
            "status": "unavailable",
            "error": message,
            "persistent_reference_reused": False,
        }
        if comparison.get("reference_sb_percent") is not None:
            reference_record["reference_sb_percent"] = comparison.get("reference_sb_percent")
        _json_write(reference_summary_path, reference_record)
        raise

    _json_write(reference_summary_path, reference_record)
    settings_fingerprint, _ = _transport_settings_fingerprint(section)
    compatible_results = collect_compatible_transport_results(
        cfg.output_dir, settings_fingerprint
    )
    comparison_rows, comparison_warnings = build_reference_comparison(
        compatible_results, comparison, reference_record
    )
    warnings.extend(comparison_warnings)

    comparison_csv = cfg.output_dir / "conductivity_comparison.csv"
    comparison_json = cfg.output_dir / "conductivity_comparison.json"
    _csv_write(comparison_csv, comparison_rows)
    _json_write(comparison_json, comparison_rows)

    results_json = cfg.output_dir / "conductivity_results.json"
    if results_json.exists():
        try:
            payload = json.loads(results_json.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {}
        payload["reference"] = reference_record
        payload["comparison"] = comparison_rows
        payload["comparison_compatible_target_count"] = len(compatible_results)
        existing_warnings = payload.get("warnings", []) or []
        # A successful reference-only rebuild supersedes reference-unavailable
        # diagnostics left behind by an earlier failed attempt. Preserve unrelated
        # target warnings, but remove stale reference failures so the GUI reflects
        # the current calculated reference state.
        stale_reference_prefixes = (
            "Conductivity reference unavailable:",
            "Reference conductivity is not available yet.",
            "ATO 5% Sb reference unavailable:",  # legacy
            "ATO 5% Sb reference conductivity is not available yet.",  # legacy
        )
        existing_warnings = [
            warning
            for warning in existing_warnings
            if not any(str(warning).startswith(prefix) for prefix in stale_reference_prefixes)
        ]
        payload["warnings"] = list(dict.fromkeys([*existing_warnings, *warnings]))
        _json_write(results_json, payload)

    return comparison_json

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
            "n_oxygen_vacancies": t.n_vacancies if t.vacancy_species == "O" else 0,
            **t.metadata,
        }
        for t in targets
    ]
    _json_write(cfg.output_dir / "selected_structures.json", selection)
    transport_settings_fingerprint, transport_settings_payload = (
        _transport_settings_fingerprint(section)
    )

    reference_record = None
    if section.get("comparison", {}).get("enabled", False):
        reference_summary_path = cfg.output_dir / "conductivity_reference.json"
        reference_summary_path.unlink(missing_ok=True)
        try:
            reference_record, reference_warnings = prepare_persistent_reference(
                raw, root, cfg, section, dry_run=dry_run
            )
            warnings.extend(reference_warnings)
        except Exception as exc:
            message = (
                "Conductivity reference unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
            warnings.append(message)
            reference_record = {
                "reference_label": section.get("comparison", {}).get(
                    "reference_label", "Reference"
                ),
                "reference_composition": section.get("comparison", {}).get(
                    "reference_composition", ""
                ),
                "comparison_basis": section.get("comparison", {}).get(
                    "basis", "reference-benchmark"
                ),
                "status": "unavailable",
                "error": message,
                "persistent_reference_reused": False,
            }
            if section.get("fail_fast", False) and not dry_run:
                _json_write(
                    cfg.output_dir / "conductivity_reference.json",
                    reference_record,
                )
                raise
        if reference_record is not None:
            _json_write(
                cfg.output_dir / "conductivity_reference.json",
                reference_record,
            )

    results, csv_rows, structure_index = [], [], []
    for target in targets:
        target_dir = _target_output_dir(cfg, target)
        record = {
            "target_id": target.target_id,
            "kind": target.kind,
            "n_oxygen_vacancies": (
                target.n_vacancies if target.vacancy_species == "O" else 0
            ),
            "structure_path": str(target.structure_path),
            "output_directory": str(target_dir),
            "status": "selected",
            "transport_settings_fingerprint": transport_settings_fingerprint,
            "transport_settings": transport_settings_payload,
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
        _json_write(target_dir / "conductivity.json", record)
        summary = {
            "target_id": target.target_id,
            "structure_kind": target.kind,
            "n_oxygen_vacancies": record["n_oxygen_vacancies"],
            "structure_path": str(target.structure_path),
            "status": record["status"],
            "dft_reused": record.get("dft_reused"),
            "gpw_file": record.get("gpw_file"),
            "error": record.get("error"),
            "transport_settings_fingerprint": record.get(
                "transport_settings_fingerprint"
            ),
            "output_directory": str(target_dir),
        }
        _json_write(target_dir / "summary.json", summary)
        structure_index.append(summary)

    compatible_results = collect_compatible_transport_results(
        cfg.output_dir,
        transport_settings_fingerprint,
        results,
    )
    comparison_rows, comparison_warnings = build_reference_comparison(
        compatible_results, section.get("comparison", {}), reference_record
    )
    warnings.extend(comparison_warnings)
    _csv_write(cfg.output_dir / "conductivity_comparison.csv", comparison_rows)
    _json_write(cfg.output_dir / "conductivity_comparison.json", comparison_rows)

    _csv_write(cfg.output_dir / "conductivity_structure_index.csv", structure_index)
    _json_write(cfg.output_dir / "conductivity_structure_index.json", structure_index)
    output = cfg.output_dir / "conductivity_results.json"
    _json_write(
        output,
        {
            "schema_version": 1,
            "dry_run": dry_run,
            "selection": selection,
            "warnings": warnings,
            "results": results,
            "comparison": comparison_rows,
            "comparison_compatible_target_count": len(compatible_results),
            "reference": reference_record,
            "settings": section,
            "limitations": [
                "Target discovery uses the same source_root, vacancy toggles, and target_include rules as oxidation-state analysis.",
                "Band-like transport is a hypothesis requiring localization checks; polaron hopping and AMSET scattering are not calculated by this stage.",
                "sigma/tau is not absolute conductivity. The primary human-readable unit is S cm^-1 fs^-1; raw SI S m^-1 s^-1 is retained. Any sigma uses the explicitly assumed relaxation time.",
                "Reference-normalized comparisons use the selected persistent vacancy-free benchmark at the same temperature and excess-electron concentration. The same reference is shared across all compatible screened structures and vacancy counts.",
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


def rebuild_reference_comparison_from_toml(config_path):
    path = Path(config_path).resolve()
    return rebuild_reference_comparison(
        tomllib.loads(path.read_text()), path.parent
    )

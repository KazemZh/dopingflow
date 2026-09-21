from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from pymatgen.core import Element, Structure

from dopingflow.oxidation import (
    OptionalMethodUnavailable,
    OxidationConfig,
    StructureTarget,
    base_method_result,
)


def _by_site(records: Sequence[dict[str, Any]], key: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for rec in records:
        if rec.get("site_index") is None or rec.get(key) is None:
            continue
        try:
            out[int(rec["site_index"])] = float(rec[key])
        except (TypeError, ValueError):
            pass
    return out


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    c = median(values)
    return float(median([abs(float(v) - c) for v in values]))


def _candidate_states(symbol: str, bader_mean: float | None = None) -> list[int]:
    element = Element(symbol)
    values: list[int] = []
    for value in (*element.common_oxidation_states, *element.oxidation_states):
        value = int(value)
        if value not in values:
            values.append(value)
    if symbol == "O":
        return [-2, -1, 0]
    if bader_mean is not None and bader_mean > 0.25:
        positive = [value for value in values if value > 0]
        if positive:
            values = positive
    elif bader_mean is not None and bader_mean < -0.25:
        negative = [value for value in values if value < 0]
        if negative:
            values = negative
    return values[:6] or [0]


def _short_oo(structure: Structure, cutoff: float) -> bool:
    oxygen = [i for i, site in enumerate(structure) if site.specie.symbol == "O"]
    return any(
        structure.get_distance(i, j) <= cutoff
        for pos, i in enumerate(oxygen)
        for j in oxygen[pos + 1 :]
    )


def _local_bv_prior(
    structure: Structure,
    settings: dict[str, Any],
) -> tuple[list[int] | None, dict[str, Any]]:
    """Local bond-valence prior without forcing global ionic charge neutrality."""
    radius = float(settings.get("bond_valence_max_radius", 4.0))
    try:
        from pymatgen.analysis.bond_valence import calculate_bv_sum

        sums: list[float] = []
        states: list[int] = []
        for site in structure:
            value = float(calculate_bv_sum(site, structure.get_neighbors(site, radius)))
            sums.append(value)
            candidates = _candidate_states(site.specie.symbol)
            signed = [
                state
                for state in candidates
                if value == 0
                or state == 0
                or math.copysign(1.0, state) == math.copysign(1.0, value)
            ]
            pool = signed or candidates
            states.append(min(pool, key=lambda state: abs(float(state) - value)))
        return states, {
            "status": "available",
            "implementation": "pymatgen.analysis.bond_valence.calculate_bv_sum",
            "local_bond_valence_sums": sums,
            "max_radius_angstrom": radius,
            "charge_neutrality_forced": False,
        }
    except Exception as exc:
        return None, {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _fallback_uniform_prior(
    structure: Structure,
    bader: dict[int, float],
    settings: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, site in enumerate(structure):
        groups[site.specie.symbol].append(i)

    symbols = list(groups)
    candidates: dict[str, list[int]] = {}
    ranks: dict[str, dict[int, int]] = {}
    has_oo = _short_oo(
        structure,
        float(settings.get("peroxide_oo_cutoff_angstrom", 1.8)),
    )
    for symbol in symbols:
        q = [bader[i] for i in groups[symbol] if i in bader]
        values = _candidate_states(symbol, sum(q) / len(q) if q else None)
        if symbol == "O" and not has_oo:
            values = [-2]
        candidates[symbol] = values
        ranks[symbol] = {value: rank for rank, value in enumerate(values)}

    carrier_penalty = float(settings.get("fallback_carrier_penalty", 0.20))
    uncommon_penalty = float(settings.get("fallback_uncommon_state_penalty", 1.0))
    max_comp = float(settings.get("max_abs_electronic_compensation_e", 12.0))
    best: tuple[float, tuple[int, ...], float] | None = None
    for choice in product(*(candidates[s] for s in symbols)):
        formal_sum = float(sum(len(groups[s]) * state for s, state in zip(symbols, choice)))
        if abs(formal_sum) > max_comp:
            continue
        score = carrier_penalty * abs(formal_sum) + uncommon_penalty * sum(
            ranks[s][state] for s, state in zip(symbols, choice)
        )
        candidate = (score, tuple(choice), formal_sum)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        choice = tuple(candidates[s][0] for s in symbols)
        formal_sum = float(sum(len(groups[s]) * state for s, state in zip(symbols, choice)))
    else:
        _, choice, formal_sum = best
    chosen = dict(zip(symbols, choice))
    return [int(chosen[site.specie.symbol]) for site in structure], {
        "status": "fallback",
        "element_states": chosen,
        "formal_charge_sum_e": formal_sum,
    }


def _load_refs(cfg: OxidationConfig, settings: dict[str, Any]) -> dict[str, Any]:
    refs = dict(settings.get("bader_reference_fingerprints", {}) or {})
    raw = str(settings.get("bader_reference_file") or "").strip()
    if not raw:
        return refs
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (cfg.root / path).resolve()
    if not path.is_file():
        raise OptionalMethodUnavailable(f"Bader reference fingerprint file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Bader reference fingerprint file must contain a JSON object")
    payload.update(refs)
    return payload


def _reference_pick(
    symbol: str,
    q: float,
    refs: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[int, float] | None:
    raw = refs.get(symbol)
    if not isinstance(raw, dict):
        return None
    default_sigma = float(settings.get("reference_default_sigma_e", 0.08))
    scored: list[tuple[float, int]] = []
    for state_raw, item in raw.items():
        try:
            state = int(state_raw)
        except (TypeError, ValueError):
            continue
        mean = None
        sigma = default_sigma
        if isinstance(item, (int, float)):
            mean = float(item)
        elif isinstance(item, list) and item:
            vals = [float(value) for value in item]
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                variance = sum((value - mean) ** 2 for value in vals) / (len(vals) - 1)
                sigma = max(default_sigma, math.sqrt(max(variance, 0.0)))
        elif isinstance(item, dict):
            if item.get("mean") is not None:
                mean = float(item["mean"])
            elif item.get("bader_partial_charge") is not None:
                mean = float(item["bader_partial_charge"])
            if item.get("std") is not None:
                sigma = max(default_sigma, abs(float(item["std"])))
        if mean is not None:
            scored.append((abs(q - mean) / max(default_sigma, sigma), state))
    if not scored:
        return None
    scored.sort()
    best_z, state = scored[0]
    second_z = scored[1][0] if len(scored) > 1 else math.inf
    if (
        best_z <= float(settings.get("reference_max_z", 2.5))
        and second_z - best_z >= float(settings.get("reference_min_z_gap", 1.0))
    ):
        return state, best_z
    return None


def _mixed_supported(
    indices: Sequence[int],
    states: Sequence[int],
    bader: dict[int, float],
    magmom: dict[int, float],
    settings: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, state in zip(indices, states):
        groups[int(state)].append(int(index))
    if len(groups) <= 1:
        return True, {"supported": True, "reason": "uniform"}

    ordered = sorted(groups)
    q_medians: dict[int, float] = {}
    q_mads: dict[int, float] = {}
    q_ok = True
    for state in ordered:
        vals = [bader[i] for i in groups[state] if i in bader]
        if len(vals) != len(groups[state]):
            q_ok = False
            break
        q_medians[state] = float(median(vals))
        q_mads[state] = _mad(vals)
    bader_support = q_ok
    deltas: list[float] = []
    if q_ok:
        for low, high in zip(ordered[:-1], ordered[1:]):
            delta = q_medians[high] - q_medians[low]
            need = max(
                float(settings.get("bader_mixed_min_separation_e", 0.12)),
                3.0 * max(q_mads.get(low, 0.0), q_mads.get(high, 0.0), 0.01),
            )
            deltas.append(delta)
            if delta < need:
                bader_support = False

    m_medians: dict[int, float] = {}
    m_ok = True
    for state in ordered:
        vals = [abs(magmom[i]) for i in groups[state] if i in magmom]
        if len(vals) != len(groups[state]):
            m_ok = False
            break
        m_medians[state] = float(median(vals))
    mag_support = (
        m_ok
        and len(ordered) == 2
        and abs(m_medians[ordered[1]] - m_medians[ordered[0]])
        >= float(settings.get("magmom_mixed_min_separation_muB", 0.50))
    )
    return bader_support or mag_support, {
        "supported": bader_support or mag_support,
        "states": ordered,
        "group_sizes": {str(state): len(groups[state]) for state in ordered},
        "bader_medians": q_medians,
        "bader_adjacent_deltas": deltas,
        "bader_supported": bader_support,
        "magmom_medians_abs": m_medians,
        "magmom_supported": mag_support,
    }


def synthesize_oxidation_states(
    structure: Structure,
    *,
    prior_states: Sequence[int],
    bader_records: Sequence[dict[str, Any]],
    magnetic_records: Sequence[dict[str, Any]] = (),
    settings: dict[str, Any] | None = None,
    reference_fingerprints: dict[str, Any] | None = None,
    band_edge_analysis: dict[str, Any] | None = None,
    wannier_analysis: dict[str, Any] | None = None,
    cell_charge_e: float = 0.0,
) -> dict[str, Any]:
    settings = dict(settings or {})
    refs = dict(reference_fingerprints or {})
    if len(prior_states) != len(structure):
        raise ValueError("prior oxidation-state list does not match structure")

    states = [int(value) for value in prior_states]
    initial = list(states)
    bader = _by_site(bader_records, "bader_partial_charge")
    magmom = _by_site(magnetic_records, "magnetic_moment")
    evidence: dict[int, list[str]] = defaultdict(list)
    confidence = [0.55] * len(structure)
    locked: set[int] = set()

    for i, site in enumerate(structure):
        if i not in bader:
            continue
        picked = _reference_pick(site.specie.symbol, bader[i], refs, settings)
        if picked is not None:
            states[i], z = picked
            locked.add(i)
            confidence[i] += 0.25
            evidence[i].append(
                f"same-method Bader reference favors {states[i]:+d} (z={z:.2f})"
            )

    groups: dict[str, list[int]] = defaultdict(list)
    for i, site in enumerate(structure):
        groups[site.specie.symbol].append(i)

    mixed_diagnostics: dict[str, Any] = {}
    has_oo = _short_oo(
        structure,
        float(settings.get("peroxide_oo_cutoff_angstrom", 1.8)),
    )
    for element, indices in groups.items():
        vals = [states[i] for i in indices]
        if len(set(vals)) == 1:
            q = [bader[i] for i in indices if i in bader]
            if len(q) >= 2 and max(q) - min(q) <= float(
                settings.get("bader_uniform_max_range_e", 0.10)
            ):
                for i in indices:
                    confidence[i] += 0.12
                    evidence[i].append(
                        f"{element} Bader charges form one narrow population"
                    )
            continue

        if element == "O" and has_oo:
            mixed_diagnostics[element] = {"supported": True, "reason": "short O-O pair"}
            continue

        supported, diag = _mixed_supported(indices, vals, bader, magmom, settings)
        mixed_diagnostics[element] = diag
        if supported:
            for i in indices:
                confidence[i] += 0.18
                evidence[i].append("DFT descriptors support mixed-valence site separation")
            continue

        unlocked = [i for i in indices if i not in locked]
        if not unlocked:
            continue
        dominant = Counter(states[i] for i in unlocked).most_common(1)[0][0]
        for i in unlocked:
            states[i] = dominant
            confidence[i] += 0.08
            evidence[i].append(
                f"unsupported mixed-{element} prior collapsed to dominant {element}{dominant:+d}"
            )

    formal_sum = float(sum(states))
    compensation_charge = float(cell_charge_e) - formal_sum
    tol = float(settings.get("electronic_compensation_tolerance_e", 0.25))
    if abs(compensation_charge) <= tol:
        comp_type, comp_count = "none", 0.0
    elif compensation_charge < 0:
        comp_type, comp_count = "electrons", abs(compensation_charge)
    else:
        comp_type, comp_count = "holes", abs(compensation_charge)

    localization = "not-evaluated"
    edge_used = None
    if band_edge_analysis and comp_type != "none":
        edge_used = "HOMO" if comp_type == "electrons" else "LUMO"
        localization = str(
            (band_edge_analysis.get(edge_used.lower(), {}) or {}).get("localization")
            or "not-evaluated"
        )
        if localization == "delocalized":
            confidence = [value + 0.05 for value in confidence]
            for i in range(len(structure)):
                evidence[i].append(
                    f"{edge_used} is spatially delocalized, supporting electronic compensation"
                )

    wannier_support = None
    if wannier_analysis and comp_type == "electrons":
        n_out = int(wannier_analysis.get("n_delocalized_spread_outliers") or 0)
        epwf = wannier_analysis.get("electrons_per_wf")
        if epwf is None:
            represented = wannier_analysis.get("represented_electrons")
            ncentres = int(wannier_analysis.get("n_wannier_centres") or 0)
            if represented is not None and ncentres:
                epwf = float(represented) / ncentres
        if epwf is not None:
            out_e = n_out * float(epwf)
            match = math.isclose(
                out_e,
                comp_count,
                rel_tol=0.0,
                abs_tol=float(settings.get("wannier_compensation_match_tolerance_e", 0.5)),
            )
            wannier_support = {
                "n_delocalized_spread_outliers": n_out,
                "electrons_per_wf": float(epwf),
                "delocalized_outlier_electron_equivalent": out_e,
                "matches_electronic_compensation": match,
            }
            if match:
                confidence = [value + 0.05 for value in confidence]
                for i in range(len(structure)):
                    evidence[i].append(
                        "delocalized Wannier electron equivalent matches compensation"
                    )

    sites: list[dict[str, Any]] = []
    for i, site in enumerate(structure):
        score = max(0.05, min(0.98, confidence[i]))
        label = "high" if score >= 0.80 else ("medium" if score >= 0.60 else "low")
        sites.append(
            {
                "site_index": i,
                "element": site.specie.symbol,
                "formal_oxidation_state": int(states[i]),
                "structural_prior_oxidation_state": int(initial[i]),
                "bader_partial_charge": bader.get(i),
                "magnetic_moment": magmom.get(i),
                "confidence": round(score, 3),
                "confidence_label": label,
                "assignment_status": "suggested",
                "evidence": evidence[i],
            }
        )

    return {
        "sites": sites,
        "formal_charge_sum_e": formal_sum,
        "cell_charge_e": float(cell_charge_e),
        "electronic_compensation": {
            "charge_e": compensation_charge,
            "type": comp_type,
            "count": comp_count,
            "localization": localization,
            "band_edge_used": edge_used,
        },
        "mixed_valence_diagnostics": mixed_diagnostics,
        "wannier_compensation_support": wannier_support,
        "reference_calibration_used": bool(refs),
    }


def _wf_site_weights(
    structure: Structure,
    calc: Any,
    *,
    band: int,
    spin: int,
    chunk: int,
    delocalized_fraction: float,
    localized_fraction: float,
) -> dict[str, Any]:
    import numpy as np

    psi = calc.get_pseudo_wave_function(band=band, kpt=0, spin=spin)
    density = np.abs(np.asarray(psi)) ** 2
    flat = density.ravel()
    weights = np.zeros(len(structure), dtype=float)
    for start in range(0, flat.size, chunk):
        stop = min(start + chunk, flat.size)
        linear = np.arange(start, stop)
        ix, iy, iz = np.unravel_index(linear, density.shape)
        frac = np.column_stack(
            (
                ix / float(density.shape[0]),
                iy / float(density.shape[1]),
                iz / float(density.shape[2]),
            )
        )
        distances = structure.lattice.get_all_distances(frac, structure.frac_coords)
        nearest = np.argmin(distances, axis=1)
        weights += np.bincount(
            nearest,
            weights=flat[start:stop],
            minlength=len(structure),
        )
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("wavefunction density sums to zero")
    fractions = weights / total
    ipr = float(np.sum(fractions**2))
    pn = 1.0 / ipr if ipr > 0 else 0.0
    fraction_atoms = pn / len(structure)
    if fraction_atoms >= delocalized_fraction:
        localization = "delocalized"
    elif fraction_atoms <= localized_fraction:
        localization = "localized"
    else:
        localization = "intermediate"

    element_weights: dict[str, float] = defaultdict(float)
    for i, site in enumerate(structure):
        element_weights[site.specie.symbol] += 100.0 * float(fractions[i])
    order = np.argsort(fractions)[::-1]
    return {
        "band": int(band),
        "spin": int(spin),
        "ipr_site_partition": ipr,
        "participation_number_sites": float(pn),
        "participation_fraction_of_atoms": float(fraction_atoms),
        "localization": localization,
        "element_weights_percent": dict(element_weights),
        "top_sites": [
            {
                "site_index": int(i),
                "element": structure[int(i)].specie.symbol,
                "weight_percent": 100.0 * float(fractions[int(i)]),
            }
            for i in order[: min(20, len(order))]
        ],
        "site_weights": [
            {
                "site_index": i,
                "element": structure[i].specie.symbol,
                "weight_percent": 100.0 * float(fractions[i]),
            }
            for i in range(len(structure))
        ],
    }


def analyze_band_edges(
    structure: Structure,
    gpw_path: Path,
    workdir: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from dopingflow import oxidation_dft as dft

    atoms, calc = dft._gpaw_restart(gpw_path)
    if list(atoms.get_chemical_symbols()) != [
        site.specie.symbol for site in structure
    ]:
        raise RuntimeError("GPAW site order/species do not match structure")

    try:
        kpts = np.asarray(calc.get_bz_k_points(), dtype=float)
    except Exception:
        kpts = np.asarray(calc.get_ibz_k_points(), dtype=float)
    if len(kpts) != 1:
        raise OptionalMethodUnavailable(
            "Automatic real-space band-edge localization currently supports one k-point."
        )

    occ_tol = float(settings.get("occupation_tolerance", 1.0e-4))
    occupied: list[tuple[float, int, int]] = []
    empty: list[tuple[float, int, int]] = []
    partial: list[dict[str, Any]] = []
    for spin in range(int(calc.get_number_of_spins())):
        occ = np.asarray(
            calc.get_occupation_numbers(kpt=0, spin=spin, raw=True),
            dtype=float,
        )
        eig = np.asarray(calc.get_eigenvalues(kpt=0, spin=spin), dtype=float)
        for band, (o, e) in enumerate(zip(occ, eig)):
            if o >= 1.0 - occ_tol:
                occupied.append((float(e), spin, band))
            elif o <= occ_tol:
                empty.append((float(e), spin, band))
            else:
                partial.append(
                    {
                        "spin": spin,
                        "band": band,
                        "occupation": float(o),
                        "energy_eV": float(e),
                    }
                )
    if not occupied or not empty:
        raise OptionalMethodUnavailable("Could not identify both occupied and empty band edges")

    he, hs, hb = max(occupied, key=lambda item: item[0])
    le, ls, lb = min(empty, key=lambda item: item[0])
    common = {
        "chunk": int(settings.get("band_edge_chunk_size", 4000)),
        "delocalized_fraction": float(
            settings.get("band_edge_delocalized_fraction", 0.20)
        ),
        "localized_fraction": float(
            settings.get("band_edge_localized_fraction", 0.08)
        ),
    }
    homo = _wf_site_weights(structure, calc, band=hb, spin=hs, **common)
    lumo = _wf_site_weights(structure, calc, band=lb, spin=ls, **common)
    homo["energy_eV"] = he
    lumo["energy_eV"] = le
    result = {
        "homo": homo,
        "lumo": lumo,
        "gap_eV": float(le - he),
        "partial_occupations": partial,
        "interpretation": (
            "Nearest-atom partitions of GPAW pseudo-wavefunction density are localization "
            "descriptors, not atomic charges."
        ),
    }
    (workdir / "band_edge_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (workdir / "band_edge_site_weights.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["state", "site_index", "element", "weight_percent"],
        )
        writer.writeheader()
        for state in ("homo", "lumo"):
            for row in result[state]["site_weights"]:
                writer.writerow({"state": state.upper(), **row})
    return result


def _write_outputs(workdir: Path, synthesis: dict[str, Any]) -> dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    json_path = workdir / "oxidation_state_suggestions.json"
    csv_path = workdir / "oxidation_state_suggestions.csv"
    json_path.write_text(
        json.dumps(synthesis, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fields = [
        "site_index",
        "element",
        "formal_oxidation_state",
        "structural_prior_oxidation_state",
        "bader_partial_charge",
        "magnetic_moment",
        "confidence",
        "confidence_label",
        "assignment_status",
        "evidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in synthesis["sites"]:
            out = {key: row.get(key) for key in fields}
            out["evidence"] = json.dumps(out.get("evidence") or [])
            writer.writerow(out)
    return {"json": str(json_path), "csv": str(csv_path)}


def run_dft_auto(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    from dopingflow import oxidation_dft as dft

    structure = Structure.from_file(target.structure_path)
    execute = bool(settings.get("execute", False))
    reuse = bool(settings.get("reuse_existing", True))
    output_root = str(
        settings.get("output_root")
        or (cfg.settings.get("dft_electronic", {}) or {}).get("output_root")
        or "dft_oxidation"
    )

    electronic_settings = dict(cfg.settings.get("dft_electronic", {}) or {})
    for key in (
        "xc", "mode", "ecut_eV", "kpts", "gamma", "smearing_eV",
        "convergence_density", "maxiter", "charge", "spinpol",
        "initial_magmoms", "dos_emin_eV", "dos_emax_eV",
        "dos_npoints", "dos_width_eV", "gpw_file", "txt_file", "nbands", "cache_root",
    ):
        if key in settings:
            electronic_settings[key] = settings[key]
    electronic_settings["output_root"] = output_root
    electronic_settings["save_wavefunctions"] = bool(
        settings.get("save_wavefunctions", True)
    )

    workdir = dft._workdir(target, cfg, electronic_settings)
    gpw_path = workdir / str(
        electronic_settings.get("gpw_file", "oxidation.gpw")
    )
    electronic_settings["execute"] = execute
    electronic_settings["reuse_existing"] = reuse
    electronic = dft._run_dft_electronic(
        target, cfg, electronic_settings
    )
    component_status: dict[str, Any] = {
        "dft-electronic": {
            "status": electronic.get("assignment_status"),
            "reused_existing": bool(
                electronic.get("provenance", {}).get("reused_existing", False)
            ),
        }
    }
    from dopingflow.dft_cache import derived_current, record_derived
    limitations: list[str] = []

    band_edges: dict[str, Any] = {}
    if bool(settings.get("band_edge_analysis", True)):
        try:
            band_edges = analyze_band_edges(
                structure, gpw_path, workdir, settings
            )
            component_status["band-edge"] = {"status": "descriptors-only"}
        except Exception as exc:
            component_status["band-edge"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
            limitations.append(
                f"Band-edge localization unavailable: {type(exc).__name__}: {exc}"
            )

    bader_settings = dict(cfg.settings.get("bader", {}) or {})
    bader_settings["output_root"] = output_root
    bader_settings["gpw_file"] = electronic_settings.get(
        "gpw_file", "oxidation.gpw"
    )
    acf = workdir / str(bader_settings.get("acf_file", "ACF.dat"))
    bader_reuse = reuse and derived_current(acf, gpw_path)
    bader_settings["execute"] = execute and not bader_reuse
    try:
        if acf.exists() and not bader_reuse and not execute:
            raise OptionalMethodUnavailable("Existing Bader results are not verified against this GPW; enable execute to regenerate")
        bader = dft._run_bader(target, cfg, bader_settings)
        if acf.exists() and gpw_path.exists():
            record_derived(acf, gpw_path)
        component_status["bader"] = {
            "status": bader.get("assignment_status"),
            "reused_existing": bool(
                bader_reuse
            ),
        }
    except Exception as exc:
        bader = {"bader_partial_charges": []}
        component_status["bader"] = {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }
        limitations.append(
            f"Bader evidence unavailable: {type(exc).__name__}: {exc}"
        )

    prior, prior_meta = _local_bv_prior(structure, settings)
    bader_by_site = _by_site(
        bader.get("bader_partial_charges", []),
        "bader_partial_charge",
    )
    if prior is None:
        prior, fallback = _fallback_uniform_prior(
            structure, bader_by_site, settings
        )
        prior_meta = {**prior_meta, "fallback": fallback}

    refs = _load_refs(cfg, settings)
    cell_charge = float(electronic_settings.get("charge", 0.0))
    synthesis = synthesize_oxidation_states(
        structure,
        prior_states=prior,
        bader_records=bader.get("bader_partial_charges", []),
        magnetic_records=electronic.get("magnetic_moments", []),
        settings=settings,
        reference_fingerprints=refs,
        band_edge_analysis=band_edges,
        cell_charge_e=cell_charge,
    )

    wannier_mode = str(
        settings.get("wannier_mode", "auto")
    ).strip().lower()
    if wannier_mode not in {"auto", "always", "never"}:
        raise ValueError(
            "[oxidation.dft_auto].wannier_mode must be auto, always, or never"
        )
    need_wannier = wannier_mode == "always" or (
        wannier_mode == "auto"
        and float(synthesis["electronic_compensation"]["count"])
        > float(settings.get("wannier_trigger_compensation_e", 0.5))
    )
    wannier_result: dict[str, Any] = {}
    if need_wannier:
        wset = dict(cfg.settings.get("wannier", {}) or {})
        wset["output_root"] = output_root
        wset.setdefault(
            "gpw_file",
            electronic_settings.get("gpw_file", "oxidation.gpw"),
        )
        seed = str(wset.get("seed", "wannier90")).strip() or "wannier90"
        centres = workdir / str(
            wset.get("centres_file", f"{seed}_centres.xyz")
        )
        wannier_reuse = reuse and derived_current(centres, gpw_path)
        wset["execute"] = execute and not wannier_reuse
        try:
            if centres.exists() and not wannier_reuse and not execute:
                raise OptionalMethodUnavailable("Existing Wannier results are not verified against this GPW; enable execute to regenerate")
            wannier_result = dft._run_wannier(target, cfg, wset)
            if centres.exists() and gpw_path.exists():
                record_derived(centres, gpw_path)
            component_status["wannier"] = {
                "status": wannier_result.get("assignment_status"),
                "reused_existing": bool(
                    wannier_reuse
                ),
            }
            synthesis = synthesize_oxidation_states(
                structure,
                prior_states=prior,
                bader_records=bader.get("bader_partial_charges", []),
                magnetic_records=electronic.get("magnetic_moments", []),
                settings=settings,
                reference_fingerprints=refs,
                band_edge_analysis=band_edges,
                wannier_analysis=wannier_result.get(
                    "wannier_analysis", {}
                ),
                cell_charge_e=cell_charge,
            )
        except Exception as exc:
            component_status["wannier"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
            limitations.append(
                f"Wannier evidence unavailable: {type(exc).__name__}: {exc}"
            )
    else:
        component_status["wannier"] = {
            "status": "not-run",
            "reason": f"wannier_mode={wannier_mode}",
        }

    files = _write_outputs(workdir, synthesis)
    result = base_method_result(
        method="dft-auto",
        target=target,
        scope="site-resolved",
        status="assigned",
        formal_oxidation_states=synthesis["sites"],
        bader_partial_charges=bader.get("bader_partial_charges", []),
        orbital_populations=electronic.get("orbital_populations", []),
        magnetic_moments=electronic.get("magnetic_moments", []),
        provenance={
            "implementation": "automated DFT oxidation-state suggestion",
            "workdir": str(workdir),
            "execute": execute,
            "reuse_existing": reuse,
            "component_status": component_status,
            "structural_prior": prior_meta,
            "bader_reference_calibration_used": bool(refs),
            "output_files": files,
        },
        limitations=[
            "Formal oxidation state is not a direct quantum-mechanical observable. dft-auto reports a chemically constrained, DFT-supported suggestion with per-site confidence.",
            "Absolute Bader-to-oxidation-state discrimination is strongest when same-method reference fingerprints are available.",
            "When DFT evidence does not support localized mixed valence, dft-auto reports explicit electronic compensation instead of forcing arbitrary reduced/oxidized atoms.",
            *limitations,
        ],
    )
    result["assignment_summary"] = synthesis
    result["electronic_compensation"] = synthesis[
        "electronic_compensation"
    ]
    result["band_edge_analysis"] = band_edges
    result["component_status"] = component_status
    if wannier_result:
        result["wannier_analysis"] = wannier_result.get(
            "wannier_analysis", {}
        )
        result["wannier_site_summary"] = wannier_result.get(
            "wannier_site_summary", []
        )
    return result

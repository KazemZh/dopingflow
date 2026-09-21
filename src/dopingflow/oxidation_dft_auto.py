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



def _safe_reference_token(value: str) -> str:
    text = str(value).strip().replace("\\", "_").replace("/", "_")
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in text) or "reference"


def _resolve_reference_path(cfg: OxidationConfig, raw: str | Path, *, base: Path | None = None) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base or cfg.root) / path).resolve()


def _reference_cache_root(
    cfg: OxidationConfig,
    settings: dict[str, Any],
    output_root: str,
) -> Path:
    raw = str(settings.get("reference_cache_root") or "").strip()
    if raw:
        return _resolve_reference_path(cfg, raw)
    base = Path(output_root).expanduser()
    if not base.is_absolute():
        base = cfg.source_root / base
    return (base / "reference_calibration").resolve()


def _reference_structure_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates: list[Path] = []
    for pattern in ("POSCAR", "CONTCAR", "*.vasp", "*.cif"):
        candidates.extend(path for path in root.rglob(pattern) if path.is_file())
    # Prefer one POSCAR/CONTCAR-like file per directory to avoid double-counting
    # the same reference when both source and relaxed names coexist.
    selected: dict[str, Path] = {}
    for path in sorted({p.resolve() for p in candidates}, key=lambda p: str(p)):
        if path.name.upper() in {"POSCAR", "CONTCAR"}:
            key = str(path.parent.resolve())
            old = selected.get(key)
            if old is None or (old.name.upper() == "POSCAR" and path.name.upper() == "CONTCAR"):
                selected[key] = path
        else:
            selected[str(path)] = path
    return list(selected.values())


def _infer_binary_oxide_reference(
    path: Path,
    *,
    peroxide_oo_cutoff_angstrom: float = 1.8,
) -> dict[str, Any] | None:
    """Infer a single-cation formal state from a simple binary oxide reference.

    Only M_xO_y structures with an integer M oxidation state under O2- are
    accepted automatically. Short O-O bonded structures are rejected so
    peroxides/superoxides are not silently treated as ordinary O2- oxides.
    """
    try:
        structure = Structure.from_file(path)
    except Exception:
        return None
    symbols = sorted({site.specie.symbol for site in structure})
    if "O" not in symbols:
        return None
    cations = [symbol for symbol in symbols if symbol != "O"]
    if len(cations) != 1:
        return None
    if _short_oo(structure, peroxide_oo_cutoff_angstrom):
        return None
    amounts = structure.composition.get_el_amt_dict()
    element = cations[0]
    n_m = float(amounts.get(element, 0.0))
    n_o = float(amounts.get("O", 0.0))
    if n_m <= 0 or n_o <= 0:
        return None
    nominal = 2.0 * n_o / n_m
    integer = int(round(nominal))
    if integer <= 0 or not math.isclose(nominal, integer, abs_tol=1.0e-8):
        return None
    try:
        allowed = {int(value) for value in Element(element).oxidation_states}
    except Exception:
        allowed = set()
    if allowed and integer not in allowed:
        return None
    return {
        "id": f"{structure.composition.reduced_formula}_{path.parent.name}",
        "path": str(path.resolve()),
        "element": element,
        "oxidation_state": integer,
        "source": "auto-binary-oxide",
        "nominal_oxygen_state": -2,
    }


def _manifest_reference_entries(
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    raw = str(settings.get("reference_manifest") or "").strip()
    default = cfg.root / "reference_structures" / "oxidation_states" / "manifest.json"
    path = _resolve_reference_path(cfg, raw) if raw else default.resolve()
    if not path.is_file():
        return [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("references", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Oxidation reference manifest must contain a list or {'references': [...]}")
    entries: list[dict[str, Any]] = []
    for pos, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict) or raw_row.get("enabled", True) is False:
            continue
        structure_raw = raw_row.get("path") or raw_row.get("structure_path")
        element = str(raw_row.get("element") or "").strip()
        state = raw_row.get("oxidation_state")
        if not structure_raw or not element or state is None:
            continue
        structure_path = _resolve_reference_path(cfg, str(structure_raw), base=path.parent)
        if not structure_path.is_file():
            continue
        numeric = float(state)
        integer = int(round(numeric))
        if not math.isclose(numeric, integer, abs_tol=1.0e-8):
            raise ValueError(f"Reference manifest oxidation_state must be integer: {state}")
        entry = {
            "id": str(raw_row.get("id") or f"{element}_{integer:+d}_{pos:03d}"),
            "path": str(structure_path),
            "element": element,
            "oxidation_state": integer,
            "source": "manifest",
        }
        if isinstance(raw_row.get("initial_magmoms"), dict):
            entry["initial_magmoms"] = {
                str(k): float(v) for k, v in raw_row["initial_magmoms"].items()
            }
        if raw_row.get("kpts") is not None:
            entry["kpts"] = [int(value) for value in raw_row["kpts"]]
        entries.append(entry)
    return entries, str(path)


def _discover_reference_entries(
    cfg: OxidationConfig,
    settings: dict[str, Any],
    *,
    elements: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_entries, manifest_path = _manifest_reference_entries(cfg, settings)
    raw_roots = settings.get("reference_roots")
    if raw_roots is None:
        raw_roots = [
            "reference_structures/relaxed/refs",
            "reference_structures/oxidation_states",
            "reference_structures/oxides",
        ]
    elif isinstance(raw_roots, str):
        raw_roots = [item.strip() for item in raw_roots.split(",") if item.strip()]
    elif not isinstance(raw_roots, (list, tuple)):
        raise ValueError("[oxidation.dft_auto].reference_roots must be a list or comma-separated string")

    # Existing dopingflow projects historically store the relaxed reference
    # phases in reference_structures/relaxed/refs.  Include that canonical
    # directory by default even when an older saved GUI configuration still
    # contains only oxidation_states/ and oxides/.
    canonical_relaxed = (cfg.root / "reference_structures" / "relaxed" / "refs").resolve()
    roots = [_resolve_reference_path(cfg, item) for item in raw_roots]
    if bool(settings.get("reference_include_project_relaxed_refs", True)):
        if canonical_relaxed.exists() and canonical_relaxed not in roots:
            roots.insert(0, canonical_relaxed)

    entries: list[dict[str, Any]] = list(manifest_entries)
    manifest_paths = {Path(entry["path"]).resolve() for entry in manifest_entries}
    cutoff = float(settings.get("peroxide_oo_cutoff_angstrom", 1.8))
    for root in roots:
        for path in _reference_structure_files(root):
            if path.resolve() in manifest_paths:
                continue
            inferred = _infer_binary_oxide_reference(
                path,
                peroxide_oo_cutoff_angstrom=cutoff,
            )
            if inferred is not None:
                entries.append(inferred)

    # Exact path/element/state triples are unique. A manifest entry wins over
    # an automatically inferred entry for the same structure.
    unique: dict[tuple[str, str, int], dict[str, Any]] = {}
    for entry in entries:
        element = str(entry["element"])
        if element not in elements or element == "O":
            continue
        key = (
            str(Path(entry["path"]).resolve()),
            element,
            int(entry["oxidation_state"]),
        )
        if key not in unique or entry.get("source") == "manifest":
            unique[key] = entry

    # Some long-lived projects already contain additional relaxed calibration
    # oxides under reference_structures/corrections/*/relaxed_calibration
    # (for example SnO or NbO2) but not in relaxed/refs.  Use those only to
    # supplement an oxidation state that is completely absent from the primary
    # reference roots; do not duplicate states already covered by canonical refs.
    supplemental_used: list[str] = []
    if bool(settings.get("reference_include_correction_calibration", True)):
        present_states = {
            (str(entry["element"]), int(entry["oxidation_state"]))
            for entry in unique.values()
        }
        correction_root = (cfg.root / "reference_structures" / "corrections").resolve()
        candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        if correction_root.exists():
            for relaxed_root in sorted(correction_root.glob("*/relaxed_calibration")):
                for path in _reference_structure_files(relaxed_root):
                    inferred = _infer_binary_oxide_reference(
                        path,
                        peroxide_oo_cutoff_angstrom=cutoff,
                    )
                    if inferred is None:
                        continue
                    element = str(inferred["element"])
                    state = int(inferred["oxidation_state"])
                    if element not in elements or element == "O":
                        continue
                    key_state = (element, state)
                    if key_state in present_states:
                        continue
                    inferred["source"] = "auto-correction-calibration-supplement"
                    candidates[key_state].append(inferred)

        # Deterministically select one existing relaxed structure per missing
        # element/oxidation-state pair. This avoids counting duplicated
        # correction snapshots as independent calibration observations.
        for key_state, options in sorted(candidates.items()):
            chosen = sorted(options, key=lambda item: str(item["path"]))[0]
            key = (
                str(Path(chosen["path"]).resolve()),
                str(chosen["element"]),
                int(chosen["oxidation_state"]),
            )
            unique[key] = chosen
            present_states.add(key_state)
            supplemental_used.append(str(chosen["path"]))

    final = sorted(
        unique.values(),
        key=lambda item: (
            str(item["element"]),
            int(item["oxidation_state"]),
            str(item["id"]),
        ),
    )
    return final, {
        "manifest": manifest_path,
        "roots": [str(root) for root in roots],
        "canonical_relaxed_refs_included": canonical_relaxed in roots,
        "correction_calibration_supplements": supplemental_used,
        "n_discovered": len(final),
    }


def _reference_kpts(
    target_structure: Structure,
    reference_structure: Structure,
    target_kpts: Sequence[int],
    settings: dict[str, Any],
    entry: dict[str, Any],
) -> list[int]:
    if entry.get("kpts") is not None:
        values = [int(value) for value in entry["kpts"]]
        if len(values) != 3 or any(value < 1 for value in values):
            raise ValueError("Reference manifest kpts must contain three positive integers")
        return values
    mode = str(settings.get("reference_kpoint_mode", "match-density")).strip().lower()
    if mode not in {"match-density", "same-grid"}:
        raise ValueError(
            "[oxidation.dft_auto].reference_kpoint_mode must be 'match-density' or 'same-grid'"
        )
    base = [int(value) for value in target_kpts]
    if mode == "same-grid":
        return base
    maximum = max(1, int(settings.get("reference_kpts_max", 8)))
    result: list[int] = []
    for n_target, l_target, l_ref in zip(
        base,
        target_structure.lattice.abc,
        reference_structure.lattice.abc,
    ):
        if l_ref <= 0:
            result.append(n_target)
            continue
        estimate = int(round(float(n_target) * float(l_target) / float(l_ref)))
        result.append(max(1, min(maximum, estimate)))
    return result


def _nominal_transition_metal_seed(element: str, oxidation_state: int) -> float:
    """Conservative high-spin-like initial seed; it is not an OS descriptor."""
    try:
        el = Element(element)
        if not el.is_transition_metal or el.group is None:
            return 0.0
        d_count = max(0, min(10, int(el.group) - int(oxidation_state)))
        return float(min(d_count, 10 - d_count))
    except Exception:
        return 0.0


def _reference_fingerprint_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(float(value) for value in values) / len(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _calculate_reference_fingerprint(
    cfg: OxidationConfig,
    settings: dict[str, Any],
    electronic_settings: dict[str, Any],
    *,
    target_structure: Structure,
    output_root: str,
    entry: dict[str, Any],
    execute: bool,
    reuse: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from dopingflow import oxidation_dft as dft
    from dopingflow.dft_cache import calculation_key, derived_current, ensure_gpaw, record_derived

    path = Path(entry["path"]).resolve()
    structure = Structure.from_file(path)
    element = str(entry["element"])
    state = int(entry["oxidation_state"])
    reference_target = StructureTarget(
        target_id=f"oxidation-reference/{_safe_reference_token(str(entry['id']))}",
        parent_id=f"oxidation-reference/{element}",
        kind="oxidation-reference",
        structure_path=path,
        n_vacancies=0,
        vacancy_species=None,
        metadata={
            "reference_element": element,
            "reference_oxidation_state": state,
            "reference_source": entry.get("source"),
        },
    )

    ref_settings = dict(electronic_settings)
    ref_settings["charge"] = 0.0
    ref_settings["save_wavefunctions"] = False
    ref_settings["reuse_existing"] = reuse
    ref_settings["execute"] = execute
    reference_spin = settings.get("reference_spinpol", "auto")
    if isinstance(reference_spin, bool):
        reference_spin = "true" if reference_spin else "false"
    reference_spin = str(reference_spin).strip().lower()
    if reference_spin not in {"auto", "true", "false"}:
        raise ValueError(
            "[oxidation.dft_auto].reference_spinpol must be auto, true, or false"
        )
    ref_settings["spinpol"] = reference_spin
    ref_settings["kpts"] = _reference_kpts(
        target_structure,
        structure,
        electronic_settings.get("kpts", [1, 1, 1]),
        settings,
        entry,
    )

    initial = dict(
        settings.get("reference_initial_magmoms")
        or ref_settings.get("initial_magmoms")
        or {}
    )
    if isinstance(entry.get("initial_magmoms"), dict):
        initial.update(entry["initial_magmoms"])
    if bool(settings.get("reference_auto_magnetic_seed", True)) and element not in initial:
        seed = _nominal_transition_metal_seed(element, state)
        if seed > 0:
            initial[element] = seed
    ref_settings["initial_magmoms"] = initial

    key, identity = calculation_key(reference_target, ref_settings)
    cache_root = _reference_cache_root(cfg, settings, output_root)
    workdir = (
        cache_root
        / element
        / f"OS_{state:+d}"
        / _safe_reference_token(str(entry["id"]))
        / key[:12]
    )
    ref_settings["workdir"] = str(workdir)
    fingerprint_file = workdir / "oxidation_reference_fingerprint.json"
    if reuse and fingerprint_file.is_file():
        try:
            cached = json.loads(fingerprint_file.read_text(encoding="utf-8"))
            if cached.get("calculation_key") == key:
                return cached, {
                    "id": entry["id"],
                    "element": element,
                    "oxidation_state": state,
                    "status": "reused-fingerprint",
                    "workdir": str(workdir),
                }
        except Exception:
            pass

    try:
        gpw_path, gpw_reused = ensure_gpaw(reference_target, cfg, ref_settings)
    except Exception as exc:
        return None, {
            "id": entry["id"],
            "element": element,
            "oxidation_state": state,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "workdir": str(workdir),
        }

    bader_settings = dict(cfg.settings.get("bader", {}) or {})
    bader_settings.update(
        {
            "workdir": str(workdir),
            "gpw_file": gpw_path.name,
            "gridrefinement": int(
                settings.get(
                    "reference_bader_gridrefinement",
                    bader_settings.get("gridrefinement", 2),
                )
            ),
        }
    )
    acf = workdir / str(bader_settings.get("acf_file", "ACF.dat"))
    bader_reuse = reuse and acf.is_file() and derived_current(acf, gpw_path)
    bader_settings["execute"] = execute and not bader_reuse
    if acf.is_file() and not bader_reuse and not execute:
        return None, {
            "id": entry["id"],
            "element": element,
            "oxidation_state": state,
            "status": "unavailable",
            "error": "Existing Bader reference is not verified against the current GPAW result",
            "workdir": str(workdir),
        }

    try:
        bader = dft._run_bader(reference_target, cfg, bader_settings)
        if acf.is_file() and gpw_path.is_file():
            record_derived(acf, gpw_path)
    except Exception as exc:
        return None, {
            "id": entry["id"],
            "element": element,
            "oxidation_state": state,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "workdir": str(workdir),
        }

    values = [
        float(record["bader_partial_charge"])
        for record in bader.get("bader_partial_charges", [])
        if record.get("element") == element
        and record.get("bader_partial_charge") is not None
    ]
    if not values:
        return None, {
            "id": entry["id"],
            "element": element,
            "oxidation_state": state,
            "status": "unavailable",
            "error": f"No Bader values found for reference element {element}",
            "workdir": str(workdir),
        }

    fingerprint = {
        "schema_version": 1,
        "id": str(entry["id"]),
        "element": element,
        "oxidation_state": state,
        "structure_path": str(path),
        "reference_source": entry.get("source"),
        "calculation_key": key,
        "calculation_identity": identity,
        "workdir": str(workdir),
        "kpts": list(ref_settings["kpts"]),
        "initial_magmoms": initial,
        "bader_values": values,
        "mean": sum(values) / len(values),
        "std": _reference_fingerprint_std(values),
        "median": float(median(values)),
        "mad": _mad(values),
        "n_sites": len(values),
        "gpw_reused": bool(gpw_reused),
        "bader_reused": bool(bader_reuse),
    }
    workdir.mkdir(parents=True, exist_ok=True)
    fingerprint_file.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return fingerprint, {
        "id": entry["id"],
        "element": element,
        "oxidation_state": state,
        "status": "calculated",
        "workdir": str(workdir),
        "gpw_reused": bool(gpw_reused),
        "bader_reused": bool(bader_reuse),
    }


def _aggregate_reference_fingerprints(
    fingerprints: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for fingerprint in fingerprints:
        grouped[str(fingerprint["element"])][
            int(fingerprint["oxidation_state"])
        ].append(fingerprint)

    library: dict[str, Any] = {}
    for element, by_state in grouped.items():
        library[element] = {}
        for state, records in by_state.items():
            values = [
                float(value)
                for record in records
                for value in record.get("bader_values", [])
            ]
            if not values:
                continue
            library[element][str(state)] = {
                "mean": sum(values) / len(values),
                "std": _reference_fingerprint_std(values),
                "median": float(median(values)),
                "mad": _mad(values),
                "n": len(values),
                "values": values,
                "references": [
                    {
                        "id": record.get("id"),
                        "structure_path": record.get("structure_path"),
                        "calculation_key": record.get("calculation_key"),
                        "kpts": record.get("kpts"),
                    }
                    for record in records
                ],
            }
    return library


def _merge_reference_libraries(
    automatic: dict[str, Any],
    explicit: dict[str, Any],
) -> dict[str, Any]:
    merged = json.loads(json.dumps(automatic)) if automatic else {}
    for element, values in explicit.items():
        if isinstance(values, dict):
            merged.setdefault(str(element), {}).update(values)
        else:
            merged[str(element)] = values
    return merged


def _build_automatic_reference_library(
    cfg: OxidationConfig,
    settings: dict[str, Any],
    electronic_settings: dict[str, Any],
    *,
    structure: Structure,
    output_root: str,
    execute: bool,
    reuse: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = str(settings.get("reference_calibration", "auto")).strip().lower()
    if mode not in {"off", "auto", "require"}:
        raise ValueError(
            "[oxidation.dft_auto].reference_calibration must be off, auto, or require"
        )
    if mode == "off":
        return {}, {"mode": "off", "status": "disabled"}

    elements = {site.specie.symbol for site in structure if site.specie.symbol != "O"}
    entries, discovery = _discover_reference_entries(
        cfg,
        settings,
        elements=elements,
    )
    reference_execute = execute and bool(settings.get("run_missing_references", True))
    fingerprints: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for entry in entries:
        fingerprint, record = _calculate_reference_fingerprint(
            cfg,
            settings,
            electronic_settings,
            target_structure=structure,
            output_root=output_root,
            entry=entry,
            execute=reference_execute,
            reuse=reuse,
        )
        records.append(record)
        if fingerprint is not None:
            fingerprints.append(fingerprint)

    library = _aggregate_reference_fingerprints(fingerprints)
    min_states = max(2, int(settings.get("reference_min_states", 2)))
    states_available = {
        element: sorted(int(state) for state in values)
        for element, values in library.items()
        if isinstance(values, dict)
    }
    missing = sorted(
        element
        for element in elements
        if len(states_available.get(element, [])) < min_states
    )

    cache_root = _reference_cache_root(cfg, settings, output_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    library_path = cache_root / "bader_reference_fingerprints.json"
    payload = {
        "schema_version": 1,
        "mode": mode,
        "target_id": getattr(structure, "composition", None).reduced_formula
        if getattr(structure, "composition", None) is not None
        else None,
        "library": library,
        "states_available": states_available,
        "elements_missing_minimum_states": missing,
        "discovery": discovery,
        "reference_runs": records,
    }
    library_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    meta = {
        "mode": mode,
        "status": "complete" if not missing else "partial",
        "library_file": str(library_path),
        "states_available": states_available,
        "elements_missing_minimum_states": missing,
        "discovery": discovery,
        "reference_runs": records,
        "run_missing_references": reference_execute,
        "min_reference_states": min_states,
    }
    if mode == "require" and missing:
        raise OptionalMethodUnavailable(
            "Automatic Bader reference calibration is required but fewer than "
            f"{min_states} oxidation-state references are available for: {', '.join(missing)}"
        )
    return library, meta


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
    if isinstance(payload.get("library"), dict):
        payload = dict(payload["library"])
    payload.update(refs)
    return payload


def _reference_match(
    symbol: str,
    q: float,
    refs: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    raw = refs.get(symbol)
    if not isinstance(raw, dict):
        return {
            "status": "no-reference",
            "selected_state": None,
            "confidence": None,
            "candidates": [],
        }

    default_sigma = max(1.0e-6, float(settings.get("reference_default_sigma_e", 0.08)))
    candidates: list[dict[str, Any]] = []
    for state_raw, item in raw.items():
        try:
            state = int(state_raw)
        except (TypeError, ValueError):
            continue
        mean = None
        sigma = default_sigma
        n = None
        source_refs: list[Any] = []
        if isinstance(item, (int, float)):
            mean = float(item)
        elif isinstance(item, list) and item:
            values = [float(value) for value in item]
            mean = sum(values) / len(values)
            n = len(values)
            if len(values) > 1:
                sigma = max(default_sigma, _reference_fingerprint_std(values))
        elif isinstance(item, dict):
            if item.get("mean") is not None:
                mean = float(item["mean"])
            elif item.get("bader_partial_charge") is not None:
                mean = float(item["bader_partial_charge"])
            values = item.get("values")
            if mean is None and isinstance(values, list) and values:
                numeric = [float(value) for value in values]
                mean = sum(numeric) / len(numeric)
            if item.get("std") is not None:
                sigma = max(default_sigma, abs(float(item["std"])))
            elif isinstance(values, list) and len(values) > 1:
                sigma = max(default_sigma, _reference_fingerprint_std([float(v) for v in values]))
            if item.get("n") is not None:
                n = int(item["n"])
            elif isinstance(values, list):
                n = len(values)
            if isinstance(item.get("references"), list):
                source_refs = list(item["references"])
        if mean is None:
            continue
        z = abs(float(q) - mean) / max(default_sigma, sigma)
        likelihood = math.exp(-0.5 * z * z)
        candidates.append(
            {
                "oxidation_state": state,
                "mean_bader_charge": mean,
                "sigma_e": sigma,
                "z_distance": z,
                "likelihood": likelihood,
                "n_reference_sites": n,
                "references": source_refs,
            }
        )

    candidates.sort(key=lambda row: (float(row["z_distance"]), int(row["oxidation_state"])))
    min_states = max(2, int(settings.get("reference_min_states", 2)))
    if len(candidates) < min_states:
        return {
            "status": "insufficient-reference-states",
            "selected_state": None,
            "confidence": None,
            "candidates": candidates,
        }

    total = sum(float(row["likelihood"]) for row in candidates)
    if total <= 0:
        for row in candidates:
            row["probability"] = 0.0
    else:
        for row in candidates:
            row["probability"] = float(row["likelihood"]) / total

    ranked = sorted(
        candidates,
        key=lambda row: (-float(row.get("probability", 0.0)), float(row["z_distance"])),
    )
    best = ranked[0]
    second = ranked[1]
    best_probability = float(best.get("probability", 0.0))
    probability_margin = best_probability - float(second.get("probability", 0.0))
    z_gap = float(second["z_distance"]) - float(best["z_distance"])
    decisive = (
        float(best["z_distance"]) <= float(settings.get("reference_max_z", 2.5))
        and z_gap >= float(settings.get("reference_min_z_gap", 0.75))
        and best_probability >= float(settings.get("reference_min_probability", 0.70))
        and probability_margin >= float(settings.get("reference_min_probability_margin", 0.20))
    )
    return {
        "status": "calibrated" if decisive else "ambiguous",
        "selected_state": int(best["oxidation_state"]) if decisive else None,
        "best_candidate_state": int(best["oxidation_state"]),
        "best_z": float(best["z_distance"]),
        "confidence": best_probability,
        "probability_margin": probability_margin,
        "z_gap": z_gap,
        "candidates": ranked,
    }


def _reference_pick(
    symbol: str,
    q: float,
    refs: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[int, float] | None:
    match = _reference_match(symbol, q, refs, settings)
    if match.get("status") != "calibrated":
        return None
    return int(match["selected_state"]), float(match["best_z"])


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
    calibration_matches: dict[int, dict[str, Any]] = {}

    for i, site in enumerate(structure):
        if site.specie.symbol == "O":
            calibration_matches[i] = {
                "status": "not-applicable",
                "selected_state": None,
                "confidence": None,
                "candidates": [],
            }
            continue
        if i not in bader:
            continue
        match = _reference_match(site.specie.symbol, bader[i], refs, settings)
        calibration_matches[i] = match
        status = str(match.get("status") or "no-reference")
        if status == "calibrated":
            states[i] = int(match["selected_state"])
            locked.add(i)
            probability = float(match.get("confidence") or 0.0)
            confidence[i] += 0.20 + 0.15 * probability
            evidence[i].append(
                "same-method Bader references calibrate "
                f"{site.specie.symbol} to {states[i]:+d} "
                f"(z={float(match['best_z']):.2f}, relative likelihood={probability:.2f})"
            )
        elif status == "ambiguous":
            confidence[i] -= 0.15
            best = match.get("best_candidate_state")
            probability = match.get("confidence")
            evidence[i].append(
                "same-method Bader calibration is ambiguous"
                + (
                    f"; closest reference state is {int(best):+d}"
                    if best is not None
                    else ""
                )
                + (
                    f" (relative likelihood={float(probability):.2f})"
                    if probability is not None
                    else ""
                )
            )
        elif status == "insufficient-reference-states":
            confidence[i] -= 0.05
            evidence[i].append(
                "fewer than two same-method reference oxidation states are available; "
                "Bader calibration cannot discriminate absolute oxidation state"
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
    calibration_status_counts: Counter[str] = Counter()
    for i, site in enumerate(structure):
        score = max(0.05, min(0.98, confidence[i]))
        label = "high" if score >= 0.80 else ("medium" if score >= 0.60 else "low")
        match = calibration_matches.get(
            i,
            {
                "status": "no-reference",
                "selected_state": None,
                "confidence": None,
                "candidates": [],
            },
        )
        calibration_status = str(match.get("status") or "no-reference")
        calibration_status_counts[calibration_status] += 1
        candidates = [
            {
                "oxidation_state": int(row["oxidation_state"]),
                "probability": (
                    round(float(row["probability"]), 4)
                    if row.get("probability") is not None
                    else None
                ),
                "z_distance": round(float(row["z_distance"]), 4),
                "mean_bader_charge": round(float(row["mean_bader_charge"]), 6),
            }
            for row in match.get("candidates", [])
        ]
        if calibration_status == "calibrated":
            oxidation_state_status = "calibrated"
        elif calibration_status == "ambiguous":
            oxidation_state_status = "suggested-ambiguous-calibration"
        elif calibration_status == "insufficient-reference-states":
            oxidation_state_status = "suggested-insufficient-calibration"
        elif calibration_status == "not-applicable":
            oxidation_state_status = "reference-calibration-not-applicable"
        else:
            oxidation_state_status = "suggested-uncalibrated"
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
                "oxidation_state_status": oxidation_state_status,
                "reference_calibration_status": calibration_status,
                "calibrated_oxidation_state": match.get("selected_state"),
                "calibration_confidence": (
                    round(float(match["confidence"]), 4)
                    if match.get("confidence") is not None
                    else None
                ),
                "calibration_candidates": candidates,
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
        "reference_calibration_summary": {
            "status_counts": dict(calibration_status_counts),
            "n_calibrated_sites": int(calibration_status_counts.get("calibrated", 0)),
            "n_ambiguous_sites": int(calibration_status_counts.get("ambiguous", 0)),
            "n_insufficient_reference_sites": int(
                calibration_status_counts.get("insufficient-reference-states", 0)
            ),
            "n_uncalibrated_sites": int(calibration_status_counts.get("no-reference", 0)),
            "n_not_applicable_sites": int(
                calibration_status_counts.get("not-applicable", 0)
            ),
        },
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
        "oxidation_state_status",
        "reference_calibration_status",
        "calibrated_oxidation_state",
        "calibration_confidence",
        "calibration_candidates",
        "assignment_status",
        "evidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in synthesis["sites"]:
            out = {key: row.get(key) for key in fields}
            out["evidence"] = json.dumps(out.get("evidence") or [])
            out["calibration_candidates"] = json.dumps(
                out.get("calibration_candidates") or []
            )
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

    explicit_refs = _load_refs(cfg, settings)
    automatic_refs: dict[str, Any] = {}
    reference_calibration_meta: dict[str, Any] = {
        "mode": str(settings.get("reference_calibration", "auto")),
        "status": "not-run",
    }
    try:
        automatic_refs, reference_calibration_meta = _build_automatic_reference_library(
            cfg,
            settings,
            electronic_settings,
            structure=structure,
            output_root=output_root,
            execute=execute,
            reuse=reuse,
        )
        component_status["reference-calibration"] = {
            "status": reference_calibration_meta.get("status"),
            "states_available": reference_calibration_meta.get("states_available", {}),
            "elements_missing_minimum_states": reference_calibration_meta.get(
                "elements_missing_minimum_states", []
            ),
            "library_file": reference_calibration_meta.get("library_file"),
        }
        if reference_calibration_meta.get("elements_missing_minimum_states"):
            limitations.append(
                "Automatic reference calibration has fewer than the requested number "
                "of oxidation-state references for: "
                + ", ".join(reference_calibration_meta["elements_missing_minimum_states"])
            )
    except OptionalMethodUnavailable:
        raise
    except Exception as exc:
        component_status["reference-calibration"] = {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }
        limitations.append(
            f"Automatic reference calibration unavailable: {type(exc).__name__}: {exc}"
        )

    refs = _merge_reference_libraries(automatic_refs, explicit_refs)
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
            "automatic_reference_calibration": reference_calibration_meta,
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
    result["reference_calibration"] = reference_calibration_meta
    if wannier_result:
        result["wannier_analysis"] = wannier_result.get(
            "wannier_analysis", {}
        )
        result["wannier_site_summary"] = wannier_result.get(
            "wannier_site_summary", []
        )
    return result

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from pymatgen.core import Structure

from dopingflow.ml_backends import (
    build_ase_calculator,
    normalize_backend_config,
    prepare_backend_runtime,
)
from dopingflow.ml_relaxation import (
    relax_structure_with_calculator,
    structure_energy_with_calculator,
)

_REFERENCE_MODES = {"reference_file", "same_calculator", "explicit", "none"}
_ENERGY_SOURCES = {"relaxed_only", "relaxed_or_single_point"}
_DEFAULT_POINTS = (0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0)
_DEFAULT_TEMPERATURES = (300.0, 600.0, 900.0, 1200.0, 1500.0)
K_B_EV_PER_K = 8.617333262145e-5
KJ_PER_MOL_PER_EV = 96.4853321233
NIST_O2_H0_MINUS_H298_KJ_PER_MOL = -8.683
NIST_O2_SHOMATE_SOURCE = (
    "NIST Chemistry WebBook SRD 69; O2 gas Shomate coefficients, Chase 1998"
)
# (T_min, T_max, A, B, C, D, E, F, G, H); t = T / 1000.
NIST_O2_SHOMATE_COEFFICIENTS = (
    (100.0, 700.0, 31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471, 246.7945, 0.0),
    (700.0, 2000.0, 30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468, 236.1663, 0.0),
    (2000.0, 6000.0, 20.91111, 10.72071, -2.020498, 0.146449, 9.245722, 5.337651, 237.6185, 0.0),
)
STATIC_LATTICE_APPROXIMATION = (
    "Static-lattice approximation: solid free energies are approximated by 0 K "
    "relaxed ML energies. Temperature and pressure enter only through the "
    "oxygen-gas chemical potential. Vibrational, configurational, electronic, "
    "magnetic and anharmonic free-energy contributions of the solid are neglected."
)
NEGLECTED_SOLID_TERMS = (
    "vibrational",
    "zero_point",
    "configurational",
    "thermal_electronic",
    "magnetic",
    "anharmonic",
    "thermal_expansion",
    "solid_pV",
)


@dataclass(frozen=True)
class VacancyAnalysisConfig:
    enabled: bool
    oxygen_reference_mode: str
    oxygen_reference_file: Path
    oxygen_reference_structure: Path
    oxygen_reference_relax: bool
    mu_O_reference_eV: float | None
    allow_unverified_oxygen_reference: bool
    delta_mu_O_min_eV: float
    delta_mu_O_max_eV: float
    delta_mu_O_points_eV: tuple[float, ...]
    thermodynamic_tolerance_eV: float
    analysis_energy_source: str
    exclude_unconverged: bool
    pressure_mapping: bool
    temperatures_K: tuple[float, ...]
    standard_oxygen_pressure_bar: float
    log10_pO2_min_bar: float
    log10_pO2_max_bar: float
    log10_pO2_step: float
    oxygen_standard_state_mode: str
    oxygen_standard_state_temperatures_K: tuple[float, ...]
    oxygen_standard_state_delta_mu_eV_per_O: tuple[float, ...]


def parse_vacancy_analysis_config(
    section: dict[str, Any], root: Path
) -> VacancyAnalysisConfig:
    enabled = bool(
        section.get(
            "static_thermodynamic_analysis",
            section.get("thermodynamic_analysis", False),
        )
    )
    mode = str(section.get("oxygen_reference_mode", "reference_file")).strip().lower()
    if mode not in _REFERENCE_MODES:
        raise ValueError(
            "[vacancies].oxygen_reference_mode must be one of: "
            + ", ".join(sorted(_REFERENCE_MODES))
        )

    def resolve(key: str, default: str) -> Path:
        value = Path(str(section.get(key, default))).expanduser()
        return (value if value.is_absolute() else root / value).resolve()

    explicit_raw = section.get("mu_O_reference_eV")
    explicit = None if explicit_raw is None else float(explicit_raw)
    if mode == "explicit" and explicit is None:
        raise ValueError(
            "[vacancies].mu_O_reference_eV is required when "
            "oxygen_reference_mode='explicit' (value is per oxygen atom)"
        )
    lower = float(section.get("delta_mu_O_min_eV", -3.0))
    upper = float(section.get("delta_mu_O_max_eV", 0.0))
    if lower > upper:
        raise ValueError(
            "[vacancies].delta_mu_O_min_eV must be <= delta_mu_O_max_eV"
        )
    if upper > 0.0:
        raise ValueError("[vacancies].delta_mu_O_max_eV must be <= 0")
    points_raw = section.get("delta_mu_O_points_eV", list(_DEFAULT_POINTS))
    if not isinstance(points_raw, list) or not points_raw:
        raise ValueError("[vacancies].delta_mu_O_points_eV must be a non-empty array")
    points = tuple(float(value) for value in points_raw)
    if any(value > 0.0 for value in points):
        raise ValueError("[vacancies].delta_mu_O_points_eV values must be <= 0")
    tolerance = float(section.get("thermodynamic_tolerance_eV", 1.0e-8))
    if tolerance <= 0:
        raise ValueError("[vacancies].thermodynamic_tolerance_eV must be > 0")
    source = str(
        section.get(
            "static_energy_source",
            section.get("analysis_energy_source", "relaxed_only"),
        )
    ).strip().lower()
    if source not in _ENERGY_SOURCES:
        raise ValueError(
            "[vacancies].static_energy_source must be 'relaxed_only' or "
            "'relaxed_or_single_point'"
        )
    temperatures_raw = section.get("temperatures_K", list(_DEFAULT_TEMPERATURES))
    if not isinstance(temperatures_raw, list) or not temperatures_raw:
        raise ValueError("[vacancies].temperatures_K must be a non-empty array")
    temperatures = tuple(float(value) for value in temperatures_raw)
    if any(value <= 0 for value in temperatures):
        raise ValueError("[vacancies].temperatures_K values must be > 0")
    standard_pressure = float(section.get("standard_oxygen_pressure_bar", 1.0))
    if standard_pressure <= 0:
        raise ValueError("[vacancies].standard_oxygen_pressure_bar must be > 0")
    pressure_min = float(section.get("log10_pO2_min_bar", -30.0))
    pressure_max = float(section.get("log10_pO2_max_bar", 1.0))
    pressure_step = float(section.get("log10_pO2_step", 0.5))
    if pressure_min > pressure_max or pressure_step <= 0:
        raise ValueError(
            "[vacancies] pressure range requires log10_pO2_min_bar <= "
            "log10_pO2_max_bar and log10_pO2_step > 0"
        )
    standard_mode = str(section.get("oxygen_standard_state_mode", "none")).strip().lower()
    if standard_mode not in {"none", "nist_shomate", "user_table"}:
        raise ValueError(
            "[vacancies].oxygen_standard_state_mode must be 'none', "
            "'nist_shomate', or 'user_table'"
        )
    table_temperatures = tuple(
        float(value)
        for value in section.get("oxygen_standard_state_temperatures_K", [])
    )
    table_values = tuple(
        float(value)
        for value in section.get("oxygen_standard_state_delta_mu_eV_per_O", [])
    )
    if standard_mode == "user_table":
        if len(table_temperatures) < 2 or len(table_temperatures) != len(table_values):
            raise ValueError(
                "[vacancies] user_table requires equal arrays with at least two "
                "oxygen standard-state temperatures and delta-mu values"
            )
        if any(right <= left for left, right in zip(table_temperatures, table_temperatures[1:])):
            raise ValueError(
                "[vacancies].oxygen_standard_state_temperatures_K must increase strictly"
            )
        if min(temperatures) < table_temperatures[0] or max(temperatures) > table_temperatures[-1]:
            raise ValueError(
                "[vacancies] oxygen standard-state user table must cover all temperatures_K"
            )
    if standard_mode == "nist_shomate":
        if min(temperatures) < 100.0 or max(temperatures) > 6000.0:
            raise ValueError(
                "[vacancies] nist_shomate supports temperatures from 100 to 6000 K; "
                "extrapolation is not allowed"
            )
        if not math.isclose(standard_pressure, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "[vacancies] nist_shomate uses the NIST 1 bar standard state; "
                "standard_oxygen_pressure_bar must be 1.0"
            )
    return VacancyAnalysisConfig(
        enabled=enabled,
        oxygen_reference_mode=mode,
        oxygen_reference_file=resolve(
            "oxygen_reference_file", "reference_structures/reference_energies.json"
        ),
        oxygen_reference_structure=resolve(
            "oxygen_reference_structure", "reference_structures/gas/O2.POSCAR"
        ),
        oxygen_reference_relax=bool(section.get("oxygen_reference_relax", False)),
        mu_O_reference_eV=explicit,
        allow_unverified_oxygen_reference=bool(
            section.get("allow_unverified_oxygen_reference", False)
        ),
        delta_mu_O_min_eV=lower,
        delta_mu_O_max_eV=upper,
        delta_mu_O_points_eV=points,
        thermodynamic_tolerance_eV=tolerance,
        analysis_energy_source=source,
        exclude_unconverged=bool(section.get("exclude_unconverged", True)),
        pressure_mapping=bool(section.get("pressure_mapping", True)),
        temperatures_K=temperatures,
        standard_oxygen_pressure_bar=standard_pressure,
        log10_pO2_min_bar=pressure_min,
        log10_pO2_max_bar=pressure_max,
        log10_pO2_step=pressure_step,
        oxygen_standard_state_mode=standard_mode,
        oxygen_standard_state_temperatures_K=table_temperatures,
        oxygen_standard_state_delta_mu_eV_per_O=table_values,
    )


def _calculator_metadata(data: dict[str, Any]) -> dict[str, str] | None:
    candidates = [data]
    refs = data.get("references", {}) or {}
    oxide = data.get("oxide_mode", {}) or {}
    gas_ref = str(oxide.get("gas_ref", "O2"))
    if isinstance(refs.get(gas_ref), dict):
        candidates.insert(0, refs[gas_ref])
    for candidate in candidates:
        if all(key in candidate for key in ("backend", "model")):
            return {
                "backend": str(candidate["backend"]),
                "model": str(candidate["model"]),
                "task": str(candidate.get("task", "")),
            }
    return None


def parse_oxygen_reference_file(
    path: Path,
    *,
    backend: str,
    model: str,
    task: str,
    allow_unverified: bool,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Oxygen reference file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    oxide = data.get("oxide_mode", {}) or {}
    refs = data.get("references", {}) or {}
    gas_ref = str(oxide.get("gas_ref", "O2")).strip()
    gas = refs.get(gas_ref)
    if not isinstance(gas, dict):
        raise ValueError(f"Oxygen reference file has no references.{gas_ref} entry")
    if "E_per_molecule_eV" in gas:
        raw = float(gas["E_per_molecule_eV"])
    elif "E_total_eV" in gas:
        raw = float(gas["E_total_eV"])
    else:
        raise ValueError(
            f"references.{gas_ref} lacks E_per_molecule_eV and E_total_eV"
        )
    shift = float(oxide.get("muO_shift_ev", 0.0))
    metadata = _calculator_metadata(data)
    expected = {"backend": backend, "model": model, "task": task}
    verified = metadata == expected
    if metadata is not None and not verified:
        raise ValueError(
            "Oxygen reference calculator is incompatible: "
            f"reference={metadata}, vacancy={expected}"
        )
    if metadata is None and not allow_unverified:
        raise ValueError(
            "Oxygen reference calculator metadata cannot be verified. Set "
            "[vacancies].allow_unverified_oxygen_reference=true only after "
            "deliberately accepting this limitation."
        )
    return {
        "mu_O_reference_eV": 0.5 * (raw + 2.0 * shift),
        "oxygen_reference_energy_eV": raw + 2.0 * shift,
        "oxygen_reference_source": str(path),
        "oxygen_reference_verified": verified,
        "oxygen_reference_calculator_metadata": metadata,
        "gas_ref": gas_ref,
        "muO_shift_ev": shift,
    }


def resolve_oxygen_reference(
    cfg: VacancyAnalysisConfig,
    *,
    backend: str,
    model: str,
    task: str,
    calculator: Any = None,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    max_steps: int = 300,
) -> dict[str, Any]:
    base = {
        "oxygen_reference_mode": cfg.oxygen_reference_mode,
        "oxygen_reference_verified": False,
        "oxygen_reference_calculator_metadata": None,
        "oxygen_reference_limitation": None,
    }
    if cfg.oxygen_reference_mode == "none":
        return {
            **base,
            "mu_O_reference_eV": None,
            "oxygen_reference_energy_eV": None,
            "oxygen_reference_source": None,
        }
    if cfg.oxygen_reference_mode == "explicit":
        return {
            **base,
            "mu_O_reference_eV": cfg.mu_O_reference_eV,
            "oxygen_reference_energy_eV": None,
            "oxygen_reference_source": "user-supplied per-O value",
        }
    if cfg.oxygen_reference_mode == "reference_file":
        return {
            **base,
            **parse_oxygen_reference_file(
                cfg.oxygen_reference_file,
                backend=backend,
                model=model,
                task=task,
                allow_unverified=cfg.allow_unverified_oxygen_reference,
            ),
        }
    if calculator is None:
        raise ValueError("same_calculator oxygen reference requires the vacancy calculator")
    if not cfg.oxygen_reference_structure.exists():
        raise FileNotFoundError(
            f"Oxygen reference structure does not exist: {cfg.oxygen_reference_structure}"
        )
    molecule = Structure.from_file(cfg.oxygen_reference_structure)
    composition = molecule.composition.get_el_amt_dict()
    if set(composition) != {"O"} or int(composition.get("O", 0)) != 2:
        raise ValueError(
            "[vacancies].oxygen_reference_structure must contain exactly one O2 "
            "molecule (two oxygen atoms and no other species)"
        )
    relaxed = False
    output_structure = molecule
    if cfg.oxygen_reference_relax:
        output_structure, energy, steps, final_force, converged = (
            relax_structure_with_calculator(
                molecule,
                calculator=calculator,
                optimizer_name=optimizer,
                fmax=fmax,
                max_steps=max_steps,
                relax_mode="atoms",
                cell_filter="frechet",
            )
        )
        if not converged:
            raise RuntimeError("Same-calculator O2 relaxation did not converge")
        relaxed = True
    else:
        energy = structure_energy_with_calculator(molecule, calculator)
        steps = 0
        final_force = None
    return {
        **base,
        "mu_O_reference_eV": 0.5 * float(energy),
        "oxygen_reference_energy_eV": float(energy),
        "oxygen_reference_source": str(cfg.oxygen_reference_structure),
        "oxygen_reference_verified": True,
        "oxygen_reference_calculator_metadata": {
            "backend": backend,
            "model": model,
            "task": task,
        },
        "oxygen_reference_relaxed": relaxed,
        "oxygen_reference_relax_steps": steps,
        "oxygen_reference_final_force": final_force,
        "oxygen_reference_structure_dict": output_structure.as_dict(),
        "oxygen_reference_limitation": (
            "Calculator-consistent, but a solid-state foundation model may not "
            "accurately describe isolated O2."
        ),
    }


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _composition_identity(row: dict[str, Any]) -> tuple[tuple[tuple[str, int], ...], int]:
    raw_counts = _json_dict(
        row.get("dopant_counts_json") or row.get("dopant_counts_from_parent")
    )
    counts = {str(key): int(value) for key, value in raw_counts.items()}
    n_total = int(row.get("n_total_cations", 0))
    return tuple(sorted(counts.items())), n_total


def format_actual_composition_key(counts: dict[str, int], n_total_cations: int) -> str:
    """Stable display key; exact integer counts remain the true grouping identity."""
    parts = []
    for element in sorted(counts):
        percent = 100.0 * counts[element] / n_total_cations
        text = f"{percent:.6f}".rstrip("0").rstrip(".")
        parts.append(f"{element}{text}")
    return "_".join(parts) if parts else "undoped"


def _energy_choice(
    row: dict[str, Any], cfg: VacancyAnalysisConfig
) -> tuple[float | None, str | None, str | None]:
    relaxed = _safe_float(row.get("energy_relaxed_total_eV"))
    converged = _as_bool(row.get("converged", False))
    if relaxed is not None and (converged or not cfg.exclude_unconverged):
        return relaxed, "relaxed", None
    if relaxed is not None and cfg.exclude_unconverged:
        reason = "unconverged"
    else:
        reason = "missing_relaxed_energy"
    if cfg.analysis_energy_source == "relaxed_or_single_point":
        single = _safe_float(row.get("energy_sp_total_eV"))
        if single is not None:
            return single, "single_point", None
    return None, None, reason


def _dynamic_columns(row: dict[str, Any], dopants: Sequence[str]) -> dict[str, Any]:
    counts = _json_dict(row["dopant_counts_json"])
    percentages = _json_dict(row["dopant_percentages_json"])
    output: dict[str, Any] = {}
    for element in dopants:
        output[f"n_{element}"] = int(counts.get(element, 0))
        output[f"percent_{element}"] = float(percentages.get(element, 0.0))
    return output


def _line_ties(lines: Sequence[dict[str, Any]], x: float, tolerance: float) -> list[int]:
    values = [
        (
            float(line["grand_potential_intercept_eV"])
            + int(line["n_vacancies"]) * x,
            int(line["n_vacancies"]),
        )
        for line in lines
    ]
    minimum = min(value for value, _ in values)
    return sorted(n for value, n in values if abs(value - minimum) <= tolerance)


def exact_stability_intervals(
    lines: Sequence[dict[str, Any]], lower: float, upper: float, tolerance: float
) -> list[dict[str, Any]]:
    if not lines:
        return []
    boundaries = {float(lower), float(upper)}
    for index, left in enumerate(lines):
        for right in lines[index + 1 :]:
            dn = int(left["n_vacancies"]) - int(right["n_vacancies"])
            if dn == 0:
                continue
            crossing = (
                float(right["grand_potential_intercept_eV"])
                - float(left["grand_potential_intercept_eV"])
            ) / dn
            if lower < crossing < upper:
                boundaries.add(float(crossing))
    ordered = sorted(boundaries)
    intervals: list[dict[str, Any]] = []
    for lo, hi in zip(ordered, ordered[1:]):
        if hi - lo <= tolerance:
            continue
        midpoint = 0.5 * (lo + hi)
        tied = _line_ties(lines, midpoint, tolerance)
        for stable_n in tied:
            line = next(item for item in lines if int(item["n_vacancies"]) == stable_n)
            item = {
                **line,
                "delta_mu_O_lower_eV": lo,
                "delta_mu_O_upper_eV": hi,
                "stable_n_vacancies": stable_n,
                "lower_boundary_tied_counts": _line_ties(lines, lo, tolerance),
                "upper_boundary_tied_counts": _line_ties(lines, hi, tolerance),
            }
            merge_previous = (
                intervals
                and intervals[-1]["stable_n_vacancies"] == stable_n
                and abs(float(intervals[-1]["delta_mu_O_upper_eV"]) - lo)
                <= tolerance
            )
            if merge_previous:
                intervals[-1]["delta_mu_O_upper_eV"] = hi
                intervals[-1]["upper_boundary_tied_counts"] = item["upper_boundary_tied_counts"]
            else:
                intervals.append(item)
    return intervals


def oxygen_pressure_delta_mu_eV_per_O(
    temperature_K: float,
    oxygen_partial_pressure_bar: float,
    standard_oxygen_pressure_bar: float = 1.0,
) -> float:
    """Ideal-gas oxygen pressure contribution per oxygen atom."""
    if temperature_K <= 0 or oxygen_partial_pressure_bar <= 0:
        raise ValueError("Temperature and oxygen partial pressure must be positive")
    if standard_oxygen_pressure_bar <= 0:
        raise ValueError("Standard oxygen pressure must be positive")
    return 0.5 * K_B_EV_PER_K * temperature_K * math.log(
        oxygen_partial_pressure_bar / standard_oxygen_pressure_bar
    )


def inverse_oxygen_pressure_log10(
    delta_mu_O_total_eV_per_O: float,
    temperature_K: float,
    delta_mu_O_standard_eV_per_O: float = 0.0,
) -> float:
    """Return log10(pO2/p_standard) for a target oxygen chemical potential."""
    if temperature_K <= 0:
        raise ValueError("Temperature must be positive")
    return 2.0 * (
        delta_mu_O_total_eV_per_O - delta_mu_O_standard_eV_per_O
    ) / (K_B_EV_PER_K * temperature_K * math.log(10.0))


def nist_o2_standard_state_delta_mu_eV_per_O(temperature_K: float) -> float:
    """Return the NIST O2 standard-state correction, excluding explicit ZPE."""
    if not 100.0 <= temperature_K <= 6000.0:
        raise ValueError(
            "NIST O2 Shomate coefficients support 100 <= T <= 6000 K; "
            "extrapolation is not allowed"
        )
    coefficients = None
    for index, values in enumerate(NIST_O2_SHOMATE_COEFFICIENTS):
        lower, upper, *_ = values
        if lower <= temperature_K < upper or (
            index == len(NIST_O2_SHOMATE_COEFFICIENTS) - 1
            and temperature_K == upper
        ):
            coefficients = values
            break
    assert coefficients is not None
    _, _, a, b, c, d, e, f, g, h = coefficients
    reduced_temperature = temperature_K / 1000.0
    enthalpy_minus_h298_kj_per_mol = (
        a * reduced_temperature
        + b * reduced_temperature**2 / 2.0
        + c * reduced_temperature**3 / 3.0
        + d * reduced_temperature**4 / 4.0
        - e / reduced_temperature
        + f
        - h
    )
    entropy_j_per_mol_K = (
        a * math.log(reduced_temperature)
        + b * reduced_temperature
        + c * reduced_temperature**2 / 2.0
        + d * reduced_temperature**3 / 3.0
        - e / (2.0 * reduced_temperature**2)
        + g
    )
    enthalpy_minus_h0_kj_per_mol = (
        enthalpy_minus_h298_kj_per_mol
        - NIST_O2_H0_MINUS_H298_KJ_PER_MOL
    )
    correction_kj_per_mol_o2 = (
        enthalpy_minus_h0_kj_per_mol
        - temperature_K * entropy_j_per_mol_K / 1000.0
    )
    return correction_kj_per_mol_o2 / (2.0 * KJ_PER_MOL_PER_EV)


def oxygen_standard_state_delta_mu(
    cfg: VacancyAnalysisConfig, temperature_K: float
) -> float:
    """Interpolate the optional per-O standard-state thermal correction."""
    if cfg.oxygen_standard_state_mode == "none":
        return 0.0
    if cfg.oxygen_standard_state_mode == "nist_shomate":
        return nist_o2_standard_state_delta_mu_eV_per_O(temperature_K)
    temperatures = cfg.oxygen_standard_state_temperatures_K
    values = cfg.oxygen_standard_state_delta_mu_eV_per_O
    if temperature_K < temperatures[0] or temperature_K > temperatures[-1]:
        raise ValueError(
            f"Oxygen standard-state table does not cover T={temperature_K:g} K"
        )
    for left, right, left_value, right_value in zip(
        temperatures, temperatures[1:], values, values[1:]
    ):
        if left <= temperature_K <= right:
            fraction = (temperature_K - left) / (right - left)
            return left_value + fraction * (right_value - left_value)
    return values[-1]


def _pressure_grid(cfg: VacancyAnalysisConfig) -> list[float]:
    count = int(
        math.floor(
            (cfg.log10_pO2_max_bar - cfg.log10_pO2_min_bar)
            / cfg.log10_pO2_step
            + 1.0e-12
        )
    )
    values = [
        cfg.log10_pO2_min_bar + index * cfg.log10_pO2_step
        for index in range(count + 1)
    ]
    if not values or values[-1] < cfg.log10_pO2_max_bar - 1.0e-10:
        values.append(cfg.log10_pO2_max_bar)
    return values


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _write_outputs(path: Path, stem: str, rows: Sequence[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = path / f"{stem}.csv"
    json_path = path / f"{stem}.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def analyze_vacancy_thermodynamics(
    *,
    rows: Sequence[dict[str, Any]],
    analysis_cfg: VacancyAnalysisConfig,
    parent_root: Path,
    backend: str,
    model: str,
    task: str,
    calculator: Any = None,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    max_steps: int = 300,
    source_database: Path | None = None,
) -> dict[str, Path]:
    reference = resolve_oxygen_reference(
        analysis_cfg,
        backend=backend,
        model=model,
        task=task,
        calculator=calculator,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
    )
    exclusions: Counter[str] = Counter()
    usable: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    dopant_union: set[str] = set()
    for original in rows:
        row = dict(original)
        raw_counts = _json_dict(
            row.get("dopant_counts_json") or row.get("dopant_counts_from_parent")
        )
        counts = {str(key): int(value) for key, value in raw_counts.items()}
        n_total = int(row.get("n_total_cations", 0))
        n_host = int(row.get("n_host", 0))
        if n_total <= 0 or n_host <= 0:
            exclusions["missing_actual_composition"] += 1
            continue
        percentages = {element: 100.0 * count / n_total for element, count in counts.items()}
        row.update(
            {
                "dopant_counts_json": counts,
                "dopant_percentages_json": percentages,
                "actual_composition_key": format_actual_composition_key(counts, n_total),
                "n_total_dopants": sum(counts.values()),
                "total_dopant_percent": 100.0 * sum(counts.values()) / n_total,
            }
        )
        dopant_union.update(counts)
        prepared.append(row)
        energy, source, reason = _energy_choice(row, analysis_cfg)
        if energy is None:
            exclusions[reason or "excluded"] += 1
            continue
        row["analysis_energy_eV"] = energy
        row["energy_source"] = source
        usable.append(row)

    dopants = sorted(dopant_union)
    groups: dict[tuple[tuple[tuple[str, int], ...], int], list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        groups[_composition_identity(row)].append(row)
    all_identities = {_composition_identity(row) for row in prepared}
    display_by_identity = {
        identity: format_actual_composition_key(dict(identity[0]), identity[1])
        for identity in all_identities
    }
    display_counts = Counter(display_by_identity.values())
    for identity, display in list(display_by_identity.items()):
        if display_counts[display] > 1:
            count_suffix = "_".join(f"{element}{count}" for element, count in identity[0])
            display_by_identity[identity] = (
                f"{display}__sites{identity[1]}_counts{count_suffix or 'none'}"
            )
    minima: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = [
        {
            "actual_composition_key": display_by_identity[identity],
            "reason": "no valid analysis energies after convergence/source filtering",
        }
        for identity in sorted(all_identities - set(groups))
    ]
    missing: dict[str, list[int]] = {}
    for identity, composition_rows in sorted(groups.items()):
        calculators = {
            (
                str(row.get("backend", "")),
                str(row.get("model", "")),
                str(row.get("task", "")),
            )
            for row in composition_rows
        }
        species_signatures = {
            (
                tuple(sorted(_json_dict(row["dopant_counts_json"]).items())),
                int(row["n_host"]),
                int(row["n_total_cations"]),
            )
            for row in composition_rows
        }
        key = display_by_identity[identity]
        if len(calculators) != 1 or len(species_signatures) != 1:
            failed.append(
                {
                    "actual_composition_key": key,
                    "reason": "inconsistent integer composition or calculator metadata",
                }
            )
            continue
        parent_candidates = [row for row in composition_rows if int(row["n_vacancies"]) == 0]
        if not parent_candidates:
            failed.append(
                {
                    "actual_composition_key": key,
                    "reason": "no valid zero-vacancy parent energy",
                }
            )
            continue
        parent = min(
            parent_candidates,
            key=lambda row: (
                float(row["analysis_energy_eV"]),
                str(row.get("parent_id", "")),
            ),
        )
        by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in composition_rows:
            by_count[int(row["n_vacancies"])].append(row)
        expected = sorted(
            int(row["n_vacancies"])
            for row in prepared
            if _composition_identity(row) == identity
        )
        missing[key] = sorted(set(expected) - set(by_count))
        for n_vacancies, candidates in sorted(by_count.items()):
            winner = min(
                candidates,
                key=lambda row: (
                    float(row["analysis_energy_eV"]),
                    str(row.get("parent_id", "")),
                    str(row.get("configuration_id", "")),
                ),
            )
            delta = float(winner["analysis_energy_eV"]) - float(parent["analysis_energy_eV"])
            mu = reference["mu_O_reference_eV"]
            intercept = None if mu is None else delta + n_vacancies * float(mu)
            n_oxygen = int(
                winner.get(
                    "n_oxygen_sites_parent",
                    winner.get("vacancy_species_sites_in_parent", 0),
                )
            )
            result = {
                "composition_directory": winner.get(
                    "composition_directory", winner.get("composition", "")
                ),
                "actual_composition_key": key,
                "host_species": winner.get("host_species", ""),
                "vacancy_species": winner.get("vacancy_species", ""),
                "n_host": int(winner["n_host"]),
                "n_total_dopants": int(winner["n_total_dopants"]),
                "n_total_cations": int(winner["n_total_cations"]),
                "total_dopant_percent": float(winner["total_dopant_percent"]),
                "dopant_counts_json": winner["dopant_counts_json"],
                "dopant_percentages_json": winner["dopant_percentages_json"],
                "n_vacancies": n_vacancies,
                "n_oxygen_sites_parent": n_oxygen,
                "vacancy_fraction_of_parent_oxygen": n_vacancies / n_oxygen if n_oxygen else None,
                "vacancy_percent_of_parent_oxygen": (
                    100.0 * n_vacancies / n_oxygen if n_oxygen else None
                ),
                "vacancies_per_cation": n_vacancies / int(winner["n_total_cations"]),
                "source_parent_id": winner.get("parent_id"),
                "source_configuration_id": winner.get("configuration_id"),
                "source_relaxed_poscar_path": winner.get("relaxed_poscar_path"),
                "energy_source": winner["energy_source"],
                "parent_energy_source": parent["energy_source"],
                "converged": _as_bool(winner.get("converged", False)),
                "energy_relaxed_min_eV": float(winner["analysis_energy_eV"]),
                "parent_energy_relaxed_min_eV": float(parent["analysis_energy_eV"]),
                "delta_energy_to_parent_eV": 0.0 if n_vacancies == 0 else delta,
                "mu_O_reference_eV": mu,
                "oxygen_reference_mode": analysis_cfg.oxygen_reference_mode,
                "oxygen_reference_verified": reference["oxygen_reference_verified"],
                "oxygen_reference_source": reference["oxygen_reference_source"],
                "static_lattice_approximation": STATIC_LATTICE_APPROXIMATION,
                "neglected_solid_terms": list(NEGLECTED_SOLID_TERMS),
                "grand_potential_intercept_eV": (
                    0.0 if n_vacancies == 0 and mu is not None else intercept
                ),
                "grand_potential_intercept_per_vacancy_eV": (
                    intercept / n_vacancies
                    if intercept is not None and n_vacancies
                    else None
                ),
                "delta_Q_values": winner.get("delta_Q_values", []),
                "residual_charge_values": winner.get("residual_charge_values", []),
                "has_fully_compensated_scenario": bool(
                    winner.get("has_fully_compensated_scenario", False)
                ),
                "backend": backend,
                "model": model,
                "task": task,
            }
            result.update(_dynamic_columns(result, dopants))
            minima.append(result)

    intervals: list[dict[str, Any]] = []
    best: list[dict[str, Any]] = []
    if reference["mu_O_reference_eV"] is None:
        message = "oxygen_reference_mode='none': minima written; no cross-count stability claim"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        warning_messages = [message]
    else:
        warning_messages = []
        for key in sorted({row["actual_composition_key"] for row in minima}):
            lines = [row for row in minima if row["actual_composition_key"] == key]
            for interval in exact_stability_intervals(
                lines,
                analysis_cfg.delta_mu_O_min_eV,
                analysis_cfg.delta_mu_O_max_eV,
                analysis_cfg.thermodynamic_tolerance_eV,
            ):
                interval["stable_vacancy_percent"] = interval["vacancy_percent_of_parent_oxygen"]
                interval["stable_vacancies_per_cation"] = interval["vacancies_per_cation"]
                interval["line_slope_n_vacancies"] = interval["stable_n_vacancies"]
                intervals.append(interval)
            for point in analysis_cfg.delta_mu_O_points_eV:
                tied = _line_ties(lines, point, analysis_cfg.thermodynamic_tolerance_eV)
                values = {
                    int(line["n_vacancies"]): float(
                        line["grand_potential_intercept_eV"]
                    )
                    + int(line["n_vacancies"]) * point
                    for line in lines
                }
                representative = next(line for line in lines if int(line["n_vacancies"]) == tied[0])
                best.append({
                    **representative,
                    "delta_mu_O_eV": point,
                    "best_n_vacancies": tied[0] if len(tied) == 1 else None,
                    "best_vacancy_percent": (
                        representative["vacancy_percent_of_parent_oxygen"]
                        if len(tied) == 1
                        else None
                    ),
                    "best_vacancies_per_cation": (
                        representative["vacancies_per_cation"]
                        if len(tied) == 1
                        else None
                    ),
                    "minimum_delta_grand_potential_eV": min(values.values()),
                    "minimum_static_grand_potential_eV": min(values.values()),
                    "is_tied": len(tied) > 1,
                    "tied_n_vacancies": tied,
                })

    pressure_rows: list[dict[str, Any]] = []
    pressure_is_approximate = (
        analysis_cfg.pressure_mapping
        and analysis_cfg.oxygen_standard_state_mode == "none"
    )
    pressure_approximation = (
        "O2 standard-state thermal correction omitted"
        if pressure_is_approximate
        else (
            "NIST O2 Shomate standard-state correction evaluated continuously"
            if analysis_cfg.oxygen_standard_state_mode == "nist_shomate"
            else "User-supplied O2 standard-state thermal correction interpolated"
        )
    )
    standard_state_source = (
        NIST_O2_SHOMATE_SOURCE
        if analysis_cfg.oxygen_standard_state_mode == "nist_shomate"
        else (
            "user-supplied table"
            if analysis_cfg.oxygen_standard_state_mode == "user_table"
            else None
        )
    )
    warning_messages.append(STATIC_LATTICE_APPROXIMATION)
    for composition, missing_counts in sorted(missing.items()):
        if missing_counts:
            missing_message = (
                f"{composition}: no valid relaxed static energy for vacancy "
                f"count(s) {missing_counts}; excluded from the lower envelope."
            )
            warning_messages.append(missing_message)
            warnings.warn(missing_message, RuntimeWarning, stacklevel=2)
    if analysis_cfg.pressure_mapping and pressure_is_approximate:
        approximate_message = (
            "Approximate pressure mapping: the O2 standard-state thermal "
            "correction is omitted."
        )
        warning_messages.append(approximate_message)
        warnings.warn(approximate_message, RuntimeWarning, stacklevel=2)
    if (
        analysis_cfg.pressure_mapping
        and reference["mu_O_reference_eV"] is not None
    ):
        for key in sorted({row["actual_composition_key"] for row in minima}):
            lines = [row for row in minima if row["actual_composition_key"] == key]
            for temperature in analysis_cfg.temperatures_K:
                standard_delta = oxygen_standard_state_delta_mu(
                    analysis_cfg, temperature
                )
                for log10_pressure in _pressure_grid(analysis_cfg):
                    pressure = 10.0**log10_pressure
                    pressure_delta = oxygen_pressure_delta_mu_eV_per_O(
                        temperature,
                        pressure,
                        analysis_cfg.standard_oxygen_pressure_bar,
                    )
                    total_delta = standard_delta + pressure_delta
                    tied = _line_ties(
                        lines,
                        total_delta,
                        analysis_cfg.thermodynamic_tolerance_eV,
                    )
                    values = {
                        int(line["n_vacancies"]): float(
                            line["grand_potential_intercept_eV"]
                        )
                        + int(line["n_vacancies"]) * total_delta
                        for line in lines
                    }
                    representative = next(
                        line
                        for line in lines
                        if int(line["n_vacancies"]) == tied[0]
                    )
                    pressure_rows.append(
                        {
                            **representative,
                            "temperature_K": temperature,
                            "oxygen_partial_pressure_bar": pressure,
                            "log10_oxygen_partial_pressure_bar": log10_pressure,
                            "standard_oxygen_pressure_bar": (
                                analysis_cfg.standard_oxygen_pressure_bar
                            ),
                            "delta_mu_O_standard_eV_per_O": standard_delta,
                            "delta_mu_O_pressure_eV_per_O": pressure_delta,
                            "delta_mu_O_total_eV_per_O": total_delta,
                            "best_n_vacancies": (
                                tied[0] if len(tied) == 1 else None
                            ),
                            "best_vacancy_percent": (
                                representative[
                                    "vacancy_percent_of_parent_oxygen"
                                ]
                                if len(tied) == 1
                                else None
                            ),
                            "best_vacancies_per_cation": (
                                representative["vacancies_per_cation"]
                                if len(tied) == 1
                                else None
                            ),
                            "minimum_static_grand_potential_eV": min(
                                values.values()
                            ),
                            "is_tied": len(tied) > 1,
                            "tied_n_vacancies": tied,
                            "pressure_mapping_is_approximate": (
                                pressure_is_approximate
                            ),
                            "pressure_mapping_approximation": (
                                pressure_approximation
                            ),
                            "oxygen_standard_state_mode": (
                                analysis_cfg.oxygen_standard_state_mode
                            ),
                            "oxygen_standard_state_source": standard_state_source,
                            "oxygen_standard_state_zpe_included": False,
                            "oxygen_standard_state_valid_temperature_range_K": (
                                [100.0, 6000.0]
                                if analysis_cfg.oxygen_standard_state_mode
                                == "nist_shomate"
                                else None
                            ),
                        }
                    )

    outputs: dict[str, Path] = {}
    compact_outputs = (
        ("vacancy_minima_by_composition", minima),
        ("vacancy_stability_intervals", intervals),
        ("vacancy_best_counts", best),
    )
    for name, output_rows in compact_outputs:
        csv_path, json_path = _write_outputs(parent_root, name, output_rows)
        outputs[f"{name}_csv"] = csv_path
        outputs[f"{name}_json"] = json_path
    static_outputs = (
        ("vacancy_static_minima", minima),
        ("vacancy_static_stability_intervals", intervals),
        ("vacancy_static_best_counts", best),
        ("vacancy_static_pressure_map", pressure_rows),
    )
    for name, output_rows in static_outputs:
        csv_path, json_path = _write_outputs(parent_root, name, output_rows)
        outputs[f"{name}_csv"] = csv_path
        outputs[f"{name}_json"] = json_path
    source_checksum = None
    if source_database is not None and source_database.exists():
        source_checksum = hashlib.sha256(source_database.read_bytes()).hexdigest()
    metadata = {
        "analysis_status": "complete" if not failed else "partial",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_vacancies_database": str(source_database) if source_database else None,
        "source_file_checksum": source_checksum,
        "source_checksum": source_checksum,
        "resolved_analysis_config": {
            **asdict(analysis_cfg),
            "static_thermodynamic_analysis": analysis_cfg.enabled,
            "static_energy_source": analysis_cfg.analysis_energy_source,
            "oxygen_reference_file": str(analysis_cfg.oxygen_reference_file),
            "oxygen_reference_structure": str(
                analysis_cfg.oxygen_reference_structure
            ),
        },
        **reference,
        "oxygen_reference_energy": reference.get("oxygen_reference_energy_eV"),
        "static_lattice_approximation": STATIC_LATTICE_APPROXIMATION,
        "neglected_solid_terms": list(NEGLECTED_SOLID_TERMS),
        "delta_mu_O_range": [analysis_cfg.delta_mu_O_min_eV, analysis_cfg.delta_mu_O_max_eV],
        "delta_mu_O_points": list(analysis_cfg.delta_mu_O_points_eV),
        "temperatures_K": list(analysis_cfg.temperatures_K),
        "pressure_range": {
            "log10_pO2_min_bar": analysis_cfg.log10_pO2_min_bar,
            "log10_pO2_max_bar": analysis_cfg.log10_pO2_max_bar,
            "log10_pO2_step": analysis_cfg.log10_pO2_step,
            "standard_oxygen_pressure_bar": analysis_cfg.standard_oxygen_pressure_bar,
        },
        "oxygen_standard_state_mode": analysis_cfg.oxygen_standard_state_mode,
        "oxygen_standard_state_source": standard_state_source,
        "oxygen_standard_state_zpe_included": False,
        "oxygen_standard_state_valid_temperature_range_K": (
            [100.0, 6000.0]
            if analysis_cfg.oxygen_standard_state_mode == "nist_shomate"
            else None
        ),
        "pressure_mapping_is_approximate": pressure_is_approximate,
        "number_of_input_rows": len(rows),
        "number_of_converged_rows": sum(
            1
            for row in prepared
            if _safe_float(row.get("energy_relaxed_total_eV")) is not None
            and _as_bool(row.get("converged", False))
        ),
        "number_of_excluded_rows": sum(exclusions.values()),
        "exclusion_reasons": dict(exclusions),
        "number_of_actual_compositions": len({row["actual_composition_key"] for row in minima}),
        "missing_vacancy_counts_by_composition": missing,
        "failed_compositions": failed,
        "warnings": warning_messages,
    }
    metadata_path = parent_root / "vacancy_analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    outputs["metadata"] = metadata_path
    static_metadata_path = parent_root / "vacancy_static_analysis_metadata.json"
    static_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    outputs["static_metadata"] = static_metadata_path
    return outputs


def analyze_vacancies_database(database_path: Path, config_path: Path) -> dict[str, Path]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = raw.get("vacancies", {}) or {}
    root = config_path.resolve().parent
    cfg = parse_vacancy_analysis_config(section, root)
    if database_path.suffix.lower() == ".json":
        rows = json.loads(database_path.read_text(encoding="utf-8"))
    else:
        with database_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    backend, model, task = normalize_backend_config(
        backend=str(section.get("backend", "m3gnet")),
        model=str(section.get("model", "default")),
        task=str(section.get("task", "")),
        section_name="vacancies",
    )
    calculator = None
    if cfg.oxygen_reference_mode == "same_calculator":
        device = str(section.get("device", "cpu"))
        prepare_backend_runtime(
            backend=backend,
            device=device,
            gpu_id=int(section.get("gpu_id", 0)),
            tf_threads=int(section.get("tf_threads", 1)),
            omp_threads=int(section.get("omp_threads", 1)),
        )
        calculator = build_ase_calculator(
            backend=backend, model=model, task=task, device=device
        )
    return analyze_vacancy_thermodynamics(
        rows=rows,
        analysis_cfg=cfg,
        parent_root=database_path.parent,
        backend=backend,
        model=model,
        task=task,
        calculator=calculator,
        optimizer=str(section.get("optimizer", "bfgs")),
        fmax=float(section.get("fmax", 0.05)),
        max_steps=int(section.get("max_steps", 300)),
        source_database=database_path,
    )

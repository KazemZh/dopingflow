"""Vacancy thermodynamics with calibrated oxygen references.

Legacy static-lattice modes are delegated unchanged to :mod:`vacancy_analysis`.
The additional ``global`` and ``chemistry-specific`` oxygen-reference modes fit
an effective per-O reference from the already calculated refs-build energies
and experimental 298 K binary-oxide formation enthalpies.

For calibrated modes the temperature-pressure map uses

    mu_O(T,p) = mu_O,cal(298 H reference)
                + 1/2 [H_O2(T) - H_O2(298) - T S_O2(T)]
                + 1/2 k_B T ln(p/p0)

which mirrors the thermochemical convention used when a 298 K formation-
enthalpy calibration replaces a raw isolated-O2 electronic-energy reference.
An optional ideal oxygen-vacancy configurational entropy can be included in
T-dependent pressure maps.  It is intentionally not applied to the
T-independent delta-mu stability intervals.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

from dopingflow.ml_backends import (
    build_ase_calculator,
    normalize_backend_config,
    prepare_backend_runtime,
)
from dopingflow.oxygen_calibration import (
    OxygenCalibrationRequest,
    chemistry_elements_from_minimum,
    fit_oxygen_reference,
    write_oxygen_calibration_report,
)
from dopingflow.vacancy_analysis import (
    K_B_EV_PER_K,
    NEGLECTED_SOLID_TERMS,
    NIST_O2_SHOMATE_SOURCE,
    STATIC_LATTICE_APPROXIMATION,
    VacancyAnalysisConfig as BaseVacancyAnalysisConfig,
    analyze_vacancies_database,
    analyze_vacancy_thermodynamics,
    exact_stability_intervals,
    inverse_oxygen_pressure_log10,
    nist_o2_standard_state_delta_mu_eV_per_O,
    oxygen_pressure_delta_mu_eV_per_O,
    oxygen_standard_state_delta_mu,
    parse_vacancy_analysis_config as parse_base_vacancy_analysis_config,
)

_CALIBRATED_MODES = {"global", "chemistry-specific"}
_SOLID_CONFIG_ENTROPY_MODES = {"none", "ideal"}
_KJ_PER_MOL_PER_EV = 96.4853321233
_O2_H298_MINUS_H0_KJ_PER_MOL = 8.683
_O_H298_MINUS_H0_EV = _O2_H298_MINUS_H0_KJ_PER_MOL / (2.0 * _KJ_PER_MOL_PER_EV)


@dataclass(frozen=True)
class StaticVacancyThermodynamicsConfig(BaseVacancyAnalysisConfig):
    requested_oxygen_reference_mode: str = "reference_file"
    oxygen_calibration_experimental_source: str = "kingsbury"
    oxygen_calibration_experimental_data: Path | None = None
    oxygen_calibration_dataset_cache_dir: Path | None = None
    oxygen_calibration_min_references: int = 2
    oxygen_calibration_include_host_oxide: bool = True
    solid_configurational_entropy: str = "none"


def _resolve_optional_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_static_vacancy_thermodynamics_config(
    section: dict[str, Any], root: Path
) -> StaticVacancyThermodynamicsConfig:
    requested = str(section.get("oxygen_reference_mode", "reference_file")).strip().lower()
    requested = requested.replace("_", "-")
    base_section = dict(section)
    if requested in _CALIBRATED_MODES:
        # The base parser intentionally knows only raw/reference/explicit modes.
        # Use its ``none`` mode for validation; enhanced analysis resolves the
        # calibrated reference after the static minima have been assembled.
        base_section["oxygen_reference_mode"] = "none"
    base = parse_base_vacancy_analysis_config(base_section, root)

    source = str(
        section.get("oxygen_calibration_experimental_source", "kingsbury")
    ).strip().lower()
    if source not in {"kingsbury", "kingsbury+custom", "custom"}:
        raise ValueError(
            "[vacancies].oxygen_calibration_experimental_source must be one of: "
            "kingsbury, kingsbury+custom, custom"
        )
    custom_data = _resolve_optional_path(
        root, section.get("oxygen_calibration_experimental_data")
    )
    if requested in _CALIBRATED_MODES and source in {"custom", "kingsbury+custom"}:
        if custom_data is None:
            raise ValueError(
                f"[vacancies].oxygen_calibration_experimental_source={source!r} "
                "requires oxygen_calibration_experimental_data"
            )
    minimum = int(section.get("oxygen_calibration_min_references", 2))
    if minimum < 1:
        raise ValueError("[vacancies].oxygen_calibration_min_references must be >= 1")

    entropy_mode = str(
        section.get("solid_configurational_entropy", "none")
    ).strip().lower()
    if entropy_mode not in _SOLID_CONFIG_ENTROPY_MODES:
        raise ValueError(
            "[vacancies].solid_configurational_entropy must be 'none' or 'ideal'"
        )

    return StaticVacancyThermodynamicsConfig(
        **asdict(base),
        requested_oxygen_reference_mode=requested,
        oxygen_calibration_experimental_source=source,
        oxygen_calibration_experimental_data=custom_data,
        oxygen_calibration_dataset_cache_dir=_resolve_optional_path(
            root, section.get("oxygen_calibration_dataset_cache_dir")
        ),
        oxygen_calibration_min_references=minimum,
        oxygen_calibration_include_host_oxide=bool(
            section.get("oxygen_calibration_include_host_oxide", True)
        ),
        solid_configurational_entropy=entropy_mode,
    )


# Backward-compatible alias used by vacancies.py.
VacancyAnalysisConfig = StaticVacancyThermodynamicsConfig
parse_vacancy_analysis_config = parse_static_vacancy_thermodynamics_config


def ideal_vacancy_configurational_entropy_eV_per_K(
    n_vacancies: int, n_oxygen_sites: int
) -> float:
    """Ideal binary mixing entropy for occupied/vacant oxygen sites."""
    n_vacancies = int(n_vacancies)
    n_oxygen_sites = int(n_oxygen_sites)
    if n_oxygen_sites <= 0:
        raise ValueError("n_oxygen_sites must be > 0")
    if n_vacancies < 0 or n_vacancies > n_oxygen_sites:
        raise ValueError("n_vacancies must satisfy 0 <= n_vacancies <= n_oxygen_sites")
    if n_vacancies in {0, n_oxygen_sites}:
        return 0.0
    fraction = n_vacancies / n_oxygen_sites
    return -K_B_EV_PER_K * n_oxygen_sites * (
        fraction * math.log(fraction)
        + (1.0 - fraction) * math.log(1.0 - fraction)
    )


def _line_values(
    lines: Sequence[dict[str, Any]],
    delta_mu_o: float,
    *,
    temperature_K: float | None = None,
    include_ideal_entropy: bool = False,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for line in lines:
        n = int(line["n_vacancies"])
        value = float(line["grand_potential_intercept_eV"]) + n * float(delta_mu_o)
        if include_ideal_entropy:
            if temperature_K is None:
                raise ValueError("temperature_K is required when entropy is enabled")
            value -= float(temperature_K) * float(
                line.get("solid_configurational_entropy_eV_per_K", 0.0)
            )
        values[n] = value
    return values


def _tied_counts(values: dict[int, float], tolerance: float) -> list[int]:
    minimum = min(values.values())
    return sorted(n for n, value in values.items() if abs(value - minimum) <= tolerance)


def _pressure_grid(cfg: StaticVacancyThermodynamicsConfig) -> list[float]:
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


def _write_pair(path: Path, stem: str, rows: Sequence[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = path / f"{stem}.csv"
    json_path = path / f"{stem}.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def _calibrated_standard_state_delta_mu(
    cfg: StaticVacancyThermodynamicsConfig, temperature_K: float
) -> float:
    """Thermal O correction appropriate for a 298 K enthalpy-calibrated reference."""
    if cfg.oxygen_standard_state_mode == "none":
        return 0.0
    if cfg.oxygen_standard_state_mode == "nist_shomate":
        # Base helper is H(T)-H(0)-T*S.  The calibrated reference was fitted to
        # 298 K formation enthalpies, so change the enthalpy origin to 298 K.
        return (
            nist_o2_standard_state_delta_mu_eV_per_O(temperature_K)
            - _O_H298_MINUS_H0_EV
        )
    # User tables are taken exactly as supplied; docs require the table to use
    # the same reference convention chosen by the user.
    return oxygen_standard_state_delta_mu(cfg, temperature_K)


def _enhanced_static_approximation(entropy_mode: str) -> str:
    if entropy_mode == "ideal":
        return (
            "Calibrated static-lattice approximation: solid internal energies are 0 K "
            "relaxed ML energies; an ideal oxygen-vacancy configurational entropy is "
            "included only in T-dependent pressure maps. Solid vibrational, zero-point, "
            "thermal-electronic, magnetic, anharmonic, thermal-expansion and pV terms "
            "are neglected."
        )
    return (
        "Calibrated static-lattice approximation: solid free energies are approximated "
        "by 0 K relaxed ML energies. Gas-phase O2 enthalpy/entropy and pressure terms "
        "are included in T-pO2 maps; solid vibrational, zero-point, configurational, "
        "thermal-electronic, magnetic, anharmonic, thermal-expansion and pV terms are "
        "neglected."
    )


def _postprocess_calibrated_outputs(
    *,
    outputs: dict[str, Path],
    cfg: StaticVacancyThermodynamicsConfig,
    parent_root: Path,
    backend: str,
    model: str,
    task: str,
) -> dict[str, Path]:
    minima_path = outputs["vacancy_static_minima_json"]
    minima: list[dict[str, Any]] = json.loads(minima_path.read_text(encoding="utf-8"))
    if not minima:
        return outputs

    requested_mode = cfg.requested_oxygen_reference_mode
    calibration_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    calibration_for_composition: dict[str, dict[str, Any]] = {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in minima:
        grouped.setdefault(str(row["actual_composition_key"]), []).append(row)

    for composition_key, lines in sorted(grouped.items()):
        target = (
            ()
            if requested_mode == "global"
            else chemistry_elements_from_minimum(lines[0])
        )
        cache_key = tuple(target)
        if cache_key not in calibration_cache:
            calibration_cache[cache_key] = fit_oxygen_reference(
                OxygenCalibrationRequest(
                    reference_file=cfg.oxygen_reference_file,
                    scope=requested_mode,
                    target_elements=tuple(target),
                    experimental_source=cfg.oxygen_calibration_experimental_source,
                    experimental_data=cfg.oxygen_calibration_experimental_data,
                    dataset_cache_dir=cfg.oxygen_calibration_dataset_cache_dir,
                    min_references=cfg.oxygen_calibration_min_references,
                    include_host_oxide=cfg.oxygen_calibration_include_host_oxide,
                ),
                backend=backend,
                model=model,
                task=task,
            )
        calibration_for_composition[composition_key] = calibration_cache[cache_key]

    report_path = write_oxygen_calibration_report(
        parent_root / "oxygen_calibration_report.json",
        calibration_cache.values(),
    )
    outputs["oxygen_calibration_report"] = report_path

    entropy_enabled = cfg.solid_configurational_entropy == "ideal"
    approximation = _enhanced_static_approximation(cfg.solid_configurational_entropy)
    for composition_key, lines in grouped.items():
        calibration = calibration_for_composition[composition_key]
        mu = float(calibration["mu_O_reference_eV"])
        for line in lines:
            n = int(line["n_vacancies"])
            delta = float(line["delta_energy_to_parent_eV"])
            n_sites = int(line.get("n_oxygen_sites_parent", 0))
            entropy = (
                ideal_vacancy_configurational_entropy_eV_per_K(n, n_sites)
                if entropy_enabled and n_sites > 0
                else 0.0
            )
            line.update(
                {
                    "mu_O_reference_eV": mu,
                    "oxygen_reference_mode": requested_mode,
                    "oxygen_reference_verified": True,
                    "oxygen_reference_source": str(report_path),
                    "oxygen_calibration_scope": requested_mode,
                    "oxygen_calibration_target_elements": calibration["target_elements"],
                    "oxygen_calibration_n_references": calibration["n_references"],
                    "oxygen_calibration_rmse_mu_eV_per_O": calibration[
                        "rmse_mu_spread_eV_per_O"
                    ],
                    "oxygen_calibration_formation_rmse_eV_per_formula": calibration[
                        "formation_enthalpy_rmse_eV_per_formula"
                    ],
                    "grand_potential_intercept_eV": delta + n * mu,
                    "grand_potential_intercept_per_vacancy_eV": (
                        (delta + n * mu) / n if n else None
                    ),
                    "solid_configurational_entropy_mode": cfg.solid_configurational_entropy,
                    "solid_configurational_entropy_eV_per_K": entropy,
                    "static_lattice_approximation": approximation,
                }
            )

    flattened_minima = [row for key in sorted(grouped) for row in grouped[key]]
    intervals: list[dict[str, Any]] = []
    best: list[dict[str, Any]] = []
    for composition_key in sorted(grouped):
        lines = grouped[composition_key]
        for interval in exact_stability_intervals(
            lines,
            cfg.delta_mu_O_min_eV,
            cfg.delta_mu_O_max_eV,
            cfg.thermodynamic_tolerance_eV,
        ):
            interval["stable_vacancy_percent"] = interval[
                "vacancy_percent_of_parent_oxygen"
            ]
            interval["stable_vacancies_per_cation"] = interval["vacancies_per_cation"]
            interval["line_slope_n_vacancies"] = interval["stable_n_vacancies"]
            intervals.append(interval)
        for point in cfg.delta_mu_O_points_eV:
            values = _line_values(lines, point)
            tied = _tied_counts(values, cfg.thermodynamic_tolerance_eV)
            representative = next(
                line for line in lines if int(line["n_vacancies"]) == tied[0]
            )
            best.append(
                {
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
                    "configurational_entropy_applied": False,
                }
            )

    pressure_rows: list[dict[str, Any]] = []
    pressure_is_approximate = (
        cfg.pressure_mapping and cfg.oxygen_standard_state_mode == "none"
    )
    if cfg.pressure_mapping:
        for composition_key in sorted(grouped):
            lines = grouped[composition_key]
            for temperature in cfg.temperatures_K:
                standard_delta = _calibrated_standard_state_delta_mu(cfg, temperature)
                for log10_pressure in _pressure_grid(cfg):
                    pressure = 10.0**log10_pressure
                    pressure_delta = oxygen_pressure_delta_mu_eV_per_O(
                        temperature,
                        pressure,
                        cfg.standard_oxygen_pressure_bar,
                    )
                    total_delta = standard_delta + pressure_delta
                    values = _line_values(
                        lines,
                        total_delta,
                        temperature_K=temperature,
                        include_ideal_entropy=entropy_enabled,
                    )
                    tied = _tied_counts(values, cfg.thermodynamic_tolerance_eV)
                    representative = next(
                        line for line in lines if int(line["n_vacancies"]) == tied[0]
                    )
                    pressure_rows.append(
                        {
                            **representative,
                            "temperature_K": temperature,
                            "oxygen_partial_pressure_bar": pressure,
                            "log10_oxygen_partial_pressure_bar": log10_pressure,
                            "standard_oxygen_pressure_bar": cfg.standard_oxygen_pressure_bar,
                            "delta_mu_O_standard_eV_per_O": standard_delta,
                            "delta_mu_O_pressure_eV_per_O": pressure_delta,
                            "delta_mu_O_total_eV_per_O": total_delta,
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
                            "minimum_static_grand_potential_eV": min(values.values()),
                            "minimum_finite_T_grand_potential_eV": min(values.values()),
                            "is_tied": len(tied) > 1,
                            "tied_n_vacancies": tied,
                            "pressure_mapping_is_approximate": pressure_is_approximate,
                            "pressure_mapping_approximation": (
                                "O2 standard-state thermal correction omitted"
                                if pressure_is_approximate
                                else "calibrated 298 K enthalpy reference + gas thermochemistry"
                            ),
                            "oxygen_standard_state_mode": cfg.oxygen_standard_state_mode,
                            "oxygen_standard_state_source": (
                                NIST_O2_SHOMATE_SOURCE
                                if cfg.oxygen_standard_state_mode == "nist_shomate"
                                else (
                                    "user-supplied table"
                                    if cfg.oxygen_standard_state_mode == "user_table"
                                    else None
                                )
                            ),
                            "oxygen_standard_state_enthalpy_origin_K": (
                                298.15
                                if cfg.oxygen_standard_state_mode == "nist_shomate"
                                else None
                            ),
                            "oxygen_standard_state_zpe_included": False,
                            "solid_configurational_entropy_applied": entropy_enabled,
                        }
                    )

    for stem, rows in (
        ("vacancy_minima_by_composition", flattened_minima),
        ("vacancy_stability_intervals", intervals),
        ("vacancy_best_counts", best),
        ("vacancy_static_minima", flattened_minima),
        ("vacancy_static_stability_intervals", intervals),
        ("vacancy_static_best_counts", best),
        ("vacancy_static_pressure_map", pressure_rows),
    ):
        csv_path, json_path = _write_pair(parent_root, stem, rows)
        outputs[f"{stem}_csv"] = csv_path
        outputs[f"{stem}_json"] = json_path

    neglected = list(NEGLECTED_SOLID_TERMS)
    if entropy_enabled and "configurational" in neglected:
        neglected.remove("configurational")
    calibration_map = {
        key: {
            "mu_O_reference_eV": value["mu_O_reference_eV"],
            "target_elements": value["target_elements"],
            "n_references": value["n_references"],
            "rmse_mu_spread_eV_per_O": value["rmse_mu_spread_eV_per_O"],
        }
        for key, value in calibration_for_composition.items()
    }
    for metadata_key in ("metadata", "static_metadata"):
        metadata_path = outputs[metadata_key]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "oxygen_reference_mode": requested_mode,
                "oxygen_reference_source": str(report_path),
                "oxygen_calibration_scope": requested_mode,
                "oxygen_calibration_experimental_source": (
                    cfg.oxygen_calibration_experimental_source
                ),
                "oxygen_calibration_report": str(report_path),
                "oxygen_calibration_by_composition": calibration_map,
                "oxygen_standard_state_enthalpy_origin_K": (
                    298.15
                    if cfg.oxygen_standard_state_mode == "nist_shomate"
                    else None
                ),
                "solid_configurational_entropy": cfg.solid_configurational_entropy,
                "configurational_entropy_applied_to": (
                    ["vacancy_static_pressure_map"] if entropy_enabled else []
                ),
                "static_lattice_approximation": approximation,
                "neglected_solid_terms": neglected,
                "pressure_mapping_is_approximate": pressure_is_approximate,
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    return outputs


def analyze_static_vacancy_thermodynamics(
    *,
    rows: Sequence[dict[str, Any]],
    cfg: StaticVacancyThermodynamicsConfig,
    parent_root: Path,
    calculator: Any = None,
    backend: str,
    model: str,
    task: str,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    max_steps: int = 300,
    source_database: Path | None = None,
) -> dict[str, Path]:
    """Create static-lattice outputs, with optional calibrated oxygen references."""
    requested = cfg.requested_oxygen_reference_mode
    if requested not in _CALIBRATED_MODES:
        return analyze_vacancy_thermodynamics(
            rows=rows,
            analysis_cfg=cfg,
            parent_root=parent_root,
            backend=backend,
            model=model,
            task=task,
            calculator=calculator,
            optimizer=optimizer,
            fmax=fmax,
            max_steps=max_steps,
            source_database=source_database,
        )

    # First let the established analysis core select minimum-energy structures
    # for every vacancy count.  It deliberately makes no cross-count stability
    # claim because the calibrated oxygen reference is applied below.
    base_cfg = replace(cfg, oxygen_reference_mode="none")
    outputs = analyze_vacancy_thermodynamics(
        rows=rows,
        analysis_cfg=base_cfg,
        parent_root=parent_root,
        backend=backend,
        model=model,
        task=task,
        calculator=calculator,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        source_database=source_database,
    )
    return _postprocess_calibrated_outputs(
        outputs=outputs,
        cfg=cfg,
        parent_root=parent_root,
        backend=backend,
        model=model,
        task=task,
    )


def analyze_vacancies_database_static(
    database_path: Path, config_path: Path
) -> dict[str, Path]:
    """Reprocess an existing vacancy database with enhanced static thermodynamics."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = raw.get("vacancies", {}) or {}
    root = config_path.resolve().parent
    cfg = parse_static_vacancy_thermodynamics_config(section, root)
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
    if cfg.requested_oxygen_reference_mode == "same_calculator":
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
    return analyze_static_vacancy_thermodynamics(
        rows=rows,
        cfg=cfg,
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


__all__ = [
    "K_B_EV_PER_K",
    "NEGLECTED_SOLID_TERMS",
    "NIST_O2_SHOMATE_SOURCE",
    "STATIC_LATTICE_APPROXIMATION",
    "StaticVacancyThermodynamicsConfig",
    "VacancyAnalysisConfig",
    "analyze_static_vacancy_thermodynamics",
    "analyze_vacancies_database_static",
    "exact_stability_intervals",
    "ideal_vacancy_configurational_entropy_eV_per_K",
    "inverse_oxygen_pressure_log10",
    "nist_o2_standard_state_delta_mu_eV_per_O",
    "oxygen_pressure_delta_mu_eV_per_O",
    "oxygen_standard_state_delta_mu",
    "parse_static_vacancy_thermodynamics_config",
    "parse_vacancy_analysis_config",
]

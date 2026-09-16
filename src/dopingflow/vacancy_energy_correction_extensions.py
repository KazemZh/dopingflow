"""Optional fitted M0/M1 corrections for oxygen-vacancy thermodynamics.

This module installs a small extension around the established vacancy static-
thermodynamics parser/analyzer.  The feature is opt-in and reuses the fitted
backend-specific M0/M1 correction; it does not fit a vacancy-specific model.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

from pymatgen.core import Structure

from dopingflow import vacancy_static_thermodynamics as _base
from dopingflow.corrections import (
    CorrectionApplication,
    CorrectionModel,
    _validate_active_model_semantics,
    apply_energy_correction,
    backend_signature_from_reference,
    combine_feature_vectors,
    evaluate_feature_vector,
    load_correction_model,
    model_path,
    validate_applicability_compatibility,
    validate_backend_compatibility,
    validate_candidate_energy_provenance,
    validate_reference_energy_provenance,
)


_ORIGINAL_PARSE = getattr(
    _base,
    "_energy_correction_original_parse",
    _base.parse_static_vacancy_thermodynamics_config,
)
_ORIGINAL_ANALYZE = getattr(
    _base,
    "_energy_correction_original_analyze",
    _base.analyze_static_vacancy_thermodynamics,
)
_DOUBLE_COUNTING_MODES = {"global", "chemistry-specific"}
_KJ_PER_MOL_PER_EV = 96.4853321233
_O2_H298_MINUS_H0_KJ_PER_MOL = 8.683
_O_H298_MINUS_H0_EV = _O2_H298_MINUS_H0_KJ_PER_MOL / (2.0 * _KJ_PER_MOL_PER_EV)


@dataclass(frozen=True)
class EnergyCorrectedStaticVacancyThermodynamicsConfig(
    _base.StaticVacancyThermodynamicsConfig
):
    """Vacancy thermodynamics plus explicit fitted-correction controls."""

    apply_fitted_energy_correction: bool = False
    correction_workflow_root: Path = Path(".")
    allow_legacy_energy_correction_provenance: bool = False


def parse_static_vacancy_thermodynamics_config(
    section: dict[str, Any], root: Path
) -> EnergyCorrectedStaticVacancyThermodynamicsConfig:
    base_cfg = _ORIGINAL_PARSE(section, root)
    enabled = bool(section.get("apply_fitted_energy_correction", False))
    allow_legacy = bool(
        section.get("allow_legacy_energy_correction_provenance", False)
    )

    if enabled and base_cfg.requested_oxygen_reference_mode in _DOUBLE_COUNTING_MODES:
        raise ValueError(
            "[vacancies].apply_fitted_energy_correction=true cannot be combined with "
            "oxygen_reference_mode='global' or 'chemistry-specific'. Both use "
            "experimental formation-enthalpy information to correct oxygen-related "
            "systematic error, so combining them would double count the calibration. "
            "Use a raw same-backend oxygen reference, normally 'reference_file' or "
            "'same_calculator', when applying the fitted M0/M1 correction."
        )

    return EnergyCorrectedStaticVacancyThermodynamicsConfig(
        **asdict(base_cfg),
        apply_fitted_energy_correction=enabled,
        correction_workflow_root=root.resolve(),
        allow_legacy_energy_correction_provenance=allow_legacy,
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def _write_pair(
    parent_root: Path, stem: str, rows: Sequence[dict[str, Any]]
) -> tuple[Path, Path]:
    csv_path = parent_root / f"{stem}.csv"
    json_path = parent_root / f"{stem}.json"
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8"
    )
    return csv_path, json_path


def _snapshot_raw_outputs(
    outputs: dict[str, Path], parent_root: Path
) -> None:
    """Preserve the pre-correction thermodynamic tables for direct comparison."""

    stems = (
        "vacancy_static_minima",
        "vacancy_static_stability_intervals",
        "vacancy_static_best_counts",
        "vacancy_static_pressure_map",
        "vacancy_formation_free_energy",
    )
    for stem in stems:
        source = outputs.get(f"{stem}_json")
        if source is None or not source.is_file():
            continue
        rows = json.loads(source.read_text(encoding="utf-8"))
        raw_stem = f"{stem}_raw"
        csv_path, json_path = _write_pair(parent_root, raw_stem, rows)
        outputs[f"{raw_stem}_csv"] = csv_path
        outputs[f"{raw_stem}_json"] = json_path


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stored_file(parent_root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        local = (parent_root / path).resolve()
        if local.is_file():
            return local
    return None


def _minimum_paths(
    parent_root: Path,
    row: dict[str, Any],
) -> tuple[Path, Path]:
    """Resolve one selected minimum even after an HPC tree was copied locally."""

    n_vacancies = int(row["n_vacancies"])
    parent_id = str(row.get("source_parent_id", "")).strip()
    if not parent_id:
        raise ValueError("Vacancy minimum is missing source_parent_id")

    parts = Path(parent_id).parts
    candidate = parts[-1]
    composition_directory = str(row.get("composition_directory", "")).strip()
    if not composition_directory:
        if len(parts) < 2:
            raise ValueError(
                f"Cannot infer composition directory from source_parent_id={parent_id!r}"
            )
        composition_directory = parts[-2]
    candidate_root = parent_root / composition_directory / candidate
    stored = _stored_file(parent_root, row.get("source_relaxed_poscar_path"))

    if n_vacancies == 0:
        parent_reference = candidate_root / "05_vacancies" / "parent_reference"
        source: dict[str, Any] = {}
        source_path = parent_reference / "source.json"
        if source_path.is_file():
            try:
                source = json.loads(source_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                source = {}
        reused = _as_bool(source.get("parent_relaxation_reused", False))
        if reused:
            poscar = stored or candidate_root / "02_relax" / "POSCAR"
            meta = candidate_root / "02_relax" / "meta.json"
        else:
            poscar = stored or parent_reference / "relaxed" / "POSCAR"
            consistency_meta = parent_reference / "relaxed" / "meta.json"
            meta = (
                consistency_meta
                if consistency_meta.is_file()
                else candidate_root / "02_relax" / "meta.json"
            )
        return poscar.resolve(), meta.resolve()

    if stored is not None:
        return stored, (stored.parent / "meta.json").resolve()

    vacancy_species = str(row.get("vacancy_species", "O")).strip() or "O"
    config_id = str(row.get("source_configuration_id", "")).strip()
    if not config_id:
        raise ValueError(
            f"Vacancy minimum n={n_vacancies} is missing source_configuration_id"
        )
    relax_dir = (
        candidate_root
        / "05_vacancies"
        / f"V_{vacancy_species}_{n_vacancies:02d}"
        / config_id
        / "02_relax"
    )
    return (relax_dir / "POSCAR").resolve(), (relax_dir / "meta.json").resolve()


def _load_fitted_m0_m1_model(
    cfg: EnergyCorrectedStaticVacancyThermodynamicsConfig,
) -> CorrectionModel:
    root = cfg.correction_workflow_root
    reference_path = root / "reference_structures" / "reference_energies.json"
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Fitted vacancy energy correction requires "
            f"{reference_path}. Run refs-build and corrections-fit first."
        )
    reference_data = json.loads(reference_path.read_text(encoding="utf-8"))
    validate_reference_energy_provenance(reference_data)
    signature = backend_signature_from_reference(reference_data, root=root)
    model = load_correction_model(model_path(root, signature))
    validate_backend_compatibility(model, signature)
    validate_applicability_compatibility(model)
    _validate_active_model_semantics(model)
    if model.model_family not in {"m0", "m1"}:
        raise ValueError(
            "Vacancy fitted-energy correction requires a fitted M0 or M1 model, but "
            f"the active artifact is family={model.model_family!r}. Configure "
            "[energy_correction].model_family as 'm0', 'm1', or 'auto', run "
            "dopingflow corrections-fit, and rerun vacancy analysis."
        )
    return model


def vacancy_reaction_correction(
    model: CorrectionModel,
    defect: CorrectionApplication,
    parent: CorrectionApplication,
) -> CorrectionApplication:
    """Return C(defect)-C(parent) with correlated parameter uncertainty."""

    vector = combine_feature_vectors(
        model.correction_terms,
        ((1.0, defect.feature_vector), (-1.0, parent.feature_vector)),
    )
    matched = tuple(
        term
        for term, value in zip(model.correction_terms, vector, strict=True)
        if abs(float(value)) > 0.0
    )
    return evaluate_feature_vector(
        model,
        vector,
        matched_terms=matched,
        reason="vacancy_reaction_correction",
        oxygen_kind=defect.oxide_type,
    )


def _correction_for_minimum(
    *,
    row: dict[str, Any],
    parent_root: Path,
    model: CorrectionModel,
    allow_legacy: bool,
) -> tuple[CorrectionApplication, dict[str, Any], Path]:
    poscar, meta_path = _minimum_paths(parent_root, row)
    if not poscar.is_file():
        raise FileNotFoundError(f"Vacancy correction structure not found: {poscar}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Vacancy correction metadata not found: {meta_path}")

    structure = Structure.from_file(str(poscar))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if "fmax_target_eV_per_A" not in metadata and "fmax_target" in metadata:
        metadata["fmax_target_eV_per_A"] = metadata["fmax_target"]

    provenance = validate_candidate_energy_provenance(
        metadata,
        poscar,
        model,
        label=(
            f"vacancy {row.get('actual_composition_key')} "
            f"n={int(row['n_vacancies'])}"
        ),
        allow_legacy=allow_legacy,
    )
    application = apply_energy_correction(
        model,
        structure.composition,
        structure=structure,
    )
    return application, provenance, poscar


def _tied_counts(values: dict[int, float], tolerance: float) -> list[int]:
    minimum = min(values.values())
    return sorted(
        count for count, value in values.items() if abs(value - minimum) <= tolerance
    )


def _rebuild_static_selection_outputs(
    *,
    minima: list[dict[str, Any]],
    cfg: EnergyCorrectedStaticVacancyThermodynamicsConfig,
    parent_root: Path,
    outputs: dict[str, Path],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in minima:
        grouped.setdefault(str(row["actual_composition_key"]), []).append(row)

    intervals: list[dict[str, Any]] = []
    best: list[dict[str, Any]] = []
    for key in sorted(grouped):
        lines = sorted(grouped[key], key=lambda row: int(row["n_vacancies"]))
        if any(line.get("grand_potential_intercept_eV") is None for line in lines):
            continue

        for interval in _base.exact_stability_intervals(
            lines,
            cfg.delta_mu_O_min_eV,
            cfg.delta_mu_O_max_eV,
            cfg.thermodynamic_tolerance_eV,
        ):
            interval["stable_vacancy_percent"] = interval[
                "vacancy_percent_of_parent_oxygen"
            ]
            interval["stable_vacancies_per_cation"] = interval[
                "vacancies_per_cation"
            ]
            interval["line_slope_n_vacancies"] = interval["stable_n_vacancies"]
            intervals.append(interval)

        for point in cfg.delta_mu_O_points_eV:
            values = {
                int(line["n_vacancies"]): (
                    float(line["grand_potential_intercept_eV"])
                    + int(line["n_vacancies"]) * float(point)
                )
                for line in lines
            }
            tied = _tied_counts(values, cfg.thermodynamic_tolerance_eV)
            representative = next(
                line for line in lines if int(line["n_vacancies"]) == tied[0]
            )
            best.append(
                {
                    **representative,
                    "delta_mu_O_eV": float(point),
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
                }
            )

    for stem, rows in (
        ("vacancy_minima_by_composition", minima),
        ("vacancy_static_minima", minima),
        ("vacancy_stability_intervals", intervals),
        ("vacancy_static_stability_intervals", intervals),
        ("vacancy_best_counts", best),
        ("vacancy_static_best_counts", best),
    ):
        csv_path, json_path = _write_pair(parent_root, stem, rows)
        outputs[f"{stem}_csv"] = csv_path
        outputs[f"{stem}_json"] = json_path


def _annotate_finite_temperature_outputs(
    outputs: dict[str, Path], parent_root: Path
) -> None:
    """Add a raw value evaluated with the original raw-O2 thermal convention."""

    path = outputs.get("vacancy_formation_free_energy_json")
    if path is None or not path.is_file():
        return
    rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        raw_intercept = row.get("grand_potential_intercept_raw_eV")
        if raw_intercept is not None:
            n_vacancies = int(row.get("n_vacancies", 0))
            standard_delta = float(row.get("delta_mu_O_standard_eV_per_O", 0.0))
            if str(row.get("oxygen_standard_state_mode", "")).lower() == "nist_shomate":
                # Corrected M0/M1 finite-T output uses a 298 K enthalpy origin.
                # Undo that origin shift for the side-by-side raw reference value.
                standard_delta += _O_H298_MINUS_H0_EV
            pressure_delta = float(row.get("delta_mu_O_pressure_eV_per_O", 0.0))
            config_term = float(
                row.get("solid_configurational_free_energy_correction_eV", 0.0) or 0.0
            )
            raw_value = (
                float(raw_intercept)
                + n_vacancies * (standard_delta + pressure_delta)
                + config_term
            )
            row["vacancy_formation_free_energy_raw_eV"] = raw_value
            row["vacancy_formation_free_energy_raw_per_vacancy_eV"] = (
                raw_value / n_vacancies if n_vacancies else None
            )
        row["vacancy_formation_free_energy_corrected_eV"] = row.get(
            "vacancy_formation_free_energy_eV"
        )
        row["vacancy_formation_free_energy_corrected_per_vacancy_eV"] = row.get(
            "vacancy_formation_free_energy_per_vacancy_eV"
        )

    csv_path, json_path = _write_pair(parent_root, "vacancy_formation_free_energy", rows)
    outputs["vacancy_formation_free_energy_csv"] = csv_path
    outputs["vacancy_formation_free_energy_json"] = json_path


def _update_metadata(
    *,
    outputs: dict[str, Path],
    model: CorrectionModel,
    allow_legacy: bool,
    oxygen_standard_state_mode: str,
) -> None:
    for key in ("metadata", "static_metadata"):
        path = outputs.get(key)
        if path is None or not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(
            {
                "fitted_energy_correction_applied": True,
                "fitted_energy_correction_model_family": model.model_family,
                "fitted_energy_correction_fit_id": model.fit_id,
                "fitted_energy_correction_terms": list(model.correction_terms),
                "fitted_energy_correction_coefficients_eV_per_term": list(
                    model.coefficients_eV_per_term
                ),
                "fitted_energy_correction_uncertainty_convention": (
                    "C(defect)-C(parent) evaluated from the correlated reaction "
                    "feature vector and fitted covariance matrix"
                ),
                "allow_legacy_energy_correction_provenance": bool(allow_legacy),
                "oxygen_reference_double_counting_guard": (
                    "global and chemistry-specific oxygen calibration are forbidden "
                    "when fitted M0/M1 vacancy correction is enabled"
                ),
                "finite_T_oxygen_enthalpy_origin_K": (
                    298.15 if oxygen_standard_state_mode == "nist_shomate" else None
                ),
            }
        )
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _apply_correction_to_outputs(
    *,
    outputs: dict[str, Path],
    cfg: EnergyCorrectedStaticVacancyThermodynamicsConfig,
    parent_root: Path,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Path]:
    minima_path = outputs.get("vacancy_static_minima_json")
    if minima_path is None or not minima_path.is_file():
        return outputs
    minima: list[dict[str, Any]] = json.loads(minima_path.read_text(encoding="utf-8"))
    if not minima:
        return outputs

    _snapshot_raw_outputs(outputs, parent_root)
    model = _load_fitted_m0_m1_model(cfg)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in minima:
        grouped.setdefault(str(row["actual_composition_key"]), []).append(row)

    cache: dict[str, tuple[CorrectionApplication, dict[str, Any], Path]] = {}

    def application(row: dict[str, Any]):
        poscar, _ = _minimum_paths(parent_root, row)
        key = str(poscar.resolve())
        if key not in cache:
            cache[key] = _correction_for_minimum(
                row=row,
                parent_root=parent_root,
                model=model,
                allow_legacy=cfg.allow_legacy_energy_correction_provenance,
            )
        return cache[key]

    for composition_key, lines in sorted(grouped.items()):
        parents = [line for line in lines if int(line["n_vacancies"]) == 0]
        if not parents:
            raise ValueError(
                f"M0/M1 vacancy correction requires n=0 parent for {composition_key}"
            )
        parent = min(
            parents,
            key=lambda row: float(row.get("energy_relaxed_min_eV", math.inf)),
        )
        parent_application, parent_provenance, _ = application(parent)
        parent_raw_energy = float(parent["energy_relaxed_min_eV"])

        for line in lines:
            defect_application, defect_provenance, _ = application(line)
            reaction = vacancy_reaction_correction(
                model, defect_application, parent_application
            )
            n_vacancies = int(line["n_vacancies"])
            raw_delta = float(line.get("delta_energy_to_parent_eV", 0.0))
            corrected_delta = (
                0.0
                if n_vacancies == 0
                else raw_delta + reaction.correction_eV
            )
            raw_energy = float(line["energy_relaxed_min_eV"])
            mu = line.get("mu_O_reference_eV")
            raw_intercept = line.get("grand_potential_intercept_eV")
            corrected_intercept = (
                None
                if mu is None
                else (
                    0.0
                    if n_vacancies == 0
                    else corrected_delta + n_vacancies * float(mu)
                )
            )

            line.update(
                {
                    "fitted_energy_correction_applied": True,
                    "correction_model_family": model.model_family,
                    "correction_fit_id": model.fit_id,
                    "correction_terms": list(model.correction_terms),
                    "correction_coefficients_eV_per_term": list(
                        model.coefficients_eV_per_term
                    ),
                    "energy_relaxed_min_raw_eV": raw_energy,
                    "parent_energy_relaxed_min_raw_eV": parent_raw_energy,
                    "delta_energy_to_parent_raw_eV": raw_delta,
                    "grand_potential_intercept_raw_eV": raw_intercept,
                    "energy_correction_eV": defect_application.correction_eV,
                    "parent_energy_correction_eV": parent_application.correction_eV,
                    "energy_corrected_min_eV": (
                        raw_energy + defect_application.correction_eV
                    ),
                    "parent_energy_corrected_min_eV": (
                        parent_raw_energy + parent_application.correction_eV
                    ),
                    "vacancy_reaction_correction_eV": reaction.correction_eV,
                    "vacancy_reaction_correction_uncertainty_eV": (
                        reaction.uncertainty_eV
                    ),
                    "correction_feature_vector": list(defect_application.feature_vector),
                    "parent_correction_feature_vector": list(
                        parent_application.feature_vector
                    ),
                    "vacancy_reaction_feature_vector": list(reaction.feature_vector),
                    "correction_oxide_type": defect_application.oxide_type,
                    "correction_provenance_mode": defect_provenance.get("mode"),
                    "correction_provenance_assumptions": defect_provenance.get(
                        "assumptions", []
                    ),
                    "parent_correction_provenance_mode": parent_provenance.get("mode"),
                    "parent_correction_provenance_assumptions": parent_provenance.get(
                        "assumptions", []
                    ),
                    # These established fields become the active corrected values.
                    "delta_energy_to_parent_eV": corrected_delta,
                    "grand_potential_intercept_eV": corrected_intercept,
                    "grand_potential_intercept_per_vacancy_eV": (
                        corrected_intercept / n_vacancies
                        if corrected_intercept is not None and n_vacancies
                        else None
                    ),
                }
            )

    flattened = [row for key in sorted(grouped) for row in grouped[key]]
    _rebuild_static_selection_outputs(
        minima=flattened,
        cfg=cfg,
        parent_root=parent_root,
        outputs=outputs,
    )

    # M0/M1 is calibrated against standard formation enthalpies near 298 K, so
    # corrected finite-T results use the 298 K gas enthalpy origin. Raw snapshots
    # above retain the original uncorrected convention.
    outputs = _base.augment_vacancy_formation_free_energy(
        outputs=outputs,
        rows=rows,
        cfg=cfg,
        parent_root=parent_root,
        calibrated_reference=True,
    )
    _annotate_finite_temperature_outputs(outputs, parent_root)
    _update_metadata(
        outputs=outputs,
        model=model,
        allow_legacy=cfg.allow_legacy_energy_correction_provenance,
        oxygen_standard_state_mode=cfg.oxygen_standard_state_mode,
    )
    return outputs


def analyze_static_vacancy_thermodynamics(
    *,
    rows: Sequence[dict[str, Any]],
    cfg: _base.StaticVacancyThermodynamicsConfig,
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
    outputs = _ORIGINAL_ANALYZE(
        rows=rows,
        cfg=cfg,
        parent_root=parent_root,
        calculator=calculator,
        backend=backend,
        model=model,
        task=task,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        source_database=source_database,
    )
    if not bool(getattr(cfg, "apply_fitted_energy_correction", False)):
        return outputs
    if not isinstance(cfg, EnergyCorrectedStaticVacancyThermodynamicsConfig):
        raise TypeError(
            "Fitted vacancy correction was requested without the enhanced vacancy config"
        )
    return _apply_correction_to_outputs(
        outputs=outputs,
        cfg=cfg,
        parent_root=parent_root,
        rows=rows,
    )


def install_extensions() -> None:
    if getattr(_base, "_energy_correction_extension_installed", False):
        return
    _base._energy_correction_original_parse = _ORIGINAL_PARSE
    _base._energy_correction_original_analyze = _ORIGINAL_ANALYZE
    _base.parse_static_vacancy_thermodynamics_config = (
        parse_static_vacancy_thermodynamics_config
    )
    _base.parse_vacancy_analysis_config = parse_static_vacancy_thermodynamics_config
    _base.analyze_static_vacancy_thermodynamics = analyze_static_vacancy_thermodynamics
    _base._energy_correction_extension_installed = True


install_extensions()


__all__ = [
    "EnergyCorrectedStaticVacancyThermodynamicsConfig",
    "analyze_static_vacancy_thermodynamics",
    "install_extensions",
    "parse_static_vacancy_thermodynamics_config",
    "vacancy_reaction_correction",
]

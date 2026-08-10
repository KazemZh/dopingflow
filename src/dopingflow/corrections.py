"""Backend-specific, uncertainty-aware energy corrections.

This module implements the linear correction framework described by Wang,
Kingsbury et al. without importing any of the pre-fitted Materials Project
parameters.  The fitted coefficients are valid only for the backend signature
and calibration data stored with the model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pymatgen.analysis.structure_analyzer import oxide_type as classify_oxide_type
from pymatgen.core import Composition, Structure

log = logging.getLogger(__name__)

CORRECTIONS_DIR = Path("reference_structures/corrections")
MODEL_FILENAME = "correction_parameters.json"
FIT_REPORT_FILENAME = "correction_fit_report.json"
MODEL_SELECTION_FILENAME = "correction_model_selection.json"
CALIBRATION_EXPANSION_FILENAME = "calibration_expansion_snapshot.json"
CANDIDATE_MODELS_DIRNAME = "candidate_models"
EXPERIMENTAL_SNAPSHOT_FILENAME = "experimental_calibration_used.json"
METADATA_FILENAME = "correction_metadata.json"

KINGSBURY_DATASET = "expt_formation_enthalpy_kingsbury"
KINGSBURY_DATASET_SHA256 = (
    "a2d2ced98d40349abd2041f169d9ed9c7f49453e86a77f82cab8c61c70dcb7ca"
)
KINGSBURY_FIELDS = (
    "formula",
    "expt_form_e",
    "uncertainty",
    "phaseinfo",
    "reference",
    "likely_mpid",
)

_OXYGEN_TERMS = {"oxide", "peroxide", "superoxide"}
_OXIDE_CATION_PREFIX = "oxide_cation:"
OXIDE_CLASSIFICATION_RELATIVE_CUTOFF = 1.05
CORRECTION_FRAMEWORK_VERSION = "dopingflow_kingsbury_wls_v1"
MODEL_SELECTION_POLICY_VERSION = "dopingflow_oxide_m0_m1_loo_v1"
DEFAULT_EXCLUDED_POLYANIONS = (
    "SO4",
    "SO3",
    "CO3",
    "NO3",
    "NO2",
    "OCl3",
    "ClO3",
    "ClO4",
    "HO",
    "ClO",
    "SeO3",
    "TiO3",
    "TiO4",
    "WO4",
    "SiO3",
    "SiO4",
    "Si2O5",
    "PO3",
    "PO4",
    "P2O7",
)
_CUSTOM_REQUIRED_COLUMNS = {
    "formula",
    "formation_enthalpy",
    "uncertainty",
    "phase",
    "temperature",
    "units",
    "source",
}
_EV_PER_ATOM_UNITS = {"ev/atom", "ev_per_atom", "ev atom-1"}
_EV_PER_FORMULA_UNITS = {
    "ev/formula",
    "ev/formula_unit",
    "ev/fu",
    "ev_per_formula",
    "ev_per_formula_unit",
}


@dataclass(frozen=True)
class CorrectionConfig:
    enabled: bool
    experimental_source: str
    experimental_data: Path | None
    dataset_cache_dir: Path | None
    calibration_manifest: Path
    correction_terms: tuple[str, ...]
    allow_element_terms: bool
    model_family: str
    target_elements: tuple[str, ...]
    m1_elements: tuple[str, ...]
    calibration_selection: str
    auto_fetch_phase_structures: bool
    optimade_base_url: str
    min_element_compounds: int
    min_element_stoichiometries: int
    min_cv_improvement_eV_per_atom: float
    require_cv_one_standard_error: bool
    exclude_polyanions: tuple[str, ...]
    max_relative_experimental_uncertainty: float
    max_calculated_e_above_hull_eV_per_atom: float | None
    allow_phase_mismatch: bool
    allow_legacy_candidate_provenance: bool
    reuse_fitted: bool
    min_degrees_of_freedom: int
    min_term_support: int
    max_condition_number: float
    poor_fit_rmse_warning_eV_per_atom: float


@dataclass(frozen=True)
class ExperimentalRecord:
    formula: str
    reduced_formula: str
    formation_enthalpy_eV_per_atom: float
    formation_enthalpy_eV_per_formula: float
    uncertainty_eV_per_atom: float | None
    uncertainty_eV_per_formula: float | None
    uncertainty_source: str
    phase: str
    temperature: str
    source: str
    likely_mpid: str
    doi: str
    reference_id: str
    notes: str
    dataset: str
    original_units: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorrectionApplication:
    correction_eV: float
    uncertainty_eV: float
    feature_vector: tuple[float, ...]
    matched_terms: tuple[str, ...]
    applied: bool
    reason: str
    oxide_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorrectionModel:
    schema_version: int
    method: str
    fit_id: str
    backend_signature: Mapping[str, Any]
    correction_terms: tuple[str, ...]
    coefficients_eV_per_term: tuple[float, ...]
    covariance_eV2: tuple[tuple[float, ...], ...]
    coefficient_uncertainties_eV_per_term: tuple[float, ...]
    experimental_dataset: str
    experimental_dataset_version: str
    fit_input_hash: str
    units: Mapping[str, str]
    calibration_formulas: tuple[str, ...]
    fit_metrics: Mapping[str, float]
    activation_input_hash: str = ""
    applicability_signature: Mapping[str, Any] | None = None
    model_family: str = "manual"
    selection_run_hash: str = ""
    target_elements: tuple[str, ...] = ()
    selection_metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["backend_signature"] = dict(self.backend_signature)
        data["units"] = dict(self.units)
        data["fit_metrics"] = dict(self.fit_metrics)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorrectionModel":
        return cls(
            schema_version=int(data["schema_version"]),
            method=str(data["method"]),
            fit_id=str(data["fit_id"]),
            backend_signature=dict(data["backend_signature"]),
            correction_terms=tuple(str(x) for x in data["correction_terms"]),
            coefficients_eV_per_term=tuple(
                float(x) for x in data["coefficients_eV_per_term"]
            ),
            covariance_eV2=tuple(
                tuple(float(x) for x in row) for row in data["covariance_eV2"]
            ),
            coefficient_uncertainties_eV_per_term=tuple(
                float(x) for x in data["coefficient_uncertainties_eV_per_term"]
            ),
            experimental_dataset=str(data["experimental_dataset"]),
            experimental_dataset_version=str(data.get("experimental_dataset_version", "")),
            fit_input_hash=str(data["fit_input_hash"]),
            units=dict(data["units"]),
            calibration_formulas=tuple(str(x) for x in data["calibration_formulas"]),
            fit_metrics={str(k): float(v) for k, v in data["fit_metrics"].items()},
            activation_input_hash=str(data.get("activation_input_hash", "")),
            applicability_signature=(
                dict(data["applicability_signature"])
                if data.get("applicability_signature") is not None
                else None
            ),
            model_family=str(data.get("model_family", "manual")),
            selection_run_hash=str(data.get("selection_run_hash", "")),
            target_elements=tuple(str(x) for x in data.get("target_elements", ())),
            selection_metadata=(
                dict(data["selection_metadata"])
                if data.get("selection_metadata") is not None
                else None
            ),
        )


def _resolve_optional_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def normalize_correction_term(term: str) -> str:
    value = str(term).strip()
    lower = value.lower()
    if lower in _OXYGEN_TERMS:
        return lower
    if lower.startswith("element:") or lower.startswith(_OXIDE_CATION_PREFIX):
        prefix = "element:" if lower.startswith("element:") else _OXIDE_CATION_PREFIX
        symbol = value.split(":", 1)[1].strip()
        try:
            composition = Composition(symbol)
        except Exception as exc:
            raise ValueError(f"Invalid correction term {term!r}") from exc
        if len(composition.elements) != 1 or composition.num_atoms != 1:
            raise ValueError(
                f"Element correction term must use one element symbol, got {term!r}"
            )
        return f"{prefix}{composition.elements[0].symbol}"
    raise ValueError(
        f"Unsupported correction term {term!r}; use oxide, peroxide, superoxide, "
        "an explicit element:<symbol> term, or an oxide_cation:<symbol> term"
    )


def _element_symbols(values: Iterable[Any], *, setting: str) -> tuple[str, ...]:
    symbols: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        try:
            composition = Composition(text)
        except Exception as exc:
            raise ValueError(f"{setting} contains invalid element {text!r}") from exc
        if len(composition.elements) != 1 or composition.num_atoms != 1:
            raise ValueError(f"{setting} must contain element symbols, got {text!r}")
        symbols.add(composition.elements[0].symbol)
    return tuple(sorted(symbols))


def infer_correction_target_elements(raw_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """Infer host/dopant cations without hard-coding a material system."""

    values: list[Any] = []
    host_formula = str((raw_cfg.get("references", {}) or {}).get("host", "")).strip()
    if host_formula:
        try:
            host_composition = Composition(host_formula)
        except Exception as exc:
            raise ValueError(f"[references].host has invalid formula {host_formula!r}") from exc
        values.extend(
            element.symbol
            for element in host_composition.elements
            if element.symbol != "O"
        )

    doping = raw_cfg.get("doping", {}) or {}
    values.append(doping.get("host_species", ""))
    mode = str(doping.get("mode", "explicit")).strip().lower()
    if mode == "enumerate":
        values.extend(doping.get("dopants", ()) or ())
        values.extend(doping.get("must_include", ()) or ())
    elif mode == "explicit":
        compositions = doping.get("compositions", ()) or ()
        if isinstance(compositions, (str, bytes)) or not isinstance(
            compositions, Sequence
        ):
            raise ValueError("[doping].compositions must be an array of tables")
        for index, composition in enumerate(compositions):
            if not isinstance(composition, Mapping):
                raise ValueError(f"[doping].compositions[{index}] must be a table")
            for element, raw_amount in composition.items():
                try:
                    amount = float(raw_amount)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"[doping].compositions[{index}].{element} must be numeric"
                    ) from exc
                if not math.isfinite(amount) or amount < 0:
                    raise ValueError(
                        f"[doping].compositions[{index}].{element} must be finite "
                        "and >= 0"
                    )
                if amount > 0:
                    values.append(element)
    else:
        raise ValueError("[doping].mode must be 'explicit' or 'enumerate'")
    return tuple(symbol for symbol in _element_symbols(values, setting="[doping]") if symbol != "O")


def parse_correction_config(raw_cfg: Mapping[str, Any], root: Path) -> CorrectionConfig:
    raw = raw_cfg.get("energy_correction", {}) or {}
    enabled = bool(raw.get("enabled", False))
    default_manifest = (
        root / "reference_structures/corrections/calibration_manifest.csv"
    ).resolve()
    if not enabled:
        # A disabled optional section must not make any legacy stage fail because
        # of a stale path or partially edited advanced setting.
        return CorrectionConfig(
            enabled=False,
            experimental_source="kingsbury",
            experimental_data=None,
            dataset_cache_dir=None,
            calibration_manifest=default_manifest,
            correction_terms=("oxide",),
            allow_element_terms=False,
            model_family="manual",
            target_elements=(),
            m1_elements=(),
            calibration_selection="manifest",
            auto_fetch_phase_structures=False,
            optimade_base_url="https://optimade.materialsproject.org/v1",
            min_element_compounds=3,
            min_element_stoichiometries=3,
            min_cv_improvement_eV_per_atom=0.01,
            require_cv_one_standard_error=True,
            exclude_polyanions=DEFAULT_EXCLUDED_POLYANIONS,
            max_relative_experimental_uncertainty=0.10,
            max_calculated_e_above_hull_eV_per_atom=0.10,
            allow_phase_mismatch=False,
            allow_legacy_candidate_provenance=False,
            reuse_fitted=True,
            min_degrees_of_freedom=1,
            min_term_support=2,
            max_condition_number=1.0e8,
            poor_fit_rmse_warning_eV_per_atom=0.20,
        )

    source = str(raw.get("experimental_source", "kingsbury")).strip().lower()
    if source not in {"kingsbury", "kingsbury+custom", "custom"}:
        raise ValueError(
            "[energy_correction].experimental_source must be one of: "
            "kingsbury, kingsbury+custom, custom"
        )

    experimental_data = _resolve_optional_path(root, raw.get("experimental_data"))
    if enabled and source in {"custom", "kingsbury+custom"} and experimental_data is None:
        raise ValueError(
            f"experimental_source={source!r} requires "
            "[energy_correction].experimental_data"
        )

    model_family = str(raw.get("model_family", "manual")).strip().lower()
    if model_family not in {"manual", "m0", "m1", "auto"}:
        raise ValueError(
            "[energy_correction].model_family must be one of: manual, m0, m1, auto"
        )

    terms_raw = raw.get("correction_terms", ["oxide"])
    if not isinstance(terms_raw, (list, tuple)) or not terms_raw:
        raise ValueError("[energy_correction].correction_terms must be a non-empty array")
    terms = tuple(normalize_correction_term(str(term)) for term in terms_raw)
    if len(set(terms)) != len(terms):
        raise ValueError("[energy_correction].correction_terms contains duplicates")
    allow_element_terms = bool(raw.get("allow_element_terms", False))
    explicit_composition_terms = any(
        term.startswith("element:") or term.startswith(_OXIDE_CATION_PREFIX)
        for term in terms
    )
    if enabled and explicit_composition_terms and not allow_element_terms:
        raise ValueError(
            "Explicit element/oxide-cation correction terms require "
            "[energy_correction].allow_element_terms=true after scientific validation."
        )
    if model_family != "manual" and explicit_composition_terms:
        raise ValueError(
            "Automatic M0/M1 model families derive oxide-cation terms from workflow "
            "coverage; remove explicit element terms from correction_terms"
        )
    if model_family != "manual" and tuple(terms) != ("oxide",):
        raise ValueError(
            "M0/M1 model selection currently requires correction_terms=[\"oxide\"]"
        )

    target_elements = infer_correction_target_elements(raw_cfg)
    m1_raw = raw.get("m1_elements", "workflow")
    if isinstance(m1_raw, str) and m1_raw.strip().lower() == "workflow":
        m1_elements = target_elements
    elif isinstance(m1_raw, (list, tuple)):
        m1_elements = _element_symbols(m1_raw, setting="[energy_correction].m1_elements")
    else:
        raise ValueError(
            "[energy_correction].m1_elements must be \"workflow\" or an array of symbols"
        )
    outside_target = sorted(set(m1_elements) - set(target_elements))
    if model_family != "manual" and outside_target:
        raise ValueError(
            "[energy_correction].m1_elements must be drawn from the inferred host/dopant "
            f"scope; outside scope: {outside_target}"
        )
    if model_family in {"m1", "auto"} and not m1_elements:
        raise ValueError(
            "M1 model selection found no target elements; configure [references].host "
            "and [doping] or provide m1_elements"
        )

    calibration_selection = str(
        raw.get("calibration_selection", "manifest")
    ).strip().lower()
    if calibration_selection not in {"manifest", "phase_resolved"}:
        raise ValueError(
            "[energy_correction].calibration_selection must be manifest or phase_resolved"
        )
    auto_fetch_phase_structures = bool(raw.get("auto_fetch_phase_structures", False))
    if auto_fetch_phase_structures and calibration_selection != "phase_resolved":
        raise ValueError(
            "auto_fetch_phase_structures=true requires calibration_selection=\"phase_resolved\""
        )
    optimade_base_url = str(
        raw.get(
            "optimade_base_url",
            "https://optimade.materialsproject.org/v1",
        )
    ).strip().rstrip("/")
    if calibration_selection == "phase_resolved" and not optimade_base_url:
        raise ValueError("[energy_correction].optimade_base_url must not be empty")
    excluded_raw = raw.get("exclude_polyanions", DEFAULT_EXCLUDED_POLYANIONS)
    if not isinstance(excluded_raw, (list, tuple)):
        raise ValueError("[energy_correction].exclude_polyanions must be an array")
    exclude_polyanions = tuple(
        str(item).strip() for item in excluded_raw if str(item).strip()
    )

    max_relative = float(raw.get("max_relative_experimental_uncertainty", 0.10))
    if not math.isfinite(max_relative) or max_relative <= 0:
        raise ValueError(
            "[energy_correction].max_relative_experimental_uncertainty must be > 0"
        )

    unstable_raw = raw.get("max_calculated_e_above_hull_eV_per_atom", 0.10)
    if unstable_raw is None or str(unstable_raw).strip().lower() in {"", "none", "false"}:
        max_unstable = None
    else:
        max_unstable = float(unstable_raw)
        if not math.isfinite(max_unstable) or max_unstable < 0:
            raise ValueError(
                "[energy_correction].max_calculated_e_above_hull_eV_per_atom "
                "must be non-negative or omitted"
            )

    min_dof = int(raw.get("min_degrees_of_freedom", 1))
    min_term_support = int(raw.get("min_term_support", 2))
    min_element_compounds = int(raw.get("min_element_compounds", 3))
    min_element_stoichiometries = int(raw.get("min_element_stoichiometries", 3))
    min_cv_improvement = float(
        raw.get("min_cv_improvement_eV_per_atom", 0.01)
    )
    max_condition = float(raw.get("max_condition_number", 1.0e8))
    rmse_warning = float(raw.get("poor_fit_rmse_warning_eV_per_atom", 0.20))
    if min_dof < 1:
        raise ValueError("[energy_correction].min_degrees_of_freedom must be >= 1")
    if min_term_support < 2:
        raise ValueError("[energy_correction].min_term_support must be >= 2")
    if min_element_compounds < 3:
        raise ValueError("[energy_correction].min_element_compounds must be >= 3")
    if min_element_stoichiometries < 2:
        raise ValueError(
            "[energy_correction].min_element_stoichiometries must be >= 2"
        )
    if not math.isfinite(min_cv_improvement) or min_cv_improvement < 0:
        raise ValueError(
            "[energy_correction].min_cv_improvement_eV_per_atom must be non-negative"
        )
    if not math.isfinite(max_condition) or max_condition <= 1:
        raise ValueError("[energy_correction].max_condition_number must be > 1")
    if not math.isfinite(rmse_warning) or rmse_warning <= 0:
        raise ValueError(
            "[energy_correction].poor_fit_rmse_warning_eV_per_atom must be > 0"
        )

    manifest = _resolve_optional_path(
        root,
        raw.get(
            "calibration_manifest",
            "reference_structures/corrections/calibration_manifest.csv",
        ),
    )
    assert manifest is not None

    return CorrectionConfig(
        enabled=enabled,
        experimental_source=source,
        experimental_data=experimental_data,
        dataset_cache_dir=_resolve_optional_path(root, raw.get("dataset_cache_dir")),
        calibration_manifest=manifest,
        correction_terms=terms,
        allow_element_terms=allow_element_terms,
        model_family=model_family,
        target_elements=target_elements,
        m1_elements=m1_elements,
        calibration_selection=calibration_selection,
        auto_fetch_phase_structures=auto_fetch_phase_structures,
        optimade_base_url=optimade_base_url,
        min_element_compounds=min_element_compounds,
        min_element_stoichiometries=min_element_stoichiometries,
        min_cv_improvement_eV_per_atom=min_cv_improvement,
        require_cv_one_standard_error=bool(
            raw.get("require_cv_one_standard_error", True)
        ),
        exclude_polyanions=exclude_polyanions,
        max_relative_experimental_uncertainty=max_relative,
        max_calculated_e_above_hull_eV_per_atom=max_unstable,
        allow_phase_mismatch=bool(raw.get("allow_phase_mismatch", False)),
        allow_legacy_candidate_provenance=bool(
            raw.get("allow_legacy_candidate_provenance", False)
        ),
        reuse_fitted=bool(raw.get("reuse_fitted", True)),
        min_degrees_of_freedom=min_dof,
        min_term_support=min_term_support,
        max_condition_number=max_condition,
        poor_fit_rmse_warning_eV_per_atom=rmse_warning,
    )


def _reduced_formula(formula: str) -> tuple[str, float]:
    try:
        composition = Composition(str(formula).strip())
    except Exception as exc:
        raise ValueError(f"Invalid formula {formula!r}") from exc
    reduced, factor = composition.get_reduced_composition_and_factor()
    return reduced.reduced_formula, float(factor)


def _standard_temperature(value: Any, *, formula: str) -> str:
    """Validate that a calibration enthalpy represents standard temperature."""
    text = str(value or "").strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(?:k|kelvin)?", text.lower())
    if match is None:
        raise ValueError(
            f"Calibration temperature for {formula!r} must be explicit in kelvin"
        )
    kelvin = float(match.group(1))
    if not 297.0 <= kelvin <= 299.5:
        raise ValueError(
            f"Calibration temperature for {formula!r} is {kelvin:g} K; this model "
            "fits standard 298 K formation enthalpies and does not mix temperatures"
        )
    return "298 K"


def _normalize_experimental_record(
    *,
    formula: str,
    formation_enthalpy: Any,
    uncertainty: Any,
    units: str,
    phase: Any,
    temperature: Any,
    source: Any,
    likely_mpid: Any = "",
    doi: Any = "",
    reference_id: Any = "",
    notes: Any = "",
    dataset: str,
) -> ExperimentalRecord:
    reduced_formula, reduction_factor = _reduced_formula(formula)
    n_atoms = float(Composition(reduced_formula).num_atoms)
    try:
        value = float(formation_enthalpy)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid formation enthalpy for {formula!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite formation enthalpy for {formula!r}")

    uncertainty_value: float | None
    if uncertainty is None or str(uncertainty).strip().lower() in {"", "nan", "none"}:
        uncertainty_value = None
    else:
        uncertainty_value = float(uncertainty)
        if not math.isfinite(uncertainty_value) or uncertainty_value < 0:
            raise ValueError(f"Invalid uncertainty for {formula!r}")
        if uncertainty_value == 0:
            uncertainty_value = None

    unit_key = str(units).strip().lower()
    if unit_key in _EV_PER_ATOM_UNITS:
        value_atom = value
        value_formula = value * n_atoms
        uncertainty_atom = uncertainty_value
        uncertainty_formula = (
            uncertainty_value * n_atoms if uncertainty_value is not None else None
        )
    elif unit_key in _EV_PER_FORMULA_UNITS:
        # ``eV/formula`` refers to the formula string supplied in the row.  Store
        # and fit the equivalent reduced-formula-unit value consistently.
        value_formula = value / reduction_factor
        value_atom = value / n_atoms
        value_atom /= reduction_factor
        uncertainty_formula = (
            uncertainty_value / reduction_factor
            if uncertainty_value is not None
            else None
        )
        uncertainty_atom = (
            uncertainty_formula / n_atoms
            if uncertainty_formula is not None
            else None
        )
    else:
        raise ValueError(
            f"Unsupported or ambiguous units {units!r} for {formula!r}; use eV/atom "
            "or eV/formula_unit"
        )

    def clean_optional_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"nan", "none", "null", "<na>", "nat"} else text

    return ExperimentalRecord(
        formula=str(formula).strip(),
        reduced_formula=reduced_formula,
        formation_enthalpy_eV_per_atom=float(value_atom),
        formation_enthalpy_eV_per_formula=float(value_formula),
        uncertainty_eV_per_atom=(
            float(uncertainty_atom) if uncertainty_atom is not None else None
        ),
        uncertainty_eV_per_formula=(
            float(uncertainty_formula) if uncertainty_formula is not None else None
        ),
        uncertainty_source="reported" if uncertainty_value is not None else "missing",
        phase=clean_optional_text(phase),
        temperature=_standard_temperature(temperature, formula=str(formula)),
        source=clean_optional_text(source),
        likely_mpid=clean_optional_text(likely_mpid),
        doi=clean_optional_text(doi),
        reference_id=clean_optional_text(reference_id),
        notes=clean_optional_text(notes),
        dataset=dataset,
        original_units=str(units),
    )


def load_kingsbury_dataset(data_home: Path | None = None) -> list[ExperimentalRecord]:
    """Load and normalize the curated matminer Kingsbury dataset.

    Matminer owns downloading and its local cache.  A caller-provided
    ``data_home`` is useful for a project-local or shared reproducible cache.
    """
    try:
        from matminer.datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Kingsbury source requires matminer. Install dopingflow with "
            "the 'corrections' extra (for example: pip install 'dopingflow[corrections]')."
        ) from exc

    kwargs: dict[str, Any] = {"pbar": False}
    if data_home is not None:
        kwargs["data_home"] = str(data_home)
    frame = load_dataset(KINGSBURY_DATASET, **kwargs)
    missing = sorted(set(KINGSBURY_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Unexpected {KINGSBURY_DATASET} schema; missing columns: {missing}"
        )

    records: list[ExperimentalRecord] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            _normalize_experimental_record(
                formula=row["formula"],
                formation_enthalpy=row["expt_form_e"],
                uncertainty=row["uncertainty"],
                units="eV/atom",
                phase=row["phaseinfo"],
                temperature="298 K",
                source=row["reference"],
                likely_mpid=row["likely_mpid"],
                dataset=KINGSBURY_DATASET,
            )
        )
    return records


def load_custom_experimental_dataset(path: Path) -> list[ExperimentalRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Custom experimental dataset not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(_CUSTOM_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"Custom experimental dataset {path} is missing required columns: {missing}"
            )
        records = [
            _normalize_experimental_record(
                formula=row["formula"],
                formation_enthalpy=row["formation_enthalpy"],
                uncertainty=row["uncertainty"],
                units=row["units"],
                phase=row["phase"],
                temperature=row["temperature"],
                source=row["source"],
                likely_mpid=row.get("likely_mpid", ""),
                doi=row.get("doi", ""),
                reference_id=row.get("reference_id", ""),
                notes=row.get("notes", ""),
                dataset=f"custom:{path.name}",
            )
            for row in reader
        ]
    if not records:
        raise ValueError(f"Custom experimental dataset is empty: {path}")
    return records


def _phase_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def merge_experimental_datasets(
    curated: Sequence[ExperimentalRecord],
    custom: Sequence[ExperimentalRecord],
) -> list[ExperimentalRecord]:
    """Merge custom records, overriding only an unambiguous phase/ID match."""
    result = list(curated)
    for replacement in custom:
        candidates: list[int] = []
        if replacement.phase:
            candidates = [
                index
                for index, record in enumerate(result)
                if record.reduced_formula == replacement.reduced_formula
                and record.temperature == replacement.temperature
                and _phase_key(record.phase) == _phase_key(replacement.phase)
            ]
            if replacement.likely_mpid:
                candidates = [
                    index
                    for index in candidates
                    if result[index].likely_mpid == replacement.likely_mpid
                ]
        if len(candidates) > 1:
            raise ValueError(
                "Ambiguous custom override for "
                f"{replacement.formula!r}; specify a structure identifier or "
                "remove duplicates"
            )
        if candidates:
            result[candidates[0]] = replacement
        else:
            result.append(replacement)
    return result


def load_experimental_dataset(config: CorrectionConfig) -> list[ExperimentalRecord]:
    curated: list[ExperimentalRecord] = []
    custom: list[ExperimentalRecord] = []
    if config.experimental_source in {"kingsbury", "kingsbury+custom"}:
        curated = load_kingsbury_dataset(config.dataset_cache_dir)
    if config.experimental_source in {"custom", "kingsbury+custom"}:
        assert config.experimental_data is not None
        custom = load_custom_experimental_dataset(config.experimental_data)
    if config.experimental_source == "kingsbury+custom":
        return merge_experimental_datasets(curated, custom)
    return custom if config.experimental_source == "custom" else curated


def impute_missing_uncertainties(
    records: Sequence[ExperimentalRecord],
) -> list[ExperimentalRecord]:
    available = [
        record.uncertainty_eV_per_atom
        for record in records
        if record.uncertainty_eV_per_atom is not None
        and record.uncertainty_eV_per_atom > 0
    ]
    if not available:
        raise ValueError(
            "None of the accepted calibration records has an experimental uncertainty; "
            "uncertainty-weighted fitting is not possible"
        )
    mean_per_atom = float(np.mean(available))
    output: list[ExperimentalRecord] = []
    for record in records:
        if record.uncertainty_eV_per_atom is not None:
            output.append(record)
            continue
        n_atoms = float(Composition(record.reduced_formula).num_atoms)
        data = record.to_dict()
        data["uncertainty_eV_per_atom"] = mean_per_atom
        data["uncertainty_eV_per_formula"] = mean_per_atom * n_atoms
        data["uncertainty_source"] = "imputed_mean"
        output.append(ExperimentalRecord(**data))
    return output


def backend_signature_from_reference(
    ref: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    backend = str(ref.get("backend", ""))
    package_name = {
        "mace": "mace-torch",
        "uma": "fairchem-core",
        "m3gnet": "m3gnet",
        "grace": "tensorpotential",
    }.get(backend.lower())
    if package_name:
        try:
            package_version = version(package_name)
        except PackageNotFoundError:
            package_version = "not-installed"
    else:
        package_version = "unknown"

    signature = {
        "backend": backend,
        "model": str(ref.get("model", "")),
        "task": str(ref.get("task", "")),
        "optimizer": str(ref.get("optimizer", "")),
        "fmax_eV_per_A": float(ref.get("fmax", 0.0)),
        "max_steps": int(ref.get("max_steps", 0)),
        "device": str(ref.get("device", "")),
        "gpu_id": ref.get("gpu_id"),
        "backend_package": package_name or "unknown",
        "backend_package_version": str(
            ref.get("backend_package_version") or package_version
        ),
        "reference_energy_set_hash": hashlib.sha256(
            _canonical_json(
                {
                    "host": ref.get("host", {}),
                    "references": ref.get("references", {}),
                    "reference_mode": ref.get("reference_mode"),
                }
            ).encode("utf-8")
        ).hexdigest(),
    }
    stored_checkpoint_hash = str(ref.get("model_checkpoint_sha256") or "").strip()
    if stored_checkpoint_hash:
        signature["model_checkpoint_sha256"] = stored_checkpoint_hash
    model_path_value = str(ref.get("model", "")).strip()
    if model_path_value and not stored_checkpoint_hash:
        checkpoint = Path(model_path_value).expanduser()
        if not checkpoint.is_absolute() and root is not None:
            checkpoint = root / checkpoint
        if checkpoint.is_file():
            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            signature["model_checkpoint_sha256"] = digest.hexdigest()
    return signature


def backend_runtime_provenance(
    backend: str,
    model: str,
    *,
    root: Path | None = None,
) -> dict[str, str]:
    package_name = {
        "mace": "mace-torch",
        "uma": "fairchem-core",
        "m3gnet": "m3gnet",
        "grace": "tensorpotential",
    }.get(str(backend).lower(), "unknown")
    try:
        package_version = (
            version(package_name) if package_name != "unknown" else "unknown"
        )
    except PackageNotFoundError:
        package_version = "not-installed"
    result = {
        "backend_package": package_name,
        "backend_package_version": package_version,
    }
    checkpoint = Path(str(model)).expanduser()
    if not checkpoint.is_absolute() and root is not None:
        checkpoint = root / checkpoint
    if checkpoint.is_file():
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["model_checkpoint_sha256"] = digest.hexdigest()
    return result


def backend_slug(signature: Mapping[str, Any]) -> str:
    parts = [
        str(signature.get("backend", "unknown")),
        str(signature.get("model", "default")),
        str(signature.get("task", "")),
    ]
    text = "-".join(part for part in parts if part)
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "unknown"
    digest = hashlib.sha256(_canonical_json(dict(signature)).encode("utf-8")).hexdigest()
    return f"{base[:80]}-{digest[:10]}"


def model_directory(root: Path, signature: Mapping[str, Any]) -> Path:
    return root / CORRECTIONS_DIR / backend_slug(signature)


def model_path(root: Path, signature: Mapping[str, Any]) -> Path:
    return model_directory(root, signature) / MODEL_FILENAME


def correction_activation_hash(config: CorrectionConfig, root: Path) -> str:
    """Fingerprint lightweight calibration inputs that require a refit."""

    def file_hash(path: Path | None) -> str | None:
        if path is None:
            return None
        if not path.exists():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest = config.calibration_manifest
    manifest_structures: list[dict[str, str | None]] = []
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                value = str(
                    row.get("structure_path") or row.get("structure") or ""
                ).strip()
                if not value:
                    continue
                path = Path(value).expanduser()
                if not path.is_absolute():
                    local = manifest.parent / path
                    project = root / path
                    path = local if local.exists() or not project.exists() else project
                manifest_structures.append(
                    {
                        "path": str(path.resolve()),
                        "sha256": file_hash(path),
                    }
                )

    payload: dict[str, Any] = {
            "schema": 2,
            "framework_version": CORRECTION_FRAMEWORK_VERSION,
            "experimental_source": config.experimental_source,
            "experimental_data": str(config.experimental_data or ""),
            "experimental_data_sha256": file_hash(config.experimental_data),
            "kingsbury_sha256": (
                KINGSBURY_DATASET_SHA256
                if config.experimental_source in {"kingsbury", "kingsbury+custom"}
                else None
            ),
            "calibration_manifest": str(manifest),
            "calibration_manifest_sha256": file_hash(manifest),
            "manifest_structures": manifest_structures,
            "correction_terms": config.correction_terms,
            "allow_element_terms": config.allow_element_terms,
            "exclude_polyanions": config.exclude_polyanions,
            "max_relative_experimental_uncertainty": (
                config.max_relative_experimental_uncertainty
            ),
            "max_calculated_e_above_hull_eV_per_atom": (
                config.max_calculated_e_above_hull_eV_per_atom
            ),
            "allow_phase_mismatch": config.allow_phase_mismatch,
            "min_degrees_of_freedom": config.min_degrees_of_freedom,
            "min_term_support": config.min_term_support,
            "max_condition_number": config.max_condition_number,
            "poor_fit_rmse_warning_eV_per_atom": (
                config.poor_fit_rmse_warning_eV_per_atom
            ),
        }
    if config.model_family != "manual" or config.calibration_selection != "manifest":
        payload.update(
            {
                "schema": 3,
                "model_family": config.model_family,
                "model_selection_policy_version": MODEL_SELECTION_POLICY_VERSION,
                "target_elements": config.target_elements,
                "m1_elements": config.m1_elements,
                "calibration_selection": config.calibration_selection,
                "auto_fetch_phase_structures": config.auto_fetch_phase_structures,
                "optimade_base_url": config.optimade_base_url,
                "min_element_compounds": config.min_element_compounds,
                "min_element_stoichiometries": config.min_element_stoichiometries,
                "min_cv_improvement_eV_per_atom": (
                    config.min_cv_improvement_eV_per_atom
                ),
                "require_cv_one_standard_error": (
                    config.require_cv_one_standard_error
                ),
            }
        )
    return content_hash(payload)


def _oxide_type_from_structure(structure: Structure | None) -> str | None:
    if structure is None or "O" not in structure.composition:
        return None
    value = str(
        classify_oxide_type(
            structure,
            relative_cutoff=OXIDE_CLASSIFICATION_RELATIVE_CUTOFF,
        )
    ).strip().lower()
    if value in {"normal", "oxide"}:
        return "oxide"
    return value


def feature_vector(
    composition: Composition | str,
    correction_terms: Sequence[str],
    *,
    structure: Structure | None = None,
    known_oxide_type: str | None = None,
) -> tuple[tuple[float, ...], tuple[str, ...], str | None]:
    comp = composition if isinstance(composition, Composition) else Composition(composition)
    if (
        structure is not None
        and structure.composition.reduced_composition != comp.reduced_composition
    ):
        raise ValueError(
            f"Correction composition {comp.reduced_formula} does not match the "
            f"supplied structure {structure.composition.reduced_formula}"
        )
    if len(comp.elements) == 1:
        return tuple(0.0 for _ in correction_terms), tuple(), None

    declared_oxygen_kind = str(known_oxide_type or "").strip().lower() or None
    detected_oxygen_kind = _oxide_type_from_structure(structure)
    if declared_oxygen_kind == "normal":
        declared_oxygen_kind = "oxide"
    if detected_oxygen_kind == "normal":
        detected_oxygen_kind = "oxide"
    if (
        declared_oxygen_kind is not None
        and detected_oxygen_kind is not None
        and declared_oxygen_kind != detected_oxygen_kind
    ):
        raise ValueError(
            f"Declared oxide_type {declared_oxygen_kind!r} disagrees with the "
            f"structure-derived type {detected_oxygen_kind!r} for {comp.reduced_formula}"
        )
    oxygen_kind = declared_oxygen_kind or detected_oxygen_kind
    if oxygen_kind == "normal":
        oxygen_kind = "oxide"

    oxygen_terms_requested = bool(set(correction_terms) & _OXYGEN_TERMS) or any(
        term.startswith(_OXIDE_CATION_PREFIX) for term in correction_terms
    )
    if "O" in comp and oxygen_terms_requested:
        if oxygen_kind is None:
            raise ValueError(
                f"Cannot classify the oxygen environment for {comp.reduced_formula}; "
                "a structure or explicit oxide_type is required"
            )
        if oxygen_kind not in _OXYGEN_TERMS:
            raise ValueError(
                f"Unsupported oxygen environment {oxygen_kind!r} for "
                f"{comp.reduced_formula}"
            )
        if oxygen_kind not in correction_terms:
            raise ValueError(
                f"The fitted model has no {oxygen_kind!r} term required by "
                f"{comp.reduced_formula}"
            )

    values: list[float] = []
    matched: list[str] = []
    for term in correction_terms:
        if term in _OXYGEN_TERMS:
            value = float(comp["O"]) if oxygen_kind == term else 0.0
        elif term.startswith(_OXIDE_CATION_PREFIX):
            symbol = term.split(":", 1)[1]
            value = (
                float(comp.get(symbol, 0.0))
                if oxygen_kind == "oxide" and "O" in comp
                else 0.0
            )
        else:
            symbol = term.split(":", 1)[1]
            value = float(comp.get(symbol, 0.0))
        values.append(value)
        if value:
            matched.append(term)
    return tuple(values), tuple(matched), oxygen_kind


def combine_feature_vectors(
    terms: Sequence[str],
    components: Iterable[tuple[float, Sequence[float]]],
) -> tuple[float, ...]:
    total = np.zeros(len(terms), dtype=float)
    for scale, vector in components:
        array = np.asarray(vector, dtype=float)
        if array.shape != total.shape:
            raise ValueError("Correction feature vector has the wrong dimension")
        total += float(scale) * array
    return tuple(float(x) for x in total)


def evaluate_feature_vector(
    model: CorrectionModel,
    vector: Sequence[float],
    *,
    matched_terms: Sequence[str] | None = None,
    reason: str = "model_applicable",
    oxygen_kind: str | None = None,
) -> CorrectionApplication:
    _validate_model_arrays(model)
    features = np.asarray(vector, dtype=float)
    coefficients = np.asarray(model.coefficients_eV_per_term, dtype=float)
    covariance = np.asarray(model.covariance_eV2, dtype=float)
    if features.shape != coefficients.shape:
        raise ValueError("Correction feature vector does not match the fitted model")
    if np.any(~np.isfinite(features)):
        raise ValueError("Correction feature vector contains non-finite values")
    variance = float(features @ covariance @ features)
    if variance < 0 and abs(variance) < 1.0e-12:
        variance = 0.0
    if variance < 0:
        raise ValueError("Correction covariance produced a negative variance")
    is_nonzero = bool(np.any(np.abs(features) > 0.0))
    return CorrectionApplication(
        correction_eV=float(features @ coefficients),
        uncertainty_eV=math.sqrt(variance),
        feature_vector=tuple(float(x) for x in features),
        matched_terms=tuple(matched_terms or ()),
        applied=is_nonzero,
        reason=reason if is_nonzero else "model_evaluated_no_applicable_terms",
        oxide_type=oxygen_kind,
    )


def apply_energy_correction(
    model: CorrectionModel,
    composition: Composition | str,
    *,
    structure: Structure | None = None,
    structure_path: Path | None = None,
    known_oxide_type: str | None = None,
) -> CorrectionApplication:
    comp = composition if isinstance(composition, Composition) else Composition(composition)
    if len(comp.elements) == 1:
        return CorrectionApplication(
            correction_eV=0.0,
            uncertainty_eV=0.0,
            feature_vector=tuple(0.0 for _ in model.correction_terms),
            matched_terms=tuple(),
            applied=False,
            reason="elemental_reference_not_corrected",
            oxide_type=None,
        )
    applicability = dict(model.applicability_signature or {})
    excluded_polyanions = tuple(applicability.get("exclude_polyanions", ()))
    if any(token in comp.reduced_formula for token in excluded_polyanions):
        raise ValueError(
            f"{comp.reduced_formula} contains a polyanion excluded by correction fit "
            f"{model.fit_id}"
        )
    if structure is None and structure_path is not None:
        if not structure_path.exists():
            raise FileNotFoundError(f"Correction structure not found: {structure_path}")
        structure = Structure.from_file(str(structure_path))
    vector, matched, oxygen_kind = feature_vector(
        comp,
        model.correction_terms,
        structure=structure,
        known_oxide_type=known_oxide_type,
    )
    return evaluate_feature_vector(
        model,
        vector,
        matched_terms=matched,
        oxygen_kind=oxygen_kind,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def fit_linear_correction_model(
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    correction_terms: Sequence[str],
    backend_signature: Mapping[str, Any],
    experimental_dataset: str,
    experimental_dataset_version: str,
    fit_input_hash: str,
    min_degrees_of_freedom: int = 1,
    min_term_support: int = 2,
    max_condition_number: float = 1.0e8,
    activation_input_hash: str = "",
    exclude_polyanions: Sequence[str] = DEFAULT_EXCLUDED_POLYANIONS,
) -> tuple[CorrectionModel, dict[str, Any]]:
    """Fit the paper's simultaneous uncertainty-weighted linear model.

    Each row must contain a formula-unit feature vector, experimental and
    calculated formation energies in eV/formula unit, and a strictly positive
    experimental uncertainty in the same unit.
    """
    terms = tuple(normalize_correction_term(term) for term in correction_terms)
    n_rows = len(calibration_rows)
    n_terms = len(terms)
    if n_rows < n_terms + min_degrees_of_freedom:
        raise ValueError(
            f"Correction fit needs at least {n_terms + min_degrees_of_freedom} "
            f"accepted compounds for {n_terms} terms; got {n_rows}"
        )

    matrix = np.asarray([row["feature_vector"] for row in calibration_rows], dtype=float)
    observed = np.asarray(
        [
            float(row["experimental_formation_eV_per_formula"])
            - float(row["calculated_formation_eV_per_formula"])
            for row in calibration_rows
        ],
        dtype=float,
    )
    sigma = np.asarray(
        [float(row["experimental_uncertainty_eV_per_formula"]) for row in calibration_rows],
        dtype=float,
    )
    if matrix.shape != (n_rows, n_terms):
        raise ValueError("Calibration feature matrix has an inconsistent shape")
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(observed)):
        raise ValueError("Calibration fit contains non-finite values")
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Experimental fit uncertainties must be finite and > 0")

    support_counts = np.count_nonzero(np.abs(matrix) > 0.0, axis=0)
    insufficient_support = {
        term: int(count)
        for term, count in zip(terms, support_counts, strict=True)
        if int(count) < min_term_support
    }
    if insufficient_support:
        raise ValueError(
            "Each fitted correction term needs replicated calibration support "
            f"(minimum {min_term_support}); insufficient: {insufficient_support}"
        )

    weighted_matrix = matrix / sigma[:, None]
    weighted_observed = observed / sigma
    rank = int(np.linalg.matrix_rank(weighted_matrix))
    if rank != n_terms:
        raise ValueError(
            f"Correction model is not identifiable: weighted design rank {rank} "
            f"for {n_terms} fitted terms"
        )
    condition_number = float(np.linalg.cond(weighted_matrix))
    if not math.isfinite(condition_number) or condition_number > max_condition_number:
        raise ValueError(
            "Correction design matrix is ill-conditioned "
            f"(condition number {condition_number:.6g}, limit {max_condition_number:.6g})"
        )

    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_matrix,
        weighted_observed,
        rcond=None,
    )
    _, singular_values, right_vectors = np.linalg.svd(
        weighted_matrix,
        full_matrices=False,
    )
    covariance = (right_vectors.T * (1.0 / np.square(singular_values))) @ right_vectors
    covariance = (covariance + covariance.T) / 2.0
    predicted = matrix @ coefficients
    residual = observed - predicted

    per_atom_residual = np.asarray(
        [
            value / float(Composition(str(row["formula"])).num_atoms)
            for value, row in zip(residual, calibration_rows, strict=True)
        ]
    )
    weights = 1.0 / np.square(sigma)
    leave_one_out_residuals_per_atom: list[float] = []
    for held_out in range(n_rows):
        keep = np.arange(n_rows) != held_out
        if int(np.sum(keep)) < n_terms:
            continue
        train_matrix = weighted_matrix[keep]
        if int(np.linalg.matrix_rank(train_matrix)) != n_terms:
            continue
        train_coefficients, _, _, _ = np.linalg.lstsq(
            train_matrix,
            weighted_observed[keep],
            rcond=None,
        )
        held_residual = observed[held_out] - float(matrix[held_out] @ train_coefficients)
        held_atoms = float(Composition(str(calibration_rows[held_out]["formula"])).num_atoms)
        leave_one_out_residuals_per_atom.append(held_residual / held_atoms)

    metrics = {
        "rmse_eV_per_atom": float(np.sqrt(np.mean(np.square(per_atom_residual)))),
        "mae_eV_per_atom": float(np.mean(np.abs(per_atom_residual))),
        "max_abs_residual_eV_per_atom": float(np.max(np.abs(per_atom_residual))),
        "weighted_rmse_eV_per_formula": float(
            np.sqrt(np.sum(weights * np.square(residual)) / np.sum(weights))
        ),
        "condition_number": condition_number,
        "rank": float(rank),
        "degrees_of_freedom": float(n_rows - n_terms),
    }
    if leave_one_out_residuals_per_atom:
        metrics["leave_one_out_rmse_eV_per_atom"] = float(
            np.sqrt(np.mean(np.square(leave_one_out_residuals_per_atom)))
        )
        metrics["leave_one_out_mae_eV_per_atom"] = float(
            np.mean(np.abs(leave_one_out_residuals_per_atom))
        )

    leave_element_out: dict[str, dict[str, float]] = {}
    all_elements = sorted(
        {
            element.symbol
            for row in calibration_rows
            for element in Composition(str(row["formula"])).elements
            if element.symbol != "O"
        }
    )
    for symbol in all_elements:
        held_indices = np.asarray(
            [
                symbol in {
                    element.symbol
                    for element in Composition(str(row["formula"])).elements
                }
                for row in calibration_rows
            ],
            dtype=bool,
        )
        keep = ~held_indices
        if not np.any(held_indices) or int(np.sum(keep)) < n_terms:
            continue
        if int(np.linalg.matrix_rank(weighted_matrix[keep])) != n_terms:
            continue
        family_coefficients, _, _, _ = np.linalg.lstsq(
            weighted_matrix[keep],
            weighted_observed[keep],
            rcond=None,
        )
        family_residuals: list[float] = []
        for index in np.flatnonzero(held_indices):
            family_residual = observed[index] - float(
                matrix[index] @ family_coefficients
            )
            family_residuals.append(
                family_residual
                / float(Composition(str(calibration_rows[index]["formula"])).num_atoms)
            )
        leave_element_out[symbol] = {
            "count": float(len(family_residuals)),
            "rmse_eV_per_atom": float(
                np.sqrt(np.mean(np.square(family_residuals)))
            ),
            "mae_eV_per_atom": float(np.mean(np.abs(family_residuals))),
        }
    coefficient_uncertainties = np.sqrt(np.diag(covariance))
    fit_id = content_hash(
        {
            "backend_signature": dict(backend_signature),
            "correction_terms": terms,
            "fit_input_hash": fit_input_hash,
            "coefficients": coefficients.tolist(),
        }
    )[:16]
    model = CorrectionModel(
        schema_version=1,
        method="kingsbury_weighted_linear_composition_correction",
        fit_id=fit_id,
        backend_signature=dict(backend_signature),
        correction_terms=terms,
        coefficients_eV_per_term=tuple(float(x) for x in coefficients),
        covariance_eV2=tuple(tuple(float(x) for x in row) for row in covariance),
        coefficient_uncertainties_eV_per_term=tuple(
            float(x) for x in coefficient_uncertainties
        ),
        experimental_dataset=experimental_dataset,
        experimental_dataset_version=experimental_dataset_version,
        fit_input_hash=fit_input_hash,
        units={
            "coefficient": "eV_per_term_atom",
            "covariance": "eV^2",
            "experimental_input": "eV_per_formula_unit",
            "reported_fit_residual": "eV_per_atom",
        },
        calibration_formulas=tuple(str(row["formula"]) for row in calibration_rows),
        fit_metrics=metrics,
        activation_input_hash=activation_input_hash,
        applicability_signature={
            "schema_version": 1,
            "framework_version": CORRECTION_FRAMEWORK_VERSION,
            "feature_model": "pymatgen.analysis.structure_analyzer.oxide_type",
            "oxide_classification_relative_cutoff": (
                OXIDE_CLASSIFICATION_RELATIVE_CUTOFF
            ),
            "pymatgen_version": version("pymatgen"),
            "exclude_polyanions": list(exclude_polyanions),
            "term_scopes": {
                term: (
                    "ordinary_oxide_cation_only"
                    if term.startswith(_OXIDE_CATION_PREFIX)
                    else (
                        "all_non_elemental_compounds"
                        if term.startswith("element:")
                        else f"{term}_oxygen_environment_only"
                    )
                )
                for term in terms
            },
        },
    )

    report_rows: list[dict[str, Any]] = []
    for row, observed_value, predicted_value, residual_value in zip(
        calibration_rows,
        observed,
        predicted,
        residual,
        strict=True,
    ):
        item = dict(row)
        item.update(
            {
                "observed_correction_eV_per_formula": float(observed_value),
                "predicted_correction_eV_per_formula": float(predicted_value),
                "residual_eV_per_formula": float(residual_value),
                "residual_eV_per_atom": float(
                    residual_value / Composition(str(row["formula"])).num_atoms
                ),
            }
        )
        report_rows.append(item)
    report = {
        "fit_id": fit_id,
        "method": model.method,
        "correction_terms": list(terms),
        "coefficients_eV_per_term": dict(zip(terms, model.coefficients_eV_per_term)),
        "coefficient_uncertainties_eV_per_term": dict(
            zip(terms, model.coefficient_uncertainties_eV_per_term)
        ),
        "covariance_eV2": [list(row) for row in model.covariance_eV2],
        "metrics": metrics,
        "accepted_count": n_rows,
        "accepted_records": report_rows,
        "validation": {
            "leave_one_out_count": len(leave_one_out_residuals_per_atom),
            "leave_element_family_out": leave_element_out,
            "note": (
                "Residual and cross-validation errors are validation diagnostics, "
                "not correction-parameter uncertainty and are not combined with it."
            ),
        },
    }
    return model, report


def save_correction_model(model: CorrectionModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_correction_model(path: Path) -> CorrectionModel:
    if not path.exists():
        raise FileNotFoundError(f"Fitted correction model not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid correction model JSON: {path}") from exc
    model = CorrectionModel.from_dict(data)
    if model.schema_version != 1:
        raise ValueError(
            f"Unsupported correction model schema version {model.schema_version} in {path}"
        )
    _validate_model_arrays(model)
    return model


def _validate_model_arrays(model: CorrectionModel) -> None:
    n_terms = len(model.correction_terms)
    normalized_terms = tuple(
        normalize_correction_term(term) for term in model.correction_terms
    )
    if normalized_terms != model.correction_terms or len(set(normalized_terms)) != n_terms:
        raise ValueError("Correction model terms are invalid or duplicated")
    coefficients = np.asarray(model.coefficients_eV_per_term, dtype=float)
    covariance = np.asarray(model.covariance_eV2, dtype=float)
    uncertainties = np.asarray(
        model.coefficient_uncertainties_eV_per_term,
        dtype=float,
    )
    if coefficients.shape != (n_terms,):
        raise ValueError("Correction model coefficient dimension is inconsistent")
    if covariance.shape != (n_terms, n_terms):
        raise ValueError("Correction model covariance dimension is inconsistent")
    if uncertainties.shape != (n_terms,):
        raise ValueError("Correction model uncertainty dimension is inconsistent")
    if not (
        np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(covariance))
        and np.all(np.isfinite(uncertainties))
    ):
        raise ValueError("Correction model contains non-finite parameters")
    if np.any(uncertainties < 0):
        raise ValueError("Correction model contains a negative parameter uncertainty")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Correction model covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(np.min(eigenvalues, initial=0.0)) < -1.0e-10:
        raise ValueError("Correction model covariance is not positive semidefinite")
    expected_uncertainties = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if not np.allclose(
        uncertainties,
        expected_uncertainties,
        rtol=1.0e-8,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Correction model parameter uncertainties do not match its covariance"
        )


def validate_applicability_compatibility(model: CorrectionModel) -> None:
    expected = {
        "schema_version": 1,
        "framework_version": CORRECTION_FRAMEWORK_VERSION,
        "feature_model": "pymatgen.analysis.structure_analyzer.oxide_type",
        "oxide_classification_relative_cutoff": OXIDE_CLASSIFICATION_RELATIVE_CUTOFF,
        "pymatgen_version": version("pymatgen"),
    }
    actual = dict(model.applicability_signature or {})
    differences = {
        key: {"fitted": actual.get(key), "current": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if differences:
        raise ValueError(
            "Correction applicability/classifier mismatch; refit with the current "
            f"implementation. Differences: {differences}"
        )


def _validate_active_model_semantics(model: CorrectionModel) -> None:
    if model.method != "kingsbury_weighted_linear_composition_correction":
        raise ValueError(
            f"Unsupported active correction method {model.method!r}; rerun corrections-fit"
        )
    expected_units = {
        "coefficient": "eV_per_term_atom",
        "covariance": "eV^2",
        "experimental_input": "eV_per_formula_unit",
        "reported_fit_residual": "eV_per_atom",
    }
    differences = {
        key: {"stored": model.units.get(key), "expected": value}
        for key, value in expected_units.items()
        if model.units.get(key) != value
    }
    if differences:
        raise ValueError(
            "Correction model unit convention is incompatible; rerun "
            f"corrections-fit. Differences: {differences}"
        )
    if model.model_family not in {"manual", "m0", "m1"}:
        raise ValueError(
            f"Unsupported correction model family {model.model_family!r}; "
            "rerun corrections-fit"
        )
    if model.model_family == "m0" and any(
        term.startswith(_OXIDE_CATION_PREFIX) for term in model.correction_terms
    ):
        raise ValueError("M0 model unexpectedly contains an oxide-cation term")
    if model.model_family == "m1" and not any(
        term.startswith(_OXIDE_CATION_PREFIX) for term in model.correction_terms
    ):
        raise ValueError("M1 model contains no validated oxide-cation term")


def validate_backend_compatibility(
    model: CorrectionModel,
    current_signature: Mapping[str, Any],
) -> None:
    expected = dict(model.backend_signature)
    actual = dict(current_signature)
    if expected != actual:
        differences = {
            key: {"fitted": expected.get(key), "current": actual.get(key)}
            for key in sorted(set(expected) | set(actual))
            if expected.get(key) != actual.get(key)
        }
        raise ValueError(
            "Correction model backend/settings mismatch; refit with the current "
            f"backend. Differences: {differences}"
        )


def validate_candidate_energy_provenance(
    metadata: Mapping[str, Any],
    structure_path: Path,
    model: CorrectionModel,
    *,
    label: str,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Validate a relaxed candidate without requiring a new physical relaxation.

    Legacy adoption is deliberately explicit. It accepts only missing provenance
    fields; a present-but-conflicting value or a changed stored structure hash is
    always rejected.
    """
    if metadata.get("converged") is not True:
        raise ValueError(f"Candidate relaxation is not positively converged at {label}")
    if not structure_path.is_file():
        raise FileNotFoundError(f"Candidate relaxed structure not found: {structure_path}")

    digest = hashlib.sha256()
    with structure_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    stored_hash = str(metadata.get("relaxed_poscar_sha256") or "").strip()
    assumptions: list[str] = []
    if stored_hash:
        if stored_hash != actual_hash:
            raise ValueError(
                f"Candidate structure changed after energy evaluation at "
                f"{structure_path}"
            )
    elif allow_legacy:
        assumptions.append("missing_original_relaxed_poscar_sha256")
    else:
        raise ValueError(
            f"Candidate relaxation lacks hashed POSCAR provenance at {label}; "
            "rerun relaxation or explicitly enable legacy candidate provenance"
        )

    expected = dict(model.backend_signature)
    required_mapping = {
        "backend": "backend",
        "model": "model",
        "task": "task",
        "optimizer": "optimizer",
        "fmax_eV_per_A": "fmax_target_eV_per_A",
        "max_steps": "max_steps",
    }
    differences = {
        fitted_key: {
            "fitted": expected.get(fitted_key),
            "candidate": metadata.get(candidate_key),
        }
        for fitted_key, candidate_key in required_mapping.items()
        if expected.get(fitted_key) != metadata.get(candidate_key)
    }
    if differences:
        raise ValueError(
            f"Candidate energy backend incompatible with correction fit at {label}: "
            f"{differences}"
        )

    optional_mapping = {
        "backend_package": "backend_package",
        "backend_package_version": "backend_package_version",
    }
    if "model_checkpoint_sha256" in expected:
        optional_mapping["model_checkpoint_sha256"] = "model_checkpoint_sha256"
    for fitted_key, candidate_key in optional_mapping.items():
        candidate_value = metadata.get(candidate_key)
        if candidate_value in (None, ""):
            if allow_legacy:
                assumptions.append(f"missing_original_{candidate_key}")
                continue
            differences[fitted_key] = {
                "fitted": expected.get(fitted_key),
                "candidate": candidate_value,
            }
        elif expected.get(fitted_key) != candidate_value:
            differences[fitted_key] = {
                "fitted": expected.get(fitted_key),
                "candidate": candidate_value,
            }
    if differences:
        raise ValueError(
            f"Candidate energy backend incompatible with correction fit at {label}: "
            f"{differences}"
        )

    execution_differences: list[str] = []
    for key in ("device", "gpu_id"):
        if expected.get(key) != metadata.get(key):
            execution_differences.append(
                f"execution_{key}_differs:{metadata.get(key)!r}->{expected.get(key)!r}"
            )

    mode = "legacy_explicitly_accepted" if assumptions else "strict"
    return {
        "mode": mode,
        "assumptions": assumptions,
        "execution_differences": execution_differences,
        "current_relaxed_poscar_sha256": actual_hash,
        "original_relaxed_poscar_sha256": stored_hash or None,
        "note": (
            "Existing energy and current POSCAR were explicitly adopted; their "
            "original byte-level binding/package version cannot be reconstructed."
            if assumptions
            else "Original calculation provenance validated."
        ),
    }


def validate_reference_energy_provenance(reference_data: Mapping[str, Any]) -> None:
    """Reject unconverged reference energies before any corrected result is used."""
    def require_structure_hash(
        record: Mapping[str, Any],
        *,
        path_key: str,
        hash_key: str,
        label: str,
    ) -> None:
        path_text = str(record.get(path_key) or "").strip()
        expected_hash = str(record.get(hash_key) or "").strip()
        if not path_text or not expected_hash:
            raise ValueError(
                f"Correction application requires hashed relaxed structure "
                f"provenance for {label}; rerun refs-build"
            )
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Relaxed reference structure not found: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(
                f"Relaxed reference structure changed for {label}; rerun refs-build"
            )

    def require_relaxation_signature(
        record: Mapping[str, Any],
        *,
        label: str,
    ) -> Mapping[str, Any]:
        signature = record.get("relaxation_signature")
        if not isinstance(signature, Mapping):
            raise ValueError(
                f"Correction application requires a relaxation signature for "
                f"{label}; rerun refs-build"
            )
        return signature

    host = reference_data.get("host")
    if not isinstance(host, Mapping):
        raise ValueError(
            "Correction application requires host convergence provenance; rerun refs-build"
        )
    for key in ("unit_converged", "supercell_converged"):
        if host.get(key) is not True:
            raise ValueError(
                f"Correction application requires host.{key}=true; rerun refs-build"
            )
    host_signature = require_relaxation_signature(host, label="host")
    if host_signature.get("schema_version") != 2:
        raise ValueError(
            "Correction application requires the current reference relaxation "
            "signature schema; rerun refs-build"
        )
    signature_to_reference_key = {
        "backend": "backend",
        "model": "model",
        "task": "task",
        "backend_package": "backend_package",
        "backend_package_version": "backend_package_version",
        "optimizer": "optimizer",
        "fmax": "fmax",
        "max_steps": "max_steps",
        "device": "device",
        "gpu_id": "gpu_id",
    }
    signature_differences = {
        signature_key: {
            "reference_set": reference_data.get(reference_key),
            "host": host_signature.get(signature_key),
        }
        for signature_key, reference_key in signature_to_reference_key.items()
        if reference_data.get(reference_key) != host_signature.get(signature_key)
    }
    top_checkpoint = str(
        reference_data.get("model_checkpoint_sha256") or ""
    ).strip()
    host_checkpoint = str(
        host_signature.get("model_checkpoint_sha256") or ""
    ).strip()
    if top_checkpoint != host_checkpoint:
        signature_differences["model_checkpoint_sha256"] = {
            "reference_set": top_checkpoint or None,
            "host": host_checkpoint or None,
        }
    if signature_differences:
        raise ValueError(
            "Host relaxation signature is inconsistent with the reference-energy "
            f"backend/settings: {signature_differences}; rerun refs-build"
        )
    require_structure_hash(
        host,
        path_key="relaxed_unit_poscar",
        hash_key="relaxed_unit_sha256",
        label="host unit cell",
    )
    require_structure_hash(
        host,
        path_key="relaxed_supercell_poscar",
        hash_key="relaxed_supercell_sha256",
        label="host supercell",
    )
    for name, entry in (reference_data.get("references", {}) or {}).items():
        if not isinstance(entry, Mapping) or entry.get("converged") is not True:
            raise ValueError(
                f"Correction application requires converged reference {name!r}; "
                "rerun refs-build"
            )
        entry_signature = require_relaxation_signature(
            entry,
            label=f"reference {name!r}",
        )
        if dict(entry_signature) != dict(host_signature):
            raise ValueError(
                f"Reference {name!r} has a relaxation signature inconsistent "
                "with the host/reference set; rerun refs-build"
            )
        require_structure_hash(
            entry,
            path_key="relaxed_poscar",
            hash_key="relaxed_sha256",
            label=f"reference {name}",
        )


def load_active_correction_model(
    raw_cfg: Mapping[str, Any],
    root: Path,
    reference_data: Mapping[str, Any],
) -> CorrectionModel | None:
    config = parse_correction_config(raw_cfg, root)
    if not config.enabled:
        return None
    validate_reference_energy_provenance(reference_data)
    signature = backend_signature_from_reference(reference_data, root=root)
    path = model_path(root, signature)
    model = load_correction_model(path)
    validate_backend_compatibility(model, signature)
    validate_applicability_compatibility(model)
    _validate_active_model_semantics(model)
    if config.model_family == "manual":
        if tuple(model.correction_terms) != tuple(config.correction_terms):
            raise ValueError(
                "Fitted correction terms do not match input.toml; rerun corrections-fit"
            )
    else:
        allowed_families = {
            "m0": {"m0"},
            "m1": {"m1"},
            "auto": {"m0", "m1"},
        }[config.model_family]
        if model.model_family not in allowed_families:
            raise ValueError(
                "Selected correction model family does not match input.toml; "
                "rerun corrections-fit"
            )
        if tuple(model.correction_terms[: len(config.correction_terms)]) != tuple(
            config.correction_terms
        ):
            raise ValueError(
                "Selected correction model does not retain the configured M0 basis; "
                "rerun corrections-fit"
            )
        expanded_terms = model.correction_terms[len(config.correction_terms) :]
        if any(
            not term.startswith(_OXIDE_CATION_PREFIX)
            or term.split(":", 1)[1] not in config.m1_elements
            for term in expanded_terms
        ):
            raise ValueError(
                "Selected M1 terms fall outside the workflow-derived target domain; "
                "rerun corrections-fit"
            )
        if tuple(model.target_elements) != tuple(config.target_elements):
            raise ValueError(
                "Correction target elements changed; rerun corrections-fit"
            )
        if not model.selection_run_hash:
            raise ValueError(
                "Automatic M0/M1 model lacks selection provenance; rerun corrections-fit"
            )
        selection_path = model_directory(root, signature) / MODEL_SELECTION_FILENAME
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Correction model-selection report is missing: {selection_path}"
            )
        try:
            selection_report = json.loads(selection_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid correction model-selection report: {selection_path}"
            ) from exc
        if selection_report.get("selection_run_hash") != model.selection_run_hash:
            raise ValueError(
                "Correction model-selection report does not match the active model; "
                "rerun corrections-fit"
            )
        selection_metadata = dict(model.selection_metadata or {})
        expected_selection_report_hash = selection_metadata.get(
            "selection_report_hash"
        )
        if (
            not expected_selection_report_hash
            or content_hash(selection_report) != expected_selection_report_hash
        ):
            raise ValueError(
                "Correction model-selection report content does not match the active "
                "model; rerun corrections-fit"
            )
        for family, candidate in (
            selection_report.get("candidate_models", {}) or {}
        ).items():
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"Invalid candidate-model record for family {family!r}"
                )
            for key in ("model_file", "fit_report_file"):
                candidate_path = model_directory(root, signature) / str(
                    candidate.get(key) or ""
                )
                if not candidate_path.is_file():
                    raise FileNotFoundError(
                        f"Correction candidate artifact is missing: {candidate_path}"
                    )
            candidate_model = load_correction_model(
                model_directory(root, signature) / str(candidate["model_file"])
            )
            if (
                candidate_model.fit_id != candidate.get("fit_id")
                or candidate_model.selection_run_hash != model.selection_run_hash
            ):
                raise ValueError(
                    f"Correction candidate model {family!r} does not match the "
                    "selection report; rerun corrections-fit"
                )
        expansion_name = selection_metadata.get("calibration_expansion_snapshot")
        expansion_hash = selection_metadata.get("calibration_expansion_snapshot_hash")
        if expansion_name:
            expansion_path = model_directory(root, signature) / str(expansion_name)
            if not expansion_path.exists():
                raise FileNotFoundError(
                    f"Correction calibration-expansion snapshot is missing: {expansion_path}"
                )
            try:
                expansion_snapshot = json.loads(
                    expansion_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid correction calibration-expansion snapshot: {expansion_path}"
                ) from exc
            if content_hash(expansion_snapshot) != expansion_hash:
                raise ValueError(
                    "Correction calibration-expansion snapshot does not match the "
                    "active model; rerun corrections-fit"
                )
    fitted_exclusions = tuple(
        (model.applicability_signature or {}).get("exclude_polyanions", ())
    )
    if fitted_exclusions != config.exclude_polyanions:
        raise ValueError(
            "Fitted polyanion applicability policy does not match input.toml; "
            "rerun corrections-fit"
        )
    activation_hash = correction_activation_hash(config, root)
    if not model.activation_input_hash or model.activation_input_hash != activation_hash:
        raise ValueError(
            "Correction calibration inputs/configuration changed; rerun corrections-fit"
        )
    return model

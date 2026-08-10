"""Pure model-family selection for backend-specific oxide corrections.

The existing correction fitter deliberately accepts an explicit feature basis.
This module adds a conservative, file-system-independent selector for two nested
model families:

``M0``
    One ordinary-oxide correction per oxygen atom.

``M1``
    ``M0`` plus workflow-relevant ``oxide_cation:<Element>`` terms.  These
    element terms are defined only for cations in ordinary oxides; they are not
    FERE-like corrections for elemental phases or arbitrary compounds.

The selector compares the families on identical leave-one-out predictions.  It
prefers M0 unless M1 is independently identifiable and clears a configured
predictive-improvement threshold.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from pymatgen.core import Composition, Element

OXIDE_TERM = "oxide"
OXIDE_CATION_PREFIX = "oxide_cation:"
_ORDINARY_OXIDE_TYPES = {"normal", "ordinary", "ordinary_oxide", "oxide"}


@dataclass(frozen=True)
class ModelSelectionConfig:
    """Numerical and coverage gates for automatic M0/M1 selection."""

    min_m0_compounds: int = 3
    min_independent_cation_support: int = 3
    min_unique_oxygen_ratios: int = 2
    max_condition_number: float = 1.0e4
    min_cv_rmse_improvement_eV_per_atom: float = 0.01
    require_one_standard_error: bool = True

    def __post_init__(self) -> None:
        if self.min_m0_compounds < 3:
            raise ValueError("min_m0_compounds must be >= 3")
        if self.min_independent_cation_support < 3:
            raise ValueError("min_independent_cation_support must be >= 3")
        if self.min_unique_oxygen_ratios < 1:
            raise ValueError("min_unique_oxygen_ratios must be >= 1")
        if (
            not math.isfinite(self.max_condition_number)
            or self.max_condition_number <= 1.0
        ):
            raise ValueError("max_condition_number must be finite and > 1")
        if (
            not math.isfinite(self.min_cv_rmse_improvement_eV_per_atom)
            or self.min_cv_rmse_improvement_eV_per_atom < 0.0
        ):
            raise ValueError(
                "min_cv_rmse_improvement_eV_per_atom must be finite and >= 0"
            )
        if not isinstance(self.require_one_standard_error, bool):
            raise ValueError("require_one_standard_error must be a boolean")


@dataclass(frozen=True)
class FamilyFit:
    """A fitted correction family and its deterministic LOO diagnostics."""

    family: str
    terms: tuple[str, ...]
    coefficients_eV_per_term: tuple[float, ...]
    covariance_eV2: tuple[tuple[float, ...], ...]
    coefficient_uncertainties_eV_per_term: tuple[float, ...]
    n_observations: int
    rank: int
    condition_number: float
    max_loo_condition_number: float
    training_rmse_eV_per_atom: float
    loo_rmse_eV_per_atom: float
    loo_mae_eV_per_atom: float
    formulas: tuple[str, ...]
    observed_corrections_eV_per_formula: tuple[float, ...]
    fitted_corrections_eV_per_formula: tuple[float, ...]
    loo_predictions_eV_per_formula: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelSelectionResult:
    """Selected family plus complete model-selection provenance."""

    selected_family: str
    selected_model: FamilyFit
    m0_model: FamilyFit
    m1_model: FamilyFit | None
    report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_family": self.selected_family,
            "selected_model": self.selected_model.to_dict(),
            "m0_model": self.m0_model.to_dict(),
            "m1_model": self.m1_model.to_dict() if self.m1_model else None,
            "report": dict(self.report),
        }


@dataclass(frozen=True)
class _Observation:
    source_index: int
    formula: str
    reduced_formula: str
    composition: Composition
    observed_correction_eV_per_formula: float
    uncertainty_eV_per_formula: float
    oxide_type: str


class _AdmissionError(ValueError):
    """Internal signal that a candidate family is not scientifically admissible."""


def oxide_cation_term(element: str) -> str:
    """Return the canonical scoped term name for an element symbol."""
    symbol = _element_symbol(element)
    if symbol == "O":
        raise ValueError("O cannot be used as an oxide-cation correction term")
    return f"{OXIDE_CATION_PREFIX}{symbol}"


def _element_symbol(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Element symbol must be non-empty")
    try:
        return Element(text).symbol
    except ValueError as exc:
        raise ValueError(f"Invalid element symbol {value!r}") from exc


def _values(section: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    value = section.get(key, ())
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError(f"Configuration value {key!r} must be an array")
    return tuple(value)


def _formula_elements(value: Any, *, label: str) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    try:
        return {element.symbol for element in Composition(text).elements}
    except Exception as exc:
        raise ValueError(f"Invalid formula {text!r} in {label}") from exc


def infer_workflow_target_elements(raw_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Infer prospective oxide cations from chemistry already present in TOML data.

    The function is intentionally pure.  It uses host/dopant declarations and
    configured reference formulas, while excluding the declared anions.  Actual
    structures can be incorporated upstream by adding their elements to the
    same configuration inventory before calling the selector.
    """

    references = raw_config.get("references", {}) or {}
    doping = raw_config.get("doping", {}) or {}
    scan = raw_config.get("scan", {}) or {}
    vacancies = raw_config.get("vacancies", {}) or {}
    for name, section in (
        ("references", references),
        ("doping", doping),
        ("scan", scan),
        ("vacancies", vacancies),
    ):
        if not isinstance(section, Mapping):
            raise ValueError(f"[{name}] must be a table")

    anions = {"O"}
    for value in _values(scan, "anion_species"):
        anions.add(_element_symbol(value))
    vacancy_species = str(vacancies.get("vacancy_species") or "").strip()
    if vacancy_species:
        anions.add(_element_symbol(vacancy_species))

    elements: set[str] = set()
    elements.update(
        _formula_elements(references.get("host"), label="[references].host")
    )
    for key in ("oxides_ref", "metal_ref"):
        for value in _values(references, key):
            elements.update(
                _formula_elements(value, label=f"[references].{key}")
            )

    host_species = str(doping.get("host_species") or "").strip()
    if host_species:
        elements.add(_element_symbol(host_species))
    for key in ("dopants", "must_include"):
        for value in _values(doping, key):
            elements.add(_element_symbol(value))

    compositions = doping.get("compositions", ()) or ()
    if isinstance(compositions, (str, bytes)) or not isinstance(
        compositions, Sequence
    ):
        raise ValueError("[doping].compositions must be an array of tables")
    for index, composition in enumerate(compositions):
        if not isinstance(composition, Mapping):
            raise ValueError(
                f"[doping].compositions[{index}] must be a table"
            )
        for element in composition:
            elements.add(_element_symbol(element))

    for value in _values(vacancies, "oxidation_state_elements"):
        elements.add(_element_symbol(value))

    return tuple(sorted(elements - anions))


def correction_feature_vector(
    composition: Composition | str,
    oxide_type: str,
    terms: Sequence[str],
) -> tuple[float, ...]:
    """Build a feature vector with oxide-cation terms scoped to ordinary oxides."""

    comp = composition if isinstance(composition, Composition) else Composition(composition)
    ordinary_oxide = (
        str(oxide_type or "").strip().lower() in _ORDINARY_OXIDE_TYPES
        and "O" in comp
        and len(comp.elements) > 1
    )
    values: list[float] = []
    for term in terms:
        if term == OXIDE_TERM:
            values.append(float(comp.get("O", 0.0)) if ordinary_oxide else 0.0)
            continue
        if term.startswith(OXIDE_CATION_PREFIX):
            symbol = _element_symbol(term.removeprefix(OXIDE_CATION_PREFIX))
            if symbol == "O":
                raise ValueError(
                    "O cannot be used as an oxide-cation correction term"
                )
            values.append(float(comp.get(symbol, 0.0)) if ordinary_oxide else 0.0)
            continue
        raise ValueError(f"Unsupported model-selection correction term {term!r}")
    return tuple(values)


def _parse_observations(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[_Observation], list[dict[str, Any]]]:
    observations: list[_Observation] = []
    rejected: list[dict[str, Any]] = []
    seen_formulas: dict[str, int] = {}
    for index, row in enumerate(rows):
        formula = str(row.get("formula") or "").strip()
        if not formula:
            raise ValueError(f"Calibration row {index} is missing formula")
        try:
            composition = Composition(formula)
        except Exception as exc:
            raise ValueError(
                f"Calibration row {index} has invalid formula {formula!r}"
            ) from exc
        reduced_formula = composition.reduced_formula
        oxide_type = str(row.get("oxide_type") or "").strip().lower()
        if oxide_type not in _ORDINARY_OXIDE_TYPES:
            rejected.append(
                {
                    "index": index,
                    "formula": reduced_formula,
                    "reason": "not_ordinary_oxide",
                }
            )
            continue
        if "O" not in composition or len(composition.elements) < 2:
            rejected.append(
                {
                    "index": index,
                    "formula": reduced_formula,
                    "reason": "not_non_elemental_oxide",
                }
            )
            continue
        if reduced_formula in seen_formulas:
            first = seen_formulas[reduced_formula]
            raise ValueError(
                "Model-family selection requires one independent row per reduced "
                f"formula; {reduced_formula} occurs at rows {first} and {index}"
            )
        seen_formulas[reduced_formula] = index

        try:
            experimental = float(row["experimental_formation_eV_per_formula"])
            calculated = float(row["calculated_formation_eV_per_formula"])
            uncertainty = float(row["experimental_uncertainty_eV_per_formula"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Calibration row {index} has missing or invalid energy data"
            ) from exc
        if not (
            math.isfinite(experimental)
            and math.isfinite(calculated)
            and math.isfinite(uncertainty)
            and uncertainty > 0.0
        ):
            raise ValueError(
                f"Calibration row {index} energies must be finite and uncertainty > 0"
            )
        observations.append(
            _Observation(
                source_index=index,
                formula=formula,
                reduced_formula=reduced_formula,
                composition=composition,
                observed_correction_eV_per_formula=experimental - calculated,
                uncertainty_eV_per_formula=uncertainty,
                oxide_type="oxide",
            )
        )
    return observations, rejected


def _design_matrix(
    observations: Sequence[_Observation], terms: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            correction_feature_vector(
                observation.composition,
                observation.oxide_type,
                terms,
            )
            for observation in observations
        ],
        dtype=float,
    )


def _condition_number(matrix: np.ndarray) -> float:
    value = float(np.linalg.cond(matrix))
    return value if math.isfinite(value) else math.inf


def _fit_family(
    observations: Sequence[_Observation],
    *,
    family: str,
    terms: tuple[str, ...],
    max_condition_number: float,
) -> FamilyFit:
    n_rows = len(observations)
    n_terms = len(terms)
    if n_rows <= n_terms:
        raise _AdmissionError(
            f"{family} needs more observations than terms for leave-one-out fitting"
        )

    matrix = _design_matrix(observations, terms)
    observed = np.asarray(
        [item.observed_correction_eV_per_formula for item in observations],
        dtype=float,
    )
    sigma = np.asarray(
        [item.uncertainty_eV_per_formula for item in observations],
        dtype=float,
    )
    weighted_matrix = matrix / sigma[:, None]
    weighted_observed = observed / sigma
    rank = int(np.linalg.matrix_rank(weighted_matrix))
    if rank != n_terms:
        raise _AdmissionError(
            f"{family} weighted design is rank deficient ({rank} < {n_terms})"
        )
    condition = _condition_number(weighted_matrix)
    if condition > max_condition_number:
        raise _AdmissionError(
            f"{family} weighted design condition number {condition:.6g} exceeds "
            f"{max_condition_number:.6g}"
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
    fitted = matrix @ coefficients

    loo_predictions = np.empty(n_rows, dtype=float)
    loo_conditions: list[float] = []
    for held_out in range(n_rows):
        keep = np.arange(n_rows) != held_out
        train_matrix = weighted_matrix[keep]
        train_observed = weighted_observed[keep]
        fold_rank = int(np.linalg.matrix_rank(train_matrix))
        if fold_rank != n_terms:
            raise _AdmissionError(
                f"{family} loses full rank when leaving out "
                f"{observations[held_out].reduced_formula}"
            )
        fold_condition = _condition_number(train_matrix)
        if fold_condition > max_condition_number:
            raise _AdmissionError(
                f"{family} exceeds the condition bound when leaving out "
                f"{observations[held_out].reduced_formula}: {fold_condition:.6g}"
            )
        loo_conditions.append(fold_condition)
        fold_coefficients, _, _, _ = np.linalg.lstsq(
            train_matrix,
            train_observed,
            rcond=None,
        )
        loo_predictions[held_out] = float(matrix[held_out] @ fold_coefficients)

    atom_counts = np.asarray(
        [float(item.composition.num_atoms) for item in observations], dtype=float
    )
    training_residual_per_atom = (observed - fitted) / atom_counts
    loo_residual_per_atom = (observed - loo_predictions) / atom_counts
    uncertainties = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return FamilyFit(
        family=family,
        terms=terms,
        coefficients_eV_per_term=tuple(float(value) for value in coefficients),
        covariance_eV2=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        coefficient_uncertainties_eV_per_term=tuple(
            float(value) for value in uncertainties
        ),
        n_observations=n_rows,
        rank=rank,
        condition_number=condition,
        max_loo_condition_number=max(loo_conditions),
        training_rmse_eV_per_atom=float(
            np.sqrt(np.mean(np.square(training_residual_per_atom)))
        ),
        loo_rmse_eV_per_atom=float(
            np.sqrt(np.mean(np.square(loo_residual_per_atom)))
        ),
        loo_mae_eV_per_atom=float(np.mean(np.abs(loo_residual_per_atom))),
        formulas=tuple(item.reduced_formula for item in observations),
        observed_corrections_eV_per_formula=tuple(float(value) for value in observed),
        fitted_corrections_eV_per_formula=tuple(float(value) for value in fitted),
        loo_predictions_eV_per_formula=tuple(
            float(value) for value in loo_predictions
        ),
    )


def _cation_support(
    observations: Sequence[_Observation], element: str
) -> dict[str, Any]:
    formula_ratio_pairs: set[tuple[str, float]] = set()
    formulas: set[str] = set()
    ratios: set[float] = set()
    for observation in observations:
        amount = float(observation.composition.get(element, 0.0))
        if amount <= 0.0:
            continue
        oxygen_ratio = float(observation.composition["O"]) / amount
        rounded_ratio = round(oxygen_ratio, 12)
        formulas.add(observation.reduced_formula)
        ratios.add(rounded_ratio)
        formula_ratio_pairs.add((observation.reduced_formula, rounded_ratio))
    return {
        "unique_formulas": len(formulas),
        "unique_oxygen_ratios": len(ratios),
        "independent_formula_ratio_pairs": len(formula_ratio_pairs),
        "formulas": sorted(formulas),
        "oxygen_ratios": sorted(ratios),
    }


def _family_indices(
    observations: Sequence[_Observation], element: str
) -> tuple[int, ...]:
    return tuple(
        index
        for index, observation in enumerate(observations)
        if float(observation.composition.get(element, 0.0)) > 0.0
    )


def _loo_residuals_per_atom(
    model: FamilyFit,
    observations: Sequence[_Observation],
) -> np.ndarray:
    observed = np.asarray(model.observed_corrections_eV_per_formula, dtype=float)
    predicted = np.asarray(model.loo_predictions_eV_per_formula, dtype=float)
    atom_counts = np.asarray(
        [float(observation.composition.num_atoms) for observation in observations],
        dtype=float,
    )
    return (observed - predicted) / atom_counts


def _family_loo_rmse(
    model: FamilyFit,
    observations: Sequence[_Observation],
    indices: Sequence[int],
) -> float:
    if not indices:
        raise ValueError("Cannot calculate family-specific RMSE for an empty family")
    residuals = _loo_residuals_per_atom(model, observations)[list(indices)]
    return float(np.sqrt(np.mean(np.square(residuals))))


def _paired_squared_loss_improvement(
    m0: FamilyFit,
    m1: FamilyFit,
    observations: Sequence[_Observation],
) -> dict[str, float | bool]:
    m0_squared_loss = np.square(_loo_residuals_per_atom(m0, observations))
    m1_squared_loss = np.square(_loo_residuals_per_atom(m1, observations))
    paired_improvement = m0_squared_loss - m1_squared_loss
    mean_improvement = float(np.mean(paired_improvement))
    standard_error = float(
        np.std(paired_improvement, ddof=1) / math.sqrt(len(paired_improvement))
    )
    tolerance = 1.0e-15
    return {
        "mean_eV2_per_atom2": mean_improvement,
        "standard_error_eV2_per_atom2": standard_error,
        "passes_one_standard_error": mean_improvement > standard_error + tolerance,
    }


def select_correction_model_family(
    calibration_rows: Sequence[Mapping[str, Any]],
    raw_config: Mapping[str, Any],
    *,
    selection_config: ModelSelectionConfig | None = None,
    target_elements: Sequence[str] | None = None,
) -> ModelSelectionResult:
    """Fit and select M0 or M1 without reading or writing external state.

    Required calibration-row keys are ``formula``, ``oxide_type``,
    ``experimental_formation_eV_per_formula``,
    ``calculated_formation_eV_per_formula``, and
    ``experimental_uncertainty_eV_per_formula``.
    """

    config = selection_config or ModelSelectionConfig()
    observations, rejected_rows = _parse_observations(calibration_rows)
    if len(observations) < config.min_m0_compounds:
        raise ValueError(
            "M0 requires at least "
            f"{config.min_m0_compounds} independent ordinary-oxide formulas; "
            f"got {len(observations)}"
        )

    m0 = _fit_family(
        observations,
        family="M0",
        terms=(OXIDE_TERM,),
        max_condition_number=config.max_condition_number,
    )
    if target_elements is None:
        workflow_elements = infer_workflow_target_elements(raw_config)
        target_element_source = "raw_config"
    else:
        if isinstance(target_elements, (str, bytes)):
            raise ValueError("target_elements must be an array of element symbols")
        normalized_targets = tuple(_element_symbol(value) for value in target_elements)
        if "O" in normalized_targets:
            raise ValueError("target_elements cannot contain oxygen")
        workflow_elements = tuple(sorted(set(normalized_targets)))
        target_element_source = "explicit_override"
    coverage: dict[str, dict[str, Any]] = {}
    admitted_elements: list[str] = []
    excluded_elements: dict[str, list[str]] = {}
    for element in workflow_elements:
        diagnostics = _cation_support(observations, element)
        reasons: list[str] = []
        if (
            diagnostics["independent_formula_ratio_pairs"]
            < config.min_independent_cation_support
        ):
            reasons.append(
                "insufficient_independent_formula_ratio_support"
            )
        if diagnostics["unique_oxygen_ratios"] < config.min_unique_oxygen_ratios:
            reasons.append("insufficient_unique_oxygen_ratios")

        if not reasons:
            try:
                single_element_model = _fit_family(
                    observations,
                    family=f"M1[{element}]",
                    terms=(OXIDE_TERM, oxide_cation_term(element)),
                    max_condition_number=config.max_condition_number,
                )
            except _AdmissionError as exc:
                reasons.append(str(exc))
            else:
                indices = _family_indices(observations, element)
                m0_family_rmse = _family_loo_rmse(m0, observations, indices)
                m1_family_rmse = _family_loo_rmse(
                    single_element_model,
                    observations,
                    indices,
                )
                diagnostics["m0_family_loo_rmse_eV_per_atom"] = m0_family_rmse
                diagnostics[
                    "m1_single_term_family_loo_rmse_eV_per_atom"
                ] = m1_family_rmse
                diagnostics["family_loo_rmse_improvement_eV_per_atom"] = (
                    m0_family_rmse - m1_family_rmse
                )
                if m1_family_rmse > m0_family_rmse + 1.0e-12:
                    reasons.append("family_specific_loo_rmse_worsened")

        diagnostics["admitted"] = not reasons
        diagnostics["reasons"] = reasons
        coverage[element] = diagnostics
        if reasons:
            excluded_elements[element] = reasons
        else:
            admitted_elements.append(element)

    m1: FamilyFit | None = None
    m1_unavailable_reason: str | None = None
    combined_family_diagnostics: dict[str, dict[str, float | bool]] = {}
    combined_worsened_elements: list[str] = []
    if admitted_elements:
        m1_terms = (OXIDE_TERM,) + tuple(
            oxide_cation_term(element) for element in sorted(admitted_elements)
        )
        try:
            m1 = _fit_family(
                observations,
                family="M1",
                terms=m1_terms,
                max_condition_number=config.max_condition_number,
            )
        except _AdmissionError as exc:
            m1_unavailable_reason = str(exc)
        else:
            for element in sorted(admitted_elements):
                indices = _family_indices(observations, element)
                m0_family_rmse = _family_loo_rmse(m0, observations, indices)
                m1_family_rmse = _family_loo_rmse(m1, observations, indices)
                non_worsening = m1_family_rmse <= m0_family_rmse + 1.0e-12
                combined_family_diagnostics[element] = {
                    "m0_loo_rmse_eV_per_atom": m0_family_rmse,
                    "m1_loo_rmse_eV_per_atom": m1_family_rmse,
                    "improvement_eV_per_atom": m0_family_rmse - m1_family_rmse,
                    "non_worsening": non_worsening,
                }
                if not non_worsening:
                    combined_worsened_elements.append(element)
            if combined_worsened_elements:
                m1_unavailable_reason = (
                    "combined M1 worsens family-specific LOO RMSE for: "
                    + ", ".join(combined_worsened_elements)
                )
                m1 = None
    else:
        details = "; ".join(
            f"{element}: {', '.join(reasons)}"
            for element, reasons in sorted(excluded_elements.items())
        )
        m1_unavailable_reason = (
            "no workflow cation passed independent coverage gates"
            + (f" ({details})" if details else "")
        )

    improvement: float | None = None
    paired_loss: dict[str, float | bool] | None = None
    one_standard_error_passed: bool | None = None
    selected = m0
    selection_reason = "M1 unavailable; selected conservative M0"
    if m1 is not None:
        if m1.formulas != m0.formulas:
            raise RuntimeError("M0 and M1 were not evaluated on identical observations")
        improvement = m0.loo_rmse_eV_per_atom - m1.loo_rmse_eV_per_atom
        paired_loss = _paired_squared_loss_improvement(m0, m1, observations)
        one_standard_error_passed = bool(
            paired_loss["passes_one_standard_error"]
        )
        # Strict comparison deliberately gives an exact threshold tie to M0.
        clears_rmse_threshold = improvement > (
            config.min_cv_rmse_improvement_eV_per_atom + 1.0e-12
        )
        clears_uncertainty_gate = (
            not config.require_one_standard_error or one_standard_error_passed
        )
        if clears_rmse_threshold and clears_uncertainty_gate:
            selected = m1
            selection_reason = (
                "M1 exceeded the configured LOO RMSE improvement and uncertainty gate"
            )
        else:
            selection_reason = (
                "M1 did not exceed all configured LOO improvement gates; "
                "selected conservative M0"
            )

    report: dict[str, Any] = {
        "selection_method": (
            "identical_leave_one_out_rmse_with_paired_loss_gate_and_m0_tie_break"
        ),
        "selected_family": selected.family,
        "selection_reason": selection_reason,
        "target_elements": list(workflow_elements),
        "target_element_source": target_element_source,
        "admitted_m1_elements": sorted(admitted_elements),
        "excluded_m1_elements": excluded_elements,
        "cation_coverage": coverage,
        "combined_m1_family_loo_diagnostics": combined_family_diagnostics,
        "combined_m1_worsened_elements": combined_worsened_elements,
        "eligible_formulas": [item.reduced_formula for item in observations],
        "rejected_rows": rejected_rows,
        "m0_loo_rmse_eV_per_atom": m0.loo_rmse_eV_per_atom,
        "m1_loo_rmse_eV_per_atom": (
            m1.loo_rmse_eV_per_atom if m1 is not None else None
        ),
        "loo_rmse_improvement_eV_per_atom": improvement,
        "paired_loo_squared_loss_improvement": paired_loss,
        "require_one_standard_error": config.require_one_standard_error,
        "one_standard_error_passed": one_standard_error_passed,
        "minimum_required_improvement_eV_per_atom": (
            config.min_cv_rmse_improvement_eV_per_atom
        ),
        "m1_unavailable_reason": m1_unavailable_reason,
        "thresholds": asdict(config),
        "applicability": {
            "M0": "ordinary_oxide_oxygen_only",
            "M1": "ordinary_oxide_oxygen_plus_scoped_workflow_cations",
        },
    }
    return ModelSelectionResult(
        selected_family=selected.family,
        selected_model=selected,
        m0_model=m0,
        m1_model=m1,
        report=report,
    )

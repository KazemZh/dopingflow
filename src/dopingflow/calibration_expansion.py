"""Deterministic discovery and acquisition helpers for correction calibration.

The functions in this module deliberately stop short of fitting a correction
model.  They identify the chemically scoped, phase-resolved oxides that
can be considered by a later calibration stage and, optionally, cache the
exact OPTIMADE structure associated with a curated Materials Project ID.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen

from pymatgen.core import Composition, Element, Structure
from pymatgen.io.vasp import Poscar

from dopingflow.corrections import ExperimentalRecord


_GENERIC_PHASE_KEYS = {
    "",
    "cr",
    "cryst",
    "crystal",
    "crystalline",
    "solid",
    "none",
    "unknown",
    "na",
}

# These are the non-ordinary binary oxygen compounds present in the pinned
# Kingsbury data.  Keep the unreduced formulas: reducing, for example, Li2O2
# to LiO would erase the peroxide stoichiometry recorded by the source.
NON_ORDINARY_BINARY_OXIDE_FORMULAS = frozenset(
    {
        "BaO2",
        "CsO2",
        "K2O2",
        "KO2",
        "Li2O2",
        "Na2O2",
        "NaO2",
        "SrO2",
    }
)

_MATERIAL_ID_RE = re.compile(r"^mp-[0-9]+$", re.IGNORECASE)
_SAFE_OPTIMADE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class CalibrationDiscovery:
    """Accepted experimental rows and an auditable selection report."""

    accepted_records: tuple[ExperimentalRecord, ...]
    report: Mapping[str, Any]

    @property
    def records(self) -> tuple[ExperimentalRecord, ...]:
        """Short alias useful to callers that only need accepted records."""

        return self.accepted_records


@dataclass(frozen=True)
class CachedOptimadeStructure:
    """Provenance for one immutable cached OPTIMADE response and POSCAR."""

    material_id: str
    reduced_formula: str
    source_url: str
    response_json_path: Path
    structure_path: Path
    response_sha256: str
    structure_sha256: str
    n_sites: int
    from_cache: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["response_json_path"] = str(self.response_json_path)
        data["structure_path"] = str(self.structure_path)
        return data


OptimadeTransport = Callable[[str], bytes | str | Mapping[str, Any]]


def _phase_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def is_generic_phase(value: Any) -> bool:
    """Return whether a phase label lacks polymorph-resolving information."""

    return _phase_key(value) in _GENERIC_PHASE_KEYS


def _element_symbol(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    try:
        return Element(text).symbol
    except (ValueError, KeyError) as exc:
        raise ValueError(f"{field} contains an invalid element symbol: {value!r}") from exc


def infer_scoped_elements(raw_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Infer non-oxygen host/dopant elements from a workflow configuration.

    The returned scope is a project-wide union.  A correction model must be
    common to all entries in a phase diagram; individual structures later get
    zero-valued features for scoped terms belonging to elements they lack.
    """

    references = raw_config.get("references", {}) or {}
    doping = raw_config.get("doping", {}) or {}
    if not isinstance(references, Mapping) or not isinstance(doping, Mapping):
        raise ValueError("[references] and [doping] must be tables")

    scoped: set[str] = set()
    host = str(references.get("host") or "").strip()
    if host:
        try:
            host_composition = Composition(host)
        except Exception as exc:
            raise ValueError(
                f"[references].host must be a parseable chemical formula; got {host!r}"
            ) from exc
        scoped.update(
            element.symbol for element in host_composition.elements if element.symbol != "O"
        )
    else:
        host_species = str(doping.get("host_species") or "").strip()
        if host_species:
            symbol = _element_symbol(host_species, field="[doping].host_species")
            if symbol != "O":
                scoped.add(symbol)

    mode = str(doping.get("mode", "explicit")).strip().lower()
    if mode == "explicit":
        compositions = doping.get("compositions", []) or []
        if not isinstance(compositions, Sequence) or isinstance(compositions, (str, bytes)):
            raise ValueError("[doping].compositions must be an array of tables")
        for index, composition in enumerate(compositions):
            if not isinstance(composition, Mapping):
                raise ValueError(f"[doping].compositions[{index}] must be a table")
            for raw_symbol, raw_amount in composition.items():
                symbol = _element_symbol(
                    raw_symbol,
                    field=f"[doping].compositions[{index}]",
                )
                amount = float(raw_amount)
                if not math.isfinite(amount) or amount < 0:
                    raise ValueError(
                        f"[doping].compositions[{index}].{symbol} must be finite and >= 0"
                    )
                if amount > 0 and symbol != "O":
                    scoped.add(symbol)
    elif mode == "enumerate":
        for field in ("dopants", "must_include"):
            values = doping.get(field, []) or []
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"[doping].{field} must be an array")
            for value in values:
                symbol = _element_symbol(value, field=f"[doping].{field}")
                if symbol != "O":
                    scoped.add(symbol)
    else:
        raise ValueError("[doping].mode must be 'explicit' or 'enumerate'")

    if not scoped:
        raise ValueError("Could not infer any non-oxygen host or dopant elements")
    return tuple(sorted(scoped))


def _compact_formula(value: str) -> str:
    return re.sub(r"\s+", "", str(value))


def _record_sort_key(record: ExperimentalRecord) -> tuple[Any, ...]:
    try:
        composition = Composition(record.formula)
        cation = next(
            (
                element.symbol
                for element in composition.elements
                if element.symbol != "O"
            ),
            "",
        )
        ratio = (
            float(composition["O"]) / float(composition[cation]) if cation else math.inf
        )
    except Exception:
        cation = ""
        ratio = math.inf
    return (
        cation,
        ratio,
        str(record.reduced_formula),
        _phase_key(record.phase),
        str(record.likely_mpid).lower(),
        float(record.formation_enthalpy_eV_per_atom),
        str(record.dataset),
        str(record.source),
    )


def _known_oxygen_environment(
    record: ExperimentalRecord,
    known: Mapping[str, str] | None,
) -> str | None:
    if not known:
        return None
    keys = (
        str(record.likely_mpid),
        str(record.formula),
        str(record.reduced_formula),
    )
    for key in keys:
        if key in known:
            value = str(known[key]).strip().lower()
            return "oxide" if value == "normal" else value
    return None


def discover_phase_resolved_oxides(
    records: Sequence[ExperimentalRecord],
    scoped_elements: Sequence[str],
    *,
    max_non_oxygen_elements: int | None = None,
    known_oxygen_environments: Mapping[str, str] | None = None,
    excluded_formula_tokens: Sequence[str] = (),
    excluded_formulas: Sequence[str] = tuple(
        sorted(NON_ORDINARY_BINARY_OXIDE_FORMULAS)
    ),
) -> CalibrationDiscovery:
    """Select strict, ordinary, phase-resolved oxides for a project.

    Every non-oxygen element must be in ``scoped_elements``.  Set
    ``max_non_oxygen_elements=1`` for binary-only discovery; ``None`` includes
    every available binary, ternary, or higher oxide in the target chemistry.
    """

    if max_non_oxygen_elements is not None and max_non_oxygen_elements < 1:
        raise ValueError("max_non_oxygen_elements must be >= 1 or None")

    scope = tuple(
        sorted(
            {
                _element_symbol(value, field="scoped_elements")
                for value in scoped_elements
            }
        )
    )
    if not scope:
        raise ValueError("scoped_elements must contain at least one non-oxygen element")
    if "O" in scope:
        raise ValueError("scoped_elements must contain non-oxygen host/dopant elements only")
    scope_set = set(scope)
    excluded = {_compact_formula(value) for value in excluded_formulas}
    excluded_tokens = tuple(
        _compact_formula(value)
        for value in excluded_formula_tokens
        if _compact_formula(value)
    )
    candidates: list[ExperimentalRecord] = []
    rejections: list[dict[str, Any]] = []

    def reject(record: ExperimentalRecord, reason: str, **details: Any) -> None:
        item: dict[str, Any] = {
            "formula": str(record.formula),
            "reduced_formula": str(record.reduced_formula),
            "phase": str(record.phase),
            "likely_mpid": str(record.likely_mpid),
            "reason": reason,
        }
        item.update(details)
        rejections.append(item)

    for record in sorted(records, key=_record_sort_key):
        try:
            composition = Composition(record.formula)
        except Exception:
            reject(record, "invalid_formula")
            continue
        symbols = {element.symbol for element in composition.elements}
        compact_formula = _compact_formula(record.formula)
        compact_reduced_formula = _compact_formula(record.reduced_formula)
        matched_exclusion = next(
            (
                token
                for token in excluded_tokens
                if token in compact_formula or token in compact_reduced_formula
            ),
            None,
        )
        if matched_exclusion is not None:
            reject(
                record,
                "excluded_polyanion",
                matched_token=matched_exclusion,
            )
            continue
        if "O" not in symbols:
            reject(record, "no_oxygen")
            continue
        cations = symbols - {"O"}
        if not cations:
            reject(record, "no_non_oxygen_element")
            continue
        outside_scope = sorted(cations - scope_set)
        if outside_scope:
            reject(
                record,
                "outside_scoped_elements",
                outside_scoped_elements=outside_scope,
            )
            continue
        if (
            max_non_oxygen_elements is not None
            and len(cations) > max_non_oxygen_elements
        ):
            reject(
                record,
                "too_many_non_oxygen_elements",
                non_oxygen_element_count=len(cations),
            )
            continue
        if is_generic_phase(record.phase):
            reject(record, "generic_or_missing_phase", cations=sorted(cations))
            continue
        material_id = str(record.likely_mpid or "").strip()
        if not material_id:
            reject(record, "missing_likely_mpid", cations=sorted(cations))
            continue
        if not _MATERIAL_ID_RE.fullmatch(material_id):
            reject(record, "invalid_likely_mpid", cations=sorted(cations))
            continue
        if _compact_formula(record.formula) in excluded:
            reject(
                record,
                "known_peroxide_or_superoxide_formula",
                cations=sorted(cations),
            )
            continue
        oxygen_environment = _known_oxygen_environment(
            record,
            known_oxygen_environments,
        )
        if oxygen_environment is not None and oxygen_environment != "oxide":
            reject(
                record,
                "non_ordinary_oxygen_environment",
                cations=sorted(cations),
                oxygen_environment=oxygen_environment,
            )
            continue
        candidates.append(record)

    accepted: list[ExperimentalRecord] = []
    seen: dict[tuple[str, str, str], ExperimentalRecord] = {}
    for record in candidates:
        identity = (
            str(record.reduced_formula),
            _phase_key(record.phase),
            str(record.likely_mpid).strip().lower(),
        )
        previous = seen.get(identity)
        if previous is None:
            seen[identity] = record
            accepted.append(record)
            continue
        if previous == record:
            reject(record, "duplicate_record")
        else:
            reject(
                record,
                "conflicting_duplicate_record",
                kept_source=str(previous.source),
            )

    coverage: dict[str, dict[str, Any]] = {}
    for element in scope:
        selected = [
            record
            for record in accepted
            if element in {item.symbol for item in Composition(record.formula).elements}
        ]
        formulas = [str(record.reduced_formula) for record in selected]
        oxygen_ratios = sorted(
            {
                float(Composition(record.formula)["O"])
                / float(Composition(record.formula)[element])
                for record in selected
            }
        )
        coverage[element] = {
            "accepted_count": len(selected),
            "formulas": formulas,
            "independent_oxygen_stoichiometry_count": len(oxygen_ratios),
            "oxygen_per_cation_ratios": oxygen_ratios,
        }

    rejection_counts: dict[str, int] = {}
    for item in rejections:
        reason = str(item["reason"])
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    rejections.sort(
        key=lambda item: (
            str(item["reason"]),
            str(item["formula"]),
            str(item["phase"]),
            str(item["likely_mpid"]),
        )
    )
    report = {
        "schema_version": 1,
        "selection_policy": "strict_phase_resolved_ordinary_target_scope_oxides",
        "max_non_oxygen_elements": max_non_oxygen_elements,
        "scoped_elements": list(scope),
        "input_count": len(records),
        "accepted_count": len(accepted),
        "accepted_formulas": [str(record.reduced_formula) for record in accepted],
        "coverage_by_element": coverage,
        "rejected_count": len(rejections),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejections": rejections,
    }
    return CalibrationDiscovery(tuple(accepted), report)


def discover_phase_resolved_binary_oxides(
    records: Sequence[ExperimentalRecord],
    scoped_elements: Sequence[str],
    *,
    known_oxygen_environments: Mapping[str, str] | None = None,
    excluded_formula_tokens: Sequence[str] = (),
    excluded_formulas: Sequence[str] = tuple(
        sorted(NON_ORDINARY_BINARY_OXIDE_FORMULAS)
    ),
) -> CalibrationDiscovery:
    """Backward-compatible binary-only phase-resolved discovery helper."""

    return discover_phase_resolved_oxides(
        records,
        scoped_elements,
        max_non_oxygen_elements=1,
        known_oxygen_environments=known_oxygen_environments,
        excluded_formula_tokens=excluded_formula_tokens,
        excluded_formulas=excluded_formulas,
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ordered_species(attributes: Mapping[str, Any]) -> list[str]:
    labels = attributes.get("species_at_sites")
    if not isinstance(labels, list) or not labels:
        raise ValueError("OPTIMADE structure lacks species_at_sites")
    definitions = {
        str(item.get("name")): item
        for item in (attributes.get("species") or [])
        if isinstance(item, Mapping) and item.get("name") is not None
    }
    species: list[str] = []
    for label in labels:
        name = str(label)
        definition = definitions.get(name)
        if definition is None:
            species.append(_element_symbol(name, field="OPTIMADE species_at_sites"))
            continue
        symbols = definition.get("chemical_symbols")
        concentrations = definition.get("concentration")
        if (
            not isinstance(symbols, list)
            or len(symbols) != 1
            or str(symbols[0]).lower() == "vacancy"
            or not isinstance(concentrations, list)
            or len(concentrations) != 1
            or not math.isclose(float(concentrations[0]), 1.0, abs_tol=1.0e-12)
        ):
            raise ValueError("Disordered or vacancy-containing OPTIMADE sites are unsupported")
        species.append(
            _element_symbol(symbols[0], field=f"OPTIMADE species {name!r}")
        )
    return species


def _structure_from_optimade_payload(
    payload: Mapping[str, Any],
    *,
    material_id: str,
    expected_formula: str,
) -> Structure:
    data: Any = payload.get("data")
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError("OPTIMADE response must contain exactly one structure")
        data = data[0]
    if not isinstance(data, Mapping):
        raise ValueError("OPTIMADE response lacks a structure data object")
    response_id = str(data.get("id") or "")
    if response_id.lower() != material_id.lower():
        raise ValueError(
            f"OPTIMADE structure ID mismatch: requested {material_id!r}, got {response_id!r}"
        )
    attributes = data.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("OPTIMADE structure lacks attributes")
    dimensions = attributes.get("dimension_types")
    if dimensions is not None and list(dimensions) != [1, 1, 1]:
        raise ValueError("Only fully three-dimensional OPTIMADE structures are supported")
    n_periodic = attributes.get("nperiodic_dimensions")
    if n_periodic is not None and int(n_periodic) != 3:
        raise ValueError("Only fully three-dimensional OPTIMADE structures are supported")

    lattice = attributes.get("lattice_vectors")
    positions = attributes.get("cartesian_site_positions")
    if not isinstance(lattice, list) or len(lattice) != 3:
        raise ValueError("OPTIMADE structure has invalid lattice_vectors")
    if not isinstance(positions, list) or not positions:
        raise ValueError("OPTIMADE structure lacks cartesian_site_positions")
    species = _ordered_species(attributes)
    if len(species) != len(positions):
        raise ValueError("OPTIMADE site positions and species have different lengths")
    structure = Structure(
        lattice,
        species,
        positions,
        coords_are_cartesian=True,
    )

    expected = Composition(expected_formula).reduced_composition
    declared_formula = str(attributes.get("chemical_formula_reduced") or "").strip()
    if declared_formula:
        try:
            declared = Composition(declared_formula).reduced_composition
        except Exception as exc:
            raise ValueError(
                f"OPTIMADE response has invalid chemical_formula_reduced {declared_formula!r}"
            ) from exc
        if declared != expected:
            raise ValueError(
                "OPTIMADE declared formula does not match the experimental record: "
                f"{declared.reduced_formula} != {expected.reduced_formula}"
            )
    if structure.composition.reduced_composition != expected:
        raise ValueError(
            "OPTIMADE structure composition does not match the experimental record: "
            f"{structure.composition.reduced_formula} != {expected.reduced_formula}"
        )
    return structure


def _decode_transport_result(value: bytes | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raw = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("OPTIMADE response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("OPTIMADE response root must be a JSON object")
    return payload


def fetch_optimade_structure(
    material_id: str,
    expected_formula: str,
    cache_dir: Path,
    *,
    base_url: str = "https://optimade.materialsproject.org/v1/structures",
    timeout: float = 30.0,
    transport: OptimadeTransport | None = None,
) -> CachedOptimadeStructure:
    """Fetch and immutably cache one exact OPTIMADE structure.

    Existing complete cache entries are validated and reused without calling
    ``transport``.  A partial cache is an error rather than an invitation to
    overwrite provenance.  Tests and offline callers can inject a deterministic
    transport callable; the default uses :mod:`urllib.request`.
    """

    identifier = str(material_id).strip()
    if not identifier or not _SAFE_OPTIMADE_ID_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe or invalid OPTIMADE material ID: {material_id!r}")
    try:
        expected_reduced = Composition(expected_formula).reduced_formula
    except Exception as exc:
        raise ValueError(f"Invalid expected formula: {expected_formula!r}") from exc
    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    response_path = cache_root / f"{identifier}.optimade.json"
    structure_path = cache_root / f"{identifier}.POSCAR"
    source_url = f"{base_url.rstrip('/')}/{quote(identifier, safe='')}"

    response_exists = response_path.exists()
    structure_exists = structure_path.exists()
    if response_exists != structure_exists:
        raise ValueError(
            f"Incomplete immutable OPTIMADE cache for {identifier}: expected both "
            f"{response_path.name} and {structure_path.name}"
        )

    if response_exists:
        try:
            envelope = json.loads(response_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid cached OPTIMADE JSON: {response_path}") from exc
        if not isinstance(envelope, Mapping) or not isinstance(
            envelope.get("response"), Mapping
        ):
            raise ValueError(f"Invalid cached OPTIMADE envelope: {response_path}")
        payload = dict(envelope["response"])
        payload_bytes = _canonical_json_bytes(payload)
        poscar_bytes = structure_path.read_bytes()
        if str(envelope.get("response_sha256") or "") != _sha256(payload_bytes):
            raise ValueError(f"Cached OPTIMADE response hash mismatch: {response_path}")
        if str(envelope.get("structure_sha256") or "") != _sha256(poscar_bytes):
            raise ValueError(f"Cached OPTIMADE POSCAR hash mismatch: {structure_path}")
        payload_structure = _structure_from_optimade_payload(
            payload,
            material_id=identifier,
            expected_formula=expected_formula,
        )
        cached_structure = Structure.from_file(str(structure_path))
        if (
            cached_structure.composition.reduced_composition
            != payload_structure.composition.reduced_composition
        ):
            raise ValueError("Cached OPTIMADE JSON and POSCAR compositions disagree")
        return CachedOptimadeStructure(
            material_id=identifier,
            reduced_formula=expected_reduced,
            source_url=str(envelope.get("source_url") or source_url),
            response_json_path=response_path,
            structure_path=structure_path,
            response_sha256=_sha256(payload_bytes),
            structure_sha256=_sha256(poscar_bytes),
            n_sites=len(cached_structure),
            from_cache=True,
        )

    if transport is None:

        def default_transport(url: str) -> bytes:
            request = Request(
                url,
                headers={
                    "Accept": "application/vnd.api+json, application/json",
                    "User-Agent": "dopingflow-calibration/0.1",
                },
            )
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read()

        transport = default_transport

    payload = _decode_transport_result(transport(source_url))
    structure = _structure_from_optimade_payload(
        payload,
        material_id=identifier,
        expected_formula=expected_formula,
    )
    payload_bytes = _canonical_json_bytes(payload)
    poscar_text = Poscar(
        structure,
        comment=f"{identifier} Materials Project OPTIMADE",
    ).get_str()
    poscar_bytes = poscar_text.encode("utf-8")
    envelope = {
        "schema_version": 1,
        "material_id": identifier,
        "expected_reduced_formula": expected_reduced,
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_sha256": _sha256(payload_bytes),
        "structure_sha256": _sha256(poscar_bytes),
        "response": payload,
    }
    # Exclusive creation is intentional: a cached scientific input is never
    # overwritten in place.  A race or partial write is surfaced for review.
    with structure_path.open("xb") as handle:
        handle.write(poscar_bytes)
    try:
        with response_path.open("x", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        # Leave the partial cache visible.  The next invocation reports it as
        # incomplete instead of silently replacing scientific provenance.
        raise
    return CachedOptimadeStructure(
        material_id=identifier,
        reduced_formula=expected_reduced,
        source_url=source_url,
        response_json_path=response_path,
        structure_path=structure_path,
        response_sha256=_sha256(payload_bytes),
        structure_sha256=_sha256(poscar_bytes),
        n_sites=len(structure),
        from_cache=False,
    )

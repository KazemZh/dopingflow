"""Phase-aware Materials Project structure fallback for correction calibration.

Only crystal structures and identity metadata are retrieved here.  No Materials
Project energies enter the correction fit: every accepted calibration structure
is still evaluated with the active dopingflow ML backend.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from dopingflow.corrections import ExperimentalRecord
from dopingflow.refs import _file_sha256

_PHASE_ALIASES = {
    "cubic": "cubic",
    "cub": "cubic",
    "hex": "hexagonal",
    "hexag": "hexagonal",
    "hexagonal": "hexagonal",
    "orth": "orthorhombic",
    "orthorh": "orthorhombic",
    "orthorhombic": "orthorhombic",
    "monocl": "monoclinic",
    "monoclinic": "monoclinic",
    "tetrag": "tetragonal",
    "tetragonal": "tetragonal",
    "rhomb": "trigonal",
    "rhombohedral": "trigonal",
    "trigonal": "trigonal",
    "triclinic": "triclinic",
}
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


def _phase_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _space_group_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def expected_crystal_system(phase: Any) -> str | None:
    """Map a coarse experimental phase label to a crystal system when possible."""

    return _PHASE_ALIASES.get(_phase_key(phase))


def phase_matches_structure(phase: Any, structure: Structure) -> bool:
    """Return whether a structure is compatible with a resolvable phase label.

    Generic labels such as ``solid`` or ``cr`` are intentionally not considered
    phase verification.  Coarse Kingsbury labels (``cubic``, ``tetrag``,
    ``orth``, ...) are checked against the structure-derived crystal system.
    Explicit Hermann-Mauguin space-group labels are checked against the detected
    space group.
    """

    phase_key = _phase_key(phase)
    if phase_key in _GENERIC_PHASE_KEYS:
        return False

    analyzer = SpacegroupAnalyzer(structure)
    expected_system = expected_crystal_system(phase)
    if expected_system is not None:
        return analyzer.get_crystal_system().lower() == expected_system

    text = str(phase or "").strip()
    try:
        SpaceGroup(text)
    except ValueError:
        return False
    return _space_group_key(analyzer.get_space_group_symbol()) == _space_group_key(text)


def phase_diagnostics(phase: Any, structure: Structure) -> dict[str, Any]:
    analyzer = SpacegroupAnalyzer(structure)
    return {
        "experimental_phase": str(phase or ""),
        "expected_crystal_system": expected_crystal_system(phase),
        "detected_crystal_system": analyzer.get_crystal_system(),
        "detected_space_group": analyzer.get_space_group_symbol(),
        "detected_space_group_number": analyzer.get_space_group_number(),
        "matches": phase_matches_structure(phase, structure),
    }


@dataclass(frozen=True)
class CachedMaterialsProjectStructure:
    material_id: str
    reduced_formula: str
    source: str
    structure_path: Path
    metadata_path: Path
    structure_sha256: str
    phase_diagnostics: Mapping[str, Any]
    from_cache: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["structure_path"] = str(self.structure_path)
        data["metadata_path"] = str(self.metadata_path)
        return data


def _safe_formula(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value))


def _cache_result(
    *,
    structure: Structure,
    material_id: str,
    record: ExperimentalRecord,
    source: str,
    output_dir: Path,
    query_metadata: Mapping[str, Any],
) -> CachedMaterialsProjectStructure:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _safe_formula(material_id)
    structure_path = (output_dir / f"{suffix}.mp-api.POSCAR").resolve()
    metadata_path = (output_dir / f"{suffix}.mp-api.json").resolve()

    structure.to(fmt="poscar", filename=str(structure_path))
    diagnostics = phase_diagnostics(record.phase, structure)
    payload = {
        "schema_version": 1,
        "provider": "Materials Project mp-api",
        "purpose": "correction_calibration_structure_only",
        "energies_used": False,
        "material_id": material_id,
        "experimental_formula": record.formula,
        "experimental_reduced_formula": record.reduced_formula,
        "experimental_likely_mpid": record.likely_mpid,
        "source": source,
        "structure_path": str(structure_path),
        "structure_sha256": _file_sha256(structure_path),
        "phase_diagnostics": diagnostics,
        "query_metadata": dict(query_metadata),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return CachedMaterialsProjectStructure(
        material_id=material_id,
        reduced_formula=record.reduced_formula,
        source=source,
        structure_path=structure_path,
        metadata_path=metadata_path,
        structure_sha256=payload["structure_sha256"],
        phase_diagnostics=diagnostics,
        from_cache=False,
    )


def _load_cache(
    metadata_path: Path,
    record: ExperimentalRecord,
) -> CachedMaterialsProjectStructure | None:
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    structure_path = Path(str(payload.get("structure_path") or ""))
    expected_hash = str(payload.get("structure_sha256") or "")
    if not structure_path.is_file() or not expected_hash:
        return None
    if _file_sha256(structure_path) != expected_hash:
        return None
    if str(payload.get("experimental_reduced_formula") or "") != record.reduced_formula:
        return None
    structure = Structure.from_file(str(structure_path))
    diagnostics = phase_diagnostics(record.phase, structure)
    if not diagnostics["matches"]:
        return None
    return CachedMaterialsProjectStructure(
        material_id=str(payload.get("material_id") or ""),
        reduced_formula=record.reduced_formula,
        source=str(payload.get("source") or "materials_project_api_cache"),
        structure_path=structure_path.resolve(),
        metadata_path=metadata_path.resolve(),
        structure_sha256=expected_hash,
        phase_diagnostics=diagnostics,
        from_cache=True,
    )


def _doc_value(doc: Any, name: str) -> Any:
    if isinstance(doc, Mapping):
        return doc.get(name)
    return getattr(doc, name, None)


def _doc_structure(doc: Any) -> Structure | None:
    value = _doc_value(doc, "structure")
    return value if isinstance(value, Structure) else None


def _doc_material_id(doc: Any) -> str:
    return str(_doc_value(doc, "material_id") or "").strip()


def _formula_matches(record: ExperimentalRecord, structure: Structure) -> bool:
    return structure.composition.reduced_formula == Composition(record.formula).reduced_formula


def select_unique_phase_compatible_document(
    docs: Sequence[Any],
    record: ExperimentalRecord,
) -> tuple[Any | None, tuple[str, ...]]:
    """Select exactly one formula- and phase-compatible MP document.

    The returned ID tuple lists every compatible candidate.  Multiple matches
    are intentionally left unresolved rather than silently selecting a
    polymorph by energy or another DFT-derived ranking.
    """

    compatible: list[Any] = []
    compatible_ids: list[str] = []
    for doc in docs:
        structure = _doc_structure(doc)
        material_id = _doc_material_id(doc)
        if structure is None or not material_id:
            continue
        if not _formula_matches(record, structure):
            continue
        if not phase_matches_structure(record.phase, structure):
            continue
        compatible.append(doc)
        compatible_ids.append(material_id)
    if len(compatible) == 1:
        return compatible[0], tuple(compatible_ids)
    return None, tuple(sorted(compatible_ids))


def fetch_materials_project_phase_structure(
    record: ExperimentalRecord,
    output_dir: Path,
) -> CachedMaterialsProjectStructure:
    """Fetch a phase-compatible structure using the official Materials Project API.

    Acquisition order:

    1. the curated Kingsbury ``likely_mpid`` through the MP summary endpoint;
    2. if that ID is unavailable or phase-incompatible, all MP structures with
       the same formula, accepting only a *unique* phase-compatible candidate.

    Only ``material_id`` and ``structure`` are requested.  No MP energies are
    read or used.
    """

    try:
        from mp_api.client import MPRester
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "mp-api is not installed; install dopingflow with the 'corrections' "
            "or 'mp' extra to enable phase-compatible Materials Project fallback"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    likely_id = str(record.likely_mpid or "").strip()
    if likely_id:
        cached = _load_cache(output_dir / f"{_safe_formula(likely_id)}.mp-api.json", record)
        if cached is not None:
            return cached

    # MPRester() follows the standard mp-api credential discovery, including
    # MP_API_KEY.  Do not put API keys into workflow snapshots or configuration.
    if not os.environ.get("MP_API_KEY"):
        credential_note = "MP_API_KEY is not set"
    else:
        credential_note = "MP_API_KEY available"

    try:
        with MPRester() as mpr:
            if likely_id:
                exact_docs = mpr.materials.summary.search(
                    material_ids=[likely_id],
                    fields=["material_id", "structure"],
                )
                exact_doc, exact_ids = select_unique_phase_compatible_document(
                    exact_docs,
                    record,
                )
                if exact_doc is not None:
                    structure = _doc_structure(exact_doc)
                    assert structure is not None
                    return _cache_result(
                        structure=structure,
                        material_id=_doc_material_id(exact_doc),
                        record=record,
                        source="materials_project_api_exact_id",
                        output_dir=output_dir,
                        query_metadata={
                            "query": "material_ids",
                            "requested_material_id": likely_id,
                            "compatible_ids": list(exact_ids),
                        },
                    )

            formula_docs = mpr.materials.summary.search(
                formula=record.reduced_formula,
                fields=["material_id", "structure"],
            )
            selected, compatible_ids = select_unique_phase_compatible_document(
                formula_docs,
                record,
            )
            if selected is None:
                if compatible_ids:
                    raise RuntimeError(
                        "Materials Project formula search found multiple phase-compatible "
                        f"structures for {record.reduced_formula}/{record.phase}: "
                        f"{list(compatible_ids)}; refusing an ambiguous polymorph choice"
                    )
                raise RuntimeError(
                    "Materials Project formula search found no phase-compatible structure "
                    f"for {record.reduced_formula}/{record.phase}"
                )
            structure = _doc_structure(selected)
            assert structure is not None
            material_id = _doc_material_id(selected)
            cached = _load_cache(
                output_dir / f"{_safe_formula(material_id)}.mp-api.json",
                record,
            )
            if cached is not None:
                return cached
            return _cache_result(
                structure=structure,
                material_id=material_id,
                record=record,
                source="materials_project_api_formula_phase_unique",
                output_dir=output_dir,
                query_metadata={
                    "query": "formula",
                    "formula": record.reduced_formula,
                    "compatible_ids": list(compatible_ids),
                    "original_likely_mpid": likely_id,
                },
            )
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("Materials Project"):
            raise
        raise RuntimeError(
            f"Materials Project mp-api structure acquisition failed ({credential_note}): {exc}"
        ) from exc

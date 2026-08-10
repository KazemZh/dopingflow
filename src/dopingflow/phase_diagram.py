from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Structure

from dopingflow.corrections import (
    CorrectionApplication,
    CorrectionModel,
    apply_energy_correction,
    combine_feature_vectors,
    evaluate_feature_vector,
    load_active_correction_model,
    parse_correction_config,
    validate_candidate_energy_provenance,
)

log = logging.getLogger(__name__)

REF_JSON = Path("reference_structures/reference_energies.json")
OUT_DB = "results_database.csv"
OUT_CSV = "phase_diagram_results.csv"
OUT_DIR = Path("phase_diagrams")
RELAX_META = Path("02_relax") / "meta.json"
RELAX_POSCAR = Path("02_relax") / "POSCAR"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _composition_from_poscar(path: Path) -> Composition:
    structure = Structure.from_file(str(path))
    return structure.composition


def _energy_from_relax_meta(path: Path) -> float:
    data = _load_json(path)

    if "energy_relaxed_eV" not in data:
        raise KeyError(f"{path} missing energy_relaxed_eV")

    return float(data["energy_relaxed_eV"])


def _host_entry_from_ref(ref: dict[str, Any]) -> PDEntry:
    host = ref.get("host", {}) or {}

    formula = str(host.get("name", "")).strip()
    if not formula:
        raise KeyError("reference JSON missing host.name")

    if "E_unit_total_eV" not in host or "n_atoms_unit" not in host:
        raise KeyError(
            "reference JSON missing host.E_unit_total_eV "
            "or host.n_atoms_unit"
        )

    comp_fu = Composition(formula).reduced_composition
    atoms_per_fu = comp_fu.num_atoms

    n_atoms_unit = int(host["n_atoms_unit"])
    n_fu = float(n_atoms_unit) / float(atoms_per_fu)

    energy_per_formula_unit = float(host["E_unit_total_eV"]) / n_fu

    return PDEntry(
        comp_fu,
        energy_per_formula_unit,
        name=formula,
        attribute={
            "entry_kind": "reference",
            "structure_path": host.get("relaxed_unit_poscar"),
        },
    )


def _reference_entries_from_ref(ref: dict[str, Any]) -> List[PDEntry]:
    entries: List[PDEntry] = []

    # Host oxide, for example SnO2.
    entries.append(_host_entry_from_ref(ref))

    refs = ref.get("references", {}) or {}

    for name, entry in refs.items():
        if not isinstance(entry, dict):
            continue

        ref_type = str(entry.get("type", "")).strip().lower()

        if ref_type == "gas":
            if "E_per_molecule_eV" in entry:
                energy = float(entry["E_per_molecule_eV"])
            elif "E_total_eV" in entry:
                energy = float(entry["E_total_eV"])
            else:
                continue

            entries.append(
                PDEntry(
                    Composition(str(name)),
                    energy,
                    name=str(name),
                    attribute={
                        "entry_kind": "reference",
                        "structure_path": entry.get("relaxed_poscar"),
                    },
                )
            )

        elif ref_type in {"metal", "oxide"}:
            if "E_per_formula_unit_eV" in entry:
                energy = float(entry["E_per_formula_unit_eV"])

            elif ref_type == "metal" and "E_per_atom_eV" in entry:
                energy = float(entry["E_per_atom_eV"])

            else:
                continue

            composition = Composition(str(name)).reduced_composition

            entries.append(
                PDEntry(
                    composition,
                    energy,
                    name=str(name),
                    attribute={
                        "entry_kind": "reference",
                        "structure_path": entry.get("relaxed_poscar"),
                    },
                )
            )

    return entries


def _candidate_entries_from_database(
    root: Path,
) -> List[tuple[str, Path, PDEntry]]:
    db_path = root / OUT_DB

    if not db_path.exists():
        raise FileNotFoundError(
            f"Missing {OUT_DB}. Run collect first."
        )

    output: List[tuple[str, Path, PDEntry]] = []

    with db_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            candidate_name = (row.get("candidate") or "").strip()
            candidate_path_string = (
                row.get("candidate_path") or ""
            ).strip()

            if not candidate_name or not candidate_path_string:
                continue

            candidate_dir = Path(candidate_path_string)

            poscar = candidate_dir / RELAX_POSCAR
            meta = candidate_dir / RELAX_META

            if not poscar.exists() or not meta.exists():
                continue

            composition = _composition_from_poscar(poscar)
            metadata = _load_json(meta)
            energy = _energy_from_relax_meta(meta)

            entry_name = (
                f"{candidate_dir.parent.name}/{candidate_name}"
            )

            entry = PDEntry(
                composition,
                energy,
                name=entry_name,
                attribute={
                    "entry_kind": "candidate",
                    "structure_path": str(poscar.resolve()),
                    "metadata_path": str(meta.resolve()),
                    "backend": metadata.get("backend"),
                    "model": metadata.get("model"),
                    "task": metadata.get("task"),
                    "backend_package": metadata.get("backend_package"),
                    "backend_package_version": metadata.get(
                        "backend_package_version"
                    ),
                    "model_checkpoint_sha256": metadata.get(
                        "model_checkpoint_sha256"
                    ),
                    "optimizer": metadata.get("optimizer"),
                    "fmax_target_eV_per_A": metadata.get(
                        "fmax_target_eV_per_A"
                    ),
                    "max_steps": metadata.get("max_steps"),
                    "converged": metadata.get("converged"),
                    "relaxed_poscar_sha256": metadata.get(
                        "relaxed_poscar_sha256"
                    ),
                    "device": metadata.get("device"),
                    "gpu_id": metadata.get("gpu_id"),
                },
            )

            output.append(
                (entry_name, candidate_dir, entry)
            )

    return output


def _element_set(entry: PDEntry) -> frozenset[str]:
    return frozenset(
        str(element)
        for element in entry.composition.elements
    )


def _system_label(elements: frozenset[str]) -> str:
    return "-".join(sorted(elements))


def _system_filename(elements: frozenset[str]) -> str:
    return f"phase_diagram_{_system_label(elements)}.csv"


def _decomposition_to_string(
    decomposition: Dict[PDEntry, float],
) -> str:
    parts = []

    for entry, amount in decomposition.items():
        name = getattr(entry, "name", None)
        phase_name = name or entry.composition.reduced_formula

        parts.append(f"{amount:.6g} {phase_name}")

    return " + ".join(parts)


def _validate_terminal_references(
    system_elements: frozenset[str],
    reference_entries: List[PDEntry],
) -> None:
    """
    A closed PhaseDiagram requires one elemental terminal entry
    for each element of the chemical system.

    O2 is valid as the oxygen elemental terminal entry because it
    contains only oxygen.
    """
    available_terminal_elements = {
        next(iter(_element_set(entry)))
        for entry in reference_entries
        if len(_element_set(entry)) == 1
    }

    missing = sorted(
        system_elements - available_terminal_elements
    )

    if missing:
        raise ValueError(
            "Cannot build phase diagram for chemical system "
            f"{_system_label(system_elements)}. "
            "Missing elemental terminal reference(s): "
            f"{missing}. "
            "Add the corresponding elemental POSCAR files under "
            "reference_structures/metals/, add the elements to "
            "[references].metal_ref in input.toml, and rerun "
            "`dopingflow refs-build`."
        )


def _validate_entry_backend(
    entry: PDEntry,
    model: CorrectionModel,
    *,
    allow_legacy: bool = False,
) -> dict[str, Any] | None:
    attribute = entry.attribute if isinstance(entry.attribute, dict) else {}
    if attribute.get("entry_kind") != "candidate":
        return None
    structure_path = Path(str(attribute.get("structure_path") or ""))
    return validate_candidate_energy_provenance(
        attribute,
        structure_path,
        model,
        label=f"phase-diagram entry {entry.name!r}",
        allow_legacy=allow_legacy,
    )


def _correct_entry(
    entry: PDEntry,
    model: CorrectionModel,
    *,
    allow_legacy: bool = False,
) -> tuple[PDEntry, CorrectionApplication, dict[str, Any] | None]:
    energy_provenance = _validate_entry_backend(
        entry,
        model,
        allow_legacy=allow_legacy,
    )
    attribute = entry.attribute if isinstance(entry.attribute, dict) else {}
    structure_path_text = str(attribute.get("structure_path") or "").strip()
    structure_path = Path(structure_path_text) if structure_path_text else None
    if len(entry.composition.elements) > 1 and structure_path is None:
        raise ValueError(
            f"Non-elemental phase-diagram entry {entry.name!r} lacks a structure; "
            "a complete corrected hull cannot be built"
        )
    application = apply_energy_correction(
        model,
        entry.composition,
        structure_path=structure_path,
    )
    corrected = PDEntry(
        entry.composition,
        float(entry.energy) + application.correction_eV,
        name=entry.name,
        attribute=entry.attribute,
    )
    return corrected, application, energy_provenance


def _phase_output_has_corrections(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
    except (OSError, csv.Error):
        return False
    exact_fields = {
        "energy_raw_eV",
        "stable_raw",
        "stable_corrected",
        "decomposition_raw",
        "decomposition_corrected",
        "experimental_dataset",
    }
    correction_prefixes = (
        "correction_",
        "energy_correction_",
        "energy_corrected_",
        "energy_above_hull_raw_",
        "energy_above_hull_corrected_",
        "energy_above_hull_correction_",
    )
    return any(
        field in exact_fields or field.startswith(correction_prefixes)
        for field in fieldnames
    )


def _combined_system_output_names(path: Path) -> set[str] | None:
    """Return per-system filenames represented by a completed combined CSV."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "chemical_system" not in (reader.fieldnames or []):
                return None
            return {
                f"phase_diagram_{label}.csv"
                for row in reader
                if (label := str(row.get("chemical_system") or "").strip())
            }
    except (OSError, csv.Error):
        return None


def _remove_obsolete_system_outputs(
    output_dir: Path,
    expected_names: set[str],
) -> None:
    if not output_dir.is_dir():
        return
    for previous_system_csv in output_dir.glob("phase_diagram_*.csv"):
        if previous_system_csv.name not in expected_names:
            previous_system_csv.unlink()
            log.info(
                "Removed obsolete phase-diagram output: %s",
                previous_system_csv,
            )


def _energy_above_hull_correction_application(
    *,
    candidate_entry: PDEntry,
    candidate_application: CorrectionApplication,
    corrected_decomposition: Dict[PDEntry, float],
    application_by_corrected_id: dict[int, CorrectionApplication],
    model: CorrectionModel,
) -> CorrectionApplication:
    """Linearized correction uncertainty relative to the corrected hull facet.

    Pymatgen decomposition amounts multiply fractional (per-atom)
    compositions.  Each entry feature vector is therefore divided by that
    entry's atom count before the decomposition weights are applied.  Combining
    the vectors before evaluating the model preserves covariance and exact
    cancellation of shared fitted parameters.
    """

    candidate_atoms = float(candidate_entry.composition.num_atoms)
    components: list[tuple[float, tuple[float, ...]]] = [
        (1.0 / candidate_atoms, candidate_application.feature_vector)
    ]
    for decomposition_entry, amount in corrected_decomposition.items():
        try:
            decomposition_application = application_by_corrected_id[
                id(decomposition_entry)
            ]
        except KeyError as exc:
            raise ValueError(
                "Corrected hull decomposition contains an entry without a "
                "correction feature vector"
            ) from exc
        decomposition_atoms = float(decomposition_entry.composition.num_atoms)
        components.append(
            (
                -float(amount) / decomposition_atoms,
                decomposition_application.feature_vector,
            )
        )

    q_vector_per_atom = combine_feature_vectors(
        model.correction_terms,
        components,
    )
    return evaluate_feature_vector(
        model,
        q_vector_per_atom,
        reason="candidate_minus_corrected_decomposition_per_atom",
    )


def _write_csv(
    path: Path,
    rows: List[dict[str, Any]],
) -> None:
    fieldnames = [
        "chemical_system",
        "candidate",
        "composition_tag",
        "candidate_path",
        "formula",
        "energy_total_eV",
        "energy_per_atom_eV",
        "energy_above_hull_eV_per_atom",
        "stable",
        "decomposition",
    ]
    if any("energy_corrected_eV" in row for row in rows):
        fieldnames.extend(
            [
                "energy_raw_eV",
                "energy_correction_eV",
                "correction_uncertainty_eV",
                "energy_corrected_eV",
                "energy_corrected_per_atom_eV",
                "energy_above_hull_raw_eV_per_atom",
                "energy_above_hull_corrected_eV_per_atom",
                "energy_above_hull_correction_eV_per_atom",
                "energy_above_hull_parameter_shift_eV_per_atom",
                "energy_above_hull_correction_uncertainty_eV_per_atom",
                "energy_above_hull_correction_q_vector_per_atom_json",
                "energy_above_hull_correction_terms_json",
                "energy_above_hull_correction_provenance",
                "stable_raw",
                "stable_corrected",
                "decomposition_raw",
                "decomposition_corrected",
                "correction_applied",
                "correction_reason",
                "correction_method",
                "correction_fit_id",
                "correction_parameter_set",
                "correction_model_family",
                "correction_selection_run_hash",
                "experimental_dataset",
                "correction_backend",
                "correction_model",
                "correction_task",
                "correction_backend_package",
                "correction_backend_package_version",
                "correction_model_checkpoint_sha256",
                "candidate_energy_provenance_mode",
                "candidate_energy_provenance_assumptions_json",
                "candidate_energy_execution_differences_json",
                "candidate_energy_current_poscar_sha256",
                "candidate_energy_original_poscar_sha256",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def run_phase_diagram(
    raw_cfg: dict[str, Any],
    root: Path,
    *,
    config_path: Path | None = None,
) -> Path:
    """
    Build one phase diagram per exact candidate chemical system.

    Examples:
      Sn-Sb-O
      Ce-Sn-Sb-O
      In-Sn-Sb-O

    For each system, the hull contains:
      - reference phases whose elements are a subset of that system;
      - all relaxed candidate structures whose elements are a subset
        of that system.

    The output contains one CSV per chemical system plus a combined
    phase_diagram_results.csv file.
    """
    phase_config = raw_cfg.get("phase_diagram", {}) or {}
    skip_if_done = bool(phase_config.get("skip_if_done", True))
    stable_threshold = float(
        phase_config.get("stable_threshold_eV_per_atom", 1.0e-8)
    )
    if stable_threshold < 0.0:
        raise ValueError(
            "[phase_diagram].stable_threshold_eV_per_atom must be non-negative"
        )

    out_csv = root / OUT_CSV
    correction_config = parse_correction_config(raw_cfg, root)
    if skip_if_done and out_csv.exists() and not correction_config.enabled:
        if not _phase_output_has_corrections(out_csv):
            cached_system_files = _combined_system_output_names(out_csv)
            if cached_system_files is not None:
                _remove_obsolete_system_outputs(
                    root / OUT_DIR,
                    cached_system_files,
                )
            log.info("SKIP phase diagram: %s already exists", out_csv)
            return out_csv
        log.info(
            "REBUILD phase diagram: correction is disabled but %s contains "
            "corrected columns",
            out_csv,
        )

    ref_path = root / REF_JSON

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Missing reference JSON: {ref_path}"
        )

    ref = _load_json(ref_path)
    correction_model = load_active_correction_model(raw_cfg, root, ref)

    # A fit ID fingerprints fitted parameters, not candidate energies,
    # structures, database membership, references, or the stability threshold.
    # Corrected hulls are therefore always rebuilt from their current entries.
    if out_csv.exists() and correction_model is not None:
        log.info(
            "REBUILD phase diagram: correction fit %s is enabled and corrected "
            "hulls are input-sensitive",
            correction_model.fit_id,
        )

    reference_entries = _reference_entries_from_ref(ref)
    candidate_entries = _candidate_entries_from_database(root)

    if not candidate_entries:
        raise ValueError(
            "No candidate entries found in results_database.csv"
        )

    # Each unique exact chemical system represented by candidates.
    # Example:
    #   {"Sn", "Sb", "O"}
    #   {"Ce", "Sn", "Sb", "O"}
    #   {"In", "Sn", "Sb", "O"}
    chemical_systems = sorted(
        {
            _element_set(entry)
            for _, _, entry in candidate_entries
        },
        key=lambda system: (len(system), _system_label(system)),
    )

    output_dir = root / OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[dict[str, Any]] = []

    for system_elements in chemical_systems:
        system_name = _system_label(system_elements)

        # Reference phases allowed in this specific chemical system.
        system_references = [
            entry
            for entry in reference_entries
            if _element_set(entry).issubset(system_elements)
        ]

        _validate_terminal_references(
            system_elements,
            system_references,
        )

        # Entries used to build this hull:
        # exact-system candidates plus lower-dimensional candidates.
        hull_candidates = [
            item
            for item in candidate_entries
            if _element_set(item[2]).issubset(system_elements)
        ]

        # Only candidates whose exact chemical system matches this
        # system are written to this system's output CSV.
        evaluation_candidates = [
            item
            for item in candidate_entries
            if _element_set(item[2]) == system_elements
        ]

        all_entries = (
            system_references
            + [entry for _, _, entry in hull_candidates]
        )

        log.info(
            "Phase diagram %s: %d reference entries, "
            "%d hull candidate entries, %d evaluated candidates",
            system_name,
            len(system_references),
            len(hull_candidates),
            len(evaluation_candidates),
        )

        phase_diagram = PhaseDiagram(all_entries)

        corrected_phase_diagram: PhaseDiagram | None = None
        corrected_by_raw_id: dict[int, PDEntry] = {}
        application_by_raw_id: dict[int, CorrectionApplication] = {}
        energy_provenance_by_raw_id: dict[int, dict[str, Any] | None] = {}
        application_by_corrected_id: dict[int, CorrectionApplication] = {}
        if correction_model is not None:
            corrected_entries: List[PDEntry] = []
            try:
                for raw_entry in all_entries:
                    corrected_entry, application, energy_provenance = _correct_entry(
                        raw_entry,
                        correction_model,
                        allow_legacy=(
                            correction_config.allow_legacy_candidate_provenance
                        ),
                    )
                    corrected_entries.append(corrected_entry)
                    corrected_by_raw_id[id(raw_entry)] = corrected_entry
                    application_by_raw_id[id(raw_entry)] = application
                    energy_provenance_by_raw_id[id(raw_entry)] = energy_provenance
                    application_by_corrected_id[id(corrected_entry)] = application
            except (ValueError, FileNotFoundError) as exc:
                raise ValueError(
                    f"Cannot build a complete corrected phase diagram for {system_name}: "
                    f"{exc}"
                ) from exc
            corrected_phase_diagram = PhaseDiagram(corrected_entries)

        system_rows: List[dict[str, Any]] = []

        for _, candidate_dir, entry in evaluation_candidates:
            energy_above_hull = float(
                phase_diagram.get_e_above_hull(entry)
            )

            decomposition = phase_diagram.get_decomposition(
                entry.composition
            )

            row = {
                "chemical_system": system_name,
                "candidate": candidate_dir.name,
                "composition_tag": candidate_dir.parent.name,
                "candidate_path": str(candidate_dir.resolve()),
                "formula": entry.composition.reduced_formula,
                "energy_total_eV": float(entry.energy),
                "energy_per_atom_eV": float(
                    entry.energy_per_atom
                ),
                "energy_above_hull_eV_per_atom": energy_above_hull,
                "stable": energy_above_hull <= stable_threshold,
                "decomposition": _decomposition_to_string(
                    decomposition
                ),
            }

            if corrected_phase_diagram is not None:
                corrected_entry = corrected_by_raw_id[id(entry)]
                application = application_by_raw_id[id(entry)]
                candidate_energy_provenance = energy_provenance_by_raw_id[id(entry)]
                corrected_energy_above_hull = float(
                    corrected_phase_diagram.get_e_above_hull(corrected_entry)
                )
                corrected_decomposition = corrected_phase_diagram.get_decomposition(
                    corrected_entry.composition
                )
                assert correction_model is not None
                hull_correction_application = (
                    _energy_above_hull_correction_application(
                        candidate_entry=corrected_entry,
                        candidate_application=application,
                        corrected_decomposition=corrected_decomposition,
                        application_by_corrected_id=application_by_corrected_id,
                        model=correction_model,
                    )
                )
                row.update(
                    {
                        "energy_raw_eV": float(entry.energy),
                        "energy_correction_eV": application.correction_eV,
                        "correction_uncertainty_eV": application.uncertainty_eV,
                        "energy_corrected_eV": float(corrected_entry.energy),
                        "energy_corrected_per_atom_eV": float(
                            corrected_entry.energy_per_atom
                        ),
                        "energy_above_hull_raw_eV_per_atom": energy_above_hull,
                        "energy_above_hull_corrected_eV_per_atom": (
                            corrected_energy_above_hull
                        ),
                        "energy_above_hull_correction_eV_per_atom": (
                            corrected_energy_above_hull - energy_above_hull
                        ),
                        "energy_above_hull_parameter_shift_eV_per_atom": (
                            hull_correction_application.correction_eV
                        ),
                        "energy_above_hull_correction_uncertainty_eV_per_atom": (
                            hull_correction_application.uncertainty_eV
                        ),
                        "energy_above_hull_correction_q_vector_per_atom_json": (
                            json.dumps(
                                list(hull_correction_application.feature_vector)
                            )
                        ),
                        "energy_above_hull_correction_terms_json": json.dumps(
                            list(correction_model.correction_terms)
                        ),
                        "energy_above_hull_correction_provenance": (
                            "candidate_minus_corrected_decomposition; "
                            "uncertainty=sqrt(q^T covariance q); "
                            "fixed_corrected_decomposition_linearization; "
                            "reported_correction=corrected_e_hull-raw_e_hull; "
                            "parameter_shift=q^T beta"
                        ),
                        "stable_raw": energy_above_hull <= stable_threshold,
                        "stable_corrected": (
                            corrected_energy_above_hull <= stable_threshold
                        ),
                        "decomposition_raw": _decomposition_to_string(decomposition),
                        "decomposition_corrected": _decomposition_to_string(
                            corrected_decomposition
                        ),
                        "correction_applied": application.applied,
                        "correction_reason": application.reason,
                        "correction_method": correction_model.method,
                        "correction_fit_id": correction_model.fit_id,
                        "correction_parameter_set": correction_model.fit_id,
                        "correction_model_family": correction_model.model_family,
                        "correction_selection_run_hash": (
                            correction_model.selection_run_hash
                        ),
                        "experimental_dataset": correction_model.experimental_dataset,
                        "correction_backend": correction_model.backend_signature.get(
                            "backend"
                        ),
                        "correction_model": correction_model.backend_signature.get(
                            "model"
                        ),
                        "correction_task": correction_model.backend_signature.get("task"),
                        "correction_backend_package": (
                            correction_model.backend_signature.get("backend_package")
                        ),
                        "correction_backend_package_version": (
                            correction_model.backend_signature.get(
                                "backend_package_version"
                            )
                        ),
                        "correction_model_checkpoint_sha256": (
                            correction_model.backend_signature.get(
                                "model_checkpoint_sha256"
                            )
                        ),
                        "candidate_energy_provenance_mode": (
                            (candidate_energy_provenance or {}).get("mode", "")
                        ),
                        "candidate_energy_provenance_assumptions_json": json.dumps(
                            (candidate_energy_provenance or {}).get("assumptions", [])
                        ),
                        "candidate_energy_execution_differences_json": json.dumps(
                            (candidate_energy_provenance or {}).get(
                                "execution_differences", []
                            )
                        ),
                        "candidate_energy_current_poscar_sha256": (
                            (candidate_energy_provenance or {}).get(
                                "current_relaxed_poscar_sha256", ""
                            )
                        ),
                        "candidate_energy_original_poscar_sha256": (
                            (candidate_energy_provenance or {}).get(
                                "original_relaxed_poscar_sha256", ""
                            )
                        ),
                    }
                )

            system_rows.append(row)

        system_rows.sort(
            key=lambda row: row[
                "energy_above_hull_eV_per_atom"
            ]
        )

        system_csv = output_dir / _system_filename(
            system_elements
        )

        _write_csv(system_csv, system_rows)

        log.info(
            "DONE phase diagram %s: wrote %s",
            system_name,
            system_csv,
        )

        combined_rows.extend(system_rows)

    combined_rows.sort(
        key=lambda row: (
            row["chemical_system"],
            row["energy_above_hull_eV_per_atom"],
        )
    )

    _write_csv(out_csv, combined_rows)

    expected_system_files = {
        _system_filename(system_elements)
        for system_elements in chemical_systems
    }
    _remove_obsolete_system_outputs(output_dir, expected_system_files)

    log.info(
        "DONE phase diagram: wrote combined results to %s",
        out_csv,
    )

    return out_csv


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(
        path.read_text(encoding="utf-8")
    )


def run_phase_diagram_from_toml(
    config_path: Path,
) -> Path:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent

    return run_phase_diagram(
        raw,
        root,
        config_path=config_path,
    )

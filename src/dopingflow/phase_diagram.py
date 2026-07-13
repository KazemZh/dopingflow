from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Structure

log = logging.getLogger(__name__)

REF_JSON = Path("reference_structures/reference_energies.json")
OUT_DB = "results_database.csv"
OUT_CSV = "phase_diagram_results.csv"
OUT_DIR = Path("phase_diagrams")

RELAX_META = Path("02_relax") / "meta.json"
RELAX_POSCAR = Path("02_relax") / "POSCAR"


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

    return PDEntry(comp_fu, energy_per_formula_unit, name=formula)


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
            energy = _energy_from_relax_meta(meta)

            entry_name = (
                f"{candidate_dir.parent.name}/{candidate_name}"
            )

            entry = PDEntry(
                composition,
                energy,
                name=entry_name,
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
    if skip_if_done and out_csv.exists():
        log.info("SKIP phase diagram: %s already exists", out_csv)
        return out_csv

    ref_path = root / REF_JSON

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Missing reference JSON: {ref_path}"
        )

    ref = _load_json(ref_path)

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

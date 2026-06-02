from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core import Composition, Structure

log = logging.getLogger(__name__)

REF_JSON = Path("reference_structures/reference_energies.json")
OUT_DB = "results_database.csv"
OUT_CSV = "phase_diagram_results.csv"

RELAX_META = Path("02_relax") / "meta.json"
RELAX_POSCAR = Path("02_relax") / "POSCAR"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _composition_from_poscar(path: Path) -> Composition:
    structure = Structure.from_file(str(path))
    return structure.composition


def _energy_from_relax_meta(path: Path) -> float:
    d = _load_json(path)
    if "energy_relaxed_eV" not in d:
        raise KeyError(f"{path} missing energy_relaxed_eV")
    return float(d["energy_relaxed_eV"])


def _host_entry_from_ref(ref: dict[str, Any]) -> PDEntry:
    host = ref.get("host", {}) or {}

    formula = str(host.get("name", "")).strip()
    if not formula:
        raise KeyError("reference JSON missing host.name")

    if "E_unit_total_eV" not in host or "n_atoms_unit" not in host:
        raise KeyError("reference JSON missing host.E_unit_total_eV or host.n_atoms_unit")

    comp_fu = Composition(formula).reduced_composition
    atoms_per_fu = comp_fu.num_atoms

    n_atoms_unit = int(host["n_atoms_unit"])
    n_fu = float(n_atoms_unit) / float(atoms_per_fu)

    E_fu = float(host["E_unit_total_eV"]) / n_fu

    return PDEntry(comp_fu, E_fu, name=formula)


def _reference_entries_from_ref(ref: dict[str, Any]) -> List[PDEntry]:
    entries: List[PDEntry] = []

    # host oxide, e.g. SnO2
    entries.append(_host_entry_from_ref(ref))

    refs = ref.get("references", {}) or {}

    for name, entry in refs.items():
        if not isinstance(entry, dict):
            continue

        ref_type = entry.get("type", "")

        if ref_type == "gas":
            # For O2, use one molecule as Composition("O2")
            if "E_per_molecule_eV" in entry:
                E = float(entry["E_per_molecule_eV"])
            elif "E_total_eV" in entry:
                E = float(entry["E_total_eV"])
            else:
                continue

            entries.append(PDEntry(Composition(str(name)), E, name=str(name)))

        elif ref_type in {"metal", "oxide"}:
            if "E_per_formula_unit_eV" in entry:
                E = float(entry["E_per_formula_unit_eV"])
                comp = Composition(str(name)).reduced_composition
                entries.append(PDEntry(comp, E, name=str(name)))

            elif "E_per_atom_eV" in entry and ref_type == "metal":
                # metal elemental reference
                E = float(entry["E_per_atom_eV"])
                comp = Composition(str(name))
                entries.append(PDEntry(comp, E, name=str(name)))

    return entries


def _candidate_entries_from_database(root: Path) -> List[tuple[str, Path, PDEntry]]:
    db_path = root / OUT_DB
    if not db_path.exists():
        raise FileNotFoundError(f"Missing {OUT_DB}. Run collect first.")

    out: List[tuple[str, Path, PDEntry]] = []

    with db_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cand = (row.get("candidate") or "").strip()
            cand_path_str = (row.get("candidate_path") or "").strip()

            if not cand or not cand_path_str:
                continue

            cand_dir = Path(cand_path_str)
            poscar = cand_dir / RELAX_POSCAR
            meta = cand_dir / RELAX_META

            if not poscar.exists() or not meta.exists():
                continue

            comp = _composition_from_poscar(poscar)
            E = _energy_from_relax_meta(meta)

            entry_name = f"{cand_dir.parent.name}/{cand}"
            entry = PDEntry(comp, E, name=entry_name)

            out.append((entry_name, cand_dir, entry))

    return out


def _decomposition_to_string(decomp: Dict[PDEntry, float]) -> str:
    parts = []
    for entry, amount in decomp.items():
        name = getattr(entry, "name", None) or entry.composition.reduced_formula
        parts.append(f"{amount:.6g} {name}")
    return " + ".join(parts)


def run_phase_diagram(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> Path:
    """
    Build full Sn-Sb-O phase diagram from reference phases + doped candidates.

    Outputs:
      phase_diagram_results.csv

    Main quantity:
      energy_above_hull_eV_per_atom
    """
    ref_path = root / REF_JSON
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference JSON: {ref_path}")

    ref = _load_json(ref_path)

    ref_entries = _reference_entries_from_ref(ref)
    cand_entries = _candidate_entries_from_database(root)

    if not cand_entries:
        raise ValueError("No candidate entries found in results_database.csv")

    all_entries = ref_entries + [x[2] for x in cand_entries]

    log.info("Phase diagram: %d reference entries", len(ref_entries))
    log.info("Phase diagram: %d candidate entries", len(cand_entries))

    pd = PhaseDiagram(all_entries)

    out_csv = root / OUT_CSV

    rows = []
    for name, cand_dir, entry in cand_entries:
        e_above = float(pd.get_e_above_hull(entry))
        decomp = pd.get_decomposition(entry.composition)

        rows.append(
            {
                "candidate": cand_dir.name,
                "composition_tag": cand_dir.parent.name,
                "candidate_path": str(cand_dir.resolve()),
                "formula": entry.composition.reduced_formula,
                "energy_total_eV": float(entry.energy),
                "energy_per_atom_eV": float(entry.energy_per_atom),
                "energy_above_hull_eV_per_atom": e_above,
                "stable": e_above <= 1e-8,
                "decomposition": _decomposition_to_string(decomp),
            }
        )

    rows.sort(key=lambda r: r["energy_above_hull_eV_per_atom"])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    log.info("DONE phase diagram: wrote %s", out_csv)
    return out_csv


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_phase_diagram_from_toml(config_path: Path) -> Path:
    raw = _load_raw_toml(config_path)
    root = config_path.resolve().parent
    return run_phase_diagram(raw, root, config_path=config_path)
# src/dopingflow/formation.py
from __future__ import annotations

import csv
import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

log = logging.getLogger(__name__)

REF_JSON = Path("reference_structures/reference_energies.json")
CAND_LIST = "selected_candidates.txt"
RELAX_META = "02_relax/meta.json"
RELAX_POSCAR = "02_relax/POSCAR"
OUT_CSV = "formation_energies.csv"


@dataclass(frozen=True)
class FormationConfig:
    outdir: Path
    host_species: str
    anion_species: List[str]
    skip_if_done: bool
    normalize: str
    relative_enabled: bool
    relative_endpoint_x: float | None


@dataclass
class CandidateRecord:
    folder: Path
    candidate_dir: Path
    candidate: str
    E_doped: float
    counts: Dict[str, int]
    dopant_counts: Dict[str, int]
    x_dopant: float
    reference_results: Dict[str, Dict[str, Any]]


def _parse_formation_config(raw: dict[str, Any], root: Path) -> FormationConfig:
    st = raw.get("structure", {}) or {}
    dop = raw.get("doping", {}) or {}
    scan = raw.get("scan", {}) or {}
    form = raw.get("formation", {}) or {}

    outdir = (root / str(st.get("outdir", "random_structures"))).resolve()
    host_species = str(dop.get("host_species", "")).strip()
    if not host_species:
        raise ValueError("[doping].host_species is required")

    anion_species = [str(x) for x in (scan.get("anion_species", ["O"]) or [])]
    if not anion_species:
        raise ValueError("[scan].anion_species must be non-empty")

    normalize = str(form.get("normalize", "per_dopant")).strip().lower()
    if normalize not in {"total", "per_dopant", "per_host"}:
        raise ValueError("[formation].normalize must be one of: total, per_dopant, per_host")

    endpoint_raw = form.get("endpoint_x", "auto")
    if endpoint_raw is None or str(endpoint_raw).strip().lower() in {"", "auto"}:
        endpoint_x = None
    else:
        endpoint_x = float(endpoint_raw)
        if not 0.0 < endpoint_x <= 1.0:
            raise ValueError("[formation].endpoint_x must be in (0, 1] or 'auto'")

    return FormationConfig(
        outdir=outdir,
        host_species=host_species,
        anion_species=anion_species,
        skip_if_done=bool(form.get("skip_if_done", True)),
        normalize=normalize,
        relative_enabled=bool(form.get("relative_enabled", False)),
        relative_endpoint_x=endpoint_x,
    )


def _load_ref_json(root: Path) -> dict[str, Any]:
    path = (root / REF_JSON).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing reference file: {path}\n"
            "Run Step 00 first: dopingflow refs-build -c input.toml"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _get_pristine_energy_and_natoms(ref: dict[str, Any]) -> tuple[float, int]:
    if isinstance(ref.get("pristine"), dict):
        p = ref["pristine"]
        return float(p["E_pristine_eV"]), int(p["n_atoms_supercell"])
    if isinstance(ref.get("host"), dict):
        h = ref["host"]
        return float(h["E_supercell_total_eV"]), int(h["n_atoms_supercell"])
    raise KeyError("reference_energies.json missing 'host' or 'pristine' block.")


def _read_selected_candidates(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _get_candidate_poscars(folder: Path) -> List[Path]:
    selected = folder / CAND_LIST
    if selected.exists():
        poscars = [folder / name / RELAX_POSCAR for name in _read_selected_candidates(selected)]
        poscars = [path for path in poscars if path.exists()]
        log.info("SELECT %s: using %d candidates from %s", folder.name, len(poscars), CAND_LIST)
        return poscars

    poscars = sorted(folder.glob(f"candidate_*/{RELAX_POSCAR}"))
    log.info("SELECT %s: using glob: %d candidates", folder.name, len(poscars))
    return poscars


def _load_relax_energy(meta_path: Path) -> float:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if "energy_relaxed_eV" not in data:
        raise KeyError(f"{meta_path} missing 'energy_relaxed_eV'")
    return float(data["energy_relaxed_eV"])


def _count_species_from_poscar(poscar_path: Path) -> Dict[str, int]:
    from pymatgen.core import Structure

    structure = Structure.from_file(str(poscar_path))
    counts: Dict[str, int] = {}
    for site in structure:
        element = site.species_string
        counts[element] = counts.get(element, 0) + 1
    return counts


def _compute_substitution_dopant_counts(
    counts_doped: Dict[str, int],
    host: str,
    anions: List[str],
) -> Dict[str, int]:
    return {
        str(element): int(count)
        for element, count in counts_doped.items()
        if element != host and element not in anions
    }


def _formula_unit_energy_from_host(ref: dict[str, Any], host_formula: str) -> dict[str, Any]:
    from pymatgen.core.composition import Composition

    host = ref.get("host", {}) or {}
    if "E_unit_total_eV" not in host or "n_atoms_unit" not in host:
        raise KeyError("reference JSON missing host.E_unit_total_eV or host.n_atoms_unit")

    composition = Composition(host_formula).reduced_composition.as_dict()
    atoms_per_fu = sum(float(v) for v in composition.values())
    n_fu = float(host["n_atoms_unit"]) / atoms_per_fu
    if n_fu <= 0.0:
        raise ValueError("Could not determine number of host formula units.")

    return {
        "formula": host_formula,
        "reduced_composition": composition,
        "E_per_formula_unit_eV": float(host["E_unit_total_eV"]) / n_fu,
    }


def _oxide_reference_groups(
    ref: dict[str, Any],
    *,
    anion: str,
) -> Dict[str, List[tuple[str, Dict[str, Any]]]]:
    """Group every binary oxide in the reference cache by its cation element."""
    groups: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    refs = ref.get("references", {}) or {}

    for name, entry in refs.items():
        if not isinstance(entry, dict) or entry.get("type") != "oxide":
            continue
        composition = entry.get("reduced_composition")
        if not isinstance(composition, dict) or anion not in composition:
            continue
        cations = [str(element) for element in composition if element != anion]
        if len(cations) != 1:
            log.warning("Ignoring non-binary oxide reference %s: %s", name, composition)
            continue
        if "E_per_formula_unit_eV" not in entry:
            log.warning("Ignoring oxide reference %s without E_per_formula_unit_eV", name)
            continue
        groups.setdefault(cations[0], []).append((str(name), entry))

    for element in groups:
        groups[element].sort(key=lambda pair: pair[0])
    return groups


def _oxygen_mu(ref: dict[str, Any]) -> tuple[float, float]:
    """Return shifted oxygen chemical potential and shifted O2 molecule energy.

    If [oxide_mode].muO_shift_ev is defined, it is a per-O-atom shift.
    Therefore the effective molecule energy used in any O2 term is

        E_O2_effective = E_O2_raw + 2 * muO_shift_ev
    """
    oxide_mode = ref.get("oxide_mode", {}) or {}
    refs = ref.get("references", {}) or {}
    gas_ref = str(oxide_mode.get("gas_ref", "O2")).strip()
    gas = refs.get(gas_ref)
    if not isinstance(gas, dict):
        raise KeyError(f"oxide mode: missing gas reference '{gas_ref}'")

    if "E_per_molecule_eV" in gas:
        E_o2_raw = float(gas["E_per_molecule_eV"])
    elif "E_total_eV" in gas:
        E_o2_raw = float(gas["E_total_eV"])
    else:
        raise KeyError(f"Gas reference {gas_ref} missing E_total_eV or E_per_molecule_eV")

    muO_shift = float(oxide_mode.get("muO_shift_ev", 0.0))
    mu_oxygen = 0.5 * E_o2_raw + muO_shift
    E_o2_effective = E_o2_raw + 2.0 * muO_shift

    return mu_oxygen, E_o2_effective


def _mu_from_oxide(
    oxide_entry: Dict[str, Any],
    *,
    dopant: str,
    anion: str,
    mu_oxygen: float,
) -> float:
    composition = oxide_entry.get("reduced_composition")
    if not isinstance(composition, dict):
        raise KeyError("oxide reference missing reduced_composition")
    if dopant not in composition or anion not in composition:
        raise KeyError(f"oxide reference does not contain {dopant} and {anion}")
    E_fu = float(oxide_entry["E_per_formula_unit_eV"])
    return (E_fu - float(composition[anion]) * mu_oxygen) / float(composition[dopant])


def _host_mu(
    host_ref: Dict[str, Any],
    *,
    host_species: str,
    anion: str,
    mu_oxygen: float,
) -> float:
    composition = host_ref["reduced_composition"]
    if host_species not in composition or anion not in composition:
        raise KeyError(f"Host oxide does not contain {host_species} and {anion}")
    return (
        float(host_ref["E_per_formula_unit_eV"]) - float(composition[anion]) * mu_oxygen
    ) / float(composition[host_species])


def _scenario_label(selection: Dict[str, tuple[str, Dict[str, Any]]]) -> str:
    return "__".join(selection[dopant][0] for dopant in sorted(selection))


def _iter_oxide_scenarios(
    dopant_counts: Dict[str, int],
    reference_groups: Dict[str, List[tuple[str, Dict[str, Any]]]],
) -> Iterable[tuple[str, Dict[str, tuple[str, Dict[str, Any]]]]]:
    dopants = sorted(dopant_counts)
    missing = [dopant for dopant in dopants if dopant not in reference_groups]
    if missing:
        raise KeyError(
            "No binary oxide references found for dopant(s): "
            + ", ".join(missing)
            + ". Add them to [references].oxides_ref and rerun refs-build."
        )

    choices = [reference_groups[dopant] for dopant in dopants]
    for combination in itertools.product(*choices):
        selection = {dopant: pair for dopant, pair in zip(dopants, combination)}
        yield _scenario_label(selection), selection


def _mixing_energy(
    *,
    E_doped: float,
    counts: Dict[str, int],
    host_species: str,
    anion: str,
    dopant_counts: Dict[str, int],
    host_ref: Dict[str, Any],
    selected_oxides: Dict[str, tuple[str, Dict[str, Any]]],
    E_O2: float,
) -> Dict[str, Any]:
    host_comp = host_ref["reduced_composition"]
    n_host = float(counts.get(host_species, 0))
    n_anion = float(counts.get(anion, 0))
    if n_host <= 0.0 or n_anion <= 0.0:
        raise ValueError("Candidate must contain host atoms and the selected anion.")

    host_coeff = n_host / float(host_comp[host_species])
    E_refs = host_coeff * float(host_ref["E_per_formula_unit_eV"])
    anions_from_refs = host_coeff * float(host_comp[anion])
    reaction_left = [f"{host_coeff:g} {host_ref['formula']}"]

    for dopant in sorted(dopant_counts):
        oxide_name, oxide = selected_oxides[dopant]
        composition = oxide["reduced_composition"]
        coeff = float(dopant_counts[dopant]) / float(composition[dopant])
        E_refs += coeff * float(oxide["E_per_formula_unit_eV"])
        anions_from_refs += coeff * float(composition[anion])
        reaction_left.append(f"{coeff:g} {oxide_name}")

    n_O2_out = (anions_from_refs - n_anion) / 2.0
    E_total = float(E_doped + n_O2_out * E_O2 - E_refs)

    n_atoms = int(sum(counts.values()))
    n_cations = int(sum(count for element, count in counts.items() if element != anion))
    n_dopants = int(sum(dopant_counts.values()))
    if n_atoms <= 0 or n_cations <= 0:
        raise ValueError("Candidate has zero atoms or cations.")

    return {
        "E_mix_eV_total": E_total,
        "E_mix_eV_per_atom": E_total / n_atoms,
        "E_mix_eV_per_cation": E_total / n_cations,
        "E_mix_eV_per_dopant": E_total / n_dopants if n_dopants else E_total,
        "n_O2_out": n_O2_out,
        "reaction_reference": " + ".join(reaction_left) + f" -> doped_structure + {n_O2_out:g} O2",
    }


def _single_oxide_endpoint_energy_per_cation(
    *,
    host_ref: Dict[str, Any],
    host_species: str,
    anion: str,
    oxide: Dict[str, Any],
    E_O2: float,
) -> float:
    """Pure dopant-oxide endpoint relative to the host oxide, per cation.

    This is atom-balanced per one host cation. For SnO2 examples:
      Sb2O3: 0.5 E(Sb2O3) + 0.25 E(O2_eff) - E(SnO2)
      Sb2O5: 0.5 E(Sb2O5) - 0.25 E(O2_eff) - E(SnO2)
      SbO2 :     E(SbO2)                       - E(SnO2)
    """
    host_comp = host_ref["reduced_composition"]
    if host_species not in host_comp or anion not in host_comp:
        raise KeyError(f"Host oxide does not contain {host_species} and {anion}")

    n_host_cat = float(host_comp[host_species])
    n_host_anion = float(host_comp[anion])
    E_host_fu = float(host_ref["E_per_formula_unit_eV"])

    oxide_comp = oxide["reduced_composition"]
    dopant_elements = [str(el) for el in oxide_comp if el != anion]
    if len(dopant_elements) != 1:
        raise ValueError(f"Only binary dopant oxides are supported, got {oxide_comp}")

    dopant = dopant_elements[0]
    n_dop_cat = float(oxide_comp[dopant])
    n_oxide_anion = float(oxide_comp[anion])
    E_oxide_fu = float(oxide["E_per_formula_unit_eV"])

    E_host_per_cation = E_host_fu / n_host_cat
    E_oxide_per_cation = E_oxide_fu / n_dop_cat

    # Positive n_O2_out means O2 is on product side:
    # host oxide -> dopant oxide + n_O2_out O2
    n_O2_out_per_cation = (
        (n_host_anion / n_host_cat) - (n_oxide_anion / n_dop_cat)
    ) / 2.0

    return float(E_oxide_per_cation + n_O2_out_per_cation * E_O2 - E_host_per_cation)


def _oxide_endpoint_energies_per_cation(
    *,
    host_ref: Dict[str, Any],
    host_species: str,
    anion: str,
    selected_oxides: Dict[str, tuple[str, Dict[str, Any]]],
    E_O2: float,
) -> Dict[str, float]:
    """Endpoint energy for each dopant oxide in a reference scenario.

    Supports both single-dopant and co-doped scenarios. For co-doping, the
    relative correction is built later as sum_i x_i * E_endpoint_i, so atom
    balance is respected independently for every dopant endpoint.
    """
    endpoints: Dict[str, float] = {}
    for dopant, (_oxide_name, oxide) in sorted(selected_oxides.items()):
        endpoints[dopant] = _single_oxide_endpoint_energy_per_cation(
            host_ref=host_ref,
            host_species=host_species,
            anion=anion,
            oxide=oxide,
            E_O2=E_O2,
        )
    return endpoints


def _weighted_endpoint_correction_per_cation(
    *,
    counts: Dict[str, int],
    dopant_counts: Dict[str, int],
    anion: str,
    endpoint_by_dopant: Dict[str, float],
) -> float:
    """Return sum_i x_i * E_endpoint_i for the candidate composition."""
    n_cations = int(sum(count for element, count in counts.items() if element != anion))
    if n_cations <= 0:
        raise ValueError("Candidate has no cations.")

    correction = 0.0
    for dopant, count in dopant_counts.items():
        if dopant not in endpoint_by_dopant:
            raise KeyError(f"Missing endpoint energy for dopant {dopant}")
        x_i = float(count) / float(n_cations)
        correction += x_i * float(endpoint_by_dopant[dopant])
    return float(correction)

def _metal_chemical_potentials(ref: dict[str, Any]) -> Dict[str, float]:
    mus: Dict[str, float] = {}
    for name, entry in (ref.get("references", {}) or {}).items():
        if not isinstance(entry, dict) or entry.get("type") != "metal":
            continue
        if "E_per_atom_eV" in entry:
            mus[str(name)] = float(entry["E_per_atom_eV"])
    return mus


def _formation_result(
    *,
    E_doped: float,
    E_pristine: float,
    n_atoms_supercell: int,
    counts: Dict[str, int],
    dopant_counts: Dict[str, int],
    host_species: str,
    anion: str,
    host_mu_value: float,
    oxygen_mu_value: float,
    selected_oxides: Dict[str, tuple[str, Dict[str, Any]]],
    host_ref: Dict[str, Any],
    E_O2: float,
    normalize: str,
) -> Dict[str, Any]:
    n_dopants = int(sum(dopant_counts.values()))
    n_atoms = int(sum(counts.values()))
    n_cations = int(sum(count for element, count in counts.items() if element != anion))
    if n_cations <= 0:
        raise ValueError("Candidate has no cations.")

    mus = {host_species: host_mu_value, anion: oxygen_mu_value}
    correction = 0.0
    oxide_references: Dict[str, str] = {}
    for dopant, count in sorted(dopant_counts.items()):
        oxide_name, oxide = selected_oxides[dopant]
        mu_dopant = _mu_from_oxide(
            oxide,
            dopant=dopant,
            anion=anion,
            mu_oxygen=oxygen_mu_value,
        )
        mus[dopant] = mu_dopant
        oxide_references[dopant] = oxide_name
        correction += float(count) * (host_mu_value - mu_dopant)

    E_form_total = float(E_doped - E_pristine + correction)
    endpoint_by_dopant = _oxide_endpoint_energies_per_cation(
        host_ref=host_ref,
        host_species=host_species,
        anion=anion,
        selected_oxides=selected_oxides,
        E_O2=E_O2,
    )
    endpoint_correction_eV_per_cation = _weighted_endpoint_correction_per_cation(
        counts=counts,
        dopant_counts=dopant_counts,
        anion=anion,
        endpoint_by_dopant=endpoint_by_dopant,
    )

    result: Dict[str, Any] = {
        "oxide_references": oxide_references,
        "mu_eV_per_atom_used": mus,
        "oxide_endpoint_eV_per_cation_by_dopant": endpoint_by_dopant,
        "oxide_endpoint_correction_eV_per_cation": endpoint_correction_eV_per_cation,
        # Backward-compatible scalar. For a single dopant this is the pure endpoint;
        # for co-doping it is the composition-weighted correction sum_i x_i E_i.
        "oxide_endpoint_eV_per_cation": (
            next(iter(endpoint_by_dopant.values()))
            if len(endpoint_by_dopant) == 1
            else endpoint_correction_eV_per_cation
        ),
        "E_form_eV_total": E_form_total,
        "E_form_eV_per_atom": E_form_total / n_atoms if n_atoms else E_form_total,
        "E_form_eV_per_cation": E_form_total / n_cations,
        "E_form_eV_per_dopant": E_form_total / n_dopants if n_dopants else E_form_total,
    }

    if normalize == "total":
        result["reported"] = {"value": E_form_total, "unit": "total_eV"}
    elif normalize == "per_host":
        result["reported"] = {
            "value": E_form_total / float(n_atoms_supercell),
            "unit": "eV_per_supercell_atom",
        }
    else:
        result["reported"] = {
            "value": E_form_total / n_dopants if n_dopants else E_form_total,
            "unit": "eV_per_dopant_atom" if n_dopants else "total_eV",
        }

    result["mixing"] = _mixing_energy(
        E_doped=E_doped,
        counts=counts,
        host_species=host_species,
        anion=anion,
        dopant_counts=dopant_counts,
        host_ref=host_ref,
        selected_oxides=selected_oxides,
        E_O2=E_O2,
    )
    return result


def _write_candidate_meta(candidate_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir = candidate_dir / "04_formation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def _flat_columns(reference_results: Dict[str, Dict[str, Any]], *, relative_enabled: bool) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for label, result in sorted(reference_results.items()):
        mixing = result.get("mixing", {}) or {}

        if "oxide_endpoint_eV_per_cation" in result:
            values[f"oxide_endpoint_eV_per_cation__{label}"] = result.get(
                "oxide_endpoint_eV_per_cation"
            )
        if "oxide_endpoint_correction_eV_per_cation" in result:
            values[f"oxide_endpoint_correction_eV_per_cation__{label}"] = result.get(
                "oxide_endpoint_correction_eV_per_cation"
            )
        if "oxide_endpoint_eV_per_cation_by_dopant" in result:
            values[f"oxide_endpoint_by_dopant_json__{label}"] = json.dumps(
                result.get("oxide_endpoint_eV_per_cation_by_dopant", {})
            )

        values[f"E_form_eV_total__{label}"] = result.get("E_form_eV_total")
        values[f"E_form_eV_per_atom__{label}"] = result.get("E_form_eV_per_atom")
        values[f"E_form_eV_per_cation__{label}"] = result.get("E_form_eV_per_cation")
        values[f"E_form_eV_per_dopant__{label}"] = result.get("E_form_eV_per_dopant")

        values[f"E_mix_eV_total__{label}"] = mixing.get("E_mix_eV_total")
        values[f"E_mix_eV_per_atom__{label}"] = mixing.get("E_mix_eV_per_atom")
        values[f"E_mix_eV_per_cation__{label}"] = mixing.get("E_mix_eV_per_cation")
        values[f"E_mix_eV_per_dopant__{label}"] = mixing.get("E_mix_eV_per_dopant")
        values[f"n_O2_out__{label}"] = mixing.get("n_O2_out")
        values[f"mixing_reaction_reference__{label}"] = mixing.get("reaction_reference", "")

        if relative_enabled:
            relative = result.get("relative", {}) or {}
            values[f"E_form_rel_eV_per_cation__{label}"] = relative.get(
                "E_form_rel_eV_per_cation"
            )
            values[f"E_mix_rel_eV_per_cation__{label}"] = relative.get(
                "E_mix_rel_eV_per_cation"
            )
            values[f"relative_reference__{label}"] = relative.get("reference", "")
            values[f"relative_endpoint_x__{label}"] = relative.get("endpoint_x")
    return values


def _apply_relative_energies(
    records: List[CandidateRecord],
    *,
    endpoint_x: float | None,
) -> tuple[float, Dict[str, Dict[str, Any]]]:
    """Populate relative columns without double-correcting oxide references.

    In oxide reference mode, E_form_eV_per_cation and E_mix_eV_per_cation are
    already referenced to the selected oxide/host tie-line through the chemical
    potentials or the atom-balanced mixing reaction. Therefore the relative
    columns are aliases of the corresponding per-cation absolute columns.

    Endpoint information is still recorded for transparency. For co-doping, the
    endpoint correction is composition weighted: sum_i x_i E_endpoint_i.
    """
    if not records:
        raise ValueError("Relative energies requested, but no candidates were found.")

    X = 1.0 if endpoint_x is None else float(endpoint_x)
    if not 0.0 < X <= 1.0:
        raise ValueError("Relative endpoint_x must be in (0, 1].")

    endpoint_by_label: Dict[str, Dict[str, Any]] = {}

    for record in records:
        for label, result in record.reference_results.items():
            endpoint_by_label.setdefault(
                label,
                {
                    "reference": "oxide_reference_already_tieline_corrected",
                    "oxide_endpoint_eV_per_cation_by_dopant": result.get(
                        "oxide_endpoint_eV_per_cation_by_dopant", {}
                    ),
                    "oxide_endpoint_correction_eV_per_cation": result.get(
                        "oxide_endpoint_correction_eV_per_cation", None
                    ),
                    "E_endpoint_eV_per_cation": result.get(
                        "oxide_endpoint_eV_per_cation", None
                    ),
                },
            )

    for record in records:
        for label, result in record.reference_results.items():
            mixing = result.get("mixing", {}) or {}
            E_mix_per_cation = mixing.get("E_mix_eV_per_cation")

            result["relative"] = {
                "endpoint_x": X,
                "reference": "oxide_reference_already_tieline_corrected",
                "E_endpoint_eV_per_cation": result.get("oxide_endpoint_eV_per_cation"),
                "oxide_endpoint_eV_per_cation_by_dopant": result.get(
                    "oxide_endpoint_eV_per_cation_by_dopant", {}
                ),
                "oxide_endpoint_correction_eV_per_cation": result.get(
                    "oxide_endpoint_correction_eV_per_cation"
                ),
                "E_form_rel_eV_per_cation": float(result["E_form_eV_per_cation"]),
                "E_mix_rel_eV_per_cation": (
                    float(E_mix_per_cation) if E_mix_per_cation is not None else None
                ),
            }

    return X, endpoint_by_label

def _write_folder_csv(
    folder: Path,
    records: List[CandidateRecord],
    *,
    reference_mode: str,
    relative_enabled: bool,
) -> None:
    base_fields = [
        "candidate",
        "E_doped_eV",
        "n_dopant_atoms",
        "dopant_counts",
        "x_dopant",
        "reference_mode",
    ]
    dynamic_fields = sorted({
        field
        for record in records
        for field in _flat_columns(record.reference_results, relative_enabled=relative_enabled)
    })
    path = folder / OUT_CSV
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + dynamic_fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.E_doped):
            row: Dict[str, Any] = {
                "candidate": record.candidate,
                "E_doped_eV": record.E_doped,
                "n_dopant_atoms": sum(record.dopant_counts.values()),
                "dopant_counts": ";".join(
                    f"{element}:{count}" for element, count in sorted(record.dopant_counts.items())
                ),
                "x_dopant": record.x_dopant,
                "reference_mode": reference_mode,
            }
            row.update(_flat_columns(record.reference_results, relative_enabled=relative_enabled))
            writer.writerow({key: _fmt(value) for key, value in row.items()})
    log.info("OK   %s: wrote %s (rows=%d)", folder.name, OUT_CSV, len(records))


def run_formation(raw_cfg: dict[str, Any], root: Path, *, config_path: Path | None = None) -> None:
    """Compute formation and oxide-reference mixing energies for relaxed candidates.

    In oxide mode every binary oxide reference matching a dopant is evaluated.
    Results are written in wide format: one candidate row and one set of columns
    per reference scenario (for example ``__SbO2`` and ``__Sb2O5``).
    """
    cfg = _parse_formation_config(raw_cfg, root)
    ref = _load_ref_json(root)
    E_pristine, n_atoms_supercell = _get_pristine_energy_and_natoms(ref)
    host_formula = str((ref.get("host") or {}).get("name", "")).strip()
    if not host_formula:
        raise KeyError("reference JSON missing host.name")

    reference_mode = str(ref.get("reference_mode", "metal")).strip().lower()
    if reference_mode not in {"metal", "oxide"}:
        raise ValueError(f"Unsupported reference mode: {reference_mode!r}")

    if len(cfg.anion_species) != 1:
        raise ValueError("Formation currently supports exactly one anion species.")
    anion = cfg.anion_species[0]

    if not cfg.outdir.exists():
        raise FileNotFoundError(f"Output directory not found: {cfg.outdir}")

    host_ref: Dict[str, Any] | None = None
    mu_oxygen: float | None = None
    E_O2: float | None = None
    mu_host: float | None = None
    reference_groups: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    metal_mus: Dict[str, float] = {}

    if reference_mode == "oxide":
        host_ref = _formula_unit_energy_from_host(ref, host_formula)
        mu_oxygen, E_O2 = _oxygen_mu(ref)
        mu_host = _host_mu(
            host_ref,
            host_species=cfg.host_species,
            anion=anion,
            mu_oxygen=mu_oxygen,
        )
        reference_groups = _oxide_reference_groups(ref, anion=anion)
    else:
        metal_mus = _metal_chemical_potentials(ref)
        mu_host = metal_mus.get(cfg.host_species)
        if mu_host is None:
            raise KeyError(
                f"Missing metal chemical potential for host_species={cfg.host_species!r}"
            )

    folders = sorted(path for path in cfg.outdir.iterdir() if path.is_dir())
    active_folders: List[Path] = []
    for folder in folders:
        out_csv = folder / OUT_CSV
        if cfg.skip_if_done and out_csv.exists():
            log.info("SKIP %s: %s exists", folder.name, OUT_CSV)
        else:
            active_folders.append(folder)

    if not active_folders:
        log.info("DONE Step 06 formation: all folders skipped.")
        return

    records_by_folder: Dict[Path, List[CandidateRecord]] = {}
    all_records: List[CandidateRecord] = []

    for folder in active_folders:
        folder_records: List[CandidateRecord] = []
        for poscar in _get_candidate_poscars(folder):
            candidate_dir = poscar.parents[1]
            meta_path = candidate_dir / RELAX_META
            if not meta_path.exists():
                log.warning("%s/%s: missing %s -> skip", folder.name, candidate_dir.name, RELAX_META)
                continue

            E_doped = _load_relax_energy(meta_path)
            counts = _count_species_from_poscar(poscar)
            dopant_counts = _compute_substitution_dopant_counts(
                counts,
                cfg.host_species,
                cfg.anion_species,
            )
            if not dopant_counts:
                log.warning("%s/%s: no dopants identified -> skip", folder.name, candidate_dir.name)
                continue

            n_cations = sum(count for element, count in counts.items() if element != anion)
            if n_cations <= 0:
                log.warning("%s/%s: no cations identified -> skip", folder.name, candidate_dir.name)
                continue

            x_dopant = sum(dopant_counts.values()) / float(n_cations)
            reference_results: Dict[str, Dict[str, Any]] = {}

            if reference_mode == "oxide":
                assert host_ref is not None
                assert mu_oxygen is not None
                assert E_O2 is not None
                assert mu_host is not None

                for label, selected_oxides in _iter_oxide_scenarios(dopant_counts, reference_groups):
                    reference_results[label] = _formation_result(
                        E_doped=E_doped,
                        E_pristine=E_pristine,
                        n_atoms_supercell=n_atoms_supercell,
                        counts=counts,
                        dopant_counts=dopant_counts,
                        host_species=cfg.host_species,
                        anion=anion,
                        host_mu_value=mu_host,
                        oxygen_mu_value=mu_oxygen,
                        selected_oxides=selected_oxides,
                        host_ref=host_ref,
                        E_O2=E_O2,
                        normalize=cfg.normalize,
                    )
            else:
                missing = [dopant for dopant in dopant_counts if dopant not in metal_mus]
                if missing:
                    raise KeyError(
                        "Missing metal chemical potentials for dopant(s): " + ", ".join(sorted(missing))
                    )

                correction = sum(
                    float(count) * (float(mu_host) - metal_mus[dopant])
                    for dopant, count in dopant_counts.items()
                )
                E_form_total = float(E_doped - E_pristine + correction)
                n_atoms = sum(counts.values())
                n_dopants = sum(dopant_counts.values())
                report_value = E_form_total if cfg.normalize == "total" else (
                    E_form_total / float(n_atoms_supercell)
                    if cfg.normalize == "per_host"
                    else (E_form_total / n_dopants if n_dopants else E_form_total)
                )
                report_unit = (
                    "total_eV" if cfg.normalize == "total"
                    else ("eV_per_supercell_atom" if cfg.normalize == "per_host"
                          else ("eV_per_dopant_atom" if n_dopants else "total_eV"))
                )
                reference_results["metal"] = {
                    "oxide_references": {},
                    "mu_eV_per_atom_used": {
                        cfg.host_species: float(mu_host),
                        **{dopant: metal_mus[dopant] for dopant in dopant_counts},
                    },
                    "E_form_eV_total": E_form_total,
                    "E_form_eV_per_atom": E_form_total / n_atoms if n_atoms else E_form_total,
                    "E_form_eV_per_cation": E_form_total / n_cations,
                    "E_form_eV_per_dopant": E_form_total / n_dopants if n_dopants else E_form_total,
                    "reported": {"value": report_value, "unit": report_unit},
                    "mixing": {},
                }

            record = CandidateRecord(
                folder=folder,
                candidate_dir=candidate_dir,
                candidate=candidate_dir.name,
                E_doped=E_doped,
                counts=counts,
                dopant_counts=dopant_counts,
                x_dopant=x_dopant,
                reference_results=reference_results,
            )
            folder_records.append(record)
            all_records.append(record)

        if folder_records:
            records_by_folder[folder] = folder_records
        else:
            log.info("SKIP %s: no valid candidates", folder.name)

    if not all_records:
        log.info("DONE Step 06 formation: no valid candidates.")
        return

    relative_metadata: Dict[str, Any] = {}
    formation_writes_relative = cfg.relative_enabled and reference_mode == "oxide"
    if formation_writes_relative:
        X, endpoint_by_label = _apply_relative_energies(
            all_records,
            endpoint_x=cfg.relative_endpoint_x,
        )
        relative_metadata = {
            "enabled": True,
            "endpoint_x": X,
            "endpoint_selection": "oxide_endmember_tieline",
            "endpoint_energies": endpoint_by_label,
        }

    for record in all_records:
        first_label = sorted(record.reference_results)[0]
        primary = record.reference_results[first_label]
        payload: Dict[str, Any] = {
            "stage": "04_formation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reference_mode": reference_mode,
            "host_formula": host_formula,
            "host_species": cfg.host_species,
            "anion_species": cfg.anion_species,
            "E_doped_eV": record.E_doped,
            "E_pristine_eV": E_pristine,
            "n_atoms_supercell": n_atoms_supercell,
            "dopant_counts": record.dopant_counts,
            "x_dopant": record.x_dopant,
            "reference_results": record.reference_results,
            "relative_energy": relative_metadata,
            "primary_reference_label": first_label,
            # Backward-compatible primary fields.
            "E_form_eV_total": primary["E_form_eV_total"],
            "reported": primary["reported"],
            "mixing": primary["mixing"],
        }
        _write_candidate_meta(record.candidate_dir, payload)

    for folder, folder_records in records_by_folder.items():
        _write_folder_csv(
            folder,
            folder_records,
            reference_mode=reference_mode,
            relative_enabled=formation_writes_relative,
        )

    log.info("DONE Step 06 formation.")


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_raw_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def run_formation_from_toml(config_path: Path) -> None:
    raw = _load_raw_toml(config_path)
    run_formation(raw, config_path.resolve().parent, config_path=config_path)

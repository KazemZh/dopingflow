from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from pymatgen.core import Structure


def _pct(count: int, denominator: int) -> float | None:
    if int(denominator) <= 0:
        return None
    return 100.0 * float(count) / float(denominator)


def _fmt_pct(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    value = float(value)
    if abs(value - round(value)) <= 1e-8:
        return f"{int(round(value))}%"
    return f"{value:.1f}%"


def _nearest(
    structure: Structure,
    source_index: int,
    candidates: Sequence[int],
) -> tuple[int | None, float]:
    if not candidates:
        return None, math.inf
    pairs = [
        (float(structure.get_distance(source_index, target)), int(target))
        for target in candidates
        if int(target) != int(source_index)
    ]
    if not pairs:
        return None, math.inf
    distance, index = min(pairs)
    return index, distance


def _components(nodes: Sequence[int], edges: Mapping[int, set[int]]) -> list[list[int]]:
    remaining = set(int(node) for node in nodes)
    groups: list[list[int]] = []
    while remaining:
        start = min(remaining)
        queue: deque[int] = deque([start])
        remaining.remove(start)
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(edges.get(node, set())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        groups.append(sorted(component))
    return groups


def _expected_simple_counts(state: Mapping[str, Any]) -> dict[str, int]:
    family = str(state.get("family", ""))
    d_h = int(state.get("delta_n_H", 0))
    d_o = int(state.get("delta_n_O", 0))
    counts = {
        "protonated": 0,
        "O": 0,
        "OH": 0,
        "H2O": 0,
    }
    if family == "protonated":
        counts["protonated"] = d_h
    elif family == "O":
        counts["O"] = d_o
    elif family == "OH":
        counts["OH"] = d_o
    elif family == "H2O":
        counts["H2O"] = d_o
    elif family == "mixed-O-OH":
        counts["OH"] = d_h
        counts["O"] = max(d_o - d_h, 0)
    return counts


def _final_family(
    *,
    n_protonated: int,
    n_o: int,
    n_oh: int,
    n_h2o: int,
    n_o2: int,
    n_ooh: int,
    n_oxygen_cluster: int,
    n_lattice_oo: int,
) -> str:
    complex_oxygen = n_ooh + n_oxygen_cluster + n_lattice_oo
    active = [
        name
        for name, count in (
            ("protonated", n_protonated),
            ("O", n_o),
            ("OH", n_oh),
            ("H2O", n_h2o),
            ("O2", n_o2),
        )
        if count > 0
    ]
    if complex_oxygen > 0:
        return "reconstructed"
    if not active:
        return "clean"
    if active == ["protonated"]:
        return "protonated"
    if active == ["O"]:
        return "O"
    if active == ["OH"]:
        return "OH"
    if active == ["H2O"]:
        return "H2O"
    if active == ["O2"]:
        return "O2"
    if set(active) == {"O", "OH"}:
        return "mixed-O-OH"
    return "reconstructed"


def _final_label(
    family: str,
    *,
    n_oxygen_sites: int,
    n_cation_sites: int,
    n_protonated: int,
    n_o: int,
    n_oh: int,
    n_h2o: int,
    n_o2: int,
    n_ooh: int,
    n_oxygen_cluster: int,
    n_lattice_oo: int,
) -> str:
    proton_cov = _pct(n_protonated, n_oxygen_sites)
    o_cov = _pct(n_o, n_cation_sites)
    oh_cov = _pct(n_oh, n_cation_sites)
    h2o_cov = _pct(n_h2o, n_cation_sites)
    o2_oxygen_cov = _pct(2 * n_o2, n_cation_sites)

    if family == "clean":
        return "Clean"
    if family == "protonated":
        return f"Protonated lattice O — {_fmt_pct(proton_cov)}"
    if family == "O":
        return f"O* — {_fmt_pct(o_cov)}"
    if family == "OH":
        return f"OH* — {_fmt_pct(oh_cov)}"
    if family == "H2O":
        return f"H₂O* — {_fmt_pct(h2o_cov)}"
    if family == "mixed-O-OH":
        return f"Mixed O*/OH* — {_fmt_pct(o_cov)} O* + {_fmt_pct(oh_cov)} OH*"
    if family == "O2":
        return f"O₂* — {_fmt_pct(o2_oxygen_cov)} O-atom coverage"

    parts: list[str] = []
    if n_protonated:
        parts.append(f"{_fmt_pct(proton_cov)} protonated O")
    if n_o:
        parts.append(f"{_fmt_pct(o_cov)} O*")
    if n_oh:
        parts.append(f"{_fmt_pct(oh_cov)} OH*")
    if n_h2o:
        parts.append(f"{_fmt_pct(h2o_cov)} H₂O*")
    if n_o2:
        parts.append(f"{n_o2} O₂*")
    if n_ooh:
        parts.append(f"{n_ooh} OOH/peroxo-like")
    if n_oxygen_cluster:
        parts.append(f"{n_oxygen_cluster} Oₓ cluster(s)")
    if n_lattice_oo:
        parts.append(f"{n_lattice_oo} lattice O–O reconstruction(s)")
    return "Reconstructed surface — " + ", ".join(parts or ["unclassified"])


def _int_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, str):
        return set()
    try:
        return {int(item) for item in value}
    except TypeError:
        return set()


def analyze_relaxed_surface_state(
    parent_structure: Structure,
    generated_structure: Structure,
    relaxed_structure: Structure,
    state: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and reclassify a relaxed electrochemical surface state.

    The initial state label is only a proposal. The relaxed geometry determines
    whether the intended O/OH/H2O/protonated chemistry survived, reconstructed,
    formed an O-O species, or desorbed into the vacuum.
    """
    if not bool(cfg.get("postprocess_validate_relaxed_states", True)):
        return {
            "state_status": "validation-disabled",
            "pourbaix_eligible": True,
            "final_family": str(state.get("family", "")),
            "final_state_label": "",
            "postprocess_reason": "post-relaxation validation disabled",
        }

    n_parent = len(parent_structure)
    if len(generated_structure) != len(relaxed_structure):
        return {
            "state_status": "invalid",
            "pourbaix_eligible": False,
            "final_family": "invalid",
            "final_state_label": "Invalid relaxed structure",
            "postprocess_reason": (
                f"atom count changed: generated={len(generated_structure)}, "
                f"relaxed={len(relaxed_structure)}"
            ),
        }
    generated_species = [site.specie.symbol for site in generated_structure]
    relaxed_species = [site.specie.symbol for site in relaxed_structure]
    if generated_species != relaxed_species:
        return {
            "state_status": "invalid",
            "pourbaix_eligible": False,
            "final_family": "invalid",
            "final_state_label": "Invalid relaxed structure",
            "postprocess_reason": "atom ordering/species changed during relaxation",
        }
    if n_parent > len(relaxed_structure):
        return {
            "state_status": "invalid",
            "pourbaix_eligible": False,
            "final_family": "invalid",
            "final_state_label": "Invalid relaxed structure",
            "postprocess_reason": "relaxed structure contains fewer atoms than parent",
        }

    anions = set(cfg.get("anion_species", ["O"]))
    # Surface-Pourbaix chemistry currently treats oxygen explicitly. Other
    # configured anions remain part of the parent slab connectivity.
    parent_oxygen = [
        i for i in range(n_parent) if relaxed_structure[i].specie.symbol == "O"
    ]
    parent_cations = [
        i
        for i in range(n_parent)
        if relaxed_structure[i].specie.symbol not in anions
        and relaxed_structure[i].specie.symbol != "H"
    ]
    added_indices = list(range(n_parent, len(relaxed_structure)))
    added_oxygen = [
        i for i in added_indices if relaxed_structure[i].specie.symbol == "O"
    ]
    added_hydrogen = [
        i for i in added_indices if relaxed_structure[i].specie.symbol == "H"
    ]
    all_oxygen = parent_oxygen + added_oxygen

    oh_cutoff = float(cfg.get("postprocess_oh_bond_cutoff_A", 1.25))
    oo_cutoff = float(cfg.get("postprocess_oo_bond_cutoff_A", 1.75))
    attachment_cutoff = float(
        cfg.get("postprocess_surface_attachment_cutoff_A", 2.80)
    )
    hh_cutoff = float(cfg.get("postprocess_hh_bond_cutoff_A", 0.90))

    # Assign each newly added H to its nearest O if it has a normal O-H bond.
    h_to_o: dict[int, int] = {}
    o_to_h: dict[int, list[int]] = defaultdict(list)
    unbound_h: list[int] = []
    for h_index in added_hydrogen:
        o_index, distance = _nearest(relaxed_structure, h_index, all_oxygen)
        if o_index is not None and distance <= oh_cutoff:
            h_to_o[h_index] = o_index
            o_to_h[o_index].append(h_index)
        else:
            unbound_h.append(h_index)

    # Detect H2-like fragments among H atoms that are no longer O-bound.
    hh_edges: dict[int, set[int]] = defaultdict(set)
    for pos, first in enumerate(unbound_h):
        for second in unbound_h[pos + 1 :]:
            if float(relaxed_structure.get_distance(first, second)) <= hh_cutoff:
                hh_edges[first].add(second)
                hh_edges[second].add(first)
    h_components = _components(unbound_h, hh_edges) if unbound_h else []
    n_h2_like = sum(1 for comp in h_components if len(comp) == 2)
    n_unbound_h = sum(len(comp) for comp in h_components if len(comp) != 2)

    # Surface attachment of added O is determined by O-to-parent-cation
    # coordination. Detached O/OH/H2O/O2 in the vacuum are therefore not
    # silently counted as adsorbates.
    added_o_attachment: dict[int, tuple[int | None, float, bool]] = {}
    for o_index in added_oxygen:
        cation_index, distance = _nearest(
            relaxed_structure, o_index, parent_cations
        )
        added_o_attachment[o_index] = (
            cation_index,
            distance,
            bool(cation_index is not None and distance <= attachment_cutoff),
        )

    # Added-added O-O graph. A short O-O distance is treated structurally as
    # molecular/peroxo-like; no electronic O2/superoxo/peroxo assignment is
    # inferred from distance alone.
    oo_edges: dict[int, set[int]] = defaultdict(set)
    for pos, first in enumerate(added_oxygen):
        for second in added_oxygen[pos + 1 :]:
            if float(relaxed_structure.get_distance(first, second)) <= oo_cutoff:
                oo_edges[first].add(second)
                oo_edges[second].add(first)
    oxygen_components = _components(added_oxygen, oo_edges) if added_oxygen else []

    # Added O bonding directly to a parent lattice O is a separate surface
    # reconstruction and should not be mislabeled as ordinary O*.
    lattice_oo_pairs: list[tuple[int, int]] = []
    for added_o in added_oxygen:
        for parent_o in parent_oxygen:
            if float(relaxed_structure.get_distance(added_o, parent_o)) <= oo_cutoff:
                lattice_oo_pairs.append((added_o, parent_o))

    added_in_oo = {
        index
        for component in oxygen_components
        if len(component) >= 2
        for index in component
    }
    added_in_lattice_oo = {pair[0] for pair in lattice_oo_pairs}

    n_surface_o = 0
    n_surface_oh = 0
    n_surface_h2o = 0
    n_desorbed_o = 0
    n_desorbed_oh = 0
    n_desorbed_h2o = 0
    n_surface_o2 = 0
    n_desorbed_o2 = 0
    n_surface_ooh_like = 0
    n_desorbed_ooh_like = 0
    n_surface_oxygen_cluster = 0
    n_desorbed_oxygen_cluster = 0
    n_surface_lattice_oo_like = 0
    n_desorbed_lattice_oo_like = 0
    n_overcoordinated_added_o = 0

    final_o_site_indices: list[int] = []
    final_oh_site_indices: list[int] = []
    final_h2o_site_indices: list[int] = []
    bound_added_o: set[int] = set()

    # Simple, non-O-O added oxygen species.
    for o_index in added_oxygen:
        if o_index in added_in_oo or o_index in added_in_lattice_oo:
            continue
        cation_index, _distance, attached = added_o_attachment[o_index]
        n_h = len(o_to_h.get(o_index, []))
        if attached:
            bound_added_o.add(o_index)
        if n_h == 0:
            if attached:
                n_surface_o += 1
                if cation_index is not None:
                    final_o_site_indices.append(int(cation_index))
            else:
                n_desorbed_o += 1
        elif n_h == 1:
            if attached:
                n_surface_oh += 1
                if cation_index is not None:
                    final_oh_site_indices.append(int(cation_index))
            else:
                n_desorbed_oh += 1
        elif n_h == 2:
            if attached:
                n_surface_h2o += 1
                if cation_index is not None:
                    final_h2o_site_indices.append(int(cation_index))
            else:
                n_desorbed_h2o += 1
        else:
            n_overcoordinated_added_o += 1

    # Molecular/clustered added oxygen.
    for component in oxygen_components:
        if len(component) < 2:
            continue
        attached = any(added_o_attachment[index][2] for index in component)
        if attached:
            bound_added_o.update(component)
        total_h = sum(len(o_to_h.get(index, [])) for index in component)
        if len(component) == 2 and total_h == 0:
            if attached:
                n_surface_o2 += 1
            else:
                n_desorbed_o2 += 1
        elif len(component) == 2:
            if attached:
                n_surface_ooh_like += 1
            else:
                n_desorbed_ooh_like += 1
        else:
            if attached:
                n_surface_oxygen_cluster += 1
            else:
                n_desorbed_oxygen_cluster += 1

    for added_o, _parent_o in lattice_oo_pairs:
        if added_o_attachment[added_o][2]:
            n_surface_lattice_oo_like += 1
            bound_added_o.add(added_o)
        else:
            n_desorbed_lattice_oo_like += 1

    # Added H attached to original lattice oxygen = protonated lattice oxygen.
    protonated_parent_o_indices = sorted(
        index
        for index in parent_oxygen
        if len(o_to_h.get(index, [])) == 1
    )
    overprotonated_parent_o_indices = sorted(
        index
        for index in parent_oxygen
        if len(o_to_h.get(index, [])) >= 2
    )
    n_protonated_lattice_o = len(protonated_parent_o_indices)
    n_overprotonated_lattice_o = len(overprotonated_parent_o_indices)

    # If an original lattice oxygen has picked up two H and is no longer
    # coordinated to the slab cation network, it has effectively become a
    # desorbing water-like fragment / oxygen-vacancy event.
    n_desorbed_parent_h2o_like = 0
    for o_index in overprotonated_parent_o_indices:
        _cation, distance = _nearest(relaxed_structure, o_index, parent_cations)
        if distance > attachment_cutoff:
            n_desorbed_parent_h2o_like += 1

    expected = _expected_simple_counts(state)
    final_simple = {
        "protonated": n_protonated_lattice_o,
        "O": n_surface_o,
        "OH": n_surface_oh,
        "H2O": n_surface_h2o,
    }

    complex_surface_count = (
        n_surface_o2
        + n_surface_ooh_like
        + n_surface_oxygen_cluster
        + n_surface_lattice_oo_like
        + n_overprotonated_lattice_o
        + n_overcoordinated_added_o
    )
    desorbed_count = (
        n_desorbed_o
        + n_desorbed_oh
        + n_desorbed_h2o
        + n_desorbed_o2
        + n_desorbed_ooh_like
        + n_desorbed_oxygen_cluster
        + n_desorbed_lattice_oo_like
        + n_desorbed_parent_h2o_like
    )
    fragmented = n_unbound_h > 0 or n_h2_like > 0 or n_overcoordinated_added_o > 0

    # Decide whether the same chemical coverage survived but hopped/reordered.
    arrangement_changed = False
    family = str(state.get("family", ""))
    if final_simple == expected and complex_surface_count == 0 and desorbed_count == 0 and not fragmented:
        if family == "protonated":
            arrangement_changed = (
                set(protonated_parent_o_indices)
                != _int_set(state.get("site_indices"))
            )
        elif family in {"O", "OH", "H2O"}:
            final_sites = {
                "O": set(final_o_site_indices),
                "OH": set(final_oh_site_indices),
                "H2O": set(final_h2o_site_indices),
            }[family]
            arrangement_changed = final_sites != _int_set(state.get("site_indices"))
        elif family == "mixed-O-OH":
            arrangement_changed = (
                set(final_o_site_indices) != _int_set(state.get("o_site_indices"))
                or set(final_oh_site_indices) != _int_set(state.get("oh_site_indices"))
            )

    if desorbed_count > 0:
        status = "desorbed"
        reason = "one or more O/OH/H2O/O-O fragments detached from the slab"
    elif fragmented:
        status = "fragmented"
        reason = "unbound H/H2-like or overcoordinated fragments formed"
    elif final_simple == expected and complex_surface_count == 0:
        status = "reconstructed_same_coverage" if arrangement_changed else "retained"
        reason = (
            "surface chemistry retained but adsorption/protonation sites changed"
            if arrangement_changed
            else "intended relaxed surface chemistry retained"
        )
    else:
        status = "reclassified"
        reason = "relaxation changed the surface chemical species/coverage class"

    final_family = _final_family(
        n_protonated=n_protonated_lattice_o,
        n_o=n_surface_o,
        n_oh=n_surface_oh,
        n_h2o=n_surface_h2o,
        n_o2=n_surface_o2,
        n_ooh=n_surface_ooh_like,
        n_oxygen_cluster=n_surface_oxygen_cluster,
        n_lattice_oo=n_surface_lattice_oo_like,
    )

    n_oxygen_sites = int(state.get("eligible_surface_oxygen_sites", 0) or 0)
    n_cation_sites = int(state.get("eligible_surface_cation_sites", 0) or 0)
    final_label = _final_label(
        final_family,
        n_oxygen_sites=n_oxygen_sites,
        n_cation_sites=n_cation_sites,
        n_protonated=n_protonated_lattice_o,
        n_o=n_surface_o,
        n_oh=n_surface_oh,
        n_h2o=n_surface_h2o,
        n_o2=n_surface_o2,
        n_ooh=n_surface_ooh_like,
        n_oxygen_cluster=n_surface_oxygen_cluster,
        n_lattice_oo=n_surface_lattice_oo_like,
    )

    if status == "desorbed":
        eligible = not bool(cfg.get("postprocess_exclude_desorbed", True))
    elif status == "fragmented":
        eligible = not bool(cfg.get("postprocess_exclude_fragmented", True))
    elif status == "reclassified":
        eligible = bool(cfg.get("postprocess_allow_reclassified", True))
    else:
        eligible = True

    return {
        "state_status": status,
        "pourbaix_eligible": bool(eligible),
        "final_family": final_family,
        "final_state_label": final_label,
        "postprocess_reason": reason,
        "n_protonated_lattice_O": n_protonated_lattice_o,
        "n_overprotonated_lattice_O": n_overprotonated_lattice_o,
        "n_surface_O": n_surface_o,
        "n_surface_OH": n_surface_oh,
        "n_surface_H2O": n_surface_h2o,
        "n_surface_O2": n_surface_o2,
        "n_surface_OOH_like": n_surface_ooh_like,
        "n_surface_oxygen_cluster": n_surface_oxygen_cluster,
        "n_surface_lattice_OO_like": n_surface_lattice_oo_like,
        "n_desorbed_O": n_desorbed_o,
        "n_desorbed_OH": n_desorbed_oh,
        "n_desorbed_H2O": n_desorbed_h2o,
        "n_desorbed_O2": n_desorbed_o2,
        "n_desorbed_OOH_like": n_desorbed_ooh_like,
        "n_desorbed_oxygen_cluster": n_desorbed_oxygen_cluster,
        "n_desorbed_lattice_OO_like": n_desorbed_lattice_oo_like,
        "n_desorbed_parent_H2O_like": n_desorbed_parent_h2o_like,
        "n_unbound_H": n_unbound_h,
        "n_H2_like": n_h2_like,
        "final_protonated_coverage_pct": _pct(
            n_protonated_lattice_o, n_oxygen_sites
        ),
        "final_o_coverage_pct": _pct(n_surface_o, n_cation_sites),
        "final_oh_coverage_pct": _pct(n_surface_oh, n_cation_sites),
        "final_h2o_coverage_pct": _pct(n_surface_h2o, n_cation_sites),
        "final_o2_oxygen_coverage_pct": _pct(
            2 * n_surface_o2, n_cation_sites
        ),
        "final_total_bound_added_O_coverage_pct": _pct(
            len(bound_added_o), n_cation_sites
        ),
        "final_protonated_O_indices_json": str(protonated_parent_o_indices),
        "final_O_site_indices_json": str(sorted(final_o_site_indices)),
        "final_OH_site_indices_json": str(sorted(final_oh_site_indices)),
        "final_H2O_site_indices_json": str(sorted(final_h2o_site_indices)),
        "postprocess_oh_bond_cutoff_A": oh_cutoff,
        "postprocess_oo_bond_cutoff_A": oo_cutoff,
        "postprocess_surface_attachment_cutoff_A": attachment_cutoff,
        "postprocess_hh_bond_cutoff_A": hh_cutoff,
    }


def analyze_relaxed_surface_state_path(
    parent_structure: Structure,
    generated_structure: Structure,
    relaxed_path: Path | str,
    state: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(relaxed_path)
    if not path.exists():
        return {
            "state_status": "invalid",
            "pourbaix_eligible": False,
            "final_family": "invalid",
            "final_state_label": "Invalid / missing relaxed structure",
            "postprocess_reason": f"relaxed structure file not found: {path}",
        }
    try:
        relaxed = Structure.from_file(path)
    except Exception as exc:
        return {
            "state_status": "invalid",
            "pourbaix_eligible": False,
            "final_family": "invalid",
            "final_state_label": "Invalid relaxed structure",
            "postprocess_reason": f"{type(exc).__name__}: {exc}",
        }
    return analyze_relaxed_surface_state(
        parent_structure,
        generated_structure,
        relaxed,
        state,
        cfg,
    )

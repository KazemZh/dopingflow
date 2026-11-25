"""Periodic adsorption sites and coverage states for H and molecular CO."""
from itertools import combinations, product
from math import comb
import random

import numpy as np
from scipy.spatial import Voronoi

from dopingflow.surface_pourbaix_sampling import (
    canonical_pattern, mixed_realizable_coverages, realizable_coverages,
    select_diverse_patterns, symmetry_permutations,
)


def _site_pool(slab, exposed, normal, types):
    """Project each outer layer onto its plane and tessellate periodic images.

    Ridge midpoints between adjacent metal atoms give bridges. Voronoi vertices
    give threefold hollows on a triangular net and fourfold hollows on a rectangle.
    """
    a, b = slab.lattice.matrix[:2]
    u = a / np.linalg.norm(a)
    basis = np.array([u, np.cross(normal, u)])
    pool = {}
    for side in sorted({sign for _, sign in exposed}):
        frac = [slab[i].frac_coords for i, sign in exposed if sign == side]
        layer = np.array([slab.lattice.get_cartesian_coords(
            [f[0] % 1, f[1] % 1, f[2]]) for f in frac])
        def add(kind, xyz):
            if kind not in types:
                return
            f = slab.lattice.get_fractional_coords(xyz)
            f[:2] %= 1
            key = (side, *np.mod(np.round(f[:2], 5), 1))
            pool.setdefault(key, (kind, side, slab.lattice.get_cartesian_coords(f)))

        for xyz in layer:
            add("ontop", xyz)
        if set(types) == {"ontop"}:
            continue
        shifts = [(0, 0)] + [s for s in product(range(-2, 3), repeat=2) if s != (0, 0)]
        images = np.vstack([layer + i*a + j*b for i, j in shifts])
        vor = Voronoi(images @ basis.T)
        for i, j in vor.ridge_points:
            if min(i, j) < len(layer):
                add("bridge", (images[i] + images[j]) / 2)
        height = np.mean(layer @ normal)
        for i in range(len(layer)):
            for vertex in vor.regions[vor.point_region[i]]:
                if vertex >= 0:
                    add("hollow", vor.vertices[vertex] @ basis + height * normal)
    return [pool[k] for k in sorted(pool)]


def _occupancies(n, nh, nc, limit):
    """Yield all assignments when affordable, otherwise reproducibly sample."""
    total = comb(n, nh + nc) * comb(nh + nc, nh)
    if total <= limit:
        for occupied in combinations(range(n), nh + nc):
            for h in combinations(occupied, nh):
                yield h, tuple(sorted(set(occupied) - set(h)))
    else:
        rng = random.Random(17)
        for _ in range(limit):
            occupied = rng.sample(range(n), nh + nc)
            yield tuple(sorted(occupied[:nh])), tuple(sorted(occupied[nh:]))


def _upright_states(slab, exposed, normal, cfg, common):
    if not exposed:
        raise ValueError("H/CO coverage needs exposed surface metal atoms")
    pool = _site_pool(slab, exposed, normal, cfg["metal_site_types"])
    # Marker atoms are used only for site permutations and distance descriptors.
    template = slab.copy()
    sites = []
    for kind, side, point in pool:
        sites.append((len(template), side))
        template.append("He", point + side * normal * 1.5, coords_are_cartesian=True)
    perms = symmetry_permutations(template, sites, enabled=cfg["symmetry_reduce"],
        symprec_A=cfg["symmetry_symprec_A"], angle_tolerance_deg=cfg["symmetry_angle_tolerance_deg"],
        mapping_tolerance_A=cfg["symmetry_mapping_tolerance_A"])
    distances = template.distance_matrix[len(slab):, len(slab):]
    denominator = len(exposed)
    counts = []
    for family, key in (("H", "h_coverages_pct"), ("CO", "co_coverages_pct")):
        if family in cfg["state_families"]:
            for cov in realizable_coverages(denominator, cfg[key]):
                n = cov["count"]
                counts.append((n, 0) if family == "H" else (0, n))
    if "mixed-H-CO" in cfg["state_families"]:
        for cov in mixed_realizable_coverages(denominator, cfg["mixed_h_co_coverages_pct"]):
            counts.append((cov["n_o"], cov["n_oh"]))
    for nh, nc in dict.fromkeys(counts):
        family = "mixed-H-CO" if nh and nc else ("H" if nh else "CO")
        n = len(pool)
        if nh + nc > n:
            raise ValueError(f"Too few adsorption sites for {nh} H and {nc} CO")
        unique = {}
        examined = 0
        for h, co in _occupancies(n, nh, nc, cfg["max_raw_configurations_per_stoichiometry"]):
            examined += 1
            if any(distances[i, j] < cfg.get("metal_min_spacing_A", 1.5)
                   for i, j in combinations(h + co, 2)):
                continue
            labels = [0] * n
            for i in h:
                labels[i] = 1
            for i in co:
                labels[i] = 2
            pattern = canonical_pattern(labels, perms)
            unique[pattern] = pattern
        if not unique:
            raise ValueError(f"No admissible starting arrangement for {nh} H and {nc} CO. "
                             "Increase max_raw_configurations_per_stoichiometry or adjust metal_min_spacing_A.")
        selected = select_diverse_patterns(list(unique), sites, template,
                                          cap=cfg["max_arrangements_per_stoichiometry"])
        for arr, pattern in enumerate(selected, 1):
            candidate = slab.copy()
            props = {"selective_dynamics": [True] * 3} if "selective_dynamics" in slab.site_properties else {}
            indices = [i for i, label in enumerate(pattern) if label]
            for i in indices:
                kind, sign, point = pool[i]
                outward = sign * normal
                if pattern[i] == 1:
                    candidate.append("H", point + outward * cfg.get("h_adsorption_height_A", 1.35),
                                     coords_are_cartesian=True, properties=props)
                else:
                    carbon = point + outward * cfg.get("co_adsorption_height_A", 1.75)
                    candidate.append("C", carbon, coords_are_cartesian=True, properties=props)
                    candidate.append("O", carbon + outward * 1.13, coords_are_cartesian=True, properties=props)
            yield dict(common, state_id=f"H{nh:02d}_CO{nc:02d}_arr{arr:03d}", family=family,
                       delta_n_H=nh, delta_n_O=nc, n_CO=nc, proton_electron_pairs=nh,
                       arrangement_id=arr, site_indices=indices, site_sides=[pool[i][1] for i in indices],
                       adsorption_site_types=[pool[i][0] for i in indices],
                       actual_h_coverage_pct=100*nh/denominator, actual_co_coverage_pct=100*nc/denominator,
                       actual_coverage_pct=100*(nh+nc)/denominator,
                       raw_arrangements_total=comb(n, nh+nc)*comb(nh+nc, nh),
                       raw_arrangements_examined=examined, symmetry_unique_arrangements=len(unique),
                       symmetry_operations=len(perms), structure=candidate)


def _legacy_tilt_states(slab, exposed, normal, cfg, common):
    """Retain the original upright state and sample C-bound CO rotations."""
    tilt = float(cfg.get("co_tilt_deg", 0))
    cap = int(cfg.get("co_orientation_limit", 9))
    if not 0 <= tilt <= 90 or cap < 1:
        raise ValueError("co_tilt_deg must be 0–90 and co_orientation_limit at least 1")
    a = slab.lattice.matrix[0]
    u = a / np.linalg.norm(a)
    v = np.cross(normal, u)
    theta = np.deg2rad(tilt)
    for state in _upright_states(slab, exposed, normal, cfg, common):
        yield state
        if not tilt or not state["n_CO"] or cap == 1:
            continue
        structure = state["structure"]
        carbons = [i for i in range(len(slab), len(structure))
                   if structure[i].specie.symbol == "C"]
        # Base-five digits encode upright or four azimuths per molecule.
        # Sample the joint space without allocating 5**N geometries.
        total = 5 ** len(carbons)
        rng = random.Random(17)
        codes = list(range(1, total)) if total <= cap else []
        if total > cap:
            # Include collective tilts first, then independent molecular tilts.
            codes = [k * ((total - 1) // 4) for k in range(1, min(5, cap))]
            used = {0, *codes}
            while len(codes) < cap - 1:
                code = rng.randrange(1, total)
                if code not in used:
                    codes.append(code)
                    used.add(code)
        for code in codes:
            candidate = structure.copy()
            digits = code
            angles = []
            for i in carbons:
                digits, direction = divmod(digits, 5)
                angles.append([tilt if direction else 0, 90 * max(0, direction - 1)])
                if direction:
                    bond = structure[i + 1].coords - structure[i].coords
                    outward = bond / np.linalg.norm(bond)
                    tangent = (u, v, -u, -v)[direction - 1]
                    axis = np.cos(theta) * outward + np.sin(theta) * tangent
                    candidate.translate_sites([i + 1], np.linalg.norm(bond) * axis - bond,
                                              frac_coords=False, to_unit_cell=False)
            # Reject overlapping seeds, excluding each CO's own C–O bond.
            if any(candidate.get_distance(i + 1, j) < 1.1 for i in carbons
                   for j in range(len(candidate)) if j not in (i, i + 1)):
                continue
            yield dict(state, structure=candidate, co_seed_angles_deg=angles,
                       state_id=f"{state['state_id']}_tilt{tilt:g}_orient{code}")


def metal_states(slab, exposed, normal, cfg, common):
    """Place each CO end-on or side-on before unconstrained adsorbate relaxation."""
    if "co_binding_modes" not in cfg and cfg.get("co_tilt_deg", 0):
        yield from _legacy_tilt_states(slab, exposed, normal, cfg, common)
        return
    modes = cfg.get("co_binding_modes", ["C-down"])
    if not isinstance(modes, list) or not modes or set(modes) - {"C-down", "O-down", "side-on"}:
        raise ValueError("co_binding_modes must list C-down, O-down, or side-on")
    a = slab.lattice.matrix[0]
    u = a / np.linalg.norm(a)
    v = np.cross(normal, u)
    axes = {"side+a": u, "side-a": -u, "side+b": v, "side-b": -v}
    poses = [p for p in ("C-down", "O-down") if p in modes]
    if "side-on" in modes:
        poses.extend(axes)
    cap = int(cfg.get("co_orientation_limit", 9))
    if cap < len(poses):
        raise ValueError(f"co_orientation_limit needs at least {len(poses)} for these modes")
    for state in _upright_states(slab, exposed, normal, cfg, common):
        if not state["n_CO"]:
            yield state
            continue
        base = state["structure"]
        carbons = [i for i in range(len(slab), len(base)) if base[i].specie.symbol == "C"]
        # First cover each homogeneous geometry, then sample joint assignments.
        choices = len(poses)
        total = choices ** len(carbons)
        codes = [k * ((total - 1) // (choices - 1)) for k in range(choices)] if choices > 1 else [0]
        used = set(codes)
        rng = random.Random(17)
        while len(codes) < min(total, cap):
            code = rng.randrange(total)
            if code not in used:
                codes.append(code)
                used.add(code)
        for code in codes:
            candidate = base.copy()
            names = []
            for i in carbons:
                code, digit = divmod(code, choices)
                pose = poses[digit]
                names.append(pose)
                origin = base[i].coords
                bond = base[i + 1].coords - origin
                if pose == "C-down":
                    continue
                if pose == "O-down":
                    c_shift, o_shift = bond, -bond
                else:
                    half = axes[pose] * np.linalg.norm(bond) / 2
                    c_shift, o_shift = -half, half - bond
                for j, shift in ((i, c_shift), (i + 1, o_shift)):
                    candidate.translate_sites([j], shift, frac_coords=False, to_unit_cell=False)
            upright = all(name == "C-down" for name in names)
            if not upright and any(candidate.get_distance(k, j) < 1.1
                                   for i in carbons for k in (i, i + 1)
                                   for j in range(len(base)) if j not in (i, i + 1)):
                continue
            signature = "".join(str(("C-down", "O-down", *axes).index(name)) for name in names)
            suffix = "" if upright else "_pose" + signature
            yield dict(state, structure=candidate, co_starting_poses=names,
                       state_id=state["state_id"] + suffix)

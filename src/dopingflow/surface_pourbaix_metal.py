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


def metal_states(slab, exposed, normal, cfg, common):
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

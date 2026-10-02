from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def surface_normal(structure: Structure) -> np.ndarray:
    normal = np.cross(structure.lattice.matrix[0], structure.lattice.matrix[1])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("Surface cell vectors are degenerate")
    return np.asarray(normal, dtype=float) / norm


def infer_host_species(
    structure: Structure,
    anion_species: Sequence[str],
    explicit_host: str = "",
) -> str:
    explicit = str(explicit_host or "").strip()
    if explicit and any(site.specie.symbol == explicit for site in structure):
        return explicit
    anions = set(anion_species)
    counts: dict[str, int] = {}
    for site in structure:
        symbol = site.specie.symbol
        if symbol in anions or symbol == "H":
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
    if not counts:
        return ""
    return max(sorted(counts), key=lambda symbol: counts[symbol])


def infer_dopant_species(
    structure: Structure,
    *,
    anion_species: Sequence[str],
    host_species: str = "",
    requested_species: Sequence[str] = (),
) -> list[str]:
    present = {site.specie.symbol for site in structure}
    requested = [
        str(symbol).strip()
        for symbol in requested_species
        if str(symbol).strip() and str(symbol).strip() in present
    ]
    if requested:
        return list(dict.fromkeys(requested))
    host = infer_host_species(structure, anion_species, host_species)
    excluded = {*set(anion_species), "H"}
    if host:
        excluded.add(host)
    return sorted(symbol for symbol in present if symbol not in excluded)


def resolve_placement_side(
    structure: Structure,
    cfg: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    requested = str(cfg.get("placement_side", "dopant-nearest")).strip().lower()
    if requested in {"top", "bottom", "both"}:
        return requested, {
            "requested_placement_side": requested,
            "resolved_placement_side": requested,
            "side_target_species": list(cfg.get("side_target_species", [])),
            "top_dopant_depth_A": None,
            "bottom_dopant_depth_A": None,
            "side_selection_reason": "manual",
        }
    if requested not in {"dopant-nearest", "dopant_nearest", "near-dopants", "near_dopants"}:
        raise ValueError(
            "placement_side must be top, bottom, both, or dopant-nearest"
        )

    normal = surface_normal(structure)
    heavy = [
        float(np.dot(site.coords, normal))
        for site in structure
        if site.specie.symbol != "H"
    ]
    if not heavy:
        raise ValueError("Cannot determine slab sides from an empty/heavy-atom-free structure")
    lo, hi = min(heavy), max(heavy)

    targets = infer_dopant_species(
        structure,
        anion_species=list(cfg.get("anion_species", ["O"])),
        host_species=str(cfg.get("host_species", "")),
        requested_species=list(cfg.get("side_target_species", [])),
    )
    top_depths: list[float] = []
    bottom_depths: list[float] = []
    used: list[str] = []
    for symbol in targets:
        projections = [
            float(np.dot(site.coords, normal))
            for site in structure
            if site.specie.symbol == symbol
        ]
        if not projections:
            continue
        used.append(symbol)
        top_depths.append(min(max(hi - value, 0.0) for value in projections))
        bottom_depths.append(min(max(value - lo, 0.0) for value in projections))

    fallback = str(cfg.get("dopant_side_fallback", "top")).strip().lower()
    if fallback not in {"top", "bottom", "both"}:
        fallback = "top"
    if not used:
        return fallback, {
            "requested_placement_side": requested,
            "resolved_placement_side": fallback,
            "side_target_species": [],
            "top_dopant_depth_A": None,
            "bottom_dopant_depth_A": None,
            "side_selection_reason": "no-dopants-found-fallback",
        }

    top_score = float(np.mean(top_depths))
    bottom_score = float(np.mean(bottom_depths))
    tolerance = float(cfg.get("dopant_side_tie_tolerance_A", 0.25))
    if abs(top_score - bottom_score) <= tolerance:
        resolved = "both"
        reason = "dopant-proximity-tie"
    elif top_score < bottom_score:
        resolved = "top"
        reason = "dopants-nearer-top"
    else:
        resolved = "bottom"
        reason = "dopants-nearer-bottom"

    return resolved, {
        "requested_placement_side": requested,
        "resolved_placement_side": resolved,
        "side_target_species": used,
        "top_dopant_depth_A": top_score,
        "bottom_dopant_depth_A": bottom_score,
        "side_selection_reason": reason,
    }


def exposed_sites(
    structure: Structure,
    species: Sequence[str],
    *,
    normal: np.ndarray,
    placement_side: str,
    window_A: float,
    limit: int = 0,
) -> list[tuple[int, int]]:
    allowed = set(species)
    rows = [
        (i, float(np.dot(site.coords, normal)))
        for i, site in enumerate(structure)
        if site.specie.symbol in allowed
    ]
    if not rows:
        return []
    lo, hi = min(value for _, value in rows), max(value for _, value in rows)
    selected: list[tuple[int, int, float]] = []
    if placement_side in {"top", "both"}:
        selected.extend(
            (i, +1, hi - value)
            for i, value in rows
            if hi - value <= float(window_A) + 1e-12
        )
    if placement_side in {"bottom", "both"}:
        selected.extend(
            (i, -1, value - lo)
            for i, value in rows
            if value - lo <= float(window_A) + 1e-12
        )
    selected.sort(key=lambda item: (item[2], item[0], -item[1]))
    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for idx, sign, _ in selected:
        key = (idx, sign)
        if key in seen:
            continue
        result.append(key)
        seen.add(key)
        if int(limit) > 0 and len(result) >= int(limit):
            break
    return result


def coverage_count_options(n_sites: int, requested_pct: float) -> list[int]:
    """Map coverage to realizable site counts; keep both counts for exact half ties."""
    n_sites = int(n_sites)
    pct = float(requested_pct)
    if n_sites <= 0 or pct <= 0:
        return []
    target = n_sites * min(max(pct, 0.0), 100.0) / 100.0
    low = int(math.floor(target + 1e-12))
    high = int(math.ceil(target - 1e-12))
    candidates = sorted({max(0, min(n_sites, low)), max(0, min(n_sites, high))})
    candidates = [value for value in candidates if value > 0]
    if not candidates:
        return []
    distances = {value: abs(float(value) - target) for value in candidates}
    minimum = min(distances.values())
    return [
        value
        for value in candidates
        if abs(distances[value] - minimum) <= 1e-10
    ]


def realizable_coverages(
    n_sites: int,
    requested_percentages: Sequence[float],
) -> list[dict[str, Any]]:
    """Deduplicate requested coverages that map to the same finite-cell count."""
    by_count: dict[int, dict[str, Any]] = {}
    for requested in requested_percentages:
        for count in coverage_count_options(n_sites, float(requested)):
            item = by_count.setdefault(
                count,
                {
                    "count": count,
                    "actual_coverage_pct": 100.0 * count / float(n_sites),
                    "requested_coverages_pct": [],
                },
            )
            item["requested_coverages_pct"].append(float(requested))
    return [by_count[count] for count in sorted(by_count)]


def _periodic_distance_A(
    structure: Structure,
    frac_a: np.ndarray,
    frac_b: np.ndarray,
) -> float:
    delta = np.asarray(frac_a, dtype=float) - np.asarray(frac_b, dtype=float)
    delta -= np.round(delta)
    cart = np.dot(delta, structure.lattice.matrix)
    return float(np.linalg.norm(cart))


def symmetry_permutations(
    structure: Structure,
    sites: Sequence[tuple[int, int]],
    *,
    enabled: bool = True,
    symprec_A: float = 0.10,
    angle_tolerance_deg: float = 5.0,
    mapping_tolerance_A: float = 0.25,
) -> list[tuple[int, ...]]:
    """Return symmetry operations as permutations of the eligible surface-site list.

    Operations are obtained from the actual doped/vacancy slab, so a symmetry
    broken by a dopant is not used to eliminate chemically distinct patterns.
    Only operations mapping the full eligible site set onto itself are retained.
    """
    n_sites = len(sites)
    identity = tuple(range(n_sites))
    if not enabled or n_sites <= 1:
        return [identity]
    try:
        operations = SpacegroupAnalyzer(
            structure,
            symprec=float(symprec_A),
            angle_tolerance=float(angle_tolerance_deg),
        ).get_symmetry_operations(cartesian=False)
    except Exception:
        return [identity]

    candidate_frac = [
        np.mod(np.asarray(structure[idx].frac_coords, dtype=float), 1.0)
        for idx, _ in sites
    ]
    candidate_species = [structure[idx].specie.symbol for idx, _ in sites]
    permutations: set[tuple[int, ...]] = {identity}

    for operation in operations:
        mapped: list[int] = []
        used: set[int] = set()
        valid = True
        for pos, (idx, _sign) in enumerate(sites):
            transformed = np.mod(
                np.asarray(operation.operate(structure[idx].frac_coords), dtype=float),
                1.0,
            )
            symbol = candidate_species[pos]
            choices: list[tuple[float, int]] = []
            for target_pos, ((_, target_sign), target_frac, target_symbol) in enumerate(
                zip(sites, candidate_frac, candidate_species)
            ):
                if target_pos in used or target_symbol != symbol:
                    continue
                distance = _periodic_distance_A(structure, transformed, target_frac)
                if distance <= float(mapping_tolerance_A):
                    choices.append((distance, target_pos))
            if not choices:
                valid = False
                break
            _, target_pos = min(choices)
            mapped.append(target_pos)
            used.add(target_pos)
        if valid and len(mapped) == n_sites and len(set(mapped)) == n_sites:
            permutations.add(tuple(mapped))
    return sorted(permutations)


def canonical_pattern(
    labels: Sequence[int],
    permutations: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    labels = tuple(int(value) for value in labels)
    images: list[tuple[int, ...]] = []
    for permutation in permutations:
        if len(permutation) != len(labels):
            continue
        transformed = [0] * len(labels)
        for source, target in enumerate(permutation):
            transformed[int(target)] = labels[source]
        images.append(tuple(transformed))
    return min(images) if images else labels


def _combination_from_rank(n: int, k: int, rank: int) -> tuple[int, ...]:
    if k < 0 or k > n:
        raise ValueError("Invalid combination size")
    total = math.comb(n, k)
    if rank < 0 or rank >= total:
        raise ValueError("Combination rank out of range")
    result: list[int] = []
    next_value = 0
    remaining_rank = int(rank)
    for position in range(k):
        for value in range(next_value, n):
            left = k - position - 1
            count = math.comb(n - value - 1, left) if left >= 0 else 1
            if remaining_rank < count:
                result.append(value)
                next_value = value + 1
                break
            remaining_rank -= count
    return tuple(result)


def _sampled_combinations(n: int, k: int, max_raw: int) -> list[tuple[int, ...]]:
    total = math.comb(n, k)
    if total <= int(max_raw):
        return list(combinations(range(n), k))
    ranks = np.linspace(0, total - 1, int(max_raw), dtype=np.int64)
    return [_combination_from_rank(n, k, int(rank)) for rank in np.unique(ranks)]


def _pair_summary(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    array = np.asarray(values, dtype=float)
    return (
        float(array.min()),
        float(array.mean()),
        float(array.max()),
        float(array.std()),
    )


def _pattern_features(
    labels: Sequence[int],
    sites: Sequence[tuple[int, int]],
    structure: Structure,
    target_species: Sequence[str],
) -> np.ndarray:
    occupied = [i for i, label in enumerate(labels) if int(label) != 0]
    pair_distances = [
        float(structure.get_distance(sites[i][0], sites[j][0]))
        for ii, i in enumerate(occupied)
        for j in occupied[ii + 1 :]
    ]
    features: list[float] = list(_pair_summary(pair_distances))

    target_indices = [
        idx
        for idx, site in enumerate(structure)
        if site.specie.symbol in set(target_species)
    ]
    target_distances: list[float] = []
    if target_indices:
        for pos in occupied:
            base_idx = sites[pos][0]
            target_distances.append(
                min(
                    float(structure.get_distance(base_idx, target_idx))
                    for target_idx in target_indices
                )
            )
    features.extend(_pair_summary(target_distances))

    # Preserve chemically distinct adsorption on different underlying cations.
    site_species = sorted({structure[idx].specie.symbol for idx, _ in sites})
    for symbol in site_species:
        features.append(
            float(
                sum(
                    1
                    for pos in occupied
                    if structure[sites[pos][0]].specie.symbol == symbol
                )
            )
        )

    # Mixed O/OH patterns receive label-aware geometric descriptors.
    for label in sorted({int(value) for value in labels if int(value) != 0}):
        positions = [i for i, value in enumerate(labels) if int(value) == label]
        distances = [
            float(structure.get_distance(sites[i][0], sites[j][0]))
            for ii, i in enumerate(positions)
            for j in positions[ii + 1 :]
        ]
        features.extend(_pair_summary(distances))
    return np.asarray(features, dtype=float)


def select_diverse_patterns(
    patterns: Sequence[tuple[int, ...]],
    sites: Sequence[tuple[int, int]],
    structure: Structure,
    *,
    cap: int,
    target_species: Sequence[str] = (),
) -> list[tuple[int, ...]]:
    patterns = list(patterns)
    cap = int(cap)
    if len(patterns) <= cap:
        return patterns
    features = np.vstack(
        [
            _pattern_features(pattern, sites, structure, target_species)
            for pattern in patterns
        ]
    )
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (features - mean) / scale

    # Start with the configuration nearest to a target dopant if that descriptor
    # varies; otherwise start with the lexicographically first canonical pattern.
    if target_species and features.shape[1] >= 8:
        start = int(np.argmin(features[:, 5]))  # mean nearest-dopant distance
    else:
        start = min(range(len(patterns)), key=lambda i: patterns[i])
    selected = [start]
    remaining = set(range(len(patterns))) - {start}
    while remaining and len(selected) < cap:
        best = max(
            remaining,
            key=lambda i: (
                min(
                    float(np.linalg.norm(normalized[i] - normalized[j]))
                    for j in selected
                ),
                tuple(patterns[i]),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return [patterns[i] for i in selected]


def enumerate_binary_patterns(
    structure: Structure,
    sites: Sequence[tuple[int, int]],
    count: int,
    cfg: Mapping[str, Any],
) -> tuple[list[tuple[int, ...]], dict[str, int]]:
    n_sites = len(sites)
    count = int(count)
    if count <= 0 or count > n_sites:
        return [], {"raw_total": 0, "raw_examined": 0, "symmetry_unique": 0}
    max_raw = int(cfg.get("max_raw_configurations_per_stoichiometry", 100000))
    raw = _sampled_combinations(n_sites, count, max_raw)
    permutations = symmetry_permutations(
        structure,
        sites,
        enabled=bool(cfg.get("symmetry_reduce", True)),
        symprec_A=float(cfg.get("symmetry_symprec_A", 0.10)),
        angle_tolerance_deg=float(cfg.get("symmetry_angle_tolerance_deg", 5.0)),
        mapping_tolerance_A=float(cfg.get("symmetry_mapping_tolerance_A", 0.25)),
    )
    unique: dict[tuple[int, ...], tuple[int, ...]] = {}
    for combo in raw:
        labels = tuple(1 if i in set(combo) else 0 for i in range(n_sites))
        canonical = canonical_pattern(labels, permutations)
        unique.setdefault(canonical, canonical)
    target_species = infer_dopant_species(
        structure,
        anion_species=list(cfg.get("anion_species", ["O"])),
        host_species=str(cfg.get("host_species", "")),
        requested_species=list(cfg.get("side_target_species", [])),
    )
    selected = select_diverse_patterns(
        list(unique),
        sites,
        structure,
        cap=int(cfg.get("max_arrangements_per_stoichiometry", 8)),
        target_species=target_species,
    )
    return selected, {
        "raw_total": math.comb(n_sites, count),
        "raw_examined": len(raw),
        "symmetry_unique": len(unique),
        "symmetry_operations": len(permutations),
    }


def enumerate_mixed_patterns(
    structure: Structure,
    sites: Sequence[tuple[int, int]],
    n_o: int,
    n_oh: int,
    cfg: Mapping[str, Any],
) -> tuple[list[tuple[int, ...]], dict[str, int]]:
    n_sites = len(sites)
    n_o, n_oh = int(n_o), int(n_oh)
    total = n_o + n_oh
    if n_o < 0 or n_oh < 0 or total <= 0 or total > n_sites:
        return [], {"raw_total": 0, "raw_examined": 0, "symmetry_unique": 0}
    raw_total = math.comb(n_sites, total) * math.comb(total, n_oh)
    max_raw = int(cfg.get("max_raw_configurations_per_stoichiometry", 100000))
    permutations = symmetry_permutations(
        structure,
        sites,
        enabled=bool(cfg.get("symmetry_reduce", True)),
        symprec_A=float(cfg.get("symmetry_symprec_A", 0.10)),
        angle_tolerance_deg=float(cfg.get("symmetry_angle_tolerance_deg", 5.0)),
        mapping_tolerance_A=float(cfg.get("symmetry_mapping_tolerance_A", 0.25)),
    )

    unique: dict[tuple[int, ...], tuple[int, ...]] = {}
    examined = 0
    # For typical surface cells (10-20 sites) this is exact. A deterministic
    # cap prevents pathological combinatorics on very large cells.
    for occupied in combinations(range(n_sites), total):
        for oh_positions in combinations(range(total), n_oh):
            labels = [0] * n_sites
            oh_set = set(oh_positions)
            for local_pos, site_pos in enumerate(occupied):
                labels[site_pos] = 2 if local_pos in oh_set else 1  # 1=O*, 2=OH*
            canonical = canonical_pattern(labels, permutations)
            unique.setdefault(canonical, canonical)
            examined += 1
            if examined >= max_raw:
                break
        if examined >= max_raw:
            break

    target_species = infer_dopant_species(
        structure,
        anion_species=list(cfg.get("anion_species", ["O"])),
        host_species=str(cfg.get("host_species", "")),
        requested_species=list(cfg.get("side_target_species", [])),
    )
    selected = select_diverse_patterns(
        list(unique),
        sites,
        structure,
        cap=int(cfg.get("max_arrangements_per_stoichiometry", 8)),
        target_species=target_species,
    )
    return selected, {
        "raw_total": raw_total,
        "raw_examined": examined,
        "symmetry_unique": len(unique),
        "symmetry_operations": len(permutations),
    }


def mixed_realizable_coverages(
    n_sites: int,
    requested_pairs: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    by_counts: dict[tuple[int, int], dict[str, Any]] = {}
    for pair in requested_pairs:
        if len(pair) != 2:
            continue
        requested_o, requested_oh = float(pair[0]), float(pair[1])
        for n_o in coverage_count_options(n_sites, requested_o):
            for n_oh in coverage_count_options(n_sites, requested_oh):
                if n_o + n_oh > n_sites:
                    continue
                key = (n_o, n_oh)
                item = by_counts.setdefault(
                    key,
                    {
                        "n_o": n_o,
                        "n_oh": n_oh,
                        "actual_o_coverage_pct": 100.0 * n_o / float(n_sites),
                        "actual_oh_coverage_pct": 100.0 * n_oh / float(n_sites),
                        "requested_pairs_pct": [],
                    },
                )
                item["requested_pairs_pct"].append((requested_o, requested_oh))
    return [by_counts[key] for key in sorted(by_counts)]

from __future__ import annotations

from typing import Sequence

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def build_sublattice_symmetry_permutations(
    parent: Structure,
    sublattice_indices: Sequence[int],
    *,
    symprec: float,
    angle_tolerance: float = 5.0,
) -> list[np.ndarray]:
    """Return deterministic symmetry permutations for a selected sublattice."""
    sga = SpacegroupAnalyzer(
        parent,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    ops = sga.get_symmetry_operations(cartesian=False)
    frac = np.asarray([parent[i].frac_coords for i in sublattice_indices], dtype=float)
    n_sites = len(frac)

    def match_index(coord: np.ndarray) -> int:
        wrapped = coord % 1.0
        delta = frac - wrapped
        delta -= np.round(delta)
        dist2 = np.sum(delta * delta, axis=1)
        match = int(np.argmin(dist2))
        if dist2[match] > (float(symprec) * 10.0) ** 2:
            raise RuntimeError(
                "Failed to match a symmetry-mapped sublattice site "
                f"(minimum fractional distance squared={dist2[match]:.3g})."
            )
        return match

    permutations: list[np.ndarray] = []
    seen: set[bytes] = set()
    for operation in ops:
        rotation = np.asarray(operation.rotation_matrix, dtype=float)
        translation = np.asarray(operation.translation_vector, dtype=float)
        permutation = np.empty(n_sites, dtype=np.int32)
        for index in range(n_sites):
            permutation[index] = match_index(rotation.dot(frac[index]) + translation)
        key = permutation.tobytes()
        if key not in seen:
            seen.add(key)
            permutations.append(permutation)
    return permutations


def canonical_occupancy_key(
    labels: np.ndarray | Sequence[int], permutations: Sequence[np.ndarray]
) -> bytes:
    """Canonicalize a sublattice label vector under supplied permutations."""
    array = np.asarray(labels, dtype=np.int8)
    if not permutations:
        return array.tobytes()
    return min(array[permutation].tobytes() for permutation in permutations)

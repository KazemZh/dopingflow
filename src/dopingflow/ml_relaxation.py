from __future__ import annotations

from typing import Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def get_optimizer_class(name: str):
    name = str(name).strip().lower()

    if name == "bfgs":
        from ase.optimize import BFGS
        return BFGS
    if name == "lbfgs":
        from ase.optimize import LBFGS
        return LBFGS
    if name == "fire":
        from ase.optimize import FIRE
        return FIRE
    if name == "mdmin":
        from ase.optimize import MDMin
        return MDMin
    if name == "quasinewton":
        from ase.optimize import QuasiNewton
        return QuasiNewton

    raise ValueError(f"Unsupported optimizer: {name}")


def final_fmax(forces: np.ndarray) -> float:
    if forces.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(forces, axis=1)))


def structure_energy_with_calculator(struct: Structure, calculator) -> float:
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calculator
    return float(atoms.get_potential_energy())

def build_relaxation_target(atoms, relax_mode: str, cell_filter: str):
    relax_mode = str(relax_mode).strip().lower()
    cell_filter = str(cell_filter).strip().lower()

    if relax_mode == "atoms":
        return atoms

    if cell_filter == "frechet":
        from ase.filters import FrechetCellFilter as CellFilter
    elif cell_filter == "unit":
        from ase.filters import UnitCellFilter as CellFilter
    elif cell_filter == "exp":
        from ase.filters import ExpCellFilter as CellFilter
    else:
        raise ValueError(f"Unsupported cell_filter: {cell_filter}")

    if relax_mode == "full":
        return CellFilter(atoms)

    if relax_mode == "isotropic":
        return CellFilter(atoms, hydrostatic_strain=True)

    if relax_mode == "volume":
        from ase.constraints import FixScaled
        atoms.set_constraint([
            FixScaled(i, mask=(True, True, True))
            for i in range(len(atoms))
        ])
        return CellFilter(atoms, hydrostatic_strain=True)

    if relax_mode == "shape":
        return CellFilter(atoms, constant_volume=True)

    if relax_mode == "xy":
        return CellFilter(
            atoms,
            mask=[True, True, False, False, False, False],
        )

    if relax_mode == "cell_only":
        from ase.constraints import FixScaled
        atoms.set_constraint([
            FixScaled(i, mask=(True, True, True))
            for i in range(len(atoms))
        ])
        return CellFilter(atoms)

    raise ValueError(f"Unsupported relax_mode: {relax_mode}")

def relax_structure_with_calculator(
    struct: Structure,
    *,
    calculator,
    optimizer_name: str,
    fmax: float,
    max_steps: int,
    relax_mode: str = "atoms",
    cell_filter: str = "frechet",
) -> Tuple[Structure, float, int, float, bool]:

    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calculator

    Optimizer = get_optimizer_class(optimizer_name)

    target = build_relaxation_target(
        atoms,
        relax_mode=relax_mode,
        cell_filter=cell_filter,
    )

    dyn = Optimizer(target, logfile=None)
    dyn.run(fmax=fmax, steps=max_steps)

    e_rel = float(atoms.get_potential_energy())

    target_forces = np.asarray(target.get_forces())
    fmax_final = final_fmax(target_forces)
    converged = bool(fmax_final <= fmax)

    try:
        nsteps = int(dyn.get_number_of_steps())
    except Exception:
        nsteps = -1

    s_rel = AseAtomsAdaptor.get_structure(atoms)
    return s_rel, e_rel, nsteps, fmax_final, converged
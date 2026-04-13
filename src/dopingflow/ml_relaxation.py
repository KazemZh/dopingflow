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


def relax_structure_with_calculator(
    struct: Structure,
    *,
    calculator,
    optimizer_name: str,
    fmax: float,
    max_steps: int,
) -> Tuple[Structure, float, int, float, bool]:
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calculator

    Optimizer = get_optimizer_class(optimizer_name)
    dyn = Optimizer(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=max_steps)

    e_rel = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces())
    fmax_final = final_fmax(forces)
    converged = bool(fmax_final <= fmax)

    try:
        nsteps = int(dyn.get_number_of_steps())
    except Exception:
        nsteps = -1

    s_rel = AseAtomsAdaptor.get_structure(atoms)
    return s_rel, e_rel, nsteps, fmax_final, converged
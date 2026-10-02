from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

KB_EV_K = 8.617333262145e-5
LN10 = math.log(10.0)


def value_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(math.floor((float(stop) - float(start)) / float(step) + 1e-10))
    values = float(start) + float(step) * np.arange(count + 1, dtype=float)
    if values[-1] < float(stop) - 1e-9:
        values = np.append(values, float(stop))
    return values


def to_she_potential(u: float, *, potential_scale: str, temperature_K: float, pH: float) -> float:
    scale = str(potential_scale).upper()
    if scale == "SHE":
        return float(u)
    if scale == "RHE":
        return float(u) - KB_EV_K * float(temperature_K) * LN10 * float(pH)
    raise ValueError("potential_scale must be SHE or RHE")


def surface_state_delta_g_eV(
    state_energy_eV: float,
    clean_energy_eV: float,
    delta_n_H: int,
    delta_n_O: int,
    h2_energy_eV: float,
    h2o_energy_eV: float,
    applied_potential_V: float,
    pH: float,
    *,
    potential_scale: str = "SHE",
    temperature_K: float = 298.15,
) -> float:
    """CHE free energy for clean + dO H2O + x(H+ + e-) -> state.

    x = dH - 2*dO. This covers protonated lattice oxygen, O*, OH*, H2O*,
    and mixed O/OH states with one expression.
    """
    x = int(delta_n_H) - 2 * int(delta_n_O)
    u_she = to_she_potential(
        applied_potential_V, potential_scale=potential_scale,
        temperature_K=temperature_K, pH=pH,
    )
    ph_term = KB_EV_K * float(temperature_K) * LN10 * float(pH)
    zero = (
        float(state_energy_eV) - float(clean_energy_eV)
        - int(delta_n_O) * float(h2o_energy_eV)
        - 0.5 * x * float(h2_energy_eV)
    )
    return float(zero + x * (u_she + ph_term))


def build_surface_pourbaix_grid(
    states: Sequence[Mapping[str, Any]],
    *,
    h2_energy_eV: float,
    h2o_energy_eV: float,
    potential_values: Sequence[float],
    pH_values: Sequence[float],
    potential_scale: str,
    temperature_K: float,
    energy_key: str,
    energy_level: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = []
    for state in states:
        if "pourbaix_eligible" in state and not bool(state["pourbaix_eligible"]):
            continue
        try:
            energy = float(state[energy_key])
            d_h, d_o = int(state["delta_n_H"]), int(state["delta_n_O"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(energy):
            usable.append((state, energy, d_h, d_o))
    clean = next((item for item in usable if item[0].get("family") == "clean"), None)
    if clean is None:
        raise ValueError("A clean state with a finite energy is required")
    clean_energy = clean[1]
    gaps = {str(item[0]["state_id"]): math.inf for item in usable}
    rows = []
    for pH in pH_values:
        for potential in potential_values:
            evaluated = []
            for state, energy, d_h, d_o in usable:
                dg = surface_state_delta_g_eV(
                    energy, clean_energy, d_h, d_o, h2_energy_eV, h2o_energy_eV,
                    float(potential), float(pH), potential_scale=potential_scale,
                    temperature_K=temperature_K,
                )
                evaluated.append((state, dg))
            best_state, best_dg = min(evaluated, key=lambda item: (item[1], str(item[0]["state_id"])))
            for state, dg in evaluated:
                sid = str(state["state_id"])
                gaps[sid] = min(gaps[sid], float(dg - best_dg))
            rows.append(dict(
                applied_potential_V=float(potential), pH=float(pH),
                potential_scale=str(potential_scale).upper(), temperature_K=float(temperature_K),
                stable_state_id=str(best_state["state_id"]),
                stable_family=str(best_state.get("final_family", best_state["family"])),
                deltaG_stable_eV=float(best_dg), delta_n_H=int(best_state["delta_n_H"]),
                delta_n_O=int(best_state["delta_n_O"]),
                proton_electron_pairs=int(best_state["proton_electron_pairs"]),
                energy_level=str(energy_level),
            ))
    gap_frame = pd.DataFrame([
        dict(state_id=state_id, minimum_deltaG_above_stable_eV=value)
        for state_id, value in sorted(gaps.items())
    ])
    return pd.DataFrame(rows), gap_frame


def summarize_stable_domains(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    total, rows = len(grid), []
    for keys, group in grid.groupby(["stable_state_id", "stable_family", "energy_level"], sort=False, dropna=False):
        rows.append(dict(
            state_id=keys[0], family=keys[1], energy_level=keys[2],
            n_grid_points=int(len(group)), grid_fraction=float(len(group) / total),
            pH_min_stable=float(group["pH"].min()), pH_max_stable=float(group["pH"].max()),
            potential_min_V_stable=float(group["applied_potential_V"].min()),
            potential_max_V_stable=float(group["applied_potential_V"].max()),
        ))
    return pd.DataFrame(rows)

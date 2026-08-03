#!/usr/bin/env python3
"""Plot compact Level-1 static-lattice vacancy thermodynamic outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from plot_vacancy_analysis import (
    _vacancy_count_color,
    plot_grand_potential_lines,
    plot_potential_vs_count,
    plot_preferred_count,
    plot_stability_map,
)


STATIC_NOTE = (
    "Static-lattice approximation: temperature and pressure enter only through "
    "the oxygen reservoir.\nSolid vibrational and entropic contributions are neglected."
)


def plot_pressure_map(
    pressure: pd.DataFrame, composition: str, temperature_K: float, output: Path
) -> None:
    data = pressure[pressure["actual_composition_key"].astype(str) == composition]
    if data.empty:
        raise ValueError(f"Composition {composition!r} is absent from the pressure map")
    temperatures = np.sort(data["temperature_K"].dropna().unique().astype(float))
    pressures = np.sort(
        data["log10_oxygen_partial_pressure_bar"].dropna().unique().astype(float)
    )
    counts = np.sort(data["best_n_vacancies"].dropna().unique().astype(int))
    count_index = {count: index for index, count in enumerate(counts)}
    grid = np.full((len(temperatures), len(pressures)), np.nan)
    for row in data.itertuples():
        if pd.isna(row.best_n_vacancies):
            continue
        y_index = int(np.where(temperatures == float(row.temperature_K))[0][0])
        x_index = int(
            np.where(pressures == float(row.log10_oxygen_partial_pressure_bar))[0][0]
        )
        grid[y_index, x_index] = count_index[int(row.best_n_vacancies)]
    colormap = ListedColormap([_vacancy_count_color(int(count)) for count in counts])
    boundaries = np.arange(len(counts) + 1) - 0.5
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=(pressures[0], pressures[-1], temperatures[0], temperatures[-1]),
        interpolation="nearest",
        cmap=colormap,
        norm=BoundaryNorm(boundaries, colormap.N),
    )
    handles = [
        Patch(
            facecolor=_vacancy_count_color(int(count)),
            edgecolor="black",
            label=str(int(count)),
        )
        for count in counts
    ]
    axis.legend(
        handles=handles,
        title="Stable vacancy count",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    axis.axhline(
        temperature_K,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Selected T={temperature_K:g} K",
    )
    first = data.iloc[0]
    approximation_text = str(first.get("pressure_mapping_approximation", ""))
    approximate = str(first.get("pressure_mapping_is_approximate", "")).lower() in {
        "true",
        "1",
    } or "omitted" in approximation_text.lower()
    mode = str(first.get("oxygen_standard_state_mode", "")).strip().lower()
    if mode == "nist_shomate":
        gas_note = "O2 standard state: NIST Shomate, 1 bar; explicit ZPE excluded."
    elif mode == "user_table":
        gas_note = "O2 standard state: user-supplied thermal correction."
    else:
        gas_note = str(first.get("pressure_mapping_approximation", ""))
    title_prefix = "Approximate " if approximate else ""
    axis.set(
        xlabel=r"$\log_{10}(p_{O_2}/\mathrm{bar})$",
        ylabel="Temperature (K)",
        title=(
            f"{title_prefix}T–pO2 stability — {composition}\n"
            f"{STATIC_NOTE}\n{gas_note}"
        ),
    )
    figure.savefig(output, dpi=300)
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--minima", type=Path, required=True)
    result.add_argument("--intervals", type=Path, required=True)
    result.add_argument("--best-counts", type=Path, required=True)
    result.add_argument("--pressure-map", type=Path, required=True)
    result.add_argument("--composition", required=True)
    result.add_argument("--delta-mu-o", type=float, required=True)
    result.add_argument("--temperature", type=float, required=True)
    result.add_argument("--x-dopant")
    result.add_argument("--output-dir", type=Path, default=Path("vacancy_static_plots"))
    return result


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    minima = pd.read_csv(args.minima)
    intervals = pd.read_csv(args.intervals)
    best = pd.read_csv(args.best_counts)
    pressure = pd.read_csv(args.pressure_map)
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.composition
    )
    plot_grand_potential_lines(
        minima, intervals, args.composition,
        args.output_dir / f"static_grand_potential_lines_{safe}.png",
    )
    plot_preferred_count(
        best, args.delta_mu_o, args.x_dopant,
        args.output_dir / "static_preferred_count_vs_doping.png",
    )
    plot_stability_map(
        intervals, args.x_dopant,
        args.output_dir / "static_vacancy_stability_map.png",
    )
    plot_pressure_map(
        pressure, args.composition, args.temperature,
        args.output_dir / f"static_pressure_stability_{safe}_{args.temperature:g}K.png",
    )
    plot_potential_vs_count(
        minima, args.composition, args.delta_mu_o,
        args.output_dir / f"static_grand_potential_vs_count_{safe}.png",
    )


if __name__ == "__main__":
    main()

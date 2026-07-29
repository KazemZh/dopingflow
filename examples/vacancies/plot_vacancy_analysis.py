#!/usr/bin/env python3
"""Create publication-ready plots from compact vacancy-analysis CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _metadata_label(frame: pd.DataFrame) -> str:
    first = frame.iloc[0]
    mode = first.get("oxygen_reference_mode", "unknown reference")
    verified = first.get("oxygen_reference_verified", "")
    model = "/".join(
        str(first.get(key, ""))
        for key in ("backend", "model", "task")
        if str(first.get(key, ""))
    )
    return f"O reference: {mode} (verified={verified}); model: {model}"


def plot_grand_potential_lines(
    minima: pd.DataFrame, intervals: pd.DataFrame, composition: str, output: Path
) -> None:
    data = minima[minima["actual_composition_key"] == composition].sort_values("n_vacancies")
    if data.empty:
        raise ValueError(f"Composition {composition!r} is absent from minima")
    if data["grand_potential_intercept_eV"].isna().all():
        raise ValueError("Grand-potential plotting requires an oxygen reference")
    if intervals.empty:
        x_min, x_max = -3.0, 0.0
    else:
        selected = intervals[intervals["actual_composition_key"] == composition]
        x_min = float(selected["delta_mu_O_lower_eV"].min())
        x_max = float(selected["delta_mu_O_upper_eV"].max())
    x = np.linspace(x_min, x_max, 500)
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    values = []
    for item in data.itertuples():
        y = float(item.grand_potential_intercept_eV) + int(item.n_vacancies) * x
        values.append(y)
        axis.plot(x, y, lw=1.4, label=f"n={int(item.n_vacancies)}")
    axis.plot(x, np.min(np.vstack(values), axis=0), color="black", lw=3, label="lower envelope")
    selected = intervals[intervals["actual_composition_key"] == composition]
    boundaries = selected.get("delta_mu_O_upper_eV", pd.Series(dtype=float))
    for boundary in sorted(set(boundaries.dropna()))[:-1]:
        axis.axvline(boundary, color="0.5", ls="--", lw=0.8)
    axis.set(
        xlabel=r"$\Delta\mu_O$ (eV per O atom)",
        ylabel=r"$\Delta\Omega$ (eV)",
        title=f"Oxygen grand potential — {composition}\n{_metadata_label(data)}",
    )
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_preferred_count(
    best: pd.DataFrame, delta_mu: float, x_dopant: str | None, output: Path
) -> None:
    selected = best[np.isclose(best["delta_mu_O_eV"], delta_mu)].copy()
    selected = selected[~selected["is_tied"].astype(str).str.lower().isin({"true", "1"})]
    if selected.empty:
        raise ValueError(f"No unique best-count rows at delta_mu_O={delta_mu:g} eV")
    x_column = f"percent_{x_dopant}" if x_dopant else "total_dopant_percent"
    if x_column not in selected:
        raise ValueError(f"Column {x_column!r} is absent")
    other = sorted(
        column
        for column in selected
        if column.startswith("percent_")
        and column != x_column
        and selected[column].nunique() > 1
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if other:
        group_column = other[0]
        for value, group in selected.groupby(group_column):
            group = group.sort_values(x_column)
            axis.plot(
                group[x_column],
                group["best_n_vacancies"],
                marker="o",
                label=f"{group_column.removeprefix('percent_')}={value:g}%",
            )
        axis.legend(fontsize=8)
    else:
        selected = selected.sort_values(x_column)
        axis.plot(selected[x_column], selected["best_n_vacancies"], marker="o")
    label = x_dopant or "total dopant"
    axis.set(
        xlabel=f"Actual {label} percentage (%)",
        ylabel="Preferred oxygen-vacancy count",
        title=rf"Preferred vacancy count at $\Delta\mu_O={delta_mu:g}$ eV"
        + f"\n{_metadata_label(selected)}",
    )
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_stability_map(
    intervals: pd.DataFrame, x_dopant: str | None, output: Path
) -> None:
    if intervals.empty:
        raise ValueError("Stability interval table is empty")
    x_column = f"percent_{x_dopant}" if x_dopant else "total_dopant_percent"
    compositions = (
        intervals[["actual_composition_key", x_column]]
        .drop_duplicates()
        .sort_values([x_column, "actual_composition_key"])
    )
    y = np.linspace(
        float(intervals["delta_mu_O_lower_eV"].min()),
        float(intervals["delta_mu_O_upper_eV"].max()),
        400,
    )
    grid = np.full((len(y), len(compositions)), np.nan)
    for column, composition in enumerate(compositions["actual_composition_key"]):
        subset = intervals[intervals["actual_composition_key"] == composition]
        for item in subset.itertuples():
            mask = (y >= item.delta_mu_O_lower_eV) & (y <= item.delta_mu_O_upper_eV)
            grid[mask, column] = item.stable_n_vacancies
    fig, axis = plt.subplots(
        figsize=(max(7.2, 0.45 * len(compositions)), 4.8),
        constrained_layout=True,
    )
    image = axis.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=(-0.5, len(compositions) - 0.5, y[0], y[-1]),
        interpolation="nearest",
    )
    axis.set_xticks(
        range(len(compositions)),
        compositions["actual_composition_key"],
        rotation=45,
        ha="right",
    )
    axis.set(
        xlabel="Actual composition",
        ylabel=r"$\Delta\mu_O$ (eV per O atom)",
        title=f"Stable oxygen-vacancy count\n{_metadata_label(intervals)}",
    )
    fig.colorbar(image, ax=axis, label="Stable vacancy count")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_potential_vs_count(
    minima: pd.DataFrame, composition: str, delta_mu: float, output: Path
) -> None:
    data = minima[minima["actual_composition_key"] == composition].sort_values("n_vacancies").copy()
    data["delta_grand_potential_eV"] = (
        data["grand_potential_intercept_eV"]
        + data["n_vacancies"] * delta_mu
    )
    minimum = data["delta_grand_potential_eV"].min()
    colors = [
        "tab:red" if np.isclose(value, minimum) else "tab:blue"
        for value in data["delta_grand_potential_eV"]
    ]
    fig, axis = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    axis.plot(data["n_vacancies"], data["delta_grand_potential_eV"], color="0.5", lw=1)
    axis.scatter(data["n_vacancies"], data["delta_grand_potential_eV"], c=colors, s=55, zorder=3)
    axis.set(
        xlabel="Number of oxygen vacancies",
        ylabel=r"$\Delta\Omega$ (eV)",
        title=rf"{composition} at $\Delta\mu_O={delta_mu:g}$ eV"
        + f"\n{_metadata_label(data)}",
    )
    fig.savefig(output, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minima", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--best-counts", type=Path, required=True)
    parser.add_argument("--composition", required=True)
    parser.add_argument("--delta-mu-o", type=float, required=True)
    parser.add_argument("--x-dopant")
    parser.add_argument("--output-dir", type=Path, default=Path("vacancy_plots"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    minima = pd.read_csv(args.minima)
    intervals = pd.read_csv(args.intervals)
    best = pd.read_csv(args.best_counts)
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.composition
    )
    plot_grand_potential_lines(
        minima,
        intervals,
        args.composition,
        args.output_dir / f"grand_potential_lines_{safe}.png",
    )
    plot_preferred_count(
        best,
        args.delta_mu_o,
        args.x_dopant,
        args.output_dir / "preferred_count_vs_doping.png",
    )
    plot_stability_map(intervals, args.x_dopant, args.output_dir / "vacancy_stability_map.png")
    plot_potential_vs_count(
        minima,
        args.composition,
        args.delta_mu_o,
        args.output_dir / f"grand_potential_vs_count_{safe}.png",
    )


if __name__ == "__main__":
    main()

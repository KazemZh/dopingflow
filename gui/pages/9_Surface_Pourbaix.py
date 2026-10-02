"""Configure, run, and inspect electrochemical surface-state / Pourbaix analysis."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import toml

from dopingflow.surface_pourbaix import (
    preview_surface_pourbaix_targets,
    resolve_surface_pourbaix_output_dir,
)
from gui_config import BACKEND_CHOICES, DEVICE_CHOICES, OPTIMIZER_CHOICES

st.set_page_config(page_title="Surface Pourbaix", layout="wide")
st.title("Electrochemical surface states / Pourbaix")
st.caption(
    "Starting from segregated intact surfaces, sample protonated, O*, OH*, H2O*, and mixed O/OH "
    "states, then determine the stable surface state versus potential and pH with the CHE."
)

project_root = Path(
    st.sidebar.text_input("Project root", value=str(Path.cwd()), key="surface_pourbaix_project_root")
).expanduser().resolve()
config_path = project_root / "input.toml"
if not config_path.exists():
    st.error(f"No input.toml found at {config_path}")
    st.stop()

cfg = toml.load(str(config_path))
saved = dict(cfg.get("surface_pourbaix", {}) or {})
saved_screen = dict(saved.get("screen", {}) or {})
saved_dft = dict(saved.get("dft", {}) or {})


def _csv(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ", ".join(str(x) for x in (value or []))


def _items(text: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in text.split(",") if x.strip()))


def _ints(text: str) -> list[int]:
    return sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))


def _run(args: list[str]) -> int:
    st.code(" ".join(args))
    box = st.empty()
    lines: list[str] = []
    process = subprocess.Popen(
        args, cwd=str(project_root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-40:]))
    code = process.wait()
    if code == 0:
        st.success("Surface-Pourbaix stage finished successfully.")
    else:
        st.error(f"Surface-Pourbaix stage exited with return code {code}.")
    return code


with st.expander("Configuration & run controls", expanded=True):
    enabled = st.checkbox("Enable surface-Pourbaix stage", value=bool(saved.get("enabled", False)))

    st.subheader("Input surfaces")
    c1, c2, c3 = st.columns(3)
    source_modes = ["auto", "segregation", "final-selected", "refine-summary", "screen-selected", "screen-summary"]
    saved_mode = str(saved.get("source_mode", "auto"))
    source_mode = c1.selectbox(
        "Surface source", source_modes,
        index=source_modes.index(saved_mode) if saved_mode in source_modes else 0,
        help="auto prefers surface-segregation output when it exists.",
    )
    source_summary = c2.text_input(
        "Explicit source CSV (optional)", value=str(saved.get("source_summary", ""))
    ).strip()
    max_surfaces = c3.number_input(
        "Maximum surfaces", min_value=1, value=int(saved.get("max_surfaces", 10)), step=1
    )
    surface_include = st.text_input(
        "Surface selectors (optional, comma-separated/globs)",
        value=_csv(saved.get("surface_include", [])),
    )

    preview_cfg = dict(cfg)
    preview_section = dict(saved)
    preview_section.update(
        enabled=True, source_mode=source_mode, source_summary=source_summary,
        surface_include=_items(surface_include), max_surfaces=int(max_surfaces),
    )
    preview_cfg["surface_pourbaix"] = preview_section
    try:
        preview = preview_surface_pourbaix_targets(preview_cfg, project_root)
        st.caption(f"Selected source table: {preview.attrs.get('source_summary', '')}")
        st.dataframe(preview, width="stretch", hide_index=True)
    except Exception as exc:
        st.warning(str(exc))

    st.subheader("Surface-state search")
    families = st.multiselect(
        "State families",
        ["clean", "protonated", "O", "OH", "H2O", "mixed-O-OH"],
        default=list(saved.get(
            "state_families",
            ["clean", "protonated", "O", "OH", "H2O", "mixed-O-OH"],
        )),
    )
    s1, s2, s3, s4 = st.columns(4)
    sides = ["top", "bottom", "both"]
    saved_side = str(saved.get("placement_side", "top"))
    placement_side = s1.selectbox(
        "Surface side", sides, index=sides.index(saved_side) if saved_side in sides else 0
    )
    surface_window = s2.number_input(
        "Exposed-site window (Å)", min_value=0.1,
        value=float(saved.get("surface_window_A", 2.0)), step=0.1,
    )
    max_o_sites = s3.number_input(
        "Max surface O sites", min_value=1,
        value=int(saved.get("max_surface_oxygen_sites", 8)), step=1,
    )
    max_cat_sites = s4.number_input(
        "Max surface cation sites", min_value=1,
        value=int(saved.get("max_surface_cation_sites", 8)), step=1,
    )
    s5, s6, s7 = st.columns(3)
    h_counts_text = s5.text_input("H counts", value=_csv(saved.get("h_counts", [1, 2, 3, 4])))
    ads_counts_text = s6.text_input(
        "O/OH/H2O counts", value=_csv(saved.get("adsorbate_counts", [1, 2]))
    )
    max_arrangements = s7.number_input(
        "Max arrangements / stoichiometry", min_value=1,
        value=int(saved.get("max_arrangements_per_stoichiometry", 8)), step=1,
    )
    st.caption(
        "Mixed O/OH defaults to one O* + one OH*. Edit mixed_compositions in input.toml "
        "for additional mixed stoichiometries."
    )

    st.subheader("Electrochemical grid")
    p1, p2, p3, p4 = st.columns(4)
    ph_min = p1.number_input("pH minimum", value=float(saved.get("pH_min", -1.0)), step=0.1)
    ph_max = p2.number_input("pH maximum", value=float(saved.get("pH_max", 3.0)), step=0.1)
    ph_step = p3.number_input(
        "pH step", min_value=0.01, value=float(saved.get("pH_step", 0.1)), step=0.05
    )
    potential_scale = p4.selectbox(
        "Potential scale", ["SHE", "RHE"],
        index=0 if str(saved.get("potential_scale", "SHE")).upper() == "SHE" else 1,
    )
    u1, u2, u3, u4 = st.columns(4)
    u_min = u1.number_input(
        "Potential minimum (V)", value=float(saved.get("potential_min_V", 0.0)), step=0.1
    )
    u_max = u2.number_input(
        "Potential maximum (V)", value=float(saved.get("potential_max_V", 2.0)), step=0.1
    )
    u_step = u3.number_input(
        "Potential step (V)", min_value=0.01,
        value=float(saved.get("potential_step_V", 0.05)), step=0.01,
    )
    temperature = u4.number_input(
        "Temperature (K)", min_value=1.0,
        value=float(saved.get("temperature_K", 298.15)), step=1.0,
    )
    if potential_scale == "RHE":
        st.info(
            "At fixed RHE potential, ideal CHE H+/e− terms are pH-independent. "
            "Use SHE for the conventional pH-sloped Pourbaix representation."
        )

    st.subheader("ML screening")
    m1, m2, m3, m4 = st.columns(4)
    saved_backend = str(saved_screen.get("backend", "mace"))
    backend = m1.selectbox(
        "Backend", BACKEND_CHOICES,
        index=BACKEND_CHOICES.index(saved_backend) if saved_backend in BACKEND_CHOICES else 0,
    )
    model = m2.text_input("Model", value=str(saved_screen.get("model", "mh-1")))
    task = m3.text_input("Task", value=str(saved_screen.get("task", "matpes_r2scan")))
    saved_device = str(saved_screen.get("device", "cpu"))
    device = m4.selectbox(
        "Device", DEVICE_CHOICES,
        index=DEVICE_CHOICES.index(saved_device) if saved_device in DEVICE_CHOICES else 0,
    )
    m5, m6, m7 = st.columns(3)
    saved_optimizer = str(saved_screen.get("optimizer", "bfgs"))
    optimizer = m5.selectbox(
        "Optimizer", OPTIMIZER_CHOICES,
        index=OPTIMIZER_CHOICES.index(saved_optimizer)
        if saved_optimizer in OPTIMIZER_CHOICES else 0,
    )
    fmax = m6.number_input(
        "fmax (eV/Å)", min_value=0.001, value=float(saved_screen.get("fmax", 0.05)), step=0.01
    )
    max_steps = m7.number_input(
        "Max relaxation steps", min_value=1,
        value=int(saved_screen.get("max_steps", 300)), step=10,
    )

    with st.expander("Optional DFT refinement"):
        dft_enabled = st.checkbox(
            "Refine competitive surface states with GPAW",
            value=bool(saved_dft.get("enabled", False)),
        )
        d1, d2, d3, d4 = st.columns(4)
        dft_execute = d1.checkbox(
            "Execute missing GPAW calculations", value=bool(saved_dft.get("execute", False))
        )
        dft_ecut = d2.number_input(
            "PW cutoff (eV)", min_value=100.0,
            value=float(saved_dft.get("ecut_eV", 500.0)), step=50.0,
        )
        dft_kpts = d3.text_input(
            "k-points", value="x".join(str(x) for x in saved_dft.get("kpts", [3, 3, 1]))
        )
        dft_window = d4.number_input(
            "ML candidate window (eV)", min_value=0.0,
            value=float(saved_dft.get("candidate_window_eV", 0.30)), step=0.05,
        )
        dft_cap = st.number_input(
            "Max DFT states / surface", min_value=1,
            value=int(saved_dft.get("max_states_per_surface", 20)), step=1,
        )

    section = dict(saved)
    section.update(
        enabled=enabled, source_mode=source_mode, source_summary=source_summary,
        surface_include=_items(surface_include), max_surfaces=int(max_surfaces),
        placement_side=placement_side, state_families=families,
        h_counts=_ints(h_counts_text), adsorbate_counts=_ints(ads_counts_text),
        surface_window_A=float(surface_window), max_surface_oxygen_sites=int(max_o_sites),
        max_surface_cation_sites=int(max_cat_sites),
        max_arrangements_per_stoichiometry=int(max_arrangements),
        pH_min=float(ph_min), pH_max=float(ph_max), pH_step=float(ph_step),
        potential_scale=potential_scale, potential_min_V=float(u_min),
        potential_max_V=float(u_max), potential_step_V=float(u_step),
        temperature_K=float(temperature),
    )
    section["screen"] = {
        **saved_screen, "backend": backend, "model": model, "task": task,
        "device": device, "optimizer": optimizer, "fmax": float(fmax),
        "max_steps": int(max_steps), "relax": True,
    }
    try:
        kpts = [int(x) for x in dft_kpts.lower().replace("x", ",").split(",") if x.strip()]
    except ValueError:
        kpts = [3, 3, 1]
    section["dft"] = {
        **saved_dft, "enabled": dft_enabled, "execute": dft_execute,
        "ecut_eV": float(dft_ecut), "kpts": kpts,
        "candidate_window_eV": float(dft_window),
        "max_states_per_surface": int(dft_cap),
    }

    b1, b2, b3 = st.columns(3)
    if b1.button("Save settings", type="primary"):
        cfg["surface_pourbaix"] = section
        with config_path.open("w", encoding="utf-8") as handle:
            toml.dump(cfg, handle)
        st.success(f"Saved {config_path}")
    if b2.button("Dry run"):
        cfg["surface_pourbaix"] = section
        with config_path.open("w", encoding="utf-8") as handle:
            toml.dump(cfg, handle)
        _run(["dopingflow", "surface-pourbaix", "-c", str(config_path), "--dry-run"])
    if b3.button("Run surface-Pourbaix"):
        cfg["surface_pourbaix"] = section
        with config_path.open("w", encoding="utf-8") as handle:
            toml.dump(cfg, handle)
        _run(["dopingflow", "surface-pourbaix", "-c", str(config_path)])

st.divider()
st.subheader("Results")
try:
    resolved = resolve_surface_pourbaix_output_dir(cfg, project_root)
    grid_path = resolved / "pourbaix_grid.csv"
    states_path = resolved / "stable_surface_states.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        surface_ids = list(dict.fromkeys(grid["surface_id"].astype(str)))
        selected = st.selectbox("Surface", surface_ids)
        view = grid[grid["surface_id"].astype(str) == selected].copy()
        fig = px.scatter(
            view, x="pH", y="applied_potential_V", color="stable_state_id",
            title=f"Stable surface state: {selected}",
            labels={"applied_potential_V": f"Potential (V vs {view['potential_scale'].iloc[0]})"},
        )
        fig.update_traces(marker={"size": 9, "symbol": "square"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No surface-Pourbaix results yet.")
    if states_path.exists():
        st.dataframe(pd.read_csv(states_path), width="stretch", hide_index=True)
except Exception as exc:
    st.warning(str(exc))

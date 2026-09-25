"""Configure, preview, run, and inspect dopant leaching from selected surfaces."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import toml

from dopingflow.leaching import (
    parse_leaching_config,
    preview_leaching_sites,
    resolve_leaching_output_dir,
)
from gui_config import BACKEND_CHOICES, DEVICE_CHOICES, OPTIMIZER_CHOICES


st.set_page_config(page_title="Dopant leaching", layout="wide")
st.title("Dopant leaching")
st.caption(
    "Remove dopants from previously selected surfaces, relax the dopant-vacancy slab, "
    "calculate a metal-referenced extraction energy, and optionally evaluate a "
    "simple electrochemical dissolution model with user-supplied redox data."
)

project_root = Path(
    st.sidebar.text_input("Project root", value=str(Path.cwd()), key="leaching_project_root")
).expanduser().resolve()
config_path = project_root / "input.toml"
if not config_path.exists():
    st.error(f"No input.toml found at {config_path}")
    st.stop()

cfg = toml.load(str(config_path))
saved = dict(cfg.get("leaching", {}) or {})
saved_protonation = dict(saved.get("protonation", {}) or {})
surface = dict(cfg.get("surface", {}) or {})
refine = dict(surface.get("refine", {}) or {})
screen = dict(surface.get("screen", {}) or {})


def _csv(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ", ".join(str(x) for x in (value or []))


def _items(value: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in value.split(",") if x.strip()))


def _json_map(label: str, value: Any, help_text: str) -> tuple[dict[str, Any], str | None]:
    text = st.text_area(
        label,
        value=json.dumps(dict(value or {}), indent=2),
        help=help_text,
        height=120,
    )
    try:
        data = json.loads(text or "{}")
        if not isinstance(data, dict):
            raise ValueError("must be a JSON object")
        return data, None
    except (ValueError, json.JSONDecodeError) as exc:
        return dict(value or {}), f"{label}: {exc}"


def _default_calculator() -> dict[str, Any]:
    source = refine if bool(refine.get("enabled", False)) else screen
    if bool(refine.get("enabled", False)):
        fallback = {
            "backend": "mace", "model": "mh-1", "task": "matpes_r2scan",
            "device": "cpu", "gpu_id": 0, "optimizer": "bfgs",
            "fmax": 0.03, "max_steps": 500,
        }
    else:
        fallback = {
            "backend": "grace", "model": "GRACE-1L-OMAT", "task": "",
            "device": "cpu", "gpu_id": 0, "optimizer": "bfgs",
            "fmax": 0.05, "max_steps": 300,
        }
    fallback.update(source)
    return fallback


def _run(args: list[str]) -> None:
    with st.spinner("Running leaching stage..."):
        completed = subprocess.run(
            args,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
    st.session_state["leaching_stdout"] = completed.stdout
    st.session_state["leaching_stderr"] = completed.stderr
    st.session_state["leaching_returncode"] = completed.returncode
    if completed.returncode == 0:
        st.success("Leaching command finished successfully.")
    else:
        st.error(f"Leaching command exited with return code {completed.returncode}.")


with st.expander("Configuration & run controls", expanded=True):
    enabled = st.checkbox("Enable leaching stage", value=bool(saved.get("enabled", False)))

    st.subheader("Surface source and output")
    source_modes = ["auto", "final-selected", "screen-selected", "refine-summary", "screen-summary"]
    current_mode = str(saved.get("source_mode", "auto"))
    if current_mode not in source_modes:
        current_mode = "auto"

    inherited_source_root = (
        str(saved.get("source_root", "")).strip()
        or str(surface.get("source_root", "")).strip()
        or str((cfg.get("conductivity", {}) or {}).get("source_root", "")).strip()
        or str((cfg.get("oxidation", {}) or {}).get("source_root", "")).strip()
        or str((cfg.get("structure", {}) or {}).get("outdir", "random_structures")).strip()
    )

    c1, c2 = st.columns(2)
    source_root = c1.text_input(
        "Parent / source root",
        value=inherited_source_root,
        help=(
            "Relative leaching output is created inside this directory. "
            "By default it inherits the same source root used by the surface workflow."
        ),
    ).strip()
    source_mode = c2.selectbox("Surface table", source_modes, index=source_modes.index(current_mode))

    c3, c4 = st.columns(2)
    source_summary = c3.text_input(
        "Explicit surface CSV (optional)",
        value=str(saved.get("source_summary", "")),
        help=(
            "Overrides Surface table when set. Relative paths are resolved inside Parent / source root."
        ),
    ).strip()
    outdir = c4.text_input(
        "Output directory",
        value=str(saved.get("outdir", "09_leaching")),
        help="Relative paths are created inside Parent / source root; absolute paths are used exactly as entered.",
    ).strip()

    surface_include = _items(
        st.text_input(
            "Surface/target selector(s)",
            value=_csv(saved.get("surface_include", [])),
            help="Optional exact IDs or wildcards. Empty means every row in the selected surface table.",
        )
    )
    dopants = _items(
        st.text_input(
            "Dopant species",
            value=_csv(saved.get("dopant_species", [])),
            placeholder="Sb, Ti",
            help="Empty means infer dopants from the surface metadata.",
        )
    )
    q1, q2, q3 = st.columns(3)
    zones = q1.multiselect(
        "Depth zones",
        ["surface", "subsurface", "bulk"],
        default=[x for x in saved.get("zones", ["surface"]) if x in {"surface", "subsurface", "bulk"}]
        or ["surface"],
    )
    placement_side = q2.selectbox(
        "Surface side",
        ["top", "bottom", "both"],
        index=["top", "bottom", "both"].index(
            str(saved.get("placement_side", surface.get("placement_side", "top")))
            if str(saved.get("placement_side", surface.get("placement_side", "top"))) in {"top", "bottom", "both"}
            else "top"
        ),
    )
    max_sites = int(
        q3.number_input(
            "Max sites / surface / dopant",
            min_value=1,
            value=int(saved.get("max_sites_per_surface_species", 12)),
            step=1,
        )
    )

    g1, g2, g3 = st.columns(3)
    layer_tol = float(
        g1.number_input(
            "Cation-layer tolerance (Å)",
            min_value=0.01,
            value=float(saved.get("cation_layer_tolerance_A", surface.get("cation_layer_tolerance_A", 0.8))),
            step=0.05,
        )
    )
    layers_per_zone = int(
        g2.number_input(
            "Cation layers / zone",
            min_value=1,
            value=int(saved.get("layers_per_zone", surface.get("layers_per_zone", 1))),
            step=1,
        )
    )
    oxygen_cutoff = float(
        g3.number_input(
            "O-neighbor cutoff (Å)",
            min_value=0.1,
            value=float(saved.get("oxygen_neighbor_cutoff_A", 2.8)),
            step=0.1,
        )
    )

    st.subheader("Removal calculator")
    defaults = _default_calculator()
    r1, r2, r3 = st.columns(3)
    backend_default = str(saved.get("backend", defaults["backend"])).lower()
    backend = r1.selectbox(
        "Backend",
        BACKEND_CHOICES,
        index=BACKEND_CHOICES.index(backend_default) if backend_default in BACKEND_CHOICES else 0,
    )
    device_default = str(saved.get("device", defaults["device"])).lower()
    device = r2.selectbox(
        "Device",
        DEVICE_CHOICES,
        index=DEVICE_CHOICES.index(device_default) if device_default in DEVICE_CHOICES else 0,
    )
    gpu_id = int(
        r3.number_input(
            "GPU ID",
            min_value=0,
            value=int(saved.get("gpu_id", defaults.get("gpu_id", 0))),
            step=1,
            disabled=device != "cuda",
        )
    )
    m1, m2 = st.columns(2)
    model = m1.text_input("Model / checkpoint", value=str(saved.get("model", defaults["model"]))).strip()
    task = m2.text_input("Task / head", value=str(saved.get("task", defaults.get("task", "")))).strip()

    s1, s2, s3, s4 = st.columns(4)
    optimizer_default = str(saved.get("optimizer", defaults.get("optimizer", "bfgs"))).lower()
    optimizer = s1.selectbox(
        "Optimizer",
        OPTIMIZER_CHOICES,
        index=OPTIMIZER_CHOICES.index(optimizer_default) if optimizer_default in OPTIMIZER_CHOICES else 0,
    )
    fmax = float(
        s2.number_input(
            "fmax (eV/Å)",
            min_value=0.001,
            value=float(saved.get("fmax", defaults.get("fmax", 0.03))),
            step=0.005,
            format="%.3f",
        )
    )
    max_steps = int(
        s3.number_input(
            "Max steps",
            min_value=1,
            value=int(saved.get("max_steps", defaults.get("max_steps", 500))),
            step=25,
        )
    )
    relax_removed = s4.checkbox(
        "Relax removed slab",
        value=bool(saved.get("relax_removed_surface", True)),
    )

    t1, t2, t3, t4 = st.columns(4)
    reuse_parent = t1.checkbox(
        "Reuse compatible surface energy",
        value=bool(saved.get("reuse_surface_energy", True)),
    )
    relax_parent = t2.checkbox(
        "Relax parent if recomputed",
        value=bool(saved.get("relax_parent_if_recomputed", True)),
    )
    inherit_fixed = t3.checkbox(
        "Reuse surface fixed-atom rule",
        value=bool(saved.get("inherit_surface_fixed_atoms", True)),
    )
    resume_completed = t4.checkbox(
        "Resume completed sites",
        value=bool(saved.get("resume_completed", True)),
        help=(
            "Reuse compatible completed per-site leaching checkpoints after an interrupted run. "
            "Incomplete or failed sites are calculated again."
        ),
    )

    st.subheader("Post-leaching protonation (CHE)")
    st.caption(
        "Optionally protonate O atoms that were bonded to the removed dopant, relax those "
        "post-leaching structures, and reference the added H to H₂ using the computational "
        "hydrogen electrode (CHE). Bare-vacancy results are always retained for comparison."
    )
    protonation_enabled = st.checkbox(
        "Enable post-leaching protonation",
        value=bool(saved_protonation.get("enabled", False)),
    )

    p1, p2, p3, p4 = st.columns(4)
    saved_h_counts = [
        int(x) for x in saved_protonation.get("h_counts", [0, 1, 2, 3])
        if int(x) in range(0, 7)
    ]
    protonation_h_counts = p1.multiselect(
        "H counts to test",
        options=list(range(0, 7)),
        default=saved_h_counts or [0, 1, 2, 3],
        help="0 is the bare dopant-vacancy state. H atoms are added only to O neighbors of the removed dopant.",
        disabled=not protonation_enabled,
    )
    if 0 not in protonation_h_counts:
        protonation_h_counts = [0, *protonation_h_counts]

    protonation_neighbor_cutoff = float(
        p2.number_input(
            "Protonatable O cutoff (Å)",
            min_value=0.1,
            value=float(
                saved_protonation.get(
                    "neighbor_cutoff_A",
                    saved.get("oxygen_neighbor_cutoff_A", 2.8),
                )
            ),
            step=0.1,
            disabled=not protonation_enabled,
        )
    )
    protonation_oh_length = float(
        p3.number_input(
            "Initial O–H length (Å)",
            min_value=0.5,
            value=float(saved_protonation.get("oh_bond_length_A", 0.98)),
            step=0.01,
            format="%.2f",
            disabled=not protonation_enabled,
        )
    )
    protonation_max_arrangements = int(
        p4.number_input(
            "Max arrangements / H count",
            min_value=1,
            value=int(saved_protonation.get("max_arrangements_per_h_count", 5)),
            step=1,
            disabled=not protonation_enabled,
        )
    )

    pp1, pp2, pp3 = st.columns(3)
    relax_protonated = pp1.checkbox(
        "Relax protonated slabs",
        value=bool(saved_protonation.get("relax_protonated_surface", True)),
        disabled=not protonation_enabled,
    )
    compute_h2 = pp2.checkbox(
        "Calculate H₂ reference",
        value=bool(saved_protonation.get("compute_h2_reference", True)),
        help="Uses the same leaching MLFF and caches the H₂ energy.",
        disabled=not protonation_enabled,
    )
    relax_h2 = pp3.checkbox(
        "Relax H₂ reference",
        value=bool(saved_protonation.get("relax_h2_reference", True)),
        disabled=(not protonation_enabled) or (not compute_h2),
    )

    with st.expander("Advanced H₂ reference settings", expanded=False):
        hp1, hp2, hp3 = st.columns(3)
        manual_h2_default = saved_protonation.get("manual_h2_energy_eV", "")
        if manual_h2_default is None:
            manual_h2_default = ""
        manual_h2_text = hp1.text_input(
            "Manual E(H₂) (eV, optional)",
            value=str(manual_h2_default),
            help="If supplied, this overrides the calculated H₂ reference.",
            disabled=not protonation_enabled,
        ).strip()
        h2_bond_length = float(
            hp2.number_input(
                "Initial H–H length (Å)",
                min_value=0.2,
                value=float(saved_protonation.get("h2_bond_length_A", 0.74)),
                step=0.01,
                format="%.2f",
                disabled=(not protonation_enabled) or bool(manual_h2_text),
            )
        )
        h2_box = float(
            hp3.number_input(
                "H₂ box size (Å)",
                min_value=5.0,
                value=float(saved_protonation.get("h2_box_A", 15.0)),
                step=1.0,
                disabled=(not protonation_enabled) or bool(manual_h2_text),
            )
        )

    manual_h2_error = None
    manual_h2_value: float | str = ""
    if manual_h2_text:
        try:
            manual_h2_value = float(manual_h2_text)
        except ValueError:
            manual_h2_error = "Manual E(H₂) must be a number or left empty."
            st.error(manual_h2_error)

    st.subheader("Metal reference")
    a1, a2, a3 = st.columns(3)
    reference_file = a1.text_input(
        "Reference-energy cache",
        value=str(saved.get("reference_energies_file", "reference_structures/reference_energies.json")),
    ).strip()
    metals_dir = a2.text_input(
        "Metal POSCAR directory",
        value=str(saved.get("metals_dir", "reference_structures/metals")),
    ).strip()
    compute_missing = a3.checkbox(
        "Calculate missing metal references",
        value=bool(saved.get("compute_missing_metal_references", True)),
    )
    relax_metal = st.checkbox(
        "Relax metal reference",
        value=bool(saved.get("relax_metal_reference", False)),
    )

    st.subheader("Optional electrochemical dissolution")
    st.info(
        "Extraction energy does not require aqueous data. Only enter a standard potential "
        "when the chosen dissolved species and electron count are chemically appropriate."
    )
    j1, j2 = st.columns(2)
    with j1:
        oxidation_states, err1 = _json_map(
            "Oxidation states / electron counts",
            saved.get("oxidation_states", {}),
            'JSON object, e.g. {"Sb": 3}.',
        )
        aqueous_species, err2 = _json_map(
            "Aqueous species labels",
            saved.get("aqueous_species", {}),
            'JSON object, e.g. {"Sb": "chosen Sb aqueous species"}.',
        )
    with j2:
        standard_potentials, err3 = _json_map(
            "Standard reduction potentials (V vs SHE)",
            saved.get("standard_reduction_potentials_V_SHE", {}),
            "Use validated values for the exact redox reaction being modeled.",
        )
        ion_activities, err4 = _json_map(
            "Ion activities",
            saved.get("ion_activities", {}),
            'JSON object, e.g. {"Sb": 1e-6}.',
        )
    redox_error = next((x for x in (err1, err2, err3, err4) if x), None)
    if redox_error:
        st.error(redox_error)

    e1, e2, e3, e4 = st.columns(4)
    default_activity = float(
        e1.number_input(
            "Default ion activity",
            min_value=1e-20,
            value=float(saved.get("default_ion_activity", 1e-6)),
            format="%.2e",
        )
    )
    temperature = float(
        e2.number_input("Temperature (K)", min_value=1.0, value=float(saved.get("temperature_K", 298.15)))
    )
    pH = float(e3.number_input("pH", value=float(saved.get("pH", 0.0)), step=0.5))
    scale = e4.selectbox(
        "Potential scale",
        ["RHE", "SHE"],
        index=0 if str(saved.get("potential_scale", "RHE")).upper() == "RHE" else 1,
    )
    potentials = [
        float(x)
        for x in _items(
            st.text_input(
                "Operating potentials (V)",
                value=_csv(saved.get("potentials_V", [1.23, 1.50, 1.70])),
            )
        )
    ]

    resolved = dict(saved)
    resolved.update(
        enabled=bool(enabled),
        source_root=source_root,
        source_mode=source_mode,
        source_summary=source_summary,
        surface_include=surface_include,
        dopant_species=dopants,
        zones=zones,
        placement_side=placement_side,
        cation_layer_tolerance_A=layer_tol,
        layers_per_zone=layers_per_zone,
        max_sites_per_surface_species=max_sites,
        oxygen_neighbor_cutoff_A=oxygen_cutoff,
        outdir=outdir,
        backend=backend,
        model=model,
        task=task,
        device=device,
        gpu_id=gpu_id,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        reuse_surface_energy=bool(reuse_parent),
        relax_parent_if_recomputed=bool(relax_parent),
        relax_removed_surface=bool(relax_removed),
        inherit_surface_fixed_atoms=bool(inherit_fixed),
        resume_completed=bool(resume_completed),
        protonation={
            "enabled": bool(protonation_enabled),
            "h_counts": sorted(set(int(x) for x in protonation_h_counts) | {0}),
            "neighbor_cutoff_A": protonation_neighbor_cutoff,
            "oh_bond_length_A": protonation_oh_length,
            "max_arrangements_per_h_count": protonation_max_arrangements,
            "relax_protonated_surface": bool(relax_protonated),
            "manual_h2_energy_eV": manual_h2_value,
            "compute_h2_reference": bool(compute_h2),
            "relax_h2_reference": bool(relax_h2),
            "h2_bond_length_A": h2_bond_length,
            "h2_box_A": h2_box,
        },
        reference_energies_file=reference_file,
        metals_dir=metals_dir,
        compute_missing_metal_references=bool(compute_missing),
        relax_metal_reference=bool(relax_metal),
        oxidation_states=oxidation_states,
        standard_reduction_potentials_V_SHE=standard_potentials,
        aqueous_species=aqueous_species,
        ion_activities=ion_activities,
        default_ion_activity=default_activity,
        temperature_K=temperature,
        pH=pH,
        potential_scale=scale,
        potentials_V=potentials,
    )
    resolved_cfg = dict(cfg)
    resolved_cfg["leaching"] = resolved

    validation_error = redox_error or manual_h2_error
    if validation_error is None:
        try:
            parse_leaching_config(resolved_cfg, project_root)
        except Exception as exc:
            validation_error = str(exc)
            st.error(validation_error)

    with st.expander("Preview exact dopant-removal sites", expanded=False):
        if validation_error is None:
            try:
                preview = preview_leaching_sites(resolved_cfg, project_root)
            except Exception as exc:
                st.warning(str(exc))
            else:
                st.caption(f"{len(preview)} atom-removal job(s).")
                columns = [
                    c for c in (
                        "target_id", "miller_h", "miller_k", "miller_l", "termination_id",
                        "variant_label", "dopant", "site_index", "detected_zone",
                        "oxygen_coordination_within_cutoff", "nearest_oxygen_distance_A",
                    ) if c in preview.columns
                ]
                st.dataframe(preview[columns], use_container_width=True, hide_index=True)

    with st.expander("Preview [leaching] TOML", expanded=False):
        st.code(toml.dumps({"leaching": resolved}), language="toml")

    save_col, dry_col, run_col = st.columns(3)
    with save_col:
        save = st.button(
            "Save leaching settings",
            type="primary",
            use_container_width=True,
            disabled=validation_error is not None,
        )
    with dry_col:
        dry = st.button(
            "Run leaching dry-run",
            use_container_width=True,
            disabled=(not enabled) or validation_error is not None,
        )
    with run_col:
        run = st.button(
            "Run leaching",
            use_container_width=True,
            disabled=(not enabled) or validation_error is not None,
        )

    if not enabled:
        st.caption(
            "Enable **leaching stage** to activate the dry-run and leaching run actions."
        )

    if save or dry or run:
        cfg["leaching"] = resolved
        config_path.write_text(toml.dumps(cfg), encoding="utf-8")
        st.success(f"Saved {config_path}")
        if dry:
            _run(["dopingflow", "leaching", "-c", str(config_path), "--dry-run"])
        elif run:
            _run(["dopingflow", "leaching", "-c", str(config_path)])


if "leaching_stdout" in st.session_state:
    with st.expander("Last command output", expanded=False):
        if st.session_state.get("leaching_stdout"):
            st.code(st.session_state["leaching_stdout"])
        if st.session_state.get("leaching_stderr"):
            st.code(st.session_state["leaching_stderr"])


st.divider()
st.subheader("Results")
try:
    parsed = parse_leaching_config(cfg, project_root)
    result_dir = resolve_leaching_output_dir(cfg, parsed, project_root)
except Exception:
    fallback_source = (
        str(saved.get("source_root", "")).strip()
        or str(surface.get("source_root", "")).strip()
        or str((cfg.get("structure", {}) or {}).get("outdir", "random_structures")).strip()
    )
    source_path = Path(fallback_source).expanduser()
    if not source_path.is_absolute():
        source_path = (project_root / source_path).resolve()
    result_dir = Path(saved.get("outdir", "09_leaching")).expanduser()
    if not result_dir.is_absolute():
        result_dir = (source_path / result_dir).resolve()

summary_path = result_dir / "leaching_summary.csv"
aggregate_path = result_dir / "leaching_surface_summary.csv"
potential_path = result_dir / "leaching_potential_scan.csv"
protonation_summary_path = result_dir / "leaching_protonation_summary.csv"
protonation_scan_path = result_dir / "leaching_protonation_potential_scan.csv"

if not summary_path.exists():
    st.info("No leaching_summary.csv exists yet.")
else:
    results = pd.read_csv(summary_path)
    tabs = st.tabs(
        ["Site results", "Surface summary", "Potential scan", "Protonation", "Interpretation"]
    )

    with tabs[0]:
        st.dataframe(results, use_container_width=True, hide_index=True)
        if "extraction_energy_eV" in results.columns:
            plot = results.copy()
            plot["extraction_energy_eV"] = pd.to_numeric(plot["extraction_energy_eV"], errors="coerce")
            plot = plot[plot["extraction_energy_eV"].notna()]
            if not plot.empty:
                fig = px.scatter(
                    plot,
                    x="dopant",
                    y="extraction_energy_eV",
                    color="initial_dopant_zone" if "initial_dopant_zone" in plot.columns else None,
                    hover_data=[c for c in ("target_id", "variant_label", "site_index", "initial_dopant_zone", "surface_variant_declared_zone", "initial_depth_from_selected_surface_A") if c in plot.columns],
                    title="Metal-referenced dopant extraction energy",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if aggregate_path.exists():
            st.dataframe(pd.read_csv(aggregate_path), use_container_width=True, hide_index=True)
        else:
            st.info("No aggregate table found.")

    with tabs[2]:
        selected_scale = str(
            (cfg.get("leaching", {}) or {}).get("potential_scale", "RHE")
        ).upper()
        threshold_col = (
            "dissolution_potential_V_RHE"
            if selected_scale == "RHE"
            else "dissolution_potential_V_SHE"
        )

        if threshold_col in results.columns:
            threshold = results.copy()
            threshold[threshold_col] = pd.to_numeric(
                threshold[threshold_col], errors="coerce"
            )
            threshold = threshold[threshold[threshold_col].notna()]
            if not threshold.empty:
                st.markdown("#### Dissolution threshold potential")
                st.caption(
                    "Lower threshold potential means that the modeled dopant dissolution "
                    "becomes thermodynamically favorable at a lower applied potential."
                )
                fig_threshold = px.scatter(
                    threshold,
                    x="dopant",
                    y=threshold_col,
                    color=(
                        "initial_dopant_zone"
                        if "initial_dopant_zone" in threshold.columns
                        else None
                    ),
                    hover_data=[
                        col
                        for col in (
                            "target_id",
                            "variant_label",
                            "site_index",
                            "initial_dopant_zone",
                            "initial_depth_from_selected_surface_A",
                            "extraction_energy_eV",
                        )
                        if col in threshold.columns
                    ],
                    title=f"Dopant dissolution threshold potential vs {selected_scale}",
                    labels={
                        threshold_col: f"Dissolution potential (V vs {selected_scale})",
                        "dopant": "Dopant",
                    },
                )
                st.plotly_chart(fig_threshold, use_container_width=True)

        if potential_path.exists():
            try:
                scan = pd.read_csv(potential_path)
            except pd.errors.EmptyDataError:
                scan = pd.DataFrame()
            if scan.empty:
                st.info(
                    "No electrochemical ΔG_leach scan is available. Complete redox data "
                    "(oxidation state/electron count and standard reduction potential) "
                    "must be supplied for the dopant."
                )
            else:
                st.markdown("#### Leaching free energy at operating potentials")
                st.caption(
                    "ΔG_leach < 0 means dissolution is thermodynamically favorable in "
                    "the current simple-ion model; ΔG_leach > 0 means it is unfavorable."
                )
                st.dataframe(scan, use_container_width=True, hide_index=True)
                fig = px.line(
                    scan,
                    x="applied_potential_V",
                    y="deltaG_leach_eV",
                    color="dopant",
                    line_group="surface_id",
                    markers=True,
                    hover_data=[
                        col
                        for col in (
                            "target_id",
                            "site_index",
                            "initial_dopant_zone",
                            "initial_depth_from_selected_surface_A",
                        )
                        if col in scan.columns
                    ],
                    title=f"Leaching free energy vs applied potential ({selected_scale})",
                    labels={
                        "applied_potential_V": f"Applied potential (V vs {selected_scale})",
                        "deltaG_leach_eV": "ΔG_leach (eV)",
                        "dopant": "Dopant",
                    },
                )
                fig.add_hline(
                    y=0.0,
                    line_dash="dash",
                    annotation_text="ΔG_leach = 0",
                    annotation_position="top left",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No potential scan found.")

    with tabs[3]:
        selected_scale = str(
            (cfg.get("leaching", {}) or {}).get("potential_scale", "RHE")
        ).upper()
        adjusted_threshold_col = (
            "protonation_adjusted_dissolution_potential_V_RHE"
            if selected_scale == "RHE"
            else "protonation_adjusted_dissolution_potential_V_SHE"
        )
        bare_threshold_col = (
            "dissolution_potential_V_RHE"
            if selected_scale == "RHE"
            else "dissolution_potential_V_SHE"
        )

        if protonation_summary_path.exists():
            try:
                protonation_table = pd.read_csv(protonation_summary_path)
            except pd.errors.EmptyDataError:
                protonation_table = pd.DataFrame()
        else:
            protonation_table = pd.DataFrame()

        if protonation_table.empty:
            st.info(
                "No post-leaching protonation results are available. Enable "
                "**post-leaching protonation** and run the leaching stage."
            )
        else:
            st.markdown("#### Protonated post-leaching structures")
            st.caption(
                "ΔE_protonation(0 V) = E(defect+nH) − E(defect) − n/2 E(H₂). "
                "Negative values mean protonation stabilizes the dopant-vacancy surface "
                "relative to the bare vacancy at 0 V vs SHE and pH 0."
            )
            st.dataframe(
                protonation_table,
                use_container_width=True,
                hide_index=True,
            )

            structural = protonation_table.copy()
            structural["deltaE_protonation_zeroV_eV"] = pd.to_numeric(
                structural.get("deltaE_protonation_zeroV_eV"),
                errors="coerce",
            )
            structural["h_count"] = pd.to_numeric(
                structural.get("h_count"), errors="coerce"
            )
            structural = structural[
                structural["deltaE_protonation_zeroV_eV"].notna()
                & structural["h_count"].notna()
            ]
            if not structural.empty:
                structural["site_key"] = (
                    structural["surface_id"].astype(str)
                    + "::"
                    + structural["dopant"].astype(str)
                    + "::"
                    + structural["site_index"].astype(str)
                )
                fig_prot = px.line(
                    structural,
                    x="h_count",
                    y="deltaE_protonation_zeroV_eV",
                    color="dopant",
                    line_group="site_key",
                    markers=True,
                    hover_data=[
                        col
                        for col in (
                            "surface_id",
                            "site_index",
                            "initial_dopant_zone",
                            "arrangement_id",
                            "oxygen_indices_json",
                        )
                        if col in structural.columns
                    ],
                    title="Post-leaching protonation stabilization",
                    labels={
                        "h_count": "Number of H atoms",
                        "deltaE_protonation_zeroV_eV": "ΔE protonation at 0 V (eV)",
                        "dopant": "Dopant",
                    },
                )
                fig_prot.add_hline(
                    y=0.0,
                    line_dash="dash",
                    annotation_text="Bare vacancy reference",
                    annotation_position="top left",
                )
                st.plotly_chart(fig_prot, use_container_width=True)

        if protonation_scan_path.exists():
            try:
                proton_scan = pd.read_csv(protonation_scan_path)
            except pd.errors.EmptyDataError:
                proton_scan = pd.DataFrame()
        else:
            proton_scan = pd.DataFrame()

        if not proton_scan.empty:
            st.markdown("#### Protonation-adjusted leaching free energy")
            st.caption(
                "At each operating potential, DopingFlow compares the bare vacancy with "
                "all successfully relaxed protonated states and reports the lowest "
                "ΔG_leach. The selected H count can therefore change with potential."
            )
            st.dataframe(proton_scan, use_container_width=True, hide_index=True)
            proton_scan["site_key"] = (
                proton_scan["surface_id"].astype(str)
                + "::"
                + proton_scan["dopant"].astype(str)
                + "::"
                + proton_scan["site_index"].astype(str)
            )
            fig_scan = px.line(
                proton_scan,
                x="applied_potential_V",
                y="best_deltaG_leach_eV",
                color="dopant",
                line_group="site_key",
                markers=True,
                hover_data=[
                    col
                    for col in (
                        "surface_id",
                        "site_index",
                        "initial_dopant_zone",
                        "best_h_count",
                        "best_arrangement_id",
                        "bare_deltaG_leach_eV",
                        "deltaG_change_vs_bare_eV",
                    )
                    if col in proton_scan.columns
                ],
                title=f"Best post-leaching state: ΔG_leach vs potential ({selected_scale})",
                labels={
                    "applied_potential_V": f"Applied potential (V vs {selected_scale})",
                    "best_deltaG_leach_eV": "Best ΔG_leach (eV)",
                    "dopant": "Dopant",
                },
            )
            fig_scan.add_hline(
                y=0.0,
                line_dash="dash",
                annotation_text="ΔG_leach = 0",
                annotation_position="top left",
            )
            st.plotly_chart(fig_scan, use_container_width=True)

        if (
            bare_threshold_col in results.columns
            and adjusted_threshold_col in results.columns
        ):
            threshold_compare = results[
                [
                    col
                    for col in (
                        "dopant",
                        "site_index",
                        "initial_dopant_zone",
                        bare_threshold_col,
                        adjusted_threshold_col,
                    )
                    if col in results.columns
                ]
            ].copy()
            threshold_compare[bare_threshold_col] = pd.to_numeric(
                threshold_compare[bare_threshold_col], errors="coerce"
            )
            threshold_compare[adjusted_threshold_col] = pd.to_numeric(
                threshold_compare[adjusted_threshold_col], errors="coerce"
            )
            threshold_compare = threshold_compare.dropna(
                subset=[bare_threshold_col, adjusted_threshold_col],
                how="all",
            )
            if not threshold_compare.empty:
                melted = threshold_compare.melt(
                    id_vars=[
                        col
                        for col in ("dopant", "site_index", "initial_dopant_zone")
                        if col in threshold_compare.columns
                    ],
                    value_vars=[bare_threshold_col, adjusted_threshold_col],
                    var_name="surface_state_model",
                    value_name="dissolution_potential_V",
                ).dropna(subset=["dissolution_potential_V"])
                melted["surface_state_model"] = melted[
                    "surface_state_model"
                ].map(
                    {
                        bare_threshold_col: "Bare vacancy",
                        adjusted_threshold_col: "Best protonated state",
                    }
                )
                fig_threshold_compare = px.scatter(
                    melted,
                    x="dopant",
                    y="dissolution_potential_V",
                    color="surface_state_model",
                    symbol=(
                        "initial_dopant_zone"
                        if "initial_dopant_zone" in melted.columns
                        else None
                    ),
                    hover_data=[
                        col
                        for col in ("site_index", "initial_dopant_zone")
                        if col in melted.columns
                    ],
                    title=f"Bare vs protonation-adjusted dissolution threshold ({selected_scale})",
                    labels={
                        "dissolution_potential_V": f"Dissolution potential (V vs {selected_scale})",
                        "surface_state_model": "Post-leaching state",
                        "dopant": "Dopant",
                    },
                )
                st.plotly_chart(fig_threshold_compare, use_container_width=True)

    with tabs[4]:
        st.markdown(
            "Every row keeps the dopant's **initial relaxed-surface zone and coordinates** before removal. "
            "**Lower extraction energy** means weaker retention relative to the elemental-metal "
            "reference. With a valid aqueous redox reference, **lower dissolution potential** "
            "means dissolution becomes thermodynamically favorable at a lower electrode potential."
        )
        st.warning(
            "The electrochemical extension is a simple M^z+/M thermodynamic model. It does not "
            "include explicit solvent, charged slabs, potential-dependent hydroxylation, aqueous "
            "complex formation, kinetic barriers, or multi-atom dissolution pathways."
        )

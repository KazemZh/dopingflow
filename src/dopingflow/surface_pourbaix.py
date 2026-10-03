"""Surface electrochemistry stage: state sampling + CHE surface Pourbaix diagrams."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from pymatgen.core import Structure

from dopingflow.surface import _select_fixed_atom_indices
from dopingflow.surface_pourbaix_config import (
    SurfacePourbaixTarget,
    discover_surface_pourbaix_targets,
    parse_surface_pourbaix_config,
    preview_surface_pourbaix_targets,
    resolve_surface_pourbaix_output_dir,
    resolve_surface_pourbaix_source_summary,
)
from dopingflow.surface_pourbaix_states import (
    enumerate_surface_states,
    ml_reference_energy,
    screen_state,
)
from dopingflow.surface_pourbaix_postprocess import (
    analyze_relaxed_surface_state_path,
)
from dopingflow.surface_pourbaix_thermo import (
    build_surface_pourbaix_grid,
    summarize_stable_domains,
    surface_state_delta_g_eV,
    value_grid,
)
from dopingflow.surface_staged import _prepare_calculator, parse_surface_config

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coverage_text(value: Any) -> str:
    number = _finite_float(value)
    if number is None:
        return ""
    if abs(number - round(number)) <= 1e-8:
        return f"{int(round(number))}%"
    return f"{number:.1f}%"


def surface_state_display_label(state: Mapping[str, Any]) -> str:
    """Human-readable thermodynamic phase label for plots/tables.

    Arrangement IDs remain available separately for provenance. When
    post-relaxation validation is available, the relaxed final chemistry takes
    precedence over the originally generated state label.
    """
    final_label = str(state.get("final_state_label") or "").strip()
    if final_label:
        return final_label
    family = str(
        state.get("final_family", state.get("family", state.get("stable_family", "")))
    ).strip()
    if family == "clean":
        return "Clean"

    if family == "mixed-O-OH":
        o_cov = _coverage_text(state.get("actual_o_coverage_pct"))
        oh_cov = _coverage_text(state.get("actual_oh_coverage_pct"))
        if o_cov and oh_cov:
            return f"Mixed O*/OH* — {o_cov} O* + {oh_cov} OH*"
        return "Mixed O*/OH*"

    family_names = {
        "protonated": "Protonated lattice O",
        "O": "O*",
        "OH": "OH*",
        "H2O": "H₂O*",
    }
    name = family_names.get(family, family or "Surface state")
    coverage = _coverage_text(state.get("actual_coverage_pct"))
    return f"{name} — {coverage}" if coverage else name


def _annotate_grid_with_state_metadata(
    grid: pd.DataFrame,
    states: list[dict[str, Any]],
) -> pd.DataFrame:
    """Attach coverage/side metadata for the stable state at each U-pH point."""
    if grid.empty:
        return grid
    by_id = {str(row["state_id"]): row for row in states}
    output = grid.copy()
    metadata_keys = (
        "requested_coverage_pct",
        "actual_coverage_pct",
        "requested_o_coverage_pct",
        "requested_oh_coverage_pct",
        "actual_o_coverage_pct",
        "actual_oh_coverage_pct",
        "resolved_placement_side",
        "requested_placement_side",
        "side_target_species",
        "eligible_surface_oxygen_sites",
        "eligible_surface_cation_sites",
        "arrangement_id",
        "symmetry_unique_arrangements",
        "symmetry_operations",
        "state_status",
        "pourbaix_eligible",
        "final_family",
        "final_state_label",
        "postprocess_reason",
        "n_protonated_lattice_O",
        "n_surface_O",
        "n_surface_OH",
        "n_surface_H2O",
        "n_surface_O2",
        "n_surface_OOH_like",
        "n_desorbed_O",
        "n_desorbed_OH",
        "n_desorbed_H2O",
        "n_desorbed_O2",
        "n_unbound_H",
        "n_H2_like",
        "final_protonated_coverage_pct",
        "final_o_coverage_pct",
        "final_oh_coverage_pct",
        "final_h2o_coverage_pct",
        "final_o2_oxygen_coverage_pct",
        "final_total_bound_added_O_coverage_pct",
    )
    for key in metadata_keys:
        output[key] = [
            _scalar(by_id.get(str(state_id), {}).get(key))
            for state_id in output["stable_state_id"]
        ]
    output["stable_state_label"] = [
        surface_state_display_label(by_id.get(str(state_id), {}))
        for state_id in output["stable_state_id"]
    ]
    return output


def _record_from_state(
    target: SurfacePourbaixTarget,
    state: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(
        surface_id=target.surface_id,
        target_id=target.target_id,
        parent_id=target.parent_id,
        source_stage=target.source_stage,
        source_structure_path=str(target.structure_path),
        state_id=state["state_id"],
        family=state["family"],
        delta_n_H=int(state["delta_n_H"]),
        delta_n_O=int(state["delta_n_O"]),
        proton_electron_pairs=int(state["proton_electron_pairs"]),
        arrangement_id=int(state["arrangement_id"]),
        site_indices_json=json.dumps(state["site_indices"]),
        site_sides_json=json.dumps(state["site_sides"]),
        ml_status=result.get("status"),
        ml_energy_eV=result.get("energy_eV"),
        ml_converged=result.get("converged"),
        ml_final_fmax_eV_per_A=result.get("final_fmax_eV_per_A"),
        ml_optimizer_steps=result.get("optimizer_steps"),
        ml_checkpoint_reused=result.get("checkpoint_reused"),
        ml_initial_structure_path=result.get("initial_structure_path", ""),
        ml_structure_path=result.get("structure_path", ""),
        dft_energy_eV=None,
        dft_reused=None,
        dft_artifact="",
        requested_state_label=surface_state_display_label(state),
    )
    # Sampling provenance is kept explicitly so coverage rounding, automatic side
    # selection, and symmetry reduction are visible in the final CSV rather than
    # being hidden implementation details.
    for key in (
        "requested_coverage_pct",
        "actual_coverage_pct",
        "requested_o_coverage_pct",
        "requested_oh_coverage_pct",
        "actual_o_coverage_pct",
        "actual_oh_coverage_pct",
        "requested_placement_side",
        "resolved_placement_side",
        "side_target_species",
        "top_dopant_depth_A",
        "bottom_dopant_depth_A",
        "side_selection_reason",
        "eligible_surface_oxygen_sites",
        "eligible_surface_cation_sites",
        "raw_arrangements_total",
        "raw_arrangements_examined",
        "symmetry_unique_arrangements",
        "symmetry_operations",
        "o_site_indices",
        "oh_site_indices",
    ):
        if key in state:
            record[key] = _scalar(state[key])
    for key in (
        "miller_h", "miller_k", "miller_l", "termination_id", "termination_label",
        "host_species", "dopant_species_json", "structure_kind", "n_oxygen_vacancies",
        "composition_tag", "candidate",
    ):
        if key in target.metadata:
            record[key] = _scalar(target.metadata[key])
    return record


def _select_dft_candidates(records: list[dict[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    window = float(cfg["dft"]["candidate_window_eV"])

    def gap(row: Mapping[str, Any]) -> float:
        try:
            value = float(row.get("minimum_deltaG_above_stable_ml_eV", math.inf))
            return value if math.isfinite(value) else math.inf
        except (TypeError, ValueError):
            return math.inf

    candidates = [
        row
        for row in records
        if bool(row.get("pourbaix_eligible", True))
        and (row["family"] == "clean" or gap(row) <= window)
    ]
    candidates.sort(key=lambda row: (
        0 if row["family"] == "clean" else 1,
        gap(row),
        str(row["state_id"]),
    ))
    return candidates[: int(cfg["dft"]["max_states_per_surface"])]


def _leaching_rows(
    target: SurfacePourbaixTarget,
    domains: pd.DataFrame,
    records: list[dict[str, Any]],
    energy_level: str,
) -> list[dict[str, Any]]:
    by_id = {row["state_id"]: row for row in records}
    output = []
    for _, domain in domains.iterrows():
        row = by_id[str(domain["state_id"])]
        metadata = {
            key: _scalar(value)
            for key, value in target.metadata.items()
            if key != "surface_id"
        }
        output.append({
            **metadata,
            "surface_id": f"{target.surface_id}/electrochem_{row['state_id']}",
            "parent_surface_id": target.surface_id,
            "target_id": target.target_id,
            "parent_id": target.parent_id,
            "pourbaix_state_id": row["state_id"],
            "pourbaix_family": row.get("final_family", row["family"]),
            "pourbaix_state_label": row.get(
                "final_state_label", row.get("requested_state_label", "")
            ),
            "pourbaix_state_status": row.get("state_status", ""),
            "pourbaix_energy_level": energy_level,
            "pourbaix_delta_n_H": row["delta_n_H"],
            "pourbaix_delta_n_O": row["delta_n_O"],
            "pourbaix_grid_fraction": float(domain["grid_fraction"]),
            "pourbaix_pH_min_stable": float(domain["pH_min_stable"]),
            "pourbaix_pH_max_stable": float(domain["pH_max_stable"]),
            "pourbaix_potential_min_V_stable": float(domain["potential_min_V_stable"]),
            "pourbaix_potential_max_V_stable": float(domain["potential_max_V_stable"]),
            "pourbaix_structure_path": row["ml_structure_path"],
        })
    return output


def run_surface_pourbaix(
    raw: Mapping[str, Any], root: Path | str = Path("."), *, dry_run: bool = False
) -> Path | None:
    cfg = parse_surface_pourbaix_config(raw, root)
    if not bool(cfg["enabled"]):
        return None
    targets = discover_surface_pourbaix_targets(cfg)
    if not targets:
        raise RuntimeError("Surface-Pourbaix stage selected no usable surfaces")

    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    preview_rows = []
    for target in targets:
        planned_states = enumerate_surface_states(
            Structure.from_file(target.structure_path), cfg
        )
        clean = planned_states[0] if planned_states else {}
        preview_rows.append(dict(
            surface_id=target.surface_id,
            target_id=target.target_id,
            source_stage=target.source_stage,
            structure_path=str(target.structure_path),
            resolved_placement_side=clean.get("resolved_placement_side", ""),
            side_target_species=_scalar(clean.get("side_target_species", [])),
            top_dopant_depth_A=clean.get("top_dopant_depth_A"),
            bottom_dopant_depth_A=clean.get("bottom_dopant_depth_A"),
            eligible_surface_oxygen_sites=clean.get("eligible_surface_oxygen_sites", 0),
            eligible_surface_cation_sites=clean.get("eligible_surface_cation_sites", 0),
            n_states=len(planned_states),
        ))
    preview_path = outdir / "surface_pourbaix_preview.csv"
    pd.DataFrame(preview_rows).to_csv(preview_path, index=False)
    if dry_run:
        return preview_path

    calculator = _prepare_calculator(cfg["screen"], "Surface Pourbaix screening")
    h2_ml, h2_source = ml_reference_energy("H2", cfg, calculator)
    h2o_ml, h2o_source = ml_reference_energy("H2O", cfg, calculator)
    if h2_ml is None or h2o_ml is None:
        raise RuntimeError("Surface-Pourbaix CHE requires H2 and H2O reference energies")

    potentials = value_grid(cfg["potential_min_V"], cfg["potential_max_V"], cfg["potential_step_V"])
    ph_values = value_grid(cfg["pH_min"], cfg["pH_max"], cfg["pH_step"])
    surface_cfg = parse_surface_config(raw)
    all_states, all_grids, all_domains, leaching = [], [], [], []

    for target in targets:
        structure = Structure.from_file(target.structure_path)
        fixed = (
            _select_fixed_atom_indices(structure, dict(surface_cfg))
            if bool(cfg["inherit_surface_fixed_atoms"]) else []
        )
        target_dir = outdir / "surfaces" / target.safe_id
        records = []
        for state in enumerate_surface_states(structure, cfg):
            result = screen_state(
                state, cfg, calculator, fixed, target_dir / "states_ml"
            )
            record = _record_from_state(target, state, result)
            if result.get("status") == "ok":
                validation = analyze_relaxed_surface_state_path(
                    structure,
                    state["structure"],
                    result.get("structure_path", ""),
                    state,
                    cfg,
                )
            else:
                validation = {
                    "state_status": "calculation_failed",
                    "pourbaix_eligible": False,
                    "final_family": "invalid",
                    "final_state_label": "Failed relaxation",
                    "postprocess_reason": str(result.get("error", "ML relaxation failed")),
                }
            record.update(validation)
            records.append(record)

        ml_grid, gaps = build_surface_pourbaix_grid(
            records, h2_energy_eV=h2_ml, h2o_energy_eV=h2o_ml,
            potential_values=potentials, pH_values=ph_values,
            potential_scale=cfg["potential_scale"], temperature_K=cfg["temperature_K"],
            energy_key="ml_energy_eV", energy_level="ML",
        )
        ml_grid = _annotate_grid_with_state_metadata(ml_grid, records)
        ml_grid.insert(0, "surface_id", target.surface_id)
        gap_map = dict(zip(gaps["state_id"], gaps["minimum_deltaG_above_stable_eV"]))
        for record in records:
            record["minimum_deltaG_above_stable_ml_eV"] = gap_map.get(record["state_id"])

        chosen_grid, energy_level = ml_grid, "ML"
        if bool(cfg["dft"].get("enabled", False)):
            from dopingflow.surface_pourbaix_dft import dft_energy, dft_reference_energies

            candidates = _select_dft_candidates(records, cfg)
            for record in candidates:
                energy, reused, artifact = dft_energy(
                    Path(record["ml_structure_path"]),
                    f"{target.safe_id}/{record['state_id']}", cfg,
                )
                record["dft_energy_eV"] = energy
                record["dft_reused"] = reused
                record["dft_artifact"] = artifact
            h2_dft, h2o_dft = dft_reference_energies(cfg)
            if h2_dft is not None and h2o_dft is not None:
                try:
                    dft_grid, _ = build_surface_pourbaix_grid(
                        candidates, h2_energy_eV=h2_dft, h2o_energy_eV=h2o_dft,
                        potential_values=potentials, pH_values=ph_values,
                        potential_scale=cfg["potential_scale"], temperature_K=cfg["temperature_K"],
                        energy_key="dft_energy_eV", energy_level="DFT",
                    )
                    dft_grid = _annotate_grid_with_state_metadata(
                        dft_grid, candidates
                    )
                    dft_grid.insert(0, "surface_id", target.surface_id)
                    chosen_grid, energy_level = dft_grid, "DFT"
                    dft_grid.to_csv(target_dir / "pourbaix_grid_dft.csv", index=False)
                except ValueError:
                    pass

        domains = summarize_stable_domains(chosen_grid)
        if not domains.empty:
            domains.insert(0, "surface_id", target.surface_id)
            domains.insert(1, "target_id", target.target_id)
        leaching.extend(_leaching_rows(target, domains, records, energy_level))

        target_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(target_dir / "surface_state_summary.csv", index=False)
        ml_grid.to_csv(target_dir / "pourbaix_grid_ml.csv", index=False)
        chosen_grid.to_csv(target_dir / "pourbaix_grid.csv", index=False)
        domains.to_csv(target_dir / "stable_surface_states.csv", index=False)
        all_states.extend(records)
        all_grids.append(chosen_grid)
        all_domains.append(domains)

    states_path = outdir / "surface_state_summary.csv"
    grid_path = outdir / "pourbaix_grid.csv"
    domains_path = outdir / "stable_surface_states.csv"
    leaching_path = outdir / "leaching_surface_states.csv"
    pd.DataFrame(all_states).to_csv(states_path, index=False)
    pd.concat(all_grids, ignore_index=True).to_csv(grid_path, index=False)
    pd.concat(all_domains, ignore_index=True).to_csv(domains_path, index=False)
    pd.DataFrame(leaching).to_csv(leaching_path, index=False)

    summary = dict(
        source_summary=str(cfg["resolved_source_summary"]),
        source_mode=cfg["resolved_source_mode"],
        pH_range=[cfg["pH_min"], cfg["pH_max"], cfg["pH_step"]],
        potential_range_V=[cfg["potential_min_V"], cfg["potential_max_V"], cfg["potential_step_V"]],
        potential_scale=cfg["potential_scale"], temperature_K=cfg["temperature_K"],
        h2_reference_eV_ml=h2_ml, h2_reference_source=h2_source,
        h2o_reference_eV_ml=h2o_ml, h2o_reference_source=h2o_source,
        states_csv=str(states_path), pourbaix_grid_csv=str(grid_path),
        stable_states_csv=str(domains_path), leaching_surface_states_csv=str(leaching_path),
        postprocess_validation_enabled=bool(cfg["postprocess_validate_relaxed_states"]),
        postprocess_exclude_desorbed=bool(cfg["postprocess_exclude_desorbed"]),
        postprocess_exclude_fragmented=bool(cfg["postprocess_exclude_fragmented"]),
        postprocess_allow_reclassified=bool(cfg["postprocess_allow_reclassified"]),
        postprocess_oh_bond_cutoff_A=float(cfg["postprocess_oh_bond_cutoff_A"]),
        postprocess_oo_bond_cutoff_A=float(cfg["postprocess_oo_bond_cutoff_A"]),
        postprocess_surface_attachment_cutoff_A=float(
            cfg["postprocess_surface_attachment_cutoff_A"]
        ),
        thermodynamic_model=(
            "Computational Hydrogen Electrode using electronic/reference energies; "
            "no implicit ZPE, vibrational entropy, configurational entropy, or solvation correction"
        ),
    )
    (outdir / "surface_pourbaix_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return grid_path


def run_surface_pourbaix_from_toml(config_path: Path, *, dry_run: bool = False) -> Path | None:
    config_path = Path(config_path).expanduser().resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return run_surface_pourbaix(raw, config_path.parent, dry_run=dry_run)


__all__ = [
    "SurfacePourbaixTarget", "build_surface_pourbaix_grid",
    "discover_surface_pourbaix_targets", "enumerate_surface_states",
    "parse_surface_pourbaix_config", "preview_surface_pourbaix_targets",
    "resolve_surface_pourbaix_output_dir", "resolve_surface_pourbaix_source_summary",
    "run_surface_pourbaix", "run_surface_pourbaix_from_toml",
    "summarize_stable_domains", "surface_state_delta_g_eV",
    "surface_state_display_label",
]

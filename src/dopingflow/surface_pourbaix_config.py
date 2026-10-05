from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from dopingflow.ml_backends import normalize_backend_config
from dopingflow.surface_segregation import resolve_surface_segregation_output_dir
from dopingflow.surface_staged import ensure_surface_ids, parse_surface_config, resolve_surface_output_dir

DEFAULT_FAMILIES = ("clean", "protonated", "O", "OH", "H2O", "mixed-O-OH")


@dataclass(frozen=True)
class SurfacePourbaixTarget:
    surface_id: str
    target_id: str
    parent_id: str
    structure_path: Path
    source_stage: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def safe_id(self) -> str:
        text = self.surface_id.replace("\\", "__").replace("/", "__")
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    raise ValueError("Expected an array or comma-separated string")


def _int_list(value: Any, default: Sequence[int]) -> list[int]:
    raw = default if value is None else value
    if isinstance(raw, str):
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        values = [int(x) for x in raw]
    return sorted(set(values))


def _float_list(value: Any, default: Sequence[float]) -> list[float]:
    raw = default if value is None else value
    if isinstance(raw, str):
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        values = [float(x) for x in raw]
    return list(dict.fromkeys(values))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def source_root(raw: Mapping[str, Any], root: Path) -> Path:
    pbx = dict(raw.get("surface_pourbaix", {}) or {})
    segregation = dict(raw.get("surface_segregation", {}) or {})
    surface = dict(raw.get("surface", {}) or {})
    structure = dict(raw.get("structure", {}) or {})
    value = (
        str(pbx.get("source_root", "")).strip()
        or str(surface.get("source_root", "")).strip()
        or str(segregation.get("source_root", "")).strip()
        or str(structure.get("outdir", "random_structures")).strip()
    )
    return _resolve(root, value)


def resolve_surface_pourbaix_output_dir(raw: Mapping[str, Any], root: Path | str) -> Path:
    root = Path(root).expanduser().resolve()
    section = dict(raw.get("surface_pourbaix", {}) or {})
    return _resolve(source_root(raw, root), str(section.get("outdir", "10_surface_pourbaix")))


def _surface_candidates(raw: Mapping[str, Any], root: Path) -> dict[str, Path]:
    surface_cfg = parse_surface_config(raw)
    output = resolve_surface_output_dir(raw, surface_cfg, root)
    return {
        "final-selected": output / str(surface_cfg["refine_selected_csv"]),
        "refine-summary": output / str(surface_cfg["refine_summary_csv"]),
        "screen-selected": output / str(surface_cfg["screen_selected_csv"]),
        "screen-summary": output / str(surface_cfg["screen_summary_csv"]),
    }


def _best_direct_surface_summary(
    surfaces: Mapping[str, Path],
) -> tuple[Path, str]:
    """Return the best available table produced directly by the surface stage."""
    for name in ("final-selected", "refine-summary", "screen-selected", "screen-summary"):
        path = Path(surfaces[name])
        if path.exists():
            return path, name
    # Return the preferred path so the eventual FileNotFoundError is informative.
    return Path(surfaces["final-selected"]), "final-selected"


def resolve_surface_pourbaix_source_summary(
    raw: Mapping[str, Any], root: Path | str
) -> tuple[Path, str]:
    root = Path(root).expanduser().resolve()
    section = dict(raw.get("surface_pourbaix", {}) or {})
    explicit = str(section.get("source_summary", "")).strip()
    if explicit:
        return _resolve(source_root(raw, root), explicit), "explicit"

    mode = str(section.get("source_mode", "surface")).strip().lower()
    segregation = (
        resolve_surface_segregation_output_dir(raw, root)
        / "surface_segregation_summary.csv"
    )
    surfaces = _surface_candidates(raw, root)

    # Surface Pourbaix must not depend on running segregation. The direct
    # surface-stage tables are a first-class source and are the default.
    if mode == "surface":
        return _best_direct_surface_summary(surfaces)
    if mode == "segregation":
        return segregation, "segregation"
    if mode == "auto":
        direct_path, direct_mode = _best_direct_surface_summary(surfaces)
        if direct_path.exists():
            return direct_path, direct_mode
        if segregation.exists():
            return segregation, "segregation"
        return direct_path, direct_mode
    if mode not in surfaces:
        raise ValueError(
            "[surface_pourbaix].source_mode must be surface, auto, segregation, "
            "final-selected, refine-summary, screen-selected, or screen-summary"
        )
    return surfaces[mode], mode


def _calculator_defaults(raw: Mapping[str, Any]) -> dict[str, Any]:
    surface = dict(raw.get("surface", {}) or {})
    refine = dict(surface.get("refine", {}) or {})
    screen = dict(surface.get("screen", {}) or {})
    source = refine if bool(refine.get("enabled", False)) else screen
    defaults = dict(
        backend="mace", model="mh-1", task="matpes_r2scan", device="cpu",
        gpu_id=0, tf_threads=1, omp_threads=1, optimizer="bfgs",
        fmax=0.05, max_steps=300, relax=True,
    )
    defaults.update(source)
    return defaults


def parse_surface_pourbaix_config(
    raw: Mapping[str, Any], root: Path | str = Path(".")
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    section = dict(raw.get("surface_pourbaix", {}) or {})
    surface = dict(raw.get("surface", {}) or {})
    scan = dict(raw.get("scan", {}) or {})
    defaults = dict(
        enabled=False, source_root=str(source_root(raw, root)), source_mode="surface",
        source_summary="", surface_include=[], max_surfaces=10,
        outdir="10_surface_pourbaix", placement_side="dopant-nearest",
        protonation_side="both",
        host_species=surface.get("host_species", ""),
        side_target_species=surface.get("dopant_species", []),
        dopant_side_tie_tolerance_A=0.25, dopant_side_fallback="top",
        anion_species=surface.get("anion_species", scan.get("anion_species", ["O"])),
        state_families=list(DEFAULT_FAMILIES),
        proton_coverages_pct=[25.0, 50.0, 75.0, 100.0],
        o_coverages_pct=[25.0, 50.0, 75.0, 100.0],
        oh_coverages_pct=[25.0, 50.0, 75.0, 100.0],
        h2o_coverages_pct=[25.0, 50.0, 100.0],
        mixed_coverages_pct=[[25.0, 25.0], [25.0, 75.0], [50.0, 50.0], [75.0, 25.0]],
        surface_window_A=2.0, max_surface_oxygen_sites=0, max_surface_cation_sites=0,
        symmetry_reduce=True, symmetry_symprec_A=0.10,
        symmetry_angle_tolerance_deg=5.0, symmetry_mapping_tolerance_A=0.25,
        max_raw_configurations_per_stoichiometry=100000,
        max_arrangements_per_stoichiometry=8, oh_bond_length_A=0.98,
        adsorbate_height_A=1.85, water_oh_bond_length_A=0.9572,
        water_hoh_angle_deg=104.5, temperature_K=298.15,
        potential_scale="SHE", potential_min_V=0.0, potential_max_V=2.0,
        potential_step_V=0.05, pH_min=-1.0, pH_max=3.0, pH_step=0.1,
        manual_h2_energy_eV="", manual_h2o_energy_eV="", compute_references=True,
        reference_relax=True, reference_box_A=15.0,
        h2_free_energy_correction_eV=0.0, h2o_free_energy_correction_eV=0.0,
        postprocess_validate_relaxed_states=True,
        postprocess_oh_bond_cutoff_A=1.25,
        postprocess_oo_bond_cutoff_A=1.75,
        postprocess_surface_attachment_cutoff_A=2.80,
        postprocess_hh_bond_cutoff_A=0.90,
        postprocess_exclude_desorbed=True,
        postprocess_exclude_fragmented=True,
        postprocess_allow_reclassified=True,
        inherit_surface_fixed_atoms=True, resume_completed=True, screen={}, dft={},
    )
    for key, value in defaults.items():
        section.setdefault(key, value)

    section["root"] = root
    section["source_root"] = source_root(raw, root)
    section["output_dir"] = resolve_surface_pourbaix_output_dir(raw, root)
    summary, mode = resolve_surface_pourbaix_source_summary(raw, root)
    section["resolved_source_summary"] = summary
    section["resolved_source_mode"] = mode
    section["surface_include"] = _string_list(section["surface_include"])
    section["anion_species"] = _string_list(section["anion_species"]) or ["O"]
    section["side_target_species"] = _string_list(section["side_target_species"])
    section["host_species"] = str(section.get("host_species", "")).strip()
    section["state_families"] = _string_list(section["state_families"])
    invalid = [x for x in section["state_families"] if x not in DEFAULT_FAMILIES]
    if invalid:
        raise ValueError(f"Unsupported surface-state families: {invalid}")
    if "clean" not in section["state_families"]:
        section["state_families"].insert(0, "clean")

    coverage_defaults = {
        "proton_coverages_pct": [25.0, 50.0, 75.0, 100.0],
        "o_coverages_pct": [25.0, 50.0, 75.0, 100.0],
        "oh_coverages_pct": [25.0, 50.0, 75.0, 100.0],
        "h2o_coverages_pct": [25.0, 50.0, 100.0],
    }
    for key, default in coverage_defaults.items():
        values = _float_list(section.get(key), default)
        if not values or any(value <= 0.0 or value > 100.0 for value in values):
            raise ValueError(f"{key} values must lie in (0, 100]")
        section[key] = values

    mixed = []
    for item in section["mixed_coverages_pct"]:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("mixed_coverages_pct entries must be [O%, OH%]")
        o_pct, oh_pct = float(item[0]), float(item[1])
        if o_pct <= 0.0 or oh_pct <= 0.0 or o_pct + oh_pct > 100.0 + 1e-12:
            raise ValueError("Mixed O/OH coverages must be positive and sum to <= 100%")
        mixed.append((o_pct, oh_pct))
    section["mixed_coverages_pct"] = mixed

    section["placement_side"] = str(section["placement_side"]).lower().replace("_", "-")
    if section["placement_side"] not in {"top", "bottom", "both", "dopant-nearest"}:
        raise ValueError("placement_side must be top, bottom, both, or dopant-nearest")

    section["protonation_side"] = (
        str(section.get("protonation_side", "both")).lower().replace("_", "-")
    )
    protonation_aliases = {
        "same": "same-as-adsorbates",
        "adsorbate-side": "same-as-adsorbates",
        "same-as-adsorbate": "same-as-adsorbates",
    }
    section["protonation_side"] = protonation_aliases.get(
        section["protonation_side"], section["protonation_side"]
    )
    if section["protonation_side"] not in {
        "same-as-adsorbates", "top", "bottom", "both", "dopant-nearest"
    }:
        raise ValueError(
            "protonation_side must be same-as-adsorbates, top, bottom, both, "
            "or dopant-nearest"
        )
    section["dopant_side_fallback"] = str(section["dopant_side_fallback"]).lower()
    if section["dopant_side_fallback"] not in {"top", "bottom", "both"}:
        raise ValueError("dopant_side_fallback must be top, bottom, or both")
    section["potential_scale"] = str(section["potential_scale"]).upper()
    if section["potential_scale"] not in {"SHE", "RHE"}:
        raise ValueError("potential_scale must be SHE or RHE")

    section["max_surfaces"] = int(section["max_surfaces"])
    section["max_arrangements_per_stoichiometry"] = int(section["max_arrangements_per_stoichiometry"])
    section["max_raw_configurations_per_stoichiometry"] = int(section["max_raw_configurations_per_stoichiometry"])
    if section["max_surfaces"] <= 0 or section["max_arrangements_per_stoichiometry"] <= 0:
        raise ValueError("max_surfaces and max_arrangements_per_stoichiometry must be positive")
    if section["max_raw_configurations_per_stoichiometry"] <= 0:
        raise ValueError("max_raw_configurations_per_stoichiometry must be positive")
    # Legacy site caps changed the physical coverage denominator (for example,
    # max_surface_oxygen_sites=8 made "100%" mean 8 sites even when 10 were
    # exposed). Keep accepting old input files, but intentionally ignore these
    # caps so coverage always uses the full eligible surface site set.
    for key in ("max_surface_oxygen_sites", "max_surface_cation_sites"):
        legacy_value = int(section.get(key, 0) or 0)
        if legacy_value < 0:
            raise ValueError(f"{key} must be >= 0")
        section[key] = 0
    for key in (
        "surface_window_A", "oh_bond_length_A", "adsorbate_height_A",
        "water_oh_bond_length_A", "water_hoh_angle_deg", "temperature_K",
        "potential_min_V", "potential_max_V", "potential_step_V", "pH_min",
        "pH_max", "pH_step", "reference_box_A", "h2_free_energy_correction_eV",
        "h2o_free_energy_correction_eV", "dopant_side_tie_tolerance_A",
        "symmetry_symprec_A", "symmetry_angle_tolerance_deg",
        "symmetry_mapping_tolerance_A", "postprocess_oh_bond_cutoff_A",
        "postprocess_oo_bond_cutoff_A", "postprocess_surface_attachment_cutoff_A",
        "postprocess_hh_bond_cutoff_A",
    ):
        section[key] = float(section[key])
    if section["temperature_K"] <= 0 or section["surface_window_A"] <= 0:
        raise ValueError("temperature_K and surface_window_A must be positive")
    if section["dopant_side_tie_tolerance_A"] < 0:
        raise ValueError("dopant_side_tie_tolerance_A must be >= 0")
    if section["symmetry_symprec_A"] <= 0 or section["symmetry_mapping_tolerance_A"] <= 0:
        raise ValueError("symmetry tolerances must be positive")
    section["symmetry_reduce"] = bool(section["symmetry_reduce"])
    section["postprocess_validate_relaxed_states"] = bool(
        section["postprocess_validate_relaxed_states"]
    )
    section["postprocess_exclude_desorbed"] = bool(
        section["postprocess_exclude_desorbed"]
    )
    section["postprocess_exclude_fragmented"] = bool(
        section["postprocess_exclude_fragmented"]
    )
    section["postprocess_allow_reclassified"] = bool(
        section["postprocess_allow_reclassified"]
    )
    if (
        section["postprocess_oh_bond_cutoff_A"] <= 0
        or section["postprocess_oo_bond_cutoff_A"] <= 0
        or section["postprocess_surface_attachment_cutoff_A"] <= 0
        or section["postprocess_hh_bond_cutoff_A"] <= 0
    ):
        raise ValueError("post-relaxation validation distance cutoffs must be positive")
    if section["potential_step_V"] <= 0 or section["pH_step"] <= 0:
        raise ValueError("potential_step_V and pH_step must be positive")
    if section["potential_min_V"] >= section["potential_max_V"] or section["pH_min"] >= section["pH_max"]:
        raise ValueError("Potential and pH minima must be below maxima")
    for key in ("manual_h2_energy_eV", "manual_h2o_energy_eV"):
        value = section[key]
        section[key] = None if value is None or str(value).strip() == "" else float(value)

    screen_cfg = _calculator_defaults(raw)
    screen_cfg.update(dict(section.get("screen", {}) or {}))
    backend, model, task = normalize_backend_config(
        backend=str(screen_cfg.get("backend", "mace")).lower(),
        model=str(screen_cfg.get("model", "mh-1")), task=str(screen_cfg.get("task", "")),
        section_name="surface_pourbaix.screen",
    )
    screen_cfg.update(backend=backend, model=model, task=task)
    defaults_int = {"gpu_id": 0, "tf_threads": 1, "omp_threads": 1, "max_steps": 300}
    for key in defaults_int:
        screen_cfg[key] = int(screen_cfg.get(key, defaults_int[key]))
    screen_cfg["device"] = str(screen_cfg.get("device", "cpu")).lower()
    screen_cfg["optimizer"] = str(screen_cfg.get("optimizer", "bfgs")).lower()
    screen_cfg["fmax"] = float(screen_cfg.get("fmax", 0.05))
    screen_cfg["relax"] = bool(screen_cfg.get("relax", True))
    section["screen"] = screen_cfg

    dft = dict(section.get("dft", {}) or {})
    dft_defaults = dict(
        enabled=False, execute=False, max_states_per_surface=20, candidate_window_eV=0.30,
        code="gpaw", mode="pw", xc="PBE", ecut_eV=500.0, kpts=[3, 3, 1], gamma=True,
        smearing_eV=0.05, convergence_density=1e-5, maxiter=333, charge=0.0,
        spinpol="auto", initial_magmoms={}, nbands=None, save_wavefunctions=False,
        reuse_existing=True, cache_root="dft_cache", fail_fast=False,
    )
    for key, value in dft_defaults.items():
        dft.setdefault(key, value)
    dft["max_states_per_surface"] = int(dft["max_states_per_surface"])
    dft["candidate_window_eV"] = float(dft["candidate_window_eV"])
    section["dft"] = dft
    return section


def _present(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"nan", "none"}


def _target_path(row: Mapping[str, Any], root: Path) -> tuple[Path, str]:
    for key, stage in (
        ("best_structure_path", "segregation-best"),
        ("final_structure_path", "segregation-final"),
        ("refine_relaxed_structure_path", "refine"),
        ("screen_relaxed_structure_path", "screen"),
        ("generated_structure_path", "generated"),
    ):
        if _present(row.get(key)):
            path = _resolve(root, str(row[key]))
            if path.exists():
                return path, stage
    raise FileNotFoundError("No usable surface structure path")


def discover_surface_pourbaix_targets(cfg: Mapping[str, Any]) -> list[SurfacePourbaixTarget]:
    summary = Path(cfg["resolved_source_summary"])
    if not summary.exists():
        raise FileNotFoundError(f"Surface-Pourbaix source table not found: {summary}")
    frame = pd.read_csv(summary)
    if frame.empty:
        return []
    if "surface_id" not in frame.columns:
        frame = ensure_surface_ids(frame)
    targets, seen = [], set()
    for _, series in frame.iterrows():
        row = series.to_dict()
        values = (str(row.get("surface_id", "")), str(row.get("target_id", "")), str(row.get("termination_label", "")))
        if cfg["surface_include"] and not any(
            fnmatch.fnmatchcase(value, pattern) for pattern in cfg["surface_include"] for value in values
        ):
            continue
        sid = str(row.get("surface_id", "")).strip()
        if not sid or sid in seen:
            continue
        try:
            path, stage = _target_path(row, Path(cfg["root"]))
        except FileNotFoundError:
            continue
        targets.append(SurfacePourbaixTarget(
            surface_id=sid, target_id=str(row.get("target_id", "")),
            parent_id=str(row.get("parent_id") or row.get("target_id") or ""),
            structure_path=path, source_stage=stage, metadata=row,
        ))
        seen.add(sid)
        if len(targets) >= int(cfg["max_surfaces"]):
            break
    return targets


def preview_surface_pourbaix_targets(raw: Mapping[str, Any], root: Path | str = Path(".")) -> pd.DataFrame:
    cfg = parse_surface_pourbaix_config(raw, root)
    frame = pd.DataFrame([
        dict(surface_id=t.surface_id, target_id=t.target_id, source_stage=t.source_stage, structure_path=str(t.structure_path))
        for t in discover_surface_pourbaix_targets(cfg)
    ])
    frame.attrs["source_summary"] = str(cfg["resolved_source_summary"])
    return frame

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pymatgen.io.vasp import Poscar

from dopingflow.surface_pourbaix_states import molecule_structure


def _oxidation_cfg(cfg: Mapping[str, Any]):
    from dopingflow.oxidation import OxidationConfig
    return OxidationConfig(
        root=Path(cfg["root"]), source_root=Path(cfg["source_root"]),
        output_dir=Path(cfg["output_dir"]) / "dft", enabled=True, strategy="dft",
        methods=("dft-electronic",), include_vacancy_free=True,
        include_oxygen_vacancies=False, mapping_tolerance=0.6,
        fail_fast=bool(cfg["dft"].get("fail_fast", False)),
        dft_followup_enabled=False, dft_followup_candidate_limit=0,
        dft_followup_execute=False, dft_followup_methods=(), target_include=(), settings={},
    )


def dft_energy(
    path: Path, state_id: str, cfg: Mapping[str, Any], *, is_reference: bool = False
) -> tuple[float | None, bool, str]:
    from dopingflow.dft_cache import ensure_gpaw
    from dopingflow.oxidation import OptionalMethodUnavailable, StructureTarget
    from dopingflow.oxidation_dft import _gpaw_restart

    settings = dict(cfg["dft"])
    settings["execute"] = bool(settings.get("execute", False))
    settings["output_root"] = str(Path(cfg["output_dir"]) / "dft")
    if is_reference:
        settings["kpts"] = [1, 1, 1]
    target = StructureTarget(
        target_id=f"surface-pourbaix/{state_id}", parent_id=state_id,
        kind="surface-pourbaix", structure_path=Path(path), n_vacancies=0,
        vacancy_species=None, metadata={},
    )
    try:
        gpw, reused = ensure_gpaw(target, _oxidation_cfg(cfg), settings)
        _, calculator = _gpaw_restart(gpw)
        return float(calculator.get_potential_energy()), bool(reused), str(gpw)
    except (OptionalMethodUnavailable, RuntimeError, ValueError, OSError) as exc:
        if bool(settings.get("fail_fast", False)):
            raise
        return None, False, f"{type(exc).__name__}: {exc}"


def dft_reference_energies(cfg: Mapping[str, Any]) -> tuple[float | None, float | None]:
    root = Path(cfg["output_dir"]) / "references"
    paths = {}
    for kind in ("H2", "H2O"):
        path = root / kind / "POSCAR_for_dft"
        path.parent.mkdir(parents=True, exist_ok=True)
        Poscar(molecule_structure(kind, cfg["reference_box_A"])).write_file(str(path))
        paths[kind] = path
    h2, _, _ = dft_energy(paths["H2"], "reference/H2", cfg, is_reference=True)
    h2o, _, _ = dft_energy(paths["H2O"], "reference/H2O", cfg, is_reference=True)
    if h2 is not None:
        h2 += float(cfg["h2_free_energy_correction_eV"])
    if h2o is not None:
        h2o += float(cfg["h2o_free_energy_correction_eV"])
    return h2, h2o

from __future__ import annotations

import csv
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from dopingflow.oxidation import (
    OptionalMethodUnavailable,
    OxidationConfig,
    StructureTarget,
    base_method_result,
    site_records,
)


def _format_path(raw: str, *, target: StructureTarget) -> Path:
    text = raw.format(target_id=target.target_id, safe_id=target.safe_id, parent_id=target.parent_id)
    return Path(text).expanduser()


def _workdir(target: StructureTarget, cfg: OxidationConfig, settings: dict[str, Any]) -> Path:
    explicit = str(settings.get("workdir") or "").strip()
    if explicit:
        path = _format_path(explicit, target=target)
        return (path if path.is_absolute() else cfg.root / path).resolve()
    root_raw = str(settings.get("output_root") or "dft_oxidation").strip()
    root = Path(root_raw).expanduser()
    root = (root if root.is_absolute() else cfg.source_root / root).resolve()
    return root / target.safe_id


def _command_tokens(command: Any, *, target: StructureTarget, workdir: Path) -> list[str]:
    if isinstance(command, str):
        raw = shlex.split(command)
    elif isinstance(command, list):
        raw = [str(item) for item in command]
    else:
        raise ValueError("Post-processing command must be a string or array of strings")
    return [
        item.format(
            target_id=target.target_id,
            safe_id=target.safe_id,
            parent_id=target.parent_id,
            structure_path=str(target.structure_path),
            workdir=str(workdir),
        )
        for item in raw
    ]


def _maybe_execute(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
    *,
    stage: str,
) -> Path:
    workdir = _workdir(target, cfg, settings)
    execute = bool(settings.get("execute", False))
    if not execute:
        return workdir
    command = settings.get("command")
    if not command:
        raise OptionalMethodUnavailable(
            f"{stage} execute=true requires an explicit command. No expensive calculation is launched implicitly."
        )
    workdir.mkdir(parents=True, exist_ok=True)
    structure = Structure.from_file(target.structure_path)
    structure.to(filename=str(workdir / "structure.cif"))
    tokens = _command_tokens(command, target=target, workdir=workdir)
    completed = subprocess.run(
        tokens,
        cwd=workdir,
        check=False,
        text=True,
        capture_output=True,
    )
    (workdir / f"dopingflow_{stage}_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (workdir / f"dopingflow_{stage}_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage} command failed with exit code {completed.returncode}; see dopingflow_{stage}_stderr.txt"
        )
    return workdir


def _trapz(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    total = 0.0
    for left in range(len(x) - 1):
        dx = float(x[left + 1]) - float(x[left])
        total += 0.5 * dx * (float(y[left + 1]) + float(y[left]))
    return total


def _parse_kpts(settings: dict[str, Any]) -> tuple[int, int, int]:
    raw = settings.get("kpts", [1, 1, 1])
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.replace("x", ",").split(",") if item.strip()]
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("[oxidation.dft_electronic].kpts must contain three positive integers")
    kpts = tuple(int(value) for value in raw)
    if any(value < 1 for value in kpts):
        raise ValueError("[oxidation.dft_electronic].kpts values must be >= 1")
    return kpts


def _parse_initial_magmoms(settings: dict[str, Any]) -> dict[str, float]:
    raw = settings.get("initial_magmoms", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("[oxidation.dft_electronic].initial_magmoms must be a table/dictionary")
    return {str(element): float(value) for element, value in raw.items()}


def _gpaw_restart(path: Path):
    try:
        from gpaw import restart
    except ImportError as exc:
        raise OptionalMethodUnavailable(
            "GPAW electronic analysis requires GPAW and its PAW datasets. Install both with "
            "`conda install -c conda-forge gpaw gpaw-data`."
        ) from exc
    try:
        return restart(str(path), txt=None)
    except TypeError:
        return restart(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not read GPAW restart file {path}: {exc}") from exc


def _run_gpaw_single_point(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> Path:
    try:
        import gpaw
        from gpaw import FermiDirac, GPAW, PW
    except ImportError as exc:
        raise OptionalMethodUnavailable(
            "GPAW execution requires GPAW and its PAW datasets. Install both with "
            "`conda install -c conda-forge gpaw gpaw-data`."
        ) from exc

    mode = str(settings.get("mode", "pw")).strip().lower()
    if mode != "pw":
        raise ValueError(
            "The oxidation GPAW backend currently supports mode='pw' only for periodic doped solids"
        )

    workdir = _workdir(target, cfg, settings)
    workdir.mkdir(parents=True, exist_ok=True)
    structure = Structure.from_file(target.structure_path)
    atoms = AseAtomsAdaptor.get_atoms(structure)

    initial_map = _parse_initial_magmoms(settings)
    if initial_map:
        atoms.set_initial_magnetic_moments(
            [float(initial_map.get(symbol, 0.0)) for symbol in atoms.get_chemical_symbols()]
        )

    spin_mode = settings.get("spinpol", "auto")
    if isinstance(spin_mode, bool):
        spin_mode = "true" if spin_mode else "false"
    spin_mode = str(spin_mode).strip().lower()
    if spin_mode not in {"auto", "true", "false"}:
        raise ValueError("[oxidation.dft_electronic].spinpol must be auto, true, or false")

    ecut = float(settings.get("ecut_eV", 500.0))
    if ecut <= 0:
        raise ValueError("[oxidation.dft_electronic].ecut_eV must be positive")
    smearing = float(settings.get("smearing_eV", 0.05))
    if smearing < 0:
        raise ValueError("[oxidation.dft_electronic].smearing_eV must be >= 0")
    convergence_density = float(settings.get("convergence_density", 1.0e-5))
    if convergence_density <= 0:
        raise ValueError("[oxidation.dft_electronic].convergence_density must be positive")

    gpw_path = workdir / str(settings.get("gpw_file", "oxidation.gpw"))
    txt_path = workdir / str(settings.get("txt_file", "gpaw.txt"))
    kwargs: dict[str, Any] = {
        "mode": PW(ecut),
        "xc": str(settings.get("xc", "PBE")),
        "kpts": {"size": _parse_kpts(settings), "gamma": bool(settings.get("gamma", True))},
        "occupations": FermiDirac(smearing),
        "convergence": {"density": convergence_density},
        "txt": str(txt_path),
    }
    maxiter = int(settings.get("maxiter", 333))
    if maxiter > 0:
        kwargs["maxiter"] = maxiter
    charge = float(settings.get("charge", 0.0))
    if abs(charge) > 1.0e-12:
        kwargs["charge"] = charge
    if spin_mode == "true":
        kwargs["spinpol"] = True
    elif spin_mode == "false":
        kwargs["spinpol"] = False

    try:
        calc = GPAW(**kwargs)
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        if bool(settings.get("save_wavefunctions", False)):
            calc.write(str(gpw_path), mode="all")
        else:
            calc.write(str(gpw_path))
    except Exception as exc:
        raise RuntimeError(f"GPAW single-point calculation failed: {exc}") from exc

    metadata = {
        "code": "GPAW",
        "gpaw_version": getattr(gpaw, "__version__", None),
        "target_id": target.target_id,
        "structure_path": str(target.structure_path),
        "gpw_file": str(gpw_path),
        "txt_file": str(txt_path),
        "final_energy_eV": energy,
        "settings": {
            "mode": "pw",
            "ecut_eV": ecut,
            "xc": str(settings.get("xc", "PBE")),
            "kpts": list(_parse_kpts(settings)),
            "gamma": bool(settings.get("gamma", True)),
            "smearing_eV": smearing,
            "convergence_density": convergence_density,
            "maxiter": maxiter,
            "charge": charge,
            "spinpol": spin_mode,
            "initial_magmoms": initial_map,
            "save_wavefunctions": bool(settings.get("save_wavefunctions", False)),
        },
    }
    (workdir / "gpaw_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return gpw_path


def _run_dft_electronic(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    code = str(settings.get("code", "gpaw")).strip().lower()
    if code != "gpaw":
        raise OptionalMethodUnavailable(
            f"The oxidation dft-electronic backend is GPAW-only; requested code='{code}'."
        )

    workdir = _workdir(target, cfg, settings)
    gpw_path = workdir / str(settings.get("gpw_file", "oxidation.gpw"))
    if bool(settings.get("execute", False)):
        gpw_path = _run_gpaw_single_point(target, cfg, settings)
    if not gpw_path.is_file():
        raise OptionalMethodUnavailable(
            f"GPAW electronic analysis needs {gpw_path}. Set execute=true to run the GPAW "
            "single point directly, or point workdir/output_root to an existing GPAW result."
        )

    atoms, calc = _gpaw_restart(gpw_path)
    original = Structure.from_file(target.structure_path)
    symbols = list(atoms.get_chemical_symbols())
    expected = [site.specie.symbol for site in original]
    if len(symbols) != len(expected) or symbols != expected:
        raise RuntimeError("GPAW output site order/species do not match the analyzed relaxed structure")

    try:
        final_energy = float(atoms.get_potential_energy())
    except Exception:
        final_energy = None
    try:
        efermi = float(calc.get_fermi_level())
    except Exception:
        efermi = None

    magnetic_records: list[dict[str, Any]] = []
    try:
        try:
            moments = calc.get_magnetic_moments(atoms)
        except TypeError:
            moments = calc.get_magnetic_moments()
        if moments is not None and len(moments) == len(original):
            magnetic_records = [
                {
                    "site_index": index,
                    "element": expected[index],
                    "magnetic_moment": float(value),
                    "descriptor": "gpaw_site_magnetic_moment",
                }
                for index, value in enumerate(moments)
            ]
    except Exception:
        magnetic_records = []

    orbital_records: list[dict[str, Any]] = []
    dos_descriptors: list[dict[str, Any]] = []
    limitations = [
        "GPAW DOS/PDOS integrals and local magnetic moments are supporting electronic descriptors, not integer formal oxidation-state assignments.",
        "GPAW atomic-projector PDOS is a qualitative local-character measure because the PAW partial-wave projectors are not an orthonormal atomic basis.",
    ]
    try:
        import numpy as np

        emin = float(settings.get("dos_emin_eV", -10.0))
        emax = float(settings.get("dos_emax_eV", 5.0))
        npoints = int(settings.get("dos_npoints", 601))
        width = float(settings.get("dos_width_eV", 0.10))
        if emax <= emin or npoints < 3 or width < 0:
            raise ValueError("invalid DOS window/npoints/width")
        energies = np.linspace(emin, emax, npoints)
        doscalc = calc.dos()
        total_dos = doscalc.raw_dos(energies, spin=None, width=width)
        total_dos_fermi = (
            float(np.interp(0.0, energies, total_dos)) if emin <= 0.0 <= emax else None
        )
        dos_descriptors.append(
            {
                "efermi_eV": efermi,
                "total_dos_at_efermi_states_per_eV": total_dos_fermi,
                "energy_reference": "E-E_F",
                "descriptor": "gpaw_total_dos",
            }
        )

        with (workdir / "dos.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["energy_relative_to_efermi_eV", "total_dos_states_per_eV"])
            writer.writerows((float(e), float(d)) for e, d in zip(energies, total_dos))

        occupied = [i for i, energy in enumerate(energies) if float(energy) <= 0.0]
        stop = occupied[-1] + 1 if occupied else 0
        labels = {0: "s", 1: "p", 2: "d", 3: "f"}
        for index, element in enumerate(expected):
            populations: dict[str, float] = {}
            if stop >= 2:
                for angular, label in labels.items():
                    try:
                        pdos = doscalc.raw_pdos(
                            energies,
                            a=index,
                            l=angular,
                            spin=None,
                            width=width,
                        )
                        populations[label] = _trapz(
                            [float(value) for value in energies[:stop]],
                            [float(value) for value in pdos[:stop]],
                        )
                    except Exception:
                        continue
            orbital_records.append(
                {
                    "site_index": index,
                    "element": element,
                    "orbital_populations": populations,
                    "descriptor": "gpaw_integrated_projector_pdos_below_efermi",
                }
            )
        (workdir / "pdos_integrals.json").write_text(
            json.dumps(orbital_records, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        limitations.append(f"GPAW DOS/PDOS extraction was unavailable for this result: {type(exc).__name__}: {exc}")

    if magnetic_records:
        with (workdir / "magnetic_moments.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["site_index", "element", "magnetic_moment", "descriptor"],
            )
            writer.writeheader()
            writer.writerows(magnetic_records)

    summary = {
        "code": "GPAW",
        "workdir": str(workdir),
        "gpw_file": str(gpw_path),
        "execute": bool(settings.get("execute", False)),
        "final_energy_eV": final_energy,
        "efermi_eV": efermi,
        "dos_descriptors": dos_descriptors,
    }
    (workdir / "electronic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    result = base_method_result(
        method="dft-electronic",
        target=target,
        scope="site-resolved",
        status="descriptors-only",
        orbital_populations=orbital_records,
        magnetic_moments=magnetic_records,
        provenance=summary,
        limitations=limitations,
    )
    result["dos_descriptors"] = dos_descriptors
    return result


def _parse_acf(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 7 or not parts[0].isdigit():
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]) - 1,
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                    "electrons": float(parts[4]),
                    "min_dist": float(parts[5]),
                    "volume": float(parts[6]),
                }
            )
        except ValueError:
            continue
    return rows



def _write_gpaw_all_electron_density(
    gpw_path: Path,
    density_path: Path,
    *,
    gridrefinement: int,
) -> None:
    if gridrefinement not in {1, 2, 4}:
        raise ValueError("[oxidation.bader].gridrefinement must be one of 1, 2, or 4")
    try:
        from ase.io import write
        from ase.units import Bohr
    except ImportError as exc:
        raise OptionalMethodUnavailable("GPAW Bader preparation requires ASE") from exc
    atoms, calc = _gpaw_restart(gpw_path)
    try:
        rho = calc.get_all_electron_density(gridrefinement=gridrefinement)
        write(str(density_path), atoms, data=rho * Bohr**3)
    except Exception as exc:
        raise RuntimeError(f"Could not reconstruct GPAW all-electron density for Bader: {exc}") from exc


def _run_bader(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    workdir = _workdir(target, cfg, settings)
    gpw_path = workdir / str(settings.get("gpw_file", "oxidation.gpw"))
    density_path = workdir / str(settings.get("density_file", "density.cube"))
    acf = workdir / str(settings.get("acf_file", "ACF.dat"))
    gridrefinement = int(settings.get("gridrefinement", 4))

    if bool(settings.get("execute", False)):
        workdir.mkdir(parents=True, exist_ok=True)
        if not gpw_path.is_file():
            raise OptionalMethodUnavailable(
                f"Bader execution needs the GPAW restart {gpw_path}. Run dft-electronic first "
                "with the same output_root/workdir and execute=true, or provide an existing .gpw file."
            )
        _write_gpaw_all_electron_density(
            gpw_path,
            density_path,
            gridrefinement=gridrefinement,
        )
        command = settings.get("command") or ["bader", density_path.name]
        tokens = _command_tokens(command, target=target, workdir=workdir)
        try:
            completed = subprocess.run(
                tokens,
                cwd=workdir,
                check=False,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise OptionalMethodUnavailable(
                "The Bader executable was not found. Install the free Henkelman-group Bader program "
                "or configure [oxidation.bader].command to its executable path."
            ) from exc
        (workdir / "dopingflow_bader_stdout.txt").write_text(
            completed.stdout or "", encoding="utf-8"
        )
        (workdir / "dopingflow_bader_stderr.txt").write_text(
            completed.stderr or "", encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Bader command failed with exit code {completed.returncode}; see dopingflow_bader_stderr.txt"
            )

    if not acf.is_file():
        raise OptionalMethodUnavailable(
            f"Bader analysis needs {acf}. Set execute=true to generate an all-electron density "
            "from GPAW and run Bader, or point workdir/output_root to existing Bader outputs."
        )

    structure = Structure.from_file(target.structure_path)
    rows = _parse_acf(acf)
    if len(rows) != len(structure):
        raise RuntimeError(
            f"Bader ACF.dat contains {len(rows)} atom rows for a {len(structure)}-site structure"
        )

    charge_records = []
    for index, (site, row) in enumerate(zip(structure, rows)):
        atomic_number = int(site.specie.Z)
        partial = float(atomic_number) - float(row["electrons"])
        charge_records.append(
            {
                "site_index": index,
                "element": site.specie.symbol,
                "atomic_number": atomic_number,
                "bader_electrons": row["electrons"],
                "bader_partial_charge": partial,
                "assignment_status": "descriptor",
            }
        )

    return base_method_result(
        method="bader",
        target=target,
        scope="site-resolved",
        status="descriptors-only",
        bader_partial_charges=charge_records,
        provenance={
            "code": "GPAW+Bader",
            "workdir": str(workdir),
            "gpw_file": str(gpw_path) if gpw_path.is_file() else None,
            "density_file": str(density_path) if density_path.is_file() else None,
            "acf_file": str(acf),
            "gridrefinement": gridrefinement,
            "execute": bool(settings.get("execute", False)),
        },
        limitations=[
            "Bader charge is a partitioned continuous charge descriptor and is not, by itself, a formal integer oxidation-state assignment.",
            "When execute=true, dopingflow reconstructs GPAW's all-electron density before Bader partitioning; partial charge is Z minus the Bader electron population.",
            "No formal oxidation state is emitted by this method, even when a Bader partial charge is close to an integer.",
        ],
    )


def _parse_wannier_centres(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise RuntimeError("Wannier centres file is empty or malformed")
    centres = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4 or not parts[0].upper().startswith("X"):
            continue
        try:
            centres.append(
                {
                    "center_index": len(centres),
                    "label": parts[0],
                    "cartesian_angstrom": [float(parts[1]), float(parts[2]), float(parts[3])],
                }
            )
        except ValueError:
            continue
    return centres


def _run_wannier(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    workdir = _maybe_execute(target, cfg, settings, stage="wannier")
    centers_file = workdir / str(settings.get("centres_file", "wannier90_centres.xyz"))
    if not centers_file.is_file():
        raise OptionalMethodUnavailable(
            f"Wannier analysis needs {centers_file}. Run Wannier90 externally or set execute=true with an explicit command."
        )
    centres = _parse_wannier_centres(centers_file)
    result = base_method_result(
        method="wannier",
        target=target,
        scope="site-resolved",
        status="descriptors-only",
        provenance={
            "workdir": str(workdir),
            "centres_file": str(centers_file),
            "execute": bool(settings.get("execute", False)),
            "n_wannier_centres": len(centres),
        },
        limitations=[
            "Static Wannier-center information is supporting electronic evidence. It is not converted into formal integer oxidation states without an explicitly validated EOS/charge-pumping assignment procedure.",
        ],
    )
    result["wannier_descriptors"] = centres
    return result


def _run_eos(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    workdir = _maybe_execute(target, cfg, settings, stage="eos")
    results_file = workdir / str(settings.get("results_file", "eos_results.json"))
    if not results_file.is_file():
        raise OptionalMethodUnavailable(
            f"EOS formal assignment needs {results_file}. The adapter intentionally does not infer EOS labels from Bader charges or static Wannier centers."
        )
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("EOS results file must contain a JSON object")
    validated = bool(payload.get("validated", False))
    procedure = str(payload.get("assignment_procedure") or "").strip()
    if not validated or not procedure:
        return base_method_result(
            method="eos",
            target=target,
            scope="site-resolved",
            status="unsupported",
            provenance={
                "workdir": str(workdir),
                "results_file": str(results_file),
                "validated": validated,
                "assignment_procedure": procedure or None,
            },
            limitations=[
                "EOS formal oxidation states are accepted only from an explicitly documented result marked validated=true with a non-empty assignment_procedure."
            ],
        )
    structure = Structure.from_file(target.structure_path)
    raw_sites = payload.get("sites", [])
    if not isinstance(raw_sites, list):
        raise RuntimeError("EOS results 'sites' must be an array")
    assignments: list[int | None] = [None] * len(structure)
    for record in raw_sites:
        if not isinstance(record, dict):
            continue
        index = record.get("site_index")
        value = record.get("formal_oxidation_state")
        if index is None or value is None:
            continue
        index = int(index)
        if index < 0 or index >= len(structure):
            raise RuntimeError(f"EOS site_index {index} is outside the structure")
        element = str(record.get("element") or structure[index].specie.symbol)
        if element != structure[index].specie.symbol:
            raise RuntimeError(
                f"EOS element/site mismatch at site {index}: {element} != {structure[index].specie.symbol}"
            )
        numeric = float(value)
        integer = int(round(numeric))
        if not math.isclose(numeric, integer, abs_tol=1e-8):
            raise RuntimeError(
                f"EOS formal oxidation state at site {index} is not an integer: {value}"
            )
        assignments[index] = integer
    n_assigned = sum(value is not None for value in assignments)
    status = "assigned" if n_assigned == len(structure) else ("partial" if n_assigned else "unsupported")
    return base_method_result(
        method="eos",
        target=target,
        scope="site-resolved",
        status=status,
        formal_oxidation_states=site_records(structure, assignments, status="validated-eos"),
        method_scores=payload.get("scores", []) if isinstance(payload.get("scores", []), list) else [],
        provenance={
            "workdir": str(workdir),
            "results_file": str(results_file),
            "execute": bool(settings.get("execute", False)),
            "validated": True,
            "assignment_procedure": procedure,
            "calculation_provenance": payload.get("provenance", {}),
        },
        limitations=[
            "EOS labels are formal assignments from the explicitly documented external EOS procedure recorded in eos_results.json; Bader/DOS/magnetic descriptors remain separate supporting evidence."
        ],
    )


def run_dft_method(
    method: str,
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    if method == "dft-electronic":
        return _run_dft_electronic(target, cfg, settings)
    if method == "bader":
        return _run_bader(target, cfg, settings)
    if method == "wannier":
        return _run_wannier(target, cfg, settings)
    if method == "eos":
        return _run_eos(target, cfg, settings)
    raise ValueError(f"Unknown DFT oxidation method: {method}")

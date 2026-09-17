from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

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
        raise ValueError("DFT/post-processing command must be a string or array of strings")
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
    Poscar(structure).write_file(workdir / "POSCAR")
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


def _run_dft_electronic(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    code = str(settings.get("code", "vasp")).strip().lower()
    if code != "vasp":
        raise OptionalMethodUnavailable(
            f"The current dft-electronic post-processor supports VASP outputs only; requested code='{code}'."
        )
    workdir = _maybe_execute(target, cfg, settings, stage="dft_electronic")
    vasprun_path = workdir / str(settings.get("vasprun_file", "vasprun.xml"))
    if not vasprun_path.is_file():
        raise OptionalMethodUnavailable(
            f"VASP electronic analysis needs {vasprun_path}. Set execute=true with a command, or point workdir/output_root to existing outputs."
        )
    try:
        from pymatgen.io.vasp.outputs import Outcar, Vasprun
    except ImportError as exc:  # pragma: no cover
        raise OptionalMethodUnavailable("pymatgen VASP parsers are unavailable") from exc

    try:
        vasprun = Vasprun(
            vasprun_path,
            parse_dos=True,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not parse VASP vasprun.xml: {exc}") from exc

    original = Structure.from_file(target.structure_path)
    final_structure = vasprun.final_structure
    if len(final_structure) != len(original) or [site.specie.symbol for site in final_structure] != [
        site.specie.symbol for site in original
    ]:
        raise RuntimeError("DFT output site order/species do not match the analyzed relaxed structure")

    magnetic_records: list[dict[str, Any]] = []
    outcar_path = workdir / str(settings.get("outcar_file", "OUTCAR"))
    if outcar_path.is_file():
        try:
            outcar = Outcar(outcar_path)
            magnetization = outcar.magnetization
            if magnetization and len(magnetization) == len(original):
                for index, (site, entry) in enumerate(zip(original, magnetization)):
                    magnetic_records.append(
                        {
                            "site_index": index,
                            "element": site.specie.symbol,
                            "magnetic_moment": float(entry.get("tot", 0.0)),
                            "descriptor": "dft_outcar_site_magnetic_moment",
                        }
                    )
        except Exception:
            magnetic_records = []

    orbital_records: list[dict[str, Any]] = []
    dos_descriptors: list[dict[str, Any]] = []
    complete_dos = getattr(vasprun, "complete_dos", None)
    if complete_dos is not None:
        efermi = float(vasprun.efermi)
        try:
            interpolated = complete_dos.get_interpolated_value(efermi)
            total_dos_fermi = sum(float(value) for value in interpolated.values())
        except Exception:
            total_dos_fermi = None
        dos_descriptors.append(
            {
                "efermi_eV": efermi,
                "total_dos_at_efermi_states_per_eV": total_dos_fermi,
                "descriptor": "vasp_complete_dos",
            }
        )
        for index, site in enumerate(final_structure):
            try:
                spd = complete_dos.get_site_spd_dos(site)
            except Exception:
                continue
            populations: dict[str, float] = {}
            for orbital_type, dos in spd.items():
                energies = [float(value) for value in dos.energies]
                mask = [i for i, energy in enumerate(energies) if energy <= efermi]
                if len(mask) < 2:
                    continue
                stop = mask[-1] + 1
                try:
                    density_map = dos.densities
                    total_density = [
                        sum(float(values[i]) for values in density_map.values()) for i in range(stop)
                    ]
                    populations[str(orbital_type)] = _trapz(energies[:stop], total_density)
                except Exception:
                    continue
            orbital_records.append(
                {
                    "site_index": index,
                    "element": original[index].specie.symbol,
                    "orbital_populations": populations,
                    "descriptor": "integrated_projected_dos_below_efermi",
                }
            )

    result = base_method_result(
        method="dft-electronic",
        target=target,
        scope="site-resolved",
        status="descriptors-only",
        orbital_populations=orbital_records,
        magnetic_moments=magnetic_records,
        provenance={
            "code": "VASP",
            "workdir": str(workdir),
            "vasprun_file": str(vasprun_path),
            "outcar_file": str(outcar_path) if outcar_path.is_file() else None,
            "execute": bool(settings.get("execute", False)),
            "final_energy_eV": float(vasprun.final_energy),
            "efermi_eV": float(vasprun.efermi),
            "dos_descriptors": dos_descriptors,
        },
        limitations=[
            "Projected DOS integrals, orbital populations, and local magnetic moments are supporting electronic descriptors, not integer formal oxidation-state assignments.",
            "Projected populations depend on the projector/basis choices and integration convention used by the electronic-structure calculation.",
        ],
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


def _potcar_valence_map(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    elements = re.findall(r"VRHFIN\s*=\s*([A-Z][a-z]?)\s*:", text)
    zvals = re.findall(r"ZVAL\s*=\s*([-+0-9.Ee]+)", text)
    return {element: float(zval) for element, zval in zip(elements, zvals)}


def _run_bader(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    workdir = _maybe_execute(target, cfg, settings, stage="bader")
    acf = workdir / str(settings.get("acf_file", "ACF.dat"))
    if not acf.is_file():
        raise OptionalMethodUnavailable(
            f"Bader analysis needs {acf}. Run Bader externally or set execute=true with an explicit command."
        )
    structure = Structure.from_file(target.structure_path)
    rows = _parse_acf(acf)
    if len(rows) != len(structure):
        raise RuntimeError(
            f"Bader ACF.dat contains {len(rows)} atom rows for a {len(structure)}-site structure"
        )
    configured_valence = settings.get("valence_electrons", {}) or {}
    if not isinstance(configured_valence, dict):
        raise ValueError("[oxidation.bader].valence_electrons must be a table/dictionary")
    valence_map = _potcar_valence_map(workdir / str(settings.get("potcar_file", "POTCAR")))
    valence_map.update({str(key): float(value) for key, value in configured_valence.items()})

    charge_records = []
    missing_valence: set[str] = set()
    for index, (site, row) in enumerate(zip(structure, rows)):
        element = site.specie.symbol
        zval = valence_map.get(element)
        partial = None if zval is None else float(zval) - row["electrons"]
        if zval is None:
            missing_valence.add(element)
        charge_records.append(
            {
                "site_index": index,
                "element": element,
                "bader_electrons": row["electrons"],
                "valence_electrons": zval,
                "bader_partial_charge": partial,
                "assignment_status": "descriptor" if partial is not None else "missing-valence-reference",
            }
        )
    limitations = [
        "Bader charge is a partitioned continuous charge descriptor and is not, by itself, a formal integer oxidation-state assignment.",
        "No formal oxidation state is emitted by this method, even when Bader charges are close to integers.",
    ]
    if missing_valence:
        limitations.append(
            "Partial charges could not be computed for elements lacking a POTCAR/configured valence-electron reference: "
            + ", ".join(sorted(missing_valence))
        )
    return base_method_result(
        method="bader",
        target=target,
        scope="site-resolved",
        status="descriptors-only",
        bader_partial_charges=charge_records,
        provenance={
            "workdir": str(workdir),
            "acf_file": str(acf),
            "execute": bool(settings.get("execute", False)),
            "valence_electron_sources": valence_map,
        },
        limitations=limitations,
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

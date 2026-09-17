from __future__ import annotations

import re
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one literal match, found {count}: {old[:80]!r}")
    p.write_text(text.replace(old, new), encoding="utf-8")


def regex_once(path: str, pattern: str, repl: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    new, count = re.subn(pattern, repl, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"{path}: expected one regex match, found {count}: {pattern[:100]!r}")
    p.write_text(new, encoding="utf-8")


DFT = "src/dopingflow/oxidation_dft.py"
replace_once(DFT, "import json\nimport math\nimport re\n", "import csv\nimport json\nimport math\n")
replace_once(DFT, "from pymatgen.io.vasp import Poscar\n", "from pymatgen.io.ase import AseAtomsAdaptor\n")
replace_once(DFT, 'settings.get("output_root") or "dft_oxidation"', 'settings.get("output_root") or "gpaw_oxidation"')
replace_once(DFT, 'raise ValueError("DFT/post-processing command must be a string or array of strings")', 'raise ValueError("Post-processing command must be a string or array of strings")')
replace_once(
    DFT,
    '    Poscar(structure).write_file(workdir / "POSCAR")\n',
    '    structure.to(filename=str(workdir / "structure.cif"))\n',
)

new_dft_block = r'''def _parse_kpts(settings: dict[str, Any]) -> tuple[int, int, int]:
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
            "GPAW electronic analysis requires the optional GPAW package. "
            "Install dopingflow with [oxidation-gpaw] and install GPAW PAW setup data."
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
            "GPAW execution requires the optional GPAW package. Install dopingflow with "
            "[oxidation-gpaw] and install GPAW PAW setup data."
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


def _parse_acf'''
regex_once(
    DFT,
    r"def _run_dft_electronic\(.*?\n\ndef _parse_acf",
    new_dft_block,
)

regex_once(
    DFT,
    r"\ndef _potcar_valence_map\(.*?\n\ndef _run_bader",
    "\n\ndef _run_bader",
)

new_bader_block = r'''def _write_gpaw_all_electron_density(
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


def _parse_wannier_centres'''
regex_once(
    DFT,
    r"def _run_bader\(.*?\n\ndef _parse_wannier_centres",
    new_bader_block,
)

# GUI
GUI = "gui/pages/Oxidation_States.py"
p = Path(GUI)
text = p.read_text(encoding="utf-8")
text = text.replace('settings.get("output_root", "dft_oxidation")', 'settings.get("output_root", "gpaw_oxidation")')
p.write_text(text, encoding="utf-8")

new_gui_dft = r'''if "dft-electronic" in methods:
    with st.expander("GPAW electronic descriptors", expanded=False):
        dft = _table("dft_electronic")
        dft["code"] = "gpaw"
        st.info(
            "GPAW is the supported DFT backend for oxidation analysis. It is open source and runs "
            "directly from the relaxed structure without INCAR/KPOINTS/POTCAR-style per-structure inputs."
        )
        dft["output_root"] = st.text_input(
            "Per-target GPAW output root",
            value=str(dft.get("output_root", "gpaw_oxidation")),
            key="oxidation_dft_electronic_root",
        )
        dft["execute"] = st.checkbox(
            "Run GPAW single-point calculation",
            value=bool(dft.get("execute", False)),
            key="oxidation_dft_electronic_execute",
            help="Off means post-process an existing oxidation.gpw file only.",
        )
        f1, f2 = st.columns(2)
        dft["gpw_file"] = f1.text_input(
            "GPAW restart file", value=str(dft.get("gpw_file", "oxidation.gpw"))
        )
        dft["txt_file"] = f2.text_input(
            "GPAW log file", value=str(dft.get("txt_file", "gpaw.txt"))
        )
        g1, g2, g3 = st.columns(3)
        dft["xc"] = g1.text_input("XC functional", value=str(dft.get("xc", "PBE")))
        dft["mode"] = "pw"
        dft["ecut_eV"] = float(
            g2.number_input(
                "Plane-wave cutoff (eV)",
                min_value=50.0,
                value=float(dft.get("ecut_eV", 500.0)),
                step=25.0,
            )
        )
        dft["smearing_eV"] = float(
            g3.number_input(
                "Fermi-Dirac smearing (eV)",
                min_value=0.0,
                value=float(dft.get("smearing_eV", 0.05)),
                step=0.01,
            )
        )
        raw_kpts = dft.get("kpts", [1, 1, 1])
        if not isinstance(raw_kpts, (list, tuple)) or len(raw_kpts) != 3:
            raw_kpts = [1, 1, 1]
        k1, k2, k3, kg = st.columns(4)
        kx = int(k1.number_input("k₁", min_value=1, value=int(raw_kpts[0]), step=1))
        ky = int(k2.number_input("k₂", min_value=1, value=int(raw_kpts[1]), step=1))
        kz = int(k3.number_input("k₃", min_value=1, value=int(raw_kpts[2]), step=1))
        dft["kpts"] = [kx, ky, kz]
        dft["gamma"] = kg.checkbox("Gamma-centered", value=bool(dft.get("gamma", True)))
        c1, c2, c3 = st.columns(3)
        dft["convergence_density"] = float(
            c1.number_input(
                "Density convergence",
                min_value=1.0e-10,
                value=float(dft.get("convergence_density", 1.0e-5)),
                format="%.1e",
            )
        )
        dft["maxiter"] = int(
            c2.number_input("SCF max iterations", min_value=1, value=int(dft.get("maxiter", 333)), step=10)
        )
        dft["charge"] = float(
            c3.number_input("Net cell charge (e)", value=float(dft.get("charge", 0.0)), step=1.0)
        )
        spin_options = ["auto", "true", "false"]
        spin_current = str(dft.get("spinpol", "auto")).lower()
        if spin_current not in spin_options:
            spin_current = "auto"
        dft["spinpol"] = st.selectbox(
            "Spin polarization",
            spin_options,
            index=spin_options.index(spin_current),
            help="auto lets GPAW follow supplied initial magnetic moments; set true explicitly for magnetic systems when needed.",
        )
        magmom_text = json.dumps(dft.get("initial_magmoms", {}), indent=2)
        magmom_text = st.text_area(
            "Optional initial magnetic moments by element (JSON)",
            value=magmom_text,
            height=100,
            help='Example: {"Mn": 4.0, "Fe": 4.0, "Ni": 2.0}. Unlisted elements start at 0 μB.',
        )
        dft["initial_magmoms"] = _parse_json_mapping(magmom_text, "Initial magnetic moments")
        d1, d2, d3, d4 = st.columns(4)
        dft["dos_emin_eV"] = float(d1.number_input("DOS Emin (E-EF, eV)", value=float(dft.get("dos_emin_eV", -10.0))))
        dft["dos_emax_eV"] = float(d2.number_input("DOS Emax (E-EF, eV)", value=float(dft.get("dos_emax_eV", 5.0))))
        dft["dos_npoints"] = int(d3.number_input("DOS points", min_value=51, value=int(dft.get("dos_npoints", 601)), step=50))
        dft["dos_width_eV"] = float(d4.number_input("DOS width (eV)", min_value=0.0, value=float(dft.get("dos_width_eV", 0.10)), step=0.05))
        dft["save_wavefunctions"] = st.checkbox(
            "Store wavefunctions in .gpw (larger file)",
            value=bool(dft.get("save_wavefunctions", False)),
        )
        st.caption(
            "The calculation is single-point only. Energy cutoff, k-points, spin treatment, smearing, and convergence must be converged for the target chemistry before production use."
        )
        oxidation["dft_electronic"] = dft

if "bader" in methods:'''
regex_once(
    GUI,
    r'if "dft-electronic" in methods:.*?\nif "bader" in methods:',
    new_gui_dft,
)

new_gui_bader = r'''if "bader" in methods:
    with st.expander("GPAW + Bader charge analysis", expanded=False):
        bader = _table("bader")
        bader["output_root"] = st.text_input(
            "Per-target GPAW/Bader output root",
            value=str(bader.get("output_root", "gpaw_oxidation")),
            key="oxidation_bader_root",
            help="Use the same root as dft-electronic so Bader can reuse oxidation.gpw.",
        )
        bader["execute"] = st.checkbox(
            "Generate GPAW all-electron density and run Bader",
            value=bool(bader.get("execute", False)),
            key="oxidation_bader_execute",
            help="Requires an existing GPAW .gpw file in the same per-target directory and the free Bader executable.",
        )
        b1, b2, b3 = st.columns(3)
        bader["gpw_file"] = b1.text_input(
            "GPAW restart file", value=str(bader.get("gpw_file", "oxidation.gpw"))
        )
        bader["density_file"] = b2.text_input(
            "All-electron density cube", value=str(bader.get("density_file", "density.cube"))
        )
        bader["acf_file"] = b3.text_input(
            "Bader ACF file", value=str(bader.get("acf_file", "ACF.dat"))
        )
        grid_options = [1, 2, 4]
        grid_current = int(bader.get("gridrefinement", 4))
        if grid_current not in grid_options:
            grid_current = 4
        bader["gridrefinement"] = st.selectbox(
            "GPAW all-electron density grid refinement",
            grid_options,
            index=grid_options.index(grid_current),
        )
        command_text = st.text_input(
            "Bader command",
            value=_command_text(bader.get("command", ["bader", "density.cube"])),
            disabled=not bader["execute"],
            help="Default uses the free 'bader' executable on density.cube; no shell is used.",
        )
        bader["command"] = _parse_command(command_text)
        st.caption(
            "dopingflow reconstructs GPAW's all-electron density and reports continuous Bader charges as Z − basin electrons. It never converts them automatically into formal integer oxidation states."
        )
        oxidation["bader"] = bader

if "wannier" in methods:'''
regex_once(
    GUI,
    r'if "bader" in methods:.*?\nif "wannier" in methods:',
    new_gui_bader,
)

# Optional dependency
PYPROJECT = "pyproject.toml"
replace_once(
    PYPROJECT,
    'oxidation-bertos = [\n  "torch>=2.0",\n  "transformers>=4.30",\n]\n\n',
    'oxidation-bertos = [\n  "torch>=2.0",\n  "transformers>=4.30",\n]\n\noxidation-gpaw = [\n  "gpaw>=24.6",\n]\n\n',
)

# Tests
TESTS = "tests/test_oxidation.py"
p = Path(TESTS)
text = p.read_text(encoding="utf-8")
text = text.replace(
    '        "1 0.0 0.0 0.0 3.60 0.50 10.0\\n"\n        "2 2.5 2.5 2.5 6.30 0.50 11.0\\n",',
    '        "1 0.0 0.0 0.0 49.60 0.50 10.0\\n"\n        "2 2.5 2.5 2.5 8.30 0.50 11.0\\n",',
)
text = text.replace(
    '            "execute": False,\n            "valence_electrons": {"Sn": 4, "O": 6},\n',
    '            "execute": False,\n',
)
needle = '    assert result["bader_partial_charges"][0]["bader_partial_charge"] == pytest.approx(0.4)\n'
if needle not in text:
    raise SystemExit("tests: Bader assertion anchor missing")
text = text.replace(
    needle,
    '    assert result["bader_partial_charges"][0]["atomic_number"] == 50\n' + needle,
    1,
)
insert_anchor = '\n\ndef test_eos_requires_explicit_validated_assignment(tmp_path: Path) -> None:\n'
if insert_anchor not in text:
    raise SystemExit("tests: EOS anchor missing")
new_test = r'''

def test_dft_electronic_is_gpaw_only_and_requires_gpw_when_not_executing(tmp_path: Path) -> None:
    structure_path = tmp_path / "POSCAR"
    Poscar(_structure()).write_file(structure_path)
    target = StructureTarget(
        target_id="parent",
        parent_id="parent",
        kind="vacancy-free",
        structure_path=structure_path,
        n_vacancies=0,
        vacancy_species=None,
    )
    cfg = _cfg(tmp_path, methods=("dft-electronic",))

    with pytest.raises(OptionalMethodUnavailable, match="GPAW-only"):
        run_dft_method(
            "dft-electronic",
            target,
            cfg,
            {"code": "vasp", "execute": False},
        )

    with pytest.raises(OptionalMethodUnavailable, match="oxidation.gpw"):
        run_dft_method(
            "dft-electronic",
            target,
            cfg,
            {"code": "gpaw", "execute": False},
        )
'''
text = text.replace(insert_anchor, new_test + insert_anchor, 1)
p.write_text(text, encoding="utf-8")

# README
README = "README.md"
p = Path(README)
text = p.read_text(encoding="utf-8")
text = text.replace(
    'pip install -e ".[oxidation-bertos]"  # BERTOS composition-token model\n',
    'pip install -e ".[oxidation-bertos]"  # BERTOS composition-token model\n'
    'pip install -e ".[oxidation-gpaw]"    # GPAW single-point DFT / electronic descriptors\n',
)
text = text.replace(
    '- **DFT:** VASP electronic descriptors, Bader, Wannier descriptors, and validated\n  EOS/charge-pumping formal assignments.',
    '- **DFT:** GPAW single-point electronic descriptors, GPAW all-electron-density Bader analysis, Wannier descriptors, and validated\n  EOS/charge-pumping formal assignments.',
)
text = text.replace(
    '- New DFT calculations are opt-in: `execute = false` is the safe default, and an\n  explicit external command is required when execution is enabled.',
    '- New GPAW single-point calculations are opt-in: `execute = false` is the safe default.\n  When enabled, dopingflow runs GPAW directly from the relaxed structure; Bader execution\n  separately requires the free `bader` executable.',
)
text = text.replace(
    '  settings, DFT/Bader/Wannier/EOS post-processing, and optional candidate-limited\n  DFT follow-up. It can save `[oxidation]`, run `dopingflow oxidation`, and inspect\n  the generated CSV/JSON results. External DFT execution remains behind explicit\n  method-level and follow-up `execute` gates plus a GUI confirmation.',
    '  settings, GPAW/Bader/Wannier/EOS analysis, and optional candidate-limited\n  DFT follow-up. It can save `[oxidation]`, run `dopingflow oxidation`, and inspect\n  the generated CSV/JSON results. GPAW and Bader execution remain behind explicit\n  method-level and follow-up `execute` gates plus a GUI confirmation.',
)
anchor = 'Run it with:\n\n```bash\ndopingflow oxidation -c input.toml --strategy structural --methods bond-valence\n```\n'
if anchor not in text:
    raise SystemExit("README: oxidation smoke-test anchor missing")
gpaw_example = anchor + '''\nFor open-source DFT electronic descriptors, install GPAW and its PAW datasets, then configure one reusable parameter set rather than per-structure DFT input files:\n\n```bash\npip install -e ".[oxidation-gpaw]"\ngpaw install-data\n```\n\n```toml\n[oxidation.dft_electronic]\ncode = "gpaw"\noutput_root = "gpaw_oxidation"\nexecute = false              # true runs the single point directly\nmode = "pw"\necut_eV = 500.0\nxc = "PBE"\nkpts = [1, 1, 1]\ngamma = true\nsmearing_eV = 0.05\nconvergence_density = 1e-5\nspinpol = "auto"\n```\n\nThe generated per-target GPAW directory can contain `oxidation.gpw`, `gpaw.txt`, `dos.csv`, `pdos_integrals.json`, `magnetic_moments.csv`, and `electronic_summary.json`. Cutoff, k-point, spin, smearing, and convergence settings must be converged for the target chemistry.\n'''
text = text.replace(anchor, gpaw_example, 1)
p.write_text(text, encoding="utf-8")

# Sphinx method guide
DOC = "docs/source/methods/oxidation_states.rst"
p = Path(DOC)
text = p.read_text(encoding="utf-8")
new_dft_docs = '''DFT interpretation
------------------

The DFT strategy uses **GPAW** as its electronic-structure backend.  GPAW is
open source and is driven directly from Python/ASE, so dopingflow does not
require per-structure ``INCAR``, ``KPOINTS``, ``POTCAR``, or equivalent input
files.  Install the optional dependency and GPAW PAW datasets before execution::

   pip install -e ".[oxidation-gpaw]"
   gpaw install-data

``dft-electronic``
   Runs or reopens a GPAW ``.gpw`` ground-state calculation.  The current
   direct-execution path uses periodic plane-wave mode and records total energy,
   Fermi level, site magnetic moments, total DOS, and atom/angular-momentum
   projected DOS integrals.  ``dos.csv``, ``pdos_integrals.json``,
   ``magnetic_moments.csv``, and ``electronic_summary.json`` are written beside
   the ``oxidation.gpw`` restart file.  These quantities are descriptors only.

``bader``
   When ``execute=true``, dopingflow reopens the GPAW restart, reconstructs the
   all-electron density with ``get_all_electron_density()``, writes a cube file
   in the units expected by the Bader program, and invokes the free external
   ``bader`` executable.  Because the density contains all electrons, the
   reported continuous partial charge is ``Z - N_Bader``; no POTCAR or
   user-supplied valence-electron table is required.  Bader never emits an
   integer formal oxidation state by itself.

``wannier``
   Parses ``wannier90_centres.xyz`` and records Wannier-center information.
   Static centers are supporting descriptors and are not converted into formal
   oxidation states automatically.  GPAW/Wannier90 can be used externally to
   generate these centers when required.

``eos``
   The formal DFT-based assignment adapter.  It accepts assignments only from
   an explicitly documented external EOS/charge-pumping workflow result marked
   ``validated=true``.  It does not infer EOS labels from Bader charge, DOS, or
   static Wannier centers.

The GPAW single-point path is opt-in.  ``execute=false`` only post-processes an
existing ``oxidation.gpw``.  ``execute=true`` runs GPAW directly with the shared
settings in ``[oxidation.dft_electronic]``.  Energy cutoff, k-point sampling,
spin initialization, smearing, and SCF convergence remain scientific convergence
parameters and must be validated for the target chemistry.  Bader has its own
``execute`` gate because running the external Bader executable is a separate
post-processing action.

The EOS exchange file is intentionally simple and auditable::
'''
regex_once(
    DOC,
    r"DFT interpretation\n------------------\n.*?The EOS exchange file is intentionally simple and auditable::\n",
    new_dft_docs,
)
text = Path(DOC).read_text(encoding="utf-8")
text = text.replace('``dft_oxidation``', '``gpaw_oxidation``')
text = text.replace('output_root = "dft_oxidation"', 'output_root = "gpaw_oxidation"')
text = text.replace(
    'when the selected DFT method tables also contain the commands/work directories\nneeded for those calculations.',
    'when the selected method tables contain the GPAW/post-processing settings and work directories\nneeded for those calculations.',
)
Path(DOC).write_text(text, encoding="utf-8")

new_example5 = '''5. GPAW plus Bader and orbital analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two methods intentionally share the same per-target output root.  GPAW can
run the single point directly from the relaxed structure.  Bader can then
reconstruct GPAW's all-electron density and run the free ``bader`` executable.

::

   [oxidation]
   enabled = true
   strategy = "dft"
   methods = ["dft-electronic", "bader"]

   [oxidation.dft_electronic]
   code = "gpaw"
   output_root = "gpaw_oxidation"
   execute = true
   mode = "pw"
   ecut_eV = 500.0
   xc = "PBE"
   kpts = [1, 1, 1]
   gamma = true
   smearing_eV = 0.05
   convergence_density = 1e-5
   maxiter = 333
   spinpol = "auto"
   initial_magmoms = {}
   gpw_file = "oxidation.gpw"

   [oxidation.bader]
   output_root = "gpaw_oxidation"
   gpw_file = "oxidation.gpw"
   execute = true
   gridrefinement = 4
   density_file = "density.cube"
   acf_file = "ACF.dat"
   command = ["bader", "density.cube"]

CLI::

   dopingflow oxidation -c input.toml --strategy dft --methods dft-electronic,bader

Selecting only ``bader`` produces Bader descriptors and *no invented integer
formal oxidation-state assignment*.  If ``execute=true`` for Bader, an existing
GPAW restart must already be present in the same per-target directory.

6. DFT plus Wannier/EOS where supported
'''
regex_once(
    DOC,
    r"5\. DFT plus Bader and orbital analysis\n.*?6\. DFT plus Wannier/EOS where supported\n",
    new_example5,
)

# Final cleanup and consistency checks.
for file_name in [DFT, GUI, README, DOC]:
    text = Path(file_name).read_text(encoding="utf-8")
    forbidden = ["VASP electronic", "vasprun.xml", "OUTCAR", "POTCAR"]
    remaining = [token for token in forbidden if token in text]
    if remaining:
        raise SystemExit(f"{file_name}: retired VASP oxidation references remain: {remaining}")

print("GPAW oxidation migration applied")

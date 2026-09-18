from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected marker not found in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Backend: native GPAW -> Wannier90 occupied-manifold route.
# ---------------------------------------------------------------------------
path = "src/dopingflow/oxidation_dft.py"
replace_once(
    path,
    "import json\nimport math\nimport shlex\n",
    "import json\nimport math\nimport os\nimport shlex\n",
)

old = '''def _run_wannier(
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
'''

new = r'''def _occupied_gamma_manifold_info(
    calc: Any,
    *,
    occupation_tolerance: float,
    min_gap_eV: float,
) -> dict[str, Any]:
    """Validate the first native Wannier mode and identify its occupied manifold."""
    if occupation_tolerance <= 0 or occupation_tolerance >= 0.5:
        raise ValueError("[oxidation.wannier].occupation_tolerance must be between 0 and 0.5")
    if min_gap_eV < 0:
        raise ValueError("[oxidation.wannier].min_gap_eV must be >= 0")

    try:
        import numpy as np
    except ImportError as exc:
        raise OptionalMethodUnavailable("Native GPAW/Wannier analysis requires NumPy") from exc

    nspins = int(calc.get_number_of_spins())
    if nspins != 1:
        raise OptionalMethodUnavailable(
            "The native occupied-manifold Wannier mode currently supports non-spin-polarized GPAW results only."
        )

    try:
        bz_kpts = np.asarray(calc.get_bz_k_points(), dtype=float)
    except Exception:
        bz_kpts = np.asarray(calc.get_ibz_k_points(), dtype=float)
    if len(bz_kpts) != 1 or not np.allclose(bz_kpts[0], 0.0, atol=1.0e-10):
        raise OptionalMethodUnavailable(
            "The native occupied-manifold Wannier mode currently supports Gamma-only GPAW results."
        )

    occupations = np.asarray(calc.get_occupation_numbers(kpt=0, spin=0, raw=True), dtype=float)
    eigenvalues = np.asarray(calc.get_eigenvalues(kpt=0, spin=0), dtype=float)
    if len(occupations) != len(eigenvalues):
        raise RuntimeError("GPAW occupation/eigenvalue arrays have different lengths")

    partial = np.where(
        (occupations > occupation_tolerance)
        & (occupations < 1.0 - occupation_tolerance)
    )[0]
    if len(partial):
        raise OptionalMethodUnavailable(
            "The native occupied-manifold Wannier mode requires an isolated, integer-occupied manifold; "
            f"partially occupied bands were detected: {partial.tolist()}. Use an explicit projection/disentanglement workflow instead."
        )

    occupied = np.where(occupations >= 1.0 - occupation_tolerance)[0]
    if not len(occupied):
        raise RuntimeError("No fully occupied GPAW bands were found for Wannierization")
    noccupied = int(occupied[-1]) + 1
    if not np.array_equal(occupied, np.arange(noccupied)):
        raise OptionalMethodUnavailable(
            "The fully occupied GPAW bands are not a contiguous manifold starting from band 0."
        )
    if noccupied >= len(eigenvalues):
        raise OptionalMethodUnavailable(
            "At least one empty band is required to verify an insulating gap before native Wannierization."
        )

    gap = float(eigenvalues[noccupied] - eigenvalues[noccupied - 1])
    if gap < min_gap_eV:
        raise OptionalMethodUnavailable(
            f"The HOMO-LUMO gap is {gap:.6f} eV, below min_gap_eV={min_gap_eV:.6f}. "
            "Use an explicit disentanglement workflow for metallic/entangled states."
        )

    try:
        calc.get_pseudo_wave_function(band=0, kpt=0, spin=0)
    except Exception as exc:
        raise OptionalMethodUnavailable(
            "The GPAW restart does not expose stored wavefunctions. Re-run dft-electronic with "
            "save_wavefunctions=true so oxidation.gpw is written with mode='all'."
        ) from exc

    return {
        "n_bands_total": int(calc.get_number_of_bands()),
        "n_occupied_bands": noccupied,
        "n_spins": nspins,
        "homo_eV": float(eigenvalues[noccupied - 1]),
        "lumo_eV": float(eigenvalues[noccupied]),
        "gap_eV": gap,
        "fermi_level_eV": float(calc.get_fermi_level()),
        "occupation_tolerance": occupation_tolerance,
    }


def _write_gamma_bloch_wannier_input(
    path: Path,
    atoms: Any,
    *,
    noccupied: int,
    num_iter: int,
) -> None:
    """Write a projection-free Wannier90 input for an isolated Gamma manifold."""
    if noccupied <= 0:
        raise ValueError("Wannier occupied-band count must be positive")
    if num_iter <= 0:
        raise ValueError("[oxidation.wannier].num_iter must be positive")

    lines = [
        f"num_bands = {noccupied}",
        f"num_wann = {noccupied}",
        "mp_grid = 1 1 1",
        "gamma_only = true",
        "use_bloch_phases = true",
        f"num_iter = {num_iter}",
        "write_xyz = true",
        "write_hr = true",
        "",
        "begin unit_cell_cart",
        "ang",
    ]
    for vector in atoms.cell.array:
        lines.append("  " + " ".join(f"{float(value):.12f}" for value in vector))
    lines.extend(["end unit_cell_cart", "", "begin atoms_frac"])
    for symbol, frac in zip(
        atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=False)
    ):
        lines.append(
            f"{symbol} " + " ".join(f"{float(value):.12f}" for value in frac)
        )
    lines.extend(
        [
            "end atoms_frac",
            "",
            "begin kpoints",
            "0.000000000000 0.000000000000 0.000000000000",
            "end kpoints",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_native_gpaw_wannier(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    workdir = _workdir(target, cfg, settings)
    workdir.mkdir(parents=True, exist_ok=True)
    gpw_path = workdir / str(settings.get("gpw_file", "oxidation.gpw"))
    if not gpw_path.is_file():
        raise OptionalMethodUnavailable(
            f"Native Wannier execution needs the GPAW restart {gpw_path}. Run dft-electronic first "
            "with the same output_root/workdir and save_wavefunctions=true."
        )

    try:
        from gpaw.wannier.wannier90 import Wannier90
    except ImportError as exc:
        raise OptionalMethodUnavailable(
            "Native Wannier execution requires GPAW's Wannier90 interface and the external wannier90.x executable."
        ) from exc

    atoms, calc = _gpaw_restart(gpw_path)
    occupation_tolerance = float(settings.get("occupation_tolerance", 1.0e-4))
    min_gap_eV = float(settings.get("min_gap_eV", 1.0e-3))
    manifold = _occupied_gamma_manifold_info(
        calc,
        occupation_tolerance=occupation_tolerance,
        min_gap_eV=min_gap_eV,
    )
    noccupied = int(manifold["n_occupied_bands"])
    num_iter = int(settings.get("num_iter", 1000))
    seed = str(settings.get("seed", "wannier90")).strip() or "wannier90"
    if Path(seed).name != seed:
        raise ValueError("[oxidation.wannier].seed must be a simple filename stem")
    executable = str(settings.get("executable", "wannier90.x")).strip() or "wannier90.x"
    less_memory = bool(settings.get("less_memory", False))

    centers_file = workdir / str(settings.get("centres_file", f"{seed}_centres.xyz"))
    if centers_file.exists():
        centers_file.unlink()

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        w90 = Wannier90(
            calc,
            seed=seed,
            bands=range(noccupied),
            orbitals_ai=[[] for _ in atoms],
            spin=0,
            spinors=False,
        )
        _write_gamma_bloch_wannier_input(
            Path(f"{seed}.win"),
            atoms,
            noccupied=noccupied,
            num_iter=num_iter,
        )

        try:
            pp = subprocess.run(
                [executable, "-pp", seed],
                check=False,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise OptionalMethodUnavailable(
                f"Wannier90 executable '{executable}' was not found. Install it with "
                "`conda install -c conda-forge wannier90` or configure [oxidation.wannier].executable."
            ) from exc
        Path("dopingflow_wannier_pp_stdout.txt").write_text(
            pp.stdout or "", encoding="utf-8"
        )
        Path("dopingflow_wannier_pp_stderr.txt").write_text(
            pp.stderr or "", encoding="utf-8"
        )
        if pp.returncode != 0:
            raise RuntimeError(
                f"wannier90 preprocessing failed with exit code {pp.returncode}; "
                "see dopingflow_wannier_pp_stderr.txt"
            )

        # use_bloch_phases=true makes Wannier90 construct A_mn from the Bloch
        # states directly, so an .amn projection file is intentionally omitted.
        w90.write_eigenvalues()
        w90.write_overlaps(less_memory=less_memory)

        final = subprocess.run(
            [executable, seed],
            check=False,
            text=True,
            capture_output=True,
        )
        Path("dopingflow_wannier_stdout.txt").write_text(
            final.stdout or "", encoding="utf-8"
        )
        Path("dopingflow_wannier_stderr.txt").write_text(
            final.stderr or "", encoding="utf-8"
        )
        if final.returncode != 0:
            raise RuntimeError(
                f"wannier90 failed with exit code {final.returncode}; see dopingflow_wannier_stderr.txt"
            )
    finally:
        os.chdir(old_cwd)

    if not centers_file.is_file():
        raise RuntimeError(
            f"Wannier90 completed without producing the expected centres file {centers_file}"
        )

    metadata = {
        "code": "GPAW+Wannier90",
        "mode": "occupied-bloch",
        "workdir": str(workdir),
        "gpw_file": str(gpw_path),
        "seed": seed,
        "centres_file": str(centers_file),
        "executable": executable,
        "num_iter": num_iter,
        "less_memory": less_memory,
        **manifold,
    }
    (workdir / "wannier_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return centers_file, metadata


def _run_wannier(
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    workdir = _workdir(target, cfg, settings)
    execution_mode = str(settings.get("execution_mode", "native-gpaw")).strip().lower()
    metadata: dict[str, Any] = {}

    if bool(settings.get("execute", False)):
        if execution_mode == "native-gpaw":
            centers_file, metadata = _run_native_gpaw_wannier(target, cfg, settings)
        elif execution_mode == "external-command":
            workdir = _maybe_execute(target, cfg, settings, stage="wannier")
            centers_file = workdir / str(
                settings.get("centres_file", "wannier90_centres.xyz")
            )
        else:
            raise ValueError(
                "[oxidation.wannier].execution_mode must be native-gpaw or external-command"
            )
    else:
        centers_file = workdir / str(
            settings.get("centres_file", "wannier90_centres.xyz")
        )

    if not centers_file.is_file():
        raise OptionalMethodUnavailable(
            f"Wannier analysis needs {centers_file}. Enable native GPAW Wannier execution, "
            "run Wannier90 externally, or point workdir/output_root to existing centers."
        )
    centres = _parse_wannier_centres(centers_file)
    provenance = {
        "workdir": str(workdir),
        "centres_file": str(centers_file),
        "execute": bool(settings.get("execute", False)),
        "execution_mode": execution_mode,
        "n_wannier_centres": len(centres),
    }
    provenance.update(metadata)
    result = base_method_result(
        method="wannier",
        target=target,
        scope="site-resolved",
        status="descriptors-only",
        provenance=provenance,
        limitations=[
            "Static Wannier-center information is supporting electronic evidence. It is not converted into formal integer oxidation states without an explicitly validated EOS/charge-pumping assignment procedure.",
            "The native occupied-bloch mode currently supports isolated non-spin-polarized Gamma-only occupied manifolds; metallic, spin-polarized, multi-k, or entangled cases require an explicit projection/disentanglement workflow.",
        ],
    )
    result["wannier_descriptors"] = centres
    return result
'''
replace_once(path, old, new)


# ---------------------------------------------------------------------------
# GUI: native mode by default, while retaining the legacy command route.
# ---------------------------------------------------------------------------
gui = "gui/pages/Oxidation_States.py"
old_gui = '''if "wannier" in methods:
    with st.expander("Wannier descriptors", expanded=False):
        dft_method_panel(
            "wannier",
            "wannier",
            extra="Static Wannier centers are retained as descriptors and are not automatically converted to formal oxidation states.",
        )
        wannier = _table("wannier")
        wannier["centres_file"] = st.text_input(
            "Wannier centres file",
            value=str(wannier.get("centres_file", "wannier90_centres.xyz")),
        )
        oxidation["wannier"] = wannier
'''

new_gui = '''if "wannier" in methods:
    with st.expander("GPAW + Wannier90 descriptors", expanded=False):
        wannier = _table("wannier")
        inherited_root = _table("dft_electronic").get("output_root") or "dft_oxidation"
        wannier["output_root"] = st.text_input(
            "Per-target GPAW/Wannier output root",
            value=str(wannier.get("output_root") or inherited_root),
            key="oxidation_wannier_root",
            help="Use the same root as dft-electronic so Wannier can reuse oxidation.gpw.",
        )
        wannier["execute"] = st.checkbox(
            "Generate Wannier centres from the GPAW restart",
            value=bool(wannier.get("execute", False)),
            key="oxidation_wannier_execute",
            help="Requires oxidation.gpw written with stored wavefunctions and the wannier90.x executable.",
        )
        mode_options = ["native-gpaw", "external-command"]
        current_mode = str(wannier.get("execution_mode", "native-gpaw")).lower()
        if current_mode not in mode_options:
            current_mode = "native-gpaw"
        wannier["execution_mode"] = st.selectbox(
            "Wannier execution mode",
            mode_options,
            index=mode_options.index(current_mode),
            help="native-gpaw currently handles isolated non-spin-polarized Gamma-only occupied manifolds without disentanglement.",
        )
        w1, w2, w3 = st.columns(3)
        wannier["gpw_file"] = w1.text_input(
            "GPAW restart file", value=str(wannier.get("gpw_file", "oxidation.gpw"))
        )
        wannier["seed"] = w2.text_input(
            "Wannier90 seed", value=str(wannier.get("seed", "wannier90"))
        )
        wannier["executable"] = w3.text_input(
            "Wannier90 executable", value=str(wannier.get("executable", "wannier90.x"))
        )
        w4, w5, w6 = st.columns(3)
        wannier["num_iter"] = int(
            w4.number_input(
                "Wannier localization iterations",
                min_value=1,
                value=int(wannier.get("num_iter", 1000)),
                step=100,
            )
        )
        wannier["occupation_tolerance"] = float(
            w5.number_input(
                "Occupation tolerance",
                min_value=1.0e-8,
                max_value=0.1,
                value=float(wannier.get("occupation_tolerance", 1.0e-4)),
                format="%.1e",
            )
        )
        wannier["min_gap_eV"] = float(
            w6.number_input(
                "Minimum insulating gap (eV)",
                min_value=0.0,
                value=float(wannier.get("min_gap_eV", 1.0e-3)),
                format="%.3e",
            )
        )
        wannier["less_memory"] = st.checkbox(
            "Low-memory GPAW overlap generation",
            value=bool(wannier.get("less_memory", False)),
            help=(
                "For the current single-Gamma route leave this off unless needed. "
                "The option is retained for future multi-k workflows."
            ),
        )
        default_centres = f"{wannier['seed']}_centres.xyz"
        wannier["centres_file"] = st.text_input(
            "Wannier centres file",
            value=str(wannier.get("centres_file", default_centres)),
        )
        if wannier["execution_mode"] == "external-command":
            command_text = st.text_input(
                "External Wannier command",
                value=_command_text(wannier.get("command", [])),
                disabled=not wannier["execute"],
                help="Legacy/custom route. The native GPAW route does not need this field.",
            )
            wannier["command"] = _parse_command(command_text)
            if wannier["execute"] and not wannier["command"]:
                st.error(
                    "wannier: external-command mode requires a command when execution is enabled."
                )
        else:
            wannier.pop("command", None)
            st.info(
                "Native mode detects the fully occupied bands, verifies a finite gap, uses Bloch phases "
                "as the initial gauge (no arbitrary atomic projection choice), writes .eig/.mmn through "
                "GPAW, runs wannier90.x, and parses the resulting *_centres.xyz file."
            )
        st.caption(
            "Static Wannier centers remain descriptors only. Formal oxidation states require a validated EOS/charge-pumping procedure."
        )
        oxidation["wannier"] = wannier
'''
replace_once(gui, old_gui, new_gui)


# ---------------------------------------------------------------------------
# Tests: no GPAW/Wannier executable is required in CI for these unit checks.
# ---------------------------------------------------------------------------
tests = Path("tests/test_oxidation.py")
text = tests.read_text(encoding="utf-8")
marker = "\n\ndef test_bader_is_descriptor_only_and_never_invents_integer_state(tmp_path: Path) -> None:\n"
if marker not in text:
    raise SystemExit("Could not find oxidation test insertion marker")

test_code = r'''


def test_native_wannier_detects_isolated_occupied_gamma_manifold() -> None:
    import numpy as np

    class Calc:
        def get_number_of_spins(self):
            return 1

        def get_bz_k_points(self):
            return np.array([[0.0, 0.0, 0.0]])

        def get_occupation_numbers(self, kpt=0, spin=0, raw=True):
            assert (kpt, spin, raw) == (0, 0, True)
            return np.array([1.0, 0.999999, 0.000001, 0.0])

        def get_eigenvalues(self, kpt=0, spin=0):
            assert (kpt, spin) == (0, 0)
            return np.array([-5.0, -1.0, 0.5, 1.0])

        def get_pseudo_wave_function(self, band=0, kpt=0, spin=0):
            return np.ones((2, 2, 2))

        def get_number_of_bands(self):
            return 4

        def get_fermi_level(self):
            return -0.25

    info = oxidation_dft._occupied_gamma_manifold_info(
        Calc(), occupation_tolerance=1.0e-4, min_gap_eV=1.0e-3
    )
    assert info["n_occupied_bands"] == 2
    assert info["gap_eV"] == pytest.approx(1.5)


def test_native_wannier_rejects_partial_occupations() -> None:
    import numpy as np

    class Calc:
        def get_number_of_spins(self):
            return 1

        def get_bz_k_points(self):
            return np.array([[0.0, 0.0, 0.0]])

        def get_occupation_numbers(self, kpt=0, spin=0, raw=True):
            return np.array([1.0, 0.4, 0.0])

        def get_eigenvalues(self, kpt=0, spin=0):
            return np.array([-2.0, -0.1, 1.0])

    with pytest.raises(OptionalMethodUnavailable, match="partially occupied bands"):
        oxidation_dft._occupied_gamma_manifold_info(
            Calc(), occupation_tolerance=1.0e-4, min_gap_eV=1.0e-3
        )


def test_native_wannier_input_uses_bloch_phases_and_xyz(tmp_path: Path) -> None:
    from ase import Atoms

    atoms = Atoms(
        "SnO2",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    path = tmp_path / "wannier90.win"
    oxidation_dft._write_gamma_bloch_wannier_input(
        path, atoms, noccupied=7, num_iter=800
    )
    content = path.read_text(encoding="utf-8")
    assert "num_bands = 7" in content
    assert "num_wann = 7" in content
    assert "mp_grid = 1 1 1" in content
    assert "gamma_only = true" in content
    assert "use_bloch_phases = true" in content
    assert "num_iter = 800" in content
    assert "write_xyz = true" in content
    assert "begin projections" not in content.lower()
'''
tests.write_text(text.replace(marker, test_code + marker, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Documentation.
# ---------------------------------------------------------------------------
readme = "README.md"
replace_once(
    readme,
    '''- Static Wannier centers are descriptors. Formal DFT labels are accepted only
  from an explicitly validated EOS/charge-pumping result with a documented
  assignment procedure.
''',
    '''- Static Wannier centers are descriptors. With ``execute = true``, the native
  GPAW/Wannier90 route can generate them automatically for an isolated,
  non-spin-polarized Gamma-only occupied manifold from an ``oxidation.gpw``
  that contains wavefunctions. It detects the occupied bands, uses Bloch phases
  as the initial gauge (no arbitrary atomic projection choice), writes the GPAW
  eigenvalue/overlap files, runs ``wannier90.x``, and parses ``*_centres.xyz``.
  Metallic, spin-polarized, multi-k, or entangled cases still require an explicit
  projection/disentanglement workflow. Formal DFT labels are accepted only from
  an explicitly validated EOS/charge-pumping result with a documented assignment
  procedure.
''',
)

docs = "docs/source/methods/oxidation_states.rst"
replace_once(
    docs,
    '''``wannier``
   Parses ``wannier90_centres.xyz`` and records Wannier-center information.
   Static centers are supporting descriptors and are not converted into formal
   oxidation states automatically.  GPAW/Wannier90 can be used externally to
   generate these centers when required.
''',
    '''``wannier``
   Parses Wannier-center information and can generate it natively from an
   existing GPAW restart when ``execute=true``.  The first native mode is
   deliberately conservative: it supports an isolated, non-spin-polarized,
   Gamma-only occupied manifold, requires wavefunctions stored in
   ``oxidation.gpw``, detects the fully occupied bands and finite HOMO-LUMO gap,
   writes a Wannier90 input using Bloch phases as the initial gauge and
   ``write_xyz=true``, lets GPAW write ``.eig``/``.mmn``, and runs
   ``wannier90.x``.  Because the occupied manifold is isolated, no
   disentanglement is used and no arbitrary atomic projector set is invented.
   Metallic, spin-polarized, multi-k, or entangled cases are rejected by this
   native mode and should use an explicit projection/disentanglement workflow.
   Static centers remain supporting descriptors and are not converted into
   formal oxidation states automatically.
''',
)

# Remove the temporary implementation machinery from the final feature commit.
Path(".github/workflows/add-native-wannier.yml").unlink(missing_ok=True)
Path(".github/scripts/patch_native_wannier.py").unlink(missing_ok=True)

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_file(path: str, content: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Core integration
# ---------------------------------------------------------------------------
replace_once(
    "src/dopingflow/vacancy_static_thermodynamics.py",
    "from dopingflow.vacancy_analysis import (\n",
    "from dopingflow.vacancy_configurational_thermodynamics import (\n"
    "    augment_vacancy_formation_free_energy,\n"
    ")\n"
    "from dopingflow.vacancy_analysis import (\n",
    "configurational helper import",
)
replace_once(
    "src/dopingflow/vacancy_static_thermodynamics.py",
    '_SOLID_CONFIG_ENTROPY_MODES = {"none", "ideal"}',
    '_SOLID_CONFIG_ENTROPY_MODES = {"none", "ideal", "configurational"}',
    "entropy mode choices",
)
replace_once(
    "src/dopingflow/vacancy_static_thermodynamics.py",
    '            "[vacancies].solid_configurational_entropy must be \'none\' or \'ideal\'"\n',
    '            "[vacancies].solid_configurational_entropy must be one of: "\n'
    '            "none, ideal, configurational"\n',
    "entropy validation message",
)
replace_once(
    "src/dopingflow/vacancy_static_thermodynamics.py",
    'def _enhanced_static_approximation(entropy_mode: str) -> str:\n    if entropy_mode == "ideal":\n',
    'def _enhanced_static_approximation(entropy_mode: str) -> str:\n'
    '    if entropy_mode == "configurational":\n'
    '        return (\n'
    '            "Finite-T screening uses 0 K relaxed ML minima plus an oxygen-vacancy "\n'
    '            "configurational partition-function correction from exact symmetry orbits. "\n'
    '            "The complete relaxed spectrum is used when available; otherwise the "\n'
    '            "complete exact single-point spectrum supplies the configurational correction. "\n'
    '            "Solid vibrational, zero-point, thermal-electronic, magnetic, anharmonic, "\n'
    '            "thermal-expansion and pV terms are neglected."\n'
    '        )\n'
    '    if entropy_mode == "ideal":\n',
    "configurational approximation text",
)
replace_once(
    "src/dopingflow/vacancy_static_thermodynamics.py",
    '''    if requested not in _CALIBRATED_MODES:\n        return analyze_vacancy_thermodynamics(\n            rows=rows,\n            analysis_cfg=_base_analysis_config(cfg),\n            parent_root=parent_root,\n            backend=backend,\n            model=model,\n            task=task,\n            calculator=calculator,\n            optimizer=optimizer,\n            fmax=fmax,\n            max_steps=max_steps,\n            source_database=source_database,\n        )\n''',
    '''    if requested not in _CALIBRATED_MODES:\n        outputs = analyze_vacancy_thermodynamics(\n            rows=rows,\n            analysis_cfg=_base_analysis_config(cfg),\n            parent_root=parent_root,\n            backend=backend,\n            model=model,\n            task=task,\n            calculator=calculator,\n            optimizer=optimizer,\n            fmax=fmax,\n            max_steps=max_steps,\n            source_database=source_database,\n        )\n        return augment_vacancy_formation_free_energy(\n            outputs=outputs,\n            rows=rows,\n            cfg=cfg,\n            parent_root=parent_root,\n            calibrated_reference=False,\n        )\n''',
    "legacy finite-T augmentation",
)
replace_once(
    "src/dopingflow/vacancy_static_thermodynamics.py",
    '''    return _postprocess_calibrated_outputs(\n        outputs=outputs,\n        cfg=cfg,\n        parent_root=parent_root,\n        backend=backend,\n        model=model,\n        task=task,\n    )\n''',
    '''    outputs = _postprocess_calibrated_outputs(\n        outputs=outputs,\n        cfg=cfg,\n        parent_root=parent_root,\n        backend=backend,\n        model=model,\n        task=task,\n    )\n    return augment_vacancy_formation_free_energy(\n        outputs=outputs,\n        rows=rows,\n        cfg=cfg,\n        parent_root=parent_root,\n        calibrated_reference=True,\n    )\n''',
    "calibrated finite-T augmentation",
)

# ---------------------------------------------------------------------------
# GUI configuration and Input Builder
# ---------------------------------------------------------------------------
replace_once(
    "gui/gui_config.py",
    '    "vacancies.solid_configurational_entropy": ["none", "ideal"],',
    '    "vacancies.solid_configurational_entropy": ["none", "ideal", "configurational"],',
    "GUI entropy choices",
)
replace_once(
    "gui/app.py",
    '''            st.warning(\n                "Static-lattice approximation: solid free energies are approximated by "\n                "0 K relaxed ML energies. Temperature and pressure enter only through "\n                "the oxygen-gas chemical potential. Vibrational, configurational, "\n                "electronic, magnetic and anharmonic free-energy contributions of the "\n                "solid are neglected."\n            )\n''',
    '''            st.warning(\n                "Static-lattice baseline: 0 K relaxed ML energies are retained. Gas-phase "\n                "O2 enthalpy/entropy and pressure can be added in the T-pO2 analysis. "\n                "Solid vacancy configurational entropy can optionally be included as an "\n                "ideal-mixing or explicit partition-function term; phonon, zero-point, "\n                "thermal-electronic, magnetic and anharmonic solid terms remain neglected."\n            )\n''',
    "GUI static warning",
)
replace_once(
    "gui/app.py",
    '''            vac["oxygen_reference_mode"] = st.selectbox(\n                "Oxygen reference mode", modes, index=modes.index(current_mode), key="vac_o_ref_mode"\n            )\n            if current_mode == "reference_file":\n''',
    '''            vac["oxygen_reference_mode"] = st.selectbox(\n                "Oxygen reference mode", modes, index=modes.index(current_mode), key="vac_o_ref_mode"\n            )\n            current_mode = vac["oxygen_reference_mode"]\n            if current_mode in {"global", "chemistry-specific"}:\n                vac["oxygen_reference_file"] = st.text_input(\n                    "Reference energies JSON",\n                    value=str(vac.get("oxygen_reference_file", defaults["oxygen_reference_file"])),\n                    key="vac_o_cal_ref_file",\n                    help="Uses already calculated binary oxide and bulk-metal references from refs-build.",\n                )\n                calibration_sources = CHOICES["vacancies.oxygen_calibration_experimental_source"]\n                current_calibration_source = str(\n                    vac.get(\n                        "oxygen_calibration_experimental_source",\n                        defaults["oxygen_calibration_experimental_source"],\n                    )\n                )\n                if current_calibration_source not in calibration_sources:\n                    current_calibration_source = defaults["oxygen_calibration_experimental_source"]\n                vac["oxygen_calibration_experimental_source"] = st.selectbox(\n                    "Experimental formation-enthalpy source",\n                    calibration_sources,\n                    index=calibration_sources.index(current_calibration_source),\n                    key="vac_o_cal_source",\n                )\n                if vac["oxygen_calibration_experimental_source"] in {"custom", "kingsbury+custom"}:\n                    vac["oxygen_calibration_experimental_data"] = st.text_input(\n                        "Custom experimental CSV",\n                        value=str(vac.get("oxygen_calibration_experimental_data", "")),\n                        key="vac_o_cal_custom",\n                    )\n                vac["oxygen_calibration_dataset_cache_dir"] = st.text_input(\n                    "Experimental dataset cache directory (optional)",\n                    value=str(vac.get("oxygen_calibration_dataset_cache_dir", "")),\n                    key="vac_o_cal_cache",\n                )\n                ccal1, ccal2 = st.columns(2)\n                vac["oxygen_calibration_min_references"] = int(ccal1.number_input(\n                    "Minimum calibration oxides",\n                    min_value=1,\n                    value=int(vac.get("oxygen_calibration_min_references", 2)),\n                    step=1,\n                    key="vac_o_cal_min_refs",\n                ))\n                vac["oxygen_calibration_include_host_oxide"] = ccal2.checkbox(\n                    "Include host oxide when eligible",\n                    value=bool(vac.get("oxygen_calibration_include_host_oxide", True)),\n                    key="vac_o_cal_host",\n                )\n                st.info(\n                    "Global uses all eligible real binary reference oxides. Chemistry-specific "\n                    "uses only oxides of the actual host/dopant cations. Missing oxide "\n                    "stoichiometries are never invented."\n                )\n            elif current_mode == "reference_file":\n''',
    "GUI calibrated oxygen controls",
)
replace_once(
    "gui/app.py",
    '''            else:\n                st.info("Minima will be written, but no preferred vacancy count across oxygen contents will be claimed.")\n            vac["allow_unverified_oxygen_reference"] = st.checkbox(\n                "Allow an unverifiable reference (recorded as unverified)",\n                value=bool(vac.get("allow_unverified_oxygen_reference", False)),\n                key="vac_allow_unverified_ref",\n            )\n''',
    '''            elif current_mode == "none":\n                st.info("Minima will be written, but no preferred vacancy count across oxygen contents will be claimed.")\n            if current_mode not in {"global", "chemistry-specific"}:\n                vac["allow_unverified_oxygen_reference"] = st.checkbox(\n                    "Allow an unverifiable reference (recorded as unverified)",\n                    value=bool(vac.get("allow_unverified_oxygen_reference", False)),\n                    key="vac_allow_unverified_ref",\n                )\n            else:\n                vac["allow_unverified_oxygen_reference"] = False\n''',
    "GUI verified calibration policy",
)
replace_once(
    "gui/app.py",
    '''            vac["static_energy_source"] = c1.selectbox("Static energy source", sources, index=sources.index(current_source), key="vac_static_source")\n            vac["exclude_unconverged"] = c2.checkbox("Exclude unconverged structures", value=bool(vac.get("exclude_unconverged", True)), key="vac_exclude_unconverged")\n\n            st.markdown("##### Temperature–pressure mapping")\n''',
    '''            vac["static_energy_source"] = c1.selectbox("Static energy source", sources, index=sources.index(current_source), key="vac_static_source")\n            vac["exclude_unconverged"] = c2.checkbox("Exclude unconverged structures", value=bool(vac.get("exclude_unconverged", True)), key="vac_exclude_unconverged")\n\n            entropy_modes = CHOICES["vacancies.solid_configurational_entropy"]\n            current_entropy_mode = str(\n                vac.get("solid_configurational_entropy", defaults["solid_configurational_entropy"])\n            )\n            if current_entropy_mode not in entropy_modes:\n                current_entropy_mode = defaults["solid_configurational_entropy"]\n            vac["solid_configurational_entropy"] = st.selectbox(\n                "Solid vacancy configurational treatment",\n                entropy_modes,\n                index=entropy_modes.index(current_entropy_mode),\n                key="vac_solid_config_entropy",\n                help=(\n                    "none: static-lattice solid. ideal: ideal occupied/vacant mixing entropy. "\n                    "configurational: explicit canonical partition function over exact "\n                    "symmetry-distinct vacancy configurations."\n                ),\n            )\n            if vac["solid_configurational_entropy"] == "configurational":\n                st.warning(\n                    "Partition-function mode requires exact vacancy enumeration and exact orbit "\n                    "degeneracies. If auto mode switches to sampling, analysis stops with a "\n                    "clear error rather than inventing degeneracies. The complete relaxed "\n                    "spectrum is used when available; otherwise the full exact single-point "\n                    "spectrum supplies the configurational correction to the relaxed minimum."\n                )\n            elif vac["solid_configurational_entropy"] == "ideal":\n                st.info(\n                    "Ideal mode adds -T S_mix from occupied/vacant oxygen-site mixing to the "\n                    "finite-temperature vacancy free energy."\n                )\n\n            st.markdown("##### Temperature–pressure mapping")\n''',
    "GUI entropy control",
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
replace_once(
    "tests/test_vacancies_gui.py",
    '    assert CHOICES["vacancies.solid_configurational_entropy"] == ["none", "ideal"]\n',
    '    assert CHOICES["vacancies.solid_configurational_entropy"] == [\n        "none", "ideal", "configurational"\n    ]\n',
    "GUI test entropy choices",
)

write_file(
    "tests/test_vacancy_configurational_thermodynamics.py",
    '''from __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nimport pytest\n\nfrom dopingflow.vacancy_configurational_thermodynamics import (\n    configurational_partition_thermodynamics,\n)\nfrom dopingflow.vacancy_static_thermodynamics import (\n    analyze_static_vacancy_thermodynamics,\n    parse_static_vacancy_thermodynamics_config,\n)\n\n\ndef _row(\n    configuration_id: str,\n    n_vacancies: int,\n    relaxed: float,\n    *,\n    sp: float | None = None,\n    degeneracy: int | None = None,\n    enumeration_mode: str = "exact",\n    degeneracy_is_exact: bool = True,\n) -> dict[str, object]:\n    row: dict[str, object] = {\n        "composition_directory": "Sb20",\n        "parent_id": "parent",\n        "configuration_id": configuration_id,\n        "host_species": "Sn",\n        "vacancy_species": "O",\n        "dopant_counts_from_parent": {"Sb": 2},\n        "dopant_counts_json": {"Sb": 2},\n        "n_host": 8,\n        "n_total_cations": 10,\n        "n_oxygen_sites_parent": 20,\n        "n_vacancies": n_vacancies,\n        "energy_relaxed_total_eV": relaxed,\n        "energy_sp_total_eV": relaxed if sp is None else sp,\n        "converged": True,\n        "relaxed_poscar_path": f"{configuration_id}/POSCAR",\n        "backend": "mace",\n        "model": "small",\n        "task": "",\n    }\n    if n_vacancies > 0:\n        row.update(\n            {\n                "enumeration_mode": enumeration_mode,\n                "degeneracy": degeneracy,\n                "degeneracy_is_exact": degeneracy_is_exact,\n            }\n        )\n    return row\n\n\ndef _minimum_row() -> dict[str, object]:\n    return {\n        "actual_composition_key": "Sb20",\n        "source_parent_id": "parent",\n        "dopant_counts_json": {"Sb": 2},\n        "n_total_cations": 10,\n        "n_oxygen_sites_parent": 20,\n        "n_vacancies": 1,\n    }\n\n\ndef test_partition_function_uses_exact_orbit_degeneracies():\n    rows = [\n        _row("v1a", 1, -94.0, degeneracy=2),\n        _row("v1b", 1, -93.9, degeneracy=3),\n    ]\n    result = configurational_partition_thermodynamics(\n        rows, _minimum_row(), 600.0\n    )\n    assert result["free_energy_correction_eV"] < 0.0\n    assert result["entropy_eV_per_K"] > 0.0\n    assert result["partition_total_degeneracy"] == 5\n    assert result["partition_orbit_count"] == 2\n    assert result["partition_exact"] is True\n    assert result["partition_energy_basis"] == "complete_exact_relaxed_spectrum"\n\n\ndef test_partition_function_uses_full_single_point_spectrum_if_relaxations_incomplete():\n    rows = [\n        _row("v1a", 1, -94.0, sp=-93.8, degeneracy=2),\n        _row("v1b", 1, -93.9, sp=-93.7, degeneracy=3),\n    ]\n    rows[1]["energy_relaxed_total_eV"] = None\n    result = configurational_partition_thermodynamics(\n        rows, _minimum_row(), 600.0\n    )\n    assert result["free_energy_correction_eV"] < 0.0\n    assert result["partition_energy_basis"].startswith("complete_exact_single_point")\n\n\ndef test_partition_function_rejects_sampled_degeneracies():\n    rows = [\n        _row(\n            "v1a",\n            1,\n            -94.0,\n            degeneracy=None,\n            enumeration_mode="sample",\n            degeneracy_is_exact=False,\n        )\n    ]\n    with pytest.raises(ValueError, match="requires exact vacancy enumeration"):\n        configurational_partition_thermodynamics(rows, _minimum_row(), 600.0)\n\n\ndef test_configurational_mode_changes_reported_vacancy_formation_free_energy(\n    tmp_path: Path,\n):\n    section = {\n        "static_thermodynamic_analysis": True,\n        "oxygen_reference_mode": "explicit",\n        "mu_O_reference_eV": -2.0,\n        "solid_configurational_entropy": "configurational",\n        "oxygen_standard_state_mode": "none",\n        "temperatures_K": [600.0],\n        "standard_oxygen_pressure_bar": 1.0,\n        "log10_pO2_min_bar": 0.0,\n        "log10_pO2_max_bar": 0.0,\n        "log10_pO2_step": 1.0,\n        "delta_mu_O_points_eV": [0.0],\n    }\n    cfg = parse_static_vacancy_thermodynamics_config(section, tmp_path)\n    rows = [\n        _row("parent_reference", 0, -100.0),\n        _row("v1a", 1, -94.0, degeneracy=2),\n        _row("v1b", 1, -93.9, degeneracy=3),\n    ]\n    outputs = analyze_static_vacancy_thermodynamics(\n        rows=rows,\n        cfg=cfg,\n        parent_root=tmp_path,\n        backend="mace",\n        model="small",\n        task="",\n    )\n    free_rows = json.loads(\n        outputs["vacancy_formation_free_energy_json"].read_text(encoding="utf-8")\n    )\n    one = next(row for row in free_rows if row["n_vacancies"] == 1)\n    # Static: (-94)-(-100) + 1*(-2) = 4 eV. The configurational\n    # partition function lowers the finite-T free energy.\n    assert one["vacancy_formation_energy_static_reference_eV"] == pytest.approx(4.0)\n    assert one["solid_configurational_free_energy_correction_eV"] < 0.0\n    assert one["vacancy_formation_free_energy_eV"] < 4.0\n    assert one["configurational_partition_total_degeneracy"] == 5\n\n    pressure = json.loads(\n        outputs["vacancy_static_pressure_map_json"].read_text(encoding="utf-8")\n    )\n    assert pressure[0]["solid_configurational_entropy_applied"] is True\n    metadata = json.loads(outputs["static_metadata"].read_text(encoding="utf-8"))\n    assert metadata["solid_configurational_entropy"] == "configurational"\n    assert "vacancy_formation_free_energy" in metadata["configurational_entropy_applied_to"]\n''',
)

# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------
replace_once(
    ".github/workflows/vacancy-calibration-tests.yml",
    '      - "src/dopingflow/vacancy_static_thermodynamics.py"\n',
    '      - "src/dopingflow/vacancy_static_thermodynamics.py"\n      - "src/dopingflow/vacancy_configurational_thermodynamics.py"\n',
    "CI push module path",
)
replace_once(
    ".github/workflows/vacancy-calibration-tests.yml",
    '      - "tests/test_vacancy_static_thermodynamics.py"\n',
    '      - "tests/test_vacancy_static_thermodynamics.py"\n      - "tests/test_vacancy_configurational_thermodynamics.py"\n',
    "CI push test path",
)
# The same path anchors appear a second time under pull_request after the first replacement.
replace_once(
    ".github/workflows/vacancy-calibration-tests.yml",
    '      - "src/dopingflow/vacancy_static_thermodynamics.py"\n',
    '      - "src/dopingflow/vacancy_static_thermodynamics.py"\n      - "src/dopingflow/vacancy_configurational_thermodynamics.py"\n',
    "CI PR module path",
)
replace_once(
    ".github/workflows/vacancy-calibration-tests.yml",
    '      - "tests/test_vacancy_static_thermodynamics.py"\n',
    '      - "tests/test_vacancy_static_thermodynamics.py"\n      - "tests/test_vacancy_configurational_thermodynamics.py"\n',
    "CI PR test path",
)
replace_once(
    ".github/workflows/vacancy-calibration-tests.yml",
    '''          pytest -q \\\n            tests/test_oxygen_calibration.py \\\n            tests/test_vacancy_static_thermodynamics.py \\\n            tests/test_vacancies_gui.py\n''',
    '''          python -m py_compile gui/app.py src/dopingflow/vacancy_configurational_thermodynamics.py\n          pytest -q \\\n            tests/test_oxygen_calibration.py \\\n            tests/test_vacancy_static_thermodynamics.py \\\n            tests/test_vacancy_configurational_thermodynamics.py \\\n            tests/test_vacancies_gui.py\n''',
    "CI command",
)

# ---------------------------------------------------------------------------
# README and examples
# ---------------------------------------------------------------------------
replace_once(
    "README.md",
    'solid_configurational_entropy = "none"       # optional: "ideal"\n',
    'solid_configurational_entropy = "none"       # optional: "ideal" or "configurational"\n',
    "README config example",
)
replace_once(
    "README.md",
    '''``solid_configurational_entropy = "ideal"`` optionally adds the ideal binary\noccupied/vacant oxygen-site mixing entropy to T-dependent pressure maps only;\nT-independent delta-mu stability intervals remain static-lattice quantities.\nAll other solid vibrational, zero-point, magnetic, electronic and anharmonic\nterms remain outside this screening level.\n''',
    '''``solid_configurational_entropy = "ideal"`` adds the ideal binary occupied/vacant\noxygen-site mixing entropy. ``"configurational"`` instead evaluates a canonical\npartition function over the exact symmetry-distinct vacancy configurations and\ntheir orbit degeneracies. If every exact configuration was relaxed, the relaxed\nspectrum is used; otherwise the complete exact single-point spectrum provides\nthe configurational correction to the relaxed minimum. Sampled enumeration is\nrejected for this mode because exact degeneracies are unavailable. Both entropy\ntreatments enter only finite-temperature outputs; direct delta-mu intervals remain\nstatic-lattice quantities. ``vacancy_formation_free_energy.csv/json`` reports\nDeltaG_vac(T,pO2) for every vacancy count. Solid vibrational, zero-point, magnetic,\nelectronic and anharmonic terms remain outside this screening level.\n''',
    "README entropy theory",
)
replace_once(
    "README.md",
    '- Select raw, global-calibrated, or chemistry-specific oxygen references for vacancy thermodynamics\n',
    '- Select raw, global-calibrated, or chemistry-specific oxygen references for vacancy thermodynamics\n- Select no, ideal, or explicit partition-function vacancy configurational entropy\n',
    "README GUI feature",
)
replace_once(
    "README.md",
    '│       ├── vacancy_analysis.py\n│       ├── vacancy_static_thermodynamics.py\n',
    '│       ├── vacancy_analysis.py\n│       ├── vacancy_configurational_thermodynamics.py\n│       ├── vacancy_static_thermodynamics.py\n',
    "README project tree",
)
replace_once(
    "examples/vacancies/input.toml",
    '''# Optional ideal occupied/vacant oxygen-site entropy. It is applied only to the\n# explicitly temperature-dependent T-pO2 map. Keep "none" for static-lattice\n# screening or select "ideal" to include the configurational contribution.\nsolid_configurational_entropy = "none"\n''',
    '''# Optional solid vacancy configurational treatment for finite-T free energies:\n# "none" = static lattice; "ideal" = ideal occupied/vacant mixing;\n# "configurational" = canonical partition function over exact symmetry orbits.\n# The configurational mode requires exact enumeration/degeneracies.\nsolid_configurational_entropy = "none"\n''',
    "example input entropy comments",
)
replace_once(
    "examples/vacancies/README.md",
    '''Solid configurational entropy remains optional:\n\n```toml\nsolid_configurational_entropy = "none"   # default\n# solid_configurational_entropy = "ideal"\n```\n\n`ideal` adds the binary occupied/vacant oxygen-site mixing entropy only to the\nexplicitly temperature-dependent T-pO2 map. The direct delta-mu intervals remain\nstatic-lattice quantities. Vibrational, zero-point, magnetic,\nthermal-electronic, anharmonic, thermal-expansion and solid-pV contributions\nare not part of this screening level.\n''',
    '''Solid configurational entropy remains optional:\n\n```toml\nsolid_configurational_entropy = "none"   # default\n# solid_configurational_entropy = "ideal"\n# solid_configurational_entropy = "configurational"\n```\n\n`ideal` adds the binary occupied/vacant oxygen-site mixing entropy.\n`configurational` uses a canonical partition function over exact symmetry-distinct\nvacancy configurations and orbit degeneracies. It requires exact enumeration; if\na calculation was sampled, the analysis stops rather than guessing degeneracies.\nWhen the whole exact set is relaxed, relaxed energies enter the partition function;\notherwise the full exact single-point spectrum supplies a configurational correction\nto the relaxed static minimum. Both treatments affect finite-temperature outputs,\nincluding `vacancy_formation_free_energy.csv/json` and the T-pO2 stability map.\nDirect delta-mu intervals remain static-lattice quantities. Vibrational, zero-point,\nmagnetic, thermal-electronic, anharmonic, thermal-expansion and solid-pV\ncontributions are not part of this screening level.\n''',
    "example README entropy",
)

# ---------------------------------------------------------------------------
# GUI README
# ---------------------------------------------------------------------------
replace_once(
    "gui/README.md",
    '- Configure calculator-verified, global-calibrated, or chemistry-specific oxygen references for vacancy thermodynamics\n',
    '- Configure calculator-verified, global-calibrated, or chemistry-specific oxygen references for vacancy thermodynamics\n- Configure none, ideal-mixing, or exact-orbit partition-function vacancy entropy\n',
    "GUI README feature list",
)
replace_once(
    "gui/README.md",
    '''The T-pO2 map combines the calibrated 298 K oxygen enthalpy reference with NIST\nO2 gas enthalpy/entropy and pressure corrections. If\n`solid_configurational_entropy = "ideal"`, the ideal occupied/vacant oxygen-site\nmixing term is also applied in this T-dependent map; direct delta-mu plots remain\nstatic-lattice quantities.\n''',
    '''The T-pO2 map combines the calibrated 298 K oxygen enthalpy reference with NIST\nO2 gas enthalpy/entropy and pressure corrections. The Input Builder also exposes\n`solid_configurational_entropy = "none"`, `"ideal"`, or `"configurational"`.\nThe last option uses exact symmetry-orbit degeneracies and an explicit canonical\npartition function; it refuses sampled enumeration because those degeneracies are\nnot exact. Entropy-aware runs also write `vacancy_formation_free_energy.csv/json`.\nDirect delta-mu plots remain static-lattice quantities.\n''',
    "GUI README entropy paragraph",
)

# ---------------------------------------------------------------------------
# Methods documentation
# ---------------------------------------------------------------------------
replace_once(
    "docs/source/methods/vacancies.rst",
    '''The default solid treatment is static-lattice: solid free energies are\napproximated by 0 K relaxed ML energies. Solid vibrational, zero-point,\nthermal-electronic, magnetic, anharmonic, thermal-expansion, and solid-pV terms\nare not included. Configurational entropy is also omitted by default. If\n``solid_configurational_entropy = "ideal"`` is selected, an ideal occupied/\nvacant oxygen-site mixing entropy is included only in the explicitly\nT-dependent pressure map. The direct ``delta_mu_O`` stability intervals remain\nstatic-lattice quantities because they do not define a unique temperature. This\nis not a complete finite-temperature phase diagram.\n''',
    '''The default solid treatment is static-lattice: solid free energies are\napproximated by 0 K relaxed ML energies. Solid vibrational, zero-point,\nthermal-electronic, magnetic, anharmonic, thermal-expansion, and solid-pV terms\nare not included. Vacancy configurational entropy is optional:\n\n``solid_configurational_entropy = "none"``\n   Keep the static-lattice solid contribution.\n\n``solid_configurational_entropy = "ideal"``\n   Add ideal occupied/vacant oxygen-site mixing entropy.\n\n``solid_configurational_entropy = "configurational"``\n   Evaluate a canonical partition function over the exact symmetry-distinct\n   vacancy configurations and their orbit degeneracies. Exact enumeration is\n   required. If every exact orbit was relaxed, relaxed energies are used;\n   otherwise the complete exact single-point spectrum supplies a configurational\n   correction relative to its minimum, which is added to the relaxed static\n   minimum for that vacancy count. Sampled enumeration is rejected because its\n   degeneracies are not exact.\n\nThe finite-temperature vacancy formation free energy is then\n\n.. math::\n\n   \Delta G_{vac}(n,T,p) =\n   E_{min}(n)-E_{min}(0) + n\mu_O(T,p) + \Delta F_{config}(n,T).\n\nFor the explicit partition function,\n\n.. math::\n\n   \Delta F_{config}(n,T) =\n   -k_B T \ln\left[\sum_i g_i\n   \exp\left(-\frac{E_i-E_{min}}{k_B T}\right)\right].\n\nThe result is written for every vacancy count to\n``vacancy_formation_free_energy.csv/json``. The T-pO2 stability map minimizes\nthis finite-T quantity. Direct ``delta_mu_O`` intervals remain static-lattice\nquantities because they do not define a unique temperature. Dopant-configurational\nentropy and the other solid free-energy terms listed above remain outside this\nscreening level.\n''',
    "vacancy methods entropy theory",
)

replace_once(
    "docs/source/methods/oxygen_calibration.rst",
    '''Optional solid configurational entropy\n---------------------------------------\n\nThe default remains the static-lattice approximation:\n\n.. code-block:: toml\n\n   solid_configurational_entropy = "none"\n\nAn optional ideal occupied/vacant oxygen-site entropy is available:\n\n.. code-block:: toml\n\n   solid_configurational_entropy = "ideal"\n\nFor :math:`N_O` parent oxygen sites and :math:`N_v` vacancies,\n\n.. math::\n\n   S_{\\mathrm{config}} = -k_B N_O\n   \\left[x_v\\ln x_v + (1-x_v)\\ln(1-x_v)\\right],\n\nwith :math:`x_v=N_v/N_O`.\n\nThis term is applied only to the explicitly T-dependent ``T-pO2`` pressure map.\nThe exact ``delta_mu_O`` intervals and selected ``delta_mu_O`` points remain\nstatic-lattice quantities because they have no unique temperature.  Solid\nvibrational, zero-point, magnetic, thermal-electronic, anharmonic, thermal-\nexpansion and solid-pV terms remain neglected at this screening level.\n''',
    '''Optional solid configurational entropy\n---------------------------------------\n\nThe finite-temperature vacancy analysis supports three choices:\n\n.. code-block:: toml\n\n   solid_configurational_entropy = "none"\n   # or "ideal"\n   # or "configurational"\n\n``ideal`` uses the occupied/vacant oxygen-site mixing entropy\n\n.. math::\n\n   S_{\\mathrm{config}} = -k_B N_O\n   \\left[x_v\\ln x_v + (1-x_v)\\ln(1-x_v)\\right],\n\nwith :math:`x_v=N_v/N_O`.\n\n``configurational`` uses the actual symmetry-distinct vacancy configurations\ngenerated by the workflow. For exact orbit degeneracy :math:`g_i`,\n\n.. math::\n\n   Z_n(T) = \sum_i g_i\n   \exp\left[-\frac{E_i-E_{min}}{k_B T}\right],\n\n.. math::\n\n   \Delta F_{config}(n,T) = -k_B T \ln Z_n(T).\n\nExact enumeration is required. If all exact configurations were relaxed, their\nrelaxed energies are used. Otherwise the full exact single-point spectrum is\nused to obtain the configurational correction relative to its minimum, while\nthe static baseline remains the relaxed minimum. Sampled configurations are not\nassigned guessed degeneracies.\n\nThe finite-T output combines this solid term with the calibrated oxygen chemical\npotential and writes ``vacancy_formation_free_energy.csv/json`` for every vacancy\ncount, temperature and oxygen pressure. Direct ``delta_mu_O`` intervals remain\nstatic-lattice quantities. Solid vibrational, zero-point, magnetic, thermal-\nelectronic, anharmonic, thermal-expansion and solid-pV terms remain neglected.\n''',
    "oxygen calibration entropy docs",
)

# Add a concise vacancy input reference to the main input-file documentation.
replace_once(
    "docs/source/input_file.rst",
    '''---------------------------------------------------------------------\n\n[surface]\n---------\n''',
    '''---------------------------------------------------------------------\n\n[vacancies]\n-----------\n\nThe vacancy stage uses one flat ``[vacancies]`` table. Important controls are:\n\n- ``host_species``, ``host_oxidation_state``, ``vacancy_species`` and\n  ``vacancy_compensation_charge`` for the formal-charge search space.\n- ``oxidation_state_elements`` / ``oxidation_state_values`` for dopants.\n- ``enumeration_mode = "auto" | "exact" | "sample"`` and the exact/sampling\n  limits for symmetry-distinct vacancy generation.\n- ``backend``, ``model``, ``task`` and ``device`` for ML screening/relaxation.\n- ``topk_per_vacancy_count``, ``optimizer``, ``fmax`` and ``max_steps`` for\n  relaxation.\n- ``static_thermodynamic_analysis = true`` to compare different oxygen contents.\n- ``oxygen_reference_mode = "global" | "chemistry-specific" |\n  "reference_file" | "same_calculator" | "explicit" | "none"``.\n- ``oxygen_standard_state_mode = "nist_shomate" | "user_table" | "none"``\n  together with ``temperatures_K`` and the ``pO2`` grid.\n- ``solid_configurational_entropy = "none" | "ideal" | "configurational"``.\n  The ``configurational`` option evaluates an explicit canonical partition\n  function using exact symmetry-orbit degeneracies; it therefore requires exact\n  vacancy enumeration.\n\nFinite-temperature runs write ``vacancy_formation_free_energy.csv/json`` with\n:math:`\\Delta G_{vac}(T,p_{O_2})` for every vacancy count. See\n:doc:`methods/vacancies` and :doc:`methods/oxygen_calibration` for the equations,\ncalibration rules and approximations.\n\n---------------------------------------------------------------------\n\n[surface]\n---------\n''',
    "input-file vacancy reference",
)

# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------
replace_once(
    "CHANGELOG.md",
    '''## [Unreleased]\n### Added\n''',
    '''## [Unreleased]\n### Added\n- Finite-temperature vacancy formation free energies with optional ideal or exact-orbit configurational entropy; the explicit partition-function mode uses exact symmetry degeneracies and writes `vacancy_formation_free_energy.csv/json` while keeping direct delta-mu intervals static-lattice\n''',
    "changelog entry",
)

print("Configurational vacancy entropy patch applied successfully.")

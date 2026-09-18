from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected patch marker not found in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Core backend.
path = "src/dopingflow/oxidation.py"
replace_once(path, "import csv\nimport json\nimport logging\n", "import csv\nimport fnmatch\nimport json\nimport logging\n")
replace_once(
    path,
    "    dft_followup_methods: tuple[str, ...]\n    settings: dict[str, Any] = field(default_factory=dict)\n",
    "    dft_followup_methods: tuple[str, ...]\n    target_include: tuple[str, ...] = ()\n    settings: dict[str, Any] = field(default_factory=dict)\n",
)
helper_marker = "    return tuple(methods)\n\n\ndef parse_oxidation_config(\n"
helper_text = '''    return tuple(methods)\n\n\ndef _parse_target_include(value: Any) -> tuple[str, ...]:\n    """Parse optional exact/glob target selectors while preserving order."""\n    if value is None:\n        return ()\n    if isinstance(value, str):\n        raw = [item.strip() for item in value.split(",") if item.strip()]\n    elif isinstance(value, (list, tuple)):\n        raw = [str(item).strip() for item in value if str(item).strip()]\n    else:\n        raise ValueError("[oxidation].target_include must be an array or comma-separated string")\n    return tuple(dict.fromkeys(raw))\n\n\ndef _target_matches_include(target: StructureTarget, selectors: Sequence[str]) -> bool:\n    """Match target_id or safe_id; shell-style wildcards are supported."""\n    target_id = target.target_id.replace("\\\\", "/")\n    safe_id = target.safe_id\n    for selector in selectors:\n        normalized = str(selector).strip().replace("\\\\", "/")\n        if not normalized:\n            continue\n        safe_selector = normalized.replace("/", "__")\n        if fnmatch.fnmatchcase(target_id, normalized) or fnmatch.fnmatchcase(safe_id, safe_selector):\n            return True\n    return False\n\n\ndef parse_oxidation_config(\n'''
replace_once(path, helper_marker, helper_text)
replace_once(
    path,
    '    mapping_tolerance = float(section.get("mapping_tolerance", 1.2))\n    if mapping_tolerance <= 0:\n        raise ValueError("[oxidation].mapping_tolerance must be > 0")\n\n    followup = section.get("dft_followup", {}) or {}\n',
    '    mapping_tolerance = float(section.get("mapping_tolerance", 1.2))\n    if mapping_tolerance <= 0:\n        raise ValueError("[oxidation].mapping_tolerance must be > 0")\n\n    target_include = _parse_target_include(section.get("target_include"))\n\n    followup = section.get("dft_followup", {}) or {}\n',
)
replace_once(
    path,
    "        dft_followup_methods=followup_methods,\n        settings=settings,\n",
    "        dft_followup_methods=followup_methods,\n        target_include=target_include,\n        settings=settings,\n",
)
replace_once(
    path,
    "    targets.sort(key=lambda target: (target.parent_id, target.n_vacancies, target.target_id))\n    return targets, warnings\n",
    '''    targets.sort(key=lambda target: (target.parent_id, target.n_vacancies, target.target_id))\n\n    if cfg.target_include:\n        discovered = list(targets)\n        targets = [\n            target for target in discovered\n            if _target_matches_include(target, cfg.target_include)\n        ]\n        unmatched = [\n            selector for selector in cfg.target_include\n            if not any(_target_matches_include(target, (selector,)) for target in discovered)\n        ]\n        for selector in unmatched:\n            warnings.append(f"target_include selector matched no structure: {selector}")\n        if not targets:\n            sample = ", ".join(target.target_id for target in discovered[:8])\n            suffix = f" Available targets include: {sample}" if sample else ""\n            raise RuntimeError(\n                "[oxidation].target_include matched no discovered structures." + suffix\n            )\n\n    return targets, warnings\n''',
)

# Streamlit GUI.
gui = "gui/pages/Oxidation_States.py"
replace_once(
    gui,
    '''source_root = st.text_input(\n    "Source root",\n    value=str(oxidation.get("source_root", source_default)),\n    help="Root containing selected relaxed parents and, when enabled, vacancies_database.json.",\n)\n\nst.info(\n''',
    '''source_root = st.text_input(\n    "Source root",\n    value=str(oxidation.get("source_root", source_default)),\n    help="Root containing selected relaxed parents and, when enabled, vacancies_database.json.",\n)\n\nsaved_target_include = oxidation.get("target_include", [])\nif isinstance(saved_target_include, str):\n    saved_target_include = [\n        item.strip() for item in saved_target_include.split(",") if item.strip()\n    ]\nelif not isinstance(saved_target_include, (list, tuple)):\n    saved_target_include = []\ntarget_include_text = st.text_input(\n    "Target selector(s) (optional)",\n    value=", ".join(str(item) for item in saved_target_include),\n    help=(\n        "Leave empty to analyze every discovered target. Use exact target IDs such as "\n        "Sb5_Ti2p5/candidate_014, safe IDs such as Sb5_Ti2p5__candidate_014, or shell-style "\n        "wildcards such as Sb5_Ti2p5/*. Multiple selectors can be comma-separated."\n    ),\n)\ntarget_include = list(\n    dict.fromkeys(\n        item.strip() for item in target_include_text.split(",") if item.strip()\n    )\n)\nif target_include:\n    st.info(\n        f"Target filtering is active: only structures matching {target_include} will be analyzed."\n    )\n\nst.info(\n''',
)
replace_once(
    gui,
    '        "include_oxygen_vacancies": bool(include_oxygen_vacancies),\n        "source_root": source_root,\n',
    '        "include_oxygen_vacancies": bool(include_oxygen_vacancies),\n        "target_include": target_include,\n        "source_root": source_root,\n',
)
replace_once(
    gui,
    '''contains_dft_execution = any(\n    bool((_table(name) if name in oxidation else {}).get("execute", False))\n    for name in ("dft_electronic", "bader", "wannier", "eos")\n) or bool(followup.get("execute", False))\n\nconfirm_dft = True\n''',
    '''contains_dft_execution = any(\n    bool((_table(name) if name in oxidation else {}).get("execute", False))\n    for name in ("dft_electronic", "bader", "wannier", "eos")\n) or bool(followup.get("execute", False))\n\nif contains_dft_execution and not target_include:\n    st.warning(\n        "DFT execution is enabled but no target selector is active. The calculation can run for every "\n        "discovered structure. For an expensive smoke test, select one exact target first."\n    )\n\nconfirm_dft = True\n''',
)

# Tests.
tests = Path("tests/test_oxidation.py")
text = tests.read_text(encoding="utf-8")
test_marker = "\n\ndef test_composition_level_predictions_never_become_site_assignments(tmp_path: Path) -> None:\n"
if test_marker not in text:
    raise SystemExit("Could not find oxidation test insertion marker")
target_tests = '''\n\ndef test_target_include_filters_exact_safe_and_glob_targets(tmp_path: Path) -> None:\n    source_root = tmp_path / "random_structures"\n    _write_parent_tree(source_root)\n    parent = _structure()\n    vacancy = parent.copy()\n    vacancy.remove_sites([3])\n    vacancy_path = source_root / "vacancy_relaxed" / "POSCAR"\n    vacancy_path.parent.mkdir(parents=True)\n    Poscar(vacancy).write_file(vacancy_path)\n    (source_root / "vacancies_database.json").write_text(\n        json.dumps(\n            [\n                {\n                    "parent_id": "Sb25/candidate_0001",\n                    "configuration_id": "config_0001",\n                    "vacancy_species": "O",\n                    "n_vacancies": 1,\n                    "relaxed_poscar_path": str(vacancy_path),\n                }\n            ]\n        ),\n        encoding="utf-8",\n    )\n\n    def make_cfg(selector):\n        return parse_oxidation_config(\n            {\n                "structure": {"outdir": str(source_root)},\n                "oxidation": {\n                    "strategy": "structural",\n                    "methods": ["bond-valence"],\n                    "target_include": selector,\n                },\n            },\n            tmp_path,\n        )\n\n    exact = make_cfg(["Sb25/candidate_0001"])\n    assert exact.target_include == ("Sb25/candidate_0001",)\n    targets, _ = discover_oxidation_targets(exact)\n    assert [target.target_id for target in targets] == ["Sb25/candidate_0001"]\n\n    safe = make_cfg("Sb25__candidate_0001")\n    targets, _ = discover_oxidation_targets(safe)\n    assert [target.target_id for target in targets] == ["Sb25/candidate_0001"]\n\n    vacancy_only = make_cfg(["Sb25/candidate_0001/V_O_01/config_0001"])\n    targets, _ = discover_oxidation_targets(vacancy_only)\n    assert [target.target_id for target in targets] == [\n        "Sb25/candidate_0001/V_O_01/config_0001"\n    ]\n\n    wildcard = make_cfg(["Sb25/candidate_0001/*"])\n    targets, _ = discover_oxidation_targets(wildcard)\n    assert [target.target_id for target in targets] == [\n        "Sb25/candidate_0001/V_O_01/config_0001"\n    ]\n\n    missing = make_cfg(["does-not-exist"])\n    with pytest.raises(RuntimeError, match="target_include matched no discovered structures"):\n        discover_oxidation_targets(missing)\n'''
tests.write_text(text.replace(test_marker, target_tests + test_marker, 1), encoding="utf-8")

# Documentation.
readme = "README.md"
replace_once(
    readme,
    '''include_vacancy_free = true\ninclude_oxygen_vacancies = true\noutput_dir = "06_oxidation"\n''',
    '''include_vacancy_free = true\ninclude_oxygen_vacancies = true\n# Optional: restrict expensive analysis to exact target IDs / safe IDs / glob patterns.\n# target_include = ["Sb5_Ti2p5/candidate_014"]\noutput_dir = "06_oxidation"\n''',
)
replace_once(
    readme,
    '''The stage can analyze both selected vacancy-free parents and relaxed oxygen\nvacancy structures. When valid same-element parent mapping is available it also\nreports parent-relative changes.\n''',
    '''The stage can analyze both selected vacancy-free parents and relaxed oxygen\nvacancy structures. `[oxidation].target_include` can restrict a run to one or more\nexact target IDs, safe IDs (`/` represented as `__`), or shell-style glob patterns.\nThis is especially useful before enabling expensive GPAW, Bader, Wannier, or EOS\nexecution. When valid same-element parent mapping is available it also reports\nparent-relative changes.\n''',
)
replace_once(
    readme,
    'conda install -c conda-forge gpaw gpaw-data\npip install -e ".[gui]"\ngpaw info\npython -m streamlit run gui/app.py\n',
    'conda install -c conda-forge gpaw gpaw-data wannier90\npip install -e ".[gui]"\ngpaw info\ncommand -v wannier90.x\npython -m streamlit run gui/app.py\n',
)

docs = "docs/source/methods/oxidation_states.rst"
replace_once(
    docs,
    '''   include_vacancy_free = true\n   include_oxygen_vacancies = true\n   output_dir = "06_oxidation"\n''',
    '''   include_vacancy_free = true\n   include_oxygen_vacancies = true\n   # Optional exact IDs, safe IDs, or shell-style glob patterns:\n   # target_include = ["Sb5_Ti2p5/candidate_014"]\n   output_dir = "06_oxidation"\n''',
)
replace_once(
    docs,
    '''``include_vacancy_free`` and ``include_oxygen_vacancies`` default to ``true``.\nThe normal use for vacancy chemistry is to leave both enabled so parent-relative\nchanges can be reported when atom mapping is valid.\n''',
    '''``include_vacancy_free`` and ``include_oxygen_vacancies`` default to ``true``.\nThe normal use for vacancy chemistry is to leave both enabled so parent-relative\nchanges can be reported when atom mapping is valid.  ``target_include`` is optional;\nwhen provided, only matching discovered targets are analyzed.  A selector can be an\nexact target ID (for example ``Sb5_Ti2p5/candidate_014``), its safe-ID form\n(``Sb5_Ti2p5__candidate_014``), or a shell-style glob such as\n``Sb5_Ti2p5/*``.  This is the recommended way to constrain expensive DFT smoke\ntests to one structure before scaling up.\n''',
)

# One-shot helper files are removed from the final feature commit.
Path(".github/patch_target_filter.py").unlink()
Path(".github/workflows/add-oxidation-target-filter.yml").unlink()

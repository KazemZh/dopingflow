from pathlib import Path
import re


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"patch anchor not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Core registry: structural now means bond valence only. TOSS-GNN remains ML.
path = Path("src/dopingflow/oxidation.py")
replace_once(
    path,
    '    "structural": ("bond-valence", "toss-bayesian"),\n',
    '    "structural": ("bond-valence",),\n',
)
text = path.read_text(encoding="utf-8")
text = text.replace('    "toss": "toss-bayesian",\n', '')
text = text.replace('    "toss_bayesian": "toss-bayesian",\n', '')
path.write_text(text, encoding="utf-8")

# Structural adapter: remove all conventional Bayesian TOSS integration.
Path("src/dopingflow/oxidation_structural.py").write_text(
    '''from __future__ import annotations\n\nfrom typing import Any\n\nfrom pymatgen.core import Structure\n\nfrom dopingflow.oxidation import (\n    OptionalMethodUnavailable,\n    OxidationConfig,\n    StructureTarget,\n    base_method_result,\n    site_records,\n)\n\n\ndef _run_bond_valence(target: StructureTarget, settings: dict[str, Any]) -> dict[str, Any]:\n    try:\n        from pymatgen.analysis.bond_valence import BVAnalyzer\n    except ImportError as exc:  # pragma: no cover - pymatgen is core dependency\n        raise OptionalMethodUnavailable("pymatgen bond-valence analyzer is unavailable") from exc\n\n    structure = Structure.from_file(target.structure_path)\n    kwargs: dict[str, Any] = {}\n    if "symm_tol" in settings:\n        kwargs["symm_tol"] = float(settings["symm_tol"])\n    if "max_radius" in settings:\n        kwargs["max_radius"] = float(settings["max_radius"])\n    analyzer = BVAnalyzer(**kwargs)\n    try:\n        valences = analyzer.get_valences(structure)\n    except ValueError as exc:\n        if "Valences cannot be assigned" not in str(exc):\n            raise\n        return base_method_result(\n            method="bond-valence",\n            target=target,\n            scope="site-resolved",\n            status="unsupported",\n            provenance={\n                "implementation": "pymatgen.analysis.bond_valence.BVAnalyzer",\n                "settings": kwargs,\n            },\n            limitations=[\n                "pymatgen BVAnalyzer could not find a chemically consistent valence "\n                "assignment for this structure. This is a method limitation, not a "\n                "software or installation failure."\n            ],\n            error="Valences cannot be assigned by pymatgen BVAnalyzer",\n        )\n    formal = site_records(structure, [float(value) for value in valences])\n    return base_method_result(\n        method="bond-valence",\n        target=target,\n        scope="site-resolved",\n        status="assigned",\n        formal_oxidation_states=formal,\n        provenance={\n            "implementation": "pymatgen.analysis.bond_valence.BVAnalyzer",\n            "settings": kwargs,\n        },\n        limitations=[\n            "Bond-valence assignments are structure/parameter dependent and should be treated "\n            "cautiously for strongly distorted, defective, or unusual coordination environments."\n        ],\n    )\n\n\ndef run_structural_method(\n    method: str,\n    target: StructureTarget,\n    cfg: OxidationConfig,\n    settings: dict[str, Any],\n) -> dict[str, Any]:\n    del cfg\n    if method == "bond-valence":\n        return _run_bond_valence(target, settings)\n    raise ValueError(f"Unknown structural oxidation method: {method}")\n''',
    encoding="utf-8",
)

# GUI: remove option and its configuration panel; discard stale table on save.
path = Path("gui/pages/Oxidation_States.py")
replace_once(
    path,
    '    "structural": ["bond-valence", "toss-bayesian"],\n',
    '    "structural": ["bond-valence"],\n',
)
text = path.read_text(encoding="utf-8")
text = text.replace('    "toss-bayesian",\n', '')
old_block = '''if "toss-bayesian" in methods:\n    with st.expander("Conventional TOSS Bayesian/MAP", expanded=False):\n        toss = _table("toss_bayesian")\n        toss["repo_path"] = st.text_input(\n            "Local TOSS repository",\n            value=str(toss.get("repo_path", "")),\n            key="oxidation_toss_bayesian_repo",\n        )\n        st.caption("This is the conventional TOSS assignment, separate from the pretrained TOSS-GNN model.")\n        oxidation["toss_bayesian"] = toss\n\n'''
if old_block not in text:
    raise SystemExit("GUI conventional TOSS panel anchor not found")
text = text.replace(old_block, '', 1)
anchor = 'oxidation = dict(cfg.get("oxidation", {}) or {})\n'
if anchor not in text:
    raise SystemExit("GUI oxidation table anchor not found")
text = text.replace(
    anchor,
    anchor + '# Retired conventional Bayesian TOSS settings are ignored and removed on save.\noxidation.pop("toss_bayesian", None)\n',
    1,
)
path.write_text(text, encoding="utf-8")

# README: describe only the public, reproducible methods that remain supported.
path = Path("README.md")
text = path.read_text(encoding="utf-8")
text = text.replace(
    'pip install -e ".[oxidation-toss]"    # conventional TOSS / TOSS-GNN Python deps',
    'pip install -e ".[oxidation-toss]"    # TOSS-GNN Python deps',
)
text = text.replace(
    '- **Structural:** pymatgen bond valence and conventional Bayesian/MAP TOSS.',
    '- **Structural:** pymatgen bond valence.',
)
path.write_text(text, encoding="utf-8")

# User guide: structural strategy is bond valence only.
path = Path("docs/source/methods/oxidation_states.rst")
text = path.read_text(encoding="utf-8")
text = text.replace(
    '``structural``\n   ``bond-valence`` (pymatgen ``BVAnalyzer``) and ``toss-bayesian``\n   (conventional MAP/Bayesian TOSS).',
    '``structural``\n   ``bond-valence`` (pymatgen ``BVAnalyzer``).',
)
path.write_text(text, encoding="utf-8")

# Regression: old configs must fail clearly rather than silently map to another TOSS method.
path = Path("tests/test_oxidation.py")
text = path.read_text(encoding="utf-8")
anchor = '    assert ml.methods == ("toss-gnn",)\n\n'
insert = '''    assert ml.methods == ("toss-gnn",)\n\n    with pytest.raises(ValueError, match="Unknown oxidation-state method"):\n        parse_oxidation_config(\n            {"oxidation": {"strategy": "structural", "methods": ["toss-bayesian"]}},\n            tmp_path,\n        )\n\n'''
if anchor not in text:
    raise SystemExit("test insertion anchor not found")
path.write_text(text.replace(anchor, insert, 1), encoding="utf-8")

# Ensure the retired method/config key is gone everywhere except the intentional rejection test.
allowed = {Path("tests/test_oxidation.py")}
problems = []
for root in (Path("src"), Path("gui"), Path("docs"), Path("README.md")):
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    for file in files:
        try:
            data = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in ("toss-bayesian", "toss_bayesian", "Bayesian/MAP TOSS", "MAP/Bayesian TOSS", "conventional Bayesian TOSS", "conventional TOSS"):
            if needle.lower() in data.lower():
                problems.append((str(file), needle))
if problems:
    raise SystemExit(f"retired conventional TOSS references remain: {problems}")

print("Conventional Bayesian TOSS removed; TOSS-GNN retained.")

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from dopingflow.oxidation import (
    OptionalMethodUnavailable,
    OxidationConfig,
    StructureTarget,
    base_method_result,
    site_records,
)

TOSS_VERIFIED_COMMIT = "c45582a3cd3088480b5d83b1440d360bd4577b80"


@contextlib.contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    text = str(path)
    sys.path.insert(0, text)
    try:
        yield
    finally:
        try:
            sys.path.remove(text)
        except ValueError:
            pass


def _resolve_toss_repo(settings: dict[str, Any]) -> Path:
    repo_value = str(settings.get("repo_path") or "").strip()
    if not repo_value:
        raise OptionalMethodUnavailable(
            "Conventional Bayesian TOSS requires [oxidation.toss_bayesian].repo_path "
            "pointing to a checkout of https://github.com/yueyin19960520/TOSS."
        )
    repo = Path(repo_value).expanduser().resolve()
    required = [repo / "toss" / "Get_Initial_Guess.py", repo / "toss" / "Get_TOS.py"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise OptionalMethodUnavailable(
            f"TOSS checkout is incomplete; missing: {', '.join(str(path) for path in missing)}"
        )
    return repo


def _run_bond_valence(target: StructureTarget, settings: dict[str, Any]) -> dict[str, Any]:
    try:
        from pymatgen.analysis.bond_valence import BVAnalyzer
    except ImportError as exc:  # pragma: no cover - pymatgen is core dependency
        raise OptionalMethodUnavailable("pymatgen bond-valence analyzer is unavailable") from exc

    structure = Structure.from_file(target.structure_path)
    kwargs: dict[str, Any] = {}
    if "symm_tol" in settings:
        kwargs["symm_tol"] = float(settings["symm_tol"])
    if "max_radius" in settings:
        kwargs["max_radius"] = float(settings["max_radius"])
    analyzer = BVAnalyzer(**kwargs)
    try:
        valences = analyzer.get_valences(structure)
    except ValueError as exc:
        if "Valences cannot be assigned" not in str(exc):
            raise
        return base_method_result(
            method="bond-valence",
            target=target,
            scope="site-resolved",
            status="unsupported",
            provenance={
                "implementation": "pymatgen.analysis.bond_valence.BVAnalyzer",
                "settings": kwargs,
            },
            limitations=[
                "pymatgen BVAnalyzer could not find a chemically consistent valence "
                "assignment for this structure. This is a method limitation, not a "
                "software or installation failure."
            ],
            error="Valences cannot be assigned by pymatgen BVAnalyzer",
        )
    formal = site_records(structure, [float(value) for value in valences])
    return base_method_result(
        method="bond-valence",
        target=target,
        scope="site-resolved",
        status="assigned",
        formal_oxidation_states=formal,
        provenance={
            "implementation": "pymatgen.analysis.bond_valence.BVAnalyzer",
            "settings": kwargs,
        },
        limitations=[
            "Bond-valence assignments are structure/parameter dependent and should be treated "
            "cautiously for strongly distorted, defective, or unusual coordination environments."
        ],
    )


def _run_toss_bayesian(target: StructureTarget, settings: dict[str, Any]) -> dict[str, Any]:
    repo = _resolve_toss_repo(settings)
    toss_dir = repo / "toss"
    structure = Structure.from_file(target.structure_path)

    try:
        with _prepend_sys_path(toss_dir):
            guess_module = importlib.import_module("Get_Initial_Guess")
            tos_module = importlib.import_module("Get_TOS")
    except Exception as exc:
        raise OptionalMethodUnavailable(
            "Could not import conventional TOSS dependencies. Install the upstream TOSS requirements "
            f"in this environment. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="dopingflow-toss-") as tmpdir:
        cif_path = Path(tmpdir) / "structure.cif"
        CifWriter(structure).write_file(cif_path)
        # Upstream TOSS's server path helper prepends './' internally. Give it
        # a path relative to cwd so an absolute '/tmp/...' path is not turned
        # into the unintended relative './/tmp/...'.
        toss_filepath = os.path.relpath(cif_path, Path.cwd())
        try:
            valid_response = guess_module.get_the_valid_t(
                m_id=cif_path.name,
                server=True,
                filepath=toss_filepath,
            )
            valid_t = valid_response[-1] if isinstance(valid_response, tuple) else valid_response
            if not valid_t:
                return base_method_result(
                    method="toss-bayesian",
                    target=target,
                    scope="site-resolved",
                    status="unsupported",
                    provenance={
                        "upstream": "yueyin19960520/TOSS",
                        "verified_interface_commit": TOSS_VERIFIED_COMMIT,
                        "repo_path": str(repo),
                    },
                    limitations=[
                        "Conventional TOSS found no valid coordination tolerance for this structure."
                    ],
                )
            outputs = tos_module.get_Oxidation_States(
                m_id=cif_path.name,
                server=True,
                filepath=toss_filepath,
                input_tolerance_list=valid_t,
            )
            result = outputs[-1]
        except Exception as exc:
            raise RuntimeError(f"Conventional TOSS inference failed: {exc}") from exc

    elements = list(getattr(result, "elements_list", []))
    values = list(getattr(result, "sum_of_valence", []))
    coordination_values = list(getattr(result, "shell_CN_list", []))
    if len(values) != len(structure):
        raise RuntimeError(
            f"TOSS returned {len(values)} site assignments for a {len(structure)}-site structure"
        )
    if elements and elements != [site.specie.symbol for site in structure]:
        raise RuntimeError("TOSS output site order/elements do not match the input structure")

    formal = site_records(structure, values)
    coordination = [
        {
            "site_index": index,
            "element": structure[index].specie.symbol,
            "coordination_number": value,
        }
        for index, value in enumerate(coordination_values)
    ]
    return base_method_result(
        method="toss-bayesian",
        target=target,
        scope="site-resolved",
        status="assigned",
        formal_oxidation_states=formal,
        coordination=coordination,
        provenance={
            "upstream": "yueyin19960520/TOSS",
            "interface": "get_the_valid_t + get_Oxidation_States (MAP/Bayesian TOSS)",
            "verified_interface_commit": TOSS_VERIFIED_COMMIT,
            "repo_path": str(repo),
            "valid_tolerances": list(valid_t),
        },
        limitations=[
            "This is conventional Bayesian/MAP TOSS, not the pretrained TOSS-GNN predictor.",
            "Applicability to defective co-doped oxides should be assessed for the target chemistry "
            "rather than inferred from broad-database performance.",
        ],
    )


def run_structural_method(
    method: str,
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    del cfg
    if method == "bond-valence":
        return _run_bond_valence(target, settings)
    if method == "toss-bayesian":
        return _run_toss_bayesian(target, settings)
    raise ValueError(f"Unknown structural oxidation method: {method}")

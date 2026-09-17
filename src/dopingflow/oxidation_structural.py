from __future__ import annotations

from typing import Any

from pymatgen.core import Structure

from dopingflow.oxidation import (
    OptionalMethodUnavailable,
    OxidationConfig,
    StructureTarget,
    base_method_result,
    site_records,
)


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


def run_structural_method(
    method: str,
    target: StructureTarget,
    cfg: OxidationConfig,
    settings: dict[str, Any],
) -> dict[str, Any]:
    del cfg
    if method == "bond-valence":
        return _run_bond_valence(target, settings)
    raise ValueError(f"Unknown structural oxidation method: {method}")

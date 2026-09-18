from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected marker not found in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_between(path: str, start_marker: str, end_marker: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    p.write_text(text[:start] + new + text[end:], encoding="utf-8")


module = r'''from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pymatgen.core import Structure


_SPREAD_RE = re.compile(
    r"^\s*WF centre and spread\s+(\d+)\s+\(.*\)\s+([-+0-9.eE]+)\s*$"
)


def parse_wannier_spreads(path: Path) -> dict[int, float]:
    """Return final Wannier spreads keyed by zero-based WF index.

    Wannier90 prints the same WF repeatedly during localization.  Assignment in
    this loop deliberately overwrites earlier iterations, leaving the final
    occurrence from the converged/final summary block.
    """
    spreads: dict[int, float] = {}
    if not path.is_file():
        return spreads
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _SPREAD_RE.match(line)
            if not match:
                continue
            try:
                spreads[int(match.group(1)) - 1] = float(match.group(2))
            except ValueError:
                continue
    return spreads


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def analyze_wannier_centres(
    structure: Structure,
    centres: Sequence[dict[str, Any]],
    spreads: dict[int, float] | None = None,
    *,
    atom_center_cutoff_angstrom: float = 0.45,
    bond_center_cutoff_angstrom: float = 1.35,
    bond_distance_balance_angstrom: float = 0.30,
    delocalized_spread_threshold_ang2: float = 3.0,
    electrons_per_wf: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Classify Wannier centers geometrically using periodic minimum distances.

    The classification is intentionally a descriptor, not an oxidation-state
    assignment.  The thresholds are exposed to the user because their useful
    values are chemistry and localization dependent.
    """
    if len(structure) == 0:
        raise ValueError("Wannier-center analysis requires a non-empty structure")
    if atom_center_cutoff_angstrom <= 0:
        raise ValueError("atom_center_cutoff_angstrom must be positive")
    if bond_center_cutoff_angstrom <= 0:
        raise ValueError("bond_center_cutoff_angstrom must be positive")
    if atom_center_cutoff_angstrom > bond_center_cutoff_angstrom:
        raise ValueError("atom_center_cutoff_angstrom cannot exceed bond_center_cutoff_angstrom")
    if bond_distance_balance_angstrom < 0:
        raise ValueError("bond_distance_balance_angstrom must be >= 0")
    if delocalized_spread_threshold_ang2 <= 0:
        raise ValueError("delocalized_spread_threshold_ang2 must be positive")
    if electrons_per_wf is not None and electrons_per_wf <= 0:
        raise ValueError("electrons_per_wf must be positive when supplied")

    spreads = dict(spreads or {})
    enriched: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    geometric_counts: Counter[str] = Counter()

    site_summary: list[dict[str, Any]] = [
        {
            "site_index": index,
            "element": site.specie.symbol,
            "nearest_wf_count": 0,
            "nearest_wf_electron_equivalent": 0.0 if electrons_per_wf is not None else None,
            "atom_centered_wf_count": 0,
            "bond_centered_wf_participation": 0,
            "ambiguous_wf_participation": 0,
            "delocalized_wf_nearest_count": 0,
        }
        for index, site in enumerate(structure)
    ]

    for raw in centres:
        record = dict(raw)
        center_index = int(record.get("center_index", len(enriched)))
        cart = record.get("cartesian_angstrom")
        if not isinstance(cart, (list, tuple)) or len(cart) != 3:
            raise ValueError(f"Wannier center {center_index} has invalid Cartesian coordinates")
        cart_values = [float(value) for value in cart]
        frac = structure.lattice.get_fractional_coords(cart_values)
        distances = structure.lattice.get_all_distances([frac], structure.frac_coords)[0]
        order = sorted(range(len(structure)), key=lambda idx: float(distances[idx]))
        nearest = int(order[0])
        second = int(order[1]) if len(order) > 1 else None
        d1 = float(distances[nearest])
        d2 = float(distances[second]) if second is not None else None
        within = [
            int(idx)
            for idx in order
            if float(distances[idx]) <= bond_center_cutoff_angstrom
        ]

        if d1 <= atom_center_cutoff_angstrom:
            geometric = "atom-centered"
            associated = [nearest]
        elif (
            second is not None
            and d2 is not None
            and d2 <= bond_center_cutoff_angstrom
            and abs(d2 - d1) <= bond_distance_balance_angstrom
        ):
            geometric = "bond-centered"
            associated = [nearest, second]
        else:
            geometric = "multicenter/ambiguous"
            associated = within or [nearest]

        spread = spreads.get(center_index)
        delocalized = bool(
            spread is not None and spread >= delocalized_spread_threshold_ang2
        )
        classification = "anomalous/delocalized" if delocalized else geometric
        classification_counts[classification] += 1
        geometric_counts[geometric] += 1

        record.update(
            {
                "center_index": center_index,
                "cartesian_angstrom": cart_values,
                "spread_ang2": float(spread) if spread is not None else None,
                "classification": classification,
                "geometric_classification": geometric,
                "is_delocalized_spread_outlier": delocalized,
                "nearest_site_index": nearest,
                "nearest_element": structure[nearest].specie.symbol,
                "nearest_distance_angstrom": d1,
                "second_nearest_site_index": second,
                "second_nearest_element": (
                    structure[second].specie.symbol if second is not None else None
                ),
                "second_nearest_distance_angstrom": d2,
                "associated_site_indices": associated,
                "associated_elements": [structure[idx].specie.symbol for idx in associated],
                "n_atoms_within_bond_cutoff": len(within),
                "electron_equivalent": electrons_per_wf,
            }
        )
        enriched.append(record)

        site_summary[nearest]["nearest_wf_count"] += 1
        if electrons_per_wf is not None:
            site_summary[nearest]["nearest_wf_electron_equivalent"] += electrons_per_wf
        if geometric == "atom-centered":
            site_summary[nearest]["atom_centered_wf_count"] += 1
        elif geometric == "bond-centered":
            for idx in associated:
                site_summary[idx]["bond_centered_wf_participation"] += 1
        else:
            for idx in associated:
                site_summary[idx]["ambiguous_wf_participation"] += 1
        if delocalized:
            site_summary[nearest]["delocalized_wf_nearest_count"] += 1

    spread_values = [
        float(record["spread_ang2"])
        for record in enriched
        if record.get("spread_ang2") is not None
    ]
    max_record = max(
        (record for record in enriched if record.get("spread_ang2") is not None),
        key=lambda record: float(record["spread_ang2"]),
        default=None,
    )
    summary = {
        "n_wannier_centres": len(enriched),
        "n_spread_records": len(spread_values),
        "median_spread_ang2": _median(spread_values),
        "max_spread_ang2": (
            float(max_record["spread_ang2"]) if max_record is not None else None
        ),
        "max_spread_center_index": (
            int(max_record["center_index"]) if max_record is not None else None
        ),
        "classification_counts": dict(sorted(classification_counts.items())),
        "geometric_classification_counts": dict(sorted(geometric_counts.items())),
        "n_delocalized_spread_outliers": int(
            classification_counts.get("anomalous/delocalized", 0)
        ),
        "electrons_per_wf": electrons_per_wf,
        "represented_electrons": (
            float(electrons_per_wf * len(enriched))
            if electrons_per_wf is not None
            else None
        ),
        "thresholds": {
            "atom_center_cutoff_angstrom": atom_center_cutoff_angstrom,
            "bond_center_cutoff_angstrom": bond_center_cutoff_angstrom,
            "bond_distance_balance_angstrom": bond_distance_balance_angstrom,
            "delocalized_spread_threshold_ang2": delocalized_spread_threshold_ang2,
        },
        "interpretation": (
            "Center/site associations are periodic geometric descriptors. Nearest-center electron "
            "equivalents are not Bader charges, atomic populations, or formal oxidation states."
        ),
    }
    return enriched, site_summary, summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def write_wannier_analysis_files(
    workdir: Path,
    centres: Sequence[dict[str, Any]],
    site_summary: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    centres_csv = workdir / "wannier_centres_analysis.csv"
    sites_csv = workdir / "wannier_site_summary.csv"
    analysis_json = workdir / "wannier_analysis.json"
    _write_csv(centres_csv, centres)
    _write_csv(sites_csv, site_summary)
    analysis_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "centres": list(centres),
                "site_summary": list(site_summary),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "centres_analysis_csv": str(centres_csv),
        "site_summary_csv": str(sites_csv),
        "analysis_json": str(analysis_json),
    }
'''
Path("src/dopingflow/wannier_analysis.py").write_text(module, encoding="utf-8")

# Wire the analysis into the DFT Wannier adapter.
p = Path("src/dopingflow/oxidation_dft.py")
text = p.read_text(encoding="utf-8")
import_marker = "from pymatgen.io.ase import AseAtomsAdaptor\n\n"
if "from dopingflow.wannier_analysis import" not in text:
    text = text.replace(
        import_marker,
        import_marker
        + "from dopingflow.wannier_analysis import (\n"
        + "    analyze_wannier_centres,\n"
        + "    parse_wannier_spreads,\n"
        + "    write_wannier_analysis_files,\n"
        + ")\n\n",
        1,
    )
text = text.replace(
    '        "n_spins": nspins,\n',
    '        "n_spins": nspins,\n        "n_electrons": float(calc.get_number_of_electrons()),\n',
    1,
)
start = text.index("def _run_wannier(\n")
end = text.index("\ndef _run_eos(\n", start)
new_run = r'''def _run_wannier(
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

    metadata_file = workdir / "wannier_run_metadata.json"
    if not metadata and metadata_file.is_file():
        try:
            loaded = json.loads(metadata_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        except Exception:
            metadata = {}

    centres = _parse_wannier_centres(centers_file)
    seed = str(metadata.get("seed") or settings.get("seed", "wannier90")).strip() or "wannier90"
    wout_file = workdir / str(settings.get("wout_file", f"{seed}.wout"))
    spreads = parse_wannier_spreads(wout_file)
    structure = Structure.from_file(target.structure_path)

    paired_native = (
        str(metadata.get("mode") or "") == "occupied-bloch"
        and int(metadata.get("n_spins") or 0) == 1
    )
    electrons_per_wf = 2.0 if paired_native else None
    enriched, site_summary, analysis = analyze_wannier_centres(
        structure,
        centres,
        spreads,
        atom_center_cutoff_angstrom=float(settings.get("atom_center_cutoff_angstrom", 0.45)),
        bond_center_cutoff_angstrom=float(settings.get("bond_center_cutoff_angstrom", 1.35)),
        bond_distance_balance_angstrom=float(settings.get("bond_distance_balance_angstrom", 0.30)),
        delocalized_spread_threshold_ang2=float(
            settings.get("delocalized_spread_threshold_ang2", 3.0)
        ),
        electrons_per_wf=electrons_per_wf,
    )

    expected_centres = metadata.get("n_occupied_bands")
    if expected_centres is not None:
        expected_centres = int(expected_centres)
    analysis.update(
        {
            "target_id": target.target_id,
            "structure_kind": target.kind,
            "n_atoms": len(structure),
            "wout_file": str(wout_file) if wout_file.is_file() else None,
            "expected_centres_from_occupied_bands": expected_centres,
            "center_count_consistent_with_occupied_bands": (
                len(enriched) == expected_centres if expected_centres is not None else None
            ),
            "gpaw_electrons": (
                float(metadata["n_electrons"])
                if metadata.get("n_electrons") is not None
                else None
            ),
            "electron_count_consistent_with_gpaw": None,
            "parent_relative_comparison": (
                "not-applicable-vacancy-free"
                if target.kind == "vacancy-free"
                else "available-only-when-matched-parent-wannier-result-is-analyzed"
            ),
        }
    )
    if analysis.get("represented_electrons") is not None and analysis.get("gpaw_electrons") is not None:
        analysis["electron_count_consistent_with_gpaw"] = math.isclose(
            float(analysis["represented_electrons"]),
            float(analysis["gpaw_electrons"]),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )

    analysis_paths = write_wannier_analysis_files(
        workdir,
        enriched,
        site_summary,
        analysis,
    )
    analysis["output_files"] = analysis_paths

    provenance = {
        "workdir": str(workdir),
        "centres_file": str(centers_file),
        "wout_file": str(wout_file) if wout_file.is_file() else None,
        "execute": bool(settings.get("execute", False)),
        "execution_mode": execution_mode,
        "n_wannier_centres": len(enriched),
        "analysis_files": analysis_paths,
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
            "Atom-/bond-/multicenter labels are configurable periodic geometric classifications, not formal charges or oxidation states.",
            "Nearest-center electron equivalents are bookkeeping descriptors for paired occupied Wannier functions and must not be interpreted as atomic electron populations.",
            "The native occupied-bloch mode currently supports isolated non-spin-polarized Gamma-only occupied manifolds; metallic, spin-polarized, multi-k, or entangled cases require an explicit projection/disentanglement workflow.",
        ],
    )
    result["wannier_descriptors"] = enriched
    result["wannier_site_summary"] = site_summary
    result["wannier_analysis"] = analysis
    return result

'''
text = text[:start] + new_run + text[end:]
p.write_text(text, encoding="utf-8")

# Parent-relative Wannier deltas are optional and only appear when both parent
# and vacancy structures were actually analyzed with Wannier.
p = Path("src/dopingflow/oxidation.py")
text = p.read_text(encoding="utf-8")
start = text.index("def _add_parent_relative_changes(\n")
end = text.index("\ndef _flatten_site_rows", start)
new_parent = r'''def _add_parent_relative_changes(
    results: list[dict[str, Any]],
    targets: Sequence[StructureTarget],
    cfg: OxidationConfig,
) -> None:
    target_by_id = {target.target_id: target for target in targets}
    parent_targets: dict[str, StructureTarget] = {}
    for target in targets:
        if target.kind == "vacancy-free":
            parent_targets[target.parent_id] = target

    # Parent-relative changes are provided only when both method results exist;
    # no vacancy calculation is required for standalone parent analysis.
    by_key = {(result["target_id"], result["method"]): result for result in results}
    structures: dict[str, Structure] = {}

    for result in results:
        target = target_by_id.get(result["target_id"])
        if target is None or target.kind != "oxygen-vacancy":
            continue
        parent_target = parent_targets.get(target.parent_id)
        parent_result = by_key.get((target.parent_id, result["method"]))
        if parent_target is None or parent_result is None:
            result.setdefault("limitations", []).append(
                "Parent-relative changes unavailable because the matched parent was not analyzed with this method."
            )
            continue
        try:
            parent_structure = structures.setdefault(
                parent_target.target_id, Structure.from_file(parent_target.structure_path)
            )
            child_structure = structures.setdefault(
                target.target_id, Structure.from_file(target.structure_path)
            )
            mapping = map_child_to_parent_sites(
                parent_structure,
                child_structure,
                tolerance=cfg.mapping_tolerance,
            )
        except Exception as exc:
            result.setdefault("limitations", []).append(
                f"Parent-relative atom mapping failed: {type(exc).__name__}: {exc}"
            )
            continue

        if result.get("method") == "wannier":
            child_sites = {
                int(rec["site_index"]): rec
                for rec in result.get("wannier_site_summary", [])
                if rec.get("site_index") is not None
            }
            parent_sites = {
                int(rec["site_index"]): rec
                for rec in parent_result.get("wannier_site_summary", [])
                if rec.get("site_index") is not None
            }
            if not child_sites or not parent_sites:
                continue
            fields = (
                "nearest_wf_count",
                "nearest_wf_electron_equivalent",
                "atom_centered_wf_count",
                "bond_centered_wf_participation",
                "ambiguous_wf_participation",
                "delocalized_wf_nearest_count",
            )
            changes = []
            for child_idx, parent_idx in mapping.items():
                child_rec = child_sites.get(child_idx)
                parent_rec = parent_sites.get(parent_idx)
                if child_rec is None or parent_rec is None:
                    continue
                row: dict[str, Any] = {
                    "site_index": child_idx,
                    "parent_site_index": parent_idx,
                    "element": child_structure[child_idx].specie.symbol,
                }
                for field in fields:
                    child_value = child_rec.get(field)
                    parent_value = parent_rec.get(field)
                    row[field] = child_value
                    row[f"parent_{field}"] = parent_value
                    if isinstance(child_value, (int, float)) and isinstance(parent_value, (int, float)):
                        row[f"delta_{field}"] = float(child_value) - float(parent_value)
                    else:
                        row[f"delta_{field}"] = None
                changes.append(row)
            result["parent_relative_changes"] = changes
            continue

        child_formal = _formal_by_site(result)
        parent_formal = _formal_by_site(parent_result)
        if not child_formal or not parent_formal:
            continue
        changes = []
        for child_idx, parent_idx in mapping.items():
            child_value = child_formal.get(child_idx)
            parent_value = parent_formal.get(parent_idx)
            if child_value is None or parent_value is None:
                continue
            changes.append(
                {
                    "site_index": child_idx,
                    "parent_site_index": parent_idx,
                    "element": child_structure[child_idx].specie.symbol,
                    "formal_oxidation_state": child_value,
                    "parent_formal_oxidation_state": parent_value,
                    "delta_formal_oxidation_state": child_value - parent_value,
                }
            )
        result["parent_relative_changes"] = changes

'''
text = text[:start] + new_parent + text[end:]
text = text.replace(
    '            "coordination": "coordination_number",\n',
    '            "coordination": "coordination_number",\n            "wannier_site_summary": "nearest_wf_count",\n',
    1,
)
p.write_text(text, encoding="utf-8")

# GUI settings and result browser.
p = Path("gui/pages/Oxidation_States.py")
text = p.read_text(encoding="utf-8")
marker = '''        wannier["less_memory"] = st.checkbox(
            "Low-memory GPAW overlap generation",
            value=bool(wannier.get("less_memory", False)),
            help=(
                "For the current single-Gamma route leave this off unless needed. "
                "The option is retained for future multi-k workflows."
            ),
        )
'''
insert = marker + '''        st.markdown("**Wannier-center classification thresholds**")
        wc1, wc2, wc3, wc4 = st.columns(4)
        wannier["atom_center_cutoff_angstrom"] = float(
            wc1.number_input(
                "Atom-center cutoff (Å)",
                min_value=0.01,
                value=float(wannier.get("atom_center_cutoff_angstrom", 0.45)),
                step=0.05,
            )
        )
        wannier["bond_center_cutoff_angstrom"] = float(
            wc2.number_input(
                "Bond-center cutoff (Å)",
                min_value=0.05,
                value=float(wannier.get("bond_center_cutoff_angstrom", 1.35)),
                step=0.05,
            )
        )
        wannier["bond_distance_balance_angstrom"] = float(
            wc3.number_input(
                "Bond distance balance (Å)",
                min_value=0.0,
                value=float(wannier.get("bond_distance_balance_angstrom", 0.30)),
                step=0.05,
            )
        )
        wannier["delocalized_spread_threshold_ang2"] = float(
            wc4.number_input(
                "Delocalized spread threshold (Å²)",
                min_value=0.01,
                value=float(wannier.get("delocalized_spread_threshold_ang2", 3.0)),
                step=0.25,
            )
        )
        st.caption(
            "These cutoffs classify periodic center geometry and spread outliers only; they do not define formal oxidation states."
        )
'''
if marker not in text:
    raise SystemExit("Wannier less-memory GUI marker not found")
text = text.replace(marker, insert, 1)

result_marker = '''            if target_sites_file.exists():
'''
result_insert = r'''            if selected_method == "wannier":
                analysis = method_result.get("wannier_analysis", {}) or {}
                if analysis:
                    st.markdown("#### Wannier-center analysis")
                    wm1, wm2, wm3, wm4 = st.columns(4)
                    wm1.metric("Wannier centers", int(analysis.get("n_wannier_centres", 0)))
                    represented = analysis.get("represented_electrons")
                    wm2.metric(
                        "Electron equivalent",
                        "-" if represented is None else f"{float(represented):.0f}",
                    )
                    wm3.metric(
                        "Spread outliers",
                        int(analysis.get("n_delocalized_spread_outliers", 0)),
                    )
                    max_spread = analysis.get("max_spread_ang2")
                    wm4.metric(
                        "Max spread (Å²)",
                        "-" if max_spread is None else f"{float(max_spread):.3f}",
                    )
                    counts = analysis.get("classification_counts", {}) or {}
                    if counts:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {"classification": name, "count": count}
                                    for name, count in counts.items()
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    if analysis.get("center_count_consistent_with_occupied_bands") is False:
                        st.warning("Wannier-center count does not match the occupied-band count recorded for this run.")

                site_summary = method_result.get("wannier_site_summary", []) or []
                if site_summary:
                    st.markdown("##### Per-site geometric Wannier descriptors")
                    st.dataframe(
                        pd.DataFrame(site_summary),
                        use_container_width=True,
                        hide_index=True,
                    )

                centre_rows = method_result.get("wannier_descriptors", []) or []
                if centre_rows:
                    with st.expander("Wannier centers and spreads", expanded=False):
                        center_df = pd.DataFrame(centre_rows)
                        if "spread_ang2" in center_df.columns:
                            center_df = center_df.sort_values(
                                "spread_ang2", ascending=False, na_position="last"
                            )
                        st.dataframe(center_df, use_container_width=True, hide_index=True)

                parent_changes = method_result.get("parent_relative_changes", []) or []
                if parent_changes:
                    st.markdown("##### Parent-relative Wannier descriptor changes")
                    st.dataframe(
                        pd.DataFrame(parent_changes),
                        use_container_width=True,
                        hide_index=True,
                    )
                elif str(selected_meta.get("structure_kind", "")) == "vacancy-free":
                    st.info(
                        "This is a standalone vacancy-free Wannier analysis. No oxygen-vacancy result is required. "
                        "Parent-relative deltas will appear only after a matched vacancy structure is also analyzed."
                    )

'''
if result_marker not in text:
    raise SystemExit("GUI result marker not found")
text = text.replace(result_marker, result_insert + result_marker, 1)
p.write_text(text, encoding="utf-8")

# Tests.
p = Path("tests/test_oxidation.py")
text = p.read_text(encoding="utf-8")
if "import dopingflow.wannier_analysis as wannier_analysis" not in text:
    text = text.replace(
        "import dopingflow.oxidation_dft as oxidation_dft\n",
        "import dopingflow.oxidation_dft as oxidation_dft\nimport dopingflow.wannier_analysis as wannier_analysis\n",
        1,
    )
append = r'''


def test_wannier_xyz_and_spread_parsers_use_only_wf_centres_and_final_spreads(tmp_path: Path) -> None:
    xyz = tmp_path / "wannier90_centres.xyz"
    xyz.write_text(
        "4\nWannier centers plus atoms\n"
        "X 0.1 0.2 0.3\n"
        "X 1.1 1.2 1.3\n"
        "Sn 0.0 0.0 0.0\n"
        "O 2.0 2.0 2.0\n",
        encoding="utf-8",
    )
    centres = oxidation_dft._parse_wannier_centres(xyz)
    assert len(centres) == 2

    wout = tmp_path / "wannier90.wout"
    wout.write_text(
        " WF centre and spread    1  ( 0.0, 0.0, 0.0 )     2.50000000\n"
        " WF centre and spread    2  ( 1.0, 1.0, 1.0 )     0.70000000\n"
        " WF centre and spread    1  ( 0.0, 0.0, 0.0 )     0.65000000\n",
        encoding="utf-8",
    )
    spreads = wannier_analysis.parse_wannier_spreads(wout)
    assert spreads == {0: pytest.approx(0.65), 1: pytest.approx(0.7)}


def test_wannier_geometric_analysis_uses_periodic_distances_and_flags_spread_outlier() -> None:
    structure = Structure(
        Lattice.cubic(10.0),
        ["Sn", "O", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
    )
    centres = [
        {"center_index": 0, "cartesian_angstrom": [9.9, 0.0, 0.0]},
        {"center_index": 1, "cartesian_angstrom": [2.5, 0.0, 0.0]},
        {"center_index": 2, "cartesian_angstrom": [2.5, 2.5, 0.0]},
        {"center_index": 3, "cartesian_angstrom": [0.2, 0.0, 0.0]},
    ]
    enriched, site_summary, summary = wannier_analysis.analyze_wannier_centres(
        structure,
        centres,
        {0: 0.6, 1: 0.7, 2: 0.8, 3: 5.0},
        atom_center_cutoff_angstrom=0.4,
        bond_center_cutoff_angstrom=3.0,
        bond_distance_balance_angstrom=0.2,
        delocalized_spread_threshold_ang2=3.0,
        electrons_per_wf=2.0,
    )
    assert enriched[0]["classification"] == "atom-centered"
    assert enriched[0]["nearest_site_index"] == 0
    assert enriched[0]["nearest_distance_angstrom"] == pytest.approx(0.1)
    assert enriched[1]["classification"] == "bond-centered"
    assert enriched[2]["classification"] == "multicenter/ambiguous"
    assert enriched[3]["classification"] == "anomalous/delocalized"
    assert enriched[3]["geometric_classification"] == "atom-centered"
    assert summary["represented_electrons"] == pytest.approx(8.0)
    assert summary["n_delocalized_spread_outliers"] == 1
    assert summary["max_spread_center_index"] == 3
    assert site_summary[0]["nearest_wf_count"] >= 2
'''
if "test_wannier_xyz_and_spread_parsers_use_only_wf_centres_and_final_spreads" not in text:
    text += append
p.write_text(text, encoding="utf-8")

# Test workflow needs ASE for the already-existing Wannier input-writer test.
p = Path(".github/workflows/oxidation-tests.yml")
text = p.read_text(encoding="utf-8")
text = text.replace(
    'pip install -e ".[dev,docs]"',
    'pip install -e ".[dev,docs]" ase',
    1,
)
p.write_text(text, encoding="utf-8")

# Documentation.
p = Path("docs/source/methods/oxidation_states.rst")
text = p.read_text(encoding="utf-8")
start = text.index("``wannier``\n")
end = text.index("\n``eos``\n", start)
wannier_docs = r'''``wannier``
   Parses Wannier-center information and can generate it natively from an
   existing GPAW restart when ``execute=true``.  The first native mode is
   deliberately conservative: it supports an isolated, non-spin-polarized,
   Gamma-only occupied manifold, requires wavefunctions stored in
   ``oxidation.gpw``, detects the fully occupied bands and finite HOMO-LUMO gap,
   writes a Wannier90 input using Bloch phases as the initial gauge and
   ``write_xyz=true``, lets GPAW write ``.eig``/``.mmn``, and runs
   ``wannier90.x``.  Because the occupied manifold is isolated, no
   disentanglement is used and no arbitrary atomic projector set is invented.

   Completed Wannier results are also analyzed without rerunning Wannier90 when
   ``execute=false``.  Periodic minimum-image distances classify centers as
   atom-centered, bond-centered, or multicenter/ambiguous using configurable
   geometric thresholds.  A separate spread threshold flags anomalous or
   delocalized Wannier functions.  The defaults are ``0.45 Å`` for the
   atom-center cutoff, ``1.35 Å`` for the bond-center cutoff, ``0.30 Å`` for the
   two-neighbor distance balance, and ``3.0 Å²`` for the spread-outlier
   threshold.  These are screening heuristics, not universal chemical
   boundaries, and should be inspected for the material class.

   The post-processing writes ``wannier_centres_analysis.csv``,
   ``wannier_site_summary.csv``, and ``wannier_analysis.json`` beside the
   Wannier90 files.  Per-site counts and paired-electron equivalents are
   geometric bookkeeping descriptors; they are not atomic populations, Bader
   charges, or formal oxidation states.  Vacancy-free structures can be
   analyzed completely on their own.  If a matched oxygen-vacancy structure
   and its parent are both analyzed later, dopingflow additionally reports
   parent-relative changes in these per-site Wannier descriptors.  No vacancy
   structure is required for the standalone analysis.

   Metallic, spin-polarized, multi-k, or entangled cases are rejected by this
   native mode and should use an explicit projection/disentanglement workflow.
   Static centers remain supporting descriptors and are not converted into
   formal oxidation states automatically.
'''
text = text[:start] + wannier_docs + text[end:]
p.write_text(text, encoding="utf-8")

p = Path("README.md")
text = p.read_text(encoding="utf-8")
block = r'''
### Wannier-center descriptor analysis

The GPAW/Wannier90 oxidation route can post-process an already completed
Wannier calculation with `execute = false`; it does not require Wannier90 to be
rerun.  It parses final Wannier spreads, associates centers with atoms using
periodic minimum-image distances, classifies atom-/bond-/multicenter geometry,
and flags configurable spread outliers.  It writes
`wannier_centres_analysis.csv`, `wannier_site_summary.csv`, and
`wannier_analysis.json`.  These outputs are electronic-structure descriptors,
not automatic formal oxidation-state assignments.

A vacancy-free structure is a complete standalone analysis target.  If matched
oxygen-vacancy structures are added later and both parent and vacancy are
analyzed with Wannier, parent-relative per-site descriptor changes are generated
then; vacancy data is not required up front.

'''
if "### Wannier-center descriptor analysis" not in text:
    license_marker = "## License"
    if license_marker in text:
        text = text.replace(license_marker, block + license_marker, 1)
    else:
        text += "\n" + block
p.write_text(text, encoding="utf-8")

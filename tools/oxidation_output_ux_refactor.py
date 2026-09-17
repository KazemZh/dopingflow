from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"patch anchor not found: {label}")
    return text.replace(old, new, 1)


# 1) Treat ordinary BVAnalyzer non-assignment as an unsupported chemical case.
path = Path("src/dopingflow/oxidation_structural.py")
text = path.read_text()
old = """    analyzer = BVAnalyzer(**kwargs)
    valences = analyzer.get_valences(structure)
    formal = site_records(structure, [float(value) for value in valences])
"""
new = """    analyzer = BVAnalyzer(**kwargs)
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
"""
path.write_text(replace_once(text, old, new, "bond-valence non-assignment"))


# 2) Quiet per-structure failures, add per-structure outputs and end-of-run summary.
path = Path("src/dopingflow/oxidation.py")
text = path.read_text()
text = replace_once(
    text,
    '        log.exception("Oxidation method %s failed for %s", method, target.target_id)\n',
    """        log.debug(
            "Oxidation method %s failed for %s",
            method,
            target.target_id,
            exc_info=True,
        )
""",
    "quiet method failure logging",
)

marker = "\ndef run_oxidation(\n"
helpers = r'''

def _safe_output_parts(target_id: str) -> list[str]:
    """Convert a target ID into safe hierarchical output-directory components."""
    parts: list[str] = []
    for raw in str(target_id).replace("\\", "/").split("/"):
        raw = raw.strip()
        if not raw or raw in {".", ".."}:
            continue
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
        parts.append(safe or "target")
    return parts or ["target"]


def _target_output_dir(cfg: OxidationConfig, target: StructureTarget) -> Path:
    return cfg.output_dir / "structures" / Path(*_safe_output_parts(target.target_id))


def _write_per_structure_outputs(
    cfg: OxidationConfig,
    targets: Sequence[StructureTarget],
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write an auditable result package for every analyzed structure."""
    index_rows: list[dict[str, Any]] = []
    successful_statuses = {"assigned", "descriptors-only"}

    for target in targets:
        target_results = [r for r in results if r["target_id"] == target.target_id]
        target_dir = _target_output_dir(cfg, target)
        method_dir = target_dir / "methods"
        target_dir.mkdir(parents=True, exist_ok=True)
        method_dir.mkdir(parents=True, exist_ok=True)

        site_rows = _flatten_site_rows(target_results)
        method_statuses = {
            str(result["method"]): str(result.get("assignment_status", "unknown"))
            for result in target_results
        }
        unresolved = [
            method
            for method, status in method_statuses.items()
            if status not in successful_statuses
        ]
        successful = [
            method
            for method, status in method_statuses.items()
            if status in successful_statuses
        ]
        if unresolved and successful:
            overall_status = "partial"
        elif unresolved:
            overall_status = "unresolved"
        else:
            overall_status = "complete"

        _json_write(target_dir / "oxidation_results.json", target_results)
        _csv_write(target_dir / "oxidation_sites.csv", site_rows)
        for result in target_results:
            _json_write(method_dir / f"{result['method']}.json", result)

        summary = {
            "target_id": target.target_id,
            "parent_id": target.parent_id,
            "structure_kind": target.kind,
            "structure_path": str(target.structure_path),
            "n_oxygen_vacancies": (
                target.n_vacancies if target.vacancy_species == "O" else 0
            ),
            "metadata": target.metadata,
            "overall_status": overall_status,
            "method_statuses": method_statuses,
            "successful_methods": successful,
            "unresolved_methods": unresolved,
            "output_directory": str(target_dir),
        }
        _json_write(target_dir / "summary.json", summary)

        try:
            n_atoms = len(Structure.from_file(target.structure_path))
        except Exception:
            n_atoms = None
        index_rows.append(
            {
                **summary,
                "n_atoms": n_atoms,
                "n_methods": len(target_results),
                "n_successful_methods": len(successful),
                "n_unresolved_methods": len(unresolved),
            }
        )

    return index_rows


def _log_analysis_summary(
    results: Sequence[dict[str, Any]],
    *,
    n_targets: int,
    discovery_warnings: Sequence[str],
) -> None:
    """Report non-assignments once, at the end, without per-structure tracebacks."""
    unresolved = [
        result
        for result in results
        if result.get("assignment_status") not in {"assigned", "descriptors-only"}
    ]
    log.info(
        "Oxidation analysis completed: targets=%d method_results=%d unresolved=%d",
        n_targets,
        len(results),
        len(unresolved),
    )
    for message in discovery_warnings:
        log.warning("Oxidation discovery: %s", message)
    if not unresolved:
        return
    log.warning(
        "Oxidation analysis completed with %d structure/method case(s) without an assignment:",
        len(unresolved),
    )
    for result in unresolved:
        detail = result.get("error") or "; ".join(result.get("limitations", []))
        suffix = f" - {detail}" if detail else ""
        log.warning(
            "  %s [%s]: %s%s",
            result.get("target_id"),
            result.get("method"),
            result.get("assignment_status"),
            suffix,
        )

'''
text = replace_once(text, marker, helpers + marker, "run_oxidation helper insertion")

text = replace_once(
    text,
    """    _json_write(results_path, results)
    _json_write(cfg.output_dir / "oxidation_comparison.json", comparisons)
    _json_write(cfg.output_dir / "dft_followup_candidates.json", followup_candidates)
    _csv_write(cfg.output_dir / "oxidation_sites.csv", _flatten_site_rows(results))
    _json_write(
""",
    """    _json_write(results_path, results)
    _json_write(cfg.output_dir / "oxidation_comparison.json", comparisons)
    _json_write(cfg.output_dir / "dft_followup_candidates.json", followup_candidates)
    _csv_write(cfg.output_dir / "oxidation_sites.csv", _flatten_site_rows(results))
    structure_index = _write_per_structure_outputs(cfg, targets, results)
    _csv_write(cfg.output_dir / "oxidation_structure_index.csv", structure_index)
    _json_write(cfg.output_dir / "oxidation_structure_index.json", structure_index)
    _json_write(
""",
    "per-structure writer call",
)
text = replace_once(
    text,
    """            "n_method_results": len(results),
            "target_ids": [target.target_id for target in targets],
""",
    """            "n_method_results": len(results),
            "target_ids": [target.target_id for target in targets],
            "per_structure_root": str(cfg.output_dir / "structures"),
            "structure_index_csv": str(cfg.output_dir / "oxidation_structure_index.csv"),
""",
    "meta per-structure paths",
)
text = replace_once(
    text,
    """    )
    return results_path


try:
""",
    """    )
    _log_analysis_summary(
        results,
        n_targets=len(targets),
        discovery_warnings=discovery_warnings,
    )
    return results_path


try:
""",
    "end-of-run summary call",
)
path.write_text(text)


# 3) Replace the flat GUI dump with structure -> method -> atom browsing.
path = Path("gui/pages/Oxidation_States.py")
text = path.read_text()
start = text.index('st.divider()\nst.subheader("Results")\n')
replacement = r'''st.divider()
st.subheader("Results")
source_path = Path(source_root).expanduser()
if not source_path.is_absolute():
    source_path = (project_root / source_path).resolve()
results_root = Path(output_dir).expanduser()
if not results_root.is_absolute():
    results_root = (source_path / results_root).resolve()

sites_csv = results_root / "oxidation_sites.csv"
index_csv = results_root / "oxidation_structure_index.csv"
results_json = results_root / "oxidation_results.json"
comparison_json = results_root / "oxidation_comparison.json"
followup_json = results_root / "dft_followup_candidates.json"

st.caption(f"Resolved output: `{results_root}`")

if not index_csv.exists():
    st.info(
        "No oxidation_structure_index.csv found yet. Run the oxidation stage with the "
        "updated workflow to create per-structure result folders and the structure browser."
    )
else:
    try:
        structure_index = pd.read_csv(index_csv)
    except Exception as exc:
        st.warning(f"Could not read {index_csv.name}: {exc}")
    else:
        st.markdown("#### Analysed structures")
        overview_columns = [
            column
            for column in (
                "target_id",
                "structure_kind",
                "n_oxygen_vacancies",
                "n_atoms",
                "overall_status",
                "n_successful_methods",
                "n_unresolved_methods",
            )
            if column in structure_index.columns
        ]
        st.dataframe(
            structure_index[overview_columns],
            use_container_width=True,
            hide_index=True,
        )

        target_ids = structure_index["target_id"].astype(str).tolist()
        selected_target = st.selectbox(
            "Choose a structure",
            target_ids,
            key="oxidation_result_target",
        )
        selected_meta = structure_index[
            structure_index["target_id"].astype(str) == selected_target
        ].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Kind", str(selected_meta.get("structure_kind", "")))
        m2.metric("O vacancies", int(selected_meta.get("n_oxygen_vacancies", 0)))
        n_atoms_value = selected_meta.get("n_atoms")
        m3.metric("Atoms", "-" if pd.isna(n_atoms_value) else int(n_atoms_value))
        m4.metric("Status", str(selected_meta.get("overall_status", "unknown")))

        st.caption(f"Structure file: `{selected_meta.get('structure_path', '')}`")
        st.caption(
            f"Per-structure results: `{selected_meta.get('output_directory', '')}`"
        )

        target_result_dir = Path(str(selected_meta.get("output_directory", "")))
        target_results_file = target_result_dir / "oxidation_results.json"
        target_sites_file = target_result_dir / "oxidation_sites.csv"
        target_summary_file = target_result_dir / "summary.json"

        target_results = []
        if target_results_file.exists():
            try:
                target_results = json.loads(
                    target_results_file.read_text(encoding="utf-8")
                )
            except Exception as exc:
                st.warning(f"Could not read {target_results_file}: {exc}")

        if target_results:
            method_names = [str(item.get("method")) for item in target_results]
            selected_method = st.selectbox(
                "Oxidation method",
                method_names,
                key="oxidation_result_method",
            )
            method_result = next(
                item
                for item in target_results
                if str(item.get("method")) == selected_method
            )
            status = str(method_result.get("assignment_status", "unknown"))
            if status in {"assigned", "descriptors-only"}:
                st.success(f"{selected_method}: {status}")
            else:
                detail = method_result.get("error") or "; ".join(
                    method_result.get("limitations", [])
                )
                st.warning(
                    f"{selected_method}: {status}. "
                    + (
                        str(detail)
                        if detail
                        else "No assignment was produced for this structure."
                    )
                )

            if target_sites_file.exists():
                try:
                    target_sites = pd.read_csv(target_sites_file)
                except Exception as exc:
                    st.warning(f"Could not read {target_sites_file}: {exc}")
                else:
                    method_sites = target_sites[
                        target_sites["method"].astype(str) == selected_method
                    ].copy()
                    if "site_index" in method_sites.columns:
                        method_sites = method_sites.sort_values(
                            "site_index", na_position="last"
                        )
                    preferred = [
                        "site_index",
                        "element",
                        "formal_oxidation_state",
                        "coordination_number",
                        "bader_partial_charge",
                        "magnetic_moment",
                        "parent_site_index",
                        "parent_formal_oxidation_state",
                        "delta_formal_oxidation_state",
                        "assignment_status",
                    ]
                    display_columns = [
                        col for col in preferred if col in method_sites.columns
                    ]
                    extra_columns = [
                        col
                        for col in method_sites.columns
                        if col not in display_columns
                        and col
                        not in {
                            "target_id",
                            "parent_id",
                            "structure_kind",
                            "method",
                            "strategy",
                            "prediction_scope",
                        }
                    ]
                    display_columns.extend(extra_columns)
                    st.markdown("#### Atom-by-atom results")
                    if method_sites.empty:
                        st.info(
                            "This method produced no site/composition records for the "
                            "selected structure."
                        )
                    else:
                        st.dataframe(
                            method_sites[display_columns],
                            use_container_width=True,
                            hide_index=True,
                        )

            with st.expander("Selected method details", expanded=False):
                st.json(method_result)

        if target_summary_file.exists():
            with st.expander("Structure summary", expanded=False):
                try:
                    st.json(
                        json.loads(target_summary_file.read_text(encoding="utf-8"))
                    )
                except Exception as exc:
                    st.warning(f"Could not read {target_summary_file}: {exc}")

with st.expander("Aggregate result files", expanded=False):
    if sites_csv.exists():
        try:
            all_sites = pd.read_csv(sites_csv)
            st.caption(f"Global atom/composition table: {len(all_sites):,} rows")
        except Exception as exc:
            st.warning(f"Could not read {sites_csv}: {exc}")
    for path in (results_json, comparison_json, followup_json):
        if not path.exists():
            continue
        st.markdown(f"**{path.name}**")
        try:
            st.json(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            st.warning(f"Could not read {path}: {exc}")
'''
path.write_text(text[:start] + replacement + "\n")


# 4) Add regression coverage for quiet failures and hierarchical output packages.
path = Path("tests/test_oxidation.py")
text = path.read_text()
if "def test_per_structure_outputs_and_quiet_failure_summary" not in text:
    text += r'''


def test_per_structure_outputs_and_quiet_failure_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_root = tmp_path / "random_structures"
    _write_parent_tree(source_root)

    def runner_for(method: str):
        def runner(target, cfg, settings):
            del cfg, settings
            if method == "bond-valence":
                raise RuntimeError("synthetic non-assignment")
            structure = Structure.from_file(target.structure_path)
            return base_method_result(
                method=method,
                target=target,
                scope="site-resolved",
                formal_oxidation_states=site_records(structure, [4, 5, -2, -2]),
            )

        return runner

    monkeypatch.setattr(oxidation, "_method_runner", runner_for)
    caplog.set_level("INFO")
    raw = {
        "structure": {"outdir": str(source_root)},
        "oxidation": {
            "strategy": "combined",
            "methods": ["bond-valence", "toss-gnn"],
            "include_oxygen_vacancies": False,
        },
    }
    run_oxidation(raw, tmp_path)

    target_dir = (
        source_root / "06_oxidation" / "structures" / "Sb25" / "candidate_0001"
    )
    assert (target_dir / "summary.json").exists()
    assert (target_dir / "oxidation_sites.csv").exists()
    assert (target_dir / "methods" / "bond-valence.json").exists()
    assert (target_dir / "methods" / "toss-gnn.json").exists()
    assert (source_root / "06_oxidation" / "oxidation_structure_index.csv").exists()
    assert not [record for record in caplog.records if record.levelname == "ERROR"]
    assert "without an assignment" in caplog.text
'''
path.write_text(text)


# 5) Document the new result layout and GUI browser.
path = Path("docs/source/methods/oxidation_states.rst")
text = path.read_text()
old = """The stage writes under ``[oxidation].output_dir``:

* ``oxidation_results.json``: complete per-method records;
* ``oxidation_sites.csv``: flattened site/composition-token records;
* ``oxidation_comparison.json``: descriptive method comparison;
* ``dft_followup_candidates.json``: optional follow-up selection;
* ``meta.json``: resolved settings, target list, and provenance.
"""
new = """The stage writes aggregate files under ``[oxidation].output_dir``:

* ``oxidation_results.json``: complete per-method records;
* ``oxidation_sites.csv``: flattened site/composition-token records;
* ``oxidation_structure_index.csv`` / ``.json``: one row/object per analyzed structure;
* ``oxidation_comparison.json``: descriptive method comparison;
* ``dft_followup_candidates.json``: optional follow-up selection;
* ``meta.json``: resolved settings, target list, and provenance.

In addition, every structure gets a dedicated hierarchical result package under
``structures/<target_id>/``. For example, ``Sb5_In10/candidate_001`` is written
to ``structures/Sb5_In10/candidate_001/``. Each package contains
``summary.json``, ``oxidation_sites.csv``, ``oxidation_results.json``, and one
JSON file per requested method below ``methods/``. This layout is intended for
structure-by-structure inspection while the aggregate tables remain available
for screening and statistics.

A method that cannot assign a particular structure is recorded as
``unsupported``, ``unavailable``, or ``failed`` without interrupting other
structures. These cases are summarized once at the end of the run instead of
emitting a traceback for every ordinary non-assignment.
"""
path.write_text(replace_once(text, old, new, "docs output section"))

path = Path("README.md")
text = path.read_text()
needle = """- **Phase Diagram** — explores raw/corrected hull results and vacancy-resolved
  energy above hull.
"""
addition = needle + """- **Oxidation States** — configures structural/ML/DFT oxidation analysis and provides a
  structure browser: select an analyzed parent or vacancy structure, choose a method,
  and inspect atom-by-atom oxidation states/descriptors. Results are also written per
  structure under ``06_oxidation/structures/<target_id>/``.
"""
path.write_text(replace_once(text, needle, addition, "README oxidation GUI bullet"))

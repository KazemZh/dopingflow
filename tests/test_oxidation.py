from __future__ import annotations

import json
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

import dopingflow.oxidation as oxidation
from dopingflow.oxidation import (
    OptionalMethodUnavailable,
    OxidationConfig,
    StructureTarget,
    _compare_target_results,
    base_method_result,
    cached_model,
    clear_model_cache,
    discover_oxidation_targets,
    parse_oxidation_config,
    run_oxidation,
    site_records,
    unsupported_elements,
)
from dopingflow.oxidation_dft import run_dft_method


def _structure() -> Structure:
    return Structure(
        Lattice.cubic(5.0),
        ["Sn", "Sb", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
    )


def _write_parent_tree(root: Path) -> Path:
    structure = _structure()
    composition_dir = root / "Sb25"
    candidate = composition_dir / "candidate_0001"
    (candidate / "01_scan").mkdir(parents=True)
    (candidate / "02_relax").mkdir(parents=True)
    Poscar(structure).write_file(candidate / "01_scan" / "POSCAR")
    Poscar(structure).write_file(candidate / "02_relax" / "POSCAR")
    (composition_dir / "selected_candidates.txt").write_text(
        "candidate_0001\n", encoding="utf-8"
    )
    return candidate


def _cfg(tmp_path: Path, *, strategy: str = "dft", methods: tuple[str, ...] = ("bader",)) -> OxidationConfig:
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    return OxidationConfig(
        root=tmp_path,
        source_root=source_root,
        output_dir=source_root / "06_oxidation",
        enabled=True,
        strategy=strategy,
        methods=methods,
        include_vacancy_free=True,
        include_oxygen_vacancies=True,
        mapping_tolerance=1.2,
        fail_fast=False,
        dft_followup_enabled=False,
        dft_followup_candidate_limit=5,
        dft_followup_execute=False,
        dft_followup_methods=("dft-electronic", "bader", "eos"),
        settings={},
    )


def test_strategy_dispatch_and_individual_methods(tmp_path: Path) -> None:
    raw = {
        "structure": {"outdir": "random_structures"},
        "oxidation": {
            "strategy": "combined",
            "methods": ["bond-valence", "toss-gnn", "bader"],
        },
    }
    cfg = parse_oxidation_config(raw, tmp_path)
    assert cfg.strategy == "combined"
    assert cfg.methods == ("bond-valence", "toss-gnn", "bader")

    ml = parse_oxidation_config(
        {"oxidation": {"strategy": "ml", "methods": ["toss-gnn"]}},
        tmp_path,
    )
    assert ml.methods == ("toss-gnn",)

    with pytest.raises(ValueError, match="Unknown oxidation-state method"):
        parse_oxidation_config(
            {"oxidation": {"strategy": "structural", "methods": ["toss-bayesian"]}},
            tmp_path,
        )

    with pytest.raises(ValueError, match="use strategy='combined'"):
        parse_oxidation_config(
            {"oxidation": {"strategy": "ml", "methods": ["toss-gnn", "bader"]}},
            tmp_path,
        )


def test_legacy_gpaw_output_root_is_migrated(tmp_path: Path) -> None:
    raw = {
        "oxidation": {
            "strategy": "dft",
            "methods": ["bader"],
            "dft_electronic": {"output_root": "gpaw_oxidation"},
            "bader": {"output_root": "gpaw_oxidation"},
        }
    }
    cfg = parse_oxidation_config(raw, tmp_path)
    assert cfg.settings["dft_electronic"]["output_root"] == "dft_oxidation"
    assert cfg.settings["bader"]["output_root"] == "dft_oxidation"

    custom = {
        "oxidation": {
            "strategy": "dft",
            "methods": ["bader"],
            "dft_electronic": {"output_root": "my_dft_results"},
            "bader": {"output_root": "gpaw_oxidation"},
        }
    }
    cfg = parse_oxidation_config(custom, tmp_path)
    assert cfg.settings["dft_electronic"]["output_root"] == "my_dft_results"
    assert cfg.settings["bader"]["output_root"] == "my_dft_results"


def test_model_cache_loads_once_per_process() -> None:
    clear_model_cache()
    calls = 0

    def loader() -> object:
        nonlocal calls
        calls += 1
        return object()

    first = cached_model(("unit-test-model", "cpu"), loader)
    second = cached_model(("unit-test-model", "cpu"), loader)
    assert first is second
    assert calls == 1


def test_site_order_and_unsupported_species_are_explicit() -> None:
    structure = _structure()
    records = site_records(structure, [4, 5, -2, -2])
    assert [record["site_index"] for record in records] == [0, 1, 2, 3]
    assert [record["element"] for record in records] == ["Sn", "Sb", "O", "O"]
    assert unsupported_elements(structure, {"Sn", "O"}) == ["Sb"]


def test_discovery_includes_parent_and_oxygen_vacancy(tmp_path: Path) -> None:
    source_root = tmp_path / "random_structures"
    candidate = _write_parent_tree(source_root)
    parent = _structure()
    vacancy = parent.copy()
    vacancy.remove_sites([3])
    vacancy_path = source_root / "vacancy_relaxed" / "POSCAR"
    vacancy_path.parent.mkdir(parents=True)
    Poscar(vacancy).write_file(vacancy_path)
    rows = [
        {
            "parent_id": "Sb25/candidate_0001",
            "configuration_id": "config_0001",
            "vacancy_species": "O",
            "n_vacancies": 1,
            "relaxed_poscar_path": str(vacancy_path),
            "backend": "mace",
            "model": "small",
            "task": "",
        }
    ]
    (source_root / "vacancies_database.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    raw = {
        "structure": {"outdir": str(source_root)},
        "oxidation": {"strategy": "structural", "methods": ["bond-valence"]},
    }
    cfg = parse_oxidation_config(raw, tmp_path)
    targets, warnings = discover_oxidation_targets(cfg)
    assert warnings == []
    assert [target.kind for target in targets] == ["vacancy-free", "oxygen-vacancy"]
    assert targets[0].structure_path == (candidate / "02_relax" / "POSCAR").resolve()


def test_composition_level_predictions_never_become_site_assignments(tmp_path: Path) -> None:
    path = tmp_path / "POSCAR"
    Poscar(_structure()).write_file(path)
    target = StructureTarget(
        target_id="x",
        parent_id="x",
        kind="vacancy-free",
        structure_path=path,
        n_vacancies=0,
        vacancy_species=None,
    )
    site_result = base_method_result(
        method="bond-valence",
        target=target,
        scope="site-resolved",
        formal_oxidation_states=site_records(_structure(), [4, 5, -2, -2]),
    )
    bertos_result = base_method_result(
        method="bertos",
        target=target,
        scope="composition-level",
        formal_oxidation_states=[
            {
                "composition_token_index": 0,
                "element": "Sn",
                "formal_oxidation_state": 4,
            }
        ],
    )
    comparison = _compare_target_results([site_result, bertos_result])
    assert comparison["composition_level_methods"] == ["bertos"]
    assert comparison["site_comparison"][0]["assignments"] == {"bond-valence": 4}


def test_optional_method_failure_does_not_stop_other_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "random_structures"
    _write_parent_tree(source_root)
    calls: list[str] = []

    def runner_for(method: str):
        def runner(target, cfg, settings):
            del cfg, settings
            calls.append(method)
            if method == "toss-gnn":
                raise OptionalMethodUnavailable("checkpoint dependency missing")
            structure = Structure.from_file(target.structure_path)
            return base_method_result(
                method=method,
                target=target,
                scope="site-resolved",
                formal_oxidation_states=site_records(structure, [4, 5, -2, -2]),
            )

        return runner

    monkeypatch.setattr(oxidation, "_method_runner", runner_for)
    raw = {
        "structure": {"outdir": str(source_root)},
        "oxidation": {
            "strategy": "combined",
            "methods": ["bond-valence", "toss-gnn"],
            "include_oxygen_vacancies": False,
        },
    }
    path = run_oxidation(raw, tmp_path)
    results = json.loads(path.read_text(encoding="utf-8"))
    assert calls == ["bond-valence", "toss-gnn"]
    assert {result["method"]: result["assignment_status"] for result in results} == {
        "bond-valence": "assigned",
        "toss-gnn": "unavailable",
    }


def test_ml_only_mode_never_dispatches_dft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "random_structures"
    _write_parent_tree(source_root)
    dispatched: list[str] = []

    def runner_for(method: str):
        assert method not in {"dft-electronic", "bader", "wannier", "eos"}

        def runner(target, cfg, settings):
            del cfg, settings
            dispatched.append(method)
            structure = Structure.from_file(target.structure_path)
            return base_method_result(
                method=method,
                target=target,
                scope="site-resolved",
                formal_oxidation_states=site_records(structure, [4, 5, -2, -2]),
            )

        return runner

    monkeypatch.setattr(oxidation, "_method_runner", runner_for)
    raw = {
        "structure": {"outdir": str(source_root)},
        "oxidation": {
            "strategy": "ml",
            "methods": ["toss-gnn"],
            "include_oxygen_vacancies": False,
        },
    }
    run_oxidation(raw, tmp_path)
    assert dispatched == ["toss-gnn"]


def test_bader_is_descriptor_only_and_never_invents_integer_state(tmp_path: Path) -> None:
    structure = Structure(
        Lattice.cubic(5.0),
        ["Sn", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    structure_path = tmp_path / "POSCAR"
    Poscar(structure).write_file(structure_path)
    target = StructureTarget(
        target_id="parent",
        parent_id="parent",
        kind="vacancy-free",
        structure_path=structure_path,
        n_vacancies=0,
        vacancy_species=None,
    )
    cfg = _cfg(tmp_path)
    workdir = tmp_path / "bader-existing"
    workdir.mkdir()
    (workdir / "ACF.dat").write_text(
        "# X Y Z CHARGE MIN_DIST ATOMIC_VOL\n"
        "---------------------------------------------\n"
        "1 0.0 0.0 0.0 49.60 0.50 10.0\n"
        "2 2.5 2.5 2.5 8.30 0.50 11.0\n",
        encoding="utf-8",
    )
    result = run_dft_method(
        "bader",
        target,
        cfg,
        {
            "workdir": str(workdir),
            "execute": False,
        },
    )
    assert result["assignment_status"] == "descriptors-only"
    assert result["formal_oxidation_states"] == []
    assert result["bader_partial_charges"][0]["atomic_number"] == 50
    assert result["bader_partial_charges"][0]["bader_partial_charge"] == pytest.approx(0.4)
    assert result["bader_partial_charges"][1]["bader_partial_charge"] == pytest.approx(-0.3)


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


def test_eos_requires_explicit_validated_assignment(tmp_path: Path) -> None:
    structure = Structure(
        Lattice.cubic(5.0),
        ["Sn", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    structure_path = tmp_path / "POSCAR"
    Poscar(structure).write_file(structure_path)
    target = StructureTarget(
        target_id="parent",
        parent_id="parent",
        kind="vacancy-free",
        structure_path=structure_path,
        n_vacancies=0,
        vacancy_species=None,
    )
    cfg = _cfg(tmp_path, methods=("eos",))
    workdir = tmp_path / "eos-existing"
    workdir.mkdir()
    result_path = workdir / "eos_results.json"
    result_path.write_text(
        json.dumps(
            {
                "validated": False,
                "assignment_procedure": "",
                "sites": [{"site_index": 0, "element": "Sn", "formal_oxidation_state": 4}],
            }
        ),
        encoding="utf-8",
    )
    unsupported = run_dft_method(
        "eos",
        target,
        cfg,
        {"workdir": str(workdir), "execute": False},
    )
    assert unsupported["assignment_status"] == "unsupported"
    assert unsupported["formal_oxidation_states"] == []

    result_path.write_text(
        json.dumps(
            {
                "validated": True,
                "assignment_procedure": "adiabatic charge-pumping EOS workflow",
                "sites": [
                    {"site_index": 0, "element": "Sn", "formal_oxidation_state": 4},
                    {"site_index": 1, "element": "O", "formal_oxidation_state": -2},
                ],
            }
        ),
        encoding="utf-8",
    )
    assigned = run_dft_method(
        "eos",
        target,
        cfg,
        {"workdir": str(workdir), "execute": False},
    )
    assert assigned["assignment_status"] == "assigned"
    assert [
        item["formal_oxidation_state"] for item in assigned["formal_oxidation_states"]
    ] == [4, -2]



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

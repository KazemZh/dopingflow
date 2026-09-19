from __future__ import annotations

import json
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

import dopingflow.oxidation as oxidation
import dopingflow.oxidation_dft as oxidation_dft
import dopingflow.oxidation_dft_auto as oxidation_dft_auto
import dopingflow.wannier_analysis as wannier_analysis
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


    dft_default = parse_oxidation_config(
        {"oxidation": {"strategy": "dft"}},
        tmp_path,
    )
    assert dft_default.methods == ("dft-auto",)


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


def test_target_include_filters_exact_safe_and_glob_targets(tmp_path: Path) -> None:
    source_root = tmp_path / "random_structures"
    _write_parent_tree(source_root)
    parent = _structure()
    vacancy = parent.copy()
    vacancy.remove_sites([3])
    vacancy_path = source_root / "vacancy_relaxed" / "POSCAR"
    vacancy_path.parent.mkdir(parents=True)
    Poscar(vacancy).write_file(vacancy_path)
    (source_root / "vacancies_database.json").write_text(
        json.dumps(
            [
                {
                    "parent_id": "Sb25/candidate_0001",
                    "configuration_id": "config_0001",
                    "vacancy_species": "O",
                    "n_vacancies": 1,
                    "relaxed_poscar_path": str(vacancy_path),
                }
            ]
        ),
        encoding="utf-8",
    )

    def make_cfg(selector):
        return parse_oxidation_config(
            {
                "structure": {"outdir": str(source_root)},
                "oxidation": {
                    "strategy": "structural",
                    "methods": ["bond-valence"],
                    "target_include": selector,
                },
            },
            tmp_path,
        )

    exact = make_cfg(["Sb25/candidate_0001"])
    assert exact.target_include == ("Sb25/candidate_0001",)
    targets, _ = discover_oxidation_targets(exact)
    assert [target.target_id for target in targets] == ["Sb25/candidate_0001"]

    safe = make_cfg("Sb25__candidate_0001")
    targets, _ = discover_oxidation_targets(safe)
    assert [target.target_id for target in targets] == ["Sb25/candidate_0001"]

    vacancy_only = make_cfg(["Sb25/candidate_0001/V_O_01/config_0001"])
    targets, _ = discover_oxidation_targets(vacancy_only)
    assert [target.target_id for target in targets] == [
        "Sb25/candidate_0001/V_O_01/config_0001"
    ]

    wildcard = make_cfg(["Sb25/candidate_0001/*"])
    targets, _ = discover_oxidation_targets(wildcard)
    assert [target.target_id for target in targets] == [
        "Sb25/candidate_0001/V_O_01/config_0001"
    ]

    missing = make_cfg(["does-not-exist"])
    with pytest.raises(RuntimeError, match="target_include matched no discovered structures"):
        discover_oxidation_targets(missing)


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
        assert method not in {"dft-auto", "dft-electronic", "bader", "wannier", "eos"}

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


def test_dft_auto_keeps_uniform_sn4_and_reports_two_delocalized_electrons() -> None:
    species = ["Sn"] * 5 + ["Sb"] * 2 + ["Ti"] + ["O"] * 16
    coords = [
        [(i % 4) / 4.0, ((i // 4) % 3) / 3.0, (i // 12) / 2.0]
        for i in range(len(species))
    ]
    structure = Structure(Lattice.cubic(20.0), species, coords)

    # A charge-neutral structural prior may be tempted to label two Sn as 3+.
    # Nearly identical Bader charges provide no DFT evidence for that site split.
    prior = [4, 4, 4, 3, 3, 5, 5, 4] + [-2] * 16
    q_sn = [2.48, 2.47, 2.49, 2.46, 2.45]
    bader = [
        {"site_index": i, "element": "Sn", "bader_partial_charge": q}
        for i, q in enumerate(q_sn)
    ]
    bader.extend(
        [
            {"site_index": 5, "element": "Sb", "bader_partial_charge": 2.85},
            {"site_index": 6, "element": "Sb", "bader_partial_charge": 2.83},
            {"site_index": 7, "element": "Ti", "bader_partial_charge": 2.25},
        ]
    )
    bader.extend(
        {
            "site_index": i,
            "element": "O",
            "bader_partial_charge": -1.24,
        }
        for i in range(8, 24)
    )

    result = oxidation_dft_auto.synthesize_oxidation_states(
        structure,
        prior_states=prior,
        bader_records=bader,
        band_edge_analysis={"homo": {"localization": "delocalized"}},
        wannier_analysis={
            "n_delocalized_spread_outliers": 1,
            "electrons_per_wf": 2.0,
            "n_wannier_centres": 100,
        },
    )
    states = [row["formal_oxidation_state"] for row in result["sites"]]
    assert states[:5] == [4, 4, 4, 4, 4]
    assert states[5:8] == [5, 5, 4]
    assert states[8:] == [-2] * 16
    assert result["formal_charge_sum_e"] == pytest.approx(2.0)
    assert result["electronic_compensation"]["charge_e"] == pytest.approx(-2.0)
    assert result["electronic_compensation"]["type"] == "electrons"
    assert result["electronic_compensation"]["localization"] == "delocalized"
    assert result["wannier_compensation_support"]["matches_electronic_compensation"] is True


def test_dft_auto_preserves_mixed_valence_when_bader_populations_separate() -> None:
    structure = Structure(
        Lattice.cubic(12.0),
        ["Fe", "Fe", "Fe", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.25, 0.25, 0.75],
        ],
    )
    result = oxidation_dft_auto.synthesize_oxidation_states(
        structure,
        prior_states=[2, 3, 3, -2, -2, -2, -2],
        bader_records=[
            {"site_index": 0, "element": "Fe", "bader_partial_charge": 1.20},
            {"site_index": 1, "element": "Fe", "bader_partial_charge": 1.52},
            {"site_index": 2, "element": "Fe", "bader_partial_charge": 1.50},
            *[
                {"site_index": i, "element": "O", "bader_partial_charge": -1.20}
                for i in range(3, 7)
            ],
        ],
    )
    assert [row["formal_oxidation_state"] for row in result["sites"][:3]] == [2, 3, 3]
    assert result["formal_charge_sum_e"] == pytest.approx(0.0)
    assert result["mixed_valence_diagnostics"]["Fe"]["supported"] is True


def test_bader_default_gridrefinement_is_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    workdir = tmp_path / "bader-default-grid"
    workdir.mkdir()
    (workdir / "oxidation.gpw").write_text("stub", encoding="utf-8")
    captured: dict[str, int] = {}

    def fake_density(gpw_path, density_path, *, gridrefinement):
        del gpw_path
        captured["gridrefinement"] = gridrefinement
        density_path.write_text("stub density", encoding="utf-8")

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(tokens, *, cwd, check, text, capture_output):
        del tokens, check, text, capture_output
        Path(cwd, "ACF.dat").write_text(
            "# X Y Z CHARGE MIN_DIST ATOMIC_VOL\n"
            "---------------------------------------------\n"
            "1 0.0 0.0 0.0 49.60 0.50 10.0\n"
            "2 2.5 2.5 2.5 8.30 0.50 11.0\n",
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(oxidation_dft, "_write_gpaw_all_electron_density", fake_density)
    monkeypatch.setattr(oxidation_dft.subprocess, "run", fake_run)
    result = oxidation_dft.run_dft_method(
        "bader",
        target,
        cfg,
        {"workdir": str(workdir), "execute": True},
    )
    assert captured["gridrefinement"] == 2
    assert result["assignment_status"] == "descriptors-only"



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
    class Cell:
        array = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]

    class AtomsLike:
        cell = Cell()

        def get_chemical_symbols(self):
            return ["Sn", "O", "O"]

        def get_scaled_positions(self, wrap=False):
            assert wrap is False
            return [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]]

    atoms = AtomsLike()
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

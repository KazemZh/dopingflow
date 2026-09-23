import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from pymatgen.core import Lattice, Structure

from dopingflow import conductivity as c
from dopingflow import dft_cache as cache
from dopingflow import oxidation_dft as dft
from dopingflow.oxidation import OptionalMethodUnavailable, StructureTarget, parse_oxidation_config


def parent(root, name, energy, *, oxygen=2):
    structure = Structure(
        Lattice.cubic(5),
        ["Sn"] + ["O"] * oxygen,
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]][: 1 + oxygen],
    )
    candidate = root / "SnO2" / name
    scan = candidate / "01_scan" / "POSCAR"
    path = candidate / "02_relax" / "POSCAR"
    scan.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(scan), fmt="poscar")
    structure.to(filename=str(path), fmt="poscar")
    (path.parent / "meta.json").write_text(
        json.dumps({"energy_relaxed_eV": energy, "backend": "mace", "model": "test"})
    )
    selected = root / "SnO2" / "selected_candidates.txt"
    names = selected.read_text().splitlines() if selected.exists() else []
    if name not in names:
        selected.write_text("\n".join([*names, name]) + "\n")
    return path


def composition_parent(root, composition, name, energy):
    structure = Structure(
        Lattice.cubic(5),
        ["Sn", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    candidate = root / composition / name
    scan = candidate / "01_scan" / "POSCAR"
    path = candidate / "02_relax" / "POSCAR"
    scan.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(scan), fmt="poscar")
    structure.to(filename=str(path), fmt="poscar")
    (path.parent / "meta.json").write_text(
        json.dumps({"energy_relaxed_eV": energy, "backend": "mace", "model": "test"})
    )
    selected = root / composition / "selected_candidates.txt"
    names = selected.read_text().splitlines() if selected.exists() else []
    if name not in names:
        selected.write_text("\n".join([*names, name]) + "\n")
    return path


def test_selection_matches_oxidation_source_and_target_semantics(tmp_path):
    root = tmp_path / "structures"
    parent(root, "c1", -20)
    parent(root, "c2", -21)
    parent(root, "c3", -100, oxygen=1)
    raw = {"structure": {"outdir": str(root)}, "conductivity": {"enabled": True}}
    cfg, settings = c.parse_config(raw, tmp_path)

    selected, _ = c.select_targets(cfg, settings)
    assert {t.target_id for t in selected} == {"SnO2/c1", "SnO2/c2", "SnO2/c3"}

    cfg, settings = c.parse_config(
        {
            "structure": {"outdir": str(root)},
            "conductivity": {"enabled": True, "target_include": ["SnO2/c1"]},
        },
        tmp_path,
    )
    selected, _ = c.select_targets(cfg, settings)
    assert [t.target_id for t in selected] == ["SnO2/c1"]


def test_vacancy_selection_matches_oxidation_rules(tmp_path):
    root = tmp_path / "structures"
    p1 = parent(root, "c1", -10)
    rows = [
        {
            "parent_id": "SnO2/c1",
            "configuration_id": f"v{i}",
            "n_vacancies": 1,
            "vacancy_species": "O",
            "relaxed_poscar_path": str(p),
            "energy_relaxed_total_eV": e,
            "backend": "mace",
            "model": "test",
        }
        for i, p, e in [(1, p1, -10), (2, p1, -12)]
    ]
    (root / "vacancies_database.json").write_text(json.dumps(rows))

    cfg, settings = c.parse_config(
        {
            "structure": {"outdir": str(root)},
            "conductivity": {
                "include_vacancy_free": False,
                "include_oxygen_vacancies": True,
            },
        },
        tmp_path,
    )
    chosen, _ = c.select_targets(cfg, settings)
    assert {t.target_id for t in chosen} == {
        "SnO2/c1/V_O_01/v1",
        "SnO2/c1/V_O_01/v2",
    }

    cfg, settings = c.parse_config(
        {
            "structure": {"outdir": str(root)},
            "conductivity": {
                "include_vacancy_free": False,
                "include_oxygen_vacancies": True,
                "target_include": ["*/V_O_01/v2"],
            },
        },
        tmp_path,
    )
    chosen, _ = c.select_targets(cfg, settings)
    assert [t.target_id for t in chosen] == ["SnO2/c1/V_O_01/v2"]


def cache_setup(tmp_path, monkeypatch):
    path = parent(tmp_path / "structures", "c1", -20)
    target = StructureTarget("SnO2/c1", "SnO2/c1", "vacancy-free", path, 0, None)
    cfg = parse_oxidation_config({"structure": {"outdir": str(tmp_path / "structures")}}, tmp_path)
    calls = []

    def run(target, cfg, settings):
        destination = dft._workdir(target, cfg, settings) / "oxidation.gpw"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"validated mock GPAW result")
        calls.append(settings.copy())
        return destination

    monkeypatch.setattr(dft, "_run_gpaw_single_point", run)
    return cfg, target, calls


@pytest.mark.parametrize(
    "first,second", [("dft_oxidation", "dft_conductivity"), ("dft_conductivity", "dft_oxidation")]
)
def test_bidirectional_cache(tmp_path, monkeypatch, first, second):
    cfg, target, calls = cache_setup(tmp_path, monkeypatch)
    settings = {
        "execute": True,
        "kpts": [4, 4, 4],
        "output_root": first,
        "save_wavefunctions": True,
    }
    _, reused = cache.ensure_gpaw(target, cfg, settings)
    assert not reused
    path, reused = cache.ensure_gpaw(
        target, cfg, {**settings, "output_root": second, "execute": False}
    )
    assert reused and path.exists() and len(calls) == 1
    # A weaker wavefunction requirement can use a stronger result.
    assert cache.ensure_gpaw(target, cfg, {**settings, "save_wavefunctions": False})[1]


@pytest.mark.parametrize(
    "change",
    [
        {"kpts": [6, 6, 6]},
        {"xc": "PBE0"},
        {"charge": 1},
        {"ecut_eV": 700},
        {"spinpol": True},
        {"nbands": 30},
    ],
)
def test_cache_invalidates_physical_settings(tmp_path, monkeypatch, change):
    cfg, target, calls = cache_setup(tmp_path, monkeypatch)
    cache.ensure_gpaw(target, cfg, {"execute": True})
    with pytest.raises(OptionalMethodUnavailable):
        cache.ensure_gpaw(target, cfg, {"execute": False, **change})
    assert len(calls) == 1


def test_cache_invalidates_geometry_and_missing_wavefunctions(tmp_path, monkeypatch):
    cfg, target, calls = cache_setup(tmp_path, monkeypatch)
    cache.ensure_gpaw(target, cfg, {"execute": True})
    with pytest.raises(OptionalMethodUnavailable):
        cache.ensure_gpaw(target, cfg, {"save_wavefunctions": True})
    structure = Structure.from_file(target.structure_path)
    structure.translate_sites([0], [0.1, 0, 0])
    structure.to(filename=str(target.structure_path), fmt="poscar")
    with pytest.raises(OptionalMethodUnavailable):
        cache.ensure_gpaw(target, cfg, {})
    assert len(calls) == 1


def test_truncated_cache_not_reused(tmp_path, monkeypatch):
    cfg, target, _ = cache_setup(tmp_path, monkeypatch)
    path, _ = cache.ensure_gpaw(target, cfg, {"execute": True})
    path.write_bytes(b"")
    # A valid shared copy is still usable; damage it as well.
    for shared in (cfg.source_root / "dft_cache").glob("*/result.gpw"):
        shared.write_bytes(b"")
    with pytest.raises(OptionalMethodUnavailable):
        cache.ensure_gpaw(target, cfg, {})


def test_disabled_and_dry_run_never_execute(tmp_path, monkeypatch):
    assert c.run_conductivity({}, tmp_path) is None
    root = tmp_path / "structures"
    parent(root, "c1", -10)
    monkeypatch.setattr(cache, "ensure_gpaw", lambda *a: pytest.fail("DFT called in dry run"))
    output = c.run_conductivity(
        {"structure": {"outdir": str(root)}, "conductivity": {"enabled": True}},
        tmp_path,
        dry_run=True,
    )
    payload = json.loads(output.read_text())
    assert payload["results"][0]["status"] == "selected"

    result_root = root / "07_conductivity"
    index = json.loads((result_root / "conductivity_structure_index.json").read_text())
    assert index[0]["target_id"] == "SnO2/c1"
    assert index[0]["status"] == "selected"

    per_target = result_root / "structures" / "SnO2" / "c1"
    assert (per_target / "conductivity.json").exists()
    assert (per_target / "summary.json").exists()


@pytest.mark.parametrize(
    "section",
    [
        {"temperatures_K": [0]},
        {"temperatures_K": [float("nan")]},
        {"relaxation_time_fs": -1},
        {"dft": {"kpts": [1, 1, 1]}},
        {"interpolation_factor": 1},
        {"dos_points": 99},
        {"comparison": {"basis": "unknown"}},
        {"comparison": {"reference_sb_percent": 0}},
    ],
)
def test_bad_settings(tmp_path, section):
    with pytest.raises(ValueError):
        c.parse_config({"conductivity": section}, tmp_path)


def test_spin_channels_are_both_loaded(monkeypatch):
    pytest.importorskip("BoltzTraP2")
    atoms = Atoms("Fe", cell=np.eye(3) * 3, pbc=True)
    calc = SimpleNamespace(
        get_ibz_k_points=lambda: [[0, 0, 0], [0.5, 0, 0]],
        get_number_of_spins=lambda: 2,
        get_eigenvalues=lambda kpt, spin: np.array([spin * 10 + kpt, spin * 10 + 2 + kpt]),
        get_magnetic_moments=lambda: [2.0],
        get_number_of_electrons=lambda: 8,
        get_fermi_level=lambda: 1,
    )
    monkeypatch.setattr(dft, "_gpaw_restart", lambda p: (atoms, calc))
    data = c.gpaw_bands(Path("unused.gpw"))
    assert data.ebands.shape == (4, 2)
    assert data.dosweight == 1
    assert not np.array_equal(data.ebands[:2], data.ebands[2:])


def test_mu_solver_refines_upstream_count_residual():
    class FakeBandlib:
        @staticmethod
        def solve_for_mu(energy, dos, electrons, temperature, dosweight=2.0, refine=False):
            return 4.9

        @staticmethod
        def calc_N(energy, dos, mu, temperature, dosweight=2.0):
            return -float(mu)

    energy = np.linspace(0.0, 10.0, 101)
    dos = np.ones_like(energy)
    mu, residual, tolerance = c._solve_mu_for_count(
        FakeBandlib, energy, dos, 5.0, 300.0, 2.0
    )
    assert mu == pytest.approx(5.0, abs=1e-10)
    assert abs(residual) <= tolerance


def test_mu_solver_reports_band_range_problem_not_dos_grid():
    class FakeBandlib:
        @staticmethod
        def solve_for_mu(energy, dos, electrons, temperature, dosweight=2.0, refine=False):
            return 1.0

        @staticmethod
        def calc_N(energy, dos, mu, temperature, dosweight=2.0):
            # At most two electrons are represented by the sampled bands.
            return -min(float(mu), 2.0)

    energy = np.linspace(0.0, 2.0, 21)
    dos = np.ones_like(energy)
    with pytest.raises(ValueError, match="increase nbands / energy range rather than dos_points"):
        c._solve_mu_for_count(FakeBandlib, energy, dos, 5.0, 300.0, 2.0)

def test_real_boltztrap_parabolic_band():
    """Independent Drude limit catches atomic/SI units, volume and spin factors."""
    pytest.importorskip("BoltzTraP2")
    from BoltzTraP2.units import Angstrom
    from scipy.constants import electron_mass, elementary_charge

    ngrid = 15
    import spglib

    atoms = Atoms("Si", cell=np.eye(3) * 5, pbc=True)
    mapping, grid = spglib.get_ir_reciprocal_mesh(
        [ngrid] * 3, (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    )
    mesh = grid[np.unique(mapping)] / ngrid
    lattice = np.array(atoms.cell).T * Angstrom
    kcart = mesh @ (2 * np.pi * np.linalg.inv(lattice))
    energy = (kcart * kcart).sum(axis=1) / 2
    data = SimpleNamespace(
        atoms=atoms,
        kpoints=mesh,
        ebands=np.stack([np.full_like(energy, -1.0), energy]),
        mommat=None,
        magmom=None,
        get_lattvec=lambda: lattice,
        nelect=2.0125,
        dosweight=2,
        fermi=0,
    )
    settings = {
        "interpolation_factor": 2,
        "dos_points": 6000,
        "temperatures_K": [300.0],
        "excess_electrons_cm3": [0.0],
        "relaxation_time_fs": 10.0,
    }
    rows = c.integrate_transport(data, settings)
    tensor = np.array(rows[0]["sigma_over_tau_S_per_m_per_s"])
    # 0.0125 e / 125 Angstrom^3 = 1e20 cm^-3.
    expected = 1e26 * elementary_charge**2 / electron_mass
    assert np.trace(tensor) / 3 == pytest.approx(expected, rel=0.15)
    assert rows[0]["sigma_over_tau_trace_average_S_per_cm_per_fs"] == pytest.approx(
        expected * 1e-17, rel=0.15
    )
    assert np.asarray(rows[0]["sigma_over_tau_S_per_cm_per_fs"]) == pytest.approx(
        tensor * 1e-17, rel=1e-12
    )
    assert rows[0]["conditional_sigma_trace_average_S_per_cm"] == pytest.approx(
        expected * 1e-14 / 100, rel=0.15
    )
    assert np.diag(tensor).max() / np.diag(tensor).min() < 1.05


def _ato_reference(value=250.0):
    return {
        "target_id": "Sb5/candidate_001",
        "kind": "vacancy-free",
        "n_oxygen_vacancies": 0,
        "status": "calculated",
        "reference_label": "ATO 5% Sb",
        "reference_sb_percent": 5.0,
        "rows": [
            {
                "temperature_K": 300.0,
                "excess_electrons_cm3": 0.0,
                "sigma_over_tau_trace_average_S_per_cm_per_fs": value,
            },
            {
                "temperature_K": 353.0,
                "excess_electrons_cm3": 0.0,
                "sigma_over_tau_trace_average_S_per_cm_per_fs": value * 0.8,
            },
        ],
    }


def test_reference_comparison_uses_persistent_ato_reference():
    results = [
        {
            "target_id": "Ce2p5_Sb2p5/candidate_013",
            "kind": "vacancy-free",
            "n_oxygen_vacancies": 0,
            "status": "calculated",
            "rows": [
                {
                    "temperature_K": 300.0,
                    "excess_electrons_cm3": 0.0,
                    "sigma_over_tau_trace_average_S_per_cm_per_fs": 282.0,
                },
                {
                    "temperature_K": 353.0,
                    "excess_electrons_cm3": 0.0,
                    "sigma_over_tau_trace_average_S_per_cm_per_fs": 190.0,
                },
            ],
        }
    ]
    rows, warnings = c.build_reference_comparison(
        results,
        {
            "enabled": True,
            "reference_label": "ATO 5% Sb",
            "reference_sb_percent": 5.0,
            "basis": "ato-5pct-sb-benchmark",
        },
        _ato_reference(),
    )
    assert not warnings
    by_temp = {row["temperature_K"]: row for row in rows}
    ce_300 = by_temp[300.0]
    assert ce_300["reference_target_id"] == "Sb5/candidate_001"
    assert ce_300["relative_to_reference"] == pytest.approx(282.0 / 250.0)
    assert ce_300["percent_change_vs_reference"] == pytest.approx(12.8)
    assert ce_300["comparison_basis"] == "ato-5pct-sb-benchmark"
    assert ce_300["reference_sb_percent"] == pytest.approx(5.0)


def test_reference_comparison_applies_same_ato_to_vacancy_structures():
    results = [
        {
            "target_id": "Ce2p5_Sb2p5/candidate_013/V_O_01/v1",
            "kind": "oxygen-vacancy",
            "n_oxygen_vacancies": 1,
            "status": "calculated",
            "rows": [
                {
                    "temperature_K": 300.0,
                    "excess_electrons_cm3": 0.0,
                    "sigma_over_tau_trace_average_S_per_cm_per_fs": 300.0,
                }
            ],
        }
    ]
    rows, warnings = c.build_reference_comparison(
        results,
        {
            "enabled": True,
            "reference_label": "ATO 5% Sb",
            "reference_sb_percent": 5.0,
            "basis": "ato-5pct-sb-benchmark",
        },
        _ato_reference(),
    )
    assert not warnings
    assert len(rows) == 1
    assert rows[0]["n_oxygen_vacancies"] == 1
    assert rows[0]["reference_n_oxygen_vacancies"] == 0
    assert rows[0]["relative_to_reference"] == pytest.approx(1.2)


def test_reference_comparison_requires_available_persistent_reference():
    rows, warnings = c.build_reference_comparison(
        [],
        {"enabled": True},
        None,
    )
    assert rows == []
    assert warnings and "not available" in warnings[0]


def test_collect_compatible_results_accumulates_prior_runs(tmp_path):
    output = tmp_path / "07_conductivity"
    first = output / "structures" / "Ce2p5_Sb2p5" / "candidate_013"
    second = output / "structures" / "Ti2p5_Sb2p5" / "candidate_004"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    (first / "conductivity.json").write_text(
        json.dumps(
            {
                "target_id": "Ce2p5_Sb2p5/candidate_013",
                "status": "calculated",
                "transport_settings_fingerprint": "same",
                "rows": [],
            }
        )
    )
    (second / "conductivity.json").write_text(
        json.dumps(
            {
                "target_id": "Ti2p5_Sb2p5/candidate_004",
                "status": "calculated",
                "transport_settings_fingerprint": "different",
                "rows": [],
            }
        )
    )
    current = [
        {
            "target_id": "Nb2p5_Sb2p5/candidate_002",
            "status": "calculated",
            "transport_settings_fingerprint": "same",
            "rows": [],
        }
    ]
    records = c.collect_compatible_transport_results(output, "same", current)
    assert {record["target_id"] for record in records} == {
        "Ce2p5_Sb2p5/candidate_013",
        "Nb2p5_Sb2p5/candidate_002",
    }


def test_reference_auto_searches_structure_outdir_before_conductivity_source(tmp_path):
    normal_root = tmp_path / "random_structures"
    vacancy_root = tmp_path / "vacancy-selected"
    composition_parent(normal_root, "Sb5", "candidate_001", -20)
    parent(vacancy_root, "codoped", -10)

    raw = {
        "structure": {"outdir": str(normal_root)},
        "conductivity": {
            "enabled": True,
            "source_root": str(vacancy_root),
            "comparison": {
                "enabled": True,
                "reference_target": "Sb5/*",
                "reference_label": "ATO 5% Sb",
                "reference_sb_percent": 5.0,
            },
        },
    }
    cfg, settings = c.parse_config(raw, tmp_path)
    ref_cfg, candidates = c.discover_reference_candidates(
        raw, tmp_path, cfg, settings["comparison"]
    )
    assert ref_cfg.source_root == normal_root.resolve()
    assert [target.target_id for target in candidates] == ["Sb5/candidate_001"]


def test_explicit_reference_structure_can_live_outside_target_source(tmp_path):
    target_root = tmp_path / "vacancy-selected"
    parent(target_root, "codoped", -10)

    reference_root = tmp_path / "references"
    reference_path = parent(reference_root, "ato", -20)
    raw = {
        "structure": {"outdir": str(target_root)},
        "conductivity": {
            "enabled": True,
            "comparison": {
                "enabled": True,
                "reference_target": "Sb5/reference",
                "reference_structure_path": str(reference_path),
                "reference_label": "ATO 5% Sb",
                "reference_sb_percent": 5.0,
            },
        },
    }
    cfg, settings = c.parse_config(raw, tmp_path)
    _, candidates = c.discover_reference_candidates(
        raw, tmp_path, cfg, settings["comparison"]
    )
    assert len(candidates) == 1
    assert candidates[0].target_id == "Sb5/reference"
    assert candidates[0].structure_path == reference_path.resolve()


def test_reference_candidate_directory_and_trailing_glob_resolve_relaxed_poscar(tmp_path):
    root = tmp_path / "complete"
    path = composition_parent(root, "Sb5", "candidate_003", -20)
    candidate_dir = path.parent.parent

    resolved_dir = c._resolve_reference_structure_hint(str(candidate_dir), tmp_path)
    resolved_glob = c._resolve_reference_structure_hint(str(candidate_dir) + "/*", tmp_path)
    assert resolved_dir == path.resolve()
    assert resolved_glob == path.resolve()


def test_reference_target_selector_accepts_absolute_candidate_glob(tmp_path):
    target_root = tmp_path / "vacancy-selected"
    parent(target_root, "codoped", -10)
    ref_root = tmp_path / "complete"
    reference_path = composition_parent(ref_root, "Sb5", "candidate_003", -20)
    pasted = str(reference_path.parent.parent) + "/*"

    raw = {
        "structure": {"outdir": str(target_root)},
        "conductivity": {
            "enabled": True,
            "source_root": str(target_root),
            "comparison": {
                "enabled": True,
                "reference_target": pasted,
                "reference_label": "ATO 5% Sb",
            },
        },
    }
    cfg, settings = c.parse_config(raw, tmp_path)
    _, candidates = c.discover_reference_candidates(
        raw, tmp_path, cfg, settings["comparison"]
    )
    assert len(candidates) == 1
    assert candidates[0].target_id == "Sb5/candidate_003"
    assert candidates[0].structure_path == reference_path.resolve()

def test_reference_source_root_accepts_candidate_directory_hint(tmp_path):
    target_root = tmp_path / "vacancy-selected"
    parent(target_root, "codoped", -10)
    ref_root = tmp_path / "complete"
    reference_path = composition_parent(ref_root, "Sb5", "candidate_003", -20)

    raw = {
        "structure": {"outdir": str(target_root)},
        "conductivity": {
            "enabled": True,
            "source_root": str(target_root),
            "comparison": {
                "enabled": True,
                "reference_source_root": str(reference_path.parent.parent) + "/*",
                "reference_target": "Sb5/*",
            },
        },
    }
    cfg, settings = c.parse_config(raw, tmp_path)
    _, candidates = c.discover_reference_candidates(
        raw, tmp_path, cfg, settings["comparison"]
    )
    assert len(candidates) == 1
    assert candidates[0].target_id == "Sb5/candidate_003"
    assert candidates[0].structure_path == reference_path.resolve()


def test_reference_only_rebuild_does_not_rerun_screened_targets(tmp_path, monkeypatch):
    source = tmp_path / "vacancy-selected"
    source.mkdir()
    raw = {
        "structure": {"outdir": str(source)},
        "conductivity": {
            "enabled": True,
            "source_root": str(source),
            "temperatures_K": [300.0],
            "excess_electrons_cm3": [0.0],
            "interpolation_factor": 5,
            "dos_points": 4000,
            "comparison": {
                "enabled": True,
                "reference_target": "Sb5/candidate_003",
                "reference_label": "ATO 5% Sb",
                "reference_sb_percent": 5.0,
                "basis": "ato-5pct-sb-benchmark",
            },
            "dft": {"kpts": [2, 2, 2], "execute": False},
        },
    }
    cfg, section = c.parse_config(raw, tmp_path)
    fingerprint, payload = c._transport_settings_fingerprint(section)
    target_dir = cfg.output_dir / "structures" / "Ce2p5_Sb2p5" / "candidate_013"
    target_dir.mkdir(parents=True)
    (target_dir / "conductivity.json").write_text(
        json.dumps(
            {
                "target_id": "Ce2p5_Sb2p5/candidate_013",
                "kind": "vacancy-free",
                "n_oxygen_vacancies": 0,
                "status": "calculated",
                "transport_settings_fingerprint": fingerprint,
                "transport_settings": payload,
                "rows": [
                    {
                        "temperature_K": 300.0,
                        "excess_electrons_cm3": 0.0,
                        "sigma_over_tau_trace_average_S_per_cm_per_fs": 282.0,
                    }
                ],
            }
        )
    )

    reference = {
        "target_id": "Sb5/candidate_003",
        "kind": "vacancy-free",
        "n_oxygen_vacancies": 0,
        "status": "calculated",
        "reference_label": "ATO 5% Sb",
        "reference_sb_percent": 5.0,
        "rows": [
            {
                "temperature_K": 300.0,
                "excess_electrons_cm3": 0.0,
                "sigma_over_tau_trace_average_S_per_cm_per_fs": 250.0,
            }
        ],
    }
    monkeypatch.setattr(
        c, "prepare_persistent_reference", lambda *args, **kwargs: (reference, [])
    )
    monkeypatch.setattr(
        c, "select_targets", lambda *args, **kwargs: pytest.fail("screened targets were rediscovered/rerun")
    )

    output = c.rebuild_reference_comparison(raw, tmp_path)
    rows = json.loads(output.read_text())
    assert len(rows) == 1
    assert rows[0]["target_id"] == "Ce2p5_Sb2p5/candidate_013"
    assert rows[0]["relative_to_reference"] == pytest.approx(282.0 / 250.0)

def test_reference_only_rebuild_clears_stale_reference_warnings(tmp_path, monkeypatch):
    source = tmp_path / "vacancy-selected"
    source.mkdir()
    raw = {
        "structure": {"outdir": str(source)},
        "conductivity": {
            "enabled": True,
            "source_root": str(source),
            "temperatures_K": [300.0],
            "excess_electrons_cm3": [0.0],
            "interpolation_factor": 5,
            "dos_points": 4000,
            "comparison": {"enabled": True, "reference_target": "Sb5/candidate_003"},
            "dft": {"kpts": [2, 2, 2], "execute": False},
        },
    }
    cfg, section = c.parse_config(raw, tmp_path)
    fingerprint, payload = c._transport_settings_fingerprint(section)
    target_dir = cfg.output_dir / "structures" / "Ce2p5_Sb2p5" / "candidate_013"
    target_dir.mkdir(parents=True)
    (target_dir / "conductivity.json").write_text(
        json.dumps({
            "target_id": "Ce2p5_Sb2p5/candidate_013",
            "status": "calculated",
            "transport_settings_fingerprint": fingerprint,
            "transport_settings": payload,
            "rows": [{"temperature_K": 300.0, "excess_electrons_cm3": 0.0,
                      "sigma_over_tau_trace_average_S_per_cm_per_fs": 282.0}],
        })
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "conductivity_results.json").write_text(json.dumps({
        "warnings": [
            "ATO 5% Sb reference unavailable: ValueError: Carrier-count integration did not converge; increase dos_points",
            "ATO 5% Sb reference conductivity is not available yet.",
            "unrelated warning",
        ],
        "results": [],
    }))
    reference = {
        "target_id": "Sb5/candidate_003",
        "kind": "vacancy-free",
        "n_oxygen_vacancies": 0,
        "status": "calculated",
        "reference_label": "ATO 5% Sb",
        "reference_sb_percent": 5.0,
        "rows": [{"temperature_K": 300.0, "excess_electrons_cm3": 0.0,
                  "sigma_over_tau_trace_average_S_per_cm_per_fs": 250.0}],
    }
    monkeypatch.setattr(c, "prepare_persistent_reference", lambda *args, **kwargs: (reference, []))

    c.rebuild_reference_comparison(raw, tmp_path)
    refreshed = json.loads((cfg.output_dir / "conductivity_results.json").read_text())
    assert refreshed["warnings"] == ["unrelated warning"]
    assert refreshed["reference"]["status"] == "calculated"

def test_default_comparison_is_5pct_sb_benchmark(tmp_path):
    raw = {
        "conductivity": {
            "comparison": {"enabled": True},
        }
    }
    _, settings = c.parse_config(raw, tmp_path)
    comparison = settings["comparison"]
    assert comparison["reference_target"] == "Sb5/*"
    assert comparison["reference_label"] == "ATO 5% Sb"
    assert comparison["reference_sb_percent"] == pytest.approx(5.0)
    assert comparison["basis"] == "ato-5pct-sb-benchmark"


def test_stale_legacy_metadata_cannot_override_modern_manifest(tmp_path, monkeypatch):
    cfg, target, calls = cache_setup(tmp_path, monkeypatch)
    path, _ = cache.ensure_gpaw(target, cfg, {"execute": True})
    (path.parent / "gpaw_run_metadata.json").write_text(json.dumps({"settings": {}}))
    cache.ensure_gpaw(target, cfg, {"execute": True, "xc": "PBE0"})
    # Remove the old compatible cache, forcing examination of the changed local file.
    oldkey, _ = cache.calculation_key(target, {})
    (cfg.source_root / "dft_cache" / oldkey / "result.gpw").unlink()
    with pytest.raises(OptionalMethodUnavailable):
        cache.ensure_gpaw(target, cfg, {})
    assert len(calls) == 2


def test_derived_results_follow_source_gpw(tmp_path):
    gpw = tmp_path / "a.gpw"
    acf = tmp_path / "ACF.dat"
    gpw.write_bytes(b"first DFT")
    acf.write_text("charges")
    assert not cache.derived_current(acf, gpw)
    cache.record_derived(acf, gpw)
    assert cache.derived_current(acf, gpw)
    gpw.write_bytes(b"different DFT")
    assert not cache.derived_current(acf, gpw)


def test_custom_source_root_is_shared_with_oxidation_selection(tmp_path):
    default_root = tmp_path / "default"
    custom_root = tmp_path / "vacancy-selected"
    parent(default_root, "default_candidate", -2)
    parent(custom_root, "chosen_candidate", -1)

    cfg, settings = c.parse_config(
        {
            "structure": {"outdir": str(default_root)},
            "conductivity": {"source_root": str(custom_root)},
        },
        tmp_path,
    )
    chosen, _ = c.select_targets(cfg, settings)
    assert [t.target_id for t in chosen] == ["SnO2/chosen_candidate"]


def test_source_root_defaults_to_oxidation_source_root(tmp_path):
    default_root = tmp_path / "default"
    oxidation_root = tmp_path / "vacancy-selected"
    parent(default_root, "default_candidate", -2)
    parent(oxidation_root, "oxidation_candidate", -1)

    cfg, settings = c.parse_config(
        {
            "structure": {"outdir": str(default_root)},
            "oxidation": {"source_root": str(oxidation_root)},
            "conductivity": {},
        },
        tmp_path,
    )
    chosen, _ = c.select_targets(cfg, settings)
    assert cfg.source_root == oxidation_root.resolve()
    assert [t.target_id for t in chosen] == ["SnO2/oxidation_candidate"]


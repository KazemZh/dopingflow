import pandas as pd
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from pathlib import Path

from dopingflow import surface_pourbaix as pb
from dopingflow import surface_staged as sf
from dopingflow.surface_pourbaix_thermo import KB_EV_K, LN10, surface_state_delta_g_eV


def test_neutral_co_and_che_hydrogen():
    # Slab + 1 H + 1 CO, E_ads=-18, E_clean=-4, E_H2=-6, mu_CO=-9.
    def energy(h, u, ph, scale):
        return surface_state_delta_g_eV(-18, -4, h, 1, -6, -12, u, ph, n_co=1,
                                        co_energy_eV=-9, potential_scale=scale)
    assert energy(0, -1, 0, "SHE") == energy(0, 1, 14, "SHE") == -5
    assert energy(1, 1, 0, "SHE") == -1
    assert energy(1, 1, 14, "SHE") == pytest.approx(-1 + 14*KB_EV_K*298.15*LN10)
    assert energy(1, 1, 14, "RHE") == pytest.approx(-1)


def test_bulk_pd_to_two_coverage_grids(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/surface_pourbaix/pd_h_co.toml"
    cfg = sf.tomllib.loads(path.read_text())
    write(tmp_path / "Pd.vasp", bulk("Pd"))
    cfg["surface"]["screen"]["relax"] = False
    cfg["surface_pourbaix"].update(reference_relax=False, h_coverages_pct=[25],
        co_coverages_pct=[25], mixed_h_co_coverages_pct=[[25, 25]],
        max_arrangements_per_stoichiometry=2, potential_step_V=1.0, pH_step=7.0)
    monkeypatch.setattr(sf, "_prepare_calculator", lambda *args: EMT())
    monkeypatch.setattr(pb, "_prepare_calculator", lambda *args: EMT())
    sf.run_surface_scan(cfg, tmp_path)
    grid_path = pb.run_surface_pourbaix(cfg, tmp_path)
    grid = pd.read_csv(grid_path)
    states = pd.read_csv(grid_path.parent / "surface_state_summary.csv")
    assert len(grid) == 2 * 3 * 3
    assert states.groupby("surface_id").family.nunique().tolist() == [4, 4]
    assert set(states.eligible_adsorbate_cation_sites) == {4}
    mixed = states[states.family == "mixed-H-CO"]
    assert set(mixed.actual_co_coverage_pct) == set(mixed.actual_h_coverage_pct) == {25}
    assert set(mixed.proton_electron_pairs) == set(mixed.n_CO) == {1}
    assert set(states.miller_l) == {0, 1}

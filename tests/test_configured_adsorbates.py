from pathlib import Path

import pandas as pd
import pytest
from ase.build import bulk, fcc111, surface
from ase.calculators.lj import LennardJones
from ase.io import write
from pymatgen.core import Structure

from dopingflow import surface_pourbaix as pb
from dopingflow.surface_pourbaix_adsorbates import parse_adsorbates
from dopingflow.surface_pourbaix_postprocess import analyze_relaxed_surface_state


def test_molecular_states_on_metal_and_oxide(tmp_path, monkeypatch):
    example = (
        Path(__file__).resolve().parents[1] / "examples/surface_pourbaix/molecular_adsorbates.toml"
    )
    raw = pb.tomllib.loads(example.read_text())
    section = raw["surface_pourbaix"]
    section.update(
        slab_files=["copper.vasp", "oxide.vasp"],
        state_families=["clean"],
        max_arrangements_per_stoichiometry=2,
        pH_step=14.0,
        potential_step_V=2.0,
        reference_relax=False,
    )
    section["screen"]["relax"] = False
    section["adsorbates"]["CO"].update(binding_species=["Cu"], coverages_pct=[25])
    section["adsorbates"]["NH3"].update(binding_species=["O"], coverages_pct=[25])
    (tmp_path / "NH2.xyz").write_text("3\nNH2\nN 0 0 0\nH 0 .8 .5\nH 0 -.8 .5\n")
    section["adsorbates"]["amide"] = dict(
        xyz="NH2.xyz",
        reference={"molecule": "NH3"},
        binding_species=["Mg"],
        coverages_pct=[25],
        activity=0.01,
    )
    write(tmp_path / "copper.vasp", fcc111("Cu", size=(2, 2, 4), vacuum=8, periodic=True))
    oxide = surface(
        bulk("MgO", "rocksalt", a=4.2, cubic=True), (1, 0, 0), 3, vacuum=8, periodic=True
    )
    write(tmp_path / "oxide.vasp", oxide)
    monkeypatch.setattr(pb, "_prepare_calculator", lambda *a: LennardJones())
    grid_path = pb.run_surface_pourbaix(raw, tmp_path)
    grid = pd.read_csv(grid_path)
    states = pd.read_csv(grid_path.parent / "surface_state_summary.csv")
    eligible = states[states.pourbaix_eligible]
    assert set(eligible[eligible.surface_id == "copper"].family) == {"clean", "ads:CO"}
    assert set(eligible[eligible.surface_id == "oxide"].family) == {"clean", "ads:NH3", "ads:amide"}
    assert set(states[states.family == "ads:NH3"].proton_electron_pairs) == {0}
    amide = states[states.family == "ads:amide"].iloc[0]
    assert amide.proton_electron_pairs == -amide.molecule_count
    assert len(grid) == 8
    # A detached molecule must not compete even if its composition is preserved.
    cfg = pb.parse_surface_pourbaix_config(raw, tmp_path)
    slab = Structure.from_file(tmp_path / "oxide.vasp")
    state = next(s for s in pb.enumerate_surface_states(slab, cfg) if s["family"] == "ads:amide")
    moved = state["structure"].copy()
    moved.translate_sites(range(len(slab), len(moved)), [0, 0, 5], frac_coords=False)
    result = analyze_relaxed_surface_state(slab, state["structure"], moved, state, cfg)
    assert not result["pourbaix_eligible"]
    assert "detached" in result["postprocess_reason"]


def test_unbalanced_molecular_reservoir(tmp_path):
    with pytest.raises(ValueError, match="balance"):
        parse_adsorbates({"bad": {"molecule": "CO", "reference": {"molecule": "CO2"}}}, tmp_path)


def test_configured_thermodynamics():
    states = [
        dict(
            state_id="clean",
            family="clean",
            ml=0,
            delta_n_H=0,
            delta_n_O=0,
            proton_electron_pairs=0,
        ),
        dict(
            state_id="nh2",
            family="ads:NH2",
            ml=-14,
            delta_n_H=2,
            delta_n_O=0,
            molecule_count=1,
            adsorbate_mu_eV_ml=-10,
            proton_electron_pairs=-1,
        ),
    ]
    grid, _ = pb.build_surface_pourbaix_grid(
        states,
        h2_energy_eV=-6,
        h2o_energy_eV=0,
        potential_values=[0, 1],
        pH_values=[0],
        potential_scale="SHE",
        temperature_K=298,
        energy_key="ml",
        energy_level="ML",
    )
    assert list(grid.deltaG_stable_eV) == [-7, -8]

import json

from pymatgen.core import Lattice, Structure

import dopingflow.refs as refs_module


def _structure() -> Structure:
    return Structure(
        Lattice.cubic(4.0),
        ["Sn", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )


def _config(*, oxides: list[str], fmax: float = 0.02) -> dict:
    return {
        "references": {
            "reference_mode": "metal",
            "skip_if_done": True,
            "host": "SnO2",
            "host_dir": "reference_structures/oxides",
            "supercell": [1, 1, 1],
            "metal_ref": ["Sn"],
            "metals_dir": "reference_structures/metals",
            "oxides_ref": oxides,
            "oxides_dir": "reference_structures/oxides",
            "gas_ref": "O2",
            "gas_dir": "reference_structures/gas",
            "backend": "m3gnet",
            "model": "default",
            "optimizer": "bfgs",
            "fmax": fmax,
            "max_steps": 10,
        }
    }


def test_metal_mode_relaxes_oxide_references_and_reuses_cache(
    tmp_path, monkeypatch
):
    files = [
        "reference_structures/oxides/SnO2.POSCAR",
        "reference_structures/metals/Sn.POSCAR",
        "reference_structures/oxides/SnO.POSCAR",
        "reference_structures/gas/O2.POSCAR",
    ]
    for relative in files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    calls: list[float] = []
    monkeypatch.setattr(
        refs_module,
        "check_backend_dependency",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(refs_module, "_read_poscar", lambda path: _structure())

    def fake_relax(structure, cfg):
        calls.append(cfg.fmax)
        return structure.copy(), -float(len(structure)), 1, cfg.fmax / 2, True

    monkeypatch.setattr(refs_module, "_relax_structure_and_energy", fake_relax)

    output = refs_module.run_refs_build(_config(oxides=["SnO"]), tmp_path)
    first_call_count = len(calls)
    assert first_call_count == 5  # host unit + host supercell + metal + oxide + gas

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["reference_mode"] == "metal"
    assert data["references"]["SnO"]["type"] == "oxide"
    assert data["references"]["O2"]["type"] == "gas"

    refs_module.run_refs_build(_config(oxides=["SnO"]), tmp_path)
    assert len(calls) == first_call_count

    relaxed_oxide = (
        tmp_path / "reference_structures/relaxed/refs/SnO_relaxed.POSCAR"
    )
    relaxed_oxide.write_text("tampered", encoding="utf-8")
    refs_module.run_refs_build(_config(oxides=["SnO"]), tmp_path)
    assert len(calls) == first_call_count + 1

    new_oxide = tmp_path / "reference_structures/oxides/Sn3O4.POSCAR"
    new_oxide.write_text("new oxide", encoding="utf-8")
    refs_module.run_refs_build(_config(oxides=["SnO", "Sn3O4"]), tmp_path)
    assert len(calls) == first_call_count + 2

    updated = json.loads(output.read_text(encoding="utf-8"))
    assert set(updated["references"]) == {"Sn", "SnO", "Sn3O4", "O2"}


def test_changed_relaxation_settings_invalidate_host_and_reference_cache(
    tmp_path, monkeypatch
):
    for relative in [
        "reference_structures/oxides/SnO2.POSCAR",
        "reference_structures/metals/Sn.POSCAR",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    calls: list[float] = []
    monkeypatch.setattr(
        refs_module,
        "check_backend_dependency",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(refs_module, "_read_poscar", lambda path: _structure())

    def fake_relax(structure, cfg):
        calls.append(cfg.fmax)
        return structure.copy(), -float(len(structure)), 1, cfg.fmax / 2, True

    monkeypatch.setattr(refs_module, "_relax_structure_and_energy", fake_relax)

    refs_module.run_refs_build(_config(oxides=[]), tmp_path)
    assert calls == [0.02, 0.02, 0.02]

    refs_module.run_refs_build(_config(oxides=[], fmax=0.03), tmp_path)
    assert calls == [0.02, 0.02, 0.02, 0.03, 0.03, 0.03]

from pathlib import Path

p = Path("src/dopingflow/oxidation_dft.py")
text = p.read_text(encoding="utf-8")
old = '''    return {
        "n_bands_total": int(calc.get_number_of_bands()),
        "n_occupied_bands": noccupied,
        "n_spins": nspins,
        "n_electrons": float(calc.get_number_of_electrons()),
'''
new = '''    try:
        n_electrons = float(calc.get_number_of_electrons())
    except Exception:
        n_electrons = None

    return {
        "n_bands_total": int(calc.get_number_of_bands()),
        "n_occupied_bands": noccupied,
        "n_spins": nspins,
        "n_electrons": n_electrons,
'''
if old not in text:
    raise SystemExit("Expected n_electrons return block not found")
p.write_text(text.replace(old, new, 1), encoding="utf-8")

p = Path("src/dopingflow/wannier_analysis.py")
text = p.read_text(encoding="utf-8")
old = '''                "center_index": center_index,
                "cartesian_angstrom": cart_values,
'''
new = '''                "center_index": center_index,
                "wannier_number": center_index + 1,
                "cartesian_angstrom": cart_values,
'''
if old not in text:
    raise SystemExit("Expected center record block not found")
p.write_text(text.replace(old, new, 1), encoding="utf-8")

# Keep the permanent oxidation test suite free of an ASE-only test dependency.
p = Path("tests/test_oxidation.py")
text = p.read_text(encoding="utf-8")
old = '''def test_native_wannier_input_uses_bloch_phases_and_xyz(tmp_path: Path) -> None:
    from ase import Atoms

    atoms = Atoms(
        "SnO2",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    path = tmp_path / "wannier90.win"
'''
new = '''def test_native_wannier_input_uses_bloch_phases_and_xyz(tmp_path: Path) -> None:
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
'''
if old not in text:
    raise SystemExit("Expected ASE-dependent Wannier input test block not found")
p.write_text(text.replace(old, new, 1), encoding="utf-8")

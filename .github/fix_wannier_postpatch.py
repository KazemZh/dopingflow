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

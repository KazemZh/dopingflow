from pathlib import Path
from dopingflow.generate import run_generate

def test_generate_minimal(tmp_path):
    # Minimal fake input config
    raw = {
        "structure": {
            "base_poscar": "POSCAR",
            "supercell": [1, 1, 1],
            "outdir": "out",
        },
        "generate": {
            "poscar_order": ["Sn", "O"],
            "seed_base": 0,
            "clean_outdir": True,
        },
        "doping": {
            "mode": "explicit",
            "host_species": "Sn",
            "compositions": [{"Sb": 5.0}],
        },
    }

    # Write minimal POSCAR
    poscar = tmp_path / "POSCAR"
    poscar.write_text("""\
SnO2
1.0
1 0 0
0 1 0
0 0 1
Sn O
1 2
Direct
0 0 0
0.5 0.5 0
0.5 0 0.5
""")

    run_generate(raw, tmp_path)

    assert (tmp_path / "out").exists()

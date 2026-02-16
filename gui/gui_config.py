# gui/gui_config.py

SUPER_CELL_PRESETS = {
    "Smoke test (3,1,1)": (3, 1, 1),
    "Medium (4,1,1)": (4, 1, 1),
    "Larger (5,1,1)": (5, 1, 1),
}

DOPING_MODE_CHOICES = ["explicit"]  # extend later if you add modes

# Keep these as *allowed dopants* for dropdowns
ALLOWED_DOPANTS = ["Sb", "Ti", "Zr", "Nb", "Zn", "Ni", "Mn", "Ba", "W"]

# Typical run presets
RUN_PRESETS = {
    "Smoke test (refs → filter)": {"from": "refs", "until": "filter"},
    "Full workflow (refs → collect)": {"from": "refs", "until": "collect"},
    "Scan → Relax": {"from": "scan", "until": "relax"},
    "Filter only (recompute)": {"from": "filter", "until": "filter"},
}

# gui/gui_config.py

STEP_KEYS = ["refs", "generate", "scan", "relax", "filter", "bandgap", "formation", "collect"]

CHOICES = {
    "references.source": ["local", "mp"],
    "doping.mode": ["explicit", "enumerate"],
    "filter.mode": ["window", "topn"],
    "formation.normalize": ["total", "per_dopant", "per_host"],
}

DEFAULTS = {
    "structure": {
        "base_poscar": "reference_structures/base.POSCAR",
        "supercell": [3, 1, 1],
        "outdir": "random_structures",
    },
    "references": {
        "source": "local",
        "bulk_dir": "reference_structures/",
        "fmax": 0.02,
        "skip_if_done": True,
        # "mp_ids": {},  # optional dict
    },
    "generate": {
        "poscar_order": ["Zr", "Ti", "Sb", "Sn", "O"],
        "seed_base": 12345,
    },
    "doping": {
        "mode": "explicit",
        "host_species": "Sn",
        "compositions": [{"Sb": 5.0, "Ti": 5.0}],
        # enumerate-mode defaults:
        "must_include": ["Sb"],
        "dopants": ["Ti", "Zr", "Sb"],
        "max_dopants_total": 2,
        "allowed_totals": [5, 10, 15],
        "levels": [5, 10],
    },
    "scan": {
        "poscar_in": "POSCAR",
        "topk": 15,
        "symprec": 1e-3,
        "nproc": 12,
        "chunksize": 50,
        "max_enum": 50_000_000,
        "max_unique": 200_000,
        "anion_species": ["O"],
        "skip_if_done": True,
    },
    "relax": {
        "fmax": 0.05,
        "n_workers": 6,
        "tf_threads": 1,
        "omp_threads": 1,
        "skip_if_done": True,
    },
    "filter": {
        "mode": "window",
        "window_meV": 50.0,
        "max_candidates": 12,
        "skip_if_done": True,
    },
    "bandgap": {
        "skip_if_done": True,
        "cutoff": 8.0,
        "max_neighbors": 12,
    },
    "formation": {
        "skip_if_done": True,
        "normalize": "per_dopant",
    },
}

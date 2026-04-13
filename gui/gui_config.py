# gui/gui_config.py

SUPER_CELL_PRESETS = {
    "Smoke test (3,1,1)": (3, 1, 1),
    "Medium (4,1,1)": (4, 1, 1),
    "Larger (5,1,1)": (5, 1, 1),
}

DOPING_MODE_CHOICES = ["explicit", "enumerate"]

# Keep these as *allowed dopants* for dropdowns
ALLOWED_DOPANTS = ["Sb", "Ti", "Zr", "Nb", "Zn", "Ni", "Mn", "Ba", "W"]

# Typical run presets
RUN_PRESETS = {
    "Smoke test (refs → filter)": {"from": "refs", "until": "filter"},
    "Full workflow (refs → collect)": {"from": "refs", "until": "collect"},
    "Scan → Relax": {"from": "scan", "until": "relax"},
    "Filter only (recompute)": {"from": "filter", "until": "filter"},
}

STEP_KEYS = ["refs", "generate", "scan", "relax", "filter", "bandgap", "formation", "collect"]

# -----------------------------
# Shared backend / execution choices
# -----------------------------
BACKEND_CHOICES = ["m3gnet", "uma", "mace", "grace"]
DEVICE_CHOICES = ["cpu", "cuda"]
OPTIMIZER_CHOICES = ["bfgs", "lbfgs", "fire", "mdmin", "quasinewton"]

UMA_MODEL_CHOICES = ["uma-s-1p2", "uma-s-1p1", "uma-m-1p1"]
UMA_TASK_CHOICES = ["omat", "oc20", "oc22", "oc25", "omol", "odac", "omc"]

MACE_MODEL_CHOICES = [
    "small",
    "medium",
    "large",
    "small-mpa-0",
    "medium-mpa-0",
    "large-mpa-0",
    "small-omat-0",
    "medium-omat-0",
]

GRACE_MODEL_CHOICES = [
    "GRACE-1L-OMAT",
    "GRACE-1L-OMAT-M-base",
    "GRACE-1L-OMAT-M",
    "GRACE-1L-OMAT-L-base",
    "GRACE-1L-OMAT-L",
    "GRACE-2L-OMAT",
    "GRACE-2L-OMAT-M-base",
    "GRACE-2L-OMAT-M",
    "GRACE-2L-OMAT-L-base",
    "GRACE-2L-OMAT-L",
    "GRACE-1L-OAM",
    "GRACE-1L-OAM-M",
    "GRACE-1L-OAM-L",
    "GRACE-2L-OAM",
    "GRACE-2L-OAM-M",
    "GRACE-2L-OAM-L",
    "GRACE-1L-SMAX-L",
    "GRACE-1L-SMAX-OMAT-L",
    "GRACE-2L-SMAX-M",
    "GRACE-2L-SMAX-L",
    "GRACE-2L-SMAX-OMAT-M",
    "GRACE-2L-SMAX-OMAT-L",
]

CHOICES = {
    "doping.mode": ["explicit", "enumerate"],

    "references.reference_mode": ["metal", "oxide"],
    "references.device": DEVICE_CHOICES,
    "references.backend": BACKEND_CHOICES,
    "references.optimizer": OPTIMIZER_CHOICES,
    "references.oxygen_mode": ["O-rich", "O-poor"],

    "scan.mode": ["auto", "exact", "sample"],
    "scan.device": DEVICE_CHOICES,
    "scan.backend": BACKEND_CHOICES,

    "relax.device": DEVICE_CHOICES,
    "relax.backend": BACKEND_CHOICES,
    "relax.optimizer": OPTIMIZER_CHOICES,

    "filter.mode": ["window", "topn"],
    "formation.normalize": ["total", "per_dopant", "per_host"],
}

DEFAULTS = {
    "structure": {
        "outdir": "random_structures",
    },
    "references": {
        "reference_mode": "metal",
        "skip_if_done": True,

        "fmax": 0.02,
        "max_steps": 300,
        "tf_threads": 1,
        "omp_threads": 1,

        "device": "cpu",
        "gpu_id": 0,
        "backend": "m3gnet",
        "model": "default",
        "task": "",
        "optimizer": "bfgs",

        "host": "SnO2",
        "host_dir": "reference_structures/oxides",
        "supercell": [5, 2, 1],

        "metal_ref": ["Sn", "Sb", "Ti", "Zr", "Nb"],
        "metals_dir": "reference_structures/metals",

        "oxides_ref": ["Sb2O5", "TiO2", "ZrO2", "Nb2O5"],
        "oxides_dir": "reference_structures/oxides",

        "gas_ref": "O2",
        "gas_dir": "reference_structures/gas",
        "oxygen_mode": "O-rich",
        "muO_shift_ev": 0.0,
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
        "backend": "m3gnet",
        "model": "default",
        "task": "",
        "poscar_in": "POSCAR",
        "topk": 15,
        "symprec": 1e-3,
        "n_workers": 4,
        "chunksize": 50,
        "max_enum": 300_000,
        "max_unique": 100_000,
        "anion_species": ["O"],
        "skip_if_done": True,
        "mode": "auto",
        "sample_budget": 20000,
        "sample_batch_size": 256,
        "sample_patience": 4000,
        "sample_seed": 42,
        "sample_max_saved": 50000,
        "device": "cpu",
        "gpu_id": 0,
    },
    "relax": {
        "backend": "m3gnet",
        "model": "default",
        "task": "",
        "optimizer": "bfgs",
        "fmax": 0.05,
        "max_steps": 300,
        "n_workers": 6,
        "tf_threads": 1,
        "omp_threads": 1,
        "skip_if_done": True,
        "skip_candidate_if_done": True,
        "device": "cpu",
        "gpu_id": 0,
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
        "n_workers": 4,
        "device": "cpu",
        "gpu_id": 0,
        "batch_size": 32,
    },
    "formation": {
        "skip_if_done": True,
        "normalize": "per_dopant",
    },
}
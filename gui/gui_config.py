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
    "Full workflow (refs → phase-diagram)": {"from": "refs", "until": "phase-diagram"},
    "Full workflow + vacancies": {"from": "refs", "until": "vacancies"},
    "Vacancies only": {"from": "vacancies", "until": "vacancies"},
    "Scan → Relax": {"from": "scan", "until": "relax"},
    "Filter only (recompute)": {"from": "filter", "until": "filter"},
}

STEP_KEYS = [
    "refs",
    "generate",
    "scan",
    "relax",
    "filter",
    "bandgap",
    "formation",
    "collect",
    "alloy-hull",
    "phase-diagram",
    "vacancies",
]

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
    "small-0b",
    "medium-0b",
    "small-0b2",
    "medium-0b2",
    "large-0b2",
    "medium-0b3",
    "medium-mpa-0",
    "small-omat-0",
    "medium-omat-0",
    "mace-matpes-pbe-0",
    "mace-matpes-r2scan-0",
    "mh-0",
    "mh-1",
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

    "sequential.mode": ["full", "recompute_energies"],

    "scan.mode": ["auto", "exact", "sample"],
    "scan.device": DEVICE_CHOICES,
    "scan.backend": BACKEND_CHOICES,

    "relax.device": DEVICE_CHOICES,
    "relax.backend": BACKEND_CHOICES,
    "relax.optimizer": OPTIMIZER_CHOICES,

    "relax.relax_mode": ["atoms", "full", "isotropic", "volume", "shape", "xy", "cell_only"],
    "relax.cell_filter": ["frechet", "unit", "exp"],


    "filter.mode": ["window", "topn"],
    "formation.normalize": ["total", "per_dopant", "per_host"],
    "vacancies.parent_source": ["selected_candidates", "directory"],
    "vacancies.count_mode": ["all_reachable", "nominal"],
    "vacancies.enumeration_mode": ["auto", "exact", "sample"],
    "vacancies.backend": BACKEND_CHOICES,
    "vacancies.device": DEVICE_CHOICES,
    "vacancies.energy_normalization": ["total", "per_atom", "per_vacancy"],
    "vacancies.oxygen_reference_mode": ["reference_file", "same_calculator", "explicit", "none"],
    "vacancies.analysis_energy_source": ["relaxed_only", "relaxed_or_single_point"],
    "vacancies.optimizer": OPTIMIZER_CHOICES,
    "vacancies.relax_mode": ["atoms", "full", "isotropic", "volume", "shape", "xy", "cell_only"],
    "vacancies.cell_filter": ["frechet", "unit", "exp"],
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
        "seed_base": 0,
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
    "sequential": {
        "outdir": "sequential_structures",
        "mode": "full",
    },    
    "scan": {
        "backend": "m3gnet",
        "model": "default",
        "task": "",
        "poscar_in": "POSCAR",
        "topk": 15,
        "symprec": 1e-3,
        "n_workers": 12,
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
        "relax_mode": "atoms",
        "cell_filter": "frechet",
    },
    "filter": {
        "mode": "window",
        "window_meV": 50.0,
        "max_candidates": 12,
        "skip_if_done": True,
    },
    "bandgap": {
        "enabled": True,
        "skip_if_done": True,
        "cutoff": 8.0,
        "max_neighbors": 12,
        "n_workers": 1,
        "device": "cpu",
        "gpu_id": 0,
        "batch_size": 32,
    },
    "formation": {
        "skip_if_done": True,
        "normalize": "per_dopant",
        "relative_enabled": False,
        "endpoint_x": "auto",
    },
    "phase_diagram": {
        "skip_if_done": True,
        "stable_threshold_eV_per_atom": 1.0e-8,
    },
    "vacancies": {
        "enabled": True,
        "parent_source": "selected_candidates",
        "parent_directory": "",
        "include_parent_reference": True,
        "skip_if_done": True,
        "resume": True,
        "count_mode": "all_reachable",
        "host_species": "Sn",
        "host_oxidation_state": 4,
        "vacancy_species": "O",
        "vacancy_compensation_charge": 2,
        "oxidation_state_elements": ["Sb", "Nb", "In", "Ta", "Ce", "Ru", "Mn"],
        "oxidation_state_values": [[3, 5], [5], [3], [5], [3, 4], [3, 4, 5], [2, 3, 4]],
        "extra_vacancies": 0,
        "max_vacancies_cap": 8,
        "symprec": 1.0e-3,
        "angle_tolerance": 5.0,
        "mapping_tolerance": 1.0,
        "enumeration_mode": "auto",
        "max_exact_raw_configs": 300000,
        "max_exact_unique_configs": 100000,
        "sample_budget": 20000,
        "sample_batch_size": 256,
        "sample_patience": 4000,
        "sample_seed": 42,
        "sample_max_saved": 50000,
        "minimum_vacancy_distance": 0.0,
        "backend": "m3gnet",
        "model": "default",
        "task": "",
        "device": "cpu",
        "gpu_id": 0,
        "n_workers": 1,
        "tf_threads": 1,
        "omp_threads": 1,
        "chunksize": 25,
        "topk_per_vacancy_count": 15,
        "energy_normalization": "per_vacancy",
        "optimizer": "bfgs",
        "fmax": 0.05,
        "max_steps": 300,
        "relax_mode": "atoms",
        "cell_filter": "frechet",
        "thermodynamic_analysis": False,
        "oxygen_reference_mode": "reference_file",
        "oxygen_reference_file": "reference_structures/reference_energies.json",
        "oxygen_reference_structure": "reference_structures/gas/O2.POSCAR",
        "oxygen_reference_relax": False,
        "allow_unverified_oxygen_reference": False,
        "delta_mu_O_min_eV": -3.0,
        "delta_mu_O_max_eV": 0.0,
        "delta_mu_O_points_eV": [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0],
        "thermodynamic_tolerance_eV": 1.0e-8,
        "analysis_energy_source": "relaxed_only",
        "exclude_unconverged": True,
    },
}

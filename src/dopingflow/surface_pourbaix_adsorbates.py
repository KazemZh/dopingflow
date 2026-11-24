"""Configured adsorbates and neutral molecular reservoirs for periodic slabs."""

from collections import Counter
from itertools import combinations
from functools import lru_cache
import math
from pathlib import Path
import re

import numpy as np
from ase.build import molecule
from ase.data import atomic_numbers, covalent_radii
from ase.io import read
from scipy.spatial.transform import Rotation
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from dopingflow.surface_pourbaix_metal import _occupancies, _site_pool
from dopingflow.surface_pourbaix_sampling import (
    canonical_pattern,
    exposed_sites,
    realizable_coverages,
    select_diverse_patterns,
    symmetry_permutations,
)
from dopingflow.surface_pourbaix_thermo import KB_EV_K
from dopingflow.surface_staged import _evaluate


def _geometry(options, root):
    if bool(options.get("xyz")) == bool(options.get("molecule")):
        raise ValueError("Specify one XYZ file or one ASE molecule name per geometry")
    atoms = (
        read(root / options["xyz"], format="xyz")
        if options.get("xyz")
        else molecule(options["molecule"])
    )
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError("Molecular coordinates must be finite")
    return atoms


def parse_adsorbates(entries, root):
    if not isinstance(entries, dict):
        raise ValueError("Use [surface_pourbaix.adsorbates.NAME] tables")
    parsed = {}
    for name, entry in entries.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid adsorbate name {name!r}")
        ads = _geometry(entry, root)
        ref = _geometry(entry["reference"], root) if "reference" in entry else ads
        counts = Counter(ads.get_chemical_symbols())
        counts.subtract(ref.get_chemical_symbols())
        pairs = counts.pop("H", 0)
        if any(counts.values()):
            raise ValueError(f"{name}: reservoir + H+/electron pairs must balance the adsorbate")
        anchor = entry.get("anchor_index", 0)
        if type(anchor) is not int or not 0 <= anchor < len(ads):
            raise ValueError(f"{name}: invalid zero-based anchor_index")
        sites = entry.get("site_types", ["ontop"])
        species = entry.get("binding_species", [])
        if not isinstance(sites, list) or not sites or set(sites) - {"ontop", "bridge", "hollow"}:
            raise ValueError(f"{name}: invalid adsorption site_types")
        if not isinstance(species, list) or set(species) - atomic_numbers.keys():
            raise ValueError(f"{name}: invalid binding_species")
        coverages = entry.get("coverages_pct", [25, 50, 75, 100])
        if not coverages or any(not 0 < c <= 100 for c in coverages):
            raise ValueError(f"{name}: coverages_pct must lie in (0, 100]")
        angles = np.asarray(entry.get("rotation_deg", [0, 0, 0]), dtype=float)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError(f"{name}: rotation_deg needs three finite x, y, z angles")
        offsets = Rotation.from_euler("xyz", angles, degrees=True).apply(
            ads.positions - ads.positions[anchor]
        )
        data = dict(
            entry,
            symbols=ads.get_chemical_symbols(),
            offsets=offsets.tolist(),
            ref_symbols=ref.get_chemical_symbols(),
            ref_positions=ref.positions.tolist(),
            pairs=pairs,
            site_types=sites,
            binding_species=species,
            coverages_pct=coverages,
            anchor_index=anchor,
        )
        for key, default in (
            ("height_A", 1.8),
            ("activity", 1.0),
            ("min_spacing_A", 1.5),
            ("free_energy_correction_eV", 0.0),
        ):
            data[key] = float(entry.get(key, default))
            if not math.isfinite(data[key]) or (
                key != "free_energy_correction_eV" and data[key] <= 0
            ):
                raise ValueError(f"{name}: invalid {key}")
        parsed[name] = data
    return parsed


def reference_mu(name, spec, cfg, calc, level):
    """One neutral reservoir molecule per adsorbate, evaluated at the requested level."""
    energy = spec.get(f"reference_energy_eV_{level}")
    if energy is None:
        box = cfg["reference_box_A"]
        coords = np.array(spec["ref_positions"])
        if np.max(np.ptp(coords, axis=0)) >= box:
            raise ValueError(f"Enlarge reference_box_A for {name}")
        ref = Structure(
            Lattice.cubic(box),
            spec["ref_symbols"],
            coords - coords.mean(axis=0) + box / 2,
            coords_are_cartesian=True,
        )
        path = Path(cfg["output_dir"]) / "references" / f"ads_{name}"
        if level == "ml" and cfg["compute_references"]:
            opts = dict(cfg["screen"], relax=cfg["reference_relax"])
            energy = _evaluate(ref, [], opts, calc, path).get("energy_eV")
        elif level == "dft":
            from dopingflow.surface_pourbaix_dft import dft_energy

            path.mkdir(parents=True, exist_ok=True)
            poscar = path / "POSCAR_for_dft"
            Poscar(ref).write_file(str(poscar))
            energy, _, _ = dft_energy(poscar, f"reference/ads_{name}", cfg, is_reference=True)
    if energy is None:
        if level == "dft":
            return None
        raise ValueError(f"Missing {level} reservoir energy for {name}")
    if not math.isfinite(float(energy)):
        raise ValueError(f"Nonfinite {level} reservoir energy for {name}")
    return (
        float(energy)
        + spec["free_energy_correction_eV"]
        + KB_EV_K * cfg["temperature_K"] * math.log(spec["activity"])
    )


def configured_states(slab, normal, inplane, side, cfg, common):
    elements = {site.specie.symbol for site in slab}
    outer = exposed_sites(
        slab,
        elements,
        normal=normal,
        placement_side=side,
        window_A=cfg["surface_window_A"],
        selection_mode=cfg["surface_site_selection"],
        layer_gap_A=cfg["surface_layer_gap_A"],
    )
    for name, spec in cfg.get("adsorbates", {}).items():
        allowed = set(spec["binding_species"]) or elements
        exposed = [(i, s) for i, s in outer if slab[i].specie.symbol in allowed]
        if not exposed:
            continue
        pool = _site_pool(slab, exposed, normal, spec["site_types"])
        markers = slab.copy()
        sites = []
        for _, sign, coord in pool:
            sites.append((len(markers), sign))
            markers.append(
                "He", coord + sign * normal * spec["height_A"], coords_are_cartesian=True
            )
        # An in-plane molecular orientation can break the symmetry of its anchor site.
        axial = np.allclose(np.asarray(spec["offsets"])[:, :2], 0)
        perms = symmetry_permutations(
            markers,
            sites,
            enabled=cfg["symmetry_reduce"] and axial,
            symprec_A=cfg["symmetry_symprec_A"],
            mapping_tolerance_A=cfg["symmetry_mapping_tolerance_A"],
            angle_tolerance_deg=cfg["symmetry_angle_tolerance_deg"],
        )
        distances = markers.distance_matrix[len(slab) :, len(slab) :]
        positions = []
        fractions = []
        blocked = set()
        # Reject severe initial overlaps before spending the arrangement budget.
        # Use minimum-image distances for molecules crossing lateral cell edges.
        radii = np.array([covalent_radii[atomic_numbers[s]] for s in spec["symbols"]])
        parent_radii = np.array([covalent_radii[s.specie.Z] for s in slab])
        molecule_limits = 0.65 * (radii[:, None] + radii)
        for i, (_, sign, point) in enumerate(pool):
            outward = sign * normal
            basis = np.array([inplane, np.cross(outward, inplane), outward])
            pos = point + outward * spec["height_A"] + np.asarray(spec["offsets"]) @ basis
            frac = slab.lattice.get_fractional_coords(pos)
            positions.append(pos)
            fractions.append(frac)
            if np.any(
                slab.lattice.get_all_distances(frac, slab.frac_coords)
                < 0.65 * (radii[:, None] + parent_radii)
            ):
                blocked.add(i)

        @lru_cache(maxsize=8192)
        def clashes(i, j):
            return distances[i, j] < spec["min_spacing_A"] or np.any(
                slab.lattice.get_all_distances(fractions[i], fractions[j]) < molecule_limits
            )

        for cov in realizable_coverages(len(exposed), spec["coverages_pct"]):
            count = cov["count"]
            if count > len(pool):
                raise ValueError(
                    f"Too few adsorption sites for {name} at {cov['actual_coverage_pct']}%"
                )
            patterns = set()
            examined = 0
            for occupied, _ in _occupancies(
                len(pool), count, 0, cfg["max_raw_configurations_per_stoichiometry"]
            ):
                examined += 1
                if blocked.intersection(occupied) or any(
                    clashes(i, j) for i, j in combinations(occupied, 2)
                ):
                    continue
                labels = tuple(int(i in occupied) for i in range(len(pool)))
                patterns.add(canonical_pattern(labels, perms))
            if not patterns:
                raise ValueError(f"No admissible {name} arrangements with {count} molecules")
            selected = select_diverse_patterns(
                sorted(patterns), sites, markers, cap=cfg["max_arrangements_per_stoichiometry"]
            )
            for arr, pattern in enumerate(selected, 1):
                candidate = slab.copy()
                indices = [i for i, label in enumerate(pattern) if label]
                props = (
                    {"selective_dynamics": [True] * 3}
                    if "selective_dynamics" in slab.site_properties
                    else {}
                )
                for i in indices:
                    for symbol, pos in zip(spec["symbols"], positions[i]):
                        candidate.append(symbol, pos, coords_are_cartesian=True, properties=props)
                yield dict(
                    common,
                    family=f"ads:{name}",
                    state_id=f"ads_{name}_{count}_arr{arr}",
                    adsorbate_name=name,
                    molecule_count=count,
                    eligible_binding_sites=len(exposed),
                    delta_n_H=count * spec["symbols"].count("H"),
                    delta_n_O=count * spec["symbols"].count("O"),
                    proton_electron_pairs=count * spec["pairs"],
                    arrangement_id=arr,
                    actual_coverage_pct=cov["actual_coverage_pct"],
                    site_indices=indices,
                    site_sides=[pool[i][1] for i in indices],
                    adsorption_site_types=[pool[i][0] for i in indices],
                    raw_arrangements_examined=examined,
                    raw_arrangements_total=math.comb(len(pool), count),
                    symmetry_unique_arrangements=len(patterns),
                    symmetry_operations=len(perms),
                    structure=candidate,
                )


def validate_molecules(parent, initial, relaxed, state, cfg):
    spec = cfg["adsorbates"][state["adsorbate_name"]]
    size = len(spec["symbols"])
    first = len(parent)
    issues = set()
    for start in range(first, len(relaxed), size):
        group = range(start, start + size)
        if (
            min(relaxed.get_distance(i, j) for i in group for j in range(first))
            > cfg["postprocess_surface_attachment_cutoff_A"]
        ):
            issues.add("detached")
        for i in group:
            for j in range(i + 1, len(relaxed)):
                cutoff = 1.2 * (
                    covalent_radii[relaxed[i].specie.Z] + covalent_radii[relaxed[j].specie.Z]
                )
                bonded = j in group and initial.get_distance(i, j) <= cutoff
                if bonded != (relaxed.get_distance(i, j) <= cutoff):
                    issues.add("changed molecular connectivity")
    return dict(
        pourbaix_eligible=not issues,
        state_status="invalid" if issues else "intact",
        final_family="invalid" if issues else state["family"],
        final_state_label="",
        postprocess_reason=", ".join(sorted(issues)),
    )

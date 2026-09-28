# gui/view_structure.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
def show_structure(path: Path, title: str = "", spin: bool = False, width: int = 800, height: int = 450):
    import py3Dmol
    from ase.io import read

    atoms = read(str(path), format="vasp")

    xyz = f"{len(atoms)}\n{title}\n"
    for sym, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        xyz += f"{sym} {x:.6f} {y:.6f} {z:.6f}\n"

    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.28}})
    view.zoomTo()
    if spin:
        view.spin(True)

    # Streamlit embed (no ipywidgets needed)
    html = view._make_html()
    st.components.v1.html(html, width=width, height=height, scrolling=False)

def show_site_environment(
    path: Path,
    site_index: int,
    neighbors,
    title: str = "",
    *,
    width: int = 760,
    height: int = 520,
    label_neighbors: bool = True,
):
    """Show a structure with one target site and its periodic local shell marked."""
    import py3Dmol
    from pymatgen.core import Structure

    structure = Structure.from_file(path)
    idx = int(site_index)
    if idx < 0 or idx >= len(structure):
        raise IndexError(f"site_index {idx} outside structure with {len(structure)} atoms")

    xyz = f"{len(structure)}\n{title}\n"
    for site in structure:
        x, y, z = site.coords
        xyz += f"{site.specie.symbol} {x:.6f} {y:.6f} {z:.6f}\n"

    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz, "xyz")
    view.setStyle({}, {"stick": {"radius": 0.10}, "sphere": {"scale": 0.22}})

    target = structure[idx]
    tx, ty, tz = (float(x) for x in target.coords)
    view.setStyle(
        {"serial": idx + 1},
        {
            "stick": {"radius": 0.18, "color": "crimson"},
            "sphere": {"scale": 0.62, "color": "crimson"},
        },
    )
    view.addLabel(
        f"{target.specie.symbol} site {idx}",
        {
            "position": {"x": tx, "y": ty, "z": tz},
            "fontColor": "white",
            "backgroundColor": "crimson",
            "fontSize": 13,
            "showBackground": True,
        },
    )

    records = (
        neighbors.to_dict("records")
        if hasattr(neighbors, "to_dict")
        else list(neighbors or [])
    )
    for rec in records:
        nx = float(rec["neighbor_cart_x_A"])
        ny = float(rec["neighbor_cart_y_A"])
        nz = float(rec["neighbor_cart_z_A"])
        neighbor_class = str(rec.get("neighbor_class", ""))
        if bool(rec.get("within_coordination_cutoff", False)):
            marker_color = "gold"
        elif neighbor_class == "dopant":
            marker_color = "dodgerblue"
        else:
            marker_color = "mediumseagreen"
        view.addSphere(
            {
                "center": {"x": nx, "y": ny, "z": nz},
                "radius": 0.30,
                "color": marker_color,
                "opacity": 0.88,
            }
        )
        view.addLine(
            {
                "start": {"x": tx, "y": ty, "z": tz},
                "end": {"x": nx, "y": ny, "z": nz},
                "color": marker_color,
                "linewidth": 2.0,
            }
        )
        if label_neighbors:
            view.addLabel(
                f"{rec.get('species', '?')}{int(rec.get('neighbor_index', -1))} "
                f"{float(rec.get('distance_A', float('nan'))):.2f} Å",
                {
                    "position": {"x": nx, "y": ny, "z": nz},
                    "fontColor": "black",
                    "backgroundColor": "white",
                    "fontSize": 10,
                    "showBackground": True,
                    "backgroundOpacity": 0.75,
                },
            )

    view.zoomTo()
    html = view._make_html()
    st.components.v1.html(html, width=width, height=height, scrolling=False)

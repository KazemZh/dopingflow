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
    element_colors: dict[str, str] | None = None,
    target_color: str = "#D62728",
    enable_hover: bool = True,
    show_orientation: bool = True,
    surface_side: str = "top",
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

    # Apply user-selected element colors consistently to both atoms and bonds.
    for element, color in dict(element_colors or {}).items():
        view.setStyle(
            {"elem": str(element)},
            {
                "stick": {"radius": 0.10, "color": str(color)},
                "sphere": {"scale": 0.22, "color": str(color)},
            },
        )

    if enable_hover:
        hover_callback = """
        function(atom, viewer) {
            if(!atom.label) {
                var idx = (atom.index !== undefined) ? atom.index :
                          ((atom.serial !== undefined) ? atom.serial - 1 : "?");
                atom.label = viewer.addLabel(
                    atom.elem + " site " + idx,
                    {
                        position: atom,
                        backgroundColor: "white",
                        backgroundOpacity: 0.90,
                        fontColor: "black",
                        fontSize: 13,
                        borderThickness: 1,
                        borderColor: "black"
                    }
                );
            }
        }
        """
        unhover_callback = """
        function(atom, viewer) {
            if(atom.label) {
                viewer.removeLabel(atom.label);
                delete atom.label;
            }
        }
        """
        model = view.getModel()
        model.setHoverable({}, True, hover_callback, unhover_callback)
        view.setHoverDuration(100)

    target = structure[idx]
    tx, ty, tz = (float(x) for x in target.coords)
    target_element_color = dict(element_colors or {}).get(
        target.specie.symbol
    )
    target_stick = {"radius": 0.10}
    target_sphere = {"scale": 0.27}
    if target_element_color:
        target_stick["color"] = target_element_color
        target_sphere["color"] = target_element_color
    view.setStyle(
        {"index": idx},
        {
            "stick": target_stick,
            "sphere": target_sphere,
        },
    )
    view.addLabel(
        f"{target.specie.symbol} site {idx}",
        {
            "position": {"x": tx, "y": ty, "z": tz},
            "fontColor": "white",
            "backgroundColor": "#333333",
            "fontSize": 13,
            "showBackground": True,
        },
    )

    if show_orientation:
        coords = [site.coords for site in structure]
        xs = [float(value[0]) for value in coords]
        ys = [float(value[1]) for value in coords]
        zs = [float(value[2]) for value in coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        span = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
        axis_length = min(4.0, max(2.0, 0.16 * span))

        # Put the coordinate triad just outside one lower slab corner.
        origin = {
            "x": x_min - 0.35 * axis_length,
            "y": y_min - 0.35 * axis_length,
            "z": z_min,
        }
        axes = [
            ("x", "#D62728", (axis_length, 0.0, 0.0)),
            ("y", "#2CA02C", (0.0, axis_length, 0.0)),
            ("z", "#1F77B4", (0.0, 0.0, axis_length)),
        ]
        for label, color, vector in axes:
            end = {
                "x": origin["x"] + vector[0],
                "y": origin["y"] + vector[1],
                "z": origin["z"] + vector[2],
            }
            view.addArrow(
                {
                    "start": origin,
                    "end": end,
                    "radius": 0.08,
                    "radiusRatio": 1.7,
                    "mid": 0.82,
                    "color": color,
                }
            )
            view.addLabel(
                label,
                {
                    "position": end,
                    "fontColor": color,
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.80,
                    "fontSize": 12,
                    "showBackground": True,
                },
            )

        # Explicitly mark the exposed surface/vacuum direction used by DopingFlow.
        side = str(surface_side).lower()
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        surface_color = "#FF8C00"

        def _surface_arrow(z_start: float, direction: float, text: str) -> None:
            start = {"x": center_x, "y": center_y, "z": z_start}
            end = {
                "x": center_x,
                "y": center_y,
                "z": z_start + direction * axis_length,
            }
            view.addArrow(
                {
                    "start": start,
                    "end": end,
                    "radius": 0.10,
                    "radiusRatio": 1.8,
                    "mid": 0.82,
                    "color": surface_color,
                }
            )
            view.addLabel(
                text,
                {
                    "position": end,
                    "fontColor": "black",
                    "backgroundColor": "#FFD27F",
                    "backgroundOpacity": 0.90,
                    "fontSize": 11,
                    "showBackground": True,
                },
            )

        if side in {"top", "both"}:
            _surface_arrow(z_max + 0.15, 1.0, "surface / vacuum  +z")
        if side in {"bottom", "both"}:
            _surface_arrow(z_min - 0.15, -1.0, "surface / vacuum  -z")

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

# gui/view_structure.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import streamlit as st


# Shared DopingFlow structure-viewer palette. Keep this in one place so every
# structure visualization (bulk, vacancy, surface, Pourbaix, leaching) uses the
# same element identity and appearance.
DEFAULT_ELEMENT_COLORS: dict[str, str] = {
    "O": "#E41A1C",
    "Sn": "#377EB8",
    "Sb": "#984EA3",
    "In": "#4DAF4A",
    "H": "#F2F2F2",
    "Ti": "#FF7F00",
    "Zr": "#A6CEE3",
    "Nb": "#A65628",
    "Ba": "#FFD92F",
    "Mn": "#F781BF",
    "Ni": "#1B9E77",
    "Fe": "#E6550D",
    "Zn": "#66A61E",
    "W": "#7570B3",
    "Ce": "#E6AB02",
    "Ru": "#666666",
    "Ir": "#1F78B4",
    "Pt": "#A6A6A6",
    "Au": "#E6B800",
    "La": "#66C2A5",
    "Sm": "#FC8D62",
    "Y": "#8DA0CB",
    "Nd": "#E78AC3",
    "Gd": "#A6D854",
    "Pr": "#FFD92F",
}

FALLBACK_ELEMENT_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#BAB0AC",
]


def element_color_defaults(elements: Iterable[str]) -> dict[str, str]:
    ordered = sorted({str(element) for element in elements if str(element)})
    return {
        element: DEFAULT_ELEMENT_COLORS.get(
            element,
            FALLBACK_ELEMENT_COLORS[index % len(FALLBACK_ELEMENT_COLORS)],
        )
        for index, element in enumerate(ordered)
    }


def _structure_from_path(path: Path | str):
    from pymatgen.core import Structure

    return Structure.from_file(str(Path(path)))


def _elements_from_paths(paths: Sequence[Path | str | None]) -> list[str]:
    elements: set[str] = set()
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            structure = _structure_from_path(path)
        except Exception:
            continue
        elements.update(site.specie.symbol for site in structure)
    return sorted(elements)


def structure_viewer_controls(
    paths: Sequence[Path | str | None],
    *,
    key_prefix: str,
    expanded: bool = False,
    show_orientation_default: bool = True,
    show_unit_cell_default: bool = True,
) -> dict[str, Any]:
    """Shared Streamlit controls for all DopingFlow structure viewers.

    The same returned palette can be passed to several viewers (for example,
    before/after relaxation) so an element never changes color between panels.
    """
    elements = _elements_from_paths(paths)
    defaults = element_color_defaults(elements)

    with st.expander("Atom colors & viewer appearance", expanded=expanded):
        reset = st.button(
            "Reset atom colors",
            key=f"{key_prefix}_reset_colors",
        )
        if reset:
            for element, default_color in defaults.items():
                st.session_state[f"{key_prefix}_color_{element}"] = default_color

        color_columns = st.columns(min(4, max(1, len(elements))))
        element_colors: dict[str, str] = {}
        for index, element in enumerate(elements):
            element_colors[element] = color_columns[
                index % len(color_columns)
            ].color_picker(
                f"{element} atoms",
                value=defaults[element],
                key=f"{key_prefix}_color_{element}",
            )

        c1, c2, c3 = st.columns(3)
        enable_hover = c1.checkbox(
            "Show element + site index on hover",
            value=True,
            key=f"{key_prefix}_hover",
        )
        show_orientation = c2.checkbox(
            "Show coordinate axes / surface direction",
            value=bool(show_orientation_default),
            key=f"{key_prefix}_orientation",
        )
        show_unit_cell = c3.checkbox(
            "Show simulation cell",
            value=bool(show_unit_cell_default),
            key=f"{key_prefix}_unit_cell",
        )

        s1, s2 = st.columns(2)
        sphere_scale = float(
            s1.slider(
                "Atom sphere size",
                min_value=0.12,
                max_value=0.50,
                value=0.25,
                step=0.01,
                key=f"{key_prefix}_sphere_scale",
            )
        )
        stick_radius = float(
            s2.slider(
                "Bond thickness",
                min_value=0.04,
                max_value=0.24,
                value=0.10,
                step=0.01,
                key=f"{key_prefix}_stick_radius",
            )
        )

        st.caption(
            "This palette is shared by all DopingFlow structure viewers. "
            "The same element keeps the same color in before/after comparisons."
        )

    return {
        "element_colors": element_colors or defaults,
        "enable_hover": bool(enable_hover),
        "show_orientation": bool(show_orientation),
        "show_unit_cell": bool(show_unit_cell),
        "sphere_scale": sphere_scale,
        "stick_radius": stick_radius,
    }


def _xyz_from_structure(structure, title: str) -> str:
    lines = [str(len(structure)), str(title or "")]
    for site in structure:
        x, y, z = site.coords
        lines.append(
            f"{site.specie.symbol} {float(x):.8f} {float(y):.8f} {float(z):.8f}"
        )
    return "\n".join(lines) + "\n"


def _apply_element_styles(
    view,
    elements: Iterable[str],
    *,
    element_colors: dict[str, str] | None,
    sphere_scale: float,
    stick_radius: float,
) -> dict[str, str]:
    colors = dict(element_colors or element_color_defaults(elements))
    view.setStyle(
        {},
        {
            "stick": {"radius": float(stick_radius)},
            "sphere": {"scale": float(sphere_scale)},
        },
    )
    for element in sorted(set(elements)):
        color = colors.get(element)
        if not color:
            continue
        view.setStyle(
            {"elem": str(element)},
            {
                "stick": {
                    "radius": float(stick_radius),
                    "color": str(color),
                },
                "sphere": {
                    "scale": float(sphere_scale),
                    "color": str(color),
                },
            },
        )
    return colors


def _enable_hover_labels(view) -> None:
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
                    backgroundOpacity: 0.92,
                    fontColor: "black",
                    fontSize: 13,
                    borderThickness: 1,
                    borderColor: "#444444"
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


def _add_unit_cell(view, structure) -> None:
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    origin = np.zeros(3, dtype=float)
    a, b, c = lattice
    corners = [
        origin,
        a,
        b,
        c,
        a + b,
        a + c,
        b + c,
        a + b + c,
    ]
    edges = (
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    )
    for first, second in edges:
        p0, p1 = corners[first], corners[second]
        view.addLine(
            {
                "start": {
                    "x": float(p0[0]),
                    "y": float(p0[1]),
                    "z": float(p0[2]),
                },
                "end": {
                    "x": float(p1[0]),
                    "y": float(p1[1]),
                    "z": float(p1[2]),
                },
                "color": "#555555",
                "linewidth": 1.2,
                "opacity": 0.75,
            }
        )


def _add_orientation(
    view,
    structure,
    *,
    surface_side: str | None = None,
) -> None:
    coords = np.asarray([site.coords for site in structure], dtype=float)
    if coords.size == 0:
        return
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = max(float(np.ptp(coords, axis=0).max()), 1.0)
    axis_length = min(4.0, max(1.8, 0.16 * span))

    origin = {
        "x": float(mins[0] - 0.35 * axis_length),
        "y": float(mins[1] - 0.35 * axis_length),
        "z": float(mins[2]),
    }
    axes = [
        ("x", "#D62728", np.array([axis_length, 0.0, 0.0])),
        ("y", "#2CA02C", np.array([0.0, axis_length, 0.0])),
        ("z", "#1F77B4", np.array([0.0, 0.0, axis_length])),
    ]
    start_vec = np.array([origin["x"], origin["y"], origin["z"]], dtype=float)
    for label, color, vector in axes:
        end_vec = start_vec + vector
        end = {
            "x": float(end_vec[0]),
            "y": float(end_vec[1]),
            "z": float(end_vec[2]),
        }
        view.addArrow(
            {
                "start": origin,
                "end": end,
                "radius": 0.075,
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
                "backgroundOpacity": 0.84,
                "fontSize": 12,
                "showBackground": True,
            },
        )

    side = str(surface_side or "").strip().lower()
    if side not in {"top", "bottom", "both"}:
        return

    # Surface direction follows the true slab normal a x b, not an assumed z axis.
    normal = np.cross(
        np.asarray(structure.lattice.matrix[0], dtype=float),
        np.asarray(structure.lattice.matrix[1], dtype=float),
    )
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return
    normal /= norm

    projections = coords @ normal
    center = coords.mean(axis=0)
    center_proj = float(np.dot(center, normal))
    color = "#FF8C00"

    def add_surface_arrow(which: str) -> None:
        if which == "top":
            boundary_proj = float(projections.max())
            direction = normal
            label = "surface / vacuum"
        else:
            boundary_proj = float(projections.min())
            direction = -normal
            label = "surface / vacuum"
        boundary = center + normal * (boundary_proj - center_proj)
        start = boundary + direction * 0.10
        end = start + direction * axis_length
        view.addArrow(
            {
                "start": {
                    "x": float(start[0]),
                    "y": float(start[1]),
                    "z": float(start[2]),
                },
                "end": {
                    "x": float(end[0]),
                    "y": float(end[1]),
                    "z": float(end[2]),
                },
                "radius": 0.095,
                "radiusRatio": 1.8,
                "mid": 0.82,
                "color": color,
            }
        )
        view.addLabel(
            label,
            {
                "position": {
                    "x": float(end[0]),
                    "y": float(end[1]),
                    "z": float(end[2]),
                },
                "fontColor": "black",
                "backgroundColor": "#FFD27F",
                "backgroundOpacity": 0.92,
                "fontSize": 11,
                "showBackground": True,
            },
        )

    if side in {"top", "both"}:
        add_surface_arrow("top")
    if side in {"bottom", "both"}:
        add_surface_arrow("bottom")


def show_structure(
    path: Path | str,
    title: str = "",
    spin: bool = False,
    width: int = 800,
    height: int = 450,
    *,
    viewer_mode: str = "bulk",
    element_colors: dict[str, str] | None = None,
    enable_hover: bool = True,
    show_orientation: bool = True,
    show_unit_cell: bool = True,
    surface_side: str | None = None,
    sphere_scale: float = 0.25,
    stick_radius: float = 0.10,
):
    """Render a structure using the shared DopingFlow palette and appearance."""
    import py3Dmol

    structure = _structure_from_path(Path(path))
    elements = [site.specie.symbol for site in structure]

    view = py3Dmol.view(width=width, height=height)
    view.addModel(_xyz_from_structure(structure, title), "xyz")
    _apply_element_styles(
        view,
        elements,
        element_colors=element_colors,
        sphere_scale=sphere_scale,
        stick_radius=stick_radius,
    )

    if enable_hover:
        _enable_hover_labels(view)
    if show_unit_cell:
        _add_unit_cell(view, structure)
    if show_orientation:
        inferred_side = surface_side
        if inferred_side is None and str(viewer_mode).lower() == "surface":
            inferred_side = "both"
        _add_orientation(view, structure, surface_side=inferred_side)

    view.setBackgroundColor("white")
    view.zoomTo()
    if spin:
        view.spin(True)

    st.components.v1.html(
        view._make_html(),
        width=width,
        height=height,
        scrolling=False,
    )


def show_site_environment(
    path: Path | str,
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
    show_unit_cell: bool = False,
    surface_side: str = "top",
    sphere_scale: float = 0.22,
    stick_radius: float = 0.10,
):
    """Show one target site and its periodic local shell with shared styling."""
    import py3Dmol

    structure = _structure_from_path(Path(path))
    idx = int(site_index)
    if idx < 0 or idx >= len(structure):
        raise IndexError(
            f"site_index {idx} outside structure with {len(structure)} atoms"
        )

    elements = [site.specie.symbol for site in structure]
    colors = dict(element_colors or element_color_defaults(elements))

    view = py3Dmol.view(width=width, height=height)
    view.addModel(_xyz_from_structure(structure, title), "xyz")
    _apply_element_styles(
        view,
        elements,
        element_colors=colors,
        sphere_scale=sphere_scale,
        stick_radius=stick_radius,
    )

    if enable_hover:
        _enable_hover_labels(view)

    target = structure[idx]
    tx, ty, tz = (float(x) for x in target.coords)
    target_element_color = colors.get(target.specie.symbol, target_color)
    view.setStyle(
        {"index": idx},
        {
            "stick": {
                "radius": float(stick_radius),
                "color": target_element_color,
            },
            "sphere": {
                "scale": float(sphere_scale) * 1.23,
                "color": target_element_color,
            },
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

    if show_unit_cell:
        _add_unit_cell(view, structure)
    if show_orientation:
        _add_orientation(view, structure, surface_side=surface_side)

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

    view.setBackgroundColor("white")
    view.zoomTo()
    st.components.v1.html(
        view._make_html(),
        width=width,
        height=height,
        scrolling=False,
    )

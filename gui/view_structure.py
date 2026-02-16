# gui/view_structure.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
from ase.io import read

def show_structure(path: Path, title: str = "", spin: bool = False, width: int = 800, height: int = 450):
    import py3Dmol

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

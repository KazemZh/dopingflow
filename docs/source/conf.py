# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "dopingflow"
copyright = "2026, Kazem Zhour"
author = "Kazem Zhour"
release = "3.1.0"
version = "3.1"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

language = "en"

# -----------------------------------------------------------------------------
# Autodoc / Napoleon
# -----------------------------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "mixed"
autodoc_mock_imports = [
    "tensorflow",
    "m3gnet",
    "mace",
    "fairchem",
    "tensorpotential",
    "torch",
    "dgl",
    "alignn",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# -----------------------------------------------------------------------------
# MyST
# -----------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_title = "dopingflow documentation"

html_theme_options = {
    "logo_only": True,
}

# -----------------------------------------------------------------------------
# LaTeX / PDF output
# -----------------------------------------------------------------------------
latex_logo = "_static/logo.png"

latex_elements = {
    "preamble": r"""
\usepackage{graphicx}
""",
}
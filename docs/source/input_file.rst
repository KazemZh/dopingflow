
Input File Specification (input.toml)
=====================================

The workflow is controlled via a TOML configuration file.

Sections
--------

[structure]
[dopants]
[scan]
[relax]
[filter]
[bandgap]
[formation]

Example
-------

[structure]
host_structure = "SnO2.cif"
supercell = [5,2,1]

[dopants]
elements = ["Sb", "Ti"]
concentrations = [0.05, 0.05]

[scan]
model = "m3gnet"
top_k = 5

[relax]
optimizer = "FIRE"
fmax = 0.05
steps = 500

[filter]
energy_window = 0.05

[bandgap]
model = "alignn"

[formation]
reference_set = "ml"

from __future__ import annotations

from dopingflow import phase_alias_extensions  # noqa: F401
from dopingflow.phase_structure_fallback import expected_crystal_system


def test_kingsbury_trigon_abbreviation_maps_to_trigonal():
    assert expected_crystal_system("trigon") == "trigonal"

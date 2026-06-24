import pytest

from dopingflow.alloy_hull import (
    HullPoint,
    _hull_energy_and_segment,
    _lower_convex_hull,
    _minimum_point_at_each_composition,
)


def point(x, energy, label):
    return HullPoint(
        x=x,
        energy_per_cation_eV=energy,
        label=label,
        source="candidate",
    )


def test_lower_hull_uses_intermediate_stable_vertex():
    points = [
        point(0.0, 0.0, "SnO2"),
        point(0.25, -0.10, "x=0.25"),
        point(0.50, -0.05, "x=0.50"),
        point(1.0, 0.0, "SbO2"),
    ]

    hull = _lower_convex_hull(_minimum_point_at_each_composition(points))

    assert [vertex.label for vertex in hull] == ["SnO2", "x=0.25", "SbO2"]

    energy, left, right = _hull_energy_and_segment(0.50, hull)
    assert left.label == "x=0.25"
    assert right.label == "SbO2"
    assert energy == pytest.approx(-0.0666666667)
    assert points[2].energy_per_cation_eV - energy == pytest.approx(0.0166666667)


def test_only_lowest_candidate_at_one_composition_can_define_hull():
    points = [
        point(0.0, 0.0, "SnO2"),
        point(0.5, -0.02, "candidate_high"),
        point(0.5, -0.05, "candidate_low"),
        point(1.0, 0.0, "SbO2"),
    ]

    minima = _minimum_point_at_each_composition(points)

    assert [entry.label for entry in minima] == ["SnO2", "candidate_low", "SbO2"]

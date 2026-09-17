import pytest

from dopingflow.vacancy_parent_selection_extensions import (
    filter_selected_parents,
    parse_parent_include,
)


def _parents():
    return [
        {
            "parent_id": "Sb5_Ti2p5/candidate_003",
            "composition": "Sb5_Ti2p5",
            "candidate": "candidate_003",
        },
        {
            "parent_id": "Ce2p5_Sb2p5/candidate_001",
            "composition": "Ce2p5_Sb2p5",
            "candidate": "candidate_001",
        },
        {
            "parent_id": "In7p5_Sb5/candidate_004",
            "composition": "In7p5_Sb5",
            "candidate": "candidate_004",
        },
    ]


def test_parent_include_validation():
    assert parse_parent_include({}) is None
    assert parse_parent_include({"parent_include": ["Ti_2.5Sb_5"]}) == (
        "Ti_2.5Sb_5",
    )
    with pytest.raises(ValueError, match="non-empty"):
        parse_parent_include({"parent_include": []})
    with pytest.raises(ValueError, match="duplicates"):
        parse_parent_include({"parent_include": ["Sb5", "Sb5"]})


def test_composition_selector_is_order_and_notation_insensitive():
    selected = filter_selected_parents(
        _parents(),
        ("Ti_2.5Sb_5", "Sb_2.5Ce_2.5", "In_7.5Sb_5"),
    )
    assert [row["parent_id"] for row in selected] == [
        "Sb5_Ti2p5/candidate_003",
        "Ce2p5_Sb2p5/candidate_001",
        "In7p5_Sb5/candidate_004",
    ]


def test_exact_parent_id_selector():
    selected = filter_selected_parents(
        _parents(), ("Sb5_Ti2p5/candidate_003",)
    )
    assert [row["parent_id"] for row in selected] == [
        "Sb5_Ti2p5/candidate_003"
    ]


def test_missing_parent_selector_fails_with_available_compositions():
    with pytest.raises(ValueError, match="matched no discovered parent"):
        filter_selected_parents(_parents(), ("Ru_2.5Sb_5",))

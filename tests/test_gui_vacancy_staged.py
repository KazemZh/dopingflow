from gui.vacancy_staged import (
    build_staged_command,
    format_int_list,
    parse_parent_selectors,
    parse_positive_int_list,
)


def test_parse_positive_int_list_accepts_common_separators_and_deduplicates():
    assert parse_positive_int_list("1, 2  2\n3", field_name="vacancy_counts") == [1, 2, 3]


def test_parse_positive_int_list_rejects_nonpositive_values():
    try:
        parse_positive_int_list("1, 0", field_name="vacancy_counts")
    except ValueError as exc:
        assert "positive integers" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_parent_selectors_ignores_blanks_and_duplicates():
    text = "Ti_2.5Sb_2.5\n\nSb5_Ti2p5\nTi_2.5Sb_2.5\n"
    assert parse_parent_selectors(text) == ["Ti_2.5Sb_2.5", "Sb5_Ti2p5"]


def test_build_staged_commands_use_separate_entry_points():
    search = build_staged_command(
        "search", env_name="dopingflow-grace", config_path="/work/input.toml"
    )
    finalize = build_staged_command(
        "finalize", env_name="dopingflow-mace", config_path="/work/input.toml"
    )
    assert search == [
        "conda",
        "run",
        "-n",
        "dopingflow-grace",
        "dopingflow",
        "vacancies-mc-search",
        "-c",
        "/work/input.toml",
        "--verbose",
    ]
    assert finalize[3] == "dopingflow-mace"
    assert "vacancies-finalize" in finalize


def test_format_int_list():
    assert format_int_list([1, 2, 3]) == "1, 2, 3"

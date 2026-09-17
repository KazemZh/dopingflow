from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_main_readme_documents_current_staged_workflow():
    text = _text("README.md")
    required = [
        "vacancies-mc-search",
        "vacancies-finalize",
        'parent_pick = "lowest_energy"',
        'output_directory = "vacancy-mc-grace-mace"',
        "vacancy_counts = [1, 2]",
        "supercell = [2, 2, 2]",
        'mc_backend = "grace"',
        "mc_max_steps = 200000",
        "mc_patience = 100000",
        "GRACE top-k",
    ]
    for token in required:
        assert token in text, token


def test_checked_in_vacancy_example_matches_staged_research_design():
    config = _text("examples/vacancies/input.toml")
    for token in [
        'parent_source = "directory"',
        'parent_pick = "lowest_energy"',
        'output_directory = "vacancy-mc-grace-mace"',
        "parent_include = [",
        'search_method = "monte-carlo"',
        "vacancy_counts = [1, 2]",
        "supercell = [2, 2, 2]",
        'mc_backend = "grace"',
        'mc_model = "GRACE-1L-OMAT"',
        "mc_max_steps = 200000",
        "mc_patience = 100000",
        'backend = "mace"',
        'model = "mh-1"',
    ]:
        assert token in config, token

    example_readme = _text("examples/vacancies/README.md")
    assert "vacancies-mc-search" in example_readme
    assert "vacancies-finalize" in example_readme
    assert "MACE does not yet rescore every archived GRACE" in example_readme


def test_sphinx_docs_cover_staged_commands_and_parent_routing():
    method = _text("docs/source/methods/vacancies.rst")
    staged_reference = _text("docs/source/input_file_vacancy_mc.rst")
    install = _text("docs/source/installation_and_usage.rst")
    index = _text("docs/source/index.rst")

    for text in (method, staged_reference, install):
        assert "vacancies-mc-search" in text
        assert "vacancies-finalize" in text

    for token in ("parent_include", "parent_pick", "output_directory"):
        assert token in method
        assert token in staged_reference

    assert "input_file_vacancy_mc" in index


def test_gui_readme_and_staged_page_document_split_environments():
    readme = _text("gui/README.md")
    page = _text("gui/pages/Vacancy_MC_Staged.py")
    helper = _text("gui/vacancy_staged.py")

    for token in [
        "Vacancy_MC_Staged.py",
        "vacancies-mc-search",
        "vacancies-finalize",
        "parent_include",
        "parent_pick",
        "output_directory",
    ]:
        assert token in readme, token

    assert "Run GRACE MC search" in page
    assert "Run MACE finalize" in page
    assert '"search": "vacancies-mc-search"' in helper
    assert '"finalize": "vacancies-finalize"' in helper


def test_changelog_no_longer_claims_vacancies_is_only_public_command():
    text = _text("CHANGELOG.md")
    assert "vacancies-mc-search" in text
    assert "vacancies-finalize" in text
    assert "exposed only as" not in text

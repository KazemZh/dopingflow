import importlib.util
from pathlib import Path


def _load_gui_config():
    path = Path(__file__).resolve().parents[1] / "gui" / "gui_config.py"
    spec = importlib.util.spec_from_file_location("dopingflow_gui_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gui_correction_defaults_are_opt_in_and_scientifically_conservative():
    config = _load_gui_config()
    defaults = config.DEFAULTS["energy_correction"]
    assert defaults["enabled"] is False
    assert defaults["experimental_source"] == "kingsbury"
    assert defaults["correction_terms"] == ["oxide"]
    assert defaults["model_family"] == "manual"
    assert defaults["m1_elements"] == "workflow"
    assert defaults["calibration_selection"] == "manifest"
    assert defaults["auto_fetch_phase_structures"] is False
    assert defaults["allow_phase_mismatch"] is False
    assert defaults["allow_legacy_candidate_provenance"] is False
    assert defaults["dataset_cache_dir"] == ""
    assert defaults["poor_fit_rmse_warning_eV_per_atom"] == 0.20


def test_gui_exposes_all_experimental_source_modes_and_fit_stage():
    config = _load_gui_config()
    assert config.CHOICES["energy_correction.experimental_source"] == [
        "kingsbury",
        "kingsbury+custom",
        "custom",
    ]
    assert config.CHOICES["energy_correction.model_family"] == [
        "manual",
        "m0",
        "m1",
        "auto",
    ]
    assert config.STEP_KEYS.index("corrections") == config.STEP_KEYS.index("refs") + 1


def test_gui_reference_oxygen_defaults_use_explicit_new_semantics():
    config = _load_gui_config()
    references = config.DEFAULTS["references"]

    assert references["oxygen_mode"] == "O-rich"
    assert references["oxygen_reference_correction_ev"] == 0.0
    assert references["delta_mu_O_ev"] == 0.0
    assert "muO_shift_ev" not in references


def test_gui_vacancy_oxygen_defaults_remain_independent():
    """Synchronizing [references] must not rewrite the vacancy thermodynamics API."""
    config = _load_gui_config()
    vacancies = config.DEFAULTS["vacancies"]

    assert vacancies["oxygen_reference_mode"] == "reference_file"
    assert vacancies["oxygen_reference_file"] == "reference_structures/reference_energies.json"
    assert vacancies["delta_mu_O_min_eV"] == -3.0
    assert vacancies["delta_mu_O_max_eV"] == 0.0
    assert vacancies["delta_mu_O_points_eV"] == [
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -2.5,
        -3.0,
    ]


def test_gui_app_contains_explicit_reference_oxygen_and_advanced_correction_controls():
    app_path = Path(__file__).resolve().parents[1] / "gui" / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert '"oxygen_reference_correction_ev"' in source
    assert '"delta_mu_O_ev"' in source
    assert '"dataset_cache_dir"' in source
    assert '"poor_fit_rmse_warning_eV_per_atom"' in source

    # The vacancy UI remains a separate thermodynamic namespace and must not be
    # replaced by the [references] oxygen controls.
    assert 'vac["oxygen_reference_mode"]' in source
    assert 'vac["delta_mu_O_min_eV"]' in source
    assert 'vac["delta_mu_O_max_eV"]' in source

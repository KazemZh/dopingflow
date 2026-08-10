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

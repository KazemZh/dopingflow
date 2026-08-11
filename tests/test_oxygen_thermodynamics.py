from __future__ import annotations

import pytest

from dopingflow.oxygen_thermodynamics import (
    oxygen_energy_state,
    parse_oxygen_thermodynamics_config,
)


def _cfg(**references):
    return {
        "references": {
            "oxygen_mode": references.pop("oxygen_mode", "O-rich"),
            **references,
        },
        "energy_correction": {"enabled": False},
    }


def test_zero_legacy_shift_remains_harmless():
    raw = _cfg(muO_shift_ev=0.0)
    settings = parse_oxygen_thermodynamics_config(raw)
    assert settings.oxygen_reference_correction_eV_per_O == 0.0
    assert settings.delta_mu_O_eV_per_O == 0.0
    assert settings.legacy_interpretation == "zero_legacy_alias"


def test_nonzero_legacy_shift_preserves_historical_reference_correction_without_fit():
    raw = _cfg(muO_shift_ev=-0.25)
    settings = parse_oxygen_thermodynamics_config(raw)
    assert settings.oxygen_reference_correction_eV_per_O == -0.25
    assert settings.delta_mu_O_eV_per_O == 0.0
    assert settings.legacy_interpretation == "historical_empirical_oxygen_reference_correction"


def test_fitted_energy_correction_rejects_empirical_oxygen_reference_shift():
    raw = _cfg(oxygen_reference_correction_ev=-0.25)
    raw["energy_correction"]["enabled"] = True
    with pytest.raises(ValueError, match="double count"):
        parse_oxygen_thermodynamics_config(raw)


def test_fitted_energy_correction_rejects_nonzero_legacy_shift_as_ambiguous():
    raw = _cfg(muO_shift_ev=-0.25)
    raw["energy_correction"]["enabled"] = True
    with pytest.raises(ValueError, match="ambiguous"):
        parse_oxygen_thermodynamics_config(raw)


def test_physical_o_poor_delta_mu_is_allowed_with_fitted_correction():
    raw = _cfg(
        oxygen_mode="O-poor",
        oxygen_reference_correction_ev=0.0,
        delta_mu_O_ev=-0.5,
    )
    raw["energy_correction"]["enabled"] = True
    settings = parse_oxygen_thermodynamics_config(raw)
    assert settings.oxygen_reference_correction_eV_per_O == 0.0
    assert settings.delta_mu_O_eV_per_O == -0.5


def test_o_rich_requires_zero_physical_delta_mu():
    raw = _cfg(delta_mu_O_ev=-0.1)
    with pytest.raises(ValueError, match="O-rich"):
        parse_oxygen_thermodynamics_config(raw)


def test_o_poor_rejects_positive_delta_mu():
    raw = _cfg(oxygen_mode="O-poor", delta_mu_O_ev=0.1)
    with pytest.raises(ValueError, match="<= 0"):
        parse_oxygen_thermodynamics_config(raw)


def test_reference_correction_and_physical_delta_are_applied_in_separate_steps():
    raw = _cfg(
        oxygen_mode="O-poor",
        oxygen_reference_correction_ev=0.2,
        delta_mu_O_ev=-0.5,
    )
    settings = parse_oxygen_thermodynamics_config(raw)
    state = oxygen_energy_state(-10.0, settings)

    assert state.E_O2_raw_eV_per_molecule == pytest.approx(-10.0)
    assert state.E_O2_reference_eV_per_molecule == pytest.approx(-9.6)
    assert state.mu_O_rich_eV_per_O == pytest.approx(-4.8)
    assert state.mu_O_used_eV_per_O == pytest.approx(-5.3)
    assert state.E_O2_effective_eV_per_molecule == pytest.approx(-10.6)


def test_nonzero_legacy_and_new_keys_cannot_be_mixed():
    raw = _cfg(muO_shift_ev=-0.1, delta_mu_O_ev=-0.2, oxygen_mode="O-poor")
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_oxygen_thermodynamics_config(raw)

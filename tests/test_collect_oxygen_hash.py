from __future__ import annotations

from dopingflow.collect_relative import _with_resolved_oxygen_hash
from dopingflow.oxygen_thermodynamics import (
    oxygen_settings_hash,
    parse_oxygen_thermodynamics_config,
)


def test_collect_reconstructs_same_oxygen_hash_as_formation() -> None:
    raw = {
        "references": {
            "oxygen_mode": "O-poor",
            "oxygen_reference_correction_ev": 0.0,
            "delta_mu_O_ev": -0.25,
        },
        "energy_correction": {"enabled": True},
        "formation": {"normalize": "per_dopant"},
    }

    forwarded = _with_resolved_oxygen_hash(raw)
    expected = oxygen_settings_hash(parse_oxygen_thermodynamics_config(raw))

    assert forwarded["formation"]["_oxygen_settings_hash"] == expected
    assert forwarded["formation"]["normalize"] == "per_dopant"
    assert "_oxygen_settings_hash" not in raw["formation"]


def test_collect_hash_changes_with_physical_delta_mu_o() -> None:
    rich = {
        "references": {
            "oxygen_mode": "O-rich",
            "oxygen_reference_correction_ev": 0.0,
            "delta_mu_O_ev": 0.0,
        },
        "energy_correction": {"enabled": True},
        "formation": {},
    }
    poor = {
        "references": {
            "oxygen_mode": "O-poor",
            "oxygen_reference_correction_ev": 0.0,
            "delta_mu_O_ev": -0.5,
        },
        "energy_correction": {"enabled": True},
        "formation": {},
    }

    rich_hash = _with_resolved_oxygen_hash(rich)["formation"]["_oxygen_settings_hash"]
    poor_hash = _with_resolved_oxygen_hash(poor)["formation"]["_oxygen_settings_hash"]

    assert rich_hash != poor_hash

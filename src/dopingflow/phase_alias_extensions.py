"""Small phase-label aliases observed in the Kingsbury calibration dataset."""

from __future__ import annotations

from dopingflow import phase_structure_fallback as _phase

# Kingsbury uses ``trigon`` for trigonal in entries such as Ti2O3.  This is a
# direct abbreviation, unlike structure-type names such as ``rutile`` which
# must not be reduced to a crystal-system-only match.
_phase._PHASE_ALIASES.setdefault("trigon", "trigonal")

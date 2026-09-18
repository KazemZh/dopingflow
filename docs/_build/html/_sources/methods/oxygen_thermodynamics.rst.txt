Oxygen Energy Convention and Chemical Potential
================================================

``dopingflow`` separates an electronic reference-energy correction from the
physical oxygen chemical potential.  They are numerically similar but have
different meanings and must not be conflated.

Configuration
-------------

Use the following keys in ``[references]``::

   oxygen_mode = "O-rich"  # or "O-poor"
   oxygen_reference_correction_ev = 0.0
   delta_mu_O_ev = 0.0

Both energy-like values are in eV per O atom.

``oxygen_reference_correction_ev``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is an empirical/electronic correction to the raw same-backend O2
reference.  If the raw ML O2 energy is ``E_O2_raw``::

   E_O2_reference = E_O2_raw + 2 * oxygen_reference_correction_ev

This setting changes the electronic reference convention itself.

When ``[energy_correction].enabled = true``, this value must be zero.  The
experimental correction model is fitted against raw same-backend formation
energies, so applying an additional empirical oxygen-reference correction would
double count an oxygen-linear correction.  ``dopingflow`` raises an error rather
than silently combining both.

``delta_mu_O_ev``
~~~~~~~~~~~~~~~~~

This is a physical thermodynamic chemical-potential shift relative to the
O-rich reference::

   mu_O_rich = 0.5 * E_O2_reference
   mu_O      = mu_O_rich + delta_mu_O_ev

It does not represent an electronic-structure correction.  Therefore it may be
used together with a fitted energy-correction model.

For ``oxygen_mode = "O-rich"``, ``delta_mu_O_ev`` must be zero.  For
``oxygen_mode = "O-poor"``, it must be zero or negative.

Legacy ``muO_shift_ev``
-----------------------

``muO_shift_ev`` is retained only as a migration aid because its old meaning was
ambiguous.

- ``muO_shift_ev = 0.0`` remains harmless.
- A non-zero legacy value without ``[energy_correction]`` preserves the old
  numerical behavior and is interpreted as an empirical oxygen-reference
  correction.
- A non-zero legacy value is rejected when ``[energy_correction].enabled = true``.
- A non-zero legacy value cannot be mixed with either new oxygen key.

For new input files, use the explicit keys above and remove ``muO_shift_ev``.

Examples
--------

Fitted correction model at O-rich conditions::

   [references]
   oxygen_mode = "O-rich"
   oxygen_reference_correction_ev = 0.0
   delta_mu_O_ev = 0.0

   [energy_correction]
   enabled = true

Fitted correction model at a chosen O-poor condition::

   [references]
   oxygen_mode = "O-poor"
   oxygen_reference_correction_ev = 0.0
   delta_mu_O_ev = -0.50

   [energy_correction]
   enabled = true

No fitted energy correction, but an explicitly chosen empirical O-reference
shift::

   [references]
   oxygen_mode = "O-rich"
   oxygen_reference_correction_ev = -0.20
   delta_mu_O_ev = 0.0

   [energy_correction]
   enabled = false

Reproducibility
---------------

``refs-build`` stores the resolved convention in
``reference_structures/reference_energies.json``.  The formation stage also
stores a hash in ``reference_structures/formation_oxygen_state.json`` and
rebuilds formation outputs when the oxygen convention changes.

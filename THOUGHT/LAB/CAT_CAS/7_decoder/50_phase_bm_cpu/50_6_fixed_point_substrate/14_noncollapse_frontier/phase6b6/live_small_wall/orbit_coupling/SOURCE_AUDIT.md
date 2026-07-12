# OrbitState Coupling Source Audit

Date: 2026-07-12

Scope:

- `small_wall_runtime.c`
- `small_wall_runtime.h`
- `small_wall_worker.c`
- `live_gate_a_target.py`
- `coded_preprojection_query_model.py`
- `f10_pmc_first_light_worker.c`
- `f10_pmc_first_light_target.py`
- retained coded pre-projection checkpoints dated 2026-07-12

## Corrected Conclusion

Previous public-phase physical mappings did not test actual OrbitState-to-carrier
coupling. They tested public P/M/C stimulus classes decoded as though they represented
private-fold alternatives.

This narrows the claim of the retained evidence. It does not erase the value of the
captures: the active-query, source-phase-chop, and combined runs remain clean negative
tests for those public schedule mappings under their stated controls.

## Source Findings

The live coded pre-projection path is schedule-token driven. `small_wall_runtime.c`
declares public token sequences such as `P0..P3`, `M0..M3`, and `C0..C3`, and
`small_wall_worker.c` validates those exact public sequences and schedule hashes.

The coded/read-only path never carries a private orbit member through
`gate_a_orbit_value()`. In `small_wall_runtime.c`, `gate_a_orbit_value()` returns `-1`
whenever `gate_a_readonly_occupancy_pilot()` is active, and coded pre-projection pilots
are included in that read-only occupancy family. The later trace writer can therefore
only emit `null` or inherited placeholder values for `orbit_value` in coded runs.

The source-side coded operators are public schedule operators:

- `gate_a_occupancy_bytes(slot)` maps `P/M/C` tokens to large/equal/small footprints.
- `small_wall_source_phase_chop_reads()` maps the public token phase to four in-slot
  footprint segments.
- `gate_a_active_query_sample()` selects receiver positive/negative response subbanks
  from the public phase index.

None of these source paths takes an `OrbitState { modulus, member }` or computes
`r_theta(d) = cos(2*pi*d/N - theta)` from a hidden member.

`coded_preprojection_query_model.py` is a valid non-driving access-law model, but its
`pre_projection_private_fold_plus` and `pre_projection_private_fold_minus` entries are
analytical branch alternatives. The live coded implementation later reuses those names
for decoded public schedule groups; the names should not be interpreted as proof that a
real private member entered the physical source interaction.

The retained checkpoints confirm the same boundary:

- `CODED_PREPROJECTION_ACTIVE_QUERY_CHECKPOINT_20260712.json` binds the run to the
  public `P/C/M` phase-local sequence and an active receiver-delta law.
- `CODED_PREPROJECTION_SOURCE_PHASE_CHOP_CHECKPOINT_20260712.json` binds the run to a
  public source-side phase-chop waveform.
- `CODED_PREPROJECTION_ACTIVE_SOURCE_CHOP_CHECKPOINT_20260712.json` combines those two
  public mechanisms and remains negative.

`F10_COHERENCE_OPERATOR_CHECKPOINT_20260712.json` establishes a useful substrate
operator: byte-preserving remote same-value store moved `Change-to-Dirty` strongly
against identity/read/same-core controls. That operator is a reasonable first physical
encoder candidate, but it has not yet been coupled to a real private OrbitState member.

## Required Repair

The missing bridge is:

```text
private orbit member
-> declared public quadrature query
-> balanced source-side physical encoding
-> receiver observes only physical response and public query
-> frozen fold-odd decoder
-> measured restoration
```

The implementation in this directory must therefore introduce a real source-side
`OrbitState` object and keep the private member out of the receiver manifest and feature
extraction path until after frozen features are hashed.

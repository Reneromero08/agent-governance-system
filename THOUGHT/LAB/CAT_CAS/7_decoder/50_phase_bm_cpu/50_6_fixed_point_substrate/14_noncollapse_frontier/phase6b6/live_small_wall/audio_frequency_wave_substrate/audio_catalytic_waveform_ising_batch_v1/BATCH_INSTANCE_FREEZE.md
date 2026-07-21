# Catalytic Waveform-Ising Held-Out Batch Freeze

Status: `FROZEN_BEFORE_EXECUTION_AND_ORACLE`

This package prospectively freezes sixteen unseen five-site Ising instances for the
unchanged machine established by `audio_integrated_catalytic_computation_v1` and used by
`audio_catalytic_waveform_ising_heldout_v1`.

Only `J` and `h` may vary. The public seed, coefficient sets, generation rule, ordered
instances, exclusions, and promotion criterion are recorded byte-for-byte in
`BATCH_INSTANCE_CUSTODY.json`.

The batch is generated without native waveform execution, oracle access, result-based
selection, or machine tuning. A duplicate complete `(J,h)` pair causes an abort; it is
never replaced after observation.

The prospectively frozen promotion criterion requires all of the following:

- exactly 16 executed instances and at least 12 unique-optimum instances;
- at least 8 `ACCEPTED_CORRECT` instances and an accepted-correct rate of at least 0.50
  among unique-optimum instances;
- zero `ACCEPTED_INCORRECT` and zero `UNINTERPRETABLE` instances;
- complete machine-identity, oracle-order, causal, restoration, reuse, and control
  integrity.

The coherence threshold remains exactly `0.90`. No threshold may be moved after batch
execution begins.

This freeze does not contain, imply, or authorize any batch result.

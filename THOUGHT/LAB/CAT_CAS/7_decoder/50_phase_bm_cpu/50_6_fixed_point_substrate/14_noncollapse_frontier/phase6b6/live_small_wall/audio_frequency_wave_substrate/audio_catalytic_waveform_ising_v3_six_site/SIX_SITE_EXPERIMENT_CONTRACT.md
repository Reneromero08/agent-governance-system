# V3 Six-Site Recursive Spectral-Phase Experiment Contract

## Question

Does the verified five-site V3 causal construction extend unchanged in kind to
six sites while retaining exact unique-optimum output, tied-optimum rejection,
inverse restoration, carrier reuse, and boundary-only selection?

## Frozen dimension law

For site count `n`, the complete recursive antipodal tree contains `2^n` phase
modes. The active-bin count equals the mode count. The complex sample count is
eight times the mode count. The relational phase denominator is twice the mode
count, so the maximum bounded six-site penalty of 84 maps inside a non-wrapping
phase interval:

```text
n = 6
mode count = 64
active bins = 64
sample count = 512
relational phase denominator = 128
relation phase scale = pi / 128
maximum penalty = 2 * (15 * 2 + 6 * 2) = 84
```

The active bins begin at bin 8. Geometry anchors follow the same global
`(11 + 7*k) mod sample_count` law used by the parameterized family.

## Native/boundary separation

Native execution may use complex carrier state, recursive phase geometry,
spectral transforms, and reversible J/h phase relations. It may not calculate
energy, call an oracle, perform scalar `J@s`, decode spins into recurrence, use
an answer cache, or perform candidate selection.

Mode comparison, uniqueness-gap checking, ambiguity rejection, and antipodal
spin extraction occur only in `project_boundary`, after native evolution.

## Prospective batch

The batch contains 256 ordered six-site problems produced by the frozen public
SHA-256 generator. Complete identities already used in the 512-case development
set and duplicate generated identities are skipped. No outcome, difficulty,
uniqueness, waveform result, energy, or oracle property may influence batch
membership.

The exact source, analyzers, thresholds, batch, and promotion law must be
committed and pushed before prospective waveform execution. All native evidence
must be sealed, committed, and pushed before exact 64-state oracle adjudication.

## Promotion law

Verification requires every observed unique optimum to be produced and
accepted, every non-unique optimum to be rejected, no incorrect acceptance, no
correct unique rejection, at least 160 unique cases, all 256 strict-control
passes, all restoration and reuse passes, zero uninterpretable cases, and zero
oracle or energy calls before the pre-oracle seal.

## Claim ceiling

```text
BOUNDED_SOFTWARE_RECURSIVE_SPECTRAL_PHASE_REFERENCE_ONLY
```

This experiment cannot establish computational advantage, favorable asymptotic
scaling, physical waveform computation, hardware persistence, bit replacement,
or a Wall crossing.

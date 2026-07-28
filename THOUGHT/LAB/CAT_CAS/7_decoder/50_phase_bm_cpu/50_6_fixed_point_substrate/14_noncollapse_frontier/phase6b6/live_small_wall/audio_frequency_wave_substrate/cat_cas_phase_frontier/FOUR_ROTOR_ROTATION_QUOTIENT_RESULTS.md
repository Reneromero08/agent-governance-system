# Four-Rotor Global-Rotation Quotient Results

Evidence:

```text
/tmp/four-rotor-rotation-quotient-repaired.EGpI53
```

Bounded claim:

```text
BOUNDED_EXACT_GLOBAL_ROTATION_QUOTIENT_CYCLIC_PHASE_CARRIER_REDUCES_FOUR_ROTOR_STATE_BY_GRID_FACTOR_WITH_DEPTH_INDEPENDENT_MEMORY_ACTUAL_RESTORATION_AND_REUSE
```

For rotation-invariant four-rotor programs, the phase carrier removes the
global angle exactly. It stores three relative angles
`(theta1-theta0, theta2-theta0, theta3-theta0)`. In the conjugate basis the
fourth momentum is derived by the public constraint
`n0 = -(n1+n2+n3) mod 17`, so the free phase is native on the quotient.
Nearest-neighbor couplings remain direct diagonal phase multipliers.

This reduces resident state from `17^4 = 83,521` to `17^3 = 4,913` complex
cells, exactly one grid factor. The quotient state, public plan, and explicit
engine/wrapper signatures remain fixed over depths `1,2,4,8,16,32,64`:

```text
quotient carrier                              4,913 cells / 78,608 bytes
retained total-momentum free-phase plan                    78,880 bytes
plan-compilation explicit peak                            157,760 bytes
maximum explicit engine arrays                           236,368 bytes
wrapper arrays including verification baseline           314,976 bytes
retained inverse history                                        0 bytes
```

The quotient is lifted into the full four-angle space for verification. Full
state error against independent dense execution remains below `2.048e-14`
through depth 64, and boundary error remains below `8.438e-15`. The dense
verification path—including the dense reference, one in-place-normalized
lift, relative index arrays, quotient carrier, plan, and baseline—peaks at
2,915,840 explicit bytes. This verification materialization is not part of
the accepted quotient execution.

Depth-32 actual inverse restoration is `8.614e-15`. An unrelated depth-11
program consumes the same backing allocation at restoration generation two
and restores at `2.310e-15`; fresh/restored boundary disagreement is
`7.994e-15`. The baseline is counted and never reloaded. Missing, wrong, and
noncommuting reordered inverse controls separate.

This is an exact structural phase-coordinate quotient for the declared
rotation-invariant, total-momentum-zero-mod-17 sector. It does not apply to
the earlier onsite-kicked program, and its state still scales as
`N^(rotors-1)`. PocketFFT internal workspace is outside the explicit-array
bound. The matched classical quotient is identical, so no distinct phase
resource, computational advantage, Small Wall crossing, unbounded
computation, or physical waveform execution is established.


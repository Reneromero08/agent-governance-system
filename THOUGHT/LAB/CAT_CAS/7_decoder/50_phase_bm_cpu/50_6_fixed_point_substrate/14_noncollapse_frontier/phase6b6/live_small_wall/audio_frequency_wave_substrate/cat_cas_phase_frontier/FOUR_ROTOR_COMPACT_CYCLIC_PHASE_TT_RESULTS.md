# Four-Rotor Compact Cyclic Phase-TT Results

Evidence:

```text
/tmp/four-rotor-compact-cyclic-tt-repaired.zELuOK
```

Bounded obstruction claim:

```text
BOUNDED_CYCLIC_PHASE_TT_NATIVE_PAIR_CLOSURE_SATURATES_GRID17_CENTRAL_INTERFACE_AT_DEPTH5_WITH_ACTUAL_RESTORATION_AND_REUSE
```

The dense finite-torus update law was moved into a four-site tensor-train
carrier. Onsite phases act directly on site tensors. Each nearest-neighbor
phase factor multiplies the actual two-site amplitude and is split by a
declared-tolerance SVD. Free evolution is the exact local cyclic Fourier
phase matrix. The inverse conjugates and reverses those actual operations
without retained history.

At pair-gate L2 tolerance `2e-12`, the central physical Schmidt rank grows:

```text
depth                              1       2       3       4       5
central boundary rank             17      92      201     280     289
installed central bond rank       17      95      207     281     289
maximum logical TT cells       10,404  55,488  120,224 162,996 167,620
```

The depth-five rank equals the exact `17^2 = 289` central interface ceiling.
Logical TT storage is `2.007x` the 83,521-cell dense wave, and the pair update
materializes an 83,521-cell dense-equivalent two-site core from depth two.
No physical-coordinate dense wave is formed, but this is not a compact path.
Only logical TT and core cell counts are claimed; simultaneous SVD/workspace
peak payload is not established.

Shared boundary observables agree with the dense cyclic reference within
`5.792e-15` across depths one through five. The depth-five actual inverse is
hard-gated before closure at `1.041e-9`; baseline-free canonical closure
returns ranks `1,1,1` and restoration error `1.141e-9`. An unrelated program
consumes the same restored carrier at generation two, and fresh/restored
boundaries agree within `4.338e-15`. Missing, wrong, and noncommuting
reordered inverse controls separate.

This identifies:

```text
COMPACT_CYCLIC_PHASE_TT_CENTRAL_RANK_SATURATES_DENSE_INTERFACE
```

The matched classical cyclic TT is identical. No fixed-rank closure,
distinct phase resource, advantage, Small Wall crossing, unbounded
computation, or physical waveform execution is established. The next repair
must use an exact phase-owned quotient or conservation law rather than
another generic TT truncation or larger cyclic fixture.


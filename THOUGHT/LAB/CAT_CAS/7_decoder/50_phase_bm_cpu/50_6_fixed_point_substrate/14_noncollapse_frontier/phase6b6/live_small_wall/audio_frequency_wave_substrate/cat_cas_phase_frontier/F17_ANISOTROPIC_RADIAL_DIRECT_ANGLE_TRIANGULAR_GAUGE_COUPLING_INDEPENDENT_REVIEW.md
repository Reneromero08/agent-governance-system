# M148 independent review

Decision: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`.

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`.

Resident 51-angle carrier restoration:
`NUMERICAL_PHYSICAL_STATE_RESTORATION`.

Transient update, projection, comparison, and verification buffers:
`NO_RESTORATION_CLAIM`.

M148 removes the per-update complex-base decode, Cartesian chart scratch,
chart reconstruction, and retained Givens plan from the accepted phase update.
Each public step translates all 17 phase triplets, lets one rotating resident
hub gauge control 16 target triplets, then lets those target gauges control the
hub in a reciprocal layer.  The two layers do not commute.  Each edge is a
common phase translation of one triplet and therefore preserves that
triplet's decoded base magnitude.

The bounded causality witness begins with two carriers whose 17 decoded base
values agree within `5.888e-16` while their gauge phasors differ by `0.714`.
The same public depth-4 program separates final decoded states by `0.0131` and
final boundaries by `0.0462`.  Reverse execution restores both actual
51-angle carriers within `2.704e-15`.  Thus the base-only 17-complex quotient
is insufficient for this declared variable-gauge carrier family.

Across 21 cases at depths 1, 4, 16, 64, 256, 1024, and 4096, final angle bytes
are identical to the separately executed matched 51-angle recurrence.  The
maximum boundary difference is `4.826e-15`; maximum single-transaction
restoration error is `9.860e-13`.  Unrelated depth-1537 reuse agrees with a
fresh carrier within `1.714e-13`.  One hundred depth-64 cycles preserve the
same backing and restore within `2.780e-13`.  No snapshot, inverse history,
retained restoration baseline, or post-inverse reload is used.

The originally attempted coupling strength `5/32` is not accepted.  At depth
4096 it produces a restoration error of `1.925`, showing that the direct sine
shear can be numerically ill-conditioned.  The declared bounded strength
`1/32` restores within `9.860e-13` for the same attack.  This is a measured
bounded numerical envelope, not evidence of unbounded-depth stability.

The independent oracle imports neither M148 production nor M146.  It shares
only M145's established public phase exponent and shell weights, then
separately implements the seed chart, triangular schedule, forward/inverse,
commitments, boundary, causality witness, mutations, reuse paths, inverse
conditioning attack, and a second identical-angle recurrence.  All 129
declared comparisons pass.

Frozen package hashes are:

```text
production source  45788de89f2e772e2f8e1b11b3d63ca7901ce1dbab14c507587841afaf638d8b
oracle source      2b288ad52a83b1fd270cf809e1d97964a71c36abdf7b9f44e1454dabbed39b8f
production JSON    a4ee5cdc250c06991baf5a4d835d428e659203c5f120355dafa4477219524b64
oracle JSON        748d6f2011e7e4fa970b0cc858cffc3344101953e19d317c102146be1b2cc017
qualifier          a64dd18e88c2ce30a609ec2672ba176da492a8399f57cf1ca76a64b478a8bfb2
predecessor        27a147d45415e97de5fafd9436cfce7a3f1cb0ae0cbd3410189560282a760590
```

The accepted path retains 51 phase angles (408 bytes), no public plan, no
complex state, and no dense kernel.  Its largest named update scratch is four
`float64` cells and its named warm peak including the public program is 750
bytes.  Python, NumPy, allocator, and native-library memory remain excluded.

The strongest executed comparison is the identical 51-`float64` angle
recurrence.  It has the same resident state, operation counts, and named warm
peak and reproduces final carrier bytes exactly.  Consequently M148
establishes a direct software phase primitive and a causally relevant gauge
fiber, but no resource beyond compact scalar software.

The result establishes neither an optimal classical lower bound, general
relational contraction, exact algebraic semantics, unbounded numerical
stability, CATVM custody, a distinct phase resource, computational advantage,
Small Wall crossing, physical waveform execution, replacement of physical
bits with pi, nor unbounded catalytic computation.

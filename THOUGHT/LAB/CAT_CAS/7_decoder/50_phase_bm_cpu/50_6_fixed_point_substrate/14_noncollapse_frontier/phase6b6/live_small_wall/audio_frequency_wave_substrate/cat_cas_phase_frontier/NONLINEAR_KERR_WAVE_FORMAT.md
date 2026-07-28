# Nonlinear Kerr/Interference Wave Carrier

## Status

This bounded software experiment establishes:

```text
BOUNDED_FIXED_RANK_NONLINEAR_KERR_INTERFERENCE_WAVE_CARRIER_WITH_ACTUAL_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_FOUR_COMPLEX_CELL_NORMALIZED_KERR_SU2_WAVE_MESH_DEPTHS1_4_32_128_512_2048_DOUBLE_COMPLEX_SOFTWARE_REFERENCE_ONLY
```

Here `fixed rank` means only that the carrier dimension remains four complex
cells as public depth grows. It does not mean fixed relational or tensor
rank, bounded computational complexity, a distinct phase resource, or a
Small Wall crossing.

## Native wave law

The borrowed carrier is a normalized four-cell complex wave:

```text
psi = (psi_0, psi_1, psi_2, psi_3)
sum_j |psi_j|^2 = 1
```

Each public layer first applies an intensity-dependent Kerr phase:

```text
psi_j <- psi_j exp(i k |psi_j|^2)
```

and then applies two disjoint `SU(2)` interference couplers. Pairing
alternates between `(0,1),(2,3)` and `(1,2),(3,0)`. The couplers change
resident amplitudes by interference; no complex entry is decoded and split
back into phasors.

The inverse traverses public layers in reverse. It applies each coupler's
adjoint and then the inverse Kerr phase. The adjoints first restore the
post-Kerr cells, and the original Kerr update preserved each cell intensity,
so the inverse kick can be recomputed from actual resident state. No inverse
matrix or intermediate wave field is retained.

Only three final real boundary observables are projected. The actual forward
path is then inverted, and sixteen unrelated transactions consume the same
restored carrier.

## Evidence and limits

Depths `1,4,32,128,512,2048` pass. At depth 2048 the fixed carrier performs
`16,384` Kerr cell updates and `8,192` interference couplers across forward
and inverse execution. Maximum norm error is `5.773e-15`; restoration error
is `1.207e-11` against the predeclared `2e-10` tolerance. Reuse restoration
across sixteen varying programs remains below `2.935e-14`.

An independent eight-double scalar recurrence has the same 64-byte semantic
state and `O(depth)` work. Its quantized boundary matches through depth 512.
At depth 2048 cross-implementation parity is deliberately not claimed;
floating-point trajectory sensitivity is not normalized away.

Evidence:

```text
/tmp/nonlinear-kerr-wave-recorded.dCFXDX/evidence
```

The result removes the paired-phasor machine's host amplitude split and
memoryless gauge canonicalization. It does not yet provide CATVM-enforced
custody, an advantage over the exactly matched compact complex recurrence,
physical waveform execution, or unlimited catalytic computation.

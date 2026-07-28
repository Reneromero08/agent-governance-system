# Parity-closed Stokes harmonic-sector format

## Phase-owned reduction

The normalized Stokes carrier already enforces
`x^2+y^2+z^2=1`. Its quadratic seed and generator have even parity, while
each Lie-Poisson bracket with the quadratic generator changes total-degree
parity once. Therefore grade `g`, with degree limit `d=2+g`, can only contain
canonical monomials whose total degree has the same parity as `d`.

The compiler now allocates exactly that topology-derived invariant sector:

```text
z exponent in {0,1}
total degree <= d
total degree mod 2 = d mod 2
```

This sector is closed under the tested quadratic Stokes Lie-Poisson update
and exact sphere reduction, which changes degree only by even amounts. It is
not numerical pruning or an answer-dependent sparse allocation.

## Carrier and custody

Every legal coefficient remains an `F17` and `F19` root phase. Native Fourier
field multiplication consumes the actual resident prior grade without
decoding it. Only the final five-grade boundary is projected. Reverse
Lie-Poisson accumulation restores the actual phase cells, and the same
carrier runs a different mixer plus seven more alternating transactions.

The exact rational oracle independently computes the sphere quotient and
hashes every legal coefficient. It additionally takes each highest-degree
homogeneous part modulo `x^2+y^2+z^2`; the resulting nonzero quotient is the
exact certificate that the corresponding highest harmonic shell survives.

## Resource law and ceiling

```text
degree limit                   2    3    4    5    6
parity-admissible cells        6   10   15   21   28
highest harmonic terms         2    3    5    5    7
```

The five-grade basis falls from 135 Stokes quotient cells to 80. Dual-prime
custody falls from 270 to 160 phase cells, or from 4,320 to 2,560 logical
packed payload bytes. CPython object allocation and temporaries are not
measured.

This establishes exact parity-sector reduction and bounded survival of
highest harmonic shells through degrees two to six. It does not establish an
explicit irreducible harmonic decomposition, fixed-rank closure, unbounded
growth, a distinct phase resource, computational advantage, Small Wall
crossing, or physical waveform execution.

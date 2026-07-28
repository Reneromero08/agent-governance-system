# Fixed-rank factorized highest Stokes shell

## Exact factorization law

Let

```text
L = 24*x - 7*z
Q = L^2 / 625
```

and let `D=ad_L` be the Stokes Lie-Poisson rotation derivative. Because
`D(L)=0`,

```text
ad_Q(f) = (2/625) L D(f)
ad_Q^n(z^2) = L^n q_n
q_(n+1) = (2/625) D(q_n).
```

In the orthonormal rational frame

```text
a = (24*x-7*z)/25
b = (7*x+24*z)/25
c = y
```

every `q_n` after the first bracket lies in the four-coordinate space

```text
a*c, a*b, b*c, b^2-c^2.
```

The public factor is stored only as its axis and exponent. No degree-`n+2`
polynomial is expanded.

## Reversible phase representation

For field `Fp`, coefficient `v` occupies the complete unit-phase character
orbit

```text
chi_v(j) = exp(2*pi*i*v*j/p), j=0..p-1.
```

Multiplication by nonzero public scalar `k` is the exact phase-cell
permutation `chi_v(k*j)=chi_(k*v)(j)`. The Q4 recurrence consists only of
two coordinate swaps and four such character permutations per field. Its
inverse uses inverse permutations and reverse swaps on the actual cells.
There is no phase locking, decoded residue shadow, snapshot restoration, or
retained inverse history.

Four coordinates over complete `F17` and `F19` character orbits require
`4*(17+19)=144` unit-phase cells, or 2,304 logical packed bytes, independent
of depth. The public recurrence descriptor is 96 logical packed bytes in the
declared accounting model. Actual CPython allocation and temporaries are not
measured.

## Claim ceiling

The algebraic factorization holds for every positive depth. Software
execution and inverse/reuse evidence are bounded to depths
`1,2,4,8,32,128,512,2048`. An exact rational oracle verifies sphere-quotient
identities through depth six and every modular boundary at all executed
depths.

This is fixed-rank closure of the highest homogeneous harmonic shell for one
repeated single-axis quadratic Kerr generator. It is not fixed-rank closure
of the full multi-shell Stokes signature. A matched classical implementation
needs only eight dual-prime residue bytes, so this result establishes neither
a distinct phase resource nor computational advantage, Small Wall crossing,
unbounded catalytic computation, or physical waveform execution.

Restored-carrier reuse seals a second public factorized Q4 seed into the same
recurrence. It is an unrelated program sentinel, not a claim that the primary
Stokes seed ran under a different physical mixer.

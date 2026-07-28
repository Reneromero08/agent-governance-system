# Symmetry-Preserving Necklace Generator Results

## Result

The free phase update now acts directly on the 285-cell
global-rotation-quotiented necklace carrier. The accepted bounded claim is:

```text
BOUNDED_SYMMETRY_PRESERVING_HERMITIAN_NECKLACE_GENERATOR_PHASE_CLOSURE_ELIMINATES_OCCUPATION_EXPANSION_WITH_ACTUAL_RESTORATION_AND_REUSE
```

The tested ceiling is:

```text
EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_GRID17_FOUR_ROTOR_DEPTH8_CHEBYSHEV_DEGREE64_TESTED_NONZERO_CHIRP_COMPLEX128_SOFTWARE_ONLY
```

Final evidence:

```text
/tmp/four-rotor-necklace-generator-final.P7tXmz
```

The manifest binds the generator source, both comparison dependencies, the
qualifier, exact executed binary, and result. The same source also passes an
ASan/UBSan run.

## Direct quotient law

For each public chirp, the 17-mode single-particle free unitary is circulant.
Its Fourier eigenphases define a Hermitian circulant logarithm `H` satisfying:

```text
C = exp(i H)
```

The lifted four-boson free update is:

```text
Gamma_4(C) = exp(i dGamma(H))
```

On per-labelled-configuration necklace amplitudes, the generator acts without
leaving the quotient:

```text
(dGamma(H) s)[m]
    = sum over occupied i and all j of
      m_i H_ij s[canonical(m - e_i + e_j)]
```

Global rotation commutes with `H`, so every streamed source is immediately
canonicalized to one of the same 285 necklace cells. No 4,845-cell occupation
vector or retained 285-by-285 transition operator is used by the accepted
path.

The exponential uses a degree-64 Chebyshev recurrence with three 285-cell work
vectors. The complete omitted tail is bounded from the Bessel series by:

```text
2 B_65 / (1 - q)

B_n = (r/2)^n / n! * exp(r^2 / (4(n+1)))
q   = r / (2*66)
```

The largest bound across the tested schedule is `6.69717105552036e-41`.

## Exactness and catalytic lifecycle

- one-step error against bosonic Givens: `2.6165889829284694e-15`
- depth-eight boundary error against bosonic Givens: `2.609024107869118e-15`
- maximum single-particle eigenvalue modulus error:
  `7.771561172376096e-16`
- maximum generator Hermitian error: `3.6638411145107034e-16`
- depth-eight weighted norm error: `1.2656542480726785e-14`
- primary actual-inverse restoration error:
  `1.3877500627453903e-14`
- unrelated generation-two restoration error:
  `1.7274307843759177e-14`
- fresh/restored reuse boundary error: `1.27675647831893e-14`

Missing, wrong, and applicably reordered inverse errors are respectively
`1.4040897828772636`, `1.3464039441456184`, and
`1.075079787302776`.

## Resource change

The accepted path uses:

- resident carrier: 285 complex cells / 4,560 bytes
- temporary recurrence: 855 complex cells / 13,680 bytes
- public topology: 10,532 bytes
- 17-by-17 Hermitian generator plan: 4,664 bytes
- occupation scratch: 0 bytes
- retained transition operator: 0 bytes
- retained inverse history: 0 bytes
- maximum explicit engine payload: 33,470 bytes
- maximum explicit wrapper payload: 38,142 bytes

The preceding bosonic Givens accepted path used 77,520 bytes of occupation
scratch and a 97,447-byte explicit engine payload. The direct generator
therefore reduces explicit engine payload by `2.912x` and never releases the
factor-17 global-rotation quotient.

The comparison harness still materializes the predecessor's 4,845-cell,
77,520-byte occupation scratch; those comparison resources are explicitly
separate from the accepted path. Its conservative explicit peak is 120,110
bytes.

The repair trades memory for work. The depth-eight forward/inverse lifecycle
streams 16,868,352 generator terms through 1,024 generator applications. Its
measured warm lifecycle is about `3.36x` slower than bosonic Givens. This is
not a speed or total-work advantage.

## Scientific boundary and next diagnostic

This is a phase-machine repair: unresolved phase amplitudes remain in the
same quotient carrier throughout native collision/free composition,
restoration, and reuse.

The identical compact classical Hermitian quotient recurrence inherits the
same repair. This does not establish a distinct phase resource,
computational advantage, Small Wall crossing, physical execution, or
unbounded computation.

The surviving obstruction is:

```text
CHEBYSHEV_GENERATOR_WORK_AND_MATCHED_CLASSICAL_HERMITIAN_QUOTIENT_IDENTITY
```

The next selected bounded diagnostic is:

```text
MATCHED_COHERENT_DEPHASED_CLASSICAL_NECKLACE_GENERATOR_SMALL_WALL_TRIAD
```

It must hold public instances, carrier dimensions, protocol traffic,
projection, restoration, and reuse semantics fixed while separating coherent
phase interference from a dephased sham and from the best matched compact
classical complex recurrence. It may not claim leverage merely from the
absence of occupation expansion.

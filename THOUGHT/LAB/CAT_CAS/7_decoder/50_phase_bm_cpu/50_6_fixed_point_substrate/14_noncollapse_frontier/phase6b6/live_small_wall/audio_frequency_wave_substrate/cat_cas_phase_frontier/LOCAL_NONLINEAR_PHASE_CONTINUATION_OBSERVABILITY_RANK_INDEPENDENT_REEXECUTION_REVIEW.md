# Local nonlinear phase continuation observability independent review

## Decision

```text
INDEPENDENTLY_VERIFIED_STRICT_SCOPE
INDEPENDENT_ORACLE_REEXECUTION
NUMERICAL_PHYSICAL_STATE_RESTORATION (UNDERLYING VIRTUAL PHASE TRANSACTION)
```

The bounded local-rank result is accepted at its declared scope. It rejects a
regular differentiable local quotient below `2n-1` real dimensions only when
that quotient must preserve all declared suffix boundaries near the tested
resident chart. It is not a global nonlinear quotient theorem, an algorithmic
memory lower bound, or a distinct phase resource.

## Declared chart and continuation family

For n=4, 8, and 16, the M187 local brickwork and intensity-feedback program
of depth n produces one norm-one resident complex phase state. The public
continuation family is fixed before rank evaluation: suffix index j determines
one layer's two coupler angles, feedback strength, and public seed by the
formula in source, while every suffix uses the same final weight shift. There
are exactly n complex suffix boundaries, supplying 2n real observations.

Public continuation compilation does not inspect rank, singular values, final
boundaries, or expected answers. No intermediate amplitude vector is emitted;
the package records commitments, final scalar boundaries, ranks, singular
values, controls, and resource counts.

## Analytic tangent mechanism

The production verifier constructs an orthonormal basis of the norm-one
carrier tangent space, dimension `2n-1`. Pair-coupler derivatives are the same
complex-linear two-cell rotations. For feedback

```text
z -> z exp(i k |z|^2)
```

the real-linear tangent update uses

```text
dz -> exp(i k |z|^2) [dz + i k z 2 Re(conj(z) dz)].
```

Streaming projection of each tangent gives a `2n x (2n-1)` real
observability matrix. With a predeclared relative rank tolerance of `1e-9`,
the observed ranks are 7/7, 15/15, and 31/31. Minimum retained singular values
are approximately `4.87e-3`, `4.25e-5`, and `9.26e-8`; the n=16 value remains
above its `4.02e-9` threshold. Condition numbers rise from about 430 to
`4.34e7`, so the ceiling remains numerical and chart-local.

Removing the final suffix leaves only `2n-2` rows and produces ranks 6, 14,
and 30. Repeating one suffix produces rank 2. Zeroing feedback changes every
analytic matrix. Direct analytic state updates match the frozen M187 forward
law within floating-point roundoff.

## Independent centered-difference oracle

The separate oracle imports neither M187 nor the analytic diagnostic. It
reconstructs source states, prefix and suffix programs, forward and inverse
operations, norm-one tangent bases, projections, and rank controls directly.
For each suffix and tangent direction it normalizes `x +/- 1e-6 v`, executes
both continuations, and forms a centered finite difference.

It independently obtains ranks 7, 15, and 31. The worst relative difference
between corresponding analytic and finite-difference singular values is
`6.30e-4` at n=16; production and oracle selected final boundaries differ by
at most `1.25e-16`. Finite-difference forward counts are 56, 240, and 992 for
the primary matrices, plus equal counts for the zero-feedback controls.

## Restoration and reuse

The accepted transaction applies the actual prefix and one actual suffix,
retains only the final scalar, reverses suffix then prefix on the same backing,
and runs an unrelated prefix/suffix program on that restored carrier. Across
the three cases:

- primary restoration error is at most `6.34e-16`;
- unrelated-reuse restoration error is at most `8.53e-16`;
- 64 alternating reuse cycles accumulate at most `1.10e-14` error;
- reused and fresh boundaries agree within `2e-11`;
- no snapshot or baseline reload is used.

The restoration class is `NUMERICAL_PHYSICAL_STATE_RESTORATION` only for the
virtual complex coordinates. Response ordering is direct-process local code,
not CATVM-enforced custody.

## Resource and claim ceiling

The accepted carrier transaction remains n+2 logical complex cells. Primary
and unrelated-reuse forward/inverse lifecycles execute `4(n^2+n)` couplers and
the same number of feedback operations. Repeated reuse, missing/wrong/reordered
inverse controls, and the fresh comparison execute a separately reported
`(2*64+4)(n^2+n)` of each. Accepted boundaries stream 2n projection terms;
fresh verification streams n. Two commitments serialize `2*16*n` bytes.

The rank certificate is verification, not free phase work: production
conservatively reports `6n(2n-1)+10n-1` named real scalars, or 207, 799, and
3,135, plus 7n public suffix descriptor scalars and 14 prefix descriptor
scalars. Descriptor JSON bytes are measured per case. Analytic execution
reports `2n^2` state couplers and feedbacks, `n^2(2n-1)` tangent couplers and
feedbacks, `n^2` boundary projection terms, and `n^2(2n-1)` tangent projection
terms. The finite-difference oracle's forward counts and resulting local
operation totals remain separate.

SVD/native-library workspace, Python containers, allocator details, arithmetic
temporaries, and whole-process memory are not included in the named-scalar
figure.

Full local tangent observability means no lower-dimensional regular
differentiable chart can preserve every declared continuation boundary near
these exact resident points. It does not exclude singular, discontinuous,
program-restricted, approximate, global, or nonlocal representations. The
identical n-complex classical recurrence remains the strongest executed full
state baseline.

No CATVM custody, compact continuation message, global lower bound,
computational advantage, Small Wall crossing, physical waveform execution,
replacement of physical bits with pi, or unbounded catalytic computation is
established.

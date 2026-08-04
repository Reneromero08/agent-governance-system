# M181 independent reexecution review

Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Verified result

M181 removes M180's retained `q-1` coefficient, Gauss, gamma, and discrete-log
tables for the single-final-scalar path. It enumerates the public
multiplicative orbit directly and regenerates each required Gauss factor in a
ten-field-cell residue workspace. No coefficient, phase, logarithm, gamma, or
Hankel table is resident on the accepted path.

The accepted transaction retains the public boundary accumulator and the
quadratic Gauss factor while four channel factors are regenerated. Slots five
through nine are reused for one Gauss orbit or character search. A rank-three,
nonzero-scale forward projection makes `4*(q-1)+1` Gauss calls; the complete
forward/inverse lifecycle makes `8*(q-1)+2`. Every call visits `q-1` orbit
points, so the repair exchanges linear table residency for theta-q-squared
exact work per projected scalar.

The ten cells are the carrier workspace, not a whole-process memory bound.
Production reports five additional named exact-field temporaries at peak and
one persisted projected output during inverse, giving a conservative
16-field-scalar named lifecycle peak. Five public field-configuration integers
and nine public program/boundary scalars are reported separately. Loop-control
integers, Python objects, bigint expression temporaries, and modular-power
internals remain excluded. Exact capacity is not fixed: even the carrier alone
uses `10*ceil(log2(p))` bits.

## Exact recurrence obstruction

For determinant-character program index `a`, the source coefficients are

```text
c_j = (q-1)^-1 G(a-j).
```

Taking the cyclic Fourier transform in `j` gives

```text
DFT(c)_k = psi(g^k) chi_a(g^k).
```

Every right-hand side is nonzero. Character orthogonality therefore gives
exact cyclic Hankel rank `q-1` and periodic linear-recurrence order `q-1`.
Production and the no-import oracle independently checked the identity,
circulant rank, and Berlekamp-Massey order at q=5, 7, 11, 13, 17, 19, 23, 29,
31, 37, 41, 43, 47, and 53. Deleting one spectral mode reduced the measured
rank by exactly one in every case. The determinant-gamma sequence also had
exact full rank in every declared case, but that second observation is not
promoted to an all-prime theorem.

This rejects a uniformly smaller linear recurrence for the source coefficient
sequence. It does not reject nonlinear, nonuniform, Hasse-Davenport, Jacobi,
or other procedural descriptions. The ten-cell stream is itself such a
procedural time-space tradeoff.

## Independent boundary and lifecycle checks

The oracle imports neither production nor M180. It independently rebuilds
finite fields, logarithms, Gauss tables, coefficient sequences, circulant
ranks, a separate ten-cell streaming machine, and the materialized boundary
formula. All 16 declared transactions match, restore the same ten-cell backing
exactly, retain their projected scalar across inverse, and reuse the restored
backing for an unrelated program with fresh-carrier output and resource
signature parity.

The oracle also directly summed the original seven-dimensional open relation,
not the descriptor projection routine, at four attacked boundaries: q=5 and
q=7 rank-three points plus q=5 rank-one and rank-zero/zero-scale points. The
direct sums covered 49,600, 603,288, 49,600, and 49,600 nonzero source terms
and matched the streamed scalars exactly.

Missing and wrong inverses fail every transaction. Omitting an applicable
Mellin channel changes every boundary. The wrong gamma shift changes every
rank-three boundary. Reordered inverse failure is not applicable because
exact channel accumulation is commutative; an executed ascending inverse also
restores. No snapshot is used. The observed pass count is bookkeeping only;
no generation or lease enforcement is claimed.

## Matched baseline and ceiling

The strongest classical comparison includes both M180's materialized
`4*q-3`-cell table path and the identical ten-cell streamed character-sum
path. The table path amortizes compilation over many boundary projections;
the stream path reduces residency for one scalar by rematerializing quadratic
work. Neither dominates the other in every workload, and no speed or memory
advantage over the best matched classical method is claimed.

The result is limited to the 14 declared prime/auxiliary-field pairs, one
rank-three transaction per field, the q=5 rank-one and zero-boundary branches,
and direct-process exact residue software. It establishes no fixed exact bit
width, fixed linear rank, nonlinear compression no-go, machine-enforced hidden
intermediate, CATVM custody, distinct phase resource, computational advantage,
Small Wall crossing, physical execution, replacement of physical bits with
pi, or unbounded computation.

## Next obstruction

The next phase-owned test should use nonlinear Hasse-Davenport or Jacobi phase
relations to try to reduce the Gauss-family state/work frontier, or establish
their growing exact relation rank. Repeating larger prime fixtures or calling
the streamed classical-equivalent procedure a phase resource would not remove
the obstruction.

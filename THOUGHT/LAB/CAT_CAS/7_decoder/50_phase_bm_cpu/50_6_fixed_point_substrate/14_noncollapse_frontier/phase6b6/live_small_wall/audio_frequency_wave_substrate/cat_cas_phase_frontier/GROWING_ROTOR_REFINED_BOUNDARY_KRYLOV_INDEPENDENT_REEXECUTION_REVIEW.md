# Independent reexecution review: refined Rotor-6 boundary Krylov diagnostic

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration classification: `NO_RESTORATION_CLAIM`

The exact declared F103 boundary sequence has Berlekamp-Massey degree 2,261
when the initial `k=0` boundary is included.  Its connection polynomial has a
zero last coefficient, and an independently recomputed sequence beginning
after one complete public word has exact degree 2,260 with nonzero last
coefficient 73.  The strongest shifted scalar recurrence therefore saves only
17 dynamic field cells relative to the 2,277-cell refined phase carrier and
has 2,244 nonzero public coefficients.  It does not reconstruct the internal
phase state and is not a compact or fixed-rank phase-machine repair.

## Independent construction

The oracle imports no production or predecessor module.  It independently:

- enumerates all 74,613 six-rotor occupation histograms;
- selects the 4,389 global-rotation representatives;
- constructs the reflection quotient and obtains 2,277 bracelets;
- recomputes the nine pair and two declared triangle coordinates;
- independently obtains 2,277 refined signatures and the public boundary;
- enumerates 684,624 legal four-site scattering transitions;
- aggregates them to a verification-only 172,838-nonzero CSR operator;
- builds the F103 root-72 diagonal, source family 0, and public probe;
- executes 4,619 exact scalar samples; and
- applies a separately implemented Berlekamp-Massey recurrence search.

Production and oracle agree exactly on the topology commitment, aggregated CSR
commitment, first forward-state commitment, scalar sequence commitment, both
connection-polynomial commitments, degrees, coefficient counts, and all
training and held-out recurrence checks.  The oracle's raw transition
commitment also equals the independently established M197 commitment.

## Controls and mutations

- Reordering the first operator to diagonal-after-scattering changes the
  boundary from 83 to 29.
- Replacing the declared root by its inverse changes the first boundary from
  83 to 88.
- A deliberately undersampled 512-term certificate reports an apparent
  degree of 256 but fails 4,069 subsequent samples, preventing short-sequence
  rank promotion.
- A null diagnostic state is rejected.
- Thirty-two or more samples beyond the 4,554-term training horizon produce
  zero recurrence violations for both the `k=0` and shifted certificates.

## Resource and baseline ceiling

The retained 652,048-entry shift plan and 172,838-nonzero CSR are diagnostic
state only.  They are not attributed to the accepted M197 path, which retains
zero shift plans and rematerializes 684,624 moves and 24,767,280 triangle
monomial evaluations per scattering.  The diagnostic CSR occupies 347,954
reported integer cells across values, column indices, and row pointers, in
addition to diagonal, probe, state, sequence, and recurrence workspace.
Python containers, allocator overhead, SciPy native bytes, timing, and
whole-process peaks are excluded.

The strongest matched classical continuation is one complete M197-equivalent
full-state initialization followed by the identical exact 2,260-scalar
companion recurrence with 2,244 nonzero public coefficients.  Setup cost is
not omitted.  This diagnostic establishes neither restoration nor reuse; it
preserves the separate exact-algebraic M197 restoration/reuse evidence without
extending it.

## Claim ceiling

The result is limited to the grid-17, exchange-symmetric, global-rotation- and
reflection-invariant, two-triangle refined Rotor-6 family over F103 with root
72, repeated `step=0, tag=0`, source family 0, and the declared scalar public
boundary.  It does not establish a recurrence for arbitrary programs, an
internal-state quotient, CATVM custody, a distinct phase resource,
computational advantage, a Small Wall crossing, physical waveform execution,
replacement of physical bits with pi, or unbounded computation.

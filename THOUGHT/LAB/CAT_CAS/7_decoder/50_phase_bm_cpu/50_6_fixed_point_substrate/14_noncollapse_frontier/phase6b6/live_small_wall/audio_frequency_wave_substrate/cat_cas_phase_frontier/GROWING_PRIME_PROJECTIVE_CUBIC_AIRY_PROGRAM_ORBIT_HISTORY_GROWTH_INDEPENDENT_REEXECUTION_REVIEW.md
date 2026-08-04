# Growing-Prime Projective Cubic Airy Program-Orbit Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION` for the discrete finite-field
carrier under the in-place branch traversal and restored-carrier reuse. The
enumerated projective-code set is diagnostic state and has
`NO_RESTORATION_CLAIM`; it is not part of an accepted compact carrier law.

## Reconstructed mechanism

M177 ruled out regular exact linear message quotients below the observed
two-fiber public-orbit span, but left nonlinear canonicalization open. This
experiment tests the smallest direct nonlinear candidate: canonicalize the
complete resident `2*q^2`-cell joint state modulo one nonzero field scalar and
merge public coefficient words that have the same exact canonical code.

The production traversal applies each public coefficient pair to the actual
resident carrier, recursively visits the declared word tree, applies the exact
inverse after every branch, and checks the restored payload before continuing.
The equality key is a collision-free base-`p` integer encoding of every
canonical field cell, not a cryptographic digest. No dense carrier-history tree
is retained.

## Independent attacks and results

- The no-import oracle separately rebuilds safe-prime fields, roots, fixtures,
  Gaussian kernels and inverses, both carrier axes, controlled cubic phases,
  projective codes, recursive forward/inverse traversal, collision linear
  algebra, mutation controls, boundary projection, and restored reuse.
- At `q=5`, depths 1, 2, and 3 are collision-free: 25, 625, and 15,625 public
  words produce the same number of distinct projective full states.
- At `q=5`, depth 4 has 390,625 words but 388,125 distinct projective states.
  There are 387,500 singleton classes and exactly 625 non-singleton classes of
  size 5, for 2,500 repeated histories.
- All 3,125 histories in non-singleton classes form one rank-5 linear subspace
  of the eight public coefficients. In coefficient order
  `(a0,b0,a1,b1,a2,b2,a3,b3)`, its three independent constraints set
  `a2=b2=a3=0`. Every class is a coset of the single direction
  `(0,0,0,1,0,0,0,-1)` over `F5`.
- Every collision is exact raw-state equality with relative scalar one. The
  observed merge therefore is not created by discarding global phase.
- The quotient does not lower the depth-4 identifier width: both 390,625 words
  and 388,125 states require 19 bits for a dense fixed-horizon index.
- At `q=11`, depths 1 and 2 are collision-free with 121 and 14,641 states. The
  `q=5` collision direction fails all three exact transfer samples at `q=11`
  and all three at `q=23`; it is not a transferable growing-prime law.
- Missing, wrong, and reordered inverses fail. A global scalar changes the raw
  state while preserving its projective code, so lawful inversion still needs
  the scalar ledger. The restored `q=5` carrier then executes an unrelated
  alternate-order program on the same list backing, matches fresh execution,
  reaches restoration generation two, and uses no snapshot.

## Resource law and matched baseline

The resident dense carrier has `2*q^2` field elements and the public program
descriptor has `2*depth` field elements. Recursive rematerialization retains no
dense state tree, but the diagnostic equality set grows with the number of
distinct full states; at `q=5`, depth 4 it contains 388,125 exact large-integer
codes. Its integer payload, Python object headers, allocator state, and
interpreter process are not hidden carrier resources and are not presented as
an accepted compact algorithm.

Classical software has the identical in-place dense recurrence and the
identical compact procedural factor graph with the public coefficient word.
The sparse `q=5` equality classes neither reduce the tested fixed-horizon bit
width nor beat that factor graph. This is an obstruction result for the direct
projective-canonicalization repair, not a computational resource result.

## Claim ceiling

The ceiling is the primary two-shear family, the fixed declared Gaussian layer
sequence, primary two-fiber fixture, exact finite-field direct-process
software, `q=5` depths 1 through 4, and `q=11` depths 1 and 2. The transfer
attack covers only three declared collision-direction samples at each of
`q=11` and `q=23`.

No arbitrary-depth quotient, transferable nonlinear closure, universal state
lower bound, CATVM custody, distinct phase resource, computational advantage,
Small Wall crossing, physical waveform or silicon execution, replacement of
physical bits with pi, or unbounded catalytic computation is established.

## Durable identities before the science commit

- Production source SHA-256:
  `d95869a3103073e72715c1cba6d97637617df03befb5a44b2614572c90dc6a71`
- Sealed production result SHA-256:
  `a5153508ee1c674ab78d4c3c1ac2706bb99b677b617a2cab1cd0bd2507bba479`
- Independent oracle source SHA-256:
  `069e52b336adb06b9dedcc6c799eb35c9c2dbb685894dc7b67ae98e6a348214e`
- Sealed independent result SHA-256:
  `2ccda5f0289a3a77b520174a00430710033bc4fd345492c97708dfda96d2e1f6`
- Qualifier SHA-256:
  `c45abab66512e51a98c3eeeaefad9ac11d43e5283d2e75a98e71e898769f989a`

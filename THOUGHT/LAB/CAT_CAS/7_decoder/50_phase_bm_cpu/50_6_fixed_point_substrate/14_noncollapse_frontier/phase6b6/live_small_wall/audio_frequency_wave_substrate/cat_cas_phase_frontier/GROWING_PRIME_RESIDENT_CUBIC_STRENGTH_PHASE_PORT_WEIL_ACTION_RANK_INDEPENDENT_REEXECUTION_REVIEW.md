# Growing-Prime Resident Cubic-Strength Port Independent Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION` for the discrete resident
finite-field carrier. Public plans, boundary probes, verification copies,
classical comparison carriers, and NTT scratch carry
`NO_RESTORATION_CLAIM`.

## Reconstructed mechanism

The direct-process software carrier stores a q-coordinate cubic-strength port
together with a two-fiber q-coordinate data port in `2*q^2` finite-field
cells. The same unprojected strength coordinate controls multiple cubic phase
operations separated by local Gaussian operations on both axes. Those
operations do not commute with the controlled cubic operation. The public
word is compiled without reading the strength amplitudes, and only the final
declared scalar boundary is projected.

Here `coherent` means that the exact finite-field amplitudes remain resident
and are transformed linearly as one algebraic state. It does not assert a
physical or quantum implementation. The runtime operand is not protected by
CATVM in this experiment, and the reproducible fixture amplitudes are not
secret package data.

The independent oracle imports neither production nor any predecessor. It
separately reconstructs the safe-prime fields, cubic phases, projective
Gaussian kernels and inverses, public schedules, q-by-2q carrier recurrence,
latent/data matrix ranks, final boundary, exact reverse sequence, controls,
reuse, dense comparison recurrence, and radix-2-NTT-backed Rader recurrence.
Production JSON is used only as a comparison target.

## Executed evidence

- All eleven declared cases were independently reconstructed: primary depth
  one for q values 5, 11, 23, 29, 41, 53, 83, 89, and 113, plus primary q=113
  depth two and alternate q=41 depth two.
- The initial carrier has latent/data separation rank one. Because cubing is
  bijective in every declared multiplicative group, the first controlled
  cubic operation reaches rank q. Every later controlled-cubic checkpoint and
  every declared final state also has rank q.
- Every boundary, controlled-rank sequence, final rank, full-state
  commitment, dense work tuple, Rader work tuple, resource tuple, exact
  restoration assertion, and backing-identity assertion agrees with
  production.
- The exact Rader/NTT comparison matches the dense recurrence in all eleven
  cases. Direct prime-DFT comparisons pass for two roots at q=5 and q=11. The
  maximum bounded convolution coefficient is 5,720,512, below auxiliary
  modulus 998,244,353.
- Missing, wrong, and reordered inverses fail. Module reordering changes the
  boundary. Premature projection, null strength state, invalid family, and
  zero depth are rejected. A one-hot strength mutation does not reproduce the
  full-rank result, while the compiled plan remains independent of amplitude
  values.
- Primary depth one followed by unrelated alternate depth two on the actual
  restored q=23 backing reaches restoration generation two. The second
  boundary and complete commitment match a fresh carrier without snapshot
  reload.

## Resource law and ceiling

The accepted recurrence uses `2*q^2` resident field cells and a
`2*q^2+q` resident-plus-temporary peak. The dense matrix-free classical
recurrence executes the identical state update with the identical leading
state and peak laws. The work-reduced exact Rader/NTT recurrence retains the
same `2*q^2` state and adds `4*q-2+2*M` scratch payload cells, where `M` is the
next power of two at least `2*q-3`. Mixed field-cell and 30-bit auxiliary-cell
capacities are counted separately.

At q=113 depth two the accepted and dense peaks are 25,651 field cells. The
Rader member retains 25,538 field cells and reaches 26,500 logical payload
cells with scratch; its declared mixed-width bit-capacity upper bound is
223,264 bits. Python objects, allocator behavior, interpreter state, native
libraries, and whole-process peaks are excluded and are not presented as
material-cell measurements.

Full rank is established only for the explicit q-by-2q linear
latent/data matrix representation. It is not a lower bound against nonlinear,
program-specific, quotient, or observable-restricted algorithms. The result
therefore rejects fixed-rank factorization for this explicit shared-port
recurrence but does not establish an intrinsic quadratic lower bound.

The exact ceiling is the nine declared safe-prime pairs, the primary and
alternate public families, declared depths, two-fiber direct-process finite-
field software, and auxiliary NTT modulus 998,244,353. No fixed-width
extension beyond q=113 is established. The package does not establish CATVM
custody, machine-enforced hidden amplitudes, general relation closure, a
distinct phase resource, computational advantage, a Small Wall crossing,
physical waveform execution, replacement of physical bits with pi, or
unbounded catalytic computation.

## Durable file identities

- Production source SHA-256:
  `9faa685880cd7e1c3c9f8613583546255f327b181864461ae51498dd4e696a6b`
- Sealed production result SHA-256:
  `248c4165cee18634d5cce2d17ccc352ad680a714772d873e48a1f62097ec877d`
- Independent oracle source SHA-256:
  `90d173deb2be68a6bb6ea1485bf09353bb9ec4fef1c54990b732b28ebe79e65e`
- Sealed independent result SHA-256:
  `7232bfcaaaaf43ea84bebd7288f78dd636bbe77dbb5635eb528f63fc7280c507`

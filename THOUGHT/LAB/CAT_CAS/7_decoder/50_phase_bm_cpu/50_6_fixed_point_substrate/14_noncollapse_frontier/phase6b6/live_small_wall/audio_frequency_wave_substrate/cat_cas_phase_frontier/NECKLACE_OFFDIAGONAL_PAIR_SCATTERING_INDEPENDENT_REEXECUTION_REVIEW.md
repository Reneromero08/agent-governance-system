# M190 Independent Reexecution Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `NUMERICAL_PHYSICAL_STATE_RESTORATION`

## Reconstructed mechanism

The reference imports no production module. It independently generates all
4,845 exchange-symmetric four-boson occupation histograms and the 285
global-rotation necklace representatives. It constructs eight sparse
pair-scattering shift-distance bases from the declared public topology, checks
the full weighted Hermiticity identity, and evolves the occupation state with
SciPy sparse `expm_multiply`. Production instead streams the same quartic law
directly on the 285 necklace cells and uses a degree-64 Chebyshev recurrence.

Primary and unrelated-reuse boundaries agree with production within
`4.8e-15`. Missing, wrong, and reordered inverses, zero scattering, and
swapped phase/scattering order independently reproduce the production control
values within `3.5e-15`. The occupation oracle measures weighted-Hermitian
error `1.11e-16` and rotation-invariance error `3.71e-18`.

## Accepted scope

The accepted production path keeps 285 complex necklace cells plus 855
temporary necklace-complex Chebyshev cells. It retains no 4,845-cell occupation
vector, sparse or dense pair-transition operator, pair table, or inverse
history. It applies 16 signed momentum-conserving shifts with opposite shifts
sharing one real public weight. Terms that change two occupation modes are
present, so this is genuinely off-diagonal and is not another diagonal
occupation-signature phase.

Restoration is numerical for virtual complex coordinates. Primary and reuse
errors are `1.76e-15` and `3.46e-15`; 32-cycle drift is `1.97e-14`. The same
backing is used and no baseline reload occurs. Response ordering is
direct-process and is not CATVM-enforced.

## Resource and claim ceiling

Production counts 285 resident complex cells, 855 temporary complex cells,
15,980,544 streamed ordered-pair shift terms, 384 generator applications, and
28,830 maximum named engine bytes for the accepted primary-plus-inverse
transaction. Allocator state, expression temporaries, standard-library Bessel
work, compiler memory, and whole-process peaks are excluded. The independent
oracle separately materializes 4,845 complex occupation cells and eight sparse
bases totaling 350,608 nonzeros; none belongs to the accepted path.

The strongest compact classical implementation is the identical 285-complex
necklace pair-scattering plus diagonal phase recurrence. The result establishes
a broader off-diagonal phase-machine update, but not a distinct phase resource,
computational advantage, Small Wall crossing, CATVM custody, physical waveform
execution, replacement of physical bits with pi, or unbounded computation.

Exact ceiling:

`GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_DEPTH3_PRIMARY_DEPTH2_REUSE_SIXTEEN_SIGNED_MOMENTUM_CONSERVING_PAIR_SHIFTS_CHEBYSHEV_DEGREE64_COMPLEX128_DIRECT_PROCESS_SOFTWARE_ONLY`

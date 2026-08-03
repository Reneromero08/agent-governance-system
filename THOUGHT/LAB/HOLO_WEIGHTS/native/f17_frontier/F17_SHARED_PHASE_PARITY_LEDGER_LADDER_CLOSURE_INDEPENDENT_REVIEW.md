# M124 Independent Review: Shared Phase-Parity Ledger Ladder Closure

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident factor/scalar carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Transient four-state recurrence frontier: `NO_RESTORATION_CLAIM`

The verified claim is confined to the declared two-row open-ladder family.
Exact `Q(zeta_17)` boundaries agree for both public descriptor families at
widths `1,2,4,8,16,32,64`; independent `F103/F137` structural execution
agrees through width 128.  This is not a general external-field or
growing-treewidth closure.

## Independent reconstruction

The oracle imports neither production nor predecessor code.  It implements a
separate Fraction-valued power basis for `Q(zeta_17)`, reconstructs the public
ladder factors, and contracts the four column boundary states directly.  At
widths `1..4`, independent binary-assignment and occurrence-expanded checks
also agree with the accepted recurrence.

The grouped factor ordering has maximum separator ranks
`[2]`, `[2,4,2]`, `[2,4,8,4,2]`, and `[2,4,8,16,8,4,2]` at widths `1..4`.
Public column interleaving changes the last two profiles to
`[2,4,4,4,2]` and `[2,4,4,4,4,4,2]`; exhaustive public-order checks through
width four find optimum maximum rank four.  This closure is explained by the
two-row four-state separator, not a general rank-collapse law.

The exact PRIMARY width-two native defect signature violates one
Grassmann--Pluecker identity, with nonzero delta coefficients
`[64,0,0,0,0,-32,0,0,0,0,0,0,-32,0,0,0]`.  Broader exact
non-Gaussianity is not claimed.  Modular structural checks count violations
`0,1,9,36` at widths `1..4` in both fields.

## Restoration, controls, and resources

Only the final 16-coordinate boundary is projected.  Public factor loads and
the scalar update are reversed on the actual backing, with no retained inverse
history or snapshot reload.  Generation advances exactly and the restored
width-16 carrier executes an unrelated descriptor with fresh/restored boundary
agreement.

Controls cover missing inverse, wrong ownership, premature projection, null
carrier, wrong field-port typing, factor mutation, and absence of a snapshot
command.  Reordered inverse failure is inapplicable because the accepted
factor loads and accumulator additions commute.

The accepted path retains `10W-4` exact factor cells, four frontier cells, and
at most 19 simultaneously named transient field cells for width greater than
one.  It performs `O(W)` field operations; exact bit complexity and payload
height are reported separately.  It materializes neither the `4^W` dense
signature nor the occurrence-expanded even sectors.  Python containers,
allocator/native-library storage, bigint internals, and whole-process peak are
excluded explicitly.

## Claim ceiling

M124 closes the declared two-row family at rank four, with exact restoration
and restored-carrier reuse.  The strongest compact classical method is the
identical four-state transfer recurrence.  The result does not establish
general separator compaction, CATVM custody, a distinct phase resource,
computational advantage, Small Wall crossing, physical waveform execution,
replacement of physical bits with pi, or unbounded catalytic computation.

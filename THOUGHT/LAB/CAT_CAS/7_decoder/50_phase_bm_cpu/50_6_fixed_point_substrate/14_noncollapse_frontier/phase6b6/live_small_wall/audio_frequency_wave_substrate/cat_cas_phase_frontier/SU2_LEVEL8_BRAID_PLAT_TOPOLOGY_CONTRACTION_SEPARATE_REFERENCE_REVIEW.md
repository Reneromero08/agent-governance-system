# Separate-reference review: SU(2)_8 braid-plat topology contraction

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `SEPARATE_REFERENCE_PARITY`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Exact scope

For the two declared public eight-sweep braid families and even strand counts
`4,6,8,10,12,14,16`, M216 contracts the exact vacuum-to-vacuum plat boundary
over `Q(zeta_40)` without materializing the complete final fusion-path vector.
The production compiler derives its min-fill variable order and structural
supports from public spacetime topology and admissibility only. It does not
inspect exact coefficients or the answer.

The result is bounded. For the tested fixed eight-sweep sequence, production
induced width reaches 11 at 10 strands and remains 11 through 16 strands. This
is not an asymptotic fixed-width theorem. At fixed 16 strands, increasing sweep
depth from 1 to 16 raises induced width from 0 to 23 and the largest structural
support factor from 1 to 99,390 assignments. This directly exposes sweep depth
as the next separator-growth obstruction.

## Separate reconstruction

The reference imports the independently checked M214 exact field and public
braid-program definitions, but imports no M216 production code. It independently
derives every local spacetime factor and contracts in a fixed column-major
order rather than production's min-fill schedule. For all 14 declared cases,
the reference, production contraction, and M214 full fusion-path recurrence
give identical exact boundary commitments. The reference schedule is
deliberately not presented as a resource improvement; at 16 strands it peaks
at 1,599 and 1,790 exact factor cells for families 0 and 1, respectively.

## Resource and restoration result

For the primary 16-strand family-0 case, production peaks at 771 live exact
factor cells (39,127 exact payload bits), versus 1,430 exact field cells
(256,269 payload bits) for the full-vector verification recurrence. The public
plan is not free: it retains 120 leaf descriptors, 1,292 descriptor integer
cells, 1,114 support-assignment records containing 4,314 label integers, and
224 operation records containing 1,106 integer cells. Compilation itself peaks
at 1,114 live support assignments containing 5,416 label integers. Python
container, hash-table, allocator, interpreter, serialization, timing, and
whole-process peaks remain explicitly excluded. Consequently the field-table
reduction is not an overall memory-advantage claim.

The accepted direct-process transaction has one exact accumulator cell and no
retained inverse history. Its inverse rematerializes the public contraction,
subtracts the exact boundary value from the same backing, verifies the exact
canonical zero state, and reuses that backing for a different 12-strand,
five-sweep family-1 program. Fresh and restored reuse boundaries agree,
restoration generation reaches two, and no snapshot reload is used.

## Claim boundary

The strongest compact classical comparator is the identical exact sparse
factor elimination; the full fusion-path recurrence is an additional exact
reference. M216 therefore establishes a bounded exact final-boundary
contraction and an honest depth-width obstruction, not a distinct phase
resource, computational advantage, Small Wall crossing, CATVM custody, full
state compaction, physical waveform execution, replacement of physical bits
with pi, catalytic inference, or unbounded computation.

# M135 Independent Review: Two-Latent Cubic Rank-4 Cycle

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident cubic-factor and four-cell shared-port carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Compiler, commitment, projection, and verification buffers:
`NO_RESTORATION_CLAIM`

The reviewed scientific source is commit
`74684d968304da3886a4e4309a35993bf3cc291f`.

## Verified mechanism

The declared family has two shared Boolean latent coordinates `h,k`.  Each
branch adds three local bits and the two pure cubic factors
`alpha^(h*u*v)` and `beta^(k*u*w)`.  Direct enumeration of one branch gives a
local-to-`(h,k)` map with a four-by-four minor
`(alpha-1)^2 (beta-1)^2`.  Every accepted nonidentity factor makes this minor
nonzero.  The independently built two-branch, eight-by-eight boundary tensor
has exact rank four over `Q(zeta17)`, `F137`, and `F239`, for both declared
families.  A rank-three separator is therefore insufficient for this fixture.

The resident unresolved port has four components.  Every branch response acts
diagonally on this port; unnormalised binary Walsh transports alternate on the
`h` and `k` axes strictly between consumer groups.  The final public boundary
is projected only after the last consumer.  Disabling transport, moving it
before its consumer, merging `h=k`, copying the port instead of sharing it, or
perturbing a public phase changes the boundary.

## Independent reconstruction

The oracle does not import the production module.  It separately compiles the
public descriptor, enumerates the eight local assignments for each branch
response, reimplements the four-state recurrence and its inverse, directly
builds the complete two-branch tensor, performs exact Gaussian elimination,
and independently reconstructs commitments, byte counts, resource tuples,
payload maxima, and compiled final rows.

Exact executions cover branch counts `2,4,8,16,32,64` in `Q(zeta17)`.
Structural executions cover branch counts `2..8` in both `F137` and `F239`.
All twenty production transactions agree with the independent oracle.

## Restoration, reuse, and attacks

The inverse consumes the actual resident branch factors.  It applies each
diagonal inverse and the preceding inverse Walsh in lawful reverse order,
then unloads cubic factors in reverse order.  The factor cells and the shared
port return to exact zero on the same outer carrier, nested carrier, factor
lists, and port backing.  Projection count is part of the machine-relevant
zero predicate.  No snapshot, baseline reload, or inverse-history buffer is
used.

The focused controls execute complete lawful, reordered, and wrong-Walsh
reverse sequences.  Only the lawful sequence restores the seed.  Additional
controls cover missing inverse, wrong factor inverse, wrong lease owner,
premature projection, null carrier, rank-three truncation, overmerge,
undershare, semantic perturbation, and unavailable snapshot.  A `PRIMARY`
branch-eight transaction followed by unrelated `REUSE` branch-sixteen work
uses the same backing, matches fresh execution and resource signatures, and
advances the package-local restoration counter to two.  That counter is not a
CATVM custody generation.

The accepted result exposes the final scalar and a one-way factor commitment.
It does not return the resident port or factor values, assignment lists, a
dense tensor, or inverse history.

## Resources, baseline, and ceiling

At `b` branches the phase carrier holds `4b` exact factor cells, of which
`2b` are nontrivial public phases, plus the four-cell shared port.  The
strongest matched classical full signature is the identical `2b` public
phase factors.  The strongest runtime boundary algorithm is the identical
four-scalar recurrence; an arbitrary port input compiles to four final-row
coefficients, and a fixed sealed transaction can cache one scalar.

The classical and phase recurrences have the same measured exact port-payload
maxima.  Across exact branch counts `2..64`, that maximum grows from 205 to
13,915 bits.  The phase factor payload grows from 264 to 8,448 bits.  Fixed
logical rank therefore does not imply bounded exact storage width.  These
counts exclude Python containers and allocator state, native-library storage,
bigint/hash internals, bit-operation complexity, and whole-process peak.

The strict ceiling is the declared two-shared-latent cubic cycle with the
declared interleaved Walsh transports and scalar final boundary.  It does not
establish arbitrary cubic-hypergraph closure, arbitrary port arity, CATVM
custody, a distinct phase resource, computational advantage, Small Wall
crossing, physical waveform execution, physical bit replacement, or unbounded
catalytic computation.

This result changes the separator law from M134's chain-only bond three to an
exact rank-four shared junction.  It also establishes the obstruction: the
phase path still reduces exactly to the matched four-scalar recurrence, stores
twice the public factor signature, and inherits growing exact port width.

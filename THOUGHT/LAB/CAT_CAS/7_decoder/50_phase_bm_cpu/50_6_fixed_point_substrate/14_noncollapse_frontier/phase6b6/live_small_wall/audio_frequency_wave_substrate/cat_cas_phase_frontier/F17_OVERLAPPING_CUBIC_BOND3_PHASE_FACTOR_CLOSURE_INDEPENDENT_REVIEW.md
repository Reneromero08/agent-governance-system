# M134 Independent Review: Overlapping Cubic Bond-3 Phase Factors

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident phase-factor carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Compiler, contraction, projection, commitment, and verification buffers:
`NO_RESTORATION_CLAIM`

The reviewed scientific source is commit
`15b9f81e999455d7156dc2d3a7d55b20a3d47fae`.

## Verified mechanism

The declared chain has public bits `x_0,...,x_(m+1)` and exact factors
`theta_j^(x_j x_(j+1) x_(j+2))`.  The multiplicative third interaction
invariant is `theta_j`, which is nonidentity for every accepted factor.  The
family is therefore genuinely cubic on the same visible bits and cannot be
rewritten as the M133 unary/pairwise Gray factorization.

For an interior cut, the two crossing factors form a four-by-four matrix with
rows indexed by `(a,b)` and columns by `(c,d)`:
`theta_left^(abc) theta_right^(bcd)`.  Rows `00` and `10` coincide, while a
three-by-three minor is, up to sign,
`(theta_left-1)(theta_right-1)`.  Hence its exact rank is three.  Factors wholly
on one side only scale or replicate rows or columns, so the local certificate
transfers to the full tensor.  Independent full-tensor elimination gives cut
ranks `[2,2]` at depth one and `[2,3,...,3,2]` thereafter.  The exact minimal
maximum MPS bond dimension is therefore two at depth one and three at every
declared depth at least two.

## Independent reconstruction

The oracle does not import production code.  It separately compiles the
public factor schedule, uses a four-state last-two-bit recurrence, directly
enumerates exact assignments through depth eight over `Q(zeta17)`, and checks
local and full-tensor ranks over `Q(zeta17)`, `F103`, and `F137`.  The separate
four-state recurrence agrees with the production three-state quotient and
direct enumeration.  It also reconstructs the selected final boundary, all
resource tuples, public descriptor commitments, and the matched classical
signatures.

The production quotient closes on three state scalars:
`A'=A+B`, `B'=A+C`, `C'=A+theta*C`, with final `Z=2A+B+C`.  This recurrence is
exact for the declared partition boundary.  The compiled three coefficients
apply only to an arbitrary input in this three-state chart; they are not an
arbitrary full two-bit boundary claim.

## Restoration, ownership, and attacks

Each resident site is seeded as `(1,0)`, coupled to `(1,theta_j)`, consumed
directly, inverted in reverse order, verified as `(1,0)`, and unseeded to exact
zero on the same backing.  No snapshot, baseline reload, or retained inverse
history is used.  `PRIMARY` depth eight followed by unrelated `REUSE` depth
sixteen matches fresh execution in boundary and resource signature and
advances the package-local restoration count to two.  This direct-process
counter is not a CATVM-enforced custody generation.

Active use validates carrier type, stage, declared depth and family, and a
recomputed lease.  Executed controls cover nonidentity factors, identity rank
collapse, bond-two rejection, schedule perturbation, missing/wrong/reordered
inverse, premature projection, null carrier, wrong projection owner, wrong
inverse owner, and unavailable snapshot.  The oracle additionally checks a
same-depth identity substitution, semantic phase perturbation, reflection
symmetry of reversed public order, and distinct primary/reuse lease identities.

The accepted response contains the final scalar and a one-way factor
commitment.  It does not expose the resident factor payload, an assignment
tensor, expanded component weights, a dense transfer, or a truth table.

## Resources, baseline, and ceiling

The accepted carrier has `2m` resident exact field cells, including `m`
nontrivial public cubic phase factors.  The selected partition contraction has
three dynamic scalars and at most six named old-plus-new cells.  The strongest
matched classical full signature is the identical `m` public factors; the
selected boundary uses the identical three-state recurrence; a sealed public
word has three chart-input coefficients; and the fixed sealed transaction can
cache one scalar.

At exact depths `1,2,4,8,16,32,64,128`, resident carrier payload grows from 66
to 8448 bits and final-boundary payload grows from 36 to 1828 bits.  These are
exact payload counts for declared algebra values, not whole-process memory or
full exact bit complexity.  Python containers and allocator, native-library
storage, bigint/hash internals, bit-operation complexity, and whole-process
peak remain excluded.

The strict ceiling is this public one-dimensional overlapping pure-cubic
Boolean factor chain and its selected partition boundary.  It does not cover
arbitrary cubic hypergraphs, arbitrary boundaries, CATVM custody, a distinct
phase resource, computational advantage, Small Wall crossing, physical
waveform execution, physical replacement of bits with pi, or unbounded
catalytic computation.  Its exact bond-three closure remains identical to a
compact classical recurrence, and its exact boundary width grows with depth.

The next mechanism should add the smallest shared or branching cubic phase
geometry whose separator law is not already this public three-state chain,
while retaining exact restoration and the strongest matched classical
hypertree or tensor contraction.  Increasing only this chain's depth would not
resolve the obstruction.

# M132 Independent Review: Iterated Affine-Reflection Secant-Rank Growth

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident explicit secant carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Coupling, projection, commitment, certificate, compiler, and classical-baseline
buffers: `NO_RESTORATION_CLAIM`

The reviewed scientific source is commit
`105d1a1944b975382ad1d084ac7bbfd30d317824`. The source and sealed evidence
were reexecuted after repair of the initial resource and baseline labels.

## Verified mechanism

At level `j`, the public one-particle involution sends the binary-chart slope
`t` to `a_j-t`, with `a_j=2^j-1`. The distinct affine reflections do not
commute. Each coupling is `I+eta_j R_j` and has exact inverse
`(I-eta_j R_j)/(1-eta_j^2)` in every declared algebra.

The public support obeys
`S_j=S_(j-1) union (a_j-S_(j-1))={0,...,2^j-1}`. All weights are nonzero.
For `r=2^m` and `k=2r-2`, the normalized `r x r` Hankel catalecticant factors
as `V diag(w) V^T`; its determinant is the nonzero product of all weights and
the squared Vandermonde differences. The lower bound `r` and explicit
`r`-component upper bound establish exact normalized divided-power secant rank
`2^m` for the declared family.

Ordinary symmetric Waring-rank language is restricted to `Q(zeta17)`.
`F103` and `F137` are structural divided-power checks, with the explicit
applicability condition `2^m <= p`. Execution is bounded to `m=1..6`.

## Execution, restoration, and enumeration

The accepted execution explicitly enumerates and retains all `r=2^m`
coherent components. Each component contains one weight and 17 vector
coordinates, so resident state is `18r` field cells and the largest logical
coupling transient is `36r` field cells. This is not a fixed-rank chart.

No additional truth table, assignment buffer, occupation vector,
catalecticant, or dense operator is materialized on the accepted transaction
path. Only the final degree-one mode-one occupation scalar is projected.
Reverse couplings run in reverse order and restore exact zero on the same
backing without snapshot reload or retained inverse history. Unrelated
`PRIMARY m=3` then `REUSE m=4` execution advances generation to two and
matches fresh execution in boundary and resource signature.

## Independent reconstruction and attacks

The oracle does not import the production module. It separately compiles the
public family, reconstructs component coupling and exact inverse, proves the
support induction, computes the Vandermonde product, and directly evaluates
Hankel determinants for exact `Q(zeta17)` through `m=4` and both finite fields
through `m=6`. It independently checks all 18 transaction commitments,
boundaries, resource identities, restoration paths, and rank certificates.

Last-coupling omission changes the boundary. Wrong and reordered inverses do
not return the prior rank. Distinct-reflection noncommutation, finite-field
point-collision exclusion, final-only projection, wrong ownership, premature
projection, null carrier, and unavailable snapshot controls are hard-gated.
Intermediate component payloads and determinant values are not serialized.

## Matched baselines and ceiling

The strongest matched full-state classical representation is independently
executed as `r` atomic weights on public support `0..r-1`; support indices are
rematerialized and not retained as field cells. The simple implementation
uses `3r/2` named old-plus-new update cells at its largest step. A separate
dense full-moment recurrence uses `2r-1` field cells and is not called the
strongest compact baseline. For final-boundary-only execution, two dynamic
moments suffice; a sealed word can cache one final scalar.

Consequently the explicit phase chart is strictly larger than the strongest
matched classical state, and no computational advantage or distinct phase
resource is established. Exact payload-height tuples and full bit complexity
remain unverified; Python container/allocator, native-library, bigint,
hashlib, and whole-process storage remain excluded.

The strict ceiling is the declared `a_j=2^j-1` affine-reflection family,
bounded `m=1..6`, exact `Q(zeta17)`, structurally applicable `F103/F137`, and
direct-process software execution. It does not establish an arbitrary
interleaved-coupling rank law, fixed-rank closure, a general Gaussian no-go,
CATVM custody, computational advantage, Small Wall crossing, physical
waveform execution, physical replacement of bits with pi, or unbounded
catalytic computation.

The next experiment should test a genuinely fixed-rank Gaussian or stabilizer
phase chart for noncommuting superposition couplings, or derive a transferable
no-go that forces a different native phase resource. It should not add a
larger explicit secant fixture.

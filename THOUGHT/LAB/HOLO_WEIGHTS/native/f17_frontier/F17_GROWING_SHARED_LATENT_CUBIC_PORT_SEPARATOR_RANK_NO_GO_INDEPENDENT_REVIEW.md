# M136 Independent Review: Growing Typed Cubic Separator Rank

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident cubic-factor carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Compiler, commitment, projection, and verification buffers:
`NO_RESTORATION_CLAIM`

The reviewed scientific source is commit
`e3af2700c46fec0e6dea330d9dc133b8144754eb`.

## Verified theorem

For `k` individually typed shared Boolean latent coordinates, the declared
left and right anchor-one branch minors are Kronecker products of the exact
two-by-two matrices `[[1,1],[1,theta_i]]`.  Every accepted phase is
nonidentity, so every local determinant `theta_i-1` is nonzero.  Each branch
minor therefore has exact rank `2^k`.  The unnormalised Walsh separator is
also invertible, so the two-sided minor has exact rank `2^k`.

The determinant contribution of every left and right local factor has
exponent `2^(k-1)`.  This law was checked directly rather than inferred from
larger fixtures.  Exact dense matrices through `k=6` independently reproduce
the left, right, Walsh, and two-sided ranks over `Q(zeta17)`, `F103`, and
`F137`.  Identity left or right phases and a singular separator mutation
reduce the rank.

The resulting no-go is deliberately limited.  A fixed or uniform exact
field-linear quotient that preserves arbitrary incoming port messages and
the complete declared family of typed continuations must be injective and
retain at least `2^k` field coordinates.  The theorem does not exclude
nonlinear, program-dependent, symmetry-restricted, or global contraction
routes.

## Independent reconstruction

The oracle does not import the production module or its factor carrier.  It
separately compiles the public descriptor, reconstructs phases and
commitments, reimplements the local pair coupling and inverse, builds the
left and right branch minors and Walsh matrices, performs exact Gaussian
rank elimination, and compares the dense boundary with the factorised
`O(k)` contraction.

All seventeen production transactions agree with the independent
reconstruction.  The direct rank suite contains eighteen cases: six arities
in each of the three exact fields.  The oracle also checks phase
perturbation, wrong inverse, rank-cap, primary/reuse descriptor separation,
and the absence of a snapshot command.

The strict qualifier was reexecuted against the sealed artifacts and emitted:

```text
QUALIFIED_F17_GROWING_SHARED_LATENT_CUBIC_PORT_SEPARATOR_RANK_NO_GO_STRICT_SCOPE
```

## Accepted path, restoration, and reuse

The accepted transaction stores only the `2k` public nonidentity phase
factors in the established paired factor carrier, using `4k` exact field
cells.  It derives the symbolic certificate from the actual resident factors
without materialising a `2^k` port vector, local assignment family, dense
minor, or determinant value.  Only the rank certificate and one-way factor
commitment survive the transaction.

The inverse consumes the actual resident factors in reverse order and
returns the same backing to canonical exact zero.  Missing, wrong, and
reordered inverses fail their controls.  Type, stage, arity, family, lease,
projection owner, and null-carrier checks pass.  A primary arity-eight
transaction followed by an unrelated reuse-family arity-sixteen transaction
uses the same original backing, agrees with fresh execution and resource
signatures, advances the package-local restoration counter to two, and uses
neither snapshot reload nor retained inverse history.  This counter is not a
CATVM custody generation.

## Resources, baseline, and ceiling

The production certificate work is linear in `k`, and the accepted factor
carrier uses `4k` exact field cells.  The uniform exact linear
arbitrary-message port law nevertheless has `2^k` coordinates.  This rank is
not a general storage or computational lower bound: the same declared local
topology contracts classically as `O(k)` two-by-two Kronecker factors, and a
general public factor descriptor remains governed by the strongest matched
`poly(descriptor size) * 2^treewidth` variable-elimination baseline.

Consequently, the result establishes neither a phase resource nor an
advantage.  Its strict ceiling is the declared typed cubic branch family,
nonidentity phases, invertible typed Walsh separator, arbitrary incoming
linear messages, and full declared typed continuation family.  It does not
establish operational `2^k` phase-port closure, arbitrary cubic hypergraphs,
CATVM custody, nonlinear or program-dependent quotient impossibility,
computational advantage, Small Wall crossing, physical waveform execution,
physical bit replacement, or unbounded catalytic computation.

The verified obstruction is that a useful repair must change the quotient or
native update law without moving exponential state into projection,
verification, or a hidden classical contraction.  Merely increasing typed
port arity would repeat the theorem rather than advance the phase machine.

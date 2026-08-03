# M127 Independent Review: Exchange-Symmetric Latent Geometry

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident grid and orbit phase-carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Transient determinant, lifted-block, and compiler buffers:
`NO_RESTORATION_CLAIM`

The verified source is commit
`8d0c04b02460d1f7b5e37c03183cd6c4c77addbc`. The family is restricted to
the exchange-symmetric sector on the fixed open `4 x 4` grid module.
`Q(zeta_17)` execution covers `k=1,2`; independent `F103/F137` execution
covers `k=1,2,3,4`, with an additional `F103` reuse-family case at `k=4`.

## Independent reconstruction

The accepted phase state is the degree-`k` symmetric power of 17 modes.
Its occupation-orbit dimension is

```text
H(k) = binomial(k+16,16) = 17,153,969,4845 for k=1,2,3,4.
```

An independent labelled `17^k` tensor implementation reproduced every sealed
boundary and reversed to its exact initial state. Additional deterministic
symmetric-vector checks over `F103` confirmed that the streamed occupation
lift equals the labelled DFT17 and exact inverse at `k=1,2,3`. Exhaustive
histogram checks confirmed that `p1,...,pk` are injective on the declared
multisets, while `p1` alone has only 17 signatures and overmerges every
declared `k >= 2` family.

Exchange symmetry is required. This result does not compress the original
labelled open-chain family. The labelled tensor and cached orbit-boundary
vectors occur only inside the independent oracle and are not accepted-path
resources or matched baselines.

## Closure, restoration, and resources

Production stores two `H(k)` orbit vectors plus 48 grid-weight cells. The
exact public DFT17 plan is lifted through streamed blocks of at most `k+1`
occupation coefficients. Each grid boundary is computed and consumed inside
one orbit-shear iteration. Production materializes no labelled tensor, dense
`H(k) x H(k)` operator, orbit boundary vector, assignment table, relation
table, or inverse history.

Only one final orbit scalar is projected. The actual reverse order
rematerializes both grid closures, reverses both lifted transforms and the
chirp, unloads the original state, restores exact zero on the same backing,
and increments the restoration generation. An unrelated program reuses that
backing with fresh/restored boundary and resource-signature agreement and no
snapshot reload.

The result reports 578 retained forward-plus-inverse plan coefficients,
`18H(k)` public occupation-topology integer cells, 146 public program integer
cells, public grid coordinates, 1,496 named compiler field cells, 133 named
transaction transient field cells, 48 named transaction transient integer
cells, final-boundary payload, and observed exact coefficient heights. The
48-cell integer figure excludes the separately reported public topology.
Operation metadata, Python/native container storage, allocator internals,
runtime, bigint workspaces, and whole-process memory are not included. Full
exact bit complexity is not established.

The lease binds the program, carrier algebra, and compiled forward-plan
fingerprint. The inverse tuple is compiler-derived and restoration detects a
bad reverse path, but this is a direct-process result and not adversarial
CATVM custody.

## Claim ceiling

M127 establishes bounded growing exchange-symmetric latent geometry with
non-sum-only shared controls, final-only projection, exact restoration, and
same-backing reuse. It reduces the declared symmetric family from labelled
`17^k` state to `H(k)` orbit state, but `H(k)` still grows and exact arithmetic
height also grows in the measured cases.

The strongest matched classical method is the identical `H(k)`-coordinate
occupation-orbit recurrence with the same elementary DFT17 plan and streamed
matching closures. No fixed-rank closure, CATVM custody, distinct phase
resource, computational advantage, Small Wall crossing, physical waveform
execution, replacement of physical bits with pi, or unbounded catalytic
computation is established.

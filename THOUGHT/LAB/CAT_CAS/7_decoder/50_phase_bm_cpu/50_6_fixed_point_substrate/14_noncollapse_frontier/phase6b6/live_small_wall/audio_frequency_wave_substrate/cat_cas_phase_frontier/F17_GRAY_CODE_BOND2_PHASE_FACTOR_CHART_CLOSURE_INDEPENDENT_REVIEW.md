# M133 Independent Review: Gray-Code Bond-2 Phase-Factor Chart

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident phase-factor carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Coupling, projection, compiler, commitment, and classical-baseline buffers:
`NO_RESTORATION_CLAIM`

The reviewed scientific source is commit
`bd76fb590c0653a26de42a2db83c9dd89c4b5d90`.

## Verified mechanism

For the declared reflected-copy family, the exact component weights obey
`w_j=(w_(j-1), eta_j reverse(w_(j-1)))`. If `x_j` are the binary digits of
the component index and `x_(m+1)=0`, then the corresponding Gray bits are
`g_j=x_j XOR x_(j+1)` and
`w_m(t)=product_j eta_j^(g_j)`. The ordered weight tensor therefore has an
open-boundary nearest-neighbor MPS with maximum bond dimension two.

Across internal cut `j`, the cross-edge matrix is
`[[1,eta_j],[eta_j,1]]`. Its determinant is `1-eta_j^2`, which is nonzero for
every declared coupling. The independent oracle directly expands the small
exact tensors and obtains rank two at every internal cut. Thus the minimal
maximum bond dimension is exactly two for declared depths at least two and is
one at depth one.

The exact `Q(zeta17)` path executes depths `1,2,4,8,16,32,64,128`; structural
parity executes depths `1..6` over both `F103` and `F137`. The accepted path
does not expand the `2^m` weights, a catalecticant, or a dense operator. Its
resident carrier has `2m` field cells, of which `m` contain nontrivial public
phase factors. The largest named local coupling has four field cells.

## Restoration, ownership repair, and reuse

Each empty resident pair is seeded as `(1,0)`, transformed by the actual
invertible local coupling to `(1,eta_j)`, consumed directly by the boundary
contraction, and transformed by the exact inverse in reverse order. The
inverse verifies `(1,0)` before unseeding the site to exact zero on the same
backing. No snapshot or retained inverse history is used. An unrelated
`PRIMARY` depth-eight transaction followed by a `REUSE` depth-sixteen
transaction agrees with fresh execution in boundary and resource signature.
The counter reported by this direct-process package is explicitly a
package-local restoration counter, not CATVM-enforced custody generation.

The initial review rejected the package because wrong-owner rejection was
hardcoded while projection and inverse did not enforce the active lease. The
repaired source now validates carrier type, stage, depth, family, and a
recomputed lease before resident contraction or inverse. Executed null,
wrong-projection-owner, and wrong-inverse-owner attacks now fail. The reviewer
reexecuted the qualifier and a direct `PRIMARY`-to-`REUSE` owner mutation after
the repair; both passed their rejection conditions.

## Independent reconstruction and attacks

The oracle does not import the production module. It separately compiles the
public descriptor, reconstructs the Gray recurrence and MPS, calculates each
small expanded tensor cut rank by exact Gaussian elimination, evaluates the
closed-form prefix/suffix total and first moment, reconstructs local inverse
action, and independently verifies commitments, byte counts, payload counts,
boundaries, and matched classical signatures for all twenty transactions.

Attacks cover straight rather than reversed upper copy, swapped factors,
perturbed reflection center, wrong, missing, and reordered inverse, singular
`eta=1`, attempted bond-one factorization, second-moment nonclosure, null
carrier, wrong projection owner, wrong inverse owner, premature projection,
and unavailable snapshot. Results expose only the final boundary and a
one-way factor commitment, not the resident factor payload.

## Matched baselines and ceiling

The strongest matched full-signature classical state is the identical `m`
public exact factors on the Gray chain. The selected final boundary closes on
two dynamic scalars, total weight and first moment. A sealed public word can
compile to a lower-triangular transfer with three nonzero field entries. These
baselines independently agree with every accepted transaction; the phase
carrier is not smaller than them.

Carrier payload and boundary payload are counted for the declared exact
algebra values, and public descriptor and commitment-record byte counts are
reported. Full process peak, Python containers and allocator, bigint and
native-library workspace, hash internals, and full exact bit-operation
complexity remain excluded. No full exact bit-complexity claim is made.

The strict ceiling is the special Gray-ordered weight tensor for this declared
affine-reflection family. It does not compact the associated coherent binary
form, whose secant rank remains `2^m`; arbitrary higher-moment boundaries do
not close on two scalars. It establishes neither general coherent-polynomial
compaction nor arbitrary-boundary closure, conventional stabilizer or
Gaussian classification, CATVM custody, a distinct phase resource,
computational advantage, Small Wall crossing, physical waveform execution,
physical replacement of bits with pi, or unbounded catalytic computation.

The next experiment should perturb beyond the pairwise Gray factor chart with
the smallest exact higher-order phase coupling or shared higher-moment port
that tests whether tensor bond must grow or a broader compact phase algebra
closes. It should not merely increase the present depth.

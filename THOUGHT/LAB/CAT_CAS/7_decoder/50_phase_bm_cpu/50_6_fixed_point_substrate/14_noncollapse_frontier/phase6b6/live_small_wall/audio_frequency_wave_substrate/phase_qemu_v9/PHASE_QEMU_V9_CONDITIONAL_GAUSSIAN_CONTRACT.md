# Phase-QEMU V9 conditional-Gaussian closed-loop obstruction contract

## Scope and authority

M267 is an exact formal theorem fixture for finite-mode, finite-label,
fixed-axis conditional Gaussian processes. It isolates the entire public
conditional-Gaussian closed-loop family that contains M265's Weyl loop and
M266's nominal trapped-ion state-dependent-force loop. It does not execute a
QEMU device, retain a software or physical carrier, establish physical
custody, or prove a complexity lower bound.

The exact package claim is:

```text
FINITE_MODE_PUBLIC_FIXED_AXIS_CONDITIONAL_GAUSSIAN_LOOPS_WITH_EXACT_FAITHFUL_CARRIER_REFERENCE_IDENTITY_REDUCE_TO_A_DIRECT_CLIENT_DIAGONAL_PHASE_OR_DECLARED_DILATION_SCHUR_CHANNEL_WHILE_POSITIVE_ACCUMULATED_CP_DIVISIBLE_MARKOV_DIFFUSION_ON_A_CLAIMED_CARRIER_SUBSPACE_PRECLUDES_EXACT_SAME_MODE_CHANNEL_RETURN_ON_THAT_SUBSPACE
```

The exact claim ceiling is:

```text
FINITE_MODE_FINITE_JOINT_CLIENT_LABEL_PUBLIC_PIECEWISE_QUADRATIC_OR_AFFINE_GAUSSIAN_DYNAMICS_WITH_FIXED_COMMUTING_CLIENT_OBSERVABLES_DECLARED_COMMON_DILATION_AND_EXACT_GAUSSIAN_MOMENT_OR_LIFTED_AFFINE_SYMPLECTIC_SEMANTICS_ONLY_NO_NONCOMMUTING_AXES_NONQUADRATIC_INTERACTIONS_NON_GAUSSIAN_BOUNDARY_MEASUREMENTS_QEC_RESTRICTED_ACCESS_NONMARKOV_RECOHERENCE_INFINITE_MODE_OR_PHYSICAL_CUSTODY
```

The restoration classification is:

```text
NO_RESTORATION_CLAIM
```

The exact restoration scope is:

```text
FORMAL_REFERENCE_COMPLETE_GAUSSIAN_CHANNEL_IDENTITY_CRITERION_AND_POSITIVE_DIFFUSION_NO_RETURN_ON_DECLARED_SUPPORT_WITHOUT_EXECUTED_OR_PHYSICAL_CARRIER_RESTORATION
```

The exact resource disposition is:

```text
GENERAL_SECTOR_DIRECT_CLIENT_SHADOW_EXISTS_WITH_EXPLICIT_L_OR_L_SQUARED_COST_AND_THE_AFFINE_LABEL_COROLLARY_IS_POLYNOMIALLY_COMPACT_WHILE_POSITIVE_DIFFUSION_ON_CLAIMED_SUPPORT_FORBIDS_EXACT_REFERENCE_COMPLETE_RETURN_SO_NO_CATALYTIC_BUS_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED
```

The selected successor is:

```text
RESTRICTED_ACCESS_NON_GAUSSIAN_PHASE_EIGENSTATE_KICKBACK_ORACLE_WITH_FAITHFUL_CARRIER_RETURN_PREPARATION_PRECISION_QUERY_AND_CUSTODY_COSTS
```

M257 remains intact. This package generalizes a mechanism obstruction; it
does not establish that every phase machine, every Gaussian computation, or
every quantum client is classically easy.

## Model and accessible algebra

Let the client expose a finite joint eigenbasis `|s>`, `s=0,...,L-1`, for a
fixed commuting family. Let the carrier `B` contain `M<infinity` bosonic modes.
The public controller is piecewise affine/quadratic in carrier quadratures, so
each client sector has a lifted affine Gaussian description

```text
G_s = (X_s, d_s, Y_s, ell_s)
```

where `X_s` is the first-moment linear map, `d_s` is displacement, `Y_s` is
Gaussian covariance noise, and `ell_s` retains the metaplectic/Weyl lift.
Keeping `ell_s` is mandatory: the projected affine symplectic endpoint does
not contain winding or central cocycle phase.

The accessible client algebra is generated only by the declared fixed
projectors `|s><s|` and the selected client boundary. The carrier algebra is
the finite-mode CCR/Weyl algebra with the public piecewise Gaussian
generators. Environment degrees of freedom are inaccessible except through a
declared common Stinespring dilation and their overlap Gram matrix. The model
does not grant a hidden snapshot, compiled answer, branch history, inverse
answer, or environment reset.

## Reference-complete identity criterion

A named carrier marginal returning is not restoration. The exact criterion is

```text
for every carrier-reference state rho_BR:
    (Phi_s tensor identity_R)(rho_BR) = rho_BR.
```

For a Gaussian channel this requires, in one fixed quadrature frame,

```text
X_s = I
d_s = 0
Y_s = 0.
```

This condition is faithful to carrier entanglement with an inert reference.
A vacuum, thermal, or rotationally invariant marginal can be fixed even when
the channel is not the identity. Exact equality of a finite list of moments
for one prepared state is therefore insufficient.

## Common-dilation reduction

The common-dilation hypothesis is substantive. Every sector isometry `V_s`
must take the same declared carrier input and same declared environment
preparation into one common output space. If the reduced carrier channel of
every sector is reference-complete identity, uniqueness of the identity
channel dilation gives, on the declared input support,

```text
V_s |psi>_B = exp(i phi_s) |psi>_B tensor |e_s>_E.
```

Absorbing `phi_s` into `|e_s>`, tracing the environment maps client matrix
elements as

```text
rho_st -> K_st rho_st
K_st = <e_t|e_s>.
```

Thus the accepted client boundary has a direct Schur-channel shadow with a
positive-semidefinite Gram kernel `K`. If the environment also returns to one
common pure state, `K` is a rank-one unit-modulus phase outer product and the
shadow reduces to a direct client diagonal unitary.

The individual Gaussian sector triples do not determine cross-sector
environment overlaps. A claimed open-system result must declare the common
dilation or an equivalent Schur kernel. Production's `kappa=3/5` fixture has
two identical carrier identity triples but maps a client `|+>` state to purity
`17/25`; the same triples with overlap one preserve purity. This is why the
general comparator is charged `L^2` kernel entries rather than only `L`
sector records.

## CP-divisible diffusion obstruction

For a time-local CP-divisible Gaussian Markov process with covariance
diffusion generator `D(t)>=0`, the accumulated covariance noise is

```text
Y(T) = integral_0^T X(T,t) D(t) X(T,t)^T dt >= 0.
```

Symplectic or affine controls can rotate and redistribute the integrand, but
cannot cancel one positive-semidefinite contribution with another. If the
restriction of `Y(T)` to the claimed carrier subspace is positive, the
identity-channel requirement `Y(T)=0` fails on that subspace. Native inverse
Hamiltonian control does not reverse an unretained Markov environment.

This statement is deliberately support-qualified. Rank-deficient diffusion
can leave a dark kernel. A computation confined to an invariant dark/noiseless
subspace is outside the positive-support conclusion. The theorem also does
not cover a finite retained environment that recoheres, non-Markovian echo,
environment reversal, error correction, measurement and feedback, or an
external reset.

## Narrow affine-Weyl-force polynomial corollary

The polynomially compact corollary is not a statement about every public
conditional Gaussian program. It applies only when every one of the following
hypotheses holds:

```text
M and q are finite
z_i in {-1,+1} are fixed commuting client labels
the public segment structure has finite length K
v_k(z) = v_k0 + sum_i z_i v_ki for every force segment k
every quadratic/symplectic propagation G_k or S_k is common to all labels
no quadratic generator depends on a client label
the final carrier displacement closes for every joint label z
the surviving branch phase comes from the bilinear Weyl cocycle
```

Common label-independent symplectic propagation preserves label affinity of
the effective force vectors. The Weyl cocycle is bilinear in pairs of those
vectors, so the compiled client phase is a polynomial of degree at most two in
the `z_i`. If a force has higher label degree, a quadratic generator depends
on a label, or closure fails for any joint label, this corollary fails closed.
It does not silently materialize a generic `L=2^q` branch table and call it an
affine descriptor.

The public input descriptor is charged as

```text
O(K (M^2 + q M)) scalars
```

for dense common `2M x 2M` propagation plus label-affine force coefficients.
The compiled phase table has `O(q^2)` coefficients. One explicit dense
compilation upper bound is

```text
O(K M^3 + K q M^2 + K q^2 M) arithmetic,
```

and applying the compiled phase to one joint label costs `O(q^2)` arithmetic.
These are conservative public charges, not complexity lower bounds. `K` is
fixed for one program but is not assumed constant across a growing family;
descriptor size, compilation, and application work all remain in the ledger.

## Exact fixtures

Production emits the following names verbatim.

### `metaplectic_2pi_vs_zero`

One client sector performs zero oscillator rotation and the other performs a
`2 pi` rotation. Both projected symplectic endpoints are `I`, but the
metaplectic lifts are `+I` and `-I`. The carrier channel is identity and the
surviving client shadow is `diag(1,-1)`. This fixture forbids reduction to
only `X_s` and `d_s`.

### `metaplectic_4pi_control`

A `4 pi` rotation has projected endpoint `I` and lifted sign `+1`. It is the
winding control for the `2 pi` fixture.

### `weyl_rectangle_cocycle`

Production freezes

```text
W(v) W(w) = exp[-i sigma(v,w)/2] W(v+w)
a = 1/2
b = 1/3
xi(z0)  = (a z0,0)
eta(z1) = (0,b z1)
sequence(z0,z1) = [xi,eta,-xi,-eta]
phi(z0,z1) = -a b z0 z1 = -z0 z1/6 radians.
```

Every sector closes to zero displacement. In the declared sector order

```text
[++,+-,-+,--]
```

the exact phase exponents are `[-1/6,+1/6,+1/6,-1/6]` radians and the direct
two-qubit client diagonal is

```text
[exp(-i/6), exp(+i/6), exp(+i/6), exp(-i/6)].
```

This is the affine-force `ZZ` rectangle, not a scalar two-sector phase gate.
The carrier closes in all four sectors while the bilinear Weyl cocycle remains.

### `vacuum_rotation_marginal_false_positive`

The bus undergoes `R(pi/2)`. A rotationally invariant bus marginal is
unchanged. In the canonical `[q,p]=i`, vacuum-covariance-`I/2` convention, the
exact bus-reference TMSV covariance uses dimensionless parameters

```text
cosh(2r) = 5/4
sinh(2r) = 3/4
(5/4)^2 - (3/4)^2 = 1.
```

Its bus marginal remains `(5/8)I`, but the `3/8` cross-covariance rotates.
The full covariance changes and reference-complete identity is rejected. The
fixture has covariance condition number four and mean occupation `1/8` per
mode.

### `additive_diffusion`

With vacuum covariance `I/2` and accumulated Gramian

```text
Y = (1/8) I,
```

the output covariance is `(5/8)I`, determinant `25/64`, purity `4/5`, and
added mean occupation `1/8` in the declared quadrature convention. `Y` has
rank two and is positive on the full one-mode support, so exact channel return
is false.

### `pure_loss_fixed_point`

The pure-loss channel with `eta=1/2` fixes the prepared vacuum but is not the
identity channel. On the exact TMSV probe the bus variance changes from `5/8`
to `9/16` and the `3/8` cross-correlation scales by `1/sqrt(2)`. This fixture prevents a
fixed prepared state from being promoted to reference-complete return.

### `rank_deficient_dark_mode`

The two-mode accumulated Gramian is

```text
diag(1/8,1/8,0,0).
```

Its noisy support is mode zero and its two-dimensional kernel is mode one.
The no-return conclusion is asserted on the noisy support only. Dynamics that
remain wholly in the dark kernel are an explicit scope escape, not a failure
of the support-qualified theorem.

### `finite_environment_recurrence`

A retained two-mode beam splitter obeys

```text
a_S(theta) = cos(theta) a_S + sin(theta) a_E.
```

At `theta=pi/2`, information occupies the environment; at `theta=2pi`, the
joint Heisenberg action recurs. This finite-memory recoherence is outside the
CP-divisible Markov-diffusion assumption. Production therefore makes no claim
that every intermediate noisy-looking evolution can never recohere.

### `common_nontrivial_bus_evolution`

Both client sectors perform the same bus rotation `R(pi/2)`, while their
lifted client phases are zero and `pi/3`. The joint endpoint factorizes as a
direct client diagonal tensor a common bus rotation. The direct client shadow
exists, but the bus endpoint is not identity. Branch-relative factorization
is not carrier restoration.

### `declared_environment_schur`

Two sector carrier channels are both identity while declared environment
states have overlap `kappa=3/5`. Their client Schur kernel is

```text
[[1,   3/5],
 [3/5,   1]].
```

The `|+>` output purity is exactly `17/25`. Comparing with overlap one proves
that the branch-local Gaussian triples `(I,0,0)` alone do not fix the client
channel.

### `sector_scaling`

For `q in [1,2,4,8,12]`, production records `L=2^q`. A completely general
direct diagonal shadow has `L` phase entries; a declared general Schur kernel
has `L^2` entries. No blanket polynomial efficiency follows.

For the narrow affine-Weyl-force family defined above, the closed-loop
cocycle phase is at most quadratic in the `q` commuting labels.
The fixture deliberately reports the coarse `q^2` scaling counts

```text
[1,4,16,64,144]
```

for `q=[1,2,4,8,12]`; these are scaling receipts, not exact monomial counts.
For the displayed `M=1`, `K=4` family, the public dense input descriptor count
is `K[(2M)^2+2M(q+1)]`, producing `[32,40,56,88,120]` scalars. In general the
input is `O(K(M^2+qM))`, the compiled phase has `O(q^2)` coefficients, and the
dense compilation and per-label application work are charged as stated in the
corollary section. This does not compress an arbitrary client phase function
or environment Gram matrix.

## Strongest honest comparator and M257

Once reference-complete carrier identity is established, the strongest
equal-access comparator applies the direct client boundary map and omits the
carrier loop and any positive-cost restoration stage. It must be charged
honestly:

```text
closed unitary sectors:     L lifted client phases
declared open dilation:     L^2 Schur-kernel entries in general
affine-Weyl input:           O(K (M^2 + q M)) public scalars
compiled affine-Weyl phase: O(q^2) coefficients
compile upper bound:        O(K M^3 + K q M^2 + K q^2 M) arithmetic
one phase application:      O(q^2) arithmetic
```

Nothing here proves that arbitrary phase lists, arbitrary Schur kernels, or
arbitrary client dynamics have compact classical representations. If the
client itself carries a difficult noncommuting quantum computation, that
difficulty is not a catalytic Gaussian-bus advantage.

M257 is not escaped: under equal public access, the accepted direct client
shadow omits the finite conditional-Gaussian bus process and its restoration.
With positive accumulated Markov diffusion on claimed support, the bus also
fails exact same-mode reference-complete return.

## Resource, precision, and custody ledger

The production fixture uses exact Python rational arithmetic plus declared
symbolic `pi` and `sqrt(2)` values. Exact numerator and denominator sizes and
lifted winding descriptors are resources; a real implementation may not hide
them in an ideal real number. General label costs are stated in `L`, not
silently rewritten as polynomial in `q=log2 L`.

The formal probes instantiate at most two carrier modes. TMSV conditioning is
four and its mean energy is finite. Physical frequency, bandwidth, latency,
drive energy, bath temperature, wall-plug energy, calibration precision, and
readout precision remain uninstantiated. No physical carrier, same backing,
controller, source, detector, environment, or restoration process was
executed or held in custody.

## Forbidden promotions

M267 does not establish:

```text
physical execution or physical observation
physical or software same-backing restoration
QEMU device execution
all noise changes all states
intermediate noise can never recohere
absence of dark or noiseless subspaces
polynomial classical simulation of arbitrary client dynamics
a complexity lower bound
a resource advantage
an M257 escape
unbounded computation
REPLACE_THE_BIT_WITH_PI
```

The theorem excludes noncommuting client axes, nonquadratic interactions,
non-Gaussian states or boundary measurements used as computational resources,
QEC, measurement/adaptation, restricted access, non-Markovian recoherence,
infinite-mode limits, and physical custody. Those exclusions delimit the next
architecture search; they are not claims that those mechanisms succeed.

## Determinism and evidence boundary

`conditional_gaussian_closed_loop_obstruction.py` prints one sorted compact
JSON object to stdout and nothing to stderr on success. It includes explicit
self-checks and the SHA-256 digest of its own source bytes. That digest seals
the executable only; independent verification and resource review remain
separate authority layers.

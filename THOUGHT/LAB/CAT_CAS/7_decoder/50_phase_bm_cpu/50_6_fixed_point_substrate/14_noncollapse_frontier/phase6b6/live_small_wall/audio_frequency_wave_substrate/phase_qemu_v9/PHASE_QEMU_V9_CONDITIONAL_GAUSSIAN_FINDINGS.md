# Phase-QEMU V9 conditional-Gaussian class findings

## Authority and disposition

M267 closes a bounded mechanism class. It is a formal and executable
conformance result for finite-mode, public-descriptor, fixed-axis conditional
Gaussian loops; it is not a new physical-carrier demonstration.

The exact package claim is:

```text
FINITE_MODE_PUBLIC_FIXED_AXIS_CONDITIONAL_GAUSSIAN_LOOPS_WITH_EXACT_FAITHFUL_CARRIER_REFERENCE_IDENTITY_REDUCE_TO_A_DIRECT_CLIENT_DIAGONAL_PHASE_OR_DECLARED_DILATION_SCHUR_CHANNEL_WHILE_POSITIVE_ACCUMULATED_CP_DIVISIBLE_MARKOV_DIFFUSION_ON_A_CLAIMED_CARRIER_SUBSPACE_PRECLUDES_EXACT_SAME_MODE_CHANNEL_RETURN_ON_THAT_SUBSPACE
```

The exact claim ceiling is:

```text
FINITE_MODE_FINITE_JOINT_CLIENT_LABEL_PUBLIC_PIECEWISE_QUADRATIC_OR_AFFINE_GAUSSIAN_DYNAMICS_WITH_FIXED_COMMUTING_CLIENT_OBSERVABLES_DECLARED_COMMON_DILATION_AND_EXACT_GAUSSIAN_MOMENT_OR_LIFTED_AFFINE_SYMPLECTIC_SEMANTICS_ONLY_NO_NONCOMMUTING_AXES_NONQUADRATIC_INTERACTIONS_NON_GAUSSIAN_BOUNDARY_MEASUREMENTS_QEC_RESTRICTED_ACCESS_NONMARKOV_RECOHERENCE_INFINITE_MODE_OR_PHYSICAL_CUSTODY
```

Restoration is classified:

```text
NO_RESTORATION_CLAIM
```

at the exact scope:

```text
FORMAL_REFERENCE_COMPLETE_GAUSSIAN_CHANNEL_IDENTITY_CRITERION_AND_POSITIVE_DIFFUSION_NO_RETURN_ON_DECLARED_SUPPORT_WITHOUT_EXECUTED_OR_PHYSICAL_CARRIER_RESTORATION
```

The exact negative resource disposition is:

```text
GENERAL_SECTOR_DIRECT_CLIENT_SHADOW_EXISTS_WITH_EXPLICIT_L_OR_L_SQUARED_COST_AND_THE_AFFINE_LABEL_COROLLARY_IS_POLYNOMIALLY_COMPACT_WHILE_POSITIVE_DIFFUSION_ON_CLAIMED_SUPPORT_FORBIDS_EXACT_REFERENCE_COMPLETE_RETURN_SO_NO_CATALYTIC_BUS_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED
```

This establishes no executed carrier restoration, physical same-mode custody,
physical observation, resource advantage, Small Wall crossing, asymptotic or
unbounded computation, or replacement of bits with pi. M257 remains intact.

## Class under test

Let a finite client have `L` joint labels `s` in the simultaneous eigenbasis of
fixed commuting observables. Let the carrier contain `M` bosonic modes with
quadrature vector

```text
R = (q1, p1, ..., qM, pM),       [Rj, Rk] = i Omega_jk.
```

On each public piecewise interval and in each client sector, the declared
Hamiltonian is at most quadratic and affine:

```text
H_s(t) = (1/2) R^T G_s(t) R + h_s(t)^T R + c_s(t).
```

The corresponding closed-system carrier operation is Gaussian. Its complete
unitary record is a *lifted* affine-symplectic datum

```text
g_tilde_s = (S_s, d_s, mu_s, phi_s),
```

where `S_s` is symplectic, `d_s` is a displacement, `mu_s` identifies the
continuous metaplectic lift/Maslov sheet, and `phi_s` retains scalar phase,
including Weyl-composition cocycles. The pair `(S_s,d_s)` alone is not a
faithful unitary description: a closed classical symplectic path can finish at
`S=I,d=0` while its metaplectic lift is `-I`, and a closed displacement
rectangle can leave a nonzero geometric phase.

For open Gaussian evolution, the sector channel acts on first moments and
covariances as

```text
m -> X_s m + d_s
V -> X_s V X_s^T + Y_s,
```

with complete positivity imposed on `(X_s,Y_s)`. Client coherences additionally
require one declared *common* system-environment dilation. Independent
per-sector reduced moment maps do not determine the cross-sector coherent
channel, so they are not sufficient evidence.

This class excludes noncommuting client axes, nonquadratic carrier
interactions, non-Gaussian boundary measurements, measurement/adaptation,
error correction, restricted or secret implementation access, non-Markovian
environment reversal, infinite-mode limits, and physical custody claims.

## Reference-complete return criterion

Exact same-carrier return is a channel statement, not equality on one prepared
state. Introduce an inert reference `R_ref` that may be correlated or entangled
with the carrier `B`. The accepted criterion on the claimed carrier subspace is

```text
(E_B tensor I_ref)(rho_Bref) = rho_Bref
```

for every allowed `rho_Bref`. In the Gaussian class this is equivalently the
identity affine Gaussian channel on that invariant symplectic mode subsystem,
tested with faithful carrier-reference first moments and covariance rather
than only a local marginal. A vacuum, thermal state, or another symmetric
fixed point by itself cannot certify this criterion.

For a closed conditional Gaussian unitary, exact initial-state return requires
every sector endpoint to have

```text
S_s = I,     d_s = 0
```

on the claimed carrier subspace. The lifted endpoint may nevertheless retain
a scalar phase `exp(i phi_s)`. Thus the full conditional unitary reduces there
to

```text
U = sum_s exp(i phi_s) |s><s| tensor I_B,
```

and the exact strongest comparator applies the diagonal client phase directly,
without instantiating or restoring the carrier.

A sector-independent but nonidentity Gaussian carrier unitary also factorizes
from the client and can be omitted by a client-only forward shadow. It is not
return to the initial carrier state unless that common evolution is explicitly
outside the claimed transaction or is itself inverted; silently changing to a
co-moving frame does not establish restoration.

## Common-dilation Schur reduction

For a declared open-system model, write the common dilation in sector form as

```text
V = sum_s |s><s| tensor V_s
```

on client, carrier, and one common environment initially uncorrelated with the
carrier in its declared state, with inert purification allowed. If the
carrier-reference channel is exactly identity in every sector,
then a faithful dilation cannot leave carrier-input information in the
environment. It may still leave a sector-dependent environment record. After
the environment is traced out, the only resulting client effect is

```text
rho_C -> D [K o rho_C] D_dagger,
```

where `D=diag(exp(i phi_s))`, `o` is entrywise multiplication, and

```text
K_st = <e_t|e_s>
```

for a pure dilation. With a purified mixed environment it remains the ordinary
inner product of purified environment-output vectors, equivalently a
state-weighted trace in a controlled-unitary realization. Consequently `K` is
positive semidefinite,
`K_ss=1`, and `|K_st|<=1`. This is a direct client Schur channel. The same
conclusion is obtained from the cross maps
`Phi_st(X)=Tr_E[V_s(X tensor sigma_E)V_t_dagger]`: reference-complete carrier
identity forces the accepted cross map to be a scalar multiple of `X`.

This reduction is narrower than the assertion that every Gaussian circuit is
easy to simulate. Non-Gaussian measurements can turn a Gaussian state
preparation into a nontrivial sampling model, and those measurements are
outside this class. The result says that a carrier which returns as the exact
identity channel supplies no residual client effect beyond the direct diagonal
phase or declared Schur multiplier.

## Positive Markov diffusion obstruction

For a differentiable CP-divisible Markov Gaussian evolution, let `A(t)` be the
linear drift, `D(t)>=0` the instantaneous diffusion, and `Phi(T,t)` the
fundamental propagator. The accumulated covariance noise is the controllability
Gramian

```text
Y(T) = integral_0^T Phi(T,t) D(t) Phi(T,t)^T dt >= 0.
```

Every integrand is positive semidefinite. Hamiltonian symplectic control can
rotate, squeeze, and transport this noise, but it cannot cancel one positive
contribution with another. If the accumulated Gramian is positive on a
claimed carrier subspace, then `Y(T)` is nonzero there. The identity Gaussian
channel requires `X(T)=I`, `d(T)=0`, and `Y(T)=0`; exact reference-complete
same-mode return is therefore impossible on that support.

The support qualification is essential. Rank-deficient diffusion can leave a
dark kernel, and a carrier encoded wholly in an invariant, dynamically
decoupled kernel is not obstructed by this argument. Nor does the result cover
a finite environment which coherently returns information, a non-Markovian
recoherence, an explicit reversal of the environment, measurement and active
error correction, or replacement/recooling. Those mechanisms change the
assumptions or export the cost; they are not counterexamples inside the
declared class.

Pure loss also shows why a fixed state is not a channel certificate. The
vacuum is fixed by a quantum-limited attenuator even though the channel is not
identity. A faithful carrier-reference input exposes the attenuated
correlations. The obstruction is to identity on all allowed inputs, not to the
existence of one fixed point.

## Analytic fixture findings

The following named fixtures are the exact conformance basis. Floating-point
production and separate-reference receipts may approximate these values, but
they do not widen the analytic claim.

### `metaplectic_2pi_vs_zero`

For one harmonic mode,

```text
U(theta) = exp[-i theta (n + 1/2)].
```

At `theta=2 pi`, the quadrature endpoint is `S=I,d=0`, but `U=-I`. The
zero-length path gives `+I`. The endpoint symplectic matrix alone therefore
loses a physically coherent sign.

### `metaplectic_4pi_control`

The same oscillator at `theta=4 pi` has `S=I,d=0,U=+I`. This is the matched
lift control and distinguishes a continuous metaplectic sheet from an
arbitrary endpoint sign bit.

### `weyl_rectangle_cocycle`

With the declared Weyl convention,

```text
W(xi) W(eta) W(-xi) W(-eta)
    = exp[-i xi^T Omega eta] I.
```

Take the orthogonal conditional sides `xi=(a Z0,0)`, `eta=(0,b Z1)`, with
`a=1/2` and `b=1/3`. The carrier displacement closes, while the client obtains

```text
exp[-i Z0 Z1 / 6].
```

In the declared sector order `(z0,z1)=(+,+),(+,-),(-,+),(-,-)`, the four
client phases are therefore

```text
[exp(-i/6), exp(+i/6), exp(+i/6), exp(-i/6)].
```

This is a Weyl cocycle retained by the lift and reproduced directly by the
client comparator.

### `vacuum_rotation_marginal_false_positive`

A bus rotation by `pi/2` leaves the vacuum marginal `V=I2/2` unchanged, so a
vacuum-only test falsely reports return. Use instead a two-mode-squeezed
bus-reference covariance with

```text
cosh(2r)=5/4,     sinh(2r)=3/4,
V_Bref=(1/2) [[(5/4)I2, (3/4)Z], [(3/4)Z, (5/4)I2]].
```

The local bus covariance remains rotationally symmetric, but the bus-reference
cross block rotates and differs from its input. The faithful reference rejects
identity.

### `additive_diffusion`

For one mode,

```text
X=I2,     Y=(1/8)I2.
```

Vacuum evolves from `V=I2/2` to `V=(5/8)I2`, giving
`nbar=1/8` and purity `4/5` in the declared `[q,p]=i` convention. The
accumulated Gramian has full rank two, so no nonzero carrier quadrature lies in
its kernel.

### `pure_loss_fixed_point`

For the quantum-limited attenuator

```text
eta=1/2,     X=sqrt(eta) I2,     Y=(1-eta)I2/2,
```

the vacuum marginal is fixed exactly. The faithful two-mode-squeezed fixture
above is not: its bus-reference cross covariance is multiplied by
`sqrt(eta)=1/sqrt(2)`. Hence fixed-vacuum return is not identity-channel
return.

### `rank_deficient_dark_mode`

For two modes,

```text
X=I4,     Y=diag(1/8,1/8,0,0).
```

The Gramian has rank two and a two-dimensional kernel. The first mode fails
return; an invariant second-mode encoding can pass. A full two-mode carrier
claim still fails. This is the required dark-kernel scope control, not an
exception to the support-qualified theorem.

### `finite_environment_recurrence`

An exact two-mode beam-splitter symplectic evolution exchanges system and
finite-environment excitations at its midcycle and returns the complete joint
state after a `2 pi` cycle. The reduced system evolution recoheres and is not a
CP-divisible Markov diffusion over the full cycle. It is therefore an explicit
outside-scope control showing why the Markov/common-environment assumptions
cannot be omitted.

### `common_nontrivial_bus_evolution`

Give every client sector the same `pi/2` bus rotation. The total operation
factorizes into client evolution times that common bus rotation, so the direct
client shadow is valid. The bus nevertheless does not return for a generic
input. Only an explicitly factored external frame convention or an executed
inverse can remove the common rotation from the restoration transaction.

### `declared_environment_schur`

Two branches have identical identity carrier triples but declared environment
overlap `kappa=3/5`. The exact direct client matrix is

```text
K = [[1,3/5], [3/5,1]].
```

It is positive semidefinite. A client `|+>` state becomes
`[[1/2,3/10],[3/10,1/2]]` with purity `17/25`. The carrier can return while
the environment retains branch information; the direct Schur shadow includes
that dephasing and no carrier restoration stage.

### `sector_scaling`

For `q=[1,2,4,8,12]` binary fixed-axis client observables, a generic joint
sector description has respectively

```text
L   = [2,4,16,256,4096]
L^2 = [4,16,256,65536,16777216].
```

A generic diagonal phase costs `Theta(L)` values. A generic declared Schur
channel costs `Theta(L^2)` entries, or an explicitly charged factorization of
equivalent rank-dependent size. There is no general polynomial-compression
claim.

For the narrower affine-displacement/Weyl-force loop corollary, every public
segment has a fixed, label-independent symplectic/quadratic kernel `S_k` (or
`G_k`) and a force

```text
v_k(z) = v_k0 + sum_i z_i v_ki.
```

The segment structure and finite per-program count `K` are public and charged;
`K` is not assumed constant across a growing family. Every branch displacement
closes, and label-dependent quadratic generators are excluded.
Under exactly these hypotheses, the bilinear Weyl cocycle makes the surviving
closed-loop phase at most quadratic in the `q` binary labels. Its direct phase
coefficient table is `O(q^2)` (`1,4,16,64,144` under the deliberately coarse
`q^2` fixture accounting), while the input descriptor costs
`O(K(M^2+qM))` before compilation. Polynomial compactness belongs only to this
common-symplectic affine-force subclass with `K`, coefficient precision, and
compilation/application work charged. It does not extend to arbitrary
label-affine quadratic generators, sector phases, or Schur matrices.

## Strongest honest comparator and resource law

The comparator is constructed from exactly the public accepted endpoint, not
from a crippled surrogate:

```text
closed unitary case:
    apply D = diag(exp(i phi_s)) directly to the client

declared common-dilation case:
    apply rho -> D [K o rho] D_dagger directly to the client
```

It omits carrier allocation, Gaussian segment evolution, endpoint testing, and
restoration because the theorem has already certified that those stages leave
the accepted carrier-reference channel equal to identity. It must still pay
for all client information: `Theta(L)` generic phases and up to `Theta(L^2)`
generic Schur entries. If a public compiler derives these objects from a
smaller affine or bounded-degree descriptor, equal implementation access gives
that same compiler to the comparator.

For a family indexed by `n`, the honest accounting is therefore at least

```text
R_phase(n) = {
    M physical or modeled modes,
    2M carrier quadratures,
    L joint client sectors,
    public segment and Gaussian-generator descriptor,
    lifted metaplectic/Maslov and Weyl-cocycle data,
    exact or effective precision,
    common-dilation/environment descriptor,
    interaction count, bandwidth, energy, latency, and controller state,
    restoration or exported reset cost
}

R_shadow(n) = {
    L client phases or an honestly compressed public compiler,
    up to L^2 Schur data or an honestly charged factorization,
    exact or effective precision,
    client application work, bandwidth, and state
}.
```

The theorem supplies no asymptotic separation. If `L=2^q`, the generic direct
shadow is exponential in `q`, but the phase machine must also specify or
physically generate the corresponding generic sector structure; this package
does not erase that cost. Conversely, when the common-symplectic affine-force
subclass above makes the phase law `O(q^2)`, it gives the equal-access
comparator the same compact law and charges the public `K`-segment input
descriptor. Depth on a fixed finite Gaussian recurrence is insufficient for
the long-term unbounded-compute claim.

Positive CP-divisible diffusion makes the catalytic balance worse on its
support: exact identity cannot be recovered by more Gaussian Hamiltonian
control. Recooling, fresh-mode substitution, snapshot reload, environment
reset, and active QEC must be separately named and charged. They are not
catalytic same-carrier restoration.

## Controls against overreach

The findings do **not** assert that:

- every Gaussian quantum process is classically easy;
- arbitrary `L`-sector phases or Schur matrices have polynomial descriptors;
- one invariant vacuum or thermal marginal proves channel return;
- diffusion on one mode prevents exact use of a dynamically isolated dark
  mode;
- finite environments cannot recohere;
- Gaussian or non-Gaussian QEC is impossible under every resource model;
- noncommuting client axes reduce to a diagonal client channel;
- nonquadratic interactions or non-Gaussian measurements obey this closure;
- mathematical channel identity proves physical same-device custody; or
- a software model establishes physical energy, bandwidth, noise, or
  precision scaling.

The Gaussian-QEC no-go literature is relevant to restricted Gaussian
correction schemes, but M267 does not import it as a universal QEC theorem.
Likewise, Gaussian boson sampling is an explicit reminder that non-Gaussian
boundary measurement changes the computational class.

## Scientific basis

The phase-space, Gaussian-unitary, Gaussian-channel, and dilation conventions
follow the authoritative [Gaussian quantum information review](https://arxiv.org/abs/1110.3234).
The CP-divisibility boundary is aligned with the primary analysis of
[non-Markovian Gaussian channels](https://arxiv.org/abs/1504.00671). The
metaplectic endpoint qualification is grounded in the mathematical treatment
of the [Maslov index and metaplectic representation](https://doi.org/10.1016/0022-1236(92)90104-Q).

The exclusions are substantive. The [Gaussian quantum error-correction no-go
theorem](https://arxiv.org/abs/0811.3128) applies under its own Gaussian
assumptions; [Gaussian boson sampling](https://arxiv.org/abs/1612.01199) uses
non-Gaussian photon counting; and [dark-mode theorems for quantum
networks](https://arxiv.org/abs/2312.06274) formalize the possibility of
decoherence-free modal kernels. These sources support ingredients and scope
boundaries. They do not establish this Phase-QEMU transaction, a physical
carrier observation, or a resource advantage.

## Next mechanism

Conditional Gaussian loops are retired as a long-term resource route under
the public fixed-axis, equal-access assumptions above. Spin-dependent
squeezing and other piecewise-quadratic variants remain inside the closed
class; adding another such fixture is not a mechanism change.

The selected successor is:

```text
RESTRICTED_ACCESS_NON_GAUSSIAN_PHASE_EIGENSTATE_KICKBACK_ORACLE_WITH_FAITHFUL_CARRIER_RETURN_PREPARATION_PRECISION_QUERY_AND_CUSTODY_COSTS
```

Its minimum Phase-QEMU model must make the changed assumption explicit rather
than hide implementation from the comparator. It must identify the physical
phase eigenstate or finite-energy approximation, the genuinely non-Gaussian
interaction, lawful oracle/query boundary, carrier and source custody,
faithful return test, preparation and rematerialization work, precision/noise
law, energy, bandwidth, latency, query count, and scaling with problem size.
The strongest comparator receives every interface and descriptor the access
contract permits. If the putative oracle is merely a public deterministic
compiler or an answer-bearing prepared state, M257 kills it.

The lane remains nonterminal.

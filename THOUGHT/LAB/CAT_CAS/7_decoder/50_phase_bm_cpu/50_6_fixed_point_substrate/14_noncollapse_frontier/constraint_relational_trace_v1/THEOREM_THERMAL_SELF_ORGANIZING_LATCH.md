# Clause-Local Thermal Latch and Self-Organizing Preparation Candidate

## Status

```text
DIRECT_PUBLIC_CLAUSE_COMPILATION_ESTABLISHED
CONSTANT_NORMALIZED_ZERO_MODE_POPULATION_ESTABLISHED_REFERENCE
TERMINAL_AGNOSTIC_FLOW_BOOLEAN_EQUILIBRIUM_LAW_ESTABLISHED_REFERENCE
SMOOTH_REGION_COTANGENT_LIFT_ESTABLISHED
POLYNOMIAL_WORST_CASE_PREPARATION_NOT_ESTABLISHED
GLOBAL_REVERSIBLE_ENVIRONMENT_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## 1. Direct public carrier

Let

```text
H_F = sum_m Pi_viol,m
```

where each `Pi_viol,m` is the diagonal three-variable projector onto the unique local
row that violates clause `m`. The Hamiltonian is compiled directly from public clause
incidence and literal signs. It contains no witness, solution count, selected branch, or
clean-sector amplitude.

The physical description uses `n` binary coordinates and `m` constant-local terms.
Its reference matrix dimension is `2^n`, but that matrix is not part of the native
compiler.

## 2. Constant normalized thermal boundary

Choose

```text
beta = (n + s) ln 2
```

for a fixed positive safety constant `s`. If `F` is satisfiable, let `g >= 1` be the
zero-energy degeneracy. Since all excited energies are integers at least one,

```text
Z_excited <= 2^n exp(-beta) = 2^-s.
```

Therefore the normalized zero-energy population obeys

```text
p0 = g / Z >= 1 / (1 + 2^-s).
```

For `s=2`, `p0 >= 4/5`, including the unique-witness case. If `F` is unsatisfiable,
`p0=0` exactly because the spectrum contains no zero-energy state.

This removes the exponentially small normalized trace and exceptional-point input
coupling from the mathematical boundary. It does not prepare the Gibbs state.

Repeated energy measurements provide a one-sided bounded-error detector conditional on
Gibbs preparation. A deterministic exact boundary still requires a lawful population
or presence projection.

## 3. Terminal-agnostic clause flow

The preparation candidate uses one voltage per public variable and two memory
coordinates per clause. For clause `m`,

```text
C_m(v) = 1/2 min_i (1 - q_i,m v_i).
```

The voltage and memory vector field is clause-local and terminal agnostic. Shared
variable nodes couple every incident clause, allowing collective motion without a
candidate list. The implementation follows the memory-assisted self-organizing 3-SAT
flow introduced in arXiv:2011.06551.

The current reference establishes:

- state dimension `n + 2m`;
- exactly `3m` literal couplings;
- every satisfying Boolean corner is a projected equilibrium;
- no nonsatisfying Boolean corner is a projected equilibrium;
- a deterministic public perturbation breaks exact neutral symmetry;
- small satisfiable instances reach a public satisfying section without a witness.

It does not establish:

- exclusion of every non-Boolean equilibrium;
- convergence from every public initial condition;
- a polynomial worst-case convergence time;
- a total UNSAT stopping law.

A reference integrator is instrumentation only. The intended compute object is the
continuous relation-valued flow.

## 4. Reversible cotangent lift

For any smooth region of an autonomous flow

```text
q_dot = f_F(q),
```

define the cotangent Hamiltonian

```text
H(q,p) = p dot f_F(q).
```

Hamilton's equations give

```text
q_dot = f_F(q)
p_dot = -Df_F(q)^T p.
```

The finite transport is

```text
(q0,p0) -> (Phi_t(q0), D Phi_t(q0)^(-T) p0),
```

and negative time is the program-derived inverse. This is a genuine reversible
dilation on smooth regions, not transcript replay.

The lift exposes the compensation law:

```text
primal attraction <-> cotangent expansion.
```

Any contraction that makes solution finding robust can reappear as momentum range,
precision, or environment energy during exact restoration.

The published clause flow also contains hard coordinate projections and piecewise
minimum selectors. Global reversible closure requires either:

1. smooth bounded replacements with proved equivalent solution dynamics; or
2. a carrier that retains switching and boundary-event information without an
   exponentially growing event transcript.

## 5. Exact remaining theorem

A proof through this route requires a uniform family of clause flows satisfying all of
the following:

```text
all SAT instances reach a satisfying basin in polynomial physical time
all UNSAT instances reach a distinct valid no-solution boundary in polynomial time
required integration precision is polynomial
memory and cotangent dynamic range are polynomial
the complete system-plus-bath evolution has a native inverse
the boundary is deterministic or admits deterministic polynomial simulation
```

The new candidate is stronger than the exceptional-point latch because it begins with
local public clauses rather than an assumed scalar `#SAT/2^n`. Its unresolved burden is
preparation and reversible closure, not clean-sector access.

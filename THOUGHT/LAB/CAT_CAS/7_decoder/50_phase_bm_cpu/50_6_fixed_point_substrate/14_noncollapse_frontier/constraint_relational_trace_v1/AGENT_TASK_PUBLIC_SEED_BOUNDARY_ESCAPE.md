# Agent Task: Public-Seed Boundary-Stratum Escape

## Mission

Advance **PR #49 only** toward the public-seed completeness theorem for the reduced
exact-product phase carrier.

Do not work on PR #50, Jacobian fiber pushforward, holonomy, Family 10h, audio transfer,
retired `live_small_wall` packages, or hardware testing. Those lanes have separate
owners.

```text
repository: Reneromero08/agent-governance-system
branch: codex/constraint-relational-trace-proof-campaign
starting head: 63b7308ac4cf2111c9d928ff03d971d1ef70ef9b
PR: #49
```

The task is not another broad SAT scaling campaign. The task is to establish or falsify
**public-seed escape from every non-solution invariant set compatible with the remaining
memory-boundary strata**.

Commit and push coherent green checkpoints to the same branch without pausing for a
separate commit ceremony. Do not merge the PR.

## Claim ceiling

Maintain these exact limits throughout:

```text
PUBLIC_SEED_COMPLETENESS_NOT_ESTABLISHED
UNIFORM_POLYNOMIAL_DEADLINE_NOT_ESTABLISHED
ROBUST_TERMINAL_MARGIN_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

A finite campaign, absence of discovered counterexamples, or numerical convergence is
not a proof.

## Required starting audit

Before editing:

```bash
git status --short --branch
git rev-parse HEAD
git branch --show-current
git worktree list
```

Require:

```text
branch = codex/constraint-relational-trace-proof-campaign
HEAD descends from 63b7308ac4cf2111c9d928ff03d971d1ef70ef9b
no unrelated dirty files
```

Use the repository virtual environment. On the current Linux host the reusable base
virtual environment has been located under:

```text
/run/media/reneshizzle/860_1/CCC 2.0/AI/agent-governance-system/.venv-linux/bin/python
```

Resolve the executable rather than silently falling back to system Python.

Run the current focused qualification before changing the mechanism:

```bash
$PY -m pytest THOUGHT/LAB/CAT_CAS/tests/constraint_relational_trace -q
$PY THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/constraint_relational_trace_v1/run_reference_campaign.py
$PY THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/constraint_relational_trace_v1/run_mechanism_campaign.py
$PY CAPABILITY/TOOLS/governance/critic.py
```

If the base is not green, separate infrastructure failure from mechanism failure before
continuing.

## Frozen current mechanism

The strongest current candidate is:

```text
public 3-CNF
-> sorted public variable boundary
-> public rational S1 phase seed
-> exact clause-product violation
-> exact negative clause-product direction
-> short and long clause memories
-> clause replicator selector
-> fixed terminal assignment verification
```

The active native carrier is implemented at:

```text
2n + 5m coordinates
```

with an angle/logit chart of:

```text
n + 4m coordinates.
```

Primary implementation files:

```text
polynomial_phase_selector_flow.py
reduced_exact_product_phase_flow.py
adaptive_phase_logit_flow.py
constraint_holo.py
self_organizing_clause_flow.py
```

Do not reintroduce pair selectors into exact-product dynamics. Their active-field
decoupling is already proved and regression-tested.

## Established theorem surface

Read these before designing new work:

```text
THEOREM_EXACT_PRODUCT_PHASE_DIRECTION.md
THEOREM_EXACT_PRODUCT_REDUCTION_AND_ENERGY_WALL.md
THEOREM_SAT_STATIONARY_POINT_WALL.md
THEOREM_MEMORY_LOGIT_CLOCK.md
THEOREM_STATIONARY_MEMORY_STRATA.md
THEOREM_PUBLIC_SEED_MEMORY_REDUCTION.md
THEOREM_REDUCED_PHASE_FORWARD_RESOURCE_TRANSFER.md
THEOREM_MINIMAL_FIXED_DEADLINE_P_EQUALS_NP_OBLIGATION.md
```

Executable audits already present:

```text
exact_product_reduction_audit.py
phase_energy_monotonicity_audit.py
phase_stationary_point_audit.py
memory_logit_drift_audit.py
stationary_memory_strata_audit.py
reduced_phase_resource_bound.py
```

Do not duplicate these results.

## Exact current facts

### 1. Arbitrary-state convergence is false

The satisfiable relation

```text
(x or y or y) and (~x or ~y or ~y)
```

has a non-solution stationary carrier state at:

```text
all phase cosines = 0
all phase sines = 1
all short memories = 1
all long memories = L
all clause selectors uniform.
```

The declared public seed is not stationary there. The theorem must be specific to the
public seed.

### 2. Raw clause energy is not Lyapunov

The wrong-corner release term can increase the sum of exact clause-product violations.
Do not attempt to prove convergence using raw clause energy alone.

### 3. Exact memory clock

For each clause, with normalized long memory `r=l/L` and logits

```text
u = log(s/(1-s))
v = log(r/(1-r)),
```

the exact identity is

```text
d/dt(v/alpha-u/beta) = gamma-delta = 1/5.
```

For the declared public seed and frozen parameters:

```text
odds(r(t)) = e^t/(L-1) * odds(s(t))^(1/4).
```

Consequences already established:

```text
finite-time memory boundary contact is impossible
bounded recurrence in the full interior memory chart is impossible
long memory is reconstructible from short memory and elapsed time
```

### 4. Five forward-compatible stationary memory strata remain

The public-seed memory relation excludes three of the eight algebraic stationary types.
The remaining local types are:

```text
SHORT_ZERO__LONG_ZERO__C_ARBITRARY
SHORT_ZERO__LONG_CAP__C_ARBITRARY
SHORT_ONE__LONG_CAP__C_ARBITRARY
SHORT_INTERIOR_C_EQ_GAMMA__LONG_CAP
SHORT_ZERO__LONG_INTERIOR_C_EQ_DELTA
```

These are local memory classes only. A global invariant set must also satisfy the phase,
selector, and cross-clause consistency equations.

## Exact mathematical starting point

For one exact three-literal clause, with literal signs `q_r in {-1,+1}` and phase
cosines `c_i`:

```text
d_r = 1-q_r c_i
C_j = truth_gain * d_1 d_2 d_3 / 8.
```

The exact coordinate direction used by the phase field is:

```text
-partial C_j/partial c_i
= truth_gain * q_i * product(other two defects) / 8,
```

including repeated-variable occurrences through the ordinary sum of occurrence
contributions.

For the phase pair `(c_i,z_i)` on `S1`:

```text
c_i_dot = -z_i omega_i
z_i_dot =  c_i omega_i.
```

Therefore the clause Lie derivative is:

```text
C_j_dot = sum_i (partial C_j/partial c_i) c_i_dot.
```

For memory-boundary normal coordinates, the first-order exponents are:

```text
short at 0:       lambda_s0 =  beta (C-gamma)
short at 1:       lambda_s1 = -beta (C-gamma)
long at 0:        lambda_l0 =  alpha (C-delta)
long at L:        lambda_lL = -alpha (C-delta).
```

For an interior-memory stratum, `C=gamma` or `C=delta` is not enough by itself. An
invariant set must also satisfy the tangency chain:

```text
C_dot = 0
and, where required, higher Lie derivatives remain zero.
```

Clause-selector stationary/tangency conditions must be included. For an interior
three-state selector simplex, all three supported literal-defect costs must agree. On a
selector face, every supported cost must equal the supported weighted cost and excluded
coordinates must satisfy the appropriate normal stability condition.

## Work package A: Generic boundary-stratum audit

Create:

```text
boundary_stratum_escape_audit.py
```

The audit must represent strata symbolically and compositionally. Do not enumerate all
`5^m` global combinations as the mechanism.

At minimum provide:

1. a typed descriptor for the five forward-compatible local memory classes;
2. exact memory normal exponents;
3. exact clause violation and `C_dot` evaluation;
4. selector support/stationarity conditions;
5. phase-field residuals on a proposed stratum state;
6. a classification of a proposed state as:

```text
NOT_INVARIANT
INVARIANT_REPELLING_IN_AT_LEAST_ONE_PUBLIC_DIRECTION
INVARIANT_STABLE_OR_CENTER_UNRESOLVED
NON_SOLUTION_ATTRACTOR_CANDIDATE
SATISFYING_INVARIANT_SECTION
```

7. explicit separation between a stationary point, a moving invariant set, and an
   asymptotic numerical approach.

The audit must fail closed on dimension mismatch, NaN/Inf, unsupported stratum labels,
or incomplete selector support.

## Work package B: Exact small-formula search

Create a capped reference search over semantic-unique small 3-CNF objects. This is an
audit and counterexample search, not the native mechanism.

Recommended initial cap:

```text
n <= 4
```

For each satisfiable formula:

1. generate the declared public seed;
2. derive stationary/invariant equations on each of the five local memory classes;
3. search exact rational/algebraic solutions where possible;
4. use high-precision numerical root finding only as a candidate generator;
5. certify candidate residuals independently;
6. compute normal eigenvalues or a rigorously bounded local linearization;
7. determine whether the public seed can lie in or approach the candidate stable set.

Seal every discovered counterexample candidate with:

```text
DIMACS/public record
semantic digest
presentation digest
exact stratum labels
state coordinates
field residual
Lie-derivative residuals
Jacobian spectrum or interval bounds
independent SAT witness/reference count
solver and precision metadata
```

Do not call a tiny numerical residual an exact invariant without a separate certificate.

## Work package C: Presentation-gauge custody

`ConstraintHolo` sorts variable names, but:

```text
renaming can change the sorted variable order
clause order is preserved
literal order is preserved in the stored clause
```

The public seed is indexed by the resulting variable order. Do not assume presentation
canonicalization has removed the gauge problem.

For every small exact control, test:

```text
all feasible variable renamings/permutations
reverse and sampled clause permutations
literal permutations within clauses
semantic-equivalent duplicate-clause controls
```

The theorem currently quantifies over every declared presentation gauge. Do not weaken
that statement silently.

A revised canonical seed/compiler is allowed only as a separate candidate if a concrete
gauge counterexample is first established. Any changed seed law must rerun the complete
existing phase census, structured controls, 128-case near-threshold campaign, seed 38,
seed 86, and planted cases.

## Work package D: Public-seed reachability campaign

Numerically integrate only after the structural audit exists.

Required controls:

```text
known symmetric non-solution stationary formula
seed 38 unique-witness separator
seed 86 near-deadline unique-witness case
complete three-variable census
structured parity/pigeonhole/coloring controls
near-threshold deterministic corpus
```

For candidate hard cases, use at least two independent numerical charts or solver
families when feasible, for example:

```text
DOP853 explicit chart
Radau or BDF implicit chart
higher precision or reduced direct chart cross-check
```

Record:

```text
minimum distance to each candidate stratum
phase and memory field residuals
C, C_dot, and higher observed Lie derivatives
normal exponents
first-passage time
terminal margin
native path lower bound
memory/logit range
solver disagreement
```

A solver exception is `INVALID_CARRIER`, never UNSAT.

## Work package E: Augmented functional search

The raw energy wall is already established. Search only functionals that incorporate the
memory clock and/or boundary geometry.

Candidate families may include:

```text
weighted clause energy with state-dependent memory weights
relative-entropy terms for short memory and clause selectors
barrier terms for the five remaining boundary classes
an explicitly time-dependent functional after eliminating long memory
a topological transit count rather than a scalar Lyapunov function
```

For every candidate functional:

1. derive its directional derivative symbolically;
2. search for positive-derivative counterexamples before promoting it;
3. test the known stationary obstruction;
4. distinguish global monotonicity, monotonicity outside declared neighborhoods, and
   average descent;
5. preserve the exact failure witness if the candidate is falsified.

Do not tune a functional only against the current finite corpus and narrate it as a
theorem.

## Decision rules

### If a public-seed counterexample is found

Do not immediately modify the native law.

First commit a sealed counterexample checkpoint with:

```text
PUBLIC_SEED_COMPLETENESS_FALSIFIED_FOR_CURRENT_LAW
P_EQUALS_NP_NOT_PROVEN
```

Include exact formula custody, independent SAT verification, solver cross-checks, and the
smallest reproducible state/trajectory witness. Then propose mechanism changes in a
separate explicitly labeled experiment.

### If all five strata are locally excluded

Do not claim global convergence yet. Record:

```text
LOCAL_NON_SOLUTION_BOUNDARY_STRATA_EXCLUDED
GLOBAL_PUBLIC_SEED_COMPLETENESS_NOT_ESTABLISHED
```

The next obligations would still be:

```text
compactness/omega-limit closure
formula-uniform neighborhood escape
polynomial escape time
robust terminal margin
```

### If a polynomial escape theorem is derived

The proof must state the polynomial explicitly and bind every constant to public
parameters. Then connect it to:

```text
THEOREM_REDUCED_PHASE_FORWARD_RESOURCE_TRANSFER.md
THEOREM_MINIMAL_FIXED_DEADLINE_P_EQUALS_NP_OBLIGATION.md
```

No `P = NP` promotion is allowed until deterministic polynomial simulation and robust
terminal verification are mechanically bound.

## Required output surface

Expected new or updated artifacts:

```text
boundary_stratum_escape_audit.py
tests/constraint_relational_trace/test_boundary_stratum_escape_audit.py
THEOREM_PUBLIC_SEED_BOUNDARY_STRATUM_ESCAPE.md
results/boundary_stratum_escape_campaign.json        if a deterministic campaign exists
run_mechanism_campaign.py                            add only mechanically justified gates
CONTRACT.md
REPORT.md
CHANGELOG.md
```

Keep generated evidence deterministic and small. Do not commit large solver traces.

## Qualification before each pushed checkpoint

At minimum:

```bash
$PY -m pytest THOUGHT/LAB/CAT_CAS/tests/constraint_relational_trace/test_boundary_stratum_escape_audit.py -q
$PY -m pytest THOUGHT/LAB/CAT_CAS/tests/constraint_relational_trace -q
$PY THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/constraint_relational_trace_v1/run_reference_campaign.py
$PY THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/constraint_relational_trace_v1/run_mechanism_campaign.py
$PY CAPABILITY/TOOLS/governance/critic.py
git diff --check
git status --short --branch
```

If one full command exceeds the agent transport window, split the suite into deterministic
partitions and preserve every result. Do not reinterpret a tool timeout as a test failure
or success.

## Commit and PR behavior

- Commit and push each coherent green theorem/counterexample checkpoint automatically.
- Keep PR #49 draft.
- Update the PR description when the exact head and theorem boundary change.
- Add a concise PR comment for each major mathematical checkpoint.
- Never merge, retarget, or mix PR #50 successor work into this branch.
- Do not wait for permission to commit after the user has directed continuation.

## Completion condition for this task

This agent task is complete only when one of the following is mechanically established:

```text
A. a sealed public-seed counterexample for the current exact-product law;
B. a certified exclusion theorem for all five forward-compatible non-solution
   boundary-stratum classes, with the remaining global obligations stated exactly;
C. a stronger formula-uniform polynomial escape theorem with robust terminal margin.
```

Anything weaker is progress evidence, not completion of PR #49 or a proof of `P = NP`.

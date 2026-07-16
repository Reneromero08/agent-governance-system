# Recursive Catalytic Ising Package

**Status:** `SOURCE_CANDIDATE__EXACT_PARENT_AND_R3_QUALIFICATION_PENDING`  
**Package:** `audio_recursive_catalytic_ising_v1`  
**Root directive:** `REPLACE THE BIT WITH PI`  
**Parent results:** `AUDIO_RECURSIVE_PHASE_TREE_REFERENCE_ESTABLISHED`; `AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED`; `AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED`  
**Source commit:** `aba7dfc4030728f25db00b6f204b2575688afe7a`  
**Source Git blob SHA-1:** `a73d41b8c70b022d7f14d345056e46afaf8b6f9a`  
**Source SHA-256:** must be computed and frozen from exact committed bytes during qualification  
**Operation:** bounded ordinary-software recursive phase-native Ising emulation  
**Physical authority:** none

## 1. Purpose

This package tests the first bounded Ising sector of the phase-native architecture.

The primitive state at site `i` is not a bit. It is:

```text
RecursivePhaseSite_i {
    complete recursive phase tree T_i
    continuous global orientation theta_i in S1
}
```

The rendered site is:

```text
Psi_i(t) = exp(i * theta_i) * B_Ti(t)
```

The complete internal tree remains unchanged while `theta_i` evolves continuously. Only
after native evolution ends may the explicit collapse boundary project:

```text
theta_i near 0   -> s_i = +1
theta_i near pi  -> s_i = -1
```

Thus the bit is the antipodal shadow `{0, pi} subset S1`, not the machine atom.

## 2. Exact Relationship to the Catalytic Predecessor

R2S is an established prerequisite and must reproduce before R3 qualification. R3 does
not smuggle the R2S complex latch back into the phase recurrence and does not claim that
each Ising integration step is itself a carrier-restored catalytic cycle.

The bounded R3 claim means:

```text
the established recursive and catalytic software stack admits a phase-native Ising
sector over complete recursive beams, with the catalytic predecessor preserved and
reproduced as separate evidence
```

It does not mean:

```text
the Ising optimum was obtained by repeatedly decoding and feeding R2S latch scalars;
the physical carrier performed the Ising evolution;
or the Ising evolution itself has a separately established restoration law
```

Any stronger direct integration of catalytic restoration into every Ising step is a later
rung and requires a new prospectively frozen contract.

## 3. Frozen Native State

The reference has five sites. Each site contains the established R0 hierarchy-A recursive
tree under collision-free site-prefixed node identities. The tree geometry, frequencies,
local phases, modulation indices, depth, and canonical bytes remain invariant through
native evolution.

Only the external orientation coordinate changes:

```text
theta = (theta_0, ..., theta_4) in T^5 = (S1)^5
```

Frozen initial orientations:

```text
[0.31, 1.27, -2.11, 2.53, -0.83] radians
```

The initial state is intentionally outside the antipodal `{0, pi}` sector.

## 4. Frozen Phase Dynamics

The continuous phase velocity is:

```text
d theta_i / dt =
    sum_(j != i) J_ij * sin(theta_j - theta_i)
    - h_i * sin(theta_i)
    - lambda_k * sin(2 * theta_i)
```

The terms have distinct roles:

```text
pairwise phase-difference term    continuous relational Ising coupling
field term                        continuous orientation bias
second-harmonic term              antipodal 0/pi phase locking
```

The second-harmonic term creates the `Z2` sector without decoding a spin during evolution.

Frozen integration law:

```text
method             synchronous explicit Euler
steps              1000
time step           0.03
lambda start        0.0
lambda final        1.2
lambda schedule     linear in step index
phase wrapping      [-pi, pi)
```

The native update may read only complete site objects, continuous orientations, the
prospectively committed `J/h` problem, and the lock schedule.

## 5. Frozen Ising Instance

Coupling matrix:

```text
J =
[[ 0,  0,  1,  2, -2],
 [ 0,  0,  2, -1, -1],
 [ 1,  2,  0,  2, -1],
 [ 2, -1,  2,  0, -2],
 [-2, -1, -1, -2,  0]]
```

Field vector:

```text
h = [-0.5, 0.0, 0.5, -1.0, -0.5]
```

The matrix must remain exactly symmetric with zero diagonal.

At the final boundary only, the discrete Ising energy is:

```text
E(s) = -0.5 * s^T J s - h^T s
```

The source-authoring model found the unique bounded optimum:

```text
s*             [-1, -1, -1, -1, +1]
E(s*)          -12.5
next energy    -11.5
optimum gap     1.0
```

These values are candidate diagnostics. Qualification must independently enumerate all
`2^5 = 32` boundary states from the committed problem after native evolution completes.
The optimum may not enter the initial state, phase update, lock schedule, stopping rule,
or candidate selection.

## 6. Native Non-Collapse Law

Forbidden from the AST-reachable native call graph rooted at:

```text
native_phase_velocity
native_phase_step
evolve_phase_state
```

are:

```text
decode_spins
collapse_boundary
exact_ising_oracle
Ising energy
sign or threshold projection
argmin or argmax
winner/candidate selection
expected optimum
verification result
R2S latch response
```

`J` acts through `sin(theta_j - theta_i)` over continuous phase coordinates. A native
implementation that instead decodes `s`, evaluates `J @ s`, and resynthesizes oriented
waves is a collapsed baseline and cannot establish this package.

The complete recursive tree under every orientation must remain byte-identical throughout
native evolution.

## 7. Explicit Collapse Boundary

Only after all 1000 continuous steps return may the package compute:

```text
s_i = +1 if cos(theta_i) >= 0 else -1
```

and then evaluate energy and the independently enumerated oracle.

Required final lock gate:

```text
max_i distance(theta_i, {0, pi}) <= 1e-8
```

Required exact bounded result:

```text
projected spin vector equals the unique enumerated optimum
observed energy equals -12.5 within 1e-12
optimum gap is at least 1.0
```

This establishes one deterministic bounded reference instance only. It does not establish
a general-purpose optimizer, success probability across arbitrary instances, complexity
advantage, or scaling.

## 8. Whole-Tree Pi Action

For every site:

```text
Psi_i(theta_i + pi, t) = -Psi_i(theta_i, t)
```

while internal magnitude and relative recursive phase geometry remain unchanged.

Required tolerance:

```text
whole-beam negation error <= 1e-12
amplitude-preservation error <= 1e-12
internal tree canonical bytes unchanged exactly
```

The final spin is therefore the global orientation of a complete recursive phase object,
not a replacement flat sine wave.

## 9. Required Controls

The bounded package must include at minimum:

```text
no second-harmonic lock
negated coupling matrix
reversed/scaled field
intermediate spin decode baseline
J@s-and-resynthesis collapsed baseline
mutated initial orientation
site permutation with covariantly permuted J/h
asymmetric J rejection
nonzero diagonal rejection
nonfinite J/h rejection
step-count mutation
lock-schedule mutation
late oracle access
expected-optimum field injection
internal tree mutation
site identity collision
final state outside lock tolerance
```

Prospective candidate gates:

```text
no-lock final antipodal residual >= 1e-3
wrong J result differs from the frozen optimum and has higher frozen energy
wrong field result differs from the frozen optimum and has higher frozen energy
```

Controls must fail for their declared mechanical reason.

## 10. Candidate Source Diagnostics

An API-compatible local R0 model of the Git-bound source produced:

```text
14 PASS / 0 FAIL
initial antipodal residual         1.27
final antipodal residual           approximately 7.1e-15
no-lock residual                   approximately 0.3661
final orientations                 [pi, pi, pi, pi, approximately 0]
projected spins                    [-1, -1, -1, -1, +1]
observed energy                    -12.5
optimum gap                         1.0
wrong-coupling frozen energy        5.5
wrong-field frozen energy          -11.5
```

These are source-authoring diagnostics only. They are not committed-byte results and do
not establish the R3 token.

## 11. Required Qualification Packet

Before promotion, add strict machine-readable custody for:

```text
R3 phase-Ising contract
J/h problem
initial phase state
site recursive-tree identities
lock schedule
sampled or complete phase trajectory
final phase state
collapse receipt
exact 32-state oracle table
manifest
reference tests
reference results
```

Recommended compact trajectory format:

```text
little-endian float64
shape [1001, 5]
initial state plus every post-step state
no metadata
```

The manifest must bind source and all R0/R1/R2S parent identities, exact `J/h`, initial
orientations, integration schedule, five tree identities, trajectory bytes, final phases,
collapse receipt, oracle table, tests, results, and claim ceiling.

Qualification must recompute from committed bytes, not merely regenerate in memory.

## 12. Parent Reproduction

Before trusting R3, independently run and verify:

```text
R0: 38/38 expected established reference
R1: 78/78 expected established reference
R2S: 78/78 expected established reference
```

Bind at minimum the committed parent source, manifest, fixture-set, test, and result hashes.
The R2S parent result expected by the current branch is:

```text
source SHA-256       6c55861da950caf0738bb5ffb676f0c458a593a805ddd49419d6b2b427f6c33c
fixture manifest     5e8bfa247c513d189774ec671265b2d3dc1ea97004e5e8c40baa090f26db3cad
fixture set          e6e51ae655e184f8f43b2afa9fe0c75041046966b4cdecd6fde008b02b684aa8
reference tests      ef888d8d8b48b2fbdc7897d6d42aa2f63f8c300517f6d9b8911346bf285438c6
reference result     bee5727f68fc10ee047d666198b3f060f669058e966aa44802e270f90abbdeeb
```

Do not modify predecessor packages.

## 13. Reviews

Four independent read-only reviews are required:

```text
AUD-RCI-01-PHASE-MECHANISM
    continuous S1 flow, second-harmonic lock, complete-tree preservation

AUD-RCI-02-NONCOLLAPSE
    native call graph, boundary placement, collapsed baselines

AUD-RCI-03-CUSTODY-ORACLE
    J/h, schedule, trajectory, fixtures, exact enumeration isolation

AUD-RCI-04-CLAIMS
    bounded emulator ceiling, catalytic-parent meaning, physical boundaries
```

Every material finding must be repaired and affected gates rerun before adjudication.

## 14. Claim Law

After exact parent reproduction, source and fixture custody, continuous-trajectory replay,
complete-tree preservation, antipodal-lock closure, isolated exact oracle agreement,
negative controls, AST non-collapse proof, four PASS reviews, zero open material findings,
and branch qualification, the package may emit only:

```text
AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED
```

Meaning only:

```text
an ordinary-software five-site reference carries complete recursive phase trees under
continuous global S1 orientations; pairwise phase-difference and field dynamics plus a
second-harmonic term enter a 0/pi sector without intermediate spin decoding; the complete
trees remain unchanged; and an explicit final projection matches the unique optimum of
one prospectively frozen bounded Ising instance, while the established catalytic wave-loop
parent remains independently reproduced
```

Claim ceiling:

```text
SOFTWARE_RECURSIVE_PHASE_ISING_EMULATOR_ONLY
```

Forbidden promotion:

```text
no general Ising solver claim
no optimization, speed, energy, or complexity advantage
no claim that R2S latch values drive the Ising recurrence
no physical audio oscillator computation
no silicon-phononic computation
no physical carrier persistence or restoration
no hardware bit replacement
no Small Wall or Big Wall crossing
```

## 15. Failure Classes

```text
BLOCKED:
    the complete recursive trees cannot survive native phase coupling;
    native evolution requires intermediate spin/energy/oracle feedback;
    the frozen instance does not enter the antipodal sector;
    the final projection misses the prospectively frozen bounded optimum;
    or collapsed controls are indistinguishable from the native mechanism under the
    declared structural tests

INCONCLUSIVE:
    the mechanism remains coherent but exact parent/source custody, trajectory fixtures,
    oracle isolation, adversaries, reviews, or repository qualification are incomplete
```

## 16. Current State

```text
AUDIO_FM_WAVE_ALGEBRA_ESTABLISHED
AUDIO_RECURSIVE_PHASE_TREE_REFERENCE_ESTABLISHED
AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED
AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED
AUDIO_RECURSIVE_CATALYTIC_ISING_SOURCE_CANDIDATE
AUDIO_RECURSIVE_CATALYTIC_ISING_NOT_YET_ESTABLISHED
PHYSICAL_AUDIO_COMPUTING_NOT_ESTABLISHED
PHYSICAL_SILICON_PHONONIC_COMPUTING_NOT_ESTABLISHED
HARDWARE_BIT_REPLACEMENT_NOT_ESTABLISHED
SMALL_WALL_CROSSED_NOT_PROMOTED
```

Contact counts:

```text
audio playback       0
audio recording      0
hardware contact     0
target contact       0
SSH/SCP              0
```

Next exact boundary:

```text
EXACT_R0_R1_R2S_REPRODUCTION_AND_R3_PHASE_TRAJECTORY_QUALIFICATION
```

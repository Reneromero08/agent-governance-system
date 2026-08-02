# Family 10h Protocol Adversarial Audit

## Disposition

```text
PROSPECTIVE_PROTOCOL_DIRECTION_COHERENT
TRANSPORT_OPERATOR_NOT_YET_PHYSICALLY_DEFINED
ACCUMULATOR_COUPLING_NOT_YET_DEFINED
BOUNDED_ADVERSARY_ONLY
PROTOCOL_REPAIR_REQUIRED_BEFORE_FREEZE
NO_LIVE_EXECUTION_AUTHORIZED
```

The commutator direction is a legitimate next calibration. The current protocol must be
sharpened before it can become a frozen experiment.

## 1. The receiver must execute the loop after source death

A post-source loop-word choice is not enough if the source already executed or encoded
the transformations. The causal sequence must be:

```text
source prepares one public carrier baseline
source exits
source capabilities and descriptors close
receiver draws a held-out word
receiver applies every generator in that word
receiver reads the independent accumulator and carrier tomography
receiver applies the declared inverse sequence where required
receiver performs R2 tomography
```

`T_A`, `T_B`, and their candidate inverses must therefore be receiver-side operations on
the surviving carrier. Otherwise the source can still prepare one response per possible
word.

## 2. `H` must be an operator, not an unexplained scalar

A scalar PMU count after one sequence is not a holonomy. The protocol must first define
a frozen carrier coordinate vector:

\[
s\in E\subseteq\mathbb R^d
\]

and identify each generator as an empirical transport map:

\[
T_g:E\to E.
\]

The word transport is:

\[
T_\gamma=T_{g_L}\circ\cdots\circ T_{g_1}.
\]

A scalar readout may then be a declared gauge-invariant function:

\[
C(T_\gamma),
\]

but it cannot replace identification of the transported state or operator law.

Minimum acceptable operator evidence includes held-out prediction of:

```text
state after A
state after B
state after A followed by B
state after B followed by A
state after candidate inverse
state after composed held-out words
```

## 3. Separate carrier state from custody metadata

The previous vector mixed physical response coordinates with temperature, mapping,
policy, and custody fields. These have different roles.

Freeze:

\[
S_{carrier}=(S_{PMU},S_{probe},S_{timing},S_{other\ observed\ response})
\]

and separately:

```text
C_custody = mapping, policy, temperature, CPU, process, schedule, identity
```

R2 equivalence applies to `S_carrier`. Custody fields gate validity and define matched
strata. They are not silently counted as restored carrier degrees of freedom.

## 4. Match operation multisets within valid strata

The four-operation base stratum is:

```text
A A^-1 B B^-1
A B B^-1 A^-1
A B A^-1 B^-1
B A B^-1 A^-1
```

These words share one copy of each operation.

The eight-operation composition stratum is:

```text
L_comm L_comm
L_comm L_rev
L_rev L_comm
L_rev L_rev
matched doubled null words
```

Do not state that a four-operation word and an eight-operation word have the same
operation multiset. Composition and cancellation must be tested inside the doubled
stratum or through a model that explicitly controls word length.

## 5. Candidate inverses need empirical qualification

A generator named `A_inverse` is only a reversal candidate. Before a commutator claim,
require:

\[
T_{A^{-1}}T_A\approx I,
\qquad
T_AT_{A^{-1}}\approx I
\]

and the same for `B`, across held-out starting carrier states, mappings, and delays.
One-sided return is insufficient.

Wrong-order and wrong-amplitude controls must fail under the same equivalence law.

## 6. The accumulator must couple causally to transport

A normal memory register that records the public word or locally computes a formula is
not a holonomy accumulator.

The accumulator contract must state:

```text
initial state
allowed interaction with the carrier
forbidden direct access to source condition and expected class
readout observable
inverse behavior
carrier-off null
reference-only null
```

The accumulator must change because of carrier transport. It must not merely mirror the
receiver control sequence.

## 7. Finite experiments exclude only bounded replay classes

No finite challenge campaign can exclude an arbitrary lookup table or unrestricted
algorithmic predictor.

The protocol may claim exclusion only against a prospectively frozen class such as:

```text
maximum encoded bits
maximum predictor state dimension
maximum polynomial degree
maximum automaton states
allowed public side information
allowed route and timing features
training cross-product
```

Held-out group-law generalization is strong evidence against that bounded class. It is
not an information-theoretic rejection of every classical explanation.

The promotion token must therefore say `BOUNDED_REPLAY_CLASS_REJECTED`, not
`ANSWER_CACHE_EXCLUDED` without qualification.

## 8. Area scaling is necessary but not unique

For small controls, a commutator response proportional to `alpha*beta` is consistent
with curvature through the Baker-Campbell-Hausdorff expansion. Ordinary nonlinear and
history-dependent classical systems can also exhibit such scaling.

Area scaling must be combined with:

```text
inverse orientation
composition
cancellation
contractible nulls
commuting-pair nulls
held-out starting states
causal accumulator intervention
R2 restoration
```

Even a passing result establishes a connection-like carrier law, not the Native
Catalytic Fiber Pushforward.

## 9. Restoration must be state-wide and prospective

Before acquisition, freeze:

```text
baseline distribution
carrier coordinates
replicate hierarchy
equivalence metric
equivalence margin
measurement disturbance ceiling
natural-relaxation window
inverse deadline
```

R2 requires return under the declared inverse, not eventual overlap after unbounded
relaxation. Both fresh replicates must pass. Aggregate rescue is forbidden.

## 10. The bridge to higher pushforward remains a separate theorem

Pair commutator curvature tests whether local path composition is nontrivial. The
weighted fiber pushforward is a global aggregation over an unresolved product fiber.

The missing bridge theorem must define how local transport cells compose into a
higher-cycle invariant with the formula weight `W_F`. It must also specify how the
independent accumulator receives the modular residue without enumerating cells or
sheets.

No H0 through H5 result automatically establishes H6.

## Required repairs

1. Freeze a receiver-visible carrier vector and empirical generator maps.
2. Require all word operations to occur receiver-side after source death.
3. Split base and doubled operation-multiset strata.
4. Qualify both left and right inverse laws on held-out starting states.
5. Define a causal accumulator interaction, not just a storage option.
6. Separate R2 carrier coordinates from custody metadata.
7. Freeze a bounded replay adversary and narrow the exclusion claim.
8. Define exclusive fail-closed result classes for each rung.
9. Keep pair holonomy and higher fiber pushforward as separate claims.

## Current claim ceiling

```text
FAMILY10H_HOLONOMY_PROTOCOL_PROSPECTIVE_ONLY
FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED
NO_LIVE_AUTHORITY
```

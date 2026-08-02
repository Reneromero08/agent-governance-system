# Prospective Family 10h Holonomy Protocol

## 1. Scope and authority

This is a non-executing protocol design.

```text
NO_TARGET_CONTACT
NO_SSH_OR_SCP
NO_PMU_ACQUISITION
NO_SENSOR_INVENTORY
NO_SYSFS_WRITE
NO_LIVE_AUTHORITY
NO_RETIRED_PACKAGE_REACTIVATION
```

It does not modify or authorize the current Family 10h tomography package. The
tomography package remains the precursor that must first identify reproducible
post-source carrier coordinates and candidate order-dependent operators.

Current protocol status:

```text
PROSPECTIVE_ONLY
TRANSPORT_OPERATOR_NOT_ESTABLISHED
ACCUMULATOR_COUPLING_NOT_ESTABLISHED
BOUNDED_REPLAY_CLASS_NOT_FROZEN
```

## 2. Experimental objective

Determine whether the Family 10h carrier supports a path-relational transformation
whose closed-loop response cannot be reduced to endpoint state, operation counts,
ordinary nonlinearity, or a source-authored bounded replay mechanism.

The first target is a commutator:

\[
K_{AB}=T_AT_BT_A^{-1}T_B^{-1}.
\]

The goal is not to call every order effect holonomy. The goal is to identify a
receiver-side transport law with inverse, composition, area scaling, a causally coupled
accumulator, and R2 restoration.

## 3. Frozen causal sequence

Every future transaction must use this order:

```text
public carrier preparation selected
source prepares one carrier baseline
source exits
parent verifies source death
source IPC, descriptors, and capabilities close
receiver draws one held-out loop word
receiver applies every generator in the word
receiver measures carrier and accumulator
receiver applies the declared inverse sequence where required
receiver performs R2 tomography
feature packet freezes
analysis begins
```

The source must not know the realized word, word length, amplitudes, mapping, or delay
before closure. `T_A`, `T_B`, and every inverse candidate are receiver-side operations on
the surviving carrier.

## 4. Gate H0: carrier tomography prerequisite

Before any holonomy package can be frozen, tomography must identify:

- a receiver-visible carrier vector `s`;
- at least two controlled receiver-side transformations `T_A` and `T_B`;
- bounded persistence after source death;
- repeatable response under held-out replicate, mapping, and delay;
- prospective disturbance ceilings;
- candidate inverse or reversal controls.

No tomography result alone establishes relational memory or a Small Wall crossing.

## 5. Gate H1: carrier state and operator grammar

Freeze a carrier coordinate space:

\[
s\in E\subseteq\mathbb R^d.
\]

The carrier vector may include:

```text
PMU response coordinates
standardized probe coordinates
timing-response coordinates
other prospectively frozen receiver observables
```

Keep custody metadata separate:

```text
mapping
policy
temperature
CPU identity
process identity
schedule identity
address-layout identity
```

Custody metadata defines valid matched strata. It is not silently counted as restored
carrier state.

Each generator must bind:

```text
operator id
public amplitude
page and line population
bank and route identity
source core and receiver core
address permutation
instruction sequence
operation count
start and end barrier
delay
temperature
process custody
policy custody
```

The generator is an empirical map:

\[
T_g:E\to E.
\]

A scalar PMU count after a word is not itself a holonomy. Any scalar result must be a
predeclared function of the identified transport and accumulator response.

## 6. Gate H2: inverse qualification

An operation named `A_inverse` is only a candidate. Before commutator testing, require
both left and right inverse laws:

\[
T_{A^{-1}}T_A\approx I,
\qquad
T_AT_{A^{-1}}\approx I,
\]

\[
T_{B^{-1}}T_B\approx I,
\qquad
T_BT_{B^{-1}}\approx I.
\]

These laws must survive held-out:

```text
starting carrier states
replicates
mappings
delays
public amplitudes
```

Wrong amplitude, wrong inverse, and wrong inverse order must fail under the same frozen
equivalence law.

## 7. Gate H3: matched loop strata

### Four-operation base stratum

```text
L_null_1 = A A^-1 B B^-1
L_null_2 = A B B^-1 A^-1
L_comm   = A B A^-1 B^-1
L_rev    = B A B^-1 A^-1
```

Every base word contains exactly one copy of `A`, `B`, `A^-1`, and `B^-1`.

### Eight-operation composition stratum

```text
L_double       = L_comm L_comm
L_cancel       = L_comm L_rev
L_cancel_rev   = L_rev L_comm
L_reverse_pair = L_rev L_rev
matched doubled null words
```

Compare operation multisets within one stratum. A four-operation word and an
eight-operation word are not described as having the same multiset.

Every matched arm must preserve:

```text
pages and line population
banks and routes
operator-position counts
source work
receiver work
timing envelope
measurement windows
starting-state stratum
```

## 8. Gate H4: transport and response laws

For a word:

\[
\gamma=g_L\cdots g_1,
\]

define:

\[
T_\gamma=T_{g_L}\circ\cdots\circ T_{g_1}.
\]

Required held-out prediction includes the state after each single generator, pair,
commutator, inverse commutator, doubled word, and cancellation word.

Required laws:

\[
T_{L_{rev}}\approx T_{L_{comm}}^{-1},
\]

\[
T_{L_{cancel}}\approx I,
\]

\[
T_{\gamma_2\gamma_1}\approx T_{\gamma_2}T_{\gamma_1}.
\]

For small public amplitudes `alpha` and `beta`, a connection-like response must satisfy
a prospectively frozen area law:

\[
T_{L_{comm}}-I
=
\mathcal F_{AB}\alpha\beta
+
o(\alpha\beta).
\]

Area scaling alone is insufficient. Ordinary nonlinear systems can produce it. The
candidate must also pass inverse orientation, composition, cancellation, contractible
nulls, commuting-pair nulls, held-out starting states, causal accumulator intervention,
and R2 restoration.

## 9. Gate H5: independent causal accumulator

An endpoint-only measurement of an arbitrary restored carrier cannot reveal a
nontrivial loop invariant. The protocol therefore requires an independent comparison
channel.

The accumulator contract must freeze:

```text
initial state
allowed carrier interaction
forbidden direct access to source class and expected result
readout observable
inverse behavior
carrier-off null
reference-only null
word-only replay null
```

Candidate implementations may include a differential receiver path, phase-referenced
counter pair, standardized probe register, or separately initialized public memory.
A normal register that merely records the public word or computes a local formula is not
an admissible accumulator.

The accumulator must change because of carrier transport. A causal intervention that
removes or scrambles the carrier coupling must destroy the claimed response.

## 10. Gate H6: Restoration R2 contract

Freeze separately:

\[
S_{carrier}=(S_{PMU},S_{probe},S_{timing},S_{other}),
\]

and:

```text
C_custody = mapping, policy, temperature, CPU, process, schedule, identity
```

Measure:

```text
S_baseline
S_after_forward_loop
S_after_inverse
S_natural_relaxation
S_destructive_reset
S_wrong_inverse
S_wrong_order_inverse
```

R2 requires:

\[
S_{after\ inverse}\sim S_{baseline}
\]

under a prospectively frozen multivariate equivalence region and inverse deadline.
Both fresh replicates must pass. Aggregate rescue is forbidden.

The same gate must reject:

- wrong inverse;
- correct operators in wrong inverse order;
- natural relaxation mislabeled as inverse restoration;
- destructive reset;
- carrier-off;
- missing accumulator retention.

The output invariant must remain in the independent accumulator after carrier
restoration.

## 11. Gate H7: bounded replay exclusion

No finite campaign can exclude an arbitrary table or unrestricted algorithmic
predictor. Freeze a bounded adversary class before acquisition, including:

```text
maximum encoded bits
maximum predictor state dimension
maximum automaton states
maximum polynomial degree
allowed public side information
allowed route and timing features
training cross-product
```

The receiver must choose post-source held-out:

- words;
- word lengths;
- inverses;
- compositions;
- powers;
- commutators;
- amplitudes;
- mappings and delays.

A positive candidate must predict unseen words through one frozen composition law.
Preparation capacity alone does not exclude compressed replay. Required evidence is:

1. an independently measured capacity bound;
2. held-out group-law generalization;
3. causal operator and accumulator interventions.

The strongest allowed exclusion token is:

```text
BOUNDED_REPLAY_CLASS_REJECTED
```

not an unrestricted claim that every classical explanation was excluded.

## 12. Relation mutation

The strongest matched mutation changes only ordered incidence:

```text
same operator multiset within stratum
same public amplitudes
same pages and banks
same routes and cores
same timing envelope
same marginal occupancy
different closed word
```

The primary contrast is:

```text
contractible word
versus
commutator word
versus
inverse commutator word
```

## 13. Kill matrix

| Attack | Required killer |
|---|---|
| Static answer vector | Post-source unseen words plus held-out composition |
| Scalar endpoint memory | Same endpoint and same operation multiset within stratum |
| Route or bank artifact | Crossed mapping with exact route and bank equivalence |
| Ordinary nonlinear contention | Contractible and commuting-pair nulls |
| Generic order effect | Inverse, composition, cancellation, and area laws |
| Timing drift | Randomized interleaving and time-matched sham loops |
| Source preselection | Independent post-death word entropy and sealed source-death receipt |
| Label-derived orientation | Blinded operator-to-physical-lane assignment |
| Word recorder | Word-only replay null and accumulator-coupling intervention |
| Reset called restoration | Wrong-inverse, natural-relaxation, and destructive-reset controls |
| Accumulator leakage | Carrier-off and reference-only nulls |
| Endpoint-only fake | Independent accumulator required |
| Post-hoc threshold | Prospective equivalence region and response law |
| Unbounded replay class | Claim remains bounded to the frozen adversary |
| Classical loop simulator | Claim limited to calibration unless native carrier use is proved |

## 14. Promotion ladder

```text
H0: post-source carrier coordinates observed
H1: receiver-side generator maps identified
H2: left and right inverse laws survive
H3: composition model predicts held-out words
H4: nontrivial commutator response survives matched nulls
H5: causal accumulator retains the response
H6: R2 carrier restoration passes
H7: frozen bounded replay class is rejected
H8: separate higher-cycle theorem links transport to weighted fiber pushforward
```

Only H6 or above is relevant to catalytic holonomy. No rung in this document is already
passed. H8 is a separate theorem and does not follow automatically from pair
commutators.

## 15. Exclusive result classes

```text
FAMILY10H_HOLONOMY_CUSTODY_INVALID
FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED
FAMILY10H_ORDER_DEPENDENCE_OBSERVED
FAMILY10H_CONNECTION_LAW_CANDIDATE
FAMILY10H_CATALYTIC_HOLONOMY_CANDIDATE
```

A result must emit exactly one class. Missing custody or malformed evidence takes
precedence and yields `FAMILY10H_HOLONOMY_CUSTODY_INVALID`.

Forbidden from this package:

```text
NATIVE_CATALYTIC_FIBER_PUSHFORWARD_ESTABLISHED
RELATIONAL_CARRIER_ESTABLISHED
SMALL_WALL_CROSSED
P_EQUALS_NP_PROVEN
```

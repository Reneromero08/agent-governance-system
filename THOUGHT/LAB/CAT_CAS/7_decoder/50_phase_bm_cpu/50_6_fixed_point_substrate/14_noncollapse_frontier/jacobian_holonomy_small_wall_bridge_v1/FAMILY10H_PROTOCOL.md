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

## 2. Experimental objective

Determine whether the Family 10h carrier supports a path-relational transformation
whose closed-loop response cannot be reduced to endpoint state, operation counts,
ordinary nonlinearity, or a source-authored answer cache.

The first target is a commutator:

\[
K_{AB}=T_AT_BT_A^{-1}T_B^{-1}.
\]

The goal is not to call every order effect holonomy. The goal is to establish a
connection-like transport law with inverse, composition, area scaling, and R2
restoration.

## 3. Gate H0: carrier tomography prerequisite

Before any holonomy package can be frozen, tomography must identify:

- a receiver-visible state vector `S`;
- at least two controlled transformations `T_A` and `T_B`;
- bounded persistence after source death;
- repeatable response under held-out replicate, mapping, and delay;
- prospective disturbance ceilings;
- candidate inverse or reversal controls.

No tomography result alone establishes relational memory or a Small Wall crossing.

## 4. Gate H1: exact operator grammar

Every operation must bind:

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

Candidate inverse operations must be independently characterized. A label such as
`A_inverse` does not establish inversion.

## 5. Gate H2: matched loop family

The minimum loop family is:

```text
L_null_1 = A A^-1 B B^-1
L_null_2 = A B B^-1 A^-1
L_comm   = A B A^-1 B^-1
L_rev    = B A B^-1 A^-1
L_double = L_comm L_comm
L_cancel = L_comm L_rev
```

All arms must preserve the exact operation multiset. They must match pages, banks,
routes, order-position counts, timing envelope, source work, receiver work, and
measurement windows.

Required laws:

\[
H(L_{\mathrm{rev}})=H(L_{\mathrm{comm}})^{-1},
\]

\[
H(L_{\mathrm{cancel}})=I,
\]

\[
H(\gamma_2\gamma_1)=H(\gamma_2)H(\gamma_1).
\]

For small public operator amplitudes `alpha` and `beta`, a connection-like response must
also satisfy a prospectively frozen area law:

\[
H(L_{\mathrm{comm}})-I
=
\mathcal F_{AB}\alpha\beta
+
o(\alpha\beta).
\]

Ordinary order sensitivity without these laws remains an order artifact.

## 6. Gate H3: independent accumulator

The carrier cannot reveal a nontrivial closed-path invariant through an endpoint-only
readout while also returning to the same arbitrary state. The protocol therefore needs
an independent comparison channel.

Allowed accumulator candidates must be identified before acquisition, for example:

- a differential receiver path;
- a phase-referenced counter pair;
- a held-out standardized probe;
- a separately initialized public memory register.

The accumulator must not contain a source-authored response table.

## 7. Restoration R2 contract

Define a frozen public carrier-state vector:

\[
S=
(S_{\mathrm{PMU}},S_{\mathrm{probe}},S_{\mathrm{timing}},
S_{\mathrm{mapping}},S_{\mathrm{policy}},S_{\mathrm{temperature}}).
\]

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
S_{\mathrm{after\ inverse}}\sim S_{\mathrm{baseline}}
\]

under a prospectively frozen multivariate equivalence region.

The same gate must reject:

- wrong inverse;
- correct operators in wrong inverse order;
- natural relaxation mislabeled as inverse restoration;
- destructive reset;
- carrier-off;
- missing accumulator retention.

The output invariant must remain in the independent accumulator after carrier
restoration.

## 8. Answer-cache exclusion

A finite fixed loop family is answer-cache equivalent. The receiver must choose a loop
word after source death from a public generator family:

\[
\gamma=
g_{i_L}^{\epsilon_L}\cdots g_{i_1}^{\epsilon_1}.
\]

The campaign must include held-out:

- words;
- word lengths;
- inverses;
- compositions;
- powers;
- commutators;
- amplitudes;
- mappings and delays.

A positive candidate must predict unseen words through one frozen composition law. A
lookup table, scalar replay, bounded automaton, route model, or ordinary nonlinear
predictor must be evaluated under the same held-out split.

Preparation capacity alone does not exclude algorithmic compression. Exclusion requires
both:

1. an independently measured preparation-capacity bound;
2. held-out group-law generalization and causal operator interventions.

## 9. Relation mutation

The strongest matched mutation changes only ordered incidence:

```text
same operator multiset
same public amplitudes
same pages and banks
same routes and cores
same timing envelope
same marginal occupancy
different closed word
```

The primary contrast is not `both active` versus `separate`. It is:

```text
contractible word
versus
commutator word
versus
inverse commutator word
```

## 10. Kill matrix

| Attack | Required killer |
|---|---|
| Static answer vector | Post-source unseen words plus held-out composition |
| Scalar endpoint memory | Same endpoint and same operation multiset across loop classes |
| Route or bank artifact | Crossed mapping with exact route and bank equivalence |
| Ordinary nonlinear contention | Contractible and commuting-pair nulls |
| Generic order effect | Inverse, composition, and area laws |
| Timing drift | Randomized interleaving and time-matched sham loops |
| Source preselection | Independent post-death word entropy and sealed source-death receipt |
| Label-derived orientation | Blinded operator-to-physical-lane assignment |
| Reset called restoration | Wrong-inverse, natural-relaxation, and destructive-reset controls |
| Accumulator leakage | Carrier-off and reference-only nulls |
| Endpoint-only fake | Independent accumulator required |
| Post-hoc threshold | Prospective equivalence region and response law |
| Classical loop simulator | Claim limited to calibration unless native carrier use is proved |

## 11. Promotion ladder

```text
H0: post-source carrier coordinates observed
H1: two controlled transformations identified
H2: inverse and composition laws survive
H3: nontrivial commutator response survives matched nulls
H4: R2 carrier restoration plus retained accumulator invariant
H5: held-out word generalization excludes bounded answer caches
H6: higher-cycle construction linked to weighted fiber pushforward
```

Only H4 or above is relevant to catalytic holonomy. No rung in this document is already
passed.

## 12. Required result classes

```text
FAMILY10H_HOLONOMY_NOT_ESTABLISHED
FAMILY10H_ORDER_DEPENDENCE_OBSERVED
FAMILY10H_CONNECTION_LAW_CANDIDATE
FAMILY10H_CATALYTIC_HOLONOMY_CANDIDATE
FAMILY10H_HOLONOMY_CUSTODY_INVALID
```

Forbidden from this package:

```text
NATIVE_CATALYTIC_FIBER_PUSHFORWARD_ESTABLISHED
RELATIONAL_CARRIER_ESTABLISHED
SMALL_WALL_CROSSED
P_EQUALS_NP_PROVEN
```

# Jacobian Holonomy Bridge Kill Matrix

## Status

```text
FAIL_CLOSED_MATRIX_FROZEN_FOR_DESIGN
NO_LIVE_THRESHOLDS_FROZEN
NO_LIVE_EXECUTION_AUTHORIZED
```

Every row below can kill a future claim. Passing one row never rescues failure on
another required row.

## Mathematical bridge

| ID | Attack | Killer test | Failure disposition |
|---|---|---|---|
| MATH-01 | Unit Jacobian is asserted but not derived | Exact sparse-polynomial determinant equals one | Representation invalid |
| MATH-02 | Listed fiber points are incomplete | Exact `x=0` versus `x!=0` case split | Representation invalid |
| MATH-03 | Fiber points have hidden multiplicity | Full Jacobian nonzero at every target preimage | Fiber multiplicity claim invalid |
| MATH-04 | Selectors fail as idempotents | Exact partition and value table on all three roots | Formula weight invalid |
| MATH-05 | Null sheets contribute | Exhaustive null-sheet test | Fiber trace invalid |
| MATH-06 | Formula weight differs from SAT indicator | Exhaustive small-formula census against independent truth table | Semantic reduction invalid |
| MATH-07 | Prime sieve has positive false zero | Exhaustive count range plus product-bound proof | Total boundary invalid |
| MATH-08 | Reduced derivative is confused with full Jacobian | Separate full and elimination derivative certificates | Jacobian interpretation invalid |
| MATH-09 | Residue is promoted as efficient algorithm | Explicit non-result and enumeration audit | Claim inflation |
| MATH-10 | Phase functional is called physical holonomy | Require connection, transport, inverse, causal readout, restoration | Physical claim invalid |

## Compiler and representation

| ID | Attack | Killer test | Failure disposition |
|---|---|---|---|
| COMP-01 | Witness is embedded in preparation | Forbidden-field and source audit | No-smuggle failure |
| COMP-02 | `#SAT` or residues are precomputed | Independent regeneration and data-flow audit | No-smuggle failure |
| COMP-03 | Formula products are expanded | Circuit-node and coefficient-count manifest | Materialization failure |
| COMP-04 | Sheets are explicitly enumerated | Source trace and loop instrumentation | Materialization failure |
| COMP-05 | One mode is allocated per sheet | Physical mode census | Resource relocation |
| COMP-06 | Roots are solved before aggregation | Branch-label and root-list audit | Materialization failure |
| COMP-07 | Numerical quadrature performs the sum | Sample-count and algorithm audit | Native operator not established |
| COMP-08 | Exponential precision carries the answer | Dynamic-range and noise ledger | Resource relocation |
| COMP-09 | Exponential dwell time carries the search | Prospective deadline and scaling campaign | Resource relocation |
| COMP-10 | Hidden bath stores unresolved information | System-plus-environment accounting | Restoration invalid |

## Family 10h transport

| ID | Attack | Killer test | Failure disposition |
|---|---|---|---|
| HW-01 | Source knows realized loop | Post-source independent entropy and source-death seal | Custody invalid |
| HW-02 | Source executed the loop | Receiver-side operation receipts | Custody invalid |
| HW-03 | `A^-1` is only a label | Left and right inverse tests on held-out states | Inverse not established |
| HW-04 | Scalar order effect is called holonomy | Frozen carrier vector and operator model | Transport not established |
| HW-05 | Different operation counts explain result | Matched multiset within word-length stratum | Confounded |
| HW-06 | Route or bank explains result | Crossed mapping and exact route equivalence | Confounded |
| HW-07 | Timing drift explains result | Randomized interleaving and time-matched sham | Confounded |
| HW-08 | Ordinary contention explains result | Contractible and commuting-pair nulls | Connection law not established |
| HW-09 | Generic nonlinear order effect | Inverse, composition, cancellation, and area laws | Holonomy not established |
| HW-10 | Word recorder drives accumulator | Word-only replay null and coupling intervention | Accumulator invalid |
| HW-11 | Reference channel leaks class | Reference-only and carrier-off nulls | Accumulator invalid |
| HW-12 | Static answer table predicts fixed words | Post-source held-out words and lengths | Replay class not rejected |
| HW-13 | Compact predictor replaces table | Frozen bounded predictor suite | Replay class not rejected |
| HW-14 | Endpoint memory replaces path relation | Same endpoint and matched multiset word classes | Path relation not established |
| HW-15 | Aggregate rescues failed replicate | Both fresh replicates required | Candidate rejected |

## Restoration

| ID | Attack | Killer test | Failure disposition |
|---|---|---|---|
| R2-01 | Carrier was never displaced | Frozen forward-use floor | Ceremonial restoration |
| R2-02 | Natural relaxation restores state | Time-matched relaxation grid | R2 not established |
| R2-03 | Destructive reset is called inverse | Reset control and process continuity | R2 not established |
| R2-04 | Fresh carrier is substituted | Object identity and allocation custody | Custody invalid |
| R2-05 | Wrong inverse also passes | Same equivalence law on wrong inverse | R2 law non-identifying |
| R2-06 | Wrong inverse order also passes | Ordered inverse control | R2 law non-identifying |
| R2-07 | Carrier restores but output disappears | Independent accumulator retention | No computation extracted |
| R2-08 | Output remains but carrier does not restore | Carrier tomography | Catalytic closure absent |
| R2-09 | Environment is omitted | Environment classification ledger | R2 incomplete |
| R2-10 | Threshold is fitted after run | Hash-bound prospective contract | Custody invalid |

## Claim transitions

| Required result | Mandatory rows |
|---|---|
| Exact representation established | MATH-01 through MATH-08 |
| Native pushforward candidate | Exact representation plus COMP-01 through COMP-10 |
| Family 10h connection-law candidate | HW-01 through HW-09, HW-14, HW-15 |
| Bounded replay class rejected | HW-01, HW-02, HW-10, HW-12, HW-13 |
| R2 restoration candidate | R2-01 through R2-10 |
| Higher fiber-pushforward candidate | Separate higher-cycle theorem plus every relevant prior gate |

## Fail-closed Family 10h catalytic-holonomy transition

The token `FAMILY10H_CATALYTIC_HOLONOMY_CANDIDATE` may be emitted if and only if every
item below is present in the frozen contract, passed on the evidence, and independently
verified:

```text
PROMOTION_RUNG_REQUIREMENTS = H0, H1, H2, H3, H4, H5, H6, H7
CONNECTION_REQUIRED_ROWS = HW-01, HW-02, HW-03, HW-04, HW-05, HW-06,
                           HW-07, HW-08, HW-09, HW-14, HW-15
ACCUMULATOR_REQUIRED_ROWS = HW-10, HW-11, R2-07
BOUNDED_REPLAY_REQUIRED_ROWS = HW-01, HW-02, HW-10, HW-12, HW-13
R2_REQUIRED_ROWS = R2-01, R2-02, R2-03, R2-04, R2-05,
                   R2-06, R2-07, R2-08, R2-09, R2-10
ALL_REQUIRED_CONTROLS_PRESENT_PASSING_VERIFIED = true
```

The transition is the conjunction of all five lines. Any missing row, failed row,
unverified row, malformed custody record, or unpassed rung forbids
`FAMILY10H_CATALYTIC_HOLONOMY_CANDIDATE` and must produce one of the noncandidate result
classes defined by `FAMILY10H_PROTOCOL.md`. No aggregate statistic may rescue a failed
required row or failed fresh replicate.

## Hard claim ceiling

Failure of any required row leaves the strongest current token at:

```text
NATIVE_CATALYTIC_FIBER_PUSHFORWARD_NOT_ESTABLISHED
```

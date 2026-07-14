# Old Boundary Model Class

Status: `B_OLD_DEFINED_FOR_NEGATIVE_AUDIT`

Decision relevance: a bounded old-boundary class can be specified, but the current
Family 10h access model does not provide a principled preparation-capacity bound that
excludes the exact answer-cache explanation being tested.

## Class Definition

`B_old` is the family of predictors allowed to explain a claimed query-separated
response without invoking unresolved physical OrbitState access.

Each predictor may receive:

```text
source operation counts
marginal branch traces
total work
bank/address/route metadata
order and timing
public relation metadata
source receipts
realized query at scoring time
```

Each input belongs to the old boundary because it is either source-visible before
closure, receiver-public after query selection, or ordinary execution metadata. None
requires access to a hidden branch winner or post-boundary unresolved state.

## Allowed Predictor Families

`linear marginal model`

```text
Y_hat = beta_0 + beta_a * x_a + beta_b * x_b + beta_q * x_q
```

`additive branch model`

```text
Y_hat_q(a,b) = F_q(a) + G_q(b) + C_q
```

`nonlinear marginal-workload model`

```text
Y_hat = f_q(total_work, marginal_counts, timing, order)
```

`finite response table`

```text
Y_hat = table[(relation_id, query_id)]
```

`bank/route/order lookup`

```text
Y_hat = table[(bank_id, route_id, order_id, query_id)]
```

`timing model`

```text
Y_hat = f_q(source_start, source_stop, dwell, thermal_proxy, window_id)
```

`bounded-capacity answer cache`

```text
Y_hat = cache[q] if q in stored_queries else fail
```

## Splits

A future positive protocol must freeze disjoint sets:

```text
training relations
held-out relations
training queries
held-out queries
training seeds/sessions
held-out seeds/sessions
```

The adversary trains only on the training cross-product and scores on held-out cells.
The realized query is supplied to the adversary at scoring time.

## Metric And Tolerance

The default synthetic reference metric is normalized absolute or vector error:

```text
err = |Y - Y_hat| / max(|Y|, scale_floor)
```

A future physical protocol must define `scale_floor` from training/control data before
acquisition. This package uses `0.25` only as the relational model-consistency
tolerance inherited from the current design review vocabulary; it does not authorize
a live threshold.

Failure condition:

```text
if any B_old family predicts held-out Y within tolerance, the relational-carrier
claim is not established
```

## Complexity And Capacity Bound

The only principled way to admit a compact relation state while rejecting an answer
table is to freeze a preparation capacity:

```text
C_prep = independently measured preparation-channel capacity in bits
B_answer_raw = |Q| * bits_per_response
B_answer_min = minimum admissible ordinary answer-equivalent code length
B_relation = bits needed by the allowed relation representation
```

Capacity separation requires:

```text
B_answer_min > C_prep
B_relation <= C_prep
```

The present Family 10h proposal does not identify the persistent physical state or
its bit capacity after source closure. Cache-line ownership, bank route, order,
duration, and PMU-count coordinates are not yet reduced to a measured capacity law.
The raw table size `B_answer_raw` is only an upper-bound accounting device. It is not
an information-theoretic lower bound because compressed tables, low-rank bases,
formulas, circuits, seeds, coefficients, and compact encodings of `R` may generate the
same answers using fewer bits.

## Boundary Decision

`B_old` can be named for future protocols, but it is not yet a principled exclusion
boundary for the proposed shared-pair Family 10h carrier. Excluding finite response
tables without a measured `C_prep` would exclude the exact ordinary explanation under
test by definition rather than by evidence.

Result:

```text
QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED
```

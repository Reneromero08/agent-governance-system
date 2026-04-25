# Machine Learning Hypotheses

## H1: Descriptive Validity

The formula defines a real representation-geometry quantity rather than a
numerological artifact.

Operational claim:

- `R_simple` and/or `R_full` vary systematically across externally labeled
  pure vs mixed clusters of representations.

Failure condition:

- scores are unstable, degenerate, or indistinguishable from chance across
  repeated runs and datasets.

## H2: Incremental Validity

The formula adds value beyond simpler metrics.

Operational claim:

- `R_simple` or `R_full` beats at least three baseline metrics on externally
  labeled cluster-purity discrimination.

Failure condition:

- `E` alone, dispersion alone, or accepted geometry baselines consistently match
  or beat the formula.

## H3: Predictive Validity

The formula predicts downstream representation usefulness.

Operational claim:

- layer/model variants with higher `R` on unlabeled structure checks tend to
  produce better downstream performance on retrieval, clustering, or
  classification transfer.

Failure condition:

- no consistent relationship after controlling for simpler baselines.

## H4: Causal Usefulness

Using the formula changes training outcomes in a favorable way.

Operational claim:

- checkpoint selection, batch selection, or auxiliary regularization using `R`
  improves held-out task quality or robustness.

Failure condition:

- interventions do nothing or degrade task performance.

## H5: Cross-Domain Stability

The same operational formula works across multiple model/data regimes.

Operational claim:

- the same locked definition remains useful across at least two domains such as
  text and vision, or text classification and retrieval.

Failure condition:

- the formula only works after redefining symbols or retuning thresholds per
  domain.

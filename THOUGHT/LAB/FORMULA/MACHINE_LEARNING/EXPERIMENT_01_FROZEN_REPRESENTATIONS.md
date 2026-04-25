# Experiment 01: Frozen Representations

## Question

Can the formula discriminate externally labeled pure semantic clusters from
mixed semantic clusters in frozen embedding spaces, and does it outperform
simpler geometry baselines?

## Why This Goes First

This is the cheapest honest test:

- external labels define purity
- no training intervention is needed
- CUDA helps but is not mandatory
- failure here is informative

## Design

For a labeled text dataset:

1. sample examples per label;
2. embed them with a frozen encoder;
3. construct `pure` clusters containing items from one label only;
4. construct `mixed` clusters containing evenly mixed labels;
5. compute all metrics on each cluster;
6. treat cluster purity as the ground-truth binary label.

## Primary Outputs

- per-metric ROC AUC
- mean and standard deviation by cluster type
- ranking of metrics
- JSON artifact with all raw scores

## Initial Runtime Target

Use:

- `sentence-transformers/all-MiniLM-L6-v2`
- `glue/sst2`
- CUDA via `py -3.11`

This is intentionally modest. If the formula cannot produce a signal here, the
next experiments should be delayed until the operational definition is revised.

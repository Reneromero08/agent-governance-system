# Relation Temporal Lifetime Mutation Discovery

Run ID: `family10h_relation_temporal_lifetime_mutation_discovery_v1_0`
Archive SHA-256: `60b60f697b4fc6f6a346b1d9d77a54fd8f77376a1a9f527fb504f2a2496775f0`
Analysis SHA-256: `ddca8768d52e64ce97cdee54117fb3751696cbf57ed7dc5761413fc021131288`

| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| alive | 0.031519236 | -0.008471976 | 0.039991212 | 1.000 | 1.000 | False |
| dead | 0.042915581 | -0.006864589 | 0.049780170 | 1.245 | 6.479 | False |
| reset_double_flush | -0.025131681 | 0.010692149 | -0.035823830 | 0.896 | 8.778 | False |
| mutate_same | 0.063245603 | -0.007874496 | 0.071120100 | 1.778 | 4.216 | False |
| mutate_opposite | 0.076201845 | -0.017019978 | 0.093221823 | 2.331 | 8.475 | False |

| Method | prep/query phase rad | prep/query mag ratio | diag/skew phase rad | diag/skew mag ratio |
|---|---:|---:|---:|---:|
| alive | -0.749391 | 1.000 | 1.011455 | 1.000 |
| dead | -0.881690 | 0.483 | 0.553156 | 0.776 |
| reset_double_flush | 0.960963 | 0.482 | -2.992602 | 0.481 |
| mutate_same | -0.813825 | 1.190 | 0.818865 | 1.382 |
| mutate_opposite | -0.826089 | 1.118 | 0.653811 | 1.558 |

Discovery interpretation:
- alive D positive: `True`
- dead D positive: `True`
- source death kills below 0.25 x alive D: `False`
- dead all one-factor strata positive: `True`
- best reset method: `reset_double_flush`
- best reset kills aggregate below 0.25 x alive D: `False`
- best reset kills all one-factor strata below 0.25 x alive D: `False`
- best reset reduces dead D abs: `True`
- persistent source-written-state candidate with reset method: `False`
- mutate-same rebuilds positive D: `True`
- mutate-opposite flips D negative: `False`
- temporal lifetime mutation candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.

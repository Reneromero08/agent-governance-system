# N1 Report

Status: EXECUTED

## Pre-Registered Outcome

### stsb

- Source: glue/stsb validation
- Clusters: pure=120, mixed=120
- AUC(E): 0.4615
- AUC(R_simple): 0.4533
- AUC(R_full): 0.4547
- AUC(random): 0.5112
- E - R_simple delta: 0.0083 [-0.0158, 0.0323]

### sst2

- Source: glue/sst2 validation
- Clusters: pure=120, mixed=120
- AUC(E): 0.5915
- AUC(R_simple): 0.5555
- AUC(R_full): 0.5489
- AUC(random): 0.5280
- E - R_simple delta: 0.0355 [-0.0193, 0.0897]

### snli

- Source: snli validation
- Clusters: pure=120, mixed=120
- AUC(E): 0.4720
- AUC(R_simple): 0.4621
- AUC(R_full): 0.4654
- AUC(random): 0.5234
- E - R_simple delta: 0.0092 [-0.0209, 0.0397]

### mnli

- Source: glue/mnli validation_matched
- Clusters: pure=120, mixed=120
- AUC(E): 0.4628
- AUC(R_simple): 0.4662
- AUC(R_full): 0.4651
- AUC(random): 0.5456
- E - R_simple delta: -0.0036 [-0.0402, 0.0301]

## Overall Decision

- E wins: 0
- R_simple wins: 0
- Ties: 4
- Hypothesis status: mixed

## Anti-Pattern Checks

### stsb

- ground_truth_independent_of_metrics: PASS
- parameters_fixed_before_results: PASS
- no_grid_search: PASS
- negative_results_will_be_reported: PASS
- no_goalpost_moving: PASS
- both_classes_present: PASS
- all_scores_finite: PASS

### sst2

- ground_truth_independent_of_metrics: PASS
- parameters_fixed_before_results: PASS
- no_grid_search: PASS
- negative_results_will_be_reported: PASS
- no_goalpost_moving: PASS
- both_classes_present: PASS
- all_scores_finite: PASS

### snli

- ground_truth_independent_of_metrics: PASS
- parameters_fixed_before_results: PASS
- no_grid_search: PASS
- negative_results_will_be_reported: PASS
- no_goalpost_moving: PASS
- both_classes_present: PASS
- all_scores_finite: PASS

### mnli

- ground_truth_independent_of_metrics: PASS
- parameters_fixed_before_results: PASS
- no_grid_search: PASS
- negative_results_will_be_reported: PASS
- no_goalpost_moving: PASS
- both_classes_present: PASS
- all_scores_finite: PASS

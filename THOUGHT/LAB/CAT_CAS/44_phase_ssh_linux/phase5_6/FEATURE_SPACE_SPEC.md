# Phase 5.6 Feature Space Spec

Status: `PHASE5_6_FULL_CARRIER_FEATURES_BUILT`

The canonical harness generates real CAT_CAS T0/T1/T2/T3 carrier rows from the Phase 3B transition model instead of relying on scalar summary CSVs. Predictive features include snapshot signatures, carrier slots, T2 answer boundary slots, residual tags, .holo slots, and operator-statistic proxies. Outcome labels (`answer_correct`, `pass_label`, `class_label`) remain diagnostic and are excluded from the predictive distance body.

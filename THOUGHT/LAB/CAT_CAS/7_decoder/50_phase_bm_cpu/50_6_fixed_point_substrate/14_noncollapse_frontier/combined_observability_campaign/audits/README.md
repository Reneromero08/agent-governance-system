# Phase 6 Post-Acquisition Audits

This directory preserves forward-only audit results for the completed Phase 6
combined-observability acquisition. Historical executor, plan, authorization,
bundles, and evidence remain immutable.

## Canonical current state

```text
PHASE6_ACQUISITION_COMPLETE
PROVENANCE_VALID
RAW_IDENTITY_AND_STRUCTURE_AUDITED
RETROSPECTIVE_FULL_DATASET_NEGATIVE_ADJUDICATION
PRISTINE_FINAL_TEST_HYGIENE_NOT_PROVEN
NO_STABLE_PREDICTIVE_OPERATOR
V2_RERUN_NOT_AUTHORIZED
V2_CALIBRATION_NOT_EXECUTED
SMALL_WALL_NOT_CROSSED
```

`PHASE6_FULL_ADJUDICATION_SUMMARY.md` records the permanent scientific verdict
from the digest-bound historical `full_adjudication.json`. That is the recorded
full adjudication.

`FROZEN_ARTIFACT_AND_RECORDED_OUTPUT_BINDING_AUDIT.md` documents a separate
independent audit of immutable artifact identities, serialized structure, and
recorded-output bindings. It does not independently recompute Stage B metrics,
Stage C gains, operator NRMSE values, or the scientific verdict.

The sibling `analysis/` tools model the waveform that actually executed. They
do not mutate or reinterpret historical evidence.

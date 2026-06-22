# Frozen Artifact And Recorded Output Binding Audit

Audit name: `FROZEN_ARTIFACT_AND_RECORDED_OUTPUT_BINDING_AUDIT`

This is not an independent frozen-model replay. It verifies immutable artifact
digests, deserializes the frozen model and seed-5 caches only after their
digests pass, validates their serialized shapes, and binds them to the
digest-verified historical adjudication record.

Frozen model SHA-256:

```text
f1b8047ba0d80d027edcec9a841c8a4320dfe6796bbad7271fd43dc6a86dee5e
```

Independent Stage B, Stage C, and operator metric recomputation is not possible
from the three frozen artifacts because they do not serialize:

- Stage B ground-truth labels;
- transition controls and targets;
- Stage C transition coefficients and frozen baselines;
- mechanical verdict thresholds.

The Stage B metrics, Stage C zero-input gains, operator NRMSE values, and final
verdicts stored in `full_adjudication.json` are recorded-output bindings only.
They are not recomputed or independently confirmed by this audit.

Permanent V1 statement:

```text
RETROSPECTIVE_FULL_DATASET_NEGATIVE_ADJUDICATION
PRISTINE_FINAL_TEST_HYGIENE_NOT_PROVEN
V2_RERUN_NOT_AUTHORIZED
```

Immutable binding manifest:

```text
LAW/CONTRACTS/_runs/frozen_artifact_and_recorded_output_binding_audit_f1b8047b/FROZEN_ARTIFACT_AND_RECORDED_OUTPUT_BINDING_AUDIT.json
SHA-256: d9d4e44f0bfeee83f8030815e0477dc3c4ed789b10c63ec45d2b9c7fe50908ae
```

No model development, model selection, seed-5 retry, hardware execution, or V2
authorization occurred.

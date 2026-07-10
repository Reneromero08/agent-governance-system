# Gate A target non-executing qualification status

The authority order for this directory is:

1. `GATE_A_TARGET_NONEXECUTING_QUALIFICATION_ADJUDICATION.json` is the current truth.
2. `GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V4.json` is the current blocked authority state.
3. `GATE_A_TARGET_NONEXECUTING_QUALIFICATION_RESULT.json` and `GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V3.json` are preserved historical interpretations and are superseded for current-status and current-authority purposes.
4. `phase6b6/evidence/gate_a_target_nonexecuting_qualification_6f243b1a_bundle_abc9e50a/` is the immutable forensic packet for the one historical attempt.
5. `GATE_A_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json` is consumed authority for that historical attempt only. It cannot authorize a retry or replacement run.

The historical attempt returned success from the no-drive runner and validate-only worker, but it did not bind the inner `ps` return code or raw process listing, and cleanup asserted an empty process set without a fail-closed scan. Target non-executing qualification therefore remains incomplete. No replacement run, Gate A engineering smoke, hardware execution, calibration, scientific acquisition, restoration, target coupling, or Small Wall work is authorized.

The next boundary is `PROJECT_OWNER_DECISION_FOR_ONE_REPLACEMENT_GATE_A_TARGET_NONEXECUTING_QUALIFICATION`.

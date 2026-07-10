# Gate A target non-executing qualification status

The authority order for this directory is:

1. `GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORITY_STATE.json` is the current authority ledger. It records one authorized, unconsumed replacement qualification and zero replacement executions.
2. `GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json` is the exact committed V2 owner authority for that one replacement execution.
3. `GATE_A_TARGET_NONEXECUTING_QUALIFICATION_ADJUDICATION.json` remains the authoritative verdict on the preserved historical attempt.
4. `GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V4.json` remains the preserved pre-decision blocked authority state; it is not rewritten after the later owner decision.
5. `GATE_A_TARGET_NONEXECUTING_QUALIFICATION_RESULT.json` and `GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V3.json` are preserved historical interpretations and are superseded for current-status and current-authority purposes.
6. `phase6b6/evidence/gate_a_target_nonexecuting_qualification_6f243b1a_bundle_abc9e50a/` is the immutable forensic packet for the one historical attempt.
7. `GATE_A_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json` is consumed authority for that historical attempt only. It cannot authorize a retry or replacement run.

The historical attempt returned success from the no-drive runner and validate-only worker, but it did not bind the inner `ps` return code or raw process listing, and cleanup asserted an empty process set without a fail-closed scan. Target non-executing qualification therefore remains incomplete. The project owner has now authorized exactly one replacement target non-executing qualification under authority `gate_a_replacement_71ab1528_01`; it has not yet executed and automatic retry is false. Gate A engineering smoke, hardware execution, calibration, scientific acquisition, restoration, target coupling, and Small Wall work remain unauthorized.

The next boundary is `EXECUTE_ONE_AUTHORIZED_REPLACEMENT_GATE_A_TARGET_NONEXECUTING_QUALIFICATION`, only after the reconciliation commit is green at its exact hosted head.

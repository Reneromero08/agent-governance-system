# Gate A target non-executing qualification status

The authority order for this directory is:

1. `GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORITY_STATE.json` is the current authority ledger. It records one active, unconsumed replacement authority with zero executions and preserves the prior authority as superseded-unconsumed.
2. `GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION_SUPERSEDED_71ab1528_01.json` preserves the prior V2 owner authority byte-for-byte as `SUPERSEDED_UNCONSUMED`. Its protected runner changed before use, so it cannot authorize target contact. The active authority basename is absent from the repaired source commit.
3. `GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json` is the one active V2 authority `gate_a_replacement_593e9920_02`. It binds reviewed source commit `593e9920be533603217cee93572d79b86cc65cf9`, permits one orchestrator-only no-drive qualification, and forbids automatic retry.
4. `GATE_A_TARGET_NONEXECUTING_QUALIFICATION_ADJUDICATION.json` remains the authoritative verdict on the preserved historical attempt.
5. `GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V4.json` remains the preserved pre-decision blocked authority state; it is not rewritten after the later owner decision.
6. `GATE_A_TARGET_NONEXECUTING_QUALIFICATION_RESULT.json` and `GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V3.json` are preserved historical interpretations and are superseded for current-status and current-authority purposes.
7. `phase6b6/evidence/gate_a_target_nonexecuting_qualification_6f243b1a_bundle_abc9e50a/` is the immutable forensic packet for the one historical attempt.
8. `GATE_A_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json` is consumed authority for that historical attempt only. It cannot authorize a retry or replacement run.

The historical attempt returned success from the no-drive runner and validate-only worker, but it did not bind the inner `ps` return code or raw process listing, and cleanup asserted an empty process set without a fail-closed scan. Target non-executing qualification therefore remains incomplete. Authority `gate_a_replacement_71ab1528_01` was never consumed: its execution count is zero, its evidence namespace is absent, and automatic retry is false. It is now superseded because the protected runner first proves execution-root, transfer-stage, evidence-archive, and temp-prefix absence before any remote temp-prefix write. Inspection errors and unobservable states fail closed.

Reviewed source commit `593e9920be533603217cee93572d79b86cc65cf9` passed its exact-head hosted workflows and an independent no-findings review. The next boundary is `EXECUTE_ONE_ORCHESTRATOR_ONLY_NO_DRIVE_QUALIFICATION__NO_RETRY`, only after this authority-bearing commit is itself clean and green. Gate A engineering smoke, probing, sender execution, receiver capture, MSR access, hardware execution, calibration, scientific acquisition, restoration, target coupling, fold-odd recovery, and Small Wall work remain unauthorized.

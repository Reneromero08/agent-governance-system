# Local Agent Runbook — Combined Campaign

**Current role:** implement and verify the schedule-driven hardware runner, then execute only after preflight passes.  
**Owner authorization:** combined campaign authorized after frozen-package and machine preflight.  
**Restoration:** forbidden.

## 1. Synchronize

```bash
git fetch origin --prune
git switch phase6b/carrier-witness-closure
git pull --ff-only origin phase6b/carrier-witness-closure
git status --short
```

Read:

```text
gate_r/PROJECT_OWNER_RATIFICATION.md
gate_r/GATE_R_STATUS.md
gate_r/COMBINED_CAMPAIGN_BINDING.json
combined_observability_campaign/CAMPAIGN_CONTRACT.md
combined_observability_campaign/ANALYSIS_CONTRACT.md
combined_observability_campaign/HARDWARE_EXECUTOR_INTERFACE.md
combined_observability_campaign/EXECUTOR_OUTPUT_CONTRACT.md
```

## 2. Verify the frozen package

Run the package workflow tests locally. Do not alter orders, counts, partitions, thresholds, or sender-off semantics.

Generate or obtain the campaign plan named in `COMBINED_CAMPAIGN_BINDING.json` and verify its SHA-256. The first noncanonical PR-merge-ref artifact must never be used.

## 3. Implement the local runner

Create the schedule-driven runner outside the frozen planner directory, preferably:

```text
14_noncollapse_frontier/holo_runtime/combined_pdn_runner.c
```

Reuse the proven Slot2 affinity, TSC, drive, capture, thermal-veto, telemetry, P-state restoration, and immutable-writer primitives. Do not modify historical evidence or reinterpret `windows.jsonl`.

The runner must satisfy `HARDWARE_EXECUTOR_INTERFACE.md` and `EXECUTOR_OUTPUT_CONTRACT.md`, including true sender-off capture with no active drive thread.

Add build/tests and push the executor implementation. Record its exact commit separately; the executor commit does not change the frozen campaign plan.

## 4. Dry validation on catcas

Compile all twelve session schedules from the frozen plan. Run the runner in non-driving validation mode if available. Confirm every schedule and output path is new.

Execute the read-only preflight:

```bash
python3 combined_observability_campaign/catcas_preflight.py \
  --plan-dir PLAN_DIR \
  --repo-root REPO_ROOT \
  --output-root /root/catcas_evidence/phase6_combined_AUTHORIZED \
  --report /root/catcas_evidence/phase6_combined_preflight.json
```

Do not continue unless `acquisition_ready=true` for every check.

## 5. Execute

Use `run_combined_campaign.py` with the verified runner. Do not use automatic retries. Do not inspect scientific outcomes to alter later blocks. A thermal or integrity abort ends that session and is reported as evidence.

The execution order is the frozen session order. Seed 4 is mandatory. Sender-off windows contain no refresh, replay, or sender workload.

## 6. Return proof

Return:

- plan and campaign-manifest hashes;
- planner source commit;
- executor source commit;
- preflight report;
- commands and host identity;
- all session/run manifests and raw hashes;
- aborts or missing sessions;
- P-state restoration proof;
- compact evidence only in Git; raw evidence remains on catcas.

Do not analyze, classify persistence, or fit operators before immutable raw provenance is complete.

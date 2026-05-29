<!-- CONTENT_HASH: updated_for_phase_h -->

# CAT_CHAT Usage Guide (CLI)

This is a practical, copy-paste oriented guide for using **CAT_CHAT** from the repo root on Windows PowerShell.

## Fresh Clone Quick Start

**New to CAT_CHAT?** Run the golden demo to see the system in action:

```powershell
cd "D:\CCC 2.0\AI\agent-governance-system"
$env:PYTHONPATH = "THOUGHT\LAB\CAT_CHAT"
python THOUGHT\LAB\CAT_CHAT\golden_demo\golden_demo.py
```

This demo shows:
- Bundle creation (deterministic packaging)
- Bundle verification (hash integrity)
- Bundle execution (receipt generation)
- Receipt verification (chain integrity)

For detailed specifications, see [docs/specs/SPEC_INDEX.md](specs/SPEC_INDEX.md).

## 0) One-time setup (PowerShell)

From repo root:

```powershell
cd "D:\CCC 2.0\AI\agent-governance-system"
$env:PYTHONPATH = "THOUGHT\LAB\CAT_CHAT"
python -m catalytic_chat.cli --help
```

If you want to avoid setting `PYTHONPATH` every session:

```powershell
setx PYTHONPATH "D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CHAT"
```

(You must open a new terminal after `setx`.)

## 1) Mental model

CAT_CHAT is a deterministic pipeline with these main layers:

1. **Index**: deterministic section extraction and lookup (`build`, `verify`, `get`, `extract`)
2. **Symbols**: stable references to sections with bounded slices (`symbols ...`, `resolve`)
3. **Cassette DB**: durable jobs, steps, receipts (`cassette ...` used implicitly by `plan/execute/bundle`)
4. **Planner**: produces deterministic steps from a request (`plan request`)
5. **Execution**: executes steps and emits deterministic receipts (`execute`, `ants`, `bundle run`)
6. **Bundle**: deterministic translation protocol (`bundle build`, `bundle verify`, `bundle run`)
7. **Trust + Policy**: fail-closed gates (`trust ...`, `--strict-trust`, `--policy ...`)

## 2) Common PowerShell gotchas

### Angle brackets are not placeholders in PowerShell
Do **not** type `<JOB_ID>` or `<RUN_ID>`. Replace with real values.

Bad:
```powershell
--job-id <job_id>
```

Good:
```powershell
--job-id "job_123..."
```

### Quote `@` arguments
Always quote symbol ids and prefixes:

```powershell
python -m catalytic_chat.cli symbols list --prefix "@"
```

## 3) Indexing

### Build the section index
Run from repo root (recommended):

```powershell
python -m catalytic_chat.cli build --repo-root "D:\CCC 2.0\AI\agent-governance-system"
```

### Verify determinism
```powershell
python -m catalytic_chat.cli verify --repo-root "D:\CCC 2.0\AI\agent-governance-system"
```

### Inspect a section
You need a real `section_id` from the index output (or your own lookup tooling). Then:

```powershell
python -m catalytic_chat.cli get "<SECTION_ID>" --slice "head(50)"
```

## 4) Symbols

### List symbols
```powershell
python -m catalytic_chat.cli symbols list --prefix "@"
```

If you see `no such table: symbols`, you are either:
- using the wrong substrate DB, or
- the registry was not initialized on this substrate, or
- you are pointing at a different repo-root than you think.

### Add a symbol
You must provide a valid section id:

```powershell
python -m catalytic_chat.cli symbols add "@TEST/example" --section "<SECTION_ID>" --default-slice "lines[0:200]"
```

### Resolve a symbol (bounded slice)
```powershell
python -m catalytic_chat.cli resolve "@TEST/example"
```

## 5) Planning

### Dry-run a plan request (no DB writes)
```powershell
python -m catalytic_chat.cli plan request --request-file "THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_min.json" --dry-run
```

Dry-run must succeed even if a referenced symbol is missing, and will emit an unresolved step sentinel in `expected_outputs`.

### Create a plan request (writes to cassette DB)
```powershell
python -m catalytic_chat.cli plan request --request-file "THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_min.json"
```

It prints:
- `message_id`
- `job_id`
- number of steps

If it fails with missing symbol: add the symbol first, or use a fixture that matches your registry.

## 6) Execution (direct)

Execute PENDING steps for a job:

```powershell
python -m catalytic_chat.cli execute --run-id "test_plan_001" --job-id "job_..." --workers 1 --continue-on-fail
```

If it says `No PENDING steps found`, your plan has zero steps, or the job is already complete, or you are using the wrong run_id/job_id pair.

## 7) Ant workers (parallel execution)

### Help
```powershell
python -m catalytic_chat.cli ants --help
```

### Spawn workers
```powershell
python -m catalytic_chat.cli ants spawn --run-id "my_run" --job-id "job_..." --workers 4
```

### Alias
`ants run` is an alias for `ants spawn`.

### Status (SQLite introspection only)
```powershell
python -m catalytic_chat.cli ants status --run-id "my_run" --job-id "job_..."
```

Output is stable and line-based:
- PENDING
- LEASED
- COMMITTED
- RECEIPTS
- WORKERS_SEEN

## 8) Bundle: build, verify, run

Bundle build is fail-closed unless the job is complete:
- all steps are COMMITTED
- exactly one receipt exists per step

### Build a bundle
```powershell
python -m catalytic_chat.cli bundle build --run-id "my_run" --job-id "job_..." --out "THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/bundles/my_bundle"
```

Output directory contains:
- `bundle.json` (canonical JSON)
- `artifacts/` (bounded slice files)

### Verify a bundle
```powershell
python -m catalytic_chat.cli bundle verify --bundle "THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/bundles/my_bundle"
```

Useful flags:
- `--json` prints JSON only to stdout (trailing newline)
- `--quiet` suppresses non-error stderr lines

Example:
```powershell
python -m catalytic_chat.cli bundle verify --bundle "THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/bundles/my_bundle" --json
```

### Run a bundle (deterministic replay from artifacts only)
```powershell
python -m catalytic_chat.cli bundle run --bundle "THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/bundles/my_bundle"
```

The executor only uses bundle artifacts for reads (artifact confinement).

## 9) Receipts, chains, Merkle root

### Verify receipt chain
`bundle run` can emit receipts to an output path (your CLI may expose `--receipt-out` or similar). When receipts exist:

- `--verify-chain` validates linkage and ordering
- Merkle root can be computed only after verification

### Print Merkle root (stdout-only)
Requires `--verify-chain`.

```powershell
python -m catalytic_chat.cli bundle run --bundle "..." --verify-chain --print-merkle
```

When `--print-merkle` is used, stdout contains only the Merkle root.

## 10) Attestation

### Receipt attestation
`bundle run` can sign receipts when a signing key is provided (ed25519). Output is hex-only and deterministic.

### Merkle attestation
Signs the Merkle root message of the form:
`CAT_CHAT_MERKLE_V1:<root>|VID:<validator_id>|BID:<build_id>|PK:<public_key>`

Key flags (exact names depend on your current CLI wiring):
- `--attest-merkle`
- `--merkle-key <hex>`
- `--merkle-attestation-out <path>`
- `--verify-merkle-attestation <path>`

## 11) Trust policy

Trust policy pins which public keys are allowed and (optionally) which build_id is pinned per validator.

### Verify trust policy
```powershell
python -m catalytic_chat.cli trust verify --trust-policy "THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/TRUST_POLICY.json"
```

### Strict trust and identity
When running verifications, strict modes fail-closed if:
- key is not pinned for the required scope
- validator_id does not match pinned entry
- build_id is pinned but mismatches (when strict identity is enabled)

## 12) Execution policy gate (Phase 6.8)

Execution policy allows you to require, in a single gate, things like:
- must verify bundle before run
- must verify receipt chain and compute merkle root
- must require receipt attestation and/or merkle attestation
- must enforce strict trust and strict identity pins
- quorum requirements (if multi-validator aggregation is in use)

You can pass a policy file to `bundle run`:

```powershell
python -m catalytic_chat.cli bundle run --bundle "..." --policy "THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/EXECUTION_POLICY.json"
```

(Use the exact policy path you created in your repo.)

## 13) Exit codes (Phase 6.14)

Exit codes are standardized:
- 0 OK
- 1 Verification failed (hash, trust, policy, bounds, ordering)
- 2 Invalid input (missing file, bad JSON, schema invalid)
- 3 Internal error (unexpected exception)

`--json` outputs machine-readable JSON to stdout only.
Human logs go to stderr.

## 14) Troubleshooting

### ModuleNotFoundError: No module named 'catalytic_chat'
You ran from repo root without `PYTHONPATH`.
Fix:

```powershell
cd "D:\CCC 2.0\AI\agent-governance-system"
$env:PYTHONPATH = "THOUGHT\LAB\CAT_CHAT"
python -m catalytic_chat.cli --help
```

### no such table: symbols
You are pointing at a DB that does not contain the symbol registry. Check:
- correct `--repo-root`
- correct `--substrate`
- that the symbol registry initialization was performed for that substrate

### plan request fails on missing symbol
Either add the symbol, or expect unresolved steps (dry-run), or change the fixture.

### ants status says job not found
You used a job_id/run_id pair that does not exist in that cassette DB. Use the printed ids from `plan request`.

### execute shows 0 pending steps
That job has no steps, or steps are already complete, or you are querying the wrong run/job.

---

If you want, paste:
- the exact `plan_request_*.json` you are running, and
- the command you ran,
and I can tell you the shortest valid command sequence for your current state.

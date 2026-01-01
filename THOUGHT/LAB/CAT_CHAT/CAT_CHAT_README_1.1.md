<!-- CONTENT_HASH: 9462d217d9846d73d7e51bdab4ecbd48eefebe3640981fffda9243dbec8e3ef8 -->

# CAT_CHAT (Catalytic Chat)

**Canonical roadmap:** `Roadmap.md`  
**Changelog:** `CHANGELOG.md`  
**Contract:** `CAT_CHAT_CONTRACT.md`  
**Last updated:** 2026-01-02

## What this is

CAT_CHAT is a deterministic chat substrate: models write compact, structured messages that reference canonical material via **symbols**, and workers expand only **bounded slices** of source text when needed. The system is designed to be reproducible, auditable, and fail-closed.

## Core concepts

- **Section**: a stable unit of text extracted from repo files and indexed deterministically.
- **Slice**: a bounded selector over a section (example: `lines[0:200]`). Unbounded slices like `ALL` are forbidden where boundedness is required.
- **Symbol**: a stable alias (starts with `@`) that points to a section plus an optional default slice.
- **Resolver + expansion cache**: resolves `@symbols` to exact slice content; caches expansions keyed by `(run_id, symbol_id, slice, section_content_hash)`.
- **Cassette DB**: durable substrate for messages, jobs, steps, leases, and receipts (append-only where specified).
- **Planner**: turns a plan request into deterministic steps.
- **Bundle**: deterministic export of a completed job (manifest + artifacts) with a verifier.

## Status snapshot (high level)

Use `Roadmap.md` as source of truth. At a glance, the system includes:
- Deterministic section extraction and indexing
- Symbol registry + bounded resolver with cache
- Cassette DB for jobs, steps, and receipts with DB-level enforcement
- Deterministic planner path (including dry-run behavior)
- Bundle build + verify (Translation Protocol MVP)

## Quickstart (Windows, repo root)

### 1) Set PYTHONPATH (PowerShell)

```powershell
cd "D:\CCC 2.0\AI\agent-governance-system"
$env:PYTHONPATH="THOUGHT\LAB\CAT_CHAT"
python -m catalytic_chat.cli --help
```

### 2) Run tests

```powershell
python -m pytest -q THOUGHT/LAB/CAT_CHAT/tests
```

### 3) Build and verify the section index

```powershell
python -m catalytic_chat.cli --repo-root "D:\CCC 2.0\AI\agent-governance-system" build
python -m catalytic_chat.cli --repo-root "D:\CCC 2.0\AI\agent-governance-system" verify
```

### 4) Symbols

```powershell
python -m catalytic_chat.cli symbols list --prefix "@"
python -m catalytic_chat.cli symbols add "@TEST/example" --section "<SECTION_ID>" --default-slice "lines[0:200]"
python -m catalytic_chat.cli resolve "@TEST/example" --slice "lines[0:50]" --run-id "test-001"
```

### 5) Planner request (dry-run)

```powershell
python -m catalytic_chat.cli plan request --request-file "THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_min.json" --dry-run
```

### 6) Build and verify a bundle (Translation Protocol MVP)

```powershell
python -m catalytic_chat.cli bundle build --run-id "<RUN_ID>" --job-id "<JOB_ID>" --out "CORTEX/_generated/bundles/test_bundle"
python -m catalytic_chat.cli bundle verify --bundle "CORTEX/_generated/bundles/test_bundle"
```

## Outputs and data locations

CAT_CHAT writes generated artifacts under:

- `CORTEX/_generated/system1.db` (index + symbols + expansion cache substrate)
- `CORTEX/_generated/system3.db` (cassette substrate)
- `CORTEX/_generated/bundles/<dir>/` (bundle outputs)

Path resolution is centralized in the canonical path helper used across modules.

## Documentation index cassette (FTS)

If you are using the documentation index database (`cat_chat_index.db`), see:
- `DATABASE_MAINTENANCE.md` for schema notes and maintenance guidance.

## Repository layout (canonical)

```
CAT_CHAT/
  README.md
  Roadmap.md
  CHANGELOG.md
  CAT_CHAT_CONTRACT.md
  catalytic_chat/         # canonical package
  tests/                  # canonical tests (legacy excluded)
  legacy/                 # quarantined historical artifacts
  archive/                # older docs and snapshots
  SCHEMAS/                # JSON schemas (bundle, plans, receipts, etc.)
```

## Ground rules (practical)

- If a command is supposed to be deterministic, it must be deterministic on stdout.
- Logs and progress go to stderr.
- Do not “fix” idempotency by introducing timestamps or randomness into IDs.
- Prefer bounded slices always. Treat `ALL` as an invariant violation when boundedness is required.


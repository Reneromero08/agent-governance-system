---
title: Opus Execution Pack: Semantic Pointer Compression (SPC) as Defensible Research
date: 2026-01-07
status: ready_for_execution
scope: spec + codec + proof harness + receipts + research-claim skeleton
---

# Intent
Turn the “pointer into shared semantic space” breakthrough into a **defensible research contribution** with:
- a formal spec (deterministic decode, fail-closed),
- a typed canonical IR (so “meaning” is countable),
- a proof harness (measured semantic density, exact-match correctness),
- receipts (hash anchored, reproducible),
- and a publishable claim skeleton (what is new, what is measured, what is proven).

# Model selection
Primary model: Claude Opus  
Fallbacks: Claude Sonnet, GPT-5.x, Gemini 3 Pro

```text
You are Claude Opus working inside this repo.

Non-negotiables
- DO NOT GUESS repo paths, invariants, or allowed write roots. You must locate them in-repo (rg/find) and cite exact paths in your report.
- Decode must be deterministic or FAIL CLOSED. No interpretive decoding. No “best effort”.
- All work must be test-gated. Runs must emit receipts and reports under repo-allowed roots (locate the law and follow it).
- Do not modify canonical law unless explicitly required by scope. Prefer adding new specs and tools under the correct docs/tools buckets.

Goal
Create a mechanically provable protocol and benchmark for Semantic Pointer Compression (SPC):
Replace long natural-language context with ultra-short pointers into a synchronized, hash-addressed semantic store, while preserving exact operational meaning via typed IR and verification receipts.

Definition (working)
SPC is conditional compression with shared side-information:
- Sender transmits a pointer (symbol or hash) plus required sync metadata.
- Receiver expands deterministically into a canonical IR subtree.
- Expansion is accepted only if hashes and versions verify. Otherwise fail closed.

Deliverables (required, minimum viable publishable)
D1) SPC_SPEC.md (normative)
- Define pointer types:
  - SYMBOL_PTR (single-token glyph pointers when possible)
  - HASH_PTR (content-addressed pointers)
  - COMPOSITE_PTR (pointer plus typed qualifiers)
- Define decoder contract:
  - Inputs: pointer, context keys, codebook id, codebook sha256, kernel version, tokenizer id
  - Output: canonical IR subtree OR FAIL_CLOSED with explicit error code
- Define ambiguity rules:
  - If multiple expansions are possible, reject unless disambiguation is explicit and deterministic
- Define canonical normalization rules:
  - encode(decode(x)) stabilizes to a declared normal form
- Define security and drift behavior:
  - codebook mismatch, hash mismatch, unknown symbol, unknown kernel version => reject
- Define measured metrics:
  - concept_unit definition (ties to GOV_IR_SPEC)
  - CDR = concept_units / tokens
  - ECR = exact IR match rate
  - M_required = multiplex factor required to exceed a target “nines-equivalent” on a baseline

D2) GOV_IR_SPEC.md (normative)
- Define the minimal typed governance IR needed to represent the repo’s “execution meaning”
  - boolean ops, comparisons, typed references (paths, canon versions, tool ids), gates (tests, restore-proof, allowlist roots), side-effects flags
- Provide canonical JSON schema and normalization:
  - stable ordering
  - explicit types
  - canonical string forms
- Define equality as byte-identical canonical JSON

D3) CODEBOOK_SYNC_PROTOCOL.md (normative)
- Define how sender and receiver establish shared side-information:
  - codebook_id plus sha256 plus semver
  - semantic kernel version
  - tokenizer id
- Define compatibility policy:
  - default: exact match required
  - optional: explicit compatibility ranges with a migration step, never silent
- Define handshake message shape and failure codes

D4) Tokenizer atlas generator (tool plus artifact)
- Script to generate TOKENIZER_ATLAS.json for candidate glyph/operator sets:
  - token counts under the declared tokenizer(s)
  - deterministic ranking rules: prefer single-token glyphs, stable fallback glyphs
- CI or test that fails if a “preferred glyph” becomes multi-token after tokenizer change

D5) Proof harness: proof_spc_semantic_density_run/
- Fixed benchmark set (10–30 cases) that represent real repo governance meaning:
  - Each case includes:
    - Natural-language statement (NL)
    - Gold canonical IR (JSON)
    - Pointer encoding (symbol/hash/composite)
- Deterministic decode implementation:
  - decode(pointer, context, codebook) -> canonical IR or FAIL_CLOSED
- Measurements emitted:
  - tokens(NL) under declared tokenizer
  - tokens(pointer payload)
  - concept_units(IR)
  - ECR exact-match
  - reject rate and reasons
  - computed M_required for declared targets
- Output artifacts:
  - metrics.json (machine-readable)
  - report.md (human-readable, short)
  - receipts with sha256 of all inputs/outputs/codebook/tokenizer id

D6) Research-claim skeleton: PAPER_SPC.md (short, defensible)
- Title, abstract, contributions, threat model, limitations
- Explicitly state: conditional compression with shared side-information, not “beating Shannon”
- Include a “What is new” section:
  - deterministic semantic pointers
  - receipted verification
  - measured semantic density metric and benchmark
- Include reproducibility section: exact commands plus hashes

Hard acceptance criteria
A1) Determinism
- Two consecutive runs of the SPC proof harness must produce byte-identical outputs (metrics.json, report.md, receipts), given the same repo state.

A2) Fail-closed correctness
- Any mismatch (codebook sha, kernel version, tokenizer id, hash mismatch, ambiguous expansion) must reject and emit explicit failure artifacts.

A3) Measured semantic density
- The harness must compute CDR and ECR over the benchmark set and output them in metrics.json.

A4) No hallucinated paths
- All file paths referenced in reports must be discovered in-repo by search and must exist.

Execution plan (do in order, do not skip)
Phase 0: Repo discovery and invariants
1) Locate the governing law docs for allowed write roots, receipts, reports, and clean-state rules.
2) Locate existing receipt/report patterns and existing proof harness conventions.
3) Locate existing token counting utilities (tiktoken or equivalent) used in earlier compression proofs.
4) Locate any existing codebook, cassette, cortex, semantic index artifacts that should be reused.

Phase 1: Specs
5) Write GOV_IR_SPEC.md first. Keep it minimal and typed.
6) Write SPC_SPEC.md next, referencing GOV_IR_SPEC.md for concept_unit and canonical IR.
7) Write CODEBOOK_SYNC_PROTOCOL.md next, referencing SPC_SPEC.

Phase 2: Implementation
8) Implement canonical IR normalizer and exact-match comparator.
9) Implement tokenizer atlas generator and produce TOKENIZER_ATLAS.json.
10) Implement deterministic decode core:
   - Start with HASH_PTR (lowest ambiguity).
   - Then add SYMBOL_PTR by mapping symbol to hash or canonical IR id via codebook.
   - Then add COMPOSITE_PTR if needed (typed qualifiers only).

Phase 3: Benchmark plus proof harness
11) Build benchmark cases:
   - Choose statements from real repo constraints and invariants.
   - Provide gold canonical IR for each.
12) Implement proof runner:
   - Measure tokens(NL) vs tokens(pointer payload).
   - Decode pointer to IR.
   - Compute ECR and concept_units.
   - Emit metrics.json, report.md, receipts.

Phase 4: Tests plus gates
13) Add unit tests:
   - IR normalization stability
   - decode determinism
   - fail-closed cases
14) Add integration test:
   - runs proof harness end-to-end with firewall active (if repo supports it).

Phase 5: Research doc
15) Write PAPER_SPC.md from measured results only.
   - No claims without metrics.
   - Include a benchmark table and a small tradeoffs section.

Reporting
- Produce a final report that lists:
  - All created/modified files
  - Exact commands to reproduce
  - Key metrics: average token reduction, ECR, reject rate
  - Any unresolved TODOs

Scope guardrails
- Do not change unrelated systems (MCP, pipelines, cortex) unless the proof harness requires a small utility reuse.
- Do not invent new global governance constructs. Use existing repo patterns and roots.

Stop conditions
- If any acceptance criteria cannot be met, stop and write a failure report with:
  - what failed
  - why it failed
  - the smallest next step to fix it
  - no speculation
```

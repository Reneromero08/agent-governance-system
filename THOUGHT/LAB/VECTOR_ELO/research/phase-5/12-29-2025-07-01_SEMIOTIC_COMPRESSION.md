---
uuid: 00000000-0000-0000-0000-000000000000
title: SEMIOTIC_COMPRESSION
section: research
bucket: 2025-12/Week-01
author: System
priority: Medium
created: 2025-12-29 07:01
modified: 2026-01-06 13:09
status: Active
summary: Legacy research document migrated to canon format
tags:
- research
- legacy
hashtags: []
---
<!-- CONTENT_HASH: 89b7612e226e986446d1acce66d7c56d6489ebd0da999561c59556bfb9a6b888 -->

# Semiotic Compression Layer (SCL)

> **Document Hash**: `SHA256:SCL_SPEC_V1`
> **Canonical Location**: `CATALYTIC-DPT/LAB/RESEARCH/SEMIOTIC_COMPRESSION.md`
> **Supersedes**: `SEMIOTIC_COMPRESSION_LAYER_REPORT.md`, `SEMIOTIC_COMPRESSION_ROADMAP_PATCH.md`
> **Status**: Design spec + roadmap

---

## 1. One-Line Summary

A **macro language + compiler**: big models emit *very short* symbolic programs; deterministic tools expand them into full structured work; tiny models (optional) can operate on the symbolic layer as strict "lever pullers."

---

## 2. Why SCL (Problem Statement)

**Problem**: Frontier models burn tokens restating the same governance/procedure text repeatedly.

**Solution**: Replace repetitive governance prose with a compact **symbolic IR** that expands deterministically into JSON JobSpecs, code stubs, or tool calls.

### Symbols vs Hashes
- **Hashes**: Identity pointers to bytes. "Don't paste big files."
- **Symbols**: Semantic macros for meaning. "Don't restate governance forever."

**You want both**:
- Hashes to avoid injecting bodies
- Symbols to avoid injecting boilerplate semantics

---

## 3. Design Goals

1. **90%+ token reduction** for governance/procedure repetition
2. **Deterministic expansion** (same symbols → same output)
3. **Verifiable** (schema-valid outputs; hashes for artifacts; ledgered receipts)
4. **Human-auditable** (expand-to-text for review)
5. **Composable** (small primitives combine into complex intents)
6. **Narrow first** (AGS + Cat-DPT core), then generalize

---

## 4. Scope (Phase 1)

**In scope**:
- Governance rules / invariants
- JobSpec assembly (schema-valid JSON)
- Tool-call macros (CAS put/get, scan roots, diff, validate, ledger write)
- Code addressing by symbolic address + hash pointer

**Not Phase 1**:
- Compress arbitrary tokens losslessly
- Compress full code bodies beyond simple addressing

---

## 5. Deliverables

### Artifacts
| File | Purpose |
|------|---------|
| `SCL/CODEBOOK.json` | Symbol dictionary: symbol → meaning → expansion templates |
| `SCL/GRAMMAR.md` | Syntax rules (EBNF-ish) + examples |
| `SCL/decode.py` | Symbolic IR → expanded form (JSON JobSpec, audit text) |
| `SCL/validate.py` | Validates symbolic program + expanded output |
| `SCL/encode.py` | Natural text → Symbolic IR (optional, heuristic) |
| `SCL/tests/fixtures/` | Paired examples with expected results |

### CLI
```bash
scl decode <program>    # → JobSpec JSON
scl validate <job.json> # → PASS/FAIL
scl run <program>       # → execute + prove invariants
```

---

## 6. Execution Pipeline

```
1. Big model outputs Symbolic IR program (short)
         ↓
2. Decoder expands into:
   - JSON JobSpec(s)
   - Tool-call plan
   - Natural-language audit rendering
         ↓
3. Validator checks:
   - Symbolic syntax OK
   - Expanded JSON passes schema
   - Outputs in allowed roots
         ↓
4. If fail: error vector → prompt repair
```

---

## 7. Symbolic IR Shape

Use **compact ASCII-first** notation for tokenizer safety:

```
@LAW>=0.1.0 & !WRITE(authored_md)
JOB{scan:DOMAIN_WORKTREE, validate:JOBSPEC, ledger:append}
CALL.cas.put(file=PATH)
```

Optional pretty rendering for humans:
```
⚖️≥0.1.0 ∧ ◆📝❌
```

**Rationale**: ASCII is more predictable across model tokenizers.

---

## 8. Tiny Models (Without Fantasy)

Tiny models should NOT invent meaning. They should:
- Select from a finite set of macros
- Fill slots under validation
- Retry until schema-valid

**Practical approach**:
- Tiny model emits Symbolic IR or JobSpec JSON under hard validator
- Deterministic tools do the rest

---

## 9. Metrics

Track from day one:
- Tokens sent to frontier models per task (baseline vs SCL)
- Schema pass rate (% first try; % after N retries)
- Expansion determinism (hash of expanded output stable)
- Latency overhead (decode+validate time)
- Human audit success (% correct expansions)

---

## 10. Roadmap (Phase 1.6)

### Deliverables
- [ ] `SCL/CODEBOOK.json` symbol dictionary
- [ ] `SCL/GRAMMAR.md` syntax rules + examples
- [ ] `SCL/decode.py` symbolic IR → expanded JSON + audit
- [ ] `SCL/validate.py` symbolic/schema validation
- [ ] `SCL/tests/` paired fixtures
- [ ] `scl` CLI: decode, validate, run

### Tests
- [ ] Determinism: same program → same JSON hash
- [ ] Schema: expanded JobSpecs validate
- [ ] Negative fixtures: invalid programs fail clearly
- [ ] Token benchmark: show reduction vs baseline

### Success Demo
- [ ] Real AGS/Cat-DPT task completed using <20 tokens of symbolic instructions + deterministic expansion, without re-pasting governance text

---

## 11. Build Plan

### Step A: Freeze MVP Macro Set
Pick 30-80 macros covering 80% of repeated governance:
- Immutability constraints
- Allowed domains/roots
- Schema validate
- Ledger append
- CAS put/get
- Root scan/diff
- Expand-by-hash read requests

### Step B: Implement Decoder First
Decoder is deterministic and unlocks the whole loop:
- Parse symbolic program
- Expand into JobSpec JSON and/or audit form
- Validate

### Step C: Add Encoder Only If Needed
Encoder can be heuristic, or "human writes symbols" initially.

---

## 12. Integration Notes

- Implementable as skill modules: `symbolic_decode`, `symbolic_validate`, `symbolic_expand`
- Symbolic layer is **control plane**; expansions/tool outputs are **data plane**
- Hash-based externalization (CAS/SPECTRUM) stays as-is; SCL is additional compression for repeated semantics

---

## 13. Exit Criteria

Phase 1 is complete when:
1. `scl decode <program>` → emits JobSpec JSON
2. `scl validate <job.json>` → PASS/FAIL
3. `scl run <program>` → executes deterministic tool calls
4. Meaningful token reduction demonstrated
5. Reproducible expansions
6. Audit rendering for humans
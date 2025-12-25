# Semiotic Compression Layer (SCL) Report
**Status:** design spec + near-term execution plan  
**Purpose:** cut frontier-model token burn by replacing repetitive governance/procedure prose with a compact **symbolic IR** that expands deterministically into JSON JobSpecs, code stubs, or tool calls.

---

## 0. What this is (in one line)
A **macro language + compiler**: big models emit *very short* symbolic programs; deterministic tools expand them into full structured work; tiny models (optional) can operate on the symbolic layer as a strict “lever puller.”

---

## 1. Key definitions
### IR (Intermediate Representation)
In compiler terms: a stable, structured “middle language” between a high-level language (natural text) and the executable form (code / JSON / tool calls).  
Here: **Symbolic IR** = token-cheap instruction sequence that is reversible and machine-checkable.

### Symbolic IR vs hashes
- **Hashes**: identity pointers to bytes (externalization). Great for “don’t paste big files.”
- **Symbols**: semantic macros for repeated *meaning* (“validate schema”, “write ledger entry”, “no authored markdown edits”). Great for “don’t restate the same governance forever.”

You want both:
- hashes to avoid injecting bodies
- symbols to avoid injecting boilerplate semantics

---

## 2. Design goals
1. **90%+ token reduction** for governance/procedure repetition (where redundancy is high).
2. **Deterministic expansion** (same input symbols => same expanded output).
3. **Verifiable** (schema-valid outputs; hashes for artifacts; ledgered receipts).
4. **Human-auditable** (expand-to-text for review).
5. **Composable** (small primitives combine into complex intents).
6. **Narrow first** (AGS + Cat-DPT core), then generalize.

---

## 3. Scope (Phase 1 target)
Start where redundancy is extreme and correctness is testable:
- **Governance rules / invariants**
- **JobSpec assembly** (schema-valid JSON)
- **Tool-call macros** (CAS put/get, scan roots, diff, validate, ledger write)
- **Code addressing**: refer to code by *symbolic address* + hash pointer, not by pasting code

Not Phase 1:
- “compress arbitrary tokens losslessly”
- “compress full code bodies into symbols” beyond simple addressing + expansion hooks

---

## 4. The pipeline (minimal viable)
### 4.1 Artifacts
- `SCL/CODEBOOK.json`  
  Symbol dictionary: symbol -> canonical meaning -> expansion template(s).
- `SCL/GRAMMAR.md`  
  Syntax rules (EBNF-ish) + examples.
- `SCL/encode.py`  
  Natural text -> Symbolic IR (optional, heuristic).
- `SCL/decode.py`  
  Symbolic IR -> expanded form (JSON JobSpec, or “expanded natural text”).
- `SCL/validate.py`  
  Validates symbolic program + validates expanded output against schemas.
- `SCL/tests/fixtures/*`  
  Paired examples: (natural) <-> (symbols) <-> (expanded JSON) with expected results.

### 4.2 Execution loop
1. Big model outputs a **Symbolic IR program** (short).
2. Deterministic decoder expands it into:
   - JSON JobSpec(s)
   - tool-call plan
   - optional natural-language audit rendering
3. Local validator checks:
   - symbolic syntax OK
   - expanded JSON passes schema
   - outputs exist / are in allowed roots
4. If fail: error vector -> prompt repair (or tiny learner loop later).

---

## 5. Symbolic IR shape (recommendation)
Use a **compact ASCII-first** notation for stability and tokenizer safety, with optional Unicode “pretty” layer for humans.

Example (ASCII-first):
- `@LAW>=0.1.0 & !WRITE(authored_md)`
- `JOB{scan:DOMAIN_WORKTREE, validate:JOBSPEC, ledger:append}`
- `CALL.cas.put(file=PATH)`

Then optionally render to “pretty symbols” in UI:
- `⚖️≥0.1.0 ∧ ◆📝❌`

Why: tokenizer behavior across models is more predictable with ASCII, and you can still map to icons in the human UI.

---

## 6. Where tiny models fit (without fantasy)
Tiny models should not be asked to invent meaning. They should:
- select from a finite set of macros
- fill slots under validation
- retry until schema-valid

This aligns with the “microscopic orchestrator emits strict JSON” idea in Cat-DPT research.

Practical approach:
- Tiny model emits **Symbolic IR** or **JobSpec JSON** under a hard validator.
- Deterministic tools do the rest.

---

## 7. Metrics (make it real)
Track from day one:
- tokens sent to frontier models per task (baseline vs SCL)
- schema pass rate (% first try; % after N retries)
- expansion determinism (hash of expanded output stable)
- latency overhead (decode+validate time)
- human audit success (% correct expansions)

---

## 8. Immediate build plan (next)
### Step A: freeze an MVP “macro set”
Pick 30-80 macros that cover 80% of your repeated governance/procedure talk:
- immutability constraints
- allowed domains/roots
- schema validate
- ledger append
- CAS put/get
- root scan / diff
- “expand-by-hash read” requests

### Step B: implement decoder first
Decoder wins because it is deterministic and unlocks the whole loop.
- parse symbolic program
- expand into JobSpec JSON and/or natural audit form
- validate

### Step C: add encoder only if needed
Encoder can be heuristic, or just “human writes symbols” initially.

---

## 9. What I need from you (inputs)
Provide these as concrete artifacts (copy/paste is fine):
1. **Top 25 repeated governance statements** you keep paying tokens to restate.
2. The **first 10 macro intents** you want to express in 5-20 tokens.
3. The **expansion targets** for each macro (JobSpec JSON fields, or code templates).
4. Your **non-negotiable invariants** (things that must hard-fail).
5. A “success demo” definition (one task you consider the breakthrough).

---

## 10. Integration notes
- This is perfectly implementable as a “skill” module: `symbolic_decode`, `symbolic_validate`, `symbolic_expand`.
- The symbolic layer becomes part of the **control plane**; expansions/tool outputs remain in the data plane.

---

## 11. Exit criteria for Phase 1
You can run:
- `scl decode <program>` -> emits JobSpec JSON
- `scl validate <job.json>` -> PASS/FAIL
- `scl run <program>` -> executes deterministic tool calls and proves invariants
With:
- meaningful token reduction
- reproducible expansions
- audit rendering for humans

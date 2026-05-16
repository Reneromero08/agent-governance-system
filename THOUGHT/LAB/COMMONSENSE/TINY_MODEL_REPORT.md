---
title: COMMONSENSE as the Tiny Model - Complete Report
version: 2.1.0
last_updated: 2026-05-16
author: Agent (opencode)
status: Report
supersedes: v2.0.0
---

# COMMONSENSE as the Tiny Model - Complete Report

## 1. Original Concept: TINY_COMPRESS Lab

The "tiny model" began as a proposal in `THOUGHT/LAB/TINY_COMPRESS/`:

**Goal** (`roadmaps/TINY_COMPRESS_ROADMAP.md:10`):
> "Train a tiny model (10M-50M params) to learn **symbolic compression** without understanding meaning."

**Method**: RL training against a Validator reward signal. The model emits valid symbols by trial and error -- it never needs to understand *why* they are valid, only *that* they are. Vision: "a model that inputs intent (text) and outputs compressed Symbolic IR."

Three research lines emerged under TINY_COMPRESS:

| Lab | Achievement |
|-----|-------------|
| `llm-spectral/` | GPT-2 eigen/KV compression (5x KV cache compression on attention layers) |
| `holographic-image/` | Image VQ compression with `.holo` format (**30x smaller than JPEG**, 29.3 dB quality) |
| **`canon-symbol/`** | H(X\|S) symbol compression of AGS canon (**22.3x**, 476x for individual file refs) |

## 2. The Pivot: canon-symbol (`TINY_COMPRESS/canon-symbol/`)

The practical implementation dropped the "train a neural net" idea and instead used **deterministic SHA-256 hashing** to achieve the same end.

**Theory** (`CANON_COMPRESSION_RESULTS.md:23-28`):
> H(X|S) = H(X) - I(X;S)
> When both parties share context S (the canon): I(X;S) ≈ H(X), therefore H(X|S) ≈ 0

**Result**: 216 KB of canon -> 9.7 KB manifest = **22.3x lossless compression**.

**Mechanism**:
- `canon_compressor.py` scans `LAW/CANON/`, SHA-256 hashes each `.md` file, generates **@C symbols** (e.g., `@C:85bc78171225`)
- `symbol_resolver.py` resolves `@C:hash` back to file path + content, with hash verification
- Compression ratio for individual file references: **476x** (16-byte symbol replaces 7,595 bytes of FORMULA.md)

**Key insight**: Compression relied on **shared context** (both parties have the canon locally). The symbol is just a pointer. No neural network needed.

## 3. Generalization: CODIFIER.md (`THOUGHT/LAB/VECTOR_ELO/CODIFIER.md`)

The symbol compression concept was generalized into a **two-layer symbolic vocabulary**:

| Layer | Scope | Example | Compression |
|-------|-------|---------|-------------|
| **CJK** | Domain pointers |  (fa) -> `LAW/CANON/*` | 56,370x |
| **ASCII** | Rule precision | `C3` -> Contract Rule 3 | ~2,000x |

The Codifier is the explicit bridge document. At line 148:

> "The macro codebook (`THOUGHT/LAB/COMMONSENSE/CODEBOOK.json`) uses compact notation:
> Grammar: RADICAL[OPERATOR][NUMBER][:CONTEXT]"

It defines 10 ASCII radicals (C/I/V/L/G/S/R/J/A/P), 7 operators (*/!/?/&/|/./:), and shows token savings of 60-80% (e.g., `@CONTRACT_RULE_3` at 6 tokens -> `C3` at 2 tokens).

## 4. The Obsidian Vault Thread (Reneshizzle)

The user's personal Obsidian vault (`Shizzle`) contains the **origin notes** that drove this entire trajectory.

### 4.1 The Origin Brainstorm

`AGI/AI Chats/Common Sense Tiny Model/GPT 5.2 Logic - Common Sense DB.md` -- a conversation with GPT-5.2 about building a **Common Sense DB** of formal logic concepts. It covers non-monotonic logic, AGM belief revision, inductive inference, and using tiny models (1B-8B) for structured extraction with schema-constrained output and citation binding.

### 4.2 The Master Todo

`AGI/AGS/ AGS Notes.md` -- the user's master research document. Key entries:

> "- [ ] **Teach common sense / logic and go from there.** Philosophy -> Semiotics first -> Psychology -> How the mind works"
> "- [ ] Create tiny models that train themselves with my formula."
> "- [ ] Make it mcp so that tiny model is constantly translating"
> "- [ ] Enhance the Tiny_Compress roadmap"
> "- [ ] Teach a tiny model to use the vector db and test benchmarks like this. (lilq)"

And the mechanical insight that drove the whole pivot:

> "I don't think we need small models for translating db, I think it can literally just be mechanical. even mcp can activate py right?"

### 4.3 The META_LOGIC Spine (Fully Expanded)

`THOUGHT/LAB/COMMONSENSE/META_LOGIC.md` (renamed from `META_LOGIC`) -- a 1,923-line file that defines the entire formal logic knowledge base concept as a **meta-logic spine** with concrete operators, database primitives, research pipeline, and formal acceptance tests:

**11 nodes, 3 load-bearing loops:**
| Node | Name | Core Function |
|------|------|--------------|
| A | Representation & Predicate Governance | What statements are allowed |
| B | Inference Operators | Deduction, induction, abduction, analogy |
| C | Defeasible Reasoning (Defaults) | "Normally/unless" engine |
| D | Non-Monotonicity | Adding info can remove conclusions |
| E | Induction & Projectibility | Why some patterns generalize |
| F | Uncertainty & Degrees of Belief | Credences, confidence |
| G | Preference, Simplicity & Compression | Occam, MDL |
| H | Belief Revision & Consistency Maintenance | AGM contraction, revision |
| I | Coherence & Explanation | Global constraint satisfaction |
| J | Normativity & Bounded Rationality | Heuristics, stopping rules |

**Three loops (the real spine):**
1. **Projectibility Loop** A -> E -> G -> A. (Why "birds fly" feels legitimate)
2. **Default-Revision Loop** C -> D -> H -> C. (How "usually" survives exceptions)
3. **Explanation-Selection Loop** Abduction -> I -> G -> Abduction. (Why some explanations feel like common sense)

**Tiny model strategy (expanded in spine Section 6):**
| Size | Good for | Failure rate on logic tasks |
|------|----------|---------------------------|
| <1B | Routing, tagging, light classification (e.g., identifying postulate-bearing chunks) | ~40% |
| 1B-3B | Triage, coarse topic bucketing, schema-constrained extraction with verification | ~20% |
| 3B-8B | Structured extraction (definitions, postulates, examples) under hard schema | ~10% |
| >8B | Synthesis, cross-source compression, narrative polishing | ~5% |

Key principles from the expanded spine:
- "Never trust a tiny model with formal logic transcription -- copy verbatim or verify."
- Chunking: one conceptual unit per chunk, formal statements never split across chunks
- Verification: every `formal_statement` checked character-by-character against source text
- Online models: input-gated, output-labeled, no new claims, human-in-the-loop

### 4.4 LIL_Q Empirical Tests

`THOUGHT/LAB/LIL_Q/test_sandbox/` contains **empirical validation**: tiny models (qwen2.5-coder:3b, 0.5B) tested with vector context. Results claimed "Quantum entanglement with context enabled tiny model across ALL domains!" -- baseline tiny model fails without context, succeeds with E-gated vector retrieval.

This directly validates the META_LOGIC tiny model strategy: tiny models as structured extractors under hard schema, with larger models for synthesis.

## 5. The Destination: COMMONSENSE (`THOUGHT/LAB/COMMONSENSE/`)

The "tiny model" completed its evolution from **ML neural net -> deterministic symbolic rule engine**.

### 5.1 CODEBOOK.json (v1.0.0)

The compact macro vocabulary, upgraded from v0.2.0:
- 6 **symbol macros** in new `"symbols"` key for `@`-handle expansion (e.g., `@DOMAIN_GOVERNANCE` → `["domain:governance"]`)
- 10 **radicals** mapping to AGS domains (C=Contract, I=Invariant, V=Verification, etc.)
- 7 **operators** for compound expressions (ALL, NOT, CHECK, AND, OR, PATH, CONTEXT)
- 13 **contract rules** (C1-C13) with summary and full text
- 20 **invariants** (I1-I20, mapping to INV-001 through INV-020)
- 5 **context tags** (build, audit, security, execute, validate)
- 6 **legacy macros** with deprecation mappings (pointing to new `symbols` entries)
- Token savings measured at **80%** (verbose -> compact)

**Key fix from v0.2.0**: The v0.2.0 codebook had legacy macros under a `legacy` key but no `symbols` key. `translate.py` reads from `codebook["symbols"]`, which didn't exist, causing Phase 2 to fail. v1.0.0 adds the `symbols` key with proper predicate-list mappings.

### 5.2 resolver.py v1.0.0 -- The Tiny Model's "Brain"

A deterministic inference engine with:

1. **Symbol expansion** (`translate.py`): `@`-prefixed facts are expanded against the codebook
2. **Scope matching**: Each entry has `scope.applies_when` (all required) and `scope.not_when` (any veto disables)
3. **Sorting**: Candidates sorted by `priority desc -> confidence desc -> id asc` (fully deterministic)
4. **Logic rule evaluation**: Rules have `if_all`/`if_any`/`unless` conditions with `set`/`unset`/`emit` effects
5. **Output**: `selected_ids`, `derived_facts`, `emits`, `expanded_facts`, `unresolved_symbols`

### 5.3 Entry Types

| Kind | Purpose | From db.example.json |
|------|---------|---------------------|
| `commonsense` | Stored prior/knowledge | "Default trust is low for unverified outputs" (CS_DEFAULT_TRUST_LOW) |
| `logic_rule` | Executable if/then rule | "If invariant violated -> emit hard_fail" (LR_CONFLICT_HARD_FAIL) |

### 5.4 Current Build State

- **Phase 0 (Schema validation)**: PASS -- 10 valid + 10 invalid fixtures
- **Phase 1 (Basic resolver)**: PASS -- 5 fixture sets, schema-valid + expectations
- **Phase 2 (Symbolic expansion)**: PASS -- `@` symbols expand to predicates via `codebook["symbols"]`, rules select and emit correctly

**Root cause (resolved)**: CODEBOOK.json v0.2.0 had legacy macros under a `legacy` key, but `translate.py` reads from `codebook["symbols"]` which didn't exist. v1.0.0 adds the `"symbols"` key with 6 mappings (`@DOMAIN_GOVERNANCE` → `["domain:governance"]`, `@INVARIANT_VIOLATION` → `["violation:invariant"]`, etc.). The legacy entries remain with `deprecated: true` and `note` fields pointing to the new symbols.

### 5.5 META_LOGIC Spine Alignment

The resolver currently implements **node B (Inference Operators)** and partially **node C (Defeasible Reasoning)** from the META_LOGIC graph. The spine itself (Section 4) now provides full formal specifications for all four core operators (defeasibility, revision, projectibility, coherence) plus detailed research pipeline, model strategy, and acceptance tests. What the resolver still needs to implement:

| META_LOGIC Node | In resolver? | Spine provides | What's needed in code |
|-----------------|--------------|---------------|----------------------|
| A (Representation) | Partially | Natural kind test, predicate typing (4C, C3) | `predicate_schema` validation |
| B (Inference Operators) | Yes | Deduction, induction, abduction, analogy specs | `if_all`/`unless` covers basic deduction |
| C (Defeasible Reasoning) | Partially | Specificity ordering, priority lattice (4A, D1-D4) | Structural specificity computation |
| D (Non-Monotonicity) | No | Default logic, truth maintenance (4A, D2) | Justification tracking, retraction |
| E (Induction) | No | Predicate entrenchment, grue filter (4C) | Projectibility gate, induction engine |
| F (Uncertainty) | No | Ranking theory, calibration (Sec 3, item 6) | Confidence update, calibration check |
| G (Preference/Compression) | No | MDL, simplicity scoring (Sec 3, item 7) | Description length computation |
| H (Belief Revision) | No | AGM postulates, entrenchment ordering (4B) | Contraction/revision operators |
| I (Coherence) | No | IBE, Thagard network, 6 virtues (4D) | Hypothesis scoring, coherence network |
| J (Normativity) | No | Satisficing, metareasoning (Sec 3, item 10) | Stopping rules, computational budget |

## 6. What's Missing from the Total Vision

The COMMONSENSE resolver + META_LOGIC spine together still leave out several threads from the user's repo and vault:

### 6.1 The Formula

`R = (E/∇S) × σ(f)^Df` -- the user's foundational equation. AGS Notes say "Create tiny models that train themselves with my formula." META_LOGIC uses Bayes/MDL as scoring policies but doesn't connect to the Df/eigenvalue formula or the Platonic Compression Thesis. The `THOUGHT/LAB/FORMULA/` directory contains the research.

### 6.2 Semiotics / Platonic Compression

The `CODIFIER.md` and `PLATONIC_COMPRESSION_THESIS.md` argue that compression comes from **shared semantic space** (55,625x via CJK pointing), not just formal logic rules. The META_LOGIC spine treats language as given; the codebook treats it as compressible by pointing. This is a **separate compression axis**: formal logic (what rules to follow) vs. semiotic pointing (how to refer efficiently).

### 6.3 Psychology / Cognitive Science

AGS Notes say "Psychology -> How the mind works." META_LOGIC is purely **normative/logical** (how reasoning *should* work). It doesn't address **descriptive/cognitive** models (how humans *actually* reason -- heuristics, biases, prototypes, typicality effects from Rosch, etc.). The `THOUGHT/LAB/FERAL_RESIDENT/perception/research/papers/manifest.json` indexes cognitive science papers.

### 6.4 Feral Resident / Vector Resident

The `THOUGHT/LAB/FERAL_RESIDENT/` and `THOUGHT/LAB/VECTOR_ELO/` directories implement the **runtime** that runs on the spine -- vector memory, ELO scoring, resident daemon, particle smasher. META_LOGIC defines the reasoning spine; Feral Resident is the running system that uses it. They are complementary but not connected in documentation.

### 6.5 The Swarm Layer

The roadmap's Phase 2 (Swarm) -- tiny models (0.5B-3B) as ant workers under a Governor -- is the **deployment** architecture for the tiny model concept. META_LOGIC defines *what* the tiny model knows; the Swarm defines *how* it executes. Not yet implemented.

### 6.6 LIL_Q (Quantum Rescue)

`THOUGHT/LAB/LIL_Q/test_sandbox/` -- empirical tests showing tiny models (qwen2.5-coder:3b) can solve problems with E-gated vector context that they fail without. This is the **empirical validation** of the tiny model strategy that META_LOGIC only theorizes.

## 7. Usage in Production

COMMONSENSE is referenced from **production governance tools**:

| File | Usage |
|------|-------|
| `CAPABILITY/TOOLS/codebook_lookup.py` | Loads `CODEBOOK.json` for symbolic lookup |
| `NAVIGATION/CORTEX/network/cassette_protocol.py` | Loads codebook for cassette sync |
| `NAVIGATION/CORTEX/network/spc_decoder.py` | Loads codebook for SPC decoding |
| `NAVIGATION/CORTEX/network/spc_integration.py` | Codebook path for SPC integration |
| `CAPABILITY/PRIMITIVES/scl_validator.py` | Loads codebook for SCL validation |
| `LAW/CANON/SEMANTIC/SPC_SPEC.md` | References CODEBOOK.json as active codebook |
| `LAW/CANON/SEMANTIC/CODEBOOK_SYNC_REFERENCE.md` | Codebook artifact reference |

## 8. The Full Evolutionary Arc

```
2025-12:  TINY_COMPRESS lab proposal
          "Train a tiny model (10M-50M params) via RL for symbolic compression"
              |
2025-12:  Obsidian vault brainstorm
          User writes: @C1, @C2 codebook approach in symbolic compression notes
          User writes: "Teach common sense / logic"
          GPT-5.2 conversation: Common Sense DB of formal logic concepts
              |
2026-01:  canon-symbol/ implementation
          "No ML needed -- SHA-256 hashing achieves 22.3x compression on canon"
          "Shared context (H(X|S) ≈ 0) replaces learned compression"
              |
2026-01:  CODIFIER.md generalization
          "Two-layer symbolic compression: CJK (55,625x) + ASCII (~2,000x)"
          "ASCII macros link to THOUGHT/LAB/COMMONSENSE/CODEBOOK.json"
              |
2026-01:  COMMONSENSE v0.2.0
          "The 'tiny model' is not a neural net -- it's a deterministic symbolic
          rule engine with fact expansion, scope matching, and logic rules"
              |
2026-01:  LIL_Q empirical tests
          "Tiny model (3B) fails without context, succeeds with E-gated vectors"
          "Quantum rescue" -- empirical validation of tiny model + vector context
              |
 2026-05:  META_LOGIC spine expanded (v2.1.0)
           1,923-line full formal specification across 8 sections
           Concrete operators: defeasibility (4A), belief revision (4B),
           projectibility (4C), explanatory coherence (4D)
           9 database primitives with full YAML schemas
           Research pipeline: 5-stage extraction with per-stage schemas
           6 formal acceptance gates, 10 test fixtures
           Gap analysis vs current COMMONSENSE implementation
               |
 2026-05:  CODEBOOK.json v1.0.0 + resolver.py v1.0.0
           Fixed Phase 2: added "symbols" key with 6 predicate mappings
           translate.py now correctly expands @-handles to predicates
           All 3 test phases pass (Phase 0, 1, 2)
           Legacy macros preserved with deprecation notes
               |
 2026-05:  _v1/ archive created (v0.2.0 files preserved)
           Old CODEBOOK, fixtures, schemas, testbench moved to _v1/
           New versions at root with fixes applied
               |
PRESENT:  COMMONSENSE resolver functional (all phases pass)
           META_LOGIC defines full spine with formal operator specs
           Resolver implements node B + partial C
           Missing in code: specificity, AGM revision, induction,
           coherence, formula connection, cognitive layer, swarm
```

## 9. Compression Analysis

| Artifact | Input | Output | Ratio | Type |
|----------|-------|--------|-------|------|
| TINY_COMPRESS holographic-image | 2,165 KB JPEG | 72 KB .holo | 30x | Image VQ |
| TINY_COMPRESS canon-symbol | 216 KB canon | 9.7 KB manifest | 22.3x | H(X|S) hashing |
| CODIFIER.md CJK symbols | 56,370 tokens canon | 1 token CJK | 56,370x | Semantic pointing |
| CODIFIER.md ASCII macros | ~2,000 tokens rule | 2 tokens ASCII | ~2,000x | Rule compression |
| META_LOGIC spine (expanded) | ~148-line sketch | 1,923-line full spec | — | Meta-logic formalization with operators, primitives, pipeline, gates |
| COMMONSENSE CODEBOOK v1.0.0 | v0.2.0 (missing `symbols` key) | v1.0.0 (6 symbol macros, Phase 2 fix) | — | Symbolic expansion wired to translate.py |

## 10. Conclusion

The "tiny model" concept evolved through **three distinct compression strategies**:

1. **Neural** (TINY_COMPRESS proposal): Train a 10M-50M param RL model -- never built
2. **Hashing** (canon-symbol): H(X|S) = 0 via shared context -- built, 22.3x
3. **Symbolic** (CODIFIER + COMMONSENSE): Pointers into shared semantic space -- built, up to 56,370x

The META_LOGIC spine adds a **fourth dimension**: defining *what* common sense IS at the operator level (11 nodes, 3 loops, 20 concrete operators, 9 database primitives, 6 formal acceptance gates). The spine is now a complete 1,923-line formal specification covering defeasibility (specificity ordering, Reiter default logic), belief revision (AGM postulates, entrenchment ordering), projectibility (natural kind test, grue filter, predicate entrenchment), and explanatory coherence (IBE scoring, Thagard network).

**COMMONSENSE IS the tiny model.** It is symbolic, not neural. The resolver executes the rules (all 3 phases pass); META_LOGIC defines what those rules should be. The remaining gaps (specificity computation, AGM revision, induction engine, coherence scoring) are specified in full formal detail and ready for implementation.

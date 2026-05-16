---
title: COMMONSENSE as the Tiny Model - Complete Report
version: 2.0.0
last_updated: 2026-05-16
author: Agent (opencode)
status: Report
supersedes: v1.0.0
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

### 4.3 The META_LOGIC Spine (Added Compression)

`THOUGHT/LAB/COMMONSENSE/META_LOGIC` -- a 148-line file that compresses the entire formal logic knowledge base concept into a **meta-logic spine**:

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

**Tiny model strategy:**
| Size | Good for |
|------|----------|
| <1B | Routing, tagging, light classification |
| 1B-3B | Triage, coarse topic bucketing |
| 3B-8B | Structured extraction (schema-constrained) |
| >8B | Synthesis, cross-source compression |

Key principle: "Never trust a tiny model with formal logic transcription -- copy verbatim or verify."

### 4.4 LIL_Q Empirical Tests

`THOUGHT/LAB/LIL_Q/test_sandbox/` contains **empirical validation**: tiny models (qwen2.5-coder:3b, 0.5B) tested with vector context. Results claimed "Quantum entanglement with context enabled tiny model across ALL domains!" -- baseline tiny model fails without context, succeeds with E-gated vector retrieval.

This directly validates the META_LOGIC tiny model strategy: tiny models as structured extractors under hard schema, with larger models for synthesis.

## 5. The Destination: COMMONSENSE (`THOUGHT/LAB/COMMONSENSE/`)

The "tiny model" completed its evolution from **ML neural net -> deterministic symbolic rule engine**.

### 5.1 CODEBOOK.json (v0.2.0)

The compact macro vocabulary:
- 10 **radicals** mapping to AGS domains (C=Contract, I=Invariant, V=Verification, etc.)
- 7 **operators** for compound expressions (ALL, NOT, CHECK, AND, OR, PATH, CONTEXT)
- 13 **contract rules** (C1-C13) with summary and full text
- 20 **invariants** (I1-I20, mapping to INV-001 through INV-020)
- 5 **context tags** (build, audit, security, execute, validate)
- 6 **legacy macros** with deprecation mappings
- Token savings measured at **60%** (verbose -> compact)

### 5.2 resolver.py -- The Tiny Model's "Brain"

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
- **Phase 2 (Symbolic expansion)**: FAIL -- `@` legacy symbols resolve as "unknown_symbol", no entries selected

**Root cause**: CODEBOOK.json has legacy macros under a `legacy` key, but `translate.py` only reads from `codebook["symbols"]` which doesn't exist. The expansion pipeline was never wired up.

### 5.5 META_LOGIC Spine Alignment

The resolver currently implements **node B (Inference Operators)** and partially **node C (Defeasible Reasoning)** from the META_LOGIC graph. Missing:

| META_LOGIC Node | In resolver? | What's needed |
|-----------------|--------------|---------------|
| A (Representation) | Partially | Scope/predicate system exists, no formal typing |
| B (Inference Operators) | Yes | |if_all|/|unless| covers basic deduction |
| C (Defeasible) | Partially | Priority/confidence sorting, but no exception hierarchy |
| D (Non-Monotonicity) | No | No circumscription or default priority system |
| E (Induction) | No | No projectibility or similarity |
| F (Uncertainty) | No | Confidence is static, not updated |
| G (Preference/Compression) | No | No MDL or simplicity |
| H (Belief Revision) | No | No AGM contraction/revision |
| I (Coherence) | No | No global constraint satisfaction |
| J (Normativity) | No | No stopping rules or bounded rationality |

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
2026-05:  META_LOGIC spine (added compression)
          11-node meta-logic graph, 3 load-bearing loops
          Tiny model strategy: <1B routing, 1B-3B triage, 3B-8B extraction
          Defines what common sense IS at the operator level
              |
PRESENT:  COMMONSENSE resolver built (Phase 0-1 pass, Phase 2 broken)
          META_LOGIC defines the spine; resolver implements node B + partial C
          Missing: AGM revision, induction, coherence, formula connection,
          psychology/cognitive layer, swarm deployment
```

## 9. Compression Analysis

| Artifact | Input | Output | Ratio | Type |
|----------|-------|--------|-------|------|
| TINY_COMPRESS holographic-image | 2,165 KB JPEG | 72 KB .holo | 30x | Image VQ |
| TINY_COMPRESS canon-symbol | 216 KB canon | 9.7 KB manifest | 22.3x | H(X|S) hashing |
| CODIFIER.md CJK symbols | 56,370 tokens canon | 1 token CJK | 56,370x | Semantic pointing |
| CODIFIER.md ASCII macros | ~2,000 tokens rule | 2 tokens ASCII | ~2,000x | Rule compression |
| META_LOGIC spine | 10K+ tokens GPT chat + vault | 148 lines | ~90% | Meta-logic distillation |
| COMMONSENSE resolver | Manual rule DB + codebook | Deterministic inference | Variable | Symbolic execution |

## 10. Conclusion

The "tiny model" concept evolved through **three distinct compression strategies**:

1. **Neural** (TINY_COMPRESS proposal): Train a 10M-50M param RL model -- never built
2. **Hashing** (canon-symbol): H(X|S) = 0 via shared context -- built, 22.3x
3. **Symbolic** (CODIFIER + COMMONSENSE): Pointers into shared semantic space -- built, up to 56,370x

The META_LOGIC spine adds a **fourth dimension**: defining *what* common sense IS at the operator level (11 nodes, 3 loops), providing the formal target that the resolver mechanically implements.

**COMMONSENSE IS the tiny model.** It is symbolic, not neural. The resolver executes the rules; META_LOGIC defines what those rules should be. The remaining gaps (AGM revision, induction, coherence, formula connection, cognitive layer) define the roadmap for what comes next.

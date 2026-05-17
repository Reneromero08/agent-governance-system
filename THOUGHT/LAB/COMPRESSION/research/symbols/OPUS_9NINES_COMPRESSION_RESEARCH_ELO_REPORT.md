---
title: Opus Research Pack: Beyond 9 Nines via Semantic Density + Symbolic Compression
date: 2026-01-07
status: ready_for_execution
scope: research_to_spec_to_proof
---

# 1) What your current docs already say (so Opus doesn’t fight the wrong battle)

Your stack currently has a **measured** vector-retrieval layer and **theoretical** SCL/CAS/session layers with a token-floor ceiling:

- COMPRESSION_STACK_ANALYSIS explicitly frames **~6 nines as the “physical limit”** under the assumption “1 token = 1 concept”. That doc also points to the existing reproducible token-count proof harness.  
- The paradigm-shift report explicitly identifies that assumption as the hidden flaw and reframes the goal as **semantic density (meaning per token)**, including the “1000x concepts per token ⇒ 9 nines equivalent” logic and open questions around tokenizer behavior + objective measurement.

This pack is about turning the paradigm-shift into: **(A) spec, (B) deterministic codec, (C) proof harness, (D) regression gates.**

# 2) The missing pieces that actually block “9 nines+” from becoming mechanically provable

If you want “9 nines+” to be more than a story, these are the non-optional missing artifacts.

## 2.1 Deterministic decoder contract for multiplex symbols (no interpretive decoding)
You already describe “semantic multiplexing” (one symbol activates a concept web by context). What’s missing is a **decoder contract** with hard determinism:
- Inputs: (symbol, context-key(s), codebook_version, rule_version, type_constraints)
- Output: a canonical IR subtree, or FAIL (closed).
- No LLM involvement in decoding. No “best effort”. No ambiguity.

Deliverable: **SEMANTIC_DENSITY_SPEC.md** (normative).

## 2.2 A real metric: concept-units + expansion correctness (not just token math)
Token math alone cannot justify “9 nines+” once you claim multiplexing. You need a measured semantic metric that can be regression-tested.

Minimum metric set:
- **concept_unit**: a discrete, countable unit in your governance IR (constraints, references, gates, side-effects, etc.).
- **CDR** (concept density ratio): concept_units / tokens.
- **ECR** (expansion correctness rate): exact-match rate of decoded IR vs gold IR.
- **M_required**: multiplex factor needed for “9 nines equivalent” on a declared baseline.

Deliverable: **proof_semantic_density_run/** (fixture + harness + receipts).

## 2.3 Codebook sync protocol (shared-side-information is not optional)
Semantic multiplexing is “compression with shared side-information”. If codebooks drift, decode collapses.
You need explicit negotiation + fail-closed mismatch rules:
- codebook id + sha256
- semantic kernel version
- allowed fallback behavior (usually: none, fail closed)
- compatibility window policy (optional)

Deliverable: **CODEBOOK_SYNC_PROTOCOL.md**.

## 2.4 Tokenizer atlas (engineer around tokenization, don’t guess)
Your own report flags that some Unicode symbols tokenize into 3–5 tokens. Past 9 nines you cannot waste tokens on symbol choice.
You need a generated atlas:
- candidate glyph sets (operators + primitives)
- token counts for your target tokenizer(s)
- deterministic “best glyph” selection per operator

Deliverable: **TOKENIZER_ATLAS.json** + generator script + CI check.

## 2.5 Canonical IR (typed, minimal, stable)
You can’t measure concept_units or exact-match ECR without a stable IR and stable normalization.

Deliverable: **GOV_IR_SPEC.md** + canonical JSON printer + parser.

## 2.6 Symbol/macro governance: “Symbol ELO” + tiering
You already use ELO logic for vectors. Extend it to symbols/macros:
- promote symbols that decode correctly and compress frequent subtrees
- demote ambiguous/low-utility symbols
- freeze stable tier; experiment only in a sandbox tier

Deliverable: **CODEBOOK_ELO.md** + scoring + promotion rules + receipts.

# 3) The “Opus attack plan” (tasks that directly increase semantic density)

## 3.1 Build the smallest executable semantic kernel (50–200 ops)
You do not need a huge language. You need a small kernel with compositional power and strict types.
Examples of kernel ops:
- boolean: and/or/not
- compare: ge/eq/in/matches
- refs: path_ref, canon_ref, tool_ref
- gates: require_test, require_restore_proof, allow_write_root, deny_write_root
- effect tags: read_only, mutate_allowed(paths), network_allowed(false), etc.

## 3.2 Macro library learning (dictionary induction from your own corpus)
Past 9 nines means recursion:
- **macro**: symbol → IR subtree
- **composition**: macros compose into bigger meaning than their token length implies

Two sources:
- hand-written macros for the top 50 patterns
- induced macros (MDL/DreamCoder style) from your existing canon + prompts

## 3.3 After IR: treat the IR stream like a compressible source
Once governance meaning is a discrete stream, all classic compression applies:
- grammar coding (Sequitur)
- dictionary coding (LZ family)
- arithmetic coding / ANS
- learned entropy models (optional)

This doesn’t replace your symbol system. It tells you how to shorten frequent patterns optimally.

## 3.4 Proof harness design (what “9 nines” must mean in your receipts)
A proof harness for semantic density should look like your existing token proof:
- fixed benchmark set (10–30 governance statements)
- gold IR per statement
- deterministic encode(symbolic) and decode(expand) functions
- compute tokens, concept_units, ECR, M_required
- emit receipts with hashes + stable outputs across 2 runs

# 4) Ranked source list in descending ELO (impact-weighted for your exact target)

Scoring rubric (heuristic):
- 2900–3000: foundational, field-defining
- 2700–2899: seminal methods/surveys with direct applicability
- 2500–2699: strong leverage, modern/practical
- 2300–2499: niche but frontier-relevant to your architecture

## 4.1 3000–2900: information + compression bedrock
1. (3000) Shannon (1948) A Mathematical Theory of Communication  
2. (2975) Kolmogorov (1965) Three approaches to the quantitative definition of information  
3. (2960) Solomonoff (1964) A formal theory of inductive inference  
4. (2950) Rissanen (1978) MDL: Modeling by shortest data description  
5. (2930) Wallace (1968+) Minimum Message Length (MML) program  
6. (2920) Tishby et al. (1999) The Information Bottleneck Method  

## 4.2 2899–2750: classic practical codecs (gives you optimality targets once IR is discrete)
7. (2890) Lempel & Ziv (1977/1978) LZ77/LZ78 (dictionary coding)  
8. (2860) Witten, Neal, Cleary (1987) Arithmetic coding tutorial/implementation lineage  
9. (2840) Duda (2013) Asymmetric Numeral Systems (ANS) (practical near-arithmetic)  
10. (2800) Nevill-Manning & Witten (1997) Sequitur (grammar-based compression)  
11. (2760) Hinton & Van Camp (1993) “Keeping neural networks simple by minimizing the description length…” (bits-back lineage)  
12. (2750) Townsend et al. (2019) Practical bits-back coding (modernizing bits-back)  

## 4.3 2899–2700: vector-symbol computation (your “vectorial computing” lane)
13. (2890) Smolensky (1990) Tensor Product Representations (symbol binding)  
14. (2860) Plate (1995) Holographic Reduced Representations (HRR)  
15. (2840) Kanerva (2009) Hyperdimensional computing introduction  
16. (2820) Kleyko et al. (2022) HDC/VSA Survey Part I  
17. (2790) Kleyko et al. (2021) HDC/VSA Survey Part II  

## 4.4 2699–2550: discrete latents + “semantic hashing” (closest analog to codebooks)
18. (2690) van den Oord et al. (2017) VQ-VAE (discrete codebooks)  
19. (2760) Salakhutdinov & Hinton (2009) Semantic Hashing (compact codes for semantics)  
20. (2590) Rae et al. (2019) Compressive Transformer (learned compression of memory)  

## 4.5 2680–2550: program induction as compression (macro induction)
21. (2680) Ellis et al. (2021) DreamCoder (library learning = compression)  
22. (2580) Program synthesis surveys (neural program induction, optional)  

## 4.6 2680–2500: semantic communication (shared-side-information framing for your “multiplex” claim)
23. (2680) Semantic communication surveys (goal-oriented / task-based comms, 2021–2024)  
24. (2600) Modern “semantic coding” as task-loss minimization (various)  

## 4.7 2599–2480: ANN / pointer layer compression (reduces overhead of retrieval pointers)
25. (2590) Jégou et al. (2011) Product Quantization (PQ)  
26. (2550) Malkov & Yashunin (2018) HNSW  
27. (2520) DiskANN (2019) SSD-based ANN search  
28. (2500) ScaNN (2020) Efficient vector search at scale  

## 4.8 2499–2350: “extreme moves” that resemble your CAS + on-demand compute ideas
29. (2480) LEANN (compressed graph index + on-demand embedding recompute)  
30. (2460) Lossless compression of vector IDs / postings (ANS-style, recent arXiv line)  
31. (2440) Aggregate Semantic Grouping (ASG) embedding table compression via PQ (arXiv line)  
32. (2400) Semantic compression of multimodal representations (recent arXiv)  

# 5) Opus execution prompt

Model selection
- Primary: Claude Opus (latest available)
- Fallbacks: Claude Sonnet, GPT-5.x, Gemini Pro/3 Pro (whatever is available in your local stack)

```text
You are Opus working inside an Agent Governance System repo.

Goal
Make semantic multiplexing mechanically provable and push measured semantic-density past the “9 nines equivalent” target.

Hard invariants
- Decode must be deterministic or fail closed. No interpretive decoding.
- Every run emits receipts + reports under allowed roots.
- Preserve canonical law unless explicitly scoped; add new specs under docs roots.

Deliverables
1) GOV_IR_SPEC.md
   - Define a typed governance IR
   - Define canonical normalization + stable JSON printing

2) SEMANTIC_DENSITY_SPEC.md
   - Define concept_unit, CDR, ECR, M_required
   - Define multiplex operator semantics
   - Define fail-closed ambiguity rules
   - Define codebook versioning + sync behavior

3) proof_semantic_density_run/
   - Fixed benchmark set (10–30 governance statements + gold IR)
   - Deterministic encode/decode implementation
   - Emit:
     - metrics.json (tokens, concept_units, errors, ECR, M_required)
     - report.md (repro command + summary table)
     - receipts/ (hashes of inputs/outputs, codebook id, tokenizer id)

4) tokenizer_atlas generator
   - Generate TOKENIZER_ATLAS.json for candidate glyph/operator sets
   - Provide deterministic selection rules (prefer single-token across the declared tokenizer)

Exit criteria
- Two consecutive runs produce byte-identical outputs
- ECR is measured and reported (even if low initially)
- M_required is computed for the baseline corpus

Scope guard
- Do not modify Cassette Network unless required for proof harness dependencies.
- Obey GuardedWriter and allowed write domains.
```


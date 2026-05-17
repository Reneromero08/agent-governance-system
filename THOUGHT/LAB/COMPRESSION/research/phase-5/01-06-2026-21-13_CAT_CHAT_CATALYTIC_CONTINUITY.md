---
uuid: 00000000-0000-0000-0000-000000000000
title: CAT_CHAT catalytic continuity report (draft)
section: report
bucket: reports/v4
author: System
priority: High
created: 2026-01-06 21:13
modified: 2026-01-06 21:13
status: Active
summary: 'V4 roadmap report: CAT_CHAT catalytic continuity report (draft)'
tags:
- catalytic
- v4
- roadmap
- report
---
<!-- CONTENT_HASH: 2062cf4c0e9e27a97e5ada014c892c9dbb8e04db3a0dc684f384ec847f0a1c95 -->

# CAT_CHAT catalytic continuity report (draft)

Date: 2026-01-05  
Scope: What you are building, why it is catalytic, and how “tokens = clean space” becomes an engineered property.

## 0. One sentence

CAT_CHAT is a catalytic interface: it keeps only a tiny, task-critical working set inside the model’s context, offloads everything else as content-addressed pointers, and rehydrates on demand through governed retrieval that leaves receipts.

## 1. What you want (requirements, stated plainly)

### Core behavior
- **Tiny live context**: the model holds only what it needs to function “right now.”
- **Aggressive offload**: everything else is replaced by **hash pointers** (and optionally symbol pointers) so tokens stay low.
- **On-demand rehydration**: if the model needs something, it queries catalytic storage (CAS + indexes) to retrieve only the minimal fragments required.
- **Continuity without bloating**: “session continuity” means the system can continue a long-running thread without dragging the whole history into the prompt.

### Catalytic guarantees you are aiming for
- **Deterministic assembly** of the prompt context given identical inputs and identical storage state.
- **Fail-closed** behavior when budgets or invariants cannot be satisfied.
- **Receipted runs**: every important action produces artifacts (hashes, receipts, proofs) so it is auditable later.
- **Reversible or restoration-safe** execution: “catalysis” means the system can run without polluting or silently mutating unrelated state.

### Why this is the “groundbreaking” part
Most “memory” systems either:
- Keep stuffing more tokens into context, or
- Store summaries without strong proof properties.

Your design target is different:
- The model context is **a clean room**, not a database.
- The database is **hash-addressed**, and retrieval is **governed + receipted**.

## 2. Glossary (your system terms, tightened)

- **Token clean space**: the model’s context window treated as ephemeral compute scratchpad.
- **Offload**: replacing content with a stable pointer (hash, symbol, or index key) in durable storage.
- **Rehydrate**: pulling just-in-time fragments back into the clean space through governed retrieval.
- **CAS (content-addressed storage)**: store blobs by hash; identity is content, not filename.
- **Receipt**: machine-readable record of visibly important decisions, inputs, outputs, and hashes.
- **Run bundle**: a deterministic envelope around an execution (inputs, outputs, receipts, proofs).
- **Working set**: the subset of state allowed to enter the prompt right now.
- **Pointer set**: the offloaded remainder represented as hashes or resolvable symbols.
- **Vectors**: approximate-nearest-neighbor index for semantic recall (must be domesticated for determinism).
- **ELO (your usage here)**: a learned or updated importance score used to decide what belongs in the working set.

## 3. What already exists (from your repo work)

### 3.1 Context window management (Phase 3.2.1)
You already have a pure-logic context assembler that:
- enforces hard budgets,
- sorts by priority tiers,
- fails closed if mandatory items do not fit,
- emits a receipt with hashes.

This is the core “token clean space” primitive.

### 3.2 Tool wiring with constraints (Phase 3.3.1 concept)
A constrained tool execution layer (MCP) is the other half of catalytic continuity:
- It is the “hydration interface.”
- It must be allowlisted, fail-closed, and receipted.

### 3.3 Catalytic governance pattern (system-wide)
You have the governance concept that important actions must be:
- bounded by allowed write domains,
- reversible or restore-proofed,
- evidenced by receipts and proofs.

Even when CAT_CHAT is in LAB, the same invariants apply, just with a smaller blast radius.

## 4. The catalytic continuity architecture (how it fits together)

Think of CAT_CHAT as a 3-layer system:

### Layer A: Clean space (prompt assembly)
- Input: live messages, explicit expansions, session capsule pointers, budgets.
- Output: an assembled context list plus an assembly receipt.

This is where you keep context short on purpose.

### Layer B: Durable state (offload substrate)
Durable storage holds:
- conversation events (append-only),
- “session capsule” (small summary + pointer set),
- CAS blobs (full messages, tool outputs, proofs),
- indexes (symbols, vectors, cassette network, whatever you name them).

Nothing here is “in the model.” It is in storage.

### Layer C: Rehydration (governed retrieval)
Hydration is performed through constrained tools:
- hash lookup (exact),
- symbol resolve (exact after governance),
- vector recall (approximate, but receipted and bounded),
- context review and snippet selection.

The rehydration results are themselves content-addressed, so future runs can be compared.

## 5. The missing algorithm: “how much context to keep” (your main ask)

You want an algorithm that decides:
- what stays in the working set,
- what becomes pointers,
- when to rehydrate,
- how to keep this deterministic enough to be catalytic.

### 5.1 A practical design: scored knapsack + deterministic gates

Represent every candidate context item as:

- `id`: stable identifier (hash or canonical pointer)
- `tier`: Mandatory | Recent | Expansion | Retrieved
- `cost`: token estimate (or char proxy)
- `score`: importance (ELO, recency, user pin, constraint weight)
- `must_keep`: boolean (system prompt, latest user message, active constraints)
- `rehydratable`: boolean (has a pointer in durable storage)

Then:

1) Always include all `must_keep`.  
2) Compute remaining budget: `B = max_total - reserve_response - sum(cost(must_keep))`.  
3) Select additional items using a deterministic knapsack heuristic:
   - Primary sort: `tier` priority
   - Secondary sort: `score / cost`
   - Tie-breakers: stable `id` ordering
4) If a Mandatory item does not fit, fail closed.  
5) Emit receipt:
   - selected item ids
   - excluded item ids
   - budgets and costs
   - final hash of the assembled context

This makes “tiny context” a mechanical behavior, not a vibe.

### 5.2 Rehydration trigger logic (minimal and sharp)

When the model encounters a missing dependency, it should:
- first attempt exact lookup (hash, symbol)
- then vector recall if exact is unavailable

Trigger conditions can be explicit and deterministic:
- user asks for a named artifact not in working set
- the plan references a pointer that is not expanded
- tool execution requires a spec section not loaded
- a verification step references a receipt hash not loaded

### 5.3 Making vectors compatible with catalysis (the hard part)

Vectors are approximate and can drift, so you constrain them:

- **Bounded top-k** with stable tie-breaking.
- **Receipted retrieval**: store the query embedding hash (or query text hash), the corpus snapshot id, and the ranked results with their hashes.
- **Snapshot discipline**: retrieval depends on a known corpus state (a merkle root or content digest of the indexed set).
- **Deterministic post-filter**: once candidates are recalled, final selection is exact and stable (hash-based).

Result: vectors become a recall tool, not a source of truth.

## 6. Session continuity that is actually catalytic

“Session persistence” is not “remember everything.”

It is:

- A **session capsule** that contains:
  - minimal stable state (goals, active constraints, active symbols),
  - pointers to durable history,
  - last known corpus snapshot ids.

- A deterministic resume flow:
  1) load capsule (by hash)
  2) assemble clean space (ContextAssembler)
  3) run with constrained tools
  4) write receipts and updated capsule (new hash)

If two runs start from the same capsule hash and use the same corpus snapshot ids, you can compare behavior and outputs.

## 6.5 Non-negotiable invariants for “catalytic for real”
- **Prompt reproducibility**: assembled context must be reproducible from receipt + pointers + corpus snapshot ids.
- **Retrieval reproducibility**: hydration must record enough to replay (or at least explain) what was pulled and why.
- **Stable serialization**: any hashed JSON must be canonically serialized before hashing.
- **No hidden state**: anything that can change behavior must live in durable state and be referenced by hash.
- **Fail-closed under uncertainty**: if a dependency cannot be resolved, do not guess and do not silently proceed.

## 6.6 What to measure (so “token clean space” is not subjective)
- **Working set size**: tokens (or chars) in the assembled context per turn.
- **Offload ratio**: (pointers bytes) / (raw history bytes) and the implied token savings.
- **Hydration rate**: hydrations per turn, and tokens spent per hydration.
- **Recall precision**: fraction of hydrated snippets that end up used (a direct input to ELO updates).
- **Determinism checks**: repeated assembly hash matches for identical inputs.

## 7. Crypto safe (where it fits)

Based on your stated intent, “crypto safe” is not about hiding files from normal users.  
It is about creating a **tamper-evident seal** on specific protected artifacts (license, provenance, or protected payload), so “breaking the seal” becomes an objective event you can prove.

Catalytically, this should integrate as:
- inputs sealed (hashes and signatures),
- outputs sealed (hashes and signatures),
- receipts include seal state,
- enforcement prevents accidental leakage into public artifacts.

If you want this to support legal enforceability, focus the design on:
- unambiguous scope of what is sealed,
- unambiguous evidence of breakage,
- clean custody chain.

## 8. Concrete engineering artifacts to add next (when you are ready)

### 8.1 The session capsule schema
- Canonical JSON schema for capsule fields.
- Required hashes for:
  - conversation event log head
  - corpus snapshot ids (vectors, symbols, CAS)
  - last assembly receipt hash

### 8.2 Append-only chat event log
- Each event is content-addressed.
- The log head is a hash chain or merkle accumulator.
- Lets you prove ordering without keeping tokens.

### 8.3 Hydration receipts
For every hydration:
- query hash
- corpus snapshot id
- ranked result hashes
- selected snippet hashes
- deterministic post-filter parameters

### 8.4 ELO update rule
A simple, robust rule you can iterate:
- increase ELO when an item is retrieved and used
- decay ELO with time
- penalize items retrieved and discarded
- clamp and quantize so it is deterministic across runtimes

Store ELO changes as events, not edits.

## 9. The full picture in one diagram (text)

1) User message arrives
2) ContextAssembler builds clean space:
   - mandatory items
   - best scored items under budget
   - pointers for everything else
   - emits assembly receipt
3) Model produces plan and identifies missing dependencies
4) Tool layer rehydrates:
   - exact first, vectors second
   - receipts every retrieval
5) Model executes, emits run bundle artifacts
6) Session capsule updates:
   - new capsule hash becomes the new “session state”
7) Next turn repeats from capsule hash, not from raw history in tokens

## 10. Related work and research pointers (for later)

These are useful because your system rhymes with software supply chain security and reproducible builds:
- provenance and attestation (SLSA)
- cryptographic workflow integrity (in-toto)
- canonical serialization (JCS)
- hash trees and merkle accumulators

(See footnotes.)

## Footnotes
1. SLSA specification and provenance concepts: slsa.dev (for example: https://slsa.dev/spec/v1.0/ and provenance pages).
2. in-toto framework and attestation specs: https://in-toto.io/ and https://in-toto.io/docs/specs/
3. RFC 8785 (JSON Canonicalization Scheme): https://www.rfc-editor.org/rfc/rfc8785
4. Merkle tree overview: https://en.wikipedia.org/wiki/Merkle_tree

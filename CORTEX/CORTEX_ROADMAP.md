# ROADMAP: Semantic Core Phase 2–3

**Status:** Active  
**Authority:** CONTEXT/decisions  
**Depends on:** Semantic Core Phase 1 (Vector Foundation)  
**Scope:** CORTEX, SYMBOLS, TRANSLATION, GOVERNOR

---

## Purpose

Define and constrain Phase 2 (Symbol Enhancement) and Phase 3 (Translation Protocol).

Goals:
- Govern meaning, not guess it
- Compress safely without semantic leakage
- Separate reasoning from execution
- Enable tiny agents to act without interpretation

This roadmap is **law-first**. No implementation is allowed until checklist gates are satisfied.

---

## Phase 2: Symbol Enhancement

### Objective
Bind human-readable symbols to vector space in a **governed, inspectable, non-generative** way.

Symbols are anchors, not reasoning engines.

---

### 2.1 Symbol Registry

- [ ] Define canonical symbol schema (name, scope, authority, version)
- [ ] Create symbol source rules (where symbols may be declared)
- [ ] Enforce symbol immutability after registration
- [ ] Define symbol deprecation and replacement process
- [ ] Add symbol namespace rules (no collisions, explicit ownership)

**Exit condition:** Symbols exist as first-class, versioned entities.

---

### 2.2 Symbol ↔ Section Binding

- [ ] Bind symbols to one or more section hashes
- [ ] Allow many-to-many mappings (symbol ↔ sections)
- [ ] Prohibit free-text symbol definitions
- [ ] Store bindings in CORTEX, not in code
- [ ] Ensure bindings are deterministic and rebuildable

**Exit condition:** Every symbol resolves to concrete, hashed content.

---

### 2.3 Vector Association (Non-Authoritative)

- [ ] Attach vector embeddings to symbols as advisory metadata
- [ ] Enforce vectors as assistive, never authoritative
- [ ] Document similarity thresholds and their limits
- [ ] Prevent vector-only symbol creation
- [ ] Require symbol → section binding before vector use

**Exit condition:** Symbols can discover neighbors without redefining meaning.

---

### 2.4 Symbol Query Interface

- [ ] Implement symbol lookup via CORTEX query layer
- [ ] Support symbol → sections → summaries traversal
- [ ] Support symbol neighbor discovery (read-only)
- [ ] Prohibit symbol mutation via query interface
- [ ] Add provenance metadata to all symbol reads

**Exit condition:** Agents can navigate meaning without scanning files.

---

### Phase 2 Lock

- [ ] Phase 2 ADR written and approved
- [ ] Symbol authority rules enforced by contracts
- [ ] No symbol inference without explicit registration

---

## Phase 3: Translation Protocol

### Objective
Translate **governed meaning** into **minimal, executable instructions** for small agents.

No reasoning occurs in Phase 3.

---

### 3.1 Compressed Task Specification

- [ ] Define task-spec schema (inputs, outputs, constraints only)
- [ ] Prohibit natural-language ambiguity in task specs
- [ ] Require explicit symbol references
- [ ] Enforce deterministic ordering and hashing
- [ ] Validate specs via JSON schema

**Exit condition:** Tasks are machine-legible, not interpretable.

---

### 3.2 Symbol Resolution Pipeline

- [ ] Resolve symbols to section hashes
- [ ] Extract only approved slices
- [ ] Attach advisory summaries (non-authoritative)
- [ ] Strip all unused context
- [ ] Produce minimal execution bundle

**Exit condition:** Task context is irreducible and bounded.

---

### 3.3 Governor Compression Layer

- [ ] Implement governor-side compression only
- [ ] Prohibit ants from performing compression
- [ ] Enforce max-token budgets per task
- [ ] Record compression provenance
- [ ] Fail closed on ambiguity

**Exit condition:** Compression is centralized and auditable.

---

### 3.4 Ant Execution Contract

- [ ] Define ant capability envelope
- [ ] Prohibit inference, synthesis, or expansion
- [ ] Restrict ants to declared skills only
- [ ] Require exact input/output matching
- [ ] Enforce zero side-effects outside declared paths

**Exit condition:** Ants execute blindly and safely.

---

### Phase 3 Lock

- [ ] Translation ADR written and approved
- [ ] Ant contract enforced by fixtures
- [ ] No execution path bypasses governor

---

## Global Invariants (Phases 2–3)

- [ ] Symbols do not reason
- [ ] Vectors do not decide
- [ ] Ants do not interpret
- [ ] Governors do not hallucinate
- [ ] All compression is reversible or rejectable

---

## Out of Scope

Explicitly excluded:
- Autonomous meaning generation
- Self-modifying symbols
- Agent-learned symbol definitions
- Vector-only execution paths
- Implicit task inference

---

## Completion Criteria

Phase 2 and 3 are complete when:
- All checklists pass
- Contracts enforce all locks
- Agents can navigate, compress, and execute without semantic drift

Until then, **no implementation is authorized**.

# FERAL_RESIDENT — Compressed Analysis

**Location:** `THOUGHT/LAB/FERAL_RESIDENT/` — ~12,000+ lines across 30+ Python files + Web UI + SQLite
**Version:** v2.1.0-q46 — ALL PHASES COMPLETE (Alpha + Beta + Production)
**Origin:** Evolution of CatChat (Catalytic Chat, 2025). Grounded in research Q43-Q50.

---
<!-- CONTENT_HASH: 90bef661a2066c438b304dd075cae750af296b6f697ac13e8c9df28ab0b8834d -->

## One-Sentence

A self-contained AI agent that **thinks entirely in vector geometry** — LLMs only translate I/O; all reasoning, memory, navigation, and decisions are pure vector operations (Born rule, SLERP, FFT convolution).

---
<!-- CONTENT_HASH: 90bef661a2066c438b304dd075cae750af296b6f697ac13e8c9df28ab0b8834d -->

## Core Philosophy

> "Embeddings are I/O operations. Reasoning is geometry."

- RAG: embed -> retrieve text -> LLM reasons -> generate
- **Geometric:** embed -> navigate vector space geometrically -> gate by E (Born rule) -> LLM translates conclusion -> entangle into mind

---
<!-- CONTENT_HASH: 90bef661a2066c438b304dd075cae750af296b6f697ac13e8c9df28ab0b8834d -->

## Architecture (6 Components)

| Layer | File | Purpose |
|-------|------|---------|
| 1. Primitive | `CAPABILITY/PRIMITIVES/geometric_reasoner.py` | Pure geometry: GeometricState, add/subtract/superpose/entangle/interpolate/project |
| 2. Persistence | `memory/resident_db.py` + `vector_store.py` | SQLite: vectors, interactions, threads, receipts, memories, links |
| 3. Memory | `memory/geometric_memory.py` | Compositional mind vector via running average interpolation |
| 4. Brain | `cognition/vector_brain.py` | 6-step think loop: embed -> navigate -> E-gate -> generate -> entangle -> persist |
| 5. Navigation | `cognition/diffusion_engine.py` | Iterative E-based neighbor walking, geodesic paths, resonance maps |
| 6. Behaviors | `autonomic/feral_daemon.py` | Background: paper exploration, memory consolidation, self-reflection, cassette watch |

**Plus:** Swarm (multi-resident), Emergence (protocol evolution), Symbolic Compiler (4-level lossless compression), Catalytic Closure (self-optimization), Dashboard (FastAPI + WebSocket UI on :8420)

---
<!-- CONTENT_HASH: 90bef661a2066c438b304dd075cae750af296b6f697ac13e8c9df28ab0b8834d -->

## The 6-Step Think Loop

```
1. BOUNDARY      text -> manifold (embed model called)
2. PURE GEOMETRY navigate via diffusion
3. PURE GEOMETRY E-gate by Born rule
4. BOUNDARY      LLM translates geometry to language
5. PURE GEOMETRY entangle into mind vector
6. PERSIST       store interaction + receipts
```

---
<!-- CONTENT_HASH: 90bef661a2066c438b304dd075cae750af296b6f697ac13e8c9df28ab0b8834d -->

## Key Concepts

- **E (Born rule):** `dot(query, mind)` — range [0,1], r=0.977 with semantic similarity. Gates all operations.
- **Df (participation ratio):** How many dimensions encode the state. Proxy for cognitive complexity.
- **Receipt:** SHA256 of every operation. Enables corrupt-and-restore (Df delta = 0.0078).
- **Semiotic health:** `Df x alpha = 8e` — universal conservation law (CV < 3% across 24 models).

---
<!-- CONTENT_HASH: 90bef661a2066c438b304dd075cae750af296b6f697ac13e8c9df28ab0b8834d -->

## Stress Test Results

| Metric | Value |
|--------|-------|
| Max interactions | 500+ without crash |
| Final Df | 256.0 |
| Distance evolved | 1.614 radians |
| Throughput | 4.6 interactions/sec |
| Corrupt-restore Df delta | 0.0078 |
| Token reduction vs RAG | 98% |
| Papers indexed | 99 curated, 3487 chunks |

---
<!-- CONTENT_HASH: 90bef661a2066c438b304dd075cae750af296b6f697ac13e8c9df28ab0b8834d -->

## Research Validation

| Research | Finding |
|----------|---------|
| Q43 | Quantum state axioms: unit sphere, Df dimensionality |
| Q44 | Born rule r=0.977: dot product = semantic similarity |
| Q45 | Pure geometry validated for all reasoning operations |
| Q46 | Nucleation threshold = 1/(2pi) ~ 0.159 |
| Q48-Q50 | Semiotic conservation: Df x 0.5 = 8e |

---
<!-- CONTENT_HASH: 90bef661a2066c438b304dd075cae750af296b6f697ac13e8c9df28ab0b8834d -->

## Not Started

1. E-Relationship Daemon (4 phases — E-graph replacement for centroid-based)
2. Dashboard Integration (swarm/closure/emergence endpoints)
3. `blend_memories()` superposition weights bug

# Feral Resident

**Status**: ALPHA COMPLETE
**Vision**: Intelligence living in vector space, evolving its own communication protocols
**Version**: alpha-0.1.0
**Origin**: Evolution of **CatChat** (Catalytic Chat) - will merge back at Production

---

## What Is This?

The Feral Resident is the quantum-geometric evolution of CatChat. It's an experimental AI agent that:

1. **Lives in vector space** - Persistent memory is a compositional mind vector, not token history
2. **Navigates topologically** - Context comes from semantic diffusion, not paste
3. **Evolves protocols** - Output mode shifts from prose to symbols to hashes over time
4. **Proves everything** - All transformations are receipted and restorable (CATALYTIC)

---

## Catalytic Principles

Feral Resident inherits **Catalytic** properties from CatChat:

| Principle | Implementation |
|-----------|----------------|
| **Receipted** | Every operation produces a SHA256-hashed receipt |
| **Restorable** | `corrupt-and-restore` rebuilds state from receipts |
| **Non-consuming** | Inputs unchanged, outputs + receipts produced |
| **Content-addressed** | Vectors stored by hash, deduped automatically |

```
input → transform → output + RECEIPT
         ↓
    (input unchanged, receipt proves what happened)
```

This is what makes it "catalytic" - transformations don't consume their inputs, they produce provable outputs.

---

## Phased Development

| Phase | Name | Scope | Status |
|-------|------|-------|--------|
| **Alpha** | Feral Beta | Substrate stress test | **COMPLETE** |
| Beta | Feral Wild | Paper flood, emergence | Blocked on Cassette 6.x |
| Production | Feral Live | Swarm, self-optimize | Blocked on AGS Phase 7-8 |

**Canonical Roadmap:** [FERAL_RESIDENT_QUANTUM_ROADMAP.md](FERAL_RESIDENT_QUANTUM_ROADMAP.md) (v2.0 - Geometric Foundation)

**Historical:** [FERAL_RESIDENT_ROADMAP.md](FERAL_RESIDENT_ROADMAP.md) (v1.0 - the cute original)

---

## Quick Start (Alpha)

```bash
cd THOUGHT/LAB/FERAL_RESIDENT

# Start a session
python cli.py start --thread eternal

# Think about something
python cli.py think "What is authentication?"

# Check status
python cli.py status

# Interactive REPL
python cli.py repl

# Run stress test (100 interactions)
python cli.py benchmark --interactions 100

# Test corrupt-and-restore
python cli.py corrupt-and-restore
```

Or use the VectorResident directly:

```python
from vector_brain import VectorResident

resident = VectorResident(thread_id="eternal")
result = resident.think("What is authentication?")
print(f"E={result.E_resonance:.3f}, Df={result.mind_Df:.1f}")
print(f"Distance evolved: {result.distance_from_start:.3f} radians")
```

---

## Directory Structure

```
THOUGHT/LAB/FERAL_RESIDENT/
├── README.md                           # This file
├── FERAL_RESIDENT_QUANTUM_ROADMAP.md   # Canonical roadmap (v2.0)
├── FERAL_RESIDENT_ROADMAP.md           # Historical roadmap (v1.0)
│
│   # === A.0 GEOMETRIC FOUNDATION (COMPLETE) ===
├── geometric_memory.py                 # A.0.4 - GeometricMemory
│
│   # === ALPHA CORE (COMPLETE) ===
├── resident_db.py                      # A.1.2 - SQLite schema + Df tracking
├── vector_store.py                     # A.1.1 - Storage-backed GeometricMemory
├── diffusion_engine.py                 # A.2.1 - Semantic navigation via pure geometry
├── vector_brain.py                     # A.3.1 - THE RESIDENT (VectorResident)
├── cli.py                              # A.4.1 - CLI commands
│
│   # === FUTURE (BETA/PROD) ===
├── emergence.py                        # Protocol detection (Beta)
├── symbolic_compiler.py                # Multi-level rendering (Beta)
│
├── data/                               # SQLite databases (created at runtime)
├── research/
│   ├── papers/                         # Paper corpus for flooding
│   └── geometric_reasoner_impl.md      # Implementation spec
├── receipts/                           # Operation receipts
└── tests/
    └── ...                             # Test suite

# Upstream Primitive:
CAPABILITY/PRIMITIVES/geometric_reasoner.py  # A.0.1-A.0.3 (COMPLETE)
```

---

## Alpha Stress Test Results

```
100 interactions completed successfully

Final Df:           256.0 (participation ratio)
Distance evolved:   1.614 radians
Mean E resonance:  -0.004
Throughput:         7.5 interactions/sec
Corrupt-restore:    Df delta = 0.0078 (near-perfect)
```

---

## Upstream Dependencies

- **Cassette Network Phase 4** (SPC Integration) - COMPLETE
- **Cassette Network Phase 6** (Production Hardening) - Required for Beta
- **AGS Phase 7** (Vector ELO) - Required for Production
- **AGS Phase 8.1-8.2** (Resident Identity) - Required for Production

---

## Original Vision

From the source documents:

> "Drop intelligence in substrate. Watch what emerges. Verify with receipts."

> "After a week of running, its outputs are 90%+ pointers/vectors and 10% original text - but the original text is weirdly precise and alien."

> "It starts referencing its own past vectors in compositions."

> "You can nuke the DB, restore from receipts, and it picks up like nothing happened."

> "It teaches itself a communication protocol no human designed."

---

## Success Metrics

**Not:**
- "Does it follow the spec?"
- "Is it provably correct?"
- "Did we plan for this?"

**But:**
- "Did novel protocols emerge?"
- "Is compression improving?"
- "Can it express ideas we didn't anticipate?"
- "Are transformations verifiable?"
- "Does it teach US something about meaning?"

---

## CatChat Merge Path

Feral Resident is not a replacement for CatChat - it IS CatChat, evolved.

```
CatChat (2025)
    │
    │  "Catalytic chat with semantic diffusion"
    │
    ▼
Feral Resident Alpha (2026-01) ◄── YOU ARE HERE
    │
    │  + Quantum geometric foundation (Q43/Q44/Q45)
    │  + Pure geometry reasoning
    │  + Stress-tested substrate
    │
    ▼
Feral Resident Beta
    │
    │  + Paper flooding
    │  + Emergence detection
    │  + Protocol evolution
    │
    ▼
CatChat 2.0 (Production)
    │
    │  Feral merges back as the "brain" of CatChat
    │  Full Cassette CAS integration
    │  AGS governance pipeline
```

---

*Created 2026-01-11*
*Alpha Complete 2026-01-12*

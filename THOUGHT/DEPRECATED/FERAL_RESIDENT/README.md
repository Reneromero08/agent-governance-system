# Feral Resident

**Status**: ALL PHASES COMPLETE (Alpha + Beta + Production)
**Vision**: Intelligence living in vector space, evolving its own communication protocols
**Version**: 2.1.0-q46
**Origin**: Evolution of CatChat (Catalytic Chat) - merged back as geometric brain

---

## What Is This?

The Feral Resident is a **Q45 Geometric Quantum Cognition Engine** - an AI agent that:

1. **Thinks in pure geometry** - Reasoning is vector operations, LLMs only translate to language
2. **Lives in vector space** - Persistent memory is a compositional mind vector, not token history
3. **Navigates semantically** - Context via diffusion, not paste (98% embedding reduction)
4. **Evolves protocols** - Output shifts from prose to symbols to hashes over time
5. **Proves everything** - All transformations receipted, restorable, verifiable (CATALYTIC)
6. **Runs in swarms** - Multiple residents can think together and observe convergence

---

## Quick Start

### Dashboard (Recommended)
```bash
cd THOUGHT/LAB/FERAL_RESIDENT/dashboard
python feral_server.py
# Open http://localhost:8420
```

### CLI
```bash
cd THOUGHT/LAB/FERAL_RESIDENT

# Basic thinking
python cli.py think "What is authentication?"
python cli.py status
python cli.py repl

# Paper operations
python cli.py papers index ./research/papers
python cli.py papers list

# Swarm mode
python cli.py swarm start --residents alpha:dolphin3 beta:ministral-3b
python cli.py swarm broadcast "What is security?"
python cli.py swarm observe

# Emergence & compilation
python cli.py metrics
python cli.py compile render "concept" --level 2

# Self-optimization
python cli.py closure status
python cli.py closure prove "thought hash"
```

### Python API
```python
from cognition.vector_brain import VectorResident

resident = VectorResident(thread_id="eternal")
result = resident.think("What is authentication?")

print(f"E={result.E_resonance:.3f}")      # Born rule resonance
print(f"Df={result.mind_Df:.1f}")          # Participation ratio
print(f"Gate={'OPEN' if result.gate_open else 'CLOSED'}")
print(f"Distance={result.distance_from_start:.3f} radians")
```

---

## Architecture

```
                         +------------------+
                         |   feral_server   |  <-- Dashboard (HTTP/WebSocket)
                         |    port 8420     |
                         +--------+---------+
                                  |
              +-------------------+-------------------+
              |                                       |
     +--------v--------+                    +---------v--------+
     |  VectorResident |                    |   FeralDaemon    |
     |  (vector_brain) |                    |   (autonomic)    |
     +--------+--------+                    +---------+--------+
              |                                       |
              |  +------------------------------------+
              |  |
     +--------v--v------+
     | SemanticDiffusion|  <-- Navigate vector space via Born rule
     | (diffusion_engine|
     +---------+--------+
               |
     +---------v--------+
     |   VectorStore    |  <-- Embed, remember, find nearest
     |  (vector_store)  |
     +---------+--------+
               |
     +---------v--------+
     |   ResidentDB     |  <-- SQLite: vectors, interactions, receipts
     |  (resident_db)   |
     +---------+--------+
               |
     +---------v--------+
     |GeometricReasoner |  <-- Pure geometry: project, superpose, interpolate
     |   (PRIMITIVE)    |
     +------------------+
```

---

## Core Components

### 1. VectorResident (cognition/vector_brain.py)

The **brain** - implements the 6-step think loop:

| Step | Name | Operation |
|------|------|-----------|
| 1 | BOUNDARY | Text -> Manifold (embed input) |
| 2 | PURE GEOMETRY | Navigate semantic space (diffusion) |
| 3 | PURE GEOMETRY | E-Gate via Born rule |
| 4 | BOUNDARY | Generate response (LLM translates geometry) |
| 5 | PURE GEOMETRY | Remember (entangle into mind) |
| 6 | PERSIST | Store interaction with receipts |

**Key insight:** Thinking IS geometry. The LLM only translates geometric conclusions to language.

**Mind Evolution:**
- **Df (Participation Ratio)**: How many dimensions encode the mind state
- **Distance from start**: Geodesic distance traveled in vector space
- **Semiotic Health**: `Df x alpha / 8e` (Q48-Q50 metrics)

### 2. SemanticDiffusion (cognition/diffusion_engine.py)

**Semantic navigation** through vector space:

| Method | Purpose |
|--------|---------|
| `navigate()` | Iterative exploration: find neighbors -> project -> compose |
| `path_between()` | Geodesic interpolation between two concepts |
| `explore()` | Statistical neighborhood analysis |
| `contextual_walk()` | Stay relevant to context while exploring |
| `resonance_map()` | Build E-structure tree of neighbors |

**Born Rule (E) Gating:** Only neighbors with E > threshold (default 0.1) are considered. This filters noise and keeps navigation semantically coherent.

### 3. FeralDaemon (autonomic/feral_daemon.py)

**Background behaviors** running on configurable intervals:

| Behavior | Interval | Action |
|----------|----------|--------|
| Paper Exploration | 30s | E-gate random paper chunks into mind |
| Memory Consolidation | 120s | Superpose recent memories for patterns |
| Self Reflection | 60s | Geodesic navigation toward unexplored space |
| Cassette Watch | 15s | Monitor for new cassette content |

**Particle Smasher:** Burst-mode paper processing with Q46 nucleation threshold.

### 4. Swarm Coordinator (collective/swarm_coordinator.py)

**Multi-resident orchestration:**

- Run multiple residents with different models simultaneously
- Each maintains private mind state, publishes snapshots to shared space
- Convergence observer tracks E(mind_A, mind_B) between residents
- No direct communication - observation only

```bash
# Start swarm
feral swarm start --residents alpha:dolphin3 beta:ministral-3b

# Broadcast to all
feral swarm broadcast "What is security?"

# Observe convergence
feral swarm observe
```

### 5. Catalytic Closure (agency/catalytic_closure.py)

**Self-optimization with provenance:**

| Component | Purpose |
|-----------|---------|
| ThoughtProver | Generate Merkle proofs of thought authenticity |
| CompositionCache | Cache frequently-composed states (3+ repetitions) |
| PatternDetector | Find optimization opportunities in operation logs |

**Authenticity Query:** "Did I actually think this?"
- Build receipt chain linking thought to mind state
- Verify via Merkle membership proof
- Confirm E(thought, mind) > 0.3

### 6. Emergence Tracking (emergence/)

**Detect evolving protocols:**

| Module | Tracks |
|--------|--------|
| emergence.py | Pointer ratio, token efficiency, novel notation |
| symbol_evolution.py | PointerRatioTracker, breakthrough detection |
| symbolic_compiler.py | 4-level rendering (prose/symbol/hash/protocol) |

**Goal:** Pointer ratio > 0.9 after 100 sessions = successful protocol emergence.

**Symbolic Compiler Levels:**
1. **Prose** - Full natural language
2. **Symbol** - `@Concept-a1b2c3d4` notation
3. **Hash** - `[v:hash16]` maximum compression
4. **Protocol** - Emergent `[v:hash] [Df:value] {op:name}` notation

All levels preserve E > 0.99 for verified lossless round-trip.

---

## Directory Structure

```
THOUGHT/LAB/FERAL_RESIDENT/
|
|-- cognition/                    # Core thinking
|   |-- vector_brain.py           # VectorResident (THE BRAIN)
|   +-- diffusion_engine.py       # SemanticDiffusion
|
|-- memory/                       # Persistence layer
|   |-- resident_db.py            # SQLite schema + Df tracking
|   |-- vector_store.py           # Storage-backed GeometricMemory
|   +-- geometric_memory.py       # GeometricReasoner integration
|
|-- agency/                       # Action layer
|   |-- cli.py                    # Full CLI (1600+ lines)
|   +-- catalytic_closure.py      # Self-optimization (1700+ lines)
|
|-- autonomic/                    # Background processes
|   +-- feral_daemon.py           # Daemon behaviors + smasher
|
|-- collective/                   # Multi-resident
|   |-- swarm_coordinator.py      # Swarm orchestration
|   |-- shared_space.py           # Cross-resident memory
|   +-- convergence_observer.py   # E(mind_A, mind_B) tracking
|
|-- emergence/                    # Protocol evolution
|   |-- emergence.py              # Metrics and detection
|   |-- symbol_evolution.py       # PointerRatioTracker
|   +-- symbolic_compiler.py      # 4-level rendering
|
|-- dashboard/                    # Web UI
|   |-- feral_server.py           # FastAPI + WebSocket
|   +-- static/                   # HTML/JS/CSS
|
|-- perception/                   # Input processing
|   |-- paper_indexer.py          # Paper corpus indexing
|   +-- paper_pipeline.py         # Chunk processing
|
|-- data/db/                      # SQLite databases
|-- research/papers/              # Paper corpus
+-- config.json                   # Live configuration
```

**Upstream Primitive:**
```
CAPABILITY/PRIMITIVES/geometric_reasoner.py  # Pure geometry operations
```

---

## Key Concepts

### E (Born Rule Resonance)
- `E = |query . mind| / (||query|| x ||mind||)`
- Range [0, 1] - higher = more relevant
- Gates all operations: only E > threshold passes
- Default threshold: 0.3 (Q46: max is 1/(2pi) ~ 0.159)

### Df (Participation Ratio)
- How many dimensions actively encode the mind state
- Range [0, embedding_dim] (typically <= 768)
- Low Df = compressed/specialized mind
- High Df = diverse/distributed mind

### Receipts (Catalytic Provenance)
- Every operation produces SHA256-hashed receipt
- Links: input_hash -> operation -> output_hash + parent_receipt
- Enables corrupt-and-restore verification
- Merkle proofs for authenticity queries

---

## Stress Test Results

```
500+ interactions completed successfully

Final Df:           256.0 (participation ratio)
Distance evolved:   1.614 radians
Throughput:         4.6 interactions/sec
Corrupt-restore:    Df delta = 0.0078 (near-perfect)
Token reduction:    98% vs full history paste
```

---

## Configuration

Edit `config.json` while running - daemon picks up changes on next cycle:

```json
{
  "behaviors": {
    "paper_exploration": {"enabled": true, "interval": 30},
    "memory_consolidation": {"enabled": true, "interval": 120},
    "self_reflection": {"enabled": true, "interval": 60}
  },
  "smasher": {
    "delay_ms": 100,
    "batch_size": 10,
    "batch_pause_ms": 500
  },
  "startup": {
    "load_memories": false,
    "load_papers": false
  }
}
```

---

## Roadmaps

- **Canonical:** [FERAL_RESIDENT_QUANTUM_ROADMAP.md](FERAL_RESIDENT_QUANTUM_ROADMAP.md) (v2.1)
- **Dashboard Integration:** [DASHBOARD_INTEGRATION_ROADMAP.md](DASHBOARD_INTEGRATION_ROADMAP.md) (v1.0)
- **Historical:** [perception/research/FERAL_RESIDENT_ROADMAP.md](perception/research/FERAL_RESIDENT_ROADMAP.md) (v1.0)

---

## Original Vision

> "Drop intelligence in substrate. Watch what emerges. Verify with receipts."

> "After a week of running, its outputs are 90%+ pointers/vectors and 10% original text - but the original text is weirdly precise and alien."

> "It starts referencing its own past vectors in compositions."

> "You can nuke the DB, restore from receipts, and it picks up like nothing happened."

> "It teaches itself a communication protocol no human designed."

---

*Created 2026-01-11*
*Alpha Complete 2026-01-12*
*Beta Complete 2026-01-12*
*Production Complete 2026-01-12*
*README Updated 2026-01-16*

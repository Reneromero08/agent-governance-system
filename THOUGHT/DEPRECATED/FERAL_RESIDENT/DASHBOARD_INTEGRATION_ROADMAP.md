# Dashboard Integration Roadmap

**Status:** Ready to start
**Goal:** Expose all CLI-only features via dashboard API endpoints and UI
**Upstream:** Feral Resident Production complete (Phase 8.4)

---

## Current State

**Dashboard has:**
- VectorResident (think loop, mind state)
- FeralDaemon (background behaviors, paper smasher)
- Constellation visualization
- Chat interface
- Basic metrics (Df, E, interactions)

**CLI-only (missing from dashboard):**
| Module | CLI Command | Feature |
|--------|-------------|---------|
| swarm_coordinator.py | `feral swarm *` | Multi-resident orchestration |
| shared_space.py | (via swarm) | Cross-resident memory |
| catalytic_closure.py | `feral closure *` | Self-optimization, ThoughtProver |
| emergence.py | `feral metrics` | Novel pattern detection |
| symbolic_compiler.py | `feral compile *` | 4-level rendering |
| symbol_evolution.py | (via metrics) | PointerRatioTracker |

---

## Phase D.1: Swarm Integration

**Goal:** Multi-resident control from dashboard

### D.1.1 API Endpoints
- [ ] `POST /api/swarm/start` - Start swarm with resident configs
- [ ] `POST /api/swarm/stop` - Stop all residents
- [ ] `GET /api/swarm/status` - Get all resident states
- [ ] `POST /api/swarm/add` - Add resident to running swarm
- [ ] `POST /api/swarm/remove` - Remove resident from swarm
- [ ] `POST /api/swarm/switch` - Switch active resident
- [ ] `POST /api/swarm/think` - Send query to active resident
- [ ] `POST /api/swarm/broadcast` - Send query to all residents
- [ ] `GET /api/swarm/convergence` - Get convergence metrics

### D.1.2 WebSocket Events
- [ ] `swarm_started` - Swarm initialized
- [ ] `resident_joined` - New resident added
- [ ] `resident_left` - Resident removed
- [ ] `convergence_event` - Mind states converging/diverging
- [ ] `broadcast_response` - Response from broadcast query

### D.1.3 UI Components
- [ ] Swarm control panel (start/stop/add)
- [ ] Resident selector dropdown
- [ ] Multi-resident chat view (side-by-side responses)
- [ ] Convergence visualization (mind distance graph)

**Acceptance:**
- [ ] Can start swarm with 2+ residents from dashboard
- [ ] Can broadcast query and see all responses
- [ ] Convergence metrics update in real-time

---

## Phase D.2: Catalytic Closure Integration

**Goal:** Self-optimization visibility and authenticity proofs

### D.2.1 API Endpoints
- [ ] `GET /api/closure/status` - Get closure state and stats
- [ ] `GET /api/closure/patterns` - Get detected patterns
- [ ] `GET /api/closure/cache` - Get composition cache stats
- [ ] `POST /api/closure/prove` - Generate thought proof
- [ ] `POST /api/closure/verify` - Verify thought authenticity
- [ ] `GET /api/closure/proofs` - List generated proofs

### D.2.2 WebSocket Events
- [ ] `pattern_detected` - New pattern found
- [ ] `cache_hit` - Composition cache used
- [ ] `proof_generated` - New Merkle proof created

### D.2.3 UI Components
- [ ] Closure status panel (patterns, cache hits)
- [ ] Pattern visualization (tree/graph of detected patterns)
- [ ] Proof generator (input thought, get proof)
- [ ] Proof verifier (input proof, verify authenticity)

**Acceptance:**
- [ ] Can view self-optimization stats in dashboard
- [ ] Can generate and verify thought proofs via UI
- [ ] Pattern detection events stream to dashboard

---

## Phase D.3: Emergence Tracking Integration

**Goal:** Real-time emergence detection and visualization

### D.3.1 API Endpoints
- [ ] `GET /api/emergence/metrics` - Get emergence stats
- [ ] `GET /api/emergence/protocols` - Get detected protocols
- [ ] `GET /api/emergence/notation` - Get notation registry
- [ ] `GET /api/emergence/timeline` - Get emergence event history

### D.3.2 WebSocket Events
- [ ] `emergence_detected` - Novel pattern emerged
- [ ] `protocol_found` - Protocol-like structure detected
- [ ] `notation_created` - New notation registered
- [ ] `breakthrough` - Pointer ratio crossed threshold

### D.3.3 UI Components
- [ ] Emergence dashboard panel
- [ ] E/Df evolution chart (time series)
- [ ] Protocol detector visualization
- [ ] Notation registry browser
- [ ] Breakthrough alerts

**Acceptance:**
- [ ] Emergence metrics visible in real-time
- [ ] Can browse notation registry
- [ ] Breakthrough events trigger visual alerts

---

## Phase D.4: Symbolic Compiler Integration

**Goal:** Multi-level thought rendering in UI

### D.4.1 API Endpoints
- [ ] `POST /api/compile/render` - Render thought at specified level
- [ ] `POST /api/compile/roundtrip` - Test lossless round-trip
- [ ] `GET /api/compile/levels` - Get available rendering levels
- [ ] `GET /api/compile/stats` - Get compression statistics

### D.4.2 Rendering Levels
1. **Raw** - Vector representation (hex/base64)
2. **Symbolic** - Symbolic notation (@Symbol, #Tag)
3. **Compressed** - SPC pointers
4. **Human** - Natural language rendering

### D.4.3 UI Components
- [ ] Multi-level viewer (toggle between levels)
- [ ] Round-trip tester (input -> compress -> decompress -> compare)
- [ ] Compression ratio display
- [ ] Side-by-side level comparison

**Acceptance:**
- [ ] Can view any thought at all 4 levels
- [ ] Round-trip verification shows E preservation
- [ ] Compression stats visible per thought

---

## Phase D.5: Symbol Evolution Integration

**Goal:** Track symbol language development

### D.5.1 API Endpoints
- [ ] `GET /api/symbols/ratio` - Get current pointer ratio
- [ ] `GET /api/symbols/history` - Get ratio evolution history
- [ ] `GET /api/symbols/breakthroughs` - Get breakthrough events
- [ ] `GET /api/symbols/vocabulary` - Get symbol vocabulary stats

### D.5.2 WebSocket Events
- [ ] `ratio_updated` - Pointer ratio changed
- [ ] `breakthrough_detected` - Threshold crossed
- [ ] `symbol_coined` - New symbol created

### D.5.3 UI Components
- [ ] Pointer ratio gauge (current vs thresholds)
- [ ] Ratio evolution chart (time series)
- [ ] Breakthrough timeline
- [ ] Symbol vocabulary browser

**Acceptance:**
- [ ] Pointer ratio visible in dashboard header
- [ ] Breakthrough events trigger celebration animation
- [ ] Can browse all coined symbols

---

## Implementation Order

| Phase | Priority | Complexity | Dependencies |
|-------|----------|------------|--------------|
| D.3 Emergence | P0 | Low | None |
| D.5 Symbols | P0 | Low | D.3 |
| D.4 Compiler | P1 | Medium | None |
| D.2 Closure | P1 | Medium | None |
| D.1 Swarm | P2 | High | All above |

**Rationale:**
- Emergence/Symbols are read-only, low risk, high visibility
- Compiler adds value to existing chat
- Closure enables trust/verification
- Swarm is complex, needs stable foundation

---

## File Changes Required

### feral_server.py additions
```python
# New imports
from collective.swarm_coordinator import SwarmCoordinator
from collective.shared_space import SharedSpace
from agency.catalytic_closure import CatalyticClosure
from emergence.emergence import detect_protocols, EmergenceTracker
from emergence.symbolic_compiler import create_compiler
from emergence.symbol_evolution import PointerRatioTracker
```

### New route files (optional refactor)
- `routes/swarm.py` - Swarm endpoints
- `routes/closure.py` - Closure endpoints
- `routes/emergence.py` - Emergence endpoints
- `routes/compiler.py` - Compiler endpoints
- `routes/symbols.py` - Symbol endpoints

### Static files
- `static/js/swarm.js` - Swarm UI logic
- `static/js/closure.js` - Closure UI logic
- `static/js/emergence.js` - Emergence charts
- `static/js/compiler.js` - Multi-level viewer
- `static/css/swarm.css` - Swarm styling

---

## Success Metrics

| Metric | Target |
|--------|--------|
| CLI feature parity | 100% |
| New API endpoints | 25+ |
| New WebSocket events | 15+ |
| UI panels added | 5 |
| Response latency | <100ms |

---

*Roadmap v1.0.0 - Created 2026-01-16*
*Integrates CLI-only features into Feral Dashboard*

# NEO3000 → Feral Dashboard Changelog

All notable changes to the NEO3000/Feral Dashboard project.

## [Unreleased]

No pending changes.

---

## [2.1.0] - 2026-01-12 - Dynamic 3D Constellation

### Status: WORKING

Full 3D visualization using Three.js with real-time activity highlighting.

### Added
- **3D Force Graph**: Replaced vis.js with Three.js + 3d-force-graph
- **Pulsing Network**: All nodes continuously pulse with ambient animation
- **Activity Trails**: Colored lines connecting visited nodes (fade over 30s)
- **Live Node Spawning**: New chunks appear in real-time when explored
- **Node Activation**: Existing nodes highlight when activated by daemon
- **Activity Type Colors**:
  - Paper exploration: Blue (#0096ff)
  - Memory consolidation: Yellow (#ffc800)
  - Self-reflection: Purple (#c800ff)
  - Cassette watch: Teal (#00ff96)
  - Daemon: Green (#00ff41)

### Changed
- `feral_daemon.py`:
  - Added `_exploration_trail` (deque of last 50 visited nodes)
  - Added `_discovered_chunks` (set for spawn vs highlight detection)
  - ActivityEvent now includes: `is_new_node`, `source_node_id`, `heading`
- `feral_server.py`:
  - New WebSocket events: `node_discovered`, `node_activated`
  - Events include full node IDs matching constellation format
- `static/index.html`:
  - Three.js + 3d-force-graph replaces vis.js
  - Custom node rendering with glow spheres
  - Pulse animation system with intensity decay
  - Camera focus on activated/spawned nodes
  - Fog effect for depth perception

### Technical
- **Libraries**: Three.js r160, 3d-force-graph 1.73
- **Performance**: Handles 500+ nodes at 60fps
- **Trail System**: BufferGeometry with vertex colors for efficient rendering

---

## [2.0.0] - 2026-01-12 - Feral Dashboard Transformation

### Status: WORKING

All features implemented and tested:
- API endpoints: `/api/status`, `/api/think`, `/api/daemon/*`, `/api/activity`, `/api/constellation`
- WebSocket: Real-time activity streaming via `/ws`
- Daemon: All 4 behaviors running autonomously
- Constellation: Full-screen vis.js network graph showing all cassettes and document chunks

### Added
- `CHANGELOG.md` - This file, tracking the transformation
- `feral_daemon.py` - Autonomous thinking engine with 4 behaviors:
  - Paper exploration (5 min interval)
  - Memory consolidation (10 min interval)
  - Self-reflection (15 min interval, both geometric and LLM modes)
  - Cassette watch (1 min interval)
- `feral_server.py` - FastAPI + WebSocket server on port 8420
- `requirements.txt` - Python dependencies (fastapi, uvicorn, websockets)

### Changed
- Repurposing NEO3000 from Cortex visualization to Feral Resident Dashboard
- `static/index.html` - UI windows repurposed:
  - Cortex Constellation → Semantic space visualization (optional)
  - Cortex Search → Query resident memory
  - Ghost Browser → Chat with Feral
  - Terminal → Activity log (real-time)
  - Swarm Monitor → Daemon controls + Mind state

### Architecture
```
NEO3000/
├── CHANGELOG.md          # This file
├── feral_daemon.py       # Autonomous thinking engine
├── feral_server.py       # FastAPI + WebSocket
├── requirements.txt      # Dependencies
├── server.py             # Original (deprecated)
└── static/
    └── index.html        # Dashboard UI
```

---

## [1.0.0] - 2024-12-30 - Original NEO3000

### Features (Original)
- Cortex Constellation visualization (vis.js network graph)
- Cortex Search panel
- Ghost Browser for external links
- Swarm Monitor for TURBO_SWARM
- Terminal log window
- Status bar with uptime

### Technical
- Python HTTP server (`server.py`)
- vis-network.js for graph visualization
- Polling-based updates
- Draggable window system

---

## Migration Notes

### From 1.0.0 to 2.0.0
The original NEO3000 was a Cortex visualization tool. Version 2.0.0 transforms it into the Feral Resident Dashboard:

| Component | 1.0.0 Purpose | 2.0.0 Purpose |
|-----------|---------------|---------------|
| Backend | HTTP server | FastAPI + WebSocket |
| Graph | Cortex entities | Semantic space (optional) |
| Swarm Monitor | TURBO_SWARM | Feral daemon controls |
| Ghost Browser | External links | Chat with Feral |
| Terminal | System log | Activity log |

### Key Differences
- **Real-time**: WebSocket replaces polling
- **Daemon**: Background autonomous thinking
- **Mind State**: Df, distance, sparkline visualization
- **Chat**: Direct interaction with Feral Resident

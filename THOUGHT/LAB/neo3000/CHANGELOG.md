# NEO3000 → Feral Dashboard Changelog

All notable changes to the NEO3000/Feral Dashboard project.

## [Unreleased]

No pending changes.

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

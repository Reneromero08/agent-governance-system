# NEO3000 → Feral Dashboard Changelog

All notable changes to the NEO3000/Feral Dashboard project.

## [Unreleased]

No pending changes.

---

## [3.2.2] - 2026-01-12 - Node Reference Fix

### Fixed
- **Graph Node References**: Similarity edges now only reference nodes that exist in the graph
  - Fixed "node not found" error when similarity edges referenced chunks outside loaded set
  - Added `loaded_chunk_ids` tracking to ensure embeddings query matches graph nodes
  - Changed from loading ALL embeddings to only those for chunks in the graph
  - Used parameterized SQL query: `WHERE c.chunk_id IN (?)` with loaded IDs

### Changed
- **Development Workflow**: Disabled hot reload for cleaner development experience
  - Frontend changes (index.html): Just refresh browser
  - Backend changes (feral_server.py): Manual restart after batching edits

---

## [3.2.1] - 2026-01-12 - Similarity Links Fix

### Fixed
- **Cosine Similarity Now Works**: Fixed embedding table lookup
  - Was looking for non-existent `embeddings` table
  - Now correctly uses `geometric_index` table with `chunk_hash = doc_id` join
  - 384-dim vectors from `vector_blob` column

---

## [3.2.0] - 2026-01-12 - Cosine Similarity + Enhanced Pulsing

### Added
- **Cosine Similarity Links**: Semantic edges between chunks based on embedding similarity
  - Server computes cosine similarity from chunk embeddings in cassette databases
  - Similarity edges shown in cyan color, strength based on similarity score
  - Toggle to show/hide similarity links in Graph settings
  - Adjustable similarity threshold slider (0.50 - 0.95)
- **API Parameters**: `/api/constellation?include_similarity=true&similarity_threshold=0.7`

### Changed
- **Enhanced Pulsing Animation**: Nodes pulse more visibly
  - Base intensity increased from 0.1 to 0.4
  - Pulse frequency increased (0.5-1.0 vs 0.3-0.6)
  - Scale variation increased to 0.25 (was 0.15)
  - Emissive intensity variation increased to 0.5 (was 0.3)
  - Glow opacity variation increased
  - Decay time extended to 10s (was 5s)
- **Link Styling**: Hierarchy edges dimmed (opacity 0.2), similarity edges prominent (0.3-0.8)

### Technical
- `feral_server.py`: Added `compute_similarity_edges()` function with numpy
- `index.html`: Added `allLinks` state, `filterLinks()`, `toggleSimilarityLinks()`, `updateSimThreshold()`, `reloadConstellation()`
- Edge types: `hierarchy` (green, thin) vs `similarity` (cyan, thick based on weight)

---

## [3.1.1] - 2026-01-12 - Sidebar Toggle Fix

### Fixed
- **Sidebar Re-expansion**: Toggle button now remains visible when sidebar is collapsed, allowing re-expansion
  - Added CSS rules to hide title and center button in collapsed state
  - Button stays at 40px width for click target

---

## [3.1.0] - 2026-01-12 - Minimal UI + Fixes

### Status: COMPLETE

Streamlined UI with black/white/green only, fog control, and chat bug fix.

### Added
- **Fog Density Slider**: Control 3D graph visibility (0 to 0.01 range)
- **Collapsible Sidebar**: Toggle button collapses entire sidebar to 40px
- **Async Script Loading**: Three.js libraries load in parallel with `async` attribute
- **waitForThreeJS()**: Promise-based loading ensures graph init after scripts ready

### Changed
- **Removed Header Bar**: Layout now directly sidebar + canvas (no top bar)
- **Color Scheme**: Pure black (#000000) background, white text, green (#00ff41) accent only
- **Activity Badges**: All badges now use green instead of per-type colors
- **3D Trail Colors**: All activity types now render as green (unified)
- **Chat Panel**: Now full height from top to activity bar (was below header)
- **WebSocket Status**: Moved into sidebar header area
- **Removed 2D Transitions**: No CSS transitions on 2D UI elements

### Fixed
- **Chat Duplicate Messages**: Removed WebSocket `thought` handler since API already returns response
- **Graph Fog Color**: Changed from `#0a0a0a` to `#000000` to match background

### Technical
- Grid layout: `1fr var(--activity-height)` rows (no header row)
- Sidebar collapse: CSS class toggle `.sidebar-collapsed`
- Fog slider: Maps 0-100 to 0-0.01 density
- All `ACTIVITY_COLORS` now point to `0x00ff41`

---

## [3.0.0] - 2026-01-12 - Modern UI Redesign

### Status: COMPLETE

Complete UI overhaul from floating windows to a modern, professional dashboard layout.

### Added
- **Modern Grid Layout**: CSS Grid-based layout with fixed sidebar + canvas + activity bar
- **Inter + JetBrains Mono Typography**: Professional font stack for clean readability
- **Collapsible Sidebar Sections**: Accordion-style sections for Mind State, Daemon, and Graph settings
- **Slide-out Chat Panel**: Floating action button opens chat from the right edge
- **Activity Bar**: Bottom footer with real-time scrolling activity feed
- **iOS-style Toggle Switches**: Modern behavior toggles with smooth animations
- **Connection Status Header**: Clean header with logo, WebSocket status, and uptime

### Changed
- **Layout**: Replaced draggable floating windows with fixed sidebar layout
- **Color Palette**: Refined dark theme with `#0a0a0a` primary background
- **Typography**: From `Lucida Console` monospace to `Inter` + `JetBrains Mono`
- **Loading Screen**: Modern spinner replacing blinking text
- **Removed**: Scanline overlay effect and window dragging code
- **Chat**: Moved from floating window to slide-out panel with FAB trigger

### Technical
- CSS Variables for consistent theming
- Grid layout: `header | sidebar + canvas | activity`
- Chat panel: `400px` width, slide-in animation
- Activity bar: Horizontal scrolling feed with badges
- Section collapse state managed via CSS class toggle

### UI Components
| Component | Location | Features |
|-----------|----------|----------|
| Header | Top | Logo, WS status, uptime |
| Sidebar | Left (280px) | 3 collapsible sections |
| Canvas | Center | 3D constellation |
| Activity | Bottom (48px) | Scrolling feed |
| Chat | Right slide-out | FAB trigger, messages |

### Visual Improvements
- Removed retro scanline effects
- Cleaner stat cards with progress bars
- Modern slider styling for graph controls
- Subtle hover states and transitions
- Badge colors for activity types preserved

---

## [2.3.0] - 2026-01-12 - UI Controls + Graph Fix

### Added
- **Daemon Interval Controls**: Editable input fields for each behavior interval
  - Paper Exploration: number input (5-3600 sec)
  - Memory Consolidation: number input (10-3600 sec)
  - Self-Reflection: number input (10-3600 sec)
  - Cassette Watch: number input (5-600 sec)
  - Changes apply live via API
- **Graph Settings Panel** (Obsidian-style sliders):
  - Center Force: 0-1.0 (default 0.05)
  - Repel Force: 0-500 (default -120)
  - Link Force: 0-1.0 (default 0.50)
  - Link Distance: 10-300 (default 100)
  - Reset to Defaults button
  - All changes apply live with simulation reheat

### Fixed
- **Graph nodes no longer collapse**: Fixed forces that were too tight
  - Link distance: 15 → 100
  - Charge strength: -30 → -120
  - Center strength: 0.15 → 0.05
  - Removed overly aggressive collision force
  - Camera position: z=200 → z=400
  - Fog density: 0.008 → 0.003
  - Focus distance: 60 → 150

---

## [2.2.0] - 2026-01-12 - Active Daemon + Obsidian-Style Graph (Broken)

### Changed
- **Daemon intervals now much more active**:
  - Paper exploration: 5 min → 30 sec
  - Memory consolidation: 10 min → 2 min
  - Self-reflection: 15 min → 1 min
  - Cassette watch: 1 min → 15 sec
- **Obsidian-style tight graph clustering** (TOO TIGHT - fixed in 2.3.0):
  - Link distance: 15 (was default ~30)
  - Charge strength: -30 (was -120)
  - Center gravity: 0.15 (stronger pull to center)
  - Collision radius: 6 (prevents overlap)
  - Initial camera at z=200 to see whole graph
  - Fog density increased for depth (0.008)
  - Camera focus distance reduced to 60

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

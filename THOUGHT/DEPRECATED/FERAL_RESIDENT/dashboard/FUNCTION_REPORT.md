# Feral Resident Dashboard - Function Report

## Core Data Model

### Mind State
| Property | Type | Description |
|----------|------|-------------|
| `Df` | float (0-256) | Participation ratio - how "spread out" the mind is |
| `distance_from_start` | float (radians) | Angular distance from initial state |
| `hash` | string | 8-char hash of current mind vector |
| `interactions` | int | Total interaction count |

### Think Result (Chat Response)
| Property | Type | Description |
|----------|------|-------------|
| `response` | string | The resident's response text |
| `E_resonance` | float (-1 to 1) | Resonance with query |
| `gate_open` | bool | Whether thought was integrated |
| `mind_Df` | float | Mind Df after this thought |

---

## API Endpoints

### Status & Data
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/status` | GET | Mind state + interaction count |
| `/api/evolution` | GET | Df history (last 50) |
| `/api/constellation` | GET | Graph nodes + edges |
| `/api/activity?limit=50` | GET | Recent activity log |
| `/api/history?limit=20` | GET | Past chat interactions |

### Chat
| Endpoint | Method | Body |
|----------|--------|------|
| `/api/think` | POST | `{query: string}` |

### Daemon Control
| Endpoint | Method | Body |
|----------|--------|------|
| `/api/daemon/status` | GET | - |
| `/api/daemon/start` | POST | - |
| `/api/daemon/stop` | POST | - |
| `/api/daemon/config` | POST | `{behavior, enabled?, interval?}` |

---

## Daemon Behaviors

| Behavior | Default | Purpose |
|----------|---------|---------|
| `paper_exploration` | 30s | Read paper chunks |
| `memory_consolidation` | 120s | Find patterns |
| `self_reflection` | 60s | Generate questions |
| `cassette_watch` | 15s | Monitor for new content |

Each has: `enabled`, `interval`, `last_run`, `next_run`

---

## WebSocket Events (`/ws`)

| Event | Data | When |
|-------|------|------|
| `init` | mind + daemon status | On connect |
| `mind_update` | `{Df, distance}` | After thought |
| `activity` | `{timestamp, action, summary}` | Daemon activity |
| `thought` | `{query, response, E, gate_open}` | Chat response |
| `node_discovered` | `{node_id, label, source_id}` | New 3D node |
| `node_activated` | `{node_id, source_id}` | Node visited |

---

## UI Components Needed

### Display (Read-only)
1. Mind Df (number + progress bar)
2. Mind Distance (radians)
3. Interaction Count
4. Df Sparkline (mini chart)
5. Daemon Status (running/stopped)
6. Daemon Uptime
7. Activity Feed (scrolling)
8. WebSocket Status

### Controls
1. Start/Stop Daemon
2. Toggle Behaviors (4x)
3. Behavior Intervals (4x number inputs)
4. Graph Forces (4x sliders)
5. Reset Graph button

### Interactive
1. Chat Input + Send
2. Chat Messages (with E badges)
3. 3D Graph (orbit, click to focus)
4. Node Tooltip

---

## Layout Suggestion

```
┌─────────────────────────────────────────────────────────────┐
│ [Logo] FERAL                    [Status] ● Connected        │
├──────────┬──────────────────────────────────────────────────┤
│          │                                                  │
│ SIDEBAR  │              3D CONSTELLATION                    │
│ 280px    │              (Full remaining space)              │
│          │                                                  │
│ [Mind]   │                                                  │
│  Df: 142 │                                                  │
│  ████░░░ │                                                  │
│          │                                                  │
│ [Daemon] │                                                  │
│  ▶ ON    │                                                  │
│  Papers  │                                                  │
│  Memory  │                                                  │
│  Reflect │                                                  │
│          │                                                  │
│ [Graph]  │                                                  │
│  Forces  │                                                  │
│          │                                                  │
├──────────┴──────────────────────────────────────────────────┤
│ [Activity] Paper explored... | Memory consolidated...       │
└─────────────────────────────────────────────────────────────┘

[Chat] Slide-out from right edge
```

---

## Activity Colors
| Activity | Color |
|----------|-------|
| Paper | `#0096ff` Blue |
| Memory | `#ffc800` Yellow |
| Reflect | `#c800ff` Purple |
| Cassette | `#00ff96` Teal |

---

## Refresh Patterns
| Data | Method | Frequency |
|------|--------|-----------|
| Mind state | Poll | 10s |
| Evolution | Poll | 30s |
| Daemon status | Poll | 5s |
| Activity | WebSocket | Real-time |
| Constellation | WebSocket | Real-time |

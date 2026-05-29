// =============================================================================
// FERAL DASHBOARD - SHARED STATE
// =============================================================================
//
// Central state store for the entire dashboard.
// All modules import from here to share state.
//
// WHY THIS EXISTS:
//   ES6 modules have read-only bindings, so we need setter functions.
//   This file provides a single source of truth for all UI state.
//
// HOW TO USE:
//   import * as state from './state.js';
//   state.daemonRunning        // read
//   state.setDaemonRunning(x)  // write
//
// SECTIONS:
//   1. WebSocket & Connection
//   2. Daemon State
//   3. Smasher State
//   4. Graph State
//   5. UI State
//   6. Settings (persisted)
//   7. Setters
//
// =============================================================================

import { CONFIG } from './config.js';

// =============================================================================
// 1. WEBSOCKET & CONNECTION
// =============================================================================

// Active WebSocket connection (null when disconnected)
export let ws = null;

// =============================================================================
// 2. DAEMON STATE
// =============================================================================

// Whether the daemon loop is currently running
export let daemonRunning = false;

// Daemon behaviors configuration
// Structure: { behavior_name: { enabled: bool, interval: number } }
// Example: { memory_consolidation: { enabled: true, interval: 120 } }
export let behaviors = {};

// =============================================================================
// 3. SMASHER STATE
// =============================================================================

// Whether particle smasher is actively processing chunks
export let smasherActive = false;

// Smasher configuration (synced with server)
// delay_ms: milliseconds between chunks
// batch_size: chunks per batch before pause
export let smasherConfig = {
    delay_ms: 100,
    batch_size: 10
};

// Q46 Nucleation: dynamic threshold based on memory count
export let currentThreshold = 0.080;
export let nMemories = 0;

// Camera follows smasher cursor when false
export let staticCameraMode = true;

// ----- SMASHER VISUALIZATION QUEUE -----
// Batches smash events to prevent overwhelming the renderer
export let smashQueue = [];
export let smashRafPending = false;
export const MAX_SMASH_BATCH = CONFIG.SMASHER.MAX_BATCH_PER_FRAME;

// ----- SMASHER CURSOR (3D indicator) -----
export let smasherCursor = null;
export let smasherCurrentNodeId = null;
export let smasherTrailNodes = [];
export const SMASHER_TRAIL_LENGTH = CONFIG.TRAILS.SMASHER_TRAIL_LENGTH;

// =============================================================================
// 4. GRAPH STATE
// =============================================================================

// ForceGraph3D instance (null until initialized)
export let Graph = null;

// Node registry for fast lookups
// byId: Map<nodeId, nodeObject>
export let nodeRegistry = { byId: new Map() };

// Animation state for node pulsing (disabled in perf mode)
// Map<nodeId, { intensity, phase, frequency, lastActivity }>
export let nodePulseState = new Map();

// Exploration trail: recent path through the graph
// Array of { nodeId, timestamp, type }
export let explorationTrail = [];
export let trailLine = null;
export const MAX_TRAIL_LENGTH = CONFIG.TRAILS.MAX_LENGTH;

// All links (edges) in the graph, including filtered ones
export let allLinks = [];

// =============================================================================
// 5. UI STATE
// =============================================================================

// Chat panel open/closed
export let chatOpen = false;

// Sidebar collapsed/expanded
export let sidebarCollapsed = false;

// Whether to show similarity edges in graph
export let showSimilarityLinks = true;

// Minimum similarity score to show edge (0.0-1.0)
export let similarityThreshold = 0.35;

// =============================================================================
// 6. SETTINGS (persisted to config.json)
// =============================================================================

// Graph physics and display settings
// These are loaded from config.json and saved periodically
export let graphSettings = {
    fog: CONFIG.GRAPH.FOG_DENSITY,
    center: CONFIG.GRAPH.FORCE_CENTER_STRENGTH,
    repel: Math.abs(CONFIG.GRAPH.FORCE_CHARGE_STRENGTH),
    linkStrength: CONFIG.GRAPH.FORCE_LINK_STRENGTH,
    linkDistance: CONFIG.GRAPH.FORCE_LINK_DISTANCE
};

// =============================================================================
// 7. ACTIVITY COLORS
// =============================================================================
// Colors for different activity types in the graph
// Used by graph.js for trail coloring

export const ACTIVITY_COLORS = {
    paper: { main: 0x00ff41, glow: '#00ff41' },
    consolidate: { main: 0x00ff41, glow: '#00ff41' },
    reflect: { main: 0x00ff41, glow: '#00ff41' },
    cassette: { main: 0x00ff41, glow: '#00ff41' },
    daemon: { main: 0x00ff41, glow: '#00ff41' },
    smash: { main: 0x00ff41, glow: '#00ff41' },
    default: { main: 0x00ff41, glow: '#00ff41' }
};

// =============================================================================
// 8. SETTERS
// =============================================================================
// ES6 exports are read-only bindings, so we need functions to mutate state

// ----- Connection -----
export function setWs(val) { ws = val; }

// ----- Daemon -----
export function setDaemonRunning(val) { daemonRunning = val; }
export function setBehaviors(val) { behaviors = val; }

// ----- Smasher -----
export function setSmasherActive(val) { smasherActive = val; }
export function setSmasherConfig(val) { smasherConfig = val; }
export function setCurrentThreshold(val) { currentThreshold = val; }
export function setNMemories(val) { nMemories = val; }
export function setStaticCameraMode(val) { staticCameraMode = val; }
export function setSmashQueue(val) { smashQueue = val; }
export function setSmashRafPending(val) { smashRafPending = val; }
export function setSmasherCursor(val) { smasherCursor = val; }
export function setSmasherCurrentNodeId(val) { smasherCurrentNodeId = val; }

// Add node to smasher trail (FIFO queue)
export function addToSmasherTrail(nodeId) {
    smasherTrailNodes.push({ nodeId, timestamp: Date.now() });
    while (smasherTrailNodes.length > SMASHER_TRAIL_LENGTH) {
        smasherTrailNodes.shift();
    }
}

// ----- Graph -----
export function setGraph(val) { Graph = val; }
export function setTrailLine(val) { trailLine = val; }
export function setAllLinks(val) { allLinks = val; }

// ----- UI -----
export function setChatOpen(val) { chatOpen = val; }
export function setSidebarCollapsed(val) { sidebarCollapsed = val; }
export function setShowSimilarityLinks(val) { showSimilarityLinks = val; }
export function setSimilarityThreshold(val) { similarityThreshold = val; }

// ----- Settings -----
export function setGraphSettings(val) {
    graphSettings = { ...graphSettings, ...val };
}

export function updateGraphSetting(key, val) {
    graphSettings[key] = val;
}

// =============================================================================
// FERAL DASHBOARD - MAIN ENTRY POINT
// =============================================================================
//
// This is the main entry point for the Feral Dashboard application.
// It initializes all modules and wires up WebSocket message handling.
//
// =============================================================================
// ARCHITECTURE OVERVIEW
// =============================================================================
//
// The dashboard is built with ES6 modules, each handling a specific concern:
//
//   +------------------+     +------------------+     +------------------+
//   |    config.js     |     |    state.js      |     |     api.js       |
//   |------------------|     |------------------|     |------------------|
//   | All tunable      |     | Shared app state |     | REST API calls   |
//   | constants        |     | (singleton)      |     | WebSocket conn   |
//   +------------------+     +------------------+     +------------------+
//           |                        |                        |
//           v                        v                        v
//   +---------------------------------------------------------------+
//   |                         main.js (this file)                    |
//   |---------------------------------------------------------------|
//   | - Loads settings                                               |
//   | - Initializes graph                                            |
//   | - Connects WebSocket                                           |
//   | - Routes WS messages to handlers                               |
//   | - Sets up polling intervals                                    |
//   | - Exposes functions to window for HTML onclick                 |
//   +---------------------------------------------------------------+
//           |                        |                        |
//           v                        v                        v
//   +------------------+     +------------------+     +------------------+
//   |    graph.js      |     |   smasher.js     |     |   daemon.js      |
//   |------------------|     |------------------|     |------------------|
//   | 3D constellation |     | Particle smasher |     | Daemon controls  |
//   | ForceGraph3D     |     | visualization    |     | behavior toggles |
//   +------------------+     +------------------+     +------------------+
//           |                        |                        |
//           v                        v                        v
//   +------------------+     +------------------+     +------------------+
//   |    mind.js       |     |  activity.js     |     |    chat.js       |
//   |------------------|     |------------------|     |------------------|
//   | Mind state UI    |     | Activity feed    |     | Chat panel       |
//   | Df, sparkline    |     | bottom bar       |     | Q&A interface    |
//   +------------------+     +------------------+     +------------------+
//           |                        |                        |
//           v                        v                        v
//   +------------------+     +------------------+
//   |   settings.js    |     |     ui.js        |
//   |------------------|     |------------------|
//   | Load/save config |     | Sidebar, section |
//   | Apply graph prefs|     | Chat toggles     |
//   +------------------+     +------------------+
//
// DATA FLOW:
//   1. User loads page -> main.js init() runs
//   2. Settings loaded from config.json
//   3. Graph initialized with ForceGraph3D
//   4. Status/evolution/daemon data fetched via REST
//   5. WebSocket connected for real-time updates
//   6. WS messages routed to appropriate handlers
//   7. Polling intervals refresh non-real-time data
//
// CUSTOMIZATION:
//   - Tunable values: Edit config.js (CONFIG object)
//   - Visual styling: Edit styles.css (CSS variables in :root)
//   - Polling intervals: CONFIG.POLLING in config.js
//
// FILES:
//   config.js    - All tunable constants with documentation
//   state.js     - Shared application state
//   api.js       - REST API and WebSocket communication
//   graph.js     - 3D constellation visualization
//   smasher.js   - Particle smasher controls and visualization
//   daemon.js    - Daemon process controls
//   mind.js      - Mind state display (Df, distance, evolution)
//   activity.js  - Activity feed in bottom bar
//   chat.js      - Chat panel for Q&A with Feral
//   settings.js  - Settings persistence to config.json
//   ui.js        - Basic UI controls (sidebar, sections, chat toggle)
//
// =============================================================================

// =============================================================================
// SECTION 1: IMPORTS
// =============================================================================

import { CONFIG } from './config.js';
import * as state from './state.js';
import { connectWebSocket } from './api.js';
import { toggleSidebar, toggleSection, toggleChat } from './ui.js';
import { updateMindState, loadStatus, loadEvolution } from './mind.js';
import { addActivity } from './activity.js';
import { loadDaemonStatus, updateDaemonStatus, toggleDaemon, toggleBehavior, updateInterval } from './daemon.js';
import {
    toggleSmasher, updateSmasherUI, updateSmasherStats, updateCurrentFile,
    clearCurrentFile, queueSmashVisualization, updateSmasherSpeed, updateSmasherBatch,
    toggleStaticCamera, loadSmasherStatus, updateThresholdDisplay
} from './smasher.js';
import {
    initConstellation, spawnNode, activateNode, addToTrail,
    updateFog, updateGraphForce, resetGraphForces, reloadConstellation,
    invalidateConstellationCache
} from './graph.js';
import { loadSettings, saveSettings, applyGraphSettings, toggleSimilarityLinks, updateSimThreshold } from './settings.js';
import { sendMessage } from './chat.js';

// =============================================================================
// SECTION 2: WEBSOCKET MESSAGE HANDLER
// =============================================================================
// Routes incoming WebSocket messages to appropriate handlers
//
// Message types (from feral_server.py):
//   - init: Initial state on connect (mind + daemon status)
//   - mind_update: Mind state changed (Df, distance)
//   - activity: New activity event
//   - node_discovered: New node added to constellation
//   - node_activated: Existing node was accessed
//   - smash_hit: Particle smasher processed a chunk (batched)
//   - hot_reload: Server file changed, reload page

function handleWebSocketMessage(msg) {
    // Debug: log all messages except high-frequency mind_update
    if (msg.type !== 'mind_update') {
        console.log('[WS]', msg.type, msg.data ? Object.keys(msg.data) : 'no data');
    }

    // Route message to appropriate handler
    switch (msg.type) {
        case 'init':
            // Initial state on connect
            updateMindState(msg.data.mind);
            updateDaemonStatus(msg.data.daemon);
            break;

        case 'mind_update':
            // Mind state changed
            updateMindState(msg.data);
            break;

        case 'activity':
            // New activity event
            addActivity(msg.data);
            break;

        case 'node_discovered':
            // New node added to constellation
            spawnNode(msg.data);
            addToTrail(msg.data.node_id, msg.data.activity_type);
            break;

        case 'node_activated':
            // Existing node was accessed
            activateNode(msg.data.node_id, msg.data.activity_type);
            addToTrail(msg.data.node_id, msg.data.activity_type);
            break;

        case 'smash_hit':
            // Particle smasher processed chunks (batched for performance)
            handleSmashHit(msg.data);
            break;

        case 'hot_reload':
            // Server file changed, reload page
            console.log('[HOT RELOAD] File changed, reloading...');
            window.location.reload();
            break;
    }
}

/**
 * Handle smash_hit WebSocket message
 * Processes batched smasher events for visualization
 *
 * @param {Object} data - Smash hit data
 * @param {Array} data.batch - Array of smash events (if batched)
 * @param {string} data.node_id - Node that was processed
 * @param {number} data.E - Similarity/energy value
 * @param {boolean} data.gate_open - Whether chunk was absorbed
 * @param {number} data.rate - Current processing rate
 */
function handleSmashHit(data) {
    // Server sends batched events to reduce WebSocket flood
    const batch = data.batch || [data];
    const batchSize = batch.length;

    // Only log if small batch (avoid console spam)
    if (batchSize <= 3) {
        console.log('[SMASH_HIT]', data.node_id, 'E=', data.E);
    }

    // Update UI with latest hit only (not every item in batch)
    updateCurrentFile(data);
    updateSmasherStats(data.rate);

    // Add to activity feed and queue for graph visualization
    for (const item of batch) {
        const gateStatus = item.gate_open ? 'ABSORBED' : 'REJECTED';
        addActivity({
            timestamp: Date.now(),
            action: 'smash',
            summary: `E=${item.E.toFixed(2)} ${gateStatus}`,
            details: {
                paper: item.paper || 'unknown',
                chunk_id: item.chunk_id || item.node_id
            }
        });
        queueSmashVisualization(item);
    }
}

// =============================================================================
// SECTION 3: INITIALIZATION
// =============================================================================
// Called on page load to set up the entire application

/**
 * Initialize the Feral Dashboard
 *
 * Initialization order:
 *   1. Load settings from config.json
 *   2. Initialize 3D constellation graph
 *   3. Apply graph settings (physics, fog)
 *   4. Load initial data (status, evolution, daemon, smasher)
 *   5. Connect WebSocket for real-time updates
 *   6. Hide loading screen
 *   7. Start polling intervals
 */
async function init() {
    // Load user settings from config.json
    await loadSettings();

    // Initialize 3D graph (must be after settings for defaults)
    await initConstellation();

    // Apply graph settings from loaded config
    applyGraphSettings();

    // Load initial data via REST API
    await loadStatus();
    await loadEvolution();
    await loadDaemonStatus();
    await loadSmasherStatus();

    // Connect WebSocket for real-time updates
    connectWebSocket(handleWebSocketMessage);

    // Hide loading screen
    document.getElementById('loading').classList.add('hidden');

    // Start polling intervals for non-real-time data
    // TWEAK: Adjust intervals in CONFIG.POLLING (config.js)
    setInterval(loadStatus, CONFIG.POLLING.STATUS_INTERVAL_MS);
    setInterval(loadEvolution, CONFIG.POLLING.EVOLUTION_INTERVAL_MS);
    setInterval(loadDaemonStatus, CONFIG.POLLING.DAEMON_INTERVAL_MS);
    setInterval(loadSmasherStatus, CONFIG.POLLING.SMASHER_INTERVAL_MS);
    setInterval(saveSettings, CONFIG.POLLING.SETTINGS_SAVE_MS);
}

// =============================================================================
// SECTION 4: GLOBAL FUNCTION EXPOSURE
// =============================================================================
// Expose functions to window object for HTML onclick handlers
//
// These functions are called from index.html like:
//   onclick="toggleSidebar()"
//   oninput="updateFog(this.value)"

// UI Controls
window.toggleSidebar = toggleSidebar;
window.toggleSection = toggleSection;
window.toggleChat = toggleChat;

// Daemon Controls
window.toggleDaemon = toggleDaemon;
window.toggleBehavior = toggleBehavior;
window.updateInterval = updateInterval;

// Smasher Controls
window.toggleSmasher = toggleSmasher;
window.updateSmasherSpeed = updateSmasherSpeed;
window.updateSmasherBatch = updateSmasherBatch;
window.toggleStaticCamera = toggleStaticCamera;

// Graph Controls
window.toggleSimilarityLinks = toggleSimilarityLinks;
window.updateSimThreshold = updateSimThreshold;
window.updateFog = updateFog;
window.updateGraphForce = updateGraphForce;
window.resetGraphForces = resetGraphForces;

// Chat Controls
window.sendMessage = sendMessage;

// Settings
window.saveSettings = saveSettings;

// =============================================================================
// SECTION 5: START APPLICATION
// =============================================================================

window.onload = init;

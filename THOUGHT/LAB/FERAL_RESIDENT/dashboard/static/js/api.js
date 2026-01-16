// =============================================================================
// FERAL DASHBOARD - API & WEBSOCKET
// =============================================================================
//
// This module handles all communication with the Feral server:
// - REST API calls (fetch-based)
// - WebSocket connection for real-time updates
//
// OVERVIEW:
//   The dashboard uses a hybrid communication model:
//   - REST API for commands (start, stop, config changes)
//   - WebSocket for real-time event streaming (smash hits, activity)
//
// SECTIONS IN THIS FILE:
//   1. Imports
//   2. REST API Helper
//   3. WebSocket Connection
//
// CUSTOMIZATION:
//   - Reconnect delay is in CONFIG.WEBSOCKET.RECONNECT_DELAY_MS (config.js)
//   - API base path is '/api' (hardcoded, change if server changes)
//
// =============================================================================

// =============================================================================
// SECTION 1: IMPORTS
// =============================================================================

import { CONFIG } from './config.js';
import * as state from './state.js';

// =============================================================================
// SECTION 2: REST API HELPER
// =============================================================================
// Wrapper for fetch() that adds error handling and JSON parsing

/**
 * Make a REST API call to the Feral server
 *
 * @param {string} endpoint - API endpoint (without /api prefix, e.g., '/status')
 * @param {Object} options - Fetch options (method, body, etc.)
 * @returns {Promise<Object>} Parsed JSON response
 * @throws {Error} On non-2xx response with status and endpoint info
 *
 * Example usage:
 *   const data = await api('/daemon/status');
 *   await api('/smasher/start', { method: 'POST', body: JSON.stringify({...}) });
 */
export async function api(endpoint, options = {}) {
    const res = await fetch(`/api${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });

    if (!res.ok) {
        const error = new Error(`API error: ${res.status} ${res.statusText}`);
        error.status = res.status;
        error.endpoint = endpoint;
        console.error(`[API] ${endpoint} failed:`, res.status, res.statusText);
        throw error;
    }

    return res.json();
}

// =============================================================================
// SECTION 3: WEBSOCKET CONNECTION
// =============================================================================
// Persistent WebSocket for real-time event streaming
//
// Connection lifecycle:
//   1. connectWebSocket() called from main.js init()
//   2. On open: update UI status indicator
//   3. On message: delegate to message handler
//   4. On close: schedule reconnect attempt
//
// TWEAK: CONFIG.WEBSOCKET.RECONNECT_DELAY_MS controls reconnect timing

// Message handler is passed in to avoid circular imports
let messageHandler = null;
let reconnectTimeout = null;
let isConnecting = false;

/**
 * Connect to the Feral WebSocket server
 * Automatically reconnects on disconnect
 *
 * @param {Function} handler - Message handler function (receives parsed JSON)
 *                            Handler is called with: { type: string, data: Object }
 *
 * Message types (defined in feral_server.py):
 *   - init: Initial state on connect
 *   - mind_update: Mind state changed
 *   - activity: New activity event
 *   - node_discovered: New node in constellation
 *   - node_activated: Existing node activated
 *   - smash_hit: Particle smasher event (batched)
 *   - hot_reload: File changed, reload page
 *
 * TWEAK: UI status elements (change IDs in index.html if needed)
 *   - #ws-dot: Connection status indicator
 *   - #ws-status-text: Connection status text
 */
export function connectWebSocket(handler) {
    if (handler) {
        messageHandler = handler;
    }

    // Prevent multiple concurrent connection attempts
    if (isConnecting) {
        return;
    }

    // Clear any pending reconnect
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }

    isConnecting = true;

    // Determine protocol (ws: for http, wss: for https)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    state.setWs(ws);

    // ----- CONNECTION OPENED -----
    ws.onopen = () => {
        isConnecting = false;
        const dot = document.getElementById('ws-dot');
        const text = document.getElementById('ws-status-text');
        if (dot) dot.classList.remove('offline');
        if (text) text.innerText = 'Connected';
    };

    // ----- CONNECTION CLOSED -----
    ws.onclose = () => {
        isConnecting = false;
        const dot = document.getElementById('ws-dot');
        const text = document.getElementById('ws-status-text');
        if (dot) dot.classList.add('offline');
        if (text) text.innerText = 'Disconnected';

        // Schedule reconnect (only if not already scheduled)
        // TWEAK: CONFIG.WEBSOCKET.RECONNECT_DELAY_MS (default: 3000ms)
        if (!reconnectTimeout) {
            reconnectTimeout = setTimeout(() => {
                reconnectTimeout = null;
                connectWebSocket();
            }, CONFIG.WEBSOCKET.RECONNECT_DELAY_MS);
        }
    };

    // ----- CONNECTION ERROR -----
    ws.onerror = () => {
        isConnecting = false;
        // Let onclose handle reconnection
    };

    // ----- MESSAGE RECEIVED -----
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (messageHandler) {
            messageHandler(msg);
        }
    };
}

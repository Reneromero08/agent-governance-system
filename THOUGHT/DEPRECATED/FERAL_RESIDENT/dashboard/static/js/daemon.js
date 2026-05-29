// =============================================================================
// FERAL DASHBOARD - DAEMON CONTROLS
// =============================================================================
//
// This module manages the Feral daemon - the background process that performs
// autonomous behaviors like memory consolidation and self-reflection.
//
// OVERVIEW:
//   The daemon runs on the server and executes configurable behaviors at
//   specified intervals. This module provides the UI controls to:
//   - Start/stop the daemon
//   - Enable/disable individual behaviors
//   - Adjust behavior intervals
//
// BEHAVIORS:
//   - memory_consolidation: Finds patterns across memories
//   - self_reflection: Generates introspective questions
//   - cassette_watch: Monitors content sources
//   - paper_exploration: Explores papers (no UI toggle, always enabled)
//
// SECTIONS IN THIS FILE:
//   1. Imports
//   2. Status Loading
//   3. UI Update Functions
//   4. Daemon Toggle (start/stop)
//   5. Behavior Controls (enable/disable, intervals)
//
// =============================================================================

// =============================================================================
// SECTION 1: IMPORTS
// =============================================================================

import * as state from './state.js';
import { api } from './api.js';
import { updateThresholdDisplay } from './smasher.js';

// Debouncing: Prevent rapid toggle clicks from causing race conditions
let _daemonTogglePending = false;

// =============================================================================
// SECTION 2: STATUS LOADING
// =============================================================================
// Fetch daemon status from server (used on page load and polling)

/**
 * Load daemon status from server
 * Updates UI with current running state and behavior configuration
 *
 * Called:
 *   - On page load (init)
 *   - After any daemon control action
 *   - Every CONFIG.POLLING.DAEMON_INTERVAL_MS (polling)
 */
export async function loadDaemonStatus() {
    try {
        const data = await api('/daemon/status');
        if (data.ok) {
            updateDaemonStatus(data);
        }
    } catch (e) {
        console.error('Failed to load daemon status:', e);
    }
}

// =============================================================================
// SECTION 3: UI UPDATE FUNCTIONS
// =============================================================================
// Updates to the sidebar daemon panel based on state changes

/**
 * Update daemon UI based on status data from server
 *
 * @param {Object} data - Status data from /daemon/status endpoint
 * @param {boolean} data.running - Whether daemon is currently running
 * @param {Object} data.behaviors - Map of behavior name to {enabled, interval}
 * @param {number} data.threshold - Current Q46 absorption threshold
 * @param {number} data.n_memories - Number of absorbed memories (for nucleation formula)
 *
 * TWEAK: LED colors are set via CSS classes (see styles.css .daemon-led)
 */
export function updateDaemonStatus(data) {
    // Update global state
    state.setDaemonRunning(data.running);
    state.setBehaviors(data.behaviors || {});

    // Update toggle button (action-btn style)
    const btn = document.getElementById('daemon-toggle-btn');
    const btnText = document.getElementById('daemon-btn-text');
    if (data.running) {
        btn.classList.add('active');
        btnText.innerText = 'RUNNING';
    } else {
        btn.classList.remove('active');
        btnText.innerText = 'START';
    }

    // Update individual behavior toggles and intervals
    // Note: paper_exploration has no UI toggle (always enabled when daemon runs)
    updateBehaviorUI('memory_consolidation', 'consolidate');
    updateBehaviorUI('self_reflection', 'reflect');
    updateBehaviorUI('cassette_watch', 'cassette');

    // Q46: Update dynamic threshold display
    // The absorption threshold follows nucleation formula based on memory count
    if (data.threshold !== undefined) {
        const n = data.n_memories || 0;
        updateThresholdDisplay(data.threshold, n);
    }
}

/**
 * Update a single behavior's UI elements (toggle and interval input)
 *
 * @param {string} name - Full behavior name (e.g., 'memory_consolidation')
 * @param {string} shortName - Short name for element IDs (e.g., 'consolidate')
 *
 * Element IDs:
 *   - toggle-{shortName}: The toggle button element
 *   - input-{shortName}: The interval input element
 *
 * TWEAK: Toggle colors are set via CSS classes
 *   - .behavior-toggle = off (gray)
 *   - .behavior-toggle.on = on (green)
 */
function updateBehaviorUI(name, shortName) {
    const toggle = document.getElementById(`toggle-${shortName}`);
    const input = document.getElementById(`input-${shortName}`);

    if (!toggle) return;  // No UI element for this behavior

    if (state.behaviors[name]) {
        const enabled = state.behaviors[name].enabled;
        toggle.className = enabled ? 'behavior-toggle on' : 'behavior-toggle';
        if (input) input.value = state.behaviors[name].interval;
    }
}

// =============================================================================
// SECTION 4: DAEMON TOGGLE
// =============================================================================
// Start/stop the daemon process

/**
 * Toggle daemon running state
 * Called when user clicks the Start/Stop button
 *
 * BUG FIX: Added debouncing and error handling to prevent race conditions
 * from rapid clicks and to gracefully handle API failures.
 * Also updates UI immediately for responsive feedback.
 */
export async function toggleDaemon() {
    // Debounce: Ignore clicks while a toggle operation is in progress
    if (_daemonTogglePending) {
        console.log('[DAEMON] Toggle already pending, ignoring click');
        return;
    }

    _daemonTogglePending = true;
    const wasRunning = state.daemonRunning;

    try {
        if (wasRunning) {
            // STOP: Update UI immediately for responsive feedback
            console.log('[DAEMON] Stop requested');
            state.setDaemonRunning(false);
            updateDaemonUIOnly(false);

            await api('/daemon/stop', { method: 'POST' });
            console.log('[DAEMON] Server confirmed stop');
        } else {
            // START: Update UI immediately
            console.log('[DAEMON] Start requested');
            state.setDaemonRunning(true);
            updateDaemonUIOnly(true);

            await api('/daemon/start', { method: 'POST' });
            console.log('[DAEMON] Server confirmed start');
        }
        // Refresh full status from server to sync state
        await loadDaemonStatus();
    } catch (e) {
        console.error('Failed to toggle daemon:', e);
        // Revert UI on error
        state.setDaemonRunning(wasRunning);
        updateDaemonUIOnly(wasRunning);
    } finally {
        _daemonTogglePending = false;
    }
}

/**
 * Update only the daemon button UI (not behaviors)
 * Used for immediate feedback before server response
 */
function updateDaemonUIOnly(running) {
    const btn = document.getElementById('daemon-toggle-btn');
    const btnText = document.getElementById('daemon-btn-text');
    if (running) {
        btn.classList.add('active');
        btnText.innerText = 'RUNNING';
    } else {
        btn.classList.remove('active');
        btnText.innerText = 'START';
    }
}

// =============================================================================
// SECTION 5: BEHAVIOR CONTROLS
// =============================================================================
// Enable/disable behaviors and adjust their intervals

/**
 * Toggle a behavior's enabled state
 * Called when user clicks a behavior toggle button
 *
 * @param {string} name - Behavior name (e.g., 'memory_consolidation')
 */
export async function toggleBehavior(name) {
    if (!state.behaviors[name]) return;

    const newEnabled = !state.behaviors[name].enabled;
    await api('/daemon/config', {
        method: 'POST',
        body: JSON.stringify({ behavior: name, enabled: newEnabled })
    });
    loadDaemonStatus();
}

/**
 * Update a behavior's interval (seconds between executions)
 * Called when user changes an interval input
 *
 * @param {string} name - Behavior name (e.g., 'memory_consolidation')
 * @param {string} seconds - New interval value (from input field)
 *
 * TWEAK: Minimum interval is 5 seconds (safety limit)
 *        Maximum is set in index.html input max attribute
 */
export async function updateInterval(name, seconds) {
    // Use parseFloat first to catch decimal input, then round to integer
    const parsed = parseFloat(seconds);
    if (isNaN(parsed) || parsed < 5) return;  // Minimum 5 second interval

    const interval = Math.round(parsed);  // Round instead of truncate
    await api('/daemon/config', {
        method: 'POST',
        body: JSON.stringify({ behavior: name, interval: interval })
    });
    loadDaemonStatus();
}

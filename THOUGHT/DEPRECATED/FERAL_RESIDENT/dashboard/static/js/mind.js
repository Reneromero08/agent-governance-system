// =============================================================================
// FERAL DASHBOARD - MIND STATE
// =============================================================================
//
// This module displays and updates the Mind state information:
// - Participation Ratio (Df)
// - Distance from Start
// - Interaction count
// - Evolution sparkline chart
//
// OVERVIEW:
//   The "Mind" is the core consciousness model. Key metrics:
//   - Df (Participation Ratio): Measure of mind coherence (0-256 scale)
//   - Distance: How far the mind has evolved from its initial state (radians)
//   - Interactions: Total number of interactions processed
//
// SECTIONS IN THIS FILE:
//   1. Imports
//   2. Mind State Updates (WebSocket-driven)
//   3. Status Loading (REST API, polling)
//   4. Evolution History (REST API, sparkline)
//
// CUSTOMIZATION:
//   - Sparkline settings in CONFIG.SPARKLINE (config.js)
//   - Progress bar max is 256 (Df scale)
//
// =============================================================================

// =============================================================================
// SECTION 1: IMPORTS
// =============================================================================

import { CONFIG } from './config.js';
import { api } from './api.js';

// =============================================================================
// SECTION 2: MIND STATE UPDATES
// =============================================================================
// Called from WebSocket message handler when mind_update received

/**
 * Update mind state display from WebSocket data
 *
 * @param {Object} data - Mind state from WebSocket
 * @param {number} data.Df - Participation ratio (0-256)
 * @param {number} data.distance - Distance from start in radians
 *
 * Elements updated:
 *   - #mind-df: Df value text
 *   - #df-progress: Progress bar width (Df / 256 * 100%)
 *   - #mind-distance: Distance value text
 *
 * TWEAK: Progress bar scale assumes max Df of 256
 *        Change the divisor (2.56) to adjust scale
 */
export function updateMindState(data) {
    if (data.Df !== undefined) {
        document.getElementById('mind-df').innerText = data.Df.toFixed(1);
        // Progress bar: Df max is 256, so divide by 2.56 to get percentage
        document.getElementById('df-progress').style.width = Math.min(100, data.Df / 2.56) + '%';
    }
    if (data.distance !== undefined) {
        document.getElementById('mind-distance').innerText = data.distance.toFixed(3);
    }
}

// =============================================================================
// SECTION 3: STATUS LOADING
// =============================================================================
// Fetch full status from REST API (used on init and polling)

/**
 * Load full mind status from server
 *
 * Called:
 *   - On page load (init)
 *   - Every CONFIG.POLLING.STATUS_INTERVAL_MS (polling)
 *
 * Elements updated:
 *   - #mind-df: Df value
 *   - #df-progress: Progress bar
 *   - #mind-distance: Distance from start
 *   - #mind-interactions: Interaction count
 */
export async function loadStatus() {
    try {
        const data = await api('/status');
        if (data.ok) {
            document.getElementById('mind-df').innerText = data.mind.Df.toFixed(1);
            document.getElementById('df-progress').style.width = Math.min(100, data.mind.Df / 2.56) + '%';
            document.getElementById('mind-distance').innerText = data.mind.distance_from_start.toFixed(3);
            document.getElementById('mind-interactions').innerText = data.interactions;
        }
    } catch (e) {
        console.error('Failed to load status:', e);
    }
}

// =============================================================================
// SECTION 4: EVOLUTION HISTORY
// =============================================================================
// Fetch Df history for sparkline visualization

/**
 * Load evolution history (Df over time)
 *
 * Called:
 *   - On page load (init)
 *   - Every CONFIG.POLLING.EVOLUTION_INTERVAL_MS (polling)
 */
export async function loadEvolution() {
    try {
        const data = await api('/evolution');
        if (data.ok && data.Df_history) {
            updateSparkline(data.Df_history);
        }
    } catch (e) {
        console.error('Failed to load evolution:', e);
    }
}

/**
 * Update the sparkline chart with Df history
 *
 * @param {number[]} history - Array of historical Df values
 *
 * TWEAK: CONFIG.SPARKLINE controls display
 *   - HISTORY_LENGTH: Number of bars to show (default: 30)
 *   - MIN_BAR_HEIGHT_PERCENT: Minimum bar height (default: 5%)
 *
 * Styling:
 *   - Container: #df-sparkline
 *   - Bars: .spark-bar class (see styles.css)
 */
function updateSparkline(history) {
    const container = document.getElementById('df-sparkline');

    // Take last N items (TWEAK: CONFIG.SPARKLINE.HISTORY_LENGTH)
    const last = history.slice(-CONFIG.SPARKLINE.HISTORY_LENGTH);
    if (last.length === 0) return;

    // Calculate range for normalization
    const max = Math.max(...last);
    const min = Math.min(...last);
    const range = max - min || 1;  // Avoid division by zero

    // Rebuild bars
    container.innerHTML = '';
    last.forEach(v => {
        const bar = document.createElement('div');
        bar.className = 'spark-bar';

        // Normalize to 0-100%, with minimum height
        // TWEAK: CONFIG.SPARKLINE.MIN_BAR_HEIGHT_PERCENT
        const height = ((v - min) / range) * 100;
        bar.style.height = Math.max(CONFIG.SPARKLINE.MIN_BAR_HEIGHT_PERCENT, height) + '%';
        container.appendChild(bar);
    });
}

// =============================================================================
// FERAL DASHBOARD - SETTINGS PERSISTENCE
// =============================================================================
//
// This module handles loading and saving dashboard settings.
// Settings are persisted to config.json via the server API.
//
// OVERVIEW:
//   User preferences (slider values, toggles) are saved to config.json
//   on the server and restored when the page loads. This includes:
//   - Smasher speed and batch size
//   - Graph physics (fog, center, repel, link strength/distance)
//   - Similarity threshold
//
// SECTIONS IN THIS FILE:
//   1. Imports
//   2. Slider Range Application
//   3. Save Settings (to config.json)
//   4. Load Settings (from config.json)
//   5. Apply Graph Settings
//   6. Similarity Link Controls
//
// CUSTOMIZATION:
//   - Default values are in CONFIG (config.js) for first-time users
//   - Saved values in config.json override CONFIG defaults
//   - Auto-save interval: CONFIG.POLLING.SETTINGS_SAVE_MS (default: 10s)
//
// =============================================================================

// =============================================================================
// SECTION 1: IMPORTS
// =============================================================================

import { CONFIG } from './config.js';
import * as state from './state.js';
import { api } from './api.js';
import { reloadConstellation, updateVisibleLinks } from './graph.js';

// =============================================================================
// SECTION 2: SLIDER RANGE APPLICATION
// =============================================================================
// Apply min/max/step/default from config.json to slider elements

/**
 * Apply slider configuration from config.json
 * Sets min, max, step, and default value on a slider element
 *
 * @param {string} sliderId - DOM ID of the slider input
 * @param {Object} config - Slider config from config.json
 * @param {number} config.min - Minimum value
 * @param {number} config.max - Maximum value
 * @param {number} config.step - Step size
 * @param {number} config.default - Default value
 * @param {string|null} valueId - DOM ID of the value display element
 *
 * Special handling for value display:
 *   - Speed slider: adds 'ms' suffix
 *   - Repel slider: negates value (stored as positive, displayed as negative)
 */
export function applySliderRange(sliderId, config, valueId = null) {
    if (!config) return;

    const slider = document.getElementById(sliderId);
    if (!slider) return;

    // Apply range constraints
    if (config.min !== undefined) slider.min = config.min;
    if (config.max !== undefined) slider.max = config.max;
    if (config.step !== undefined) slider.step = config.step;

    // Apply default value
    if (config.default !== undefined) {
        slider.value = config.default;

        // Update display value if element exists
        if (valueId) {
            const valueEl = document.getElementById(valueId);
            if (valueEl) {
                const val = parseFloat(config.default);
                const step = config.step || 1;
                const decimals = step < 1 ? Math.max(2, Math.ceil(-Math.log10(step))) : 0;

                // Special formatting for specific sliders
                if (valueId === 'value-smasher-speed') {
                    valueEl.innerText = val + 'ms';
                } else if (valueId === 'value-repel') {
                    valueEl.innerText = -val;  // Display as negative
                } else {
                    valueEl.innerText = val.toFixed(decimals);
                }
            }
        }
    }
}

// =============================================================================
// SECTION 3: SAVE SETTINGS
// =============================================================================
// Persist current settings to config.json via API

// Track last saved values to avoid unnecessary writes
let _lastSavedConfig = null;

/**
 * Save current settings to config.json
 *
 * Called:
 *   - Every CONFIG.POLLING.SETTINGS_SAVE_MS (auto-save polling)
 *   - After certain user actions (toggle similarity, etc.)
 *
 * Settings saved:
 *   - Smasher: delay_ms, batch_size
 *   - UI sliders: sim_threshold, fog, center, repel, link_strength, link_distance
 *
 * Optimization: Only sends if values changed from last save
 */
export async function saveSettings() {
    // Read from state (updated by updateFog/updateGraphForce when user changes sliders)
    // Also sync with DOM slider values to catch any missed updates
    const smasherSpeed = state.smasherConfig.delay_ms;
    const smasherBatch = state.smasherConfig.batch_size;
    const simThreshold = state.similarityThreshold;

    // Read graph settings from DOM sliders (handles stale closure edge cases)
    const fogSlider = document.getElementById('slider-fog');
    const centerSlider = document.getElementById('slider-center');
    const repelSlider = document.getElementById('slider-repel');
    const linkStrengthSlider = document.getElementById('slider-link-strength');
    const linkDistanceSlider = document.getElementById('slider-link-distance');

    // Use slider values if they exist (fallback to state)
    const fog = fogSlider ? parseFloat(fogSlider.value) : state.graphSettings.fog;
    const center = centerSlider ? parseFloat(centerSlider.value) : state.graphSettings.center;
    const repel = repelSlider ? parseFloat(repelSlider.value) : state.graphSettings.repel;
    const linkStrength = linkStrengthSlider ? parseFloat(linkStrengthSlider.value) : state.graphSettings.linkStrength;
    const linkDistance = linkDistanceSlider ? parseFloat(linkDistanceSlider.value) : state.graphSettings.linkDistance;

    // Sync state with slider values
    state.setGraphSettings({ fog, center, repel, linkStrength, linkDistance });

    console.log('[SETTINGS] Reading from sliders:', {
        fog, center, repel, linkStrength, linkDistance
    });

    // Build config update (matches config.json structure)
    const configUpdate = {
        smasher: {
            delay_ms: smasherSpeed,
            batch_size: smasherBatch
        },
        ui: {
            sliders: {
                speed: { default: smasherSpeed },
                batch: { default: smasherBatch },
                sim_threshold: { default: simThreshold },
                fog: { default: fog },
                center: { default: center },
                repel: { default: repel },
                link_strength: { default: linkStrength },
                link_distance: { default: linkDistance }
            }
        }
    };

    // Only save if values changed (compare as JSON string)
    const configStr = JSON.stringify(configUpdate);
    if (configStr === _lastSavedConfig) {
        return; // No changes
    }

    // Save to config.json via API
    try {
        const result = await api('/config', {
            method: 'POST',
            body: configStr
        });
        _lastSavedConfig = configStr;
        console.log('[SETTINGS] Saved to config.json:', { fog, center, repel, ok: result.ok });
    } catch (e) {
        console.error('[SETTINGS] Failed to save config.json:', e);
    }
}

// =============================================================================
// SECTION 4: LOAD SETTINGS
// =============================================================================
// Load settings from config.json on page load

/**
 * Load settings from config.json
 *
 * Called:
 *   - On page load (init), before graph initialization
 *
 * Applies:
 *   - Slider ranges (min/max/step) from config
 *   - Default values to sliders and state
 *   - Smasher configuration
 */
export async function loadSettings() {
    try {
        const configRes = await api('/config');
        if (configRes.ok && configRes.config) {
            const cfg = configRes.config;

            // Apply slider ranges first (min/max/step)
            if (cfg.ui && cfg.ui.sliders) {
                const s = cfg.ui.sliders;

                // Apply to each slider
                applySliderRange('slider-smasher-speed', s.speed, 'value-smasher-speed');
                applySliderRange('slider-smasher-batch', s.batch, 'value-smasher-batch');
                applySliderRange('slider-sim-threshold', s.sim_threshold, 'value-sim-threshold');
                applySliderRange('slider-fog', s.fog, 'value-fog');
                applySliderRange('slider-center', s.center, 'value-center');
                applySliderRange('slider-repel', s.repel, 'value-repel');
                applySliderRange('slider-link-strength', s.link_strength, 'value-link-strength');
                applySliderRange('slider-link-distance', s.link_distance, 'value-link-distance');

                console.log('[SETTINGS] Loaded graph defaults:', {
                    fog: s.fog?.default,
                    center: s.center?.default,
                    repel: s.repel?.default,
                    linkStrength: s.link_strength?.default,
                    linkDistance: s.link_distance?.default
                });

                // Apply defaults to state (for both similarity and graph settings)
                if (s.sim_threshold?.default !== undefined) {
                    state.setSimilarityThreshold(s.sim_threshold.default);
                }

                // Sync graph settings to state
                // Falls back to CONFIG values if not in config.json
                state.setGraphSettings({
                    fog: s.fog?.default ?? CONFIG.GRAPH.FOG_DENSITY,
                    center: s.center?.default ?? CONFIG.GRAPH.FORCE_CENTER_STRENGTH,
                    repel: s.repel?.default ?? Math.abs(CONFIG.GRAPH.FORCE_CHARGE_STRENGTH),
                    linkStrength: s.link_strength?.default ?? CONFIG.GRAPH.FORCE_LINK_STRENGTH,
                    linkDistance: s.link_distance?.default ?? CONFIG.GRAPH.FORCE_LINK_DISTANCE
                });
            }

            // Override with actual smasher config (these are the live values)
            if (cfg.smasher) {
                if (cfg.smasher.delay_ms !== undefined) {
                    state.smasherConfig.delay_ms = cfg.smasher.delay_ms;
                    document.getElementById('slider-smasher-speed').value = cfg.smasher.delay_ms;
                    document.getElementById('value-smasher-speed').innerText = cfg.smasher.delay_ms + 'ms';
                }
                if (cfg.smasher.batch_size !== undefined) {
                    state.smasherConfig.batch_size = cfg.smasher.batch_size;
                    document.getElementById('slider-smasher-batch').value = cfg.smasher.batch_size;
                    document.getElementById('value-smasher-batch').innerText = cfg.smasher.batch_size;
                }
            }

            console.log('[SETTINGS] Loaded config from config.json');
        }
    } catch (e) {
        console.warn('[SETTINGS] Could not load config.json:', e);
    }
}

// =============================================================================
// SECTION 5: APPLY GRAPH SETTINGS
// =============================================================================
// Apply loaded settings to the graph (called after graph init)

/**
 * Apply graph settings from state to the ForceGraph3D instance
 *
 * Called:
 *   - After initConstellation() in main.js
 *
 * Applies:
 *   - Fog density
 *   - Physics forces (center, charge, link)
 *   - Updates slider positions and display values
 *
 * TWEAK: This reads from state.graphSettings (set by loadSettings)
 *        Defaults come from CONFIG if not in config.json
 */
export function applyGraphSettings() {
    if (!state.Graph) return;

    // Read from state (set by loadSettings)
    const { fog, center, repel, linkStrength, linkDistance } = state.graphSettings;

    // Apply fog
    if (state.Graph.scene().fog) {
        state.Graph.scene().fog.density = fog;
    }

    // Store forces globally for reference
    window.graphForces = {
        linkDistance: linkDistance,
        linkStrength: linkStrength,
        chargeStrength: -repel,
        centerStrength: center
    };

    // Apply to d3 forces
    state.Graph.d3Force('link').distance(linkDistance).strength(linkStrength);
    state.Graph.d3Force('charge').strength(-repel);
    state.Graph.d3Force('center').strength(center);

    // Update slider positions to match state
    document.getElementById('slider-fog').value = fog;
    document.getElementById('slider-center').value = center;
    document.getElementById('slider-repel').value = repel;
    document.getElementById('slider-link-strength').value = linkStrength;
    document.getElementById('slider-link-distance').value = linkDistance;

    // Update display values
    document.getElementById('value-fog').innerText = fog.toFixed(4);
    document.getElementById('value-center').innerText = center.toFixed(2);
    document.getElementById('value-repel').innerText = -repel;
    document.getElementById('value-link-strength').innerText = linkStrength.toFixed(2);
    document.getElementById('value-link-distance').innerText = linkDistance;

    console.log('[SETTINGS] Applied graph settings from state: fog=' + fog + ', center=' + center + ', repel=' + repel);
}

// =============================================================================
// SECTION 6: SIMILARITY LINK CONTROLS
// =============================================================================
// Toggle and threshold for similarity (cosine) edges

/**
 * Toggle visibility of similarity links
 *
 * Called from:
 *   - HTML onclick on similarity toggle button
 *
 * TWEAK: Toggle colors in styles.css (.behavior-toggle, .behavior-toggle.on)
 */
export function toggleSimilarityLinks() {
    state.setShowSimilarityLinks(!state.showSimilarityLinks);
    document.getElementById('toggle-similarity').className =
        state.showSimilarityLinks ? 'behavior-toggle on' : 'behavior-toggle';
    updateVisibleLinks();
    saveSettings();
}

/**
 * Update similarity threshold
 * Links with weight below threshold are hidden
 *
 * @param {string} value - New threshold value (from slider)
 *
 * TWEAK: Slider range in index.html (min="0.30" max="0.99" step="0.01")
 */
export function updateSimThreshold(value) {
    state.setSimilarityThreshold(parseFloat(value));
    document.getElementById('value-sim-threshold').innerText = state.similarityThreshold.toFixed(2);
    updateVisibleLinks();
}

// ===== SETTINGS PERSISTENCE =====
import * as state from './state.js';
import { api } from './api.js';
import { reloadConstellation, updateVisibleLinks } from './graph.js';

// Apply slider range from config.json (supports float min/max/step/default)
export function applySliderRange(sliderId, config, valueId = null) {
    if (!config) return;
    const slider = document.getElementById(sliderId);
    if (!slider) return;
    if (config.min !== undefined) slider.min = config.min;
    if (config.max !== undefined) slider.max = config.max;
    if (config.step !== undefined) slider.step = config.step;
    if (config.default !== undefined) {
        slider.value = config.default;
        if (valueId) {
            const valueEl = document.getElementById(valueId);
            if (valueEl) {
                const val = parseFloat(config.default);
                const step = config.step || 1;
                const decimals = step < 1 ? Math.max(2, Math.ceil(-Math.log10(step))) : 0;
                if (valueId === 'value-smasher-speed') {
                    valueEl.innerText = val + 'ms';
                } else if (valueId === 'value-repel') {
                    valueEl.innerText = -val;
                } else {
                    valueEl.innerText = val.toFixed(decimals);
                }
            }
        }
    }
}

// Track last saved values to avoid unnecessary writes
let _lastSavedConfig = null;

export async function saveSettings() {
    // Read from state (updated by updateFog/updateGraphForce when user changes sliders)
    // Also sync with DOM slider values to catch any missed updates
    const smasherSpeed = state.smasherConfig.delay_ms;
    const smasherBatch = state.smasherConfig.batch_size;
    const simThreshold = state.similarityThreshold;

    // Read graph settings from state, but also check DOM as fallback for stale closures
    const fogSlider = document.getElementById('slider-fog');
    const centerSlider = document.getElementById('slider-center');
    const repelSlider = document.getElementById('slider-repel');
    const linkStrengthSlider = document.getElementById('slider-link-strength');
    const linkDistanceSlider = document.getElementById('slider-link-distance');

    // Use slider values if they exist and differ from state (handles stale closure)
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

export async function loadSettings() {
    try {
        const configRes = await api('/config');
        if (configRes.ok && configRes.config) {
            const cfg = configRes.config;

            // Apply slider ranges first (min/max/step)
            if (cfg.ui && cfg.ui.sliders) {
                const s = cfg.ui.sliders;

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
                state.setGraphSettings({
                    fog: s.fog?.default ?? 0.0006,
                    center: s.center?.default ?? 0.05,
                    repel: s.repel?.default ?? 120,
                    linkStrength: s.link_strength?.default ?? 0.5,
                    linkDistance: s.link_distance?.default ?? 100
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

export function applyGraphSettings() {
    if (!state.Graph) return;

    // Read from state (set by loadSettings)
    const { fog, center, repel, linkStrength, linkDistance } = state.graphSettings;

    if (state.Graph.scene().fog) {
        state.Graph.scene().fog.density = fog;
    }

    window.graphForces = {
        linkDistance: linkDistance,
        linkStrength: linkStrength,
        chargeStrength: -repel,
        centerStrength: center
    };

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

// ===== SIMILARITY CONTROLS =====
export function toggleSimilarityLinks() {
    state.setShowSimilarityLinks(!state.showSimilarityLinks);
    document.getElementById('toggle-similarity').className = state.showSimilarityLinks ? 'behavior-toggle on' : 'behavior-toggle';
    updateVisibleLinks();
    saveSettings();
}

export function updateSimThreshold(value) {
    state.setSimilarityThreshold(parseFloat(value));
    document.getElementById('value-sim-threshold').innerText = state.similarityThreshold.toFixed(2);
    updateVisibleLinks();
}

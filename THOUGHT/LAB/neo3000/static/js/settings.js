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

export function saveSettings() {
    const settings = {
        smasher_delay_ms: state.smasherConfig.delay_ms,
        smasher_batch_size: state.smasherConfig.batch_size,
        similarity_threshold: state.similarityThreshold,
        show_similarity_links: state.showSimilarityLinks,
        static_camera: state.staticCameraMode,
        fog_density: state.Graph ? parseFloat(document.getElementById('value-fog').innerText) : 0.003,
        center_strength: state.Graph ? parseFloat(document.getElementById('value-center').innerText) : 0.05,
        repel_strength: state.Graph ? parseFloat(document.getElementById('value-repel').innerText) : -120,
        link_strength: state.Graph ? parseFloat(document.getElementById('value-link-strength').innerText) : 0.5,
        link_distance: state.Graph ? parseFloat(document.getElementById('value-link-distance').innerText) : 100
    };

    localStorage.setItem('feral_settings', JSON.stringify(settings));
}

export async function loadSettings() {
    try {
        const configRes = await api('/config');
        if (configRes.ok && configRes.config) {
            const cfg = configRes.config;

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

            if (cfg.ui && cfg.ui.sliders) {
                const s = cfg.ui.sliders;
                localStorage.removeItem('feral_settings');

                applySliderRange('slider-smasher-speed', s.speed, 'value-smasher-speed');
                applySliderRange('slider-smasher-batch', s.batch, 'value-smasher-batch');
                applySliderRange('slider-sim-threshold', s.sim_threshold, 'value-sim-threshold');
                applySliderRange('slider-fog', s.fog, 'value-fog');
                applySliderRange('slider-center', s.center, 'value-center');
                applySliderRange('slider-repel', s.repel, 'value-repel');
                applySliderRange('slider-link-strength', s.link_strength, 'value-link-strength');
                applySliderRange('slider-link-distance', s.link_distance, 'value-link-distance');

                if (s.sim_threshold?.default !== undefined) {
                    state.setSimilarityThreshold(s.sim_threshold.default);
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

    const fog = parseFloat(document.getElementById('slider-fog').value);
    const center = parseFloat(document.getElementById('slider-center').value);
    const repel = parseFloat(document.getElementById('slider-repel').value);
    const linkStrength = parseFloat(document.getElementById('slider-link-strength').value);
    const linkDistance = parseFloat(document.getElementById('slider-link-distance').value);

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

    document.getElementById('value-fog').innerText = fog.toFixed(4);
    document.getElementById('value-center').innerText = center.toFixed(2);
    document.getElementById('value-repel').innerText = -repel;
    document.getElementById('value-link-strength').innerText = linkStrength.toFixed(2);
    document.getElementById('value-link-distance').innerText = linkDistance;

    console.log('[SETTINGS] Applied graph settings: fog=' + fog + ', center=' + center + ', repel=' + repel);
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
    reloadConstellation();
}

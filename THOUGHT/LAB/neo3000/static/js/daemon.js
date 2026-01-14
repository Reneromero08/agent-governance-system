// ===== DAEMON CONTROLS =====
import * as state from './state.js';
import { api } from './api.js';
import { updateThresholdDisplay } from './smasher.js';

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

export function updateDaemonStatus(data) {
    state.setDaemonRunning(data.running);
    state.setBehaviors(data.behaviors || {});

    document.getElementById('daemon-status-text').innerText = data.running ? 'Running' : 'Stopped';
    document.getElementById('daemon-led').className = data.running ? 'daemon-led on' : 'daemon-led';
    const btn = document.getElementById('daemon-toggle-btn');
    btn.innerText = data.running ? 'Stop' : 'Start';
    btn.className = data.running ? 'daemon-btn' : 'daemon-btn primary';

    // paper_exploration has no UI toggle
    updateBehaviorUI('memory_consolidation', 'consolidate');
    updateBehaviorUI('self_reflection', 'reflect');
    updateBehaviorUI('cassette_watch', 'cassette');

    // Q46: Update dynamic threshold display
    if (data.threshold !== undefined) {
        const n = data.explored_chunks || 0;
        updateThresholdDisplay(data.threshold, n);
    }
}

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

export async function toggleDaemon() {
    if (state.daemonRunning) {
        await api('/daemon/stop', { method: 'POST' });
    } else {
        await api('/daemon/start', { method: 'POST' });
    }
    loadDaemonStatus();
}

export async function toggleBehavior(name) {
    if (!state.behaviors[name]) return;
    const newEnabled = !state.behaviors[name].enabled;
    await api('/daemon/config', {
        method: 'POST',
        body: JSON.stringify({ behavior: name, enabled: newEnabled })
    });
    loadDaemonStatus();
}

export async function updateInterval(name, seconds) {
    const interval = parseInt(seconds, 10);
    if (isNaN(interval) || interval < 5) return;
    await api('/daemon/config', {
        method: 'POST',
        body: JSON.stringify({ behavior: name, interval: interval })
    });
    loadDaemonStatus();
}

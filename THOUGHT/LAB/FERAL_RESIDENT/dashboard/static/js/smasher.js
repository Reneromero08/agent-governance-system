// ===== PARTICLE SMASHER =====
import * as state from './state.js';
import { api } from './api.js';
import { flashNode, addToTrail, focusCameraOnNode, hideSmasherCursor } from './graph.js';
import { saveSettings } from './settings.js';

export async function toggleSmasher() {
    if (state.smasherActive) {
        await stopSmasher();
    } else {
        await startSmasher();
    }
}

export async function startSmasher() {
    try {
        clearCurrentFile();
        const res = await api('/smasher/start', {
            method: 'POST',
            body: JSON.stringify({
                delay_ms: state.smasherConfig.delay_ms,
                batch_size: state.smasherConfig.batch_size,
                batch_pause_ms: 200,
                max_chunks: 0
            })
        });
        if (res.ok) {
            state.setSmasherActive(true);
            updateSmasherUI();
        }
    } catch (e) {
        console.error('Failed to start smasher:', e);
    }
}

export async function stopSmasher() {
    try {
        // Immediately clear the local visualization queue to prevent lingering animations
        state.smashQueue.length = 0;
        state.setSmashRafPending(false);

        await api('/smasher/stop', { method: 'POST' });
        state.setSmasherActive(false);
        updateSmasherUI();
        hideSmasherCursor();  // Hide the 3D cursor when stopped
        clearCurrentFile();   // Clear the current file display
    } catch (e) {
        console.error('Failed to stop smasher:', e);
    }
}

export function updateSmasherUI() {
    const led = document.getElementById('smasher-led');
    const text = document.getElementById('smasher-status-text');
    const btn = document.getElementById('smasher-toggle-btn');
    const stats = document.getElementById('smasher-stats');
    const currentFile = document.getElementById('smasher-current');

    if (state.smasherActive) {
        led.className = 'daemon-led on';
        text.innerText = 'SMASHING';
        btn.innerText = 'STOP';
        btn.className = 'daemon-btn';
        stats.style.display = 'block';
        // Don't touch currentFile here - let updateCurrentFile handle it
        // currentFile.classList.add('active');
    } else {
        led.className = 'daemon-led';
        text.innerText = 'Idle';
        btn.innerText = 'SMASH';
        btn.className = 'daemon-btn primary';
        // Only hide if NOT receiving smash updates (clearCurrentFile handles this on stop)
        // currentFile.classList.remove('active');
    }
}

export function updateSmasherStats(rate) {
    document.getElementById('smasher-rate').innerText = rate.toFixed(1);
}

export function updateCurrentFile(data) {
    const container = document.getElementById('smasher-current');
    const fileEl = document.getElementById('smasher-current-file');
    const eEl = document.getElementById('smasher-current-e');

    if (!container || !fileEl || !eEl) {
        console.error('[updateCurrentFile] Missing elements!', { container, fileEl, eEl });
        return;
    }

    container.classList.add('active');
    console.log('[updateCurrentFile] Showing:', data.node_id);

    const nodeId = data.node_id || '';
    const parts = nodeId.split(':');
    const fileName = parts.length > 2 ? `${parts[1]}/${parts[2]}` : nodeId;

    fileEl.innerText = fileName;
    fileEl.title = nodeId;

    const E = data.E || 0;
    const gateOpen = data.gate_open;
    eEl.innerText = `E=${E.toFixed(2)} ${gateOpen ? 'ABSORBED' : 'REJECTED'}`;
    eEl.className = `smasher-current-e ${gateOpen ? 'open' : 'closed'}`;
}

export function clearCurrentFile() {
    const container = document.getElementById('smasher-current');
    container.classList.remove('active');
    document.getElementById('smasher-current-file').innerText = '--';
    document.getElementById('smasher-current-e').innerText = '';
}

export function queueSmashVisualization(data) {
    state.smashQueue.push(data);
    if (!state.smashRafPending) {
        state.setSmashRafPending(true);
        requestAnimationFrame(processSmashQueue);
    }
}

function processSmashQueue() {
    state.setSmashRafPending(false);
    const batch = state.smashQueue.splice(0, state.MAX_SMASH_BATCH);
    if (batch.length === 0 || !state.Graph) return;

    const graphData = state.Graph.graphData();
    let graphUpdated = false;
    const nodesToFlash = [];

    for (const data of batch) {
        const result = processSmashItem(data, graphData);
        if (result.updated) graphUpdated = true;
        if (result.node) nodesToFlash.push({ nodeId: data.node_id, gateOpen: data.gate_open, E: data.E });
    }

    if (graphUpdated) {
        state.Graph.graphData(graphData);
    }

    for (const item of nodesToFlash) {
        flashNode(item.nodeId, item.gateOpen, item.E);
    }

    if (state.smashQueue.length > 0) {
        state.setSmashRafPending(true);
        requestAnimationFrame(processSmashQueue);
    }
}

function processSmashItem(data, graphData) {
    const nodeId = data.node_id;
    const similarTo = data.similar_to;
    const similarE = data.similar_E || 0;
    let updated = false;

    let node = state.nodeRegistry.byId.get(nodeId);

    if (!node && data.is_new_node) {
        let pos = { x: 0, y: 0, z: 0 };
        let foundAnchor = false;

        if (similarTo) {
            const anchorNode = state.nodeRegistry.byId.get(similarTo);
            if (anchorNode && anchorNode.x !== undefined) {
                const offset = 15 + (1 - similarE) * 25;
                pos = {
                    x: anchorNode.x + (Math.random() - 0.5) * offset,
                    y: anchorNode.y + (Math.random() - 0.5) * offset,
                    z: anchorNode.z + (Math.random() - 0.5) * offset
                };
                foundAnchor = true;
            }
        }

        if (!foundAnchor) {
            pos = {
                x: (Math.random() - 0.5) * 30,
                y: (Math.random() - 0.5) * 30,
                z: (Math.random() - 0.5) * 30
            };
        }

        const parts = nodeId.split(':');
        const label = parts.length > 2 ? parts[2] : 'chunk';

        node = {
            id: nodeId,
            label: label,
            group: 'page',
            val: 1,
            x: pos.x,
            y: pos.y,
            z: pos.z
        };

        state.nodeRegistry.byId.set(nodeId, node);
        graphData.nodes.push(node);
        updated = true;

        state.nodePulseState.set(nodeId, {
            intensity: 1.5,
            phase: 0,
            frequency: 2.0,
            lastActivity: Date.now()
        });
    }

    if (similarTo && similarE > 0.3) {
        const hasAnchor = state.nodeRegistry.byId.has(similarTo);
        const hasTarget = state.nodeRegistry.byId.has(nodeId);

        if (hasAnchor && hasTarget) {
            const edgeExists = graphData.links.some(l =>
                ((l.source.id || l.source) === similarTo) &&
                ((l.target.id || l.target) === nodeId) &&
                l.type === 'similarity'
            );
            if (!edgeExists) {
                graphData.links.push({
                    source: similarTo,
                    target: nodeId,
                    type: 'similarity',
                    weight: similarE
                });
                updated = true;
            }
        }
    }

    addToTrail(nodeId, 'smash');
    return { updated, node };
}

export async function updateSmasherSpeed(value) {
    state.smasherConfig.delay_ms = parseInt(value);
    document.getElementById('value-smasher-speed').innerText = value + 'ms';
    await sendSmasherConfig({ delay_ms: state.smasherConfig.delay_ms });
}

export async function updateSmasherBatch(value) {
    state.smasherConfig.batch_size = parseInt(value);
    document.getElementById('value-smasher-batch').innerText = value;
    await sendSmasherConfig({ batch_size: state.smasherConfig.batch_size });
}

async function sendSmasherConfig(updates) {
    try {
        await api('/smasher/config', {
            method: 'POST',
            body: JSON.stringify(updates)
        });
    } catch (e) {
        console.error('Failed to update smasher config:', e);
    }
}

export function toggleStaticCamera() {
    state.setStaticCameraMode(!state.staticCameraMode);
    document.getElementById('toggle-static-camera').className =
        state.staticCameraMode ? 'behavior-toggle on' : 'behavior-toggle';
    saveSettings();
}

export async function loadSmasherStatus() {
    try {
        const res = await api('/smasher/status');
        if (res.ok) {
            state.setSmasherActive(res.active);
            if (res.stats) {
                document.getElementById('smasher-processed').innerText = res.stats.chunks_processed;
                document.getElementById('smasher-absorbed').innerText = res.stats.chunks_absorbed;
                document.getElementById('smasher-rejected').innerText = res.stats.chunks_rejected;
                document.getElementById('smasher-rate').innerText = res.stats.chunks_per_second.toFixed(1);
            }
            updateSmasherUI();
        }
    } catch (e) {
        // Ignore
    }
}

export function updateThresholdDisplay(threshold, n) {
    state.setCurrentThreshold(threshold);
    state.setNMemories(n);
    document.getElementById('value-e-threshold').innerText = threshold.toFixed(3);
    document.getElementById('value-n-memories').innerText = n;
    const maxThreshold = 1.0 / (2.0 * Math.PI);
    const progress = Math.min(100, (threshold / maxThreshold) * 100);
    document.getElementById('threshold-progress').style.width = progress + '%';
}

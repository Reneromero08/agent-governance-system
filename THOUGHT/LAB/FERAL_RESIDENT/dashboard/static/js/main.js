// ===== MAIN ENTRY POINT =====
// ES6 Modules for Feral Daemon UI

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
    updateFog, updateGraphForce, resetGraphForces, reloadConstellation
} from './graph.js';
import { loadSettings, saveSettings, applyGraphSettings, toggleSimilarityLinks, updateSimThreshold } from './settings.js';
import { sendMessage } from './chat.js';

// ===== WEBSOCKET MESSAGE HANDLER =====
function handleWebSocketMessage(msg) {
    // Debug: log all messages
    if (msg.type !== 'mind_update') {
        console.log('[WS]', msg.type, msg.data ? Object.keys(msg.data) : 'no data');
    }

    if (msg.type === 'init') {
        updateMindState(msg.data.mind);
        updateDaemonStatus(msg.data.daemon);
    } else if (msg.type === 'mind_update') {
        updateMindState(msg.data);
    } else if (msg.type === 'activity') {
        addActivity(msg.data);
    } else if (msg.type === 'node_discovered') {
        spawnNode(msg.data);
        addToTrail(msg.data.node_id, msg.data.activity_type);
    } else if (msg.type === 'node_activated') {
        activateNode(msg.data.node_id, msg.data.activity_type);
        addToTrail(msg.data.node_id, msg.data.activity_type);
    } else if (msg.type === 'smash_hit') {
        // BATCHED: Server sends batched smash events to reduce WebSocket flood
        const batch = msg.data.batch || [msg.data];
        const batchSize = batch.length;

        // Only log if small batch (avoid console spam)
        if (batchSize <= 3) {
            console.log('[SMASH_HIT]', msg.data.node_id, 'E=', msg.data.E);
        }

        // Update UI with latest hit only (not every item in batch)
        updateCurrentFile(msg.data);
        updateSmasherStats(msg.data.rate);

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
    } else if (msg.type === 'hot_reload') {
        console.log('[HOT RELOAD] File changed, reloading...');
        window.location.reload();
    }
}

// ===== INIT =====
async function init() {
    await loadSettings();
    await initConstellation();
    applyGraphSettings();
    await loadStatus();
    await loadEvolution();
    await loadDaemonStatus();
    await loadSmasherStatus();
    connectWebSocket(handleWebSocketMessage);
    document.getElementById('loading').classList.add('hidden');

    // Polling intervals (reduced frequency - WebSocket handles real-time updates)
    setInterval(loadStatus, 10000);
    setInterval(loadEvolution, 30000);
    setInterval(loadDaemonStatus, 5000);
    setInterval(loadSmasherStatus, 2000);  // Reduced from 1000ms
    setInterval(saveSettings, 10000);       // Reduced from 5000ms
}

// ===== EXPOSE TO GLOBAL SCOPE FOR HTML onclick =====
// These functions are called from HTML onclick handlers
window.toggleSidebar = toggleSidebar;
window.toggleSection = toggleSection;
window.toggleChat = toggleChat;
window.toggleDaemon = toggleDaemon;
window.toggleBehavior = toggleBehavior;
window.updateInterval = updateInterval;
window.toggleSmasher = toggleSmasher;
window.updateSmasherSpeed = updateSmasherSpeed;
window.updateSmasherBatch = updateSmasherBatch;
window.toggleStaticCamera = toggleStaticCamera;
window.toggleSimilarityLinks = toggleSimilarityLinks;
window.updateSimThreshold = updateSimThreshold;
window.updateFog = updateFog;
window.updateGraphForce = updateGraphForce;
window.resetGraphForces = resetGraphForces;
window.sendMessage = sendMessage;
window.saveSettings = saveSettings;

// Start on load
window.onload = init;

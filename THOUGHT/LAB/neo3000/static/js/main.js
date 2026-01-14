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
        updateCurrentFile(msg.data);
        updateSmasherStats(msg.data.rate);
        queueSmashVisualization(msg.data);
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

    // Polling intervals
    setInterval(loadStatus, 10000);
    setInterval(loadEvolution, 30000);
    setInterval(loadDaemonStatus, 5000);
    setInterval(loadSmasherStatus, 1000);
    setInterval(saveSettings, 5000);
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

// Start on load
window.onload = init;

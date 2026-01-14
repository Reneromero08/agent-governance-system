// ===== SHARED STATE =====
// Central state store - all modules import from here

export let ws = null;
export let daemonRunning = false;
export let behaviors = {};
export let chatOpen = false;
export let sidebarCollapsed = false;
export let showSimilarityLinks = true;
export let similarityThreshold = 0.7;
export let allLinks = [];

// Particle Smasher state
export let smasherActive = false;
export let smasherConfig = { delay_ms: 100, batch_size: 10 };
export let currentThreshold = 0.080;
export let nMemories = 0;
export let staticCameraMode = true;

// Render throttling
export let smashQueue = [];
export let smashRafPending = false;
export const MAX_SMASH_BATCH = 5;

// 3D Graph state
export let Graph = null;
export let nodeRegistry = { byId: new Map() };
export let nodePulseState = new Map();
export let explorationTrail = [];
export let trailLine = null;
export const MAX_TRAIL_LENGTH = 50;

export const ACTIVITY_COLORS = {
    paper: { main: 0x00ff41, glow: '#00ff41' },
    consolidate: { main: 0x00ff41, glow: '#00ff41' },
    reflect: { main: 0x00ff41, glow: '#00ff41' },
    cassette: { main: 0x00ff41, glow: '#00ff41' },
    daemon: { main: 0x00ff41, glow: '#00ff41' },
    default: { main: 0x00ff41, glow: '#00ff41' }
};

// State setters (since ES6 exports are read-only bindings)
export function setWs(val) { ws = val; }
export function setDaemonRunning(val) { daemonRunning = val; }
export function setBehaviors(val) { behaviors = val; }
export function setChatOpen(val) { chatOpen = val; }
export function setSidebarCollapsed(val) { sidebarCollapsed = val; }
export function setShowSimilarityLinks(val) { showSimilarityLinks = val; }
export function setSimilarityThreshold(val) { similarityThreshold = val; }
export function setAllLinks(val) { allLinks = val; }
export function setSmasherActive(val) { smasherActive = val; }
export function setSmasherConfig(val) { smasherConfig = val; }
export function setCurrentThreshold(val) { currentThreshold = val; }
export function setNMemories(val) { nMemories = val; }
export function setStaticCameraMode(val) { staticCameraMode = val; }
export function setSmashQueue(val) { smashQueue = val; }
export function setSmashRafPending(val) { smashRafPending = val; }
export function setGraph(val) { Graph = val; }
export function setTrailLine(val) { trailLine = val; }

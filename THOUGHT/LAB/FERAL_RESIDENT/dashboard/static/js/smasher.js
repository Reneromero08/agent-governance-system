// =============================================================================
// FERAL DASHBOARD - PARTICLE SMASHER
// =============================================================================
//
// This module controls the "particle smasher" - the system that analyzes
// memory chunks for absorption into the mind constellation.
//
// OVERVIEW:
//   The smasher iterates through memory chunks, computing similarity (E values)
//   and deciding whether to "absorb" (integrate into mind) or "reject" each chunk.
//   Results are visualized in real-time on the 3D constellation graph.
//
// ARCHITECTURE:
//   - Server API calls control start/stop and configuration
//   - WebSocket delivers smash_hit events with E values and gate status
//   - Visualization queue batches updates for smooth animation
//   - Graph integration creates nodes and similarity edges dynamically
//
// CUSTOMIZATION:
//   Most tunable values are in config.js under CONFIG.SMASHER
//   See those settings for queue management and positioning parameters.
//
// SECTIONS IN THIS FILE:
//   1. Imports
//   2. Toggle/Start/Stop Controls
//   3. UI Update Functions
//   4. Current File Display
//   5. Visualization Queue (batched graph updates)
//   6. Smash Item Processing (node creation, edge creation)
//   7. Configuration Controls (speed, batch size)
//   8. Status Loading
//   9. Threshold Display (Q46 nucleation)
//
// =============================================================================

// =============================================================================
// SECTION 1: IMPORTS
// =============================================================================

import { CONFIG } from './config.js';
import * as state from './state.js';
import { api } from './api.js';
import { flashNode, addToTrail, focusCameraOnNode, hideSmasherCursor, invalidateConstellationCache, reloadConstellation } from './graph.js';
import { saveSettings } from './settings.js';

// Track smasher state for cache invalidation
let _lastSmasherActive = false;
let _lastAbsorbedCount = 0;

// Debouncing: Prevent rapid toggle clicks from causing race conditions
let _smasherTogglePending = false;

// =============================================================================
// SECTION 2: TOGGLE/START/STOP CONTROLS
// =============================================================================
// Main controls for the particle smasher. Called from UI buttons.

/**
 * Toggle smasher on/off
 * Called when user clicks the SMASH/STOP button
 *
 * BUG FIX: Added debouncing to prevent race conditions from rapid clicks.
 * While an API request is in-flight, additional clicks are ignored.
 */
export async function toggleSmasher() {
    // Debounce: Ignore clicks while a toggle operation is in progress
    if (_smasherTogglePending) {
        console.log('[SMASHER] Toggle already pending, ignoring click');
        return;
    }

    _smasherTogglePending = true;
    try {
        if (state.smasherActive) {
            await stopSmasher();
        } else {
            await startSmasher();
        }
    } finally {
        _smasherTogglePending = false;
    }
}

/**
 * Start the particle smasher
 * Sends configuration to server and begins chunk processing
 *
 * Configuration sent:
 *   - delay_ms: Time between chunks (from slider)
 *   - batch_size: Chunks per batch (from slider)
 *   - batch_pause_ms: Pause between batches (hardcoded 200ms)
 *   - max_chunks: 0 = unlimited
 */
export async function startSmasher() {
    try {
        clearCurrentFile();
        const res = await api('/smasher/start', {
            method: 'POST',
            body: JSON.stringify({
                delay_ms: state.smasherConfig.delay_ms,
                batch_size: state.smasherConfig.batch_size,
                batch_pause_ms: 200,  // Fixed pause between batches
                max_chunks: 0         // 0 = process all available chunks
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

/**
 * Stop the particle smasher
 * Clears local queue immediately for responsive UI
 */
export async function stopSmasher() {
    try {
        // Immediately clear local visualization queue to prevent lingering animations
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

// =============================================================================
// SECTION 3: UI UPDATE FUNCTIONS
// =============================================================================
// Updates to the sidebar smasher panel based on state changes

/**
 * Update smasher UI based on current state
 * Controls: LED indicator, status text, button style, stats visibility
 *
 * TWEAK: LED colors are set via CSS classes (see styles.css .daemon-led)
 */
export function updateSmasherUI() {
    const btn = document.getElementById('smasher-toggle-btn');
    const btnText = document.getElementById('smasher-btn-text');
    const iconLeft = document.getElementById('smasher-icon-left');
    const iconRight = document.getElementById('smasher-icon-right');
    const stats = document.getElementById('smasher-stats');

    if (state.smasherActive) {
        btn.classList.add('active');
        btnText.innerText = 'SMASHING';
        // Show both icons with smash_2_green (green on orange button)
        if (iconLeft) iconLeft.src = '/static/icons/smash_2_green.svg';
        if (iconRight) {
            iconRight.src = '/static/icons/smash_2_green.svg';
            iconRight.classList.remove('hidden');
        }
        stats.style.display = 'block';
    } else {
        btn.classList.remove('active');
        btnText.innerText = 'SMASH';
        // Show only left icon with smash_1
        if (iconLeft) iconLeft.src = '/static/icons/smash_1.svg';
        if (iconRight) iconRight.classList.add('hidden');
    }
}

/**
 * Update throughput rate display
 * @param {number} rate - Chunks per second
 */
export function updateSmasherStats(rate) {
    document.getElementById('smasher-rate').innerText = rate.toFixed(1);
}

// =============================================================================
// SECTION 4: CURRENT FILE DISPLAY
// =============================================================================
// Shows the currently-being-analyzed chunk in the activity bar

/**
 * Update the "ANALYZING:" display in the activity bar
 * Shows: filename, E value, and ABSORBED/REJECTED status
 *
 * @param {Object} data - Smash hit data from WebSocket
 * @param {string} data.node_id - Full node ID (format: "type:paper:chunk")
 * @param {number} data.E - Similarity/energy value
 * @param {boolean} data.gate_open - True if absorbed, false if rejected
 *
 * TWEAK: Colors are set via CSS classes in styles.css
 *   - .smasher-current-e.open = green (absorbed)
 *   - .smasher-current-e.closed = red (rejected)
 */
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

    // Parse node ID to extract readable filename
    // Format: "type:paper:chunk" -> "paper/chunk"
    const nodeId = data.node_id || '';
    const parts = nodeId.split(':');
    const fileName = parts.length > 2 ? `${parts[1]}/${parts[2]}` : nodeId;

    fileEl.innerText = fileName;
    fileEl.title = nodeId;  // Full ID on hover

    // Display E value and gate status
    const E = data.E || 0;
    const gateOpen = data.gate_open;
    eEl.innerText = `E=${E.toFixed(2)} ${gateOpen ? 'ABSORBED' : 'REJECTED'}`;
    eEl.className = `smasher-current-e ${gateOpen ? 'open' : 'closed'}`;
}

/**
 * Clear the current file display (when smasher stops)
 */
export function clearCurrentFile() {
    const container = document.getElementById('smasher-current');
    container.classList.remove('active');
    document.getElementById('smasher-current-file').innerText = '--';
    document.getElementById('smasher-current-e').innerText = '';
}

// =============================================================================
// SECTION 5: VISUALIZATION QUEUE
// =============================================================================
// Batched processing of smash events for smooth animation
//
// WHY QUEUE?
//   The smasher can produce events faster than the graph can render.
//   We batch them into animation frames to prevent:
//   - UI freezing from too many graph updates
//   - Memory exhaustion from unbounded queue growth
//
// TWEAK: CONFIG.SMASHER controls queue behavior
//   - MAX_QUEUE_SIZE: Maximum pending items (default: 100)
//   - QUEUE_DROP_PERCENT: How much to drop on overflow (default: 0.2 = 20%)
//   - MAX_BATCH_PER_FRAME: Items per animation frame (default: 15)

/**
 * Add a smash event to the visualization queue
 * Uses requestAnimationFrame for smooth batched rendering
 *
 * @param {Object} data - Smash hit data from WebSocket
 */
export function queueSmashVisualization(data) {
    // Prevent unbounded queue growth - drop oldest items if queue is full
    // TWEAK: Adjust CONFIG.SMASHER.MAX_QUEUE_SIZE and QUEUE_DROP_PERCENT
    if (state.smashQueue.length >= CONFIG.SMASHER.MAX_QUEUE_SIZE) {
        const dropCount = Math.floor(CONFIG.SMASHER.MAX_QUEUE_SIZE * CONFIG.SMASHER.QUEUE_DROP_PERCENT);
        state.smashQueue.splice(0, dropCount);
        console.warn(`[SMASHER] Queue overflow, dropped ${dropCount} items`);
    }

    state.smashQueue.push(data);

    // Schedule processing if not already scheduled
    if (!state.smashRafPending) {
        state.setSmashRafPending(true);
        requestAnimationFrame(processSmashQueue);
    }
}

/**
 * Process queued smash events in batched animation frame
 * Called via requestAnimationFrame for smooth rendering
 *
 * TWEAK: CONFIG.SMASHER.MAX_BATCH_PER_FRAME controls items per frame
 */
function processSmashQueue() {
    state.setSmashRafPending(false);

    // Process up to MAX_BATCH_PER_FRAME items per frame
    // TWEAK: Adjust in config.js if animation is choppy (lower) or lagging (higher)
    const batch = state.smashQueue.splice(0, CONFIG.SMASHER.MAX_BATCH_PER_FRAME);
    if (batch.length === 0 || !state.Graph) return;

    const graphData = state.Graph.graphData();
    let graphUpdated = false;
    const nodesToFlash = [];

    // Process each item in the batch
    for (const data of batch) {
        const result = processSmashItem(data, graphData);
        if (result.updated) graphUpdated = true;
        if (result.node) {
            nodesToFlash.push({
                nodeId: data.node_id,
                gateOpen: data.gate_open,
                E: data.E
            });
        }
    }

    // Single graph update for entire batch (more efficient)
    if (graphUpdated) {
        state.Graph.graphData(graphData);
    }

    // Flash all processed nodes
    for (const item of nodesToFlash) {
        flashNode(item.nodeId, item.gateOpen, item.E);
    }

    // Continue processing if more items in queue
    if (state.smashQueue.length > 0) {
        state.setSmashRafPending(true);
        requestAnimationFrame(processSmashQueue);
    }
}

// =============================================================================
// SECTION 6: SMASH ITEM PROCESSING
// =============================================================================
// Core logic for processing individual smash events
// Creates new nodes and similarity edges in the graph

/**
 * Process a single smash item
 * - Creates new nodes if is_new_node is true
 * - Adds similarity edges if similar_to is provided and E > threshold
 * - Adds node to exploration trail
 *
 * @param {Object} data - Smash hit data
 * @param {string} data.node_id - Node identifier
 * @param {boolean} data.is_new_node - Whether this is a newly discovered node
 * @param {string} data.similar_to - ID of most similar existing node
 * @param {number} data.similar_E - Similarity score to that node
 * @param {Object} graphData - Current graph data (nodes and links arrays)
 * @returns {Object} { updated: boolean, node: Object|null }
 *
 * TWEAK: Node positioning uses CONFIG.SMASHER settings:
 *   - ANCHOR_OFFSET_BASE: Minimum offset from anchor node
 *   - ANCHOR_OFFSET_SCALE: Additional offset based on similarity
 *   - RANDOM_SPAWN_RANGE: Range when no anchor found
 *   - MIN_SIMILARITY_FOR_EDGE: Minimum E to create similarity edge
 */
function processSmashItem(data, graphData) {
    const nodeId = data.node_id;
    if (!nodeId) {
        console.warn('[SMASHER] processSmashItem called with no node_id');
        return { updated: false, node: null };
    }

    const similarTo = data.similar_to;
    const similarE = data.similar_E || 0;
    let updated = false;

    // Check if node already exists
    let node = state.nodeRegistry.byId.get(nodeId);

    // ----- CREATE NEW NODE -----
    if (!node && data.is_new_node) {
        let pos = { x: 0, y: 0, z: 0 };
        let foundAnchor = false;

        // Position near similar node if available
        // TWEAK: CONFIG.SMASHER.ANCHOR_OFFSET_BASE and ANCHOR_OFFSET_SCALE
        if (similarTo && state.nodeRegistry.byId.has(similarTo)) {
            const anchorNode = state.nodeRegistry.byId.get(similarTo);
            if (anchorNode && typeof anchorNode.x === 'number') {
                // Offset inversely proportional to similarity
                // Higher similarity = closer to anchor
                const offset = CONFIG.SMASHER.ANCHOR_OFFSET_BASE +
                    (1 - similarE) * CONFIG.SMASHER.ANCHOR_OFFSET_SCALE;
                pos = {
                    x: anchorNode.x + (Math.random() - 0.5) * offset,
                    y: anchorNode.y + (Math.random() - 0.5) * offset,
                    z: anchorNode.z + (Math.random() - 0.5) * offset
                };
                foundAnchor = true;
            }
        }

        // Random position if no anchor found
        // TWEAK: CONFIG.SMASHER.RANDOM_SPAWN_RANGE
        if (!foundAnchor) {
            pos = {
                x: (Math.random() - 0.5) * CONFIG.SMASHER.RANDOM_SPAWN_RANGE,
                y: (Math.random() - 0.5) * CONFIG.SMASHER.RANDOM_SPAWN_RANGE,
                z: (Math.random() - 0.5) * CONFIG.SMASHER.RANDOM_SPAWN_RANGE
            };
        }

        // Extract label from node ID
        const parts = nodeId.split(':');
        const label = parts.length > 2 ? parts[2] : 'chunk';

        // Create node object
        node = {
            id: nodeId,
            label: label,
            group: 'page',
            val: 1,
            x: pos.x,
            y: pos.y,
            z: pos.z
        };

        // Register and add to graph
        state.nodeRegistry.byId.set(nodeId, node);
        graphData.nodes.push(node);
        updated = true;

        // Initialize pulse animation state
        state.nodePulseState.set(nodeId, {
            intensity: 1.5,
            phase: 0,
            frequency: 2.0,
            lastActivity: Date.now()
        });
    }

    // ----- CREATE SIMILARITY EDGE -----
    // TWEAK: CONFIG.SMASHER.MIN_SIMILARITY_FOR_EDGE controls minimum E for edge creation
    if (similarTo && similarE > CONFIG.SMASHER.MIN_SIMILARITY_FOR_EDGE) {
        const hasAnchor = state.nodeRegistry.byId.has(similarTo);
        const hasTarget = state.nodeRegistry.byId.has(nodeId);

        if (hasAnchor && hasTarget) {
            // Check if edge already exists (avoid duplicates)
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

    // Add to exploration trail
    addToTrail(nodeId, 'smash');

    return { updated, node };
}

// =============================================================================
// SECTION 7: CONFIGURATION CONTROLS
// =============================================================================
// Speed and batch size sliders that send updates to server

/**
 * Update smasher processing speed
 * Called from speed slider oninput
 *
 * @param {string} value - Delay in milliseconds
 *
 * TWEAK: Slider range is set in index.html (min="10" max="2000")
 */
export async function updateSmasherSpeed(value) {
    state.smasherConfig.delay_ms = parseInt(value);
    document.getElementById('value-smasher-speed').innerText = value + 'ms';
    await sendSmasherConfig({ delay_ms: state.smasherConfig.delay_ms });
}

/**
 * Update smasher batch size
 * Called from batch slider oninput
 *
 * @param {string} value - Number of chunks per batch
 *
 * TWEAK: Slider range is set in index.html (min="1" max="50")
 */
export async function updateSmasherBatch(value) {
    state.smasherConfig.batch_size = parseInt(value);
    document.getElementById('value-smasher-batch').innerText = value;
    await sendSmasherConfig({ batch_size: state.smasherConfig.batch_size });
}

/**
 * Send configuration update to server
 * @param {Object} updates - Configuration changes to send
 */
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

/**
 * Toggle follow mode (camera auto-follow)
 * When toggle is ON, camera auto-follows smasher cursor
 * When toggle is OFF, camera stays static
 *
 * Note: Internally uses staticCameraMode where:
 *   staticCameraMode = true  -> no follow (toggle OFF)
 *   staticCameraMode = false -> follow enabled (toggle ON)
 *
 * TWEAK: Default state is set in state.js (staticCameraMode = true = Follow OFF)
 */
export function toggleStaticCamera() {
    state.setStaticCameraMode(!state.staticCameraMode);
    // Invert display: toggle ON when staticCameraMode is false (follow enabled)
    document.getElementById('toggle-static-camera').className =
        state.staticCameraMode ? 'behavior-toggle' : 'behavior-toggle on';
    saveSettings();
}

// =============================================================================
// SECTION 8: STATUS LOADING
// =============================================================================
// Fetch smasher status from server (used on page load and polling)

/**
 * Load smasher status from server
 * Updates UI with current running state and statistics
 *
 * Called:
 *   - On page load (init)
 *   - Every CONFIG.POLLING.SMASHER_INTERVAL_MS (polling)
 */
export async function loadSmasherStatus() {
    try {
        const res = await api('/smasher/status');
        if (res.ok) {
            const wasActive = _lastSmasherActive;
            const isActive = res.active;
            const currentAbsorbed = res.stats?.chunks_absorbed || 0;

            state.setSmasherActive(isActive);

            // Update stats display
            if (res.stats) {
                document.getElementById('smasher-processed').innerText = res.stats.chunks_processed;
                document.getElementById('smasher-absorbed').innerText = res.stats.chunks_absorbed;
                document.getElementById('smasher-rejected').innerText = res.stats.chunks_rejected;
                document.getElementById('smasher-rate').innerText = res.stats.chunks_per_second.toFixed(1);
            }

            updateSmasherUI();

            // CACHE INVALIDATION: When smasher stops and new data was absorbed
            if (wasActive && !isActive && currentAbsorbed > _lastAbsorbedCount) {
                console.log('[SMASHER] Session ended with new absorptions - invalidating cache');
                await invalidateConstellationCache();
                // Reload constellation to show new data
                await reloadConstellation(true);  // Force refresh
            }

            // Update tracking state
            _lastSmasherActive = isActive;
            if (!isActive) {
                // Reset absorbed count when not running
                _lastAbsorbedCount = 0;
            } else {
                _lastAbsorbedCount = currentAbsorbed;
            }
        }
    } catch (e) {
        // Ignore polling errors silently
    }
}

// =============================================================================
// SECTION 9: THRESHOLD DISPLAY (Q46 NUCLEATION)
// =============================================================================
// Dynamic threshold display based on memory count (Q46 Law 3)
//
// The absorption threshold is not fixed - it follows the nucleation formula:
//   theta = 1 / (2 * pi) * (1 - exp(-N / N_0))
// where N is the number of memories and N_0 is a scale parameter.
//
// This creates emergent behavior where:
//   - Small memories have low threshold (easy to absorb)
//   - Large memories have high threshold (selective absorption)

/**
 * Update the threshold display in the UI
 * Shows current dynamic threshold and memory count
 *
 * @param {number} threshold - Current absorption threshold (0 to ~0.159)
 * @param {number} n - Current number of memories
 *
 * TWEAK: The progress bar max is 1/(2*pi) which is the theoretical maximum
 */
export function updateThresholdDisplay(threshold, n) {
    state.setCurrentThreshold(threshold);
    state.setNMemories(n);

    // Update display values
    document.getElementById('value-e-threshold').innerText = threshold.toFixed(3);
    document.getElementById('value-n-memories').innerText = n;

    // Update progress bar (max threshold is 1/(2*pi) ~ 0.159)
    const maxThreshold = 1.0 / (2.0 * Math.PI);
    const progress = Math.min(100, (threshold / maxThreshold) * 100);
    document.getElementById('threshold-progress').style.width = progress + '%';
}

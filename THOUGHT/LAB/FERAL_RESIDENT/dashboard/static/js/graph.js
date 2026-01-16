// =============================================================================
// FERAL DASHBOARD - 3D CONSTELLATION GRAPH
// =============================================================================
//
// This module handles the 3D force-directed graph visualization using Three.js
// and 3d-force-graph library. It renders memory nodes, connections, and trails.
//
// ARCHITECTURE:
//   - Uses ForceGraph3D (https://github.com/vasturiano/3d-force-graph)
//   - Custom Three.js objects for nodes (spheres with glow)
//   - Physics simulation via d3-force-3d
//
// KEY CONCEPTS:
//   - Nodes: Memory chunks and folder hierarchies
//   - Links: Hierarchy, similarity, mind-projected, co-retrieval, entanglement
//   - Trails: Path visualization showing exploration history
//   - Smasher Cursor: 3D indicator showing current analysis position
//
// CUSTOMIZATION:
//   Edit config.js to adjust:
//   - Node geometry (sphere segments, sizes)
//   - Link widths and colors
//   - Physics simulation parameters
//   - Performance thresholds
//
// SECTIONS:
//   1. Imports & Constants
//   2. Shared Geometries/Materials (performance optimization)
//   3. Graph Initialization
//   4. Node Rendering
//   5. Trail Visualization
//   6. Smasher Cursor
//   7. Node Interactions
//   8. Graph Controls
//
// =============================================================================

import * as state from './state.js';
import { CONFIG } from './config.js';

// =============================================================================
// 1. ACTIVITY COLORS
// =============================================================================
// Colors for different activity types in the exploration trail
// TWEAK: Edit these hex values to change trail/node colors by activity type

export const ACTIVITY_COLORS = {
    paper:       { main: 0x00ff41, glow: '#00ff41' },  // Green
    consolidate: { main: 0x00ff41, glow: '#00ff41' },
    reflect:     { main: 0x00ff41, glow: '#00ff41' },
    cassette:    { main: 0x00ff41, glow: '#00ff41' },
    daemon:      { main: 0x00ff41, glow: '#00ff41' },
    smash:       { main: 0x00ff41, glow: '#00ff41' },
    default:     { main: 0x00ff41, glow: '#00ff41' }
};

// =============================================================================
// 2. MODULE STATE
// =============================================================================

let trailLine = null;
let sharedGeometries = null;
let sharedMaterials = null;

// Smasher cursor components
let smasherCursorGroup = null;
let smasherCursorRing = null;
let smasherTrailLine = null;
let smasherAnimationFrame = null;

// Flash timer tracking (prevents memory leaks)
const activeFlashTimers = new Map();

// Settings save debounce
let saveSettingsTimer = null;

// =============================================================================
// 3. SHARED GEOMETRIES & MATERIALS
// =============================================================================
// Performance optimization: create geometries once, reuse for all nodes
//
// TWEAK: Adjust segment counts in config.js GRAPH section
//   - Higher segments = smoother spheres, more GPU load
//   - Lower segments = blockier look, better performance

function initSharedAssets() {
    if (sharedGeometries) return;

    const cfg = CONFIG.GRAPH;

    // ----- GEOMETRIES -----
    // Created ONCE, shared by all nodes
    sharedGeometries = {
        // Folder nodes (larger, more prominent)
        folderSphere: new THREE.SphereGeometry(
            cfg.FOLDER_SPHERE_RADIUS,
            cfg.FOLDER_SPHERE_SEGMENTS,
            cfg.FOLDER_SPHERE_SEGMENTS
        ),
        folderGlow: new THREE.SphereGeometry(
            cfg.FOLDER_GLOW_RADIUS,
            cfg.FOLDER_GLOW_SEGMENTS,
            cfg.FOLDER_GLOW_SEGMENTS
        ),
        // Chunk nodes (smaller, numerous)
        nodeSphere: new THREE.SphereGeometry(
            cfg.NODE_SPHERE_RADIUS,
            cfg.NODE_SPHERE_SEGMENTS,
            cfg.NODE_SPHERE_SEGMENTS
        ),
        nodeGlow: new THREE.SphereGeometry(
            cfg.NODE_GLOW_RADIUS,
            cfg.NODE_GLOW_SEGMENTS,
            cfg.NODE_GLOW_SEGMENTS
        ),
    };

    // ----- MATERIALS -----
    // Created ONCE, cloned when nodes need independent color (for flash)
    sharedMaterials = {
        folder: new THREE.MeshBasicMaterial({
            color: CONFIG.COLORS.FOLDER_COLOR,
            transparent: true,
            opacity: CONFIG.COLORS.NODE_OPACITY
        }),
        node: new THREE.MeshBasicMaterial({
            color: CONFIG.COLORS.NODE_COLOR,
            transparent: true,
            opacity: CONFIG.COLORS.NODE_OPACITY
        }),
        glow: new THREE.MeshBasicMaterial({
            color: CONFIG.COLORS.GLOW_COLOR,
            transparent: true,
            opacity: CONFIG.COLORS.GLOW_OPACITY
        })
    };

    console.log('[GRAPH] Shared geometries/materials initialized');
}

// =============================================================================
// 4. THREE.JS LOADING
// =============================================================================

/**
 * Wait for Three.js and ForceGraph3D to load from CDN
 * These are loaded async in index.html
 */
export function waitForThreeJS() {
    return new Promise((resolve) => {
        function check() {
            if (typeof THREE !== 'undefined' && typeof ForceGraph3D !== 'undefined') {
                resolve();
            } else {
                setTimeout(check, 100);
            }
        }
        check();
    });
}

// =============================================================================
// 5. GRAPH INITIALIZATION
// =============================================================================

/**
 * Initialize the 3D constellation graph
 *
 * This is the main entry point. It:
 *   1. Fetches node/edge data from server
 *   2. Creates ForceGraph3D instance
 *   3. Configures physics simulation
 *   4. Sets up custom node rendering
 *   5. Adds lighting and fog
 *
 * TWEAK: See config.js GRAPH section for all parameters
 */
export async function initConstellation() {
    await waitForThreeJS();

    const container = document.getElementById('constellation-background');
    const tooltip = document.getElementById('node-tooltip');

    if (!container) {
        console.error('[GRAPH] Missing container: constellation-background');
        return;
    }

    try {
        // -----------------------------------------------------------------
        // FETCH DATA
        // -----------------------------------------------------------------
        console.log('[GRAPH] Fetching constellation data...');
        const url = '/api/constellation?include_similarity=true&similarity_threshold=' + state.similarityThreshold;
        const res = await fetch(url);
        const data = await res.json();
        console.log('[GRAPH] Received:', data.nodes?.length, 'nodes,', data.edges?.length, 'edges');

        if (!data.ok || !data.nodes || data.nodes.length === 0) {
            console.log('[GRAPH] No data:', data.error || 'empty');
            return;
        }

        // -----------------------------------------------------------------
        // PREPARE NODES
        // -----------------------------------------------------------------
        const nodes = data.nodes.map(n => ({
            id: n.id,
            label: n.label,
            group: n.group,       // 'folder' or 'page'
            path: n.path || '',
            val: n.group === 'folder' ? 3 : 1  // Folder nodes are larger
        }));

        // -----------------------------------------------------------------
        // PREPARE LINKS
        // -----------------------------------------------------------------
        state.allLinks.length = 0;
        data.edges.forEach(e => {
            state.allLinks.push({
                source: e.from,
                target: e.to,
                type: e.type || 'hierarchy',
                weight: e.weight || 0.5
            });
        });

        const links = filterLinks(state.allLinks);
        console.log('[GRAPH] After filtering:', nodes.length, 'nodes,', links.length, 'links');

        // -----------------------------------------------------------------
        // PERFORMANCE MODE
        // -----------------------------------------------------------------
        // Enable optimizations for large datasets
        const cfg = CONFIG.GRAPH;
        const perfMode = nodes.length > cfg.PERF_MODE_THRESHOLD;
        if (perfMode) {
            console.log('[GRAPH] Performance mode ON (>', cfg.PERF_MODE_THRESHOLD, 'nodes)');
        }

        // Initialize shared assets
        initSharedAssets();

        // Register nodes for fast lookup
        nodes.forEach(n => state.nodeRegistry.byId.set(n.id, n));

        // Initialize pulse state (skip in perf mode)
        if (!perfMode) {
            nodes.forEach(n => {
                state.nodePulseState.set(n.id, {
                    intensity: 0.4,
                    phase: Math.random() * Math.PI * 2,
                    frequency: 0.5 + Math.random() * 0.5,
                    lastActivity: Date.now()
                });
            });
        }

        // -----------------------------------------------------------------
        // CREATE GRAPH
        // -----------------------------------------------------------------
        console.log('[GRAPH] Creating ForceGraph3D...');

        const Graph = ForceGraph3D({
            controlType: 'orbit',
            rendererConfig: {
                antialias: cfg.ANTIALIAS,
                powerPreference: 'high-performance',
                precision: perfMode ? cfg.PRECISION_PERF : cfg.PRECISION_NORMAL
            }
        })(container)
            .graphData({ nodes, links })
            .backgroundColor('#000000')
            .showNavInfo(false)
            .nodeLabel(node => `${node.label}\n${node.path || node.id}`)
            .nodeColor(node => node.group === 'folder' ? '#00ff41' : '#008f11')
            .nodeOpacity(0.9)
            .nodeResolution(perfMode ? cfg.NODE_RESOLUTION_PERF : cfg.NODE_RESOLUTION_NORMAL)
            .nodeVal(node => node.val)

            // ---------------------------------------------------------
            // LINK COLORS
            // ---------------------------------------------------------
            // TWEAK: Modify alpha calculations for different opacity curves
            .linkColor(link => {
                if (link.type === 'mind_projected') {
                    const alpha = 0.4 + (link.weight || 0.5) * 0.5;
                    return `rgba(255, 107, 107, ${alpha})`;  // Coral red
                }
                if (link.type === 'co_retrieval') {
                    const alpha = 0.4 + (link.weight || 0.5) * 0.5;
                    return `rgba(255, 217, 61, ${alpha})`;   // Gold
                }
                if (link.type === 'entanglement') {
                    const alpha = 0.5 + (link.weight || 0.5) * 0.4;
                    return `rgba(199, 125, 255, ${alpha})`;  // Purple
                }
                if (link.type === 'smash_trail') {
                    return 'rgba(255, 102, 0, 0.9)';         // Orange
                }
                if (link.type === 'similarity') {
                    const alpha = 0.15 + (link.weight || 0.5) * 0.25;
                    return `rgba(100, 255, 255, ${alpha})`;  // Cyan
                }
                return 'rgba(0, 143, 17, 0.2)';              // Dark green (hierarchy)
            })

            // ---------------------------------------------------------
            // LINK WIDTHS
            // ---------------------------------------------------------
            // TWEAK: Adjust in config.js GRAPH.LINK_WIDTH_*
            .linkWidth(link => {
                if (link.type === 'smash_trail') return cfg.LINK_WIDTH_SMASH_TRAIL;
                if (link.type === 'similarity') return cfg.LINK_WIDTH_SIMILARITY;
                if (link.type === 'mind_projected') return cfg.LINK_WIDTH_MIND_PROJECTED;
                if (link.type === 'co_retrieval') return cfg.LINK_WIDTH_CO_RETRIEVAL;
                if (link.type === 'entanglement') return cfg.LINK_WIDTH_ENTANGLEMENT;
                return cfg.LINK_WIDTH_HIERARCHY;
            })
            .linkOpacity(link => {
                if (link.type === 'smash_trail') return 1.0;
                if (link.type === 'similarity') return 0.3;
                return 0.2;
            })

            // ---------------------------------------------------------
            // PHYSICS
            // ---------------------------------------------------------
            // TWEAK: Adjust in config.js GRAPH section
            .d3AlphaDecay(perfMode ? cfg.ALPHA_DECAY_PERF : cfg.ALPHA_DECAY_NORMAL)
            .d3VelocityDecay(perfMode ? cfg.VELOCITY_DECAY_PERF : cfg.VELOCITY_DECAY_NORMAL)
            .warmupTicks(perfMode ? cfg.WARMUP_TICKS_PERF : cfg.WARMUP_TICKS_NORMAL)
            .cooldownTicks(perfMode ? cfg.COOLDOWN_TICKS_PERF : cfg.COOLDOWN_TICKS_NORMAL)
            .enableNodeDrag(false)

            // ---------------------------------------------------------
            // INTERACTIONS
            // ---------------------------------------------------------
            .onNodeHover(node => {
                if (!tooltip) return;
                if (node) {
                    tooltip.innerHTML = `<strong>${node.label}</strong><br><span style="color: var(--text-muted)">${node.path || node.id}</span>`;
                    tooltip.style.display = 'block';
                } else {
                    tooltip.style.display = 'none';
                }
            })
            .onNodeClick(node => {
                if (node) focusCameraOnNode(node, cfg.FOCUS_DURATION_MS);
            });

        state.setGraph(Graph);

        // -----------------------------------------------------------------
        // CONFIGURE FORCES
        // -----------------------------------------------------------------
        const { linkDistance, linkStrength, repel, center } = state.graphSettings;
        console.log('[GRAPH] Forces:', { linkDistance, linkStrength, repel, center });

        window.graphForces = {
            linkDistance, linkStrength,
            chargeStrength: -repel,
            centerStrength: center
        };

        Graph.d3Force('link').distance(linkDistance).strength(linkStrength);
        Graph.d3Force('charge').strength(-repel).distanceMax(cfg.FORCE_CHARGE_MAX_DISTANCE);
        Graph.d3Force('center').strength(center);

        // -----------------------------------------------------------------
        // CUSTOM NODE RENDERING
        // -----------------------------------------------------------------
        Graph.nodeThreeObject(node => {
            const group = new THREE.Group();
            const isFolder = node.group === 'folder';

            // Use shared geometry, clone material for independent color
            const geometry = isFolder ? sharedGeometries.folderSphere : sharedGeometries.nodeSphere;
            const material = (isFolder ? sharedMaterials.folder : sharedMaterials.node).clone();
            const sphere = new THREE.Mesh(geometry, material);
            group.add(sphere);
            node.__mainSphere = sphere;

            // Add glow (skip for chunks in perf mode)
            if (!perfMode || isFolder) {
                const glowGeom = isFolder ? sharedGeometries.folderGlow : sharedGeometries.nodeGlow;
                const glowMat = sharedMaterials.glow.clone();
                const glow = new THREE.Mesh(glowGeom, glowMat);
                group.add(glow);
                node.__glowSphere = glow;
            }

            return group;
        });

        // -----------------------------------------------------------------
        // LIGHTING & FOG
        // -----------------------------------------------------------------
        const scene = Graph.scene();

        // Ambient light
        scene.add(new THREE.AmbientLight(0x00ff41, 0.3));

        // Point light
        const pointLight = new THREE.PointLight(0x00ff41, 1, 500);
        pointLight.position.set(0, 100, 100);
        scene.add(pointLight);

        // Fog - TWEAK: Adjust FOG_DENSITY in config.js
        const fogDensity = state.graphSettings.fog;
        scene.fog = new THREE.FogExp2(0x000000, fogDensity);

        // Camera position
        Graph.cameraPosition(
            { x: 0, y: 0, z: cfg.INITIAL_CAMERA_Z },
            { x: 0, y: 0, z: 0 },
            0
        );

        // -----------------------------------------------------------------
        // EVENT HANDLERS
        // -----------------------------------------------------------------
        document.addEventListener('mousemove', (e) => {
            if (tooltip && tooltip.style.display === 'block') {
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            }
        });

        // Resize graph to container
        const resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    Graph.width(width).height(height);
                }
            }
        });
        resizeObserver.observe(container);

        console.log('[GRAPH] Initialized with', nodes.length, 'nodes');

    } catch (e) {
        console.error('[GRAPH] Init error:', e);
    }
}

// =============================================================================
// 6. CAMERA CONTROLS
// =============================================================================

/**
 * Smoothly focus camera on a node
 */
export function focusCameraOnNode(node, duration = 1500) {
    if (!state.Graph || !node) return;

    const distance = 150;
    const distRatio = 1 + distance / Math.max(10, Math.hypot(node.x || 0, node.y || 0, node.z || 0));

    state.Graph.cameraPosition(
        { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
        { x: node.x || 0, y: node.y || 0, z: node.z || 0 },
        duration
    );
}

// =============================================================================
// 7. EXPLORATION TRAIL
// =============================================================================
// Visualizes the daemon's path through the graph

/**
 * Add node to exploration trail
 */
export function addToTrail(nodeId, activityType) {
    state.explorationTrail.push({
        nodeId,
        timestamp: Date.now(),
        type: activityType
    });

    while (state.explorationTrail.length > CONFIG.TRAILS.MAX_LENGTH) {
        state.explorationTrail.shift();
    }

    updateTrailVisualization();
}

function updateTrailVisualization() {
    if (!state.Graph) return;
    const scene = state.Graph.scene();

    // Remove old trail
    if (trailLine) {
        scene.remove(trailLine);
        trailLine.geometry.dispose();
        trailLine.material.dispose();
        trailLine = null;
    }

    if (state.explorationTrail.length < 2) return;

    const points = [];
    const colors = [];

    state.explorationTrail.forEach((entry) => {
        const node = state.nodeRegistry.byId.get(entry.nodeId);
        if (!node || node.x === undefined) return;

        points.push(new THREE.Vector3(node.x, node.y, node.z));

        // Fade based on age
        const age = (Date.now() - entry.timestamp) / CONFIG.TRAILS.FADE_TIME_MS;
        const alpha = Math.max(CONFIG.TRAILS.MIN_OPACITY, 1 - age);

        const colorInfo = ACTIVITY_COLORS[entry.type] || ACTIVITY_COLORS.default;
        const color = new THREE.Color(colorInfo.main);
        colors.push(color.r * alpha, color.g * alpha, color.b * alpha);
    });

    if (points.length < 2) return;

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: CONFIG.TRAILS.LINE_OPACITY
    });

    trailLine = new THREE.Line(geometry, material);
    scene.add(trailLine);
}

// =============================================================================
// 8. NODE SPAWNING & ACTIVATION
// =============================================================================

/**
 * Spawn a new node in the graph
 */
export function spawnNode(nodeData) {
    if (!state.Graph) return;
    const graphData = state.Graph.graphData();

    if (state.nodeRegistry.byId.has(nodeData.node_id)) return;

    // Position near parent
    let parentPos = { x: 0, y: 0, z: 0 };
    if (nodeData.source_id) {
        const parent = state.nodeRegistry.byId.get(nodeData.source_id);
        if (parent && parent.x !== undefined) {
            parentPos = { x: parent.x, y: parent.y, z: parent.z };
        }
    }

    const newNode = {
        id: nodeData.node_id,
        label: nodeData.label || 'new chunk',
        group: 'page',
        path: nodeData.paper || '',
        val: 1,
        x: parentPos.x + (Math.random() - 0.5) * 50,
        y: parentPos.y + (Math.random() - 0.5) * 50,
        z: parentPos.z + (Math.random() - 0.5) * 50
    };

    state.nodeRegistry.byId.set(newNode.id, newNode);
    graphData.nodes.push(newNode);

    if (nodeData.source_id && state.nodeRegistry.byId.has(nodeData.source_id)) {
        graphData.links.push({ source: nodeData.source_id, target: nodeData.node_id });
    }

    state.Graph.graphData(graphData);
    state.nodePulseState.set(newNode.id, {
        intensity: 1.0, phase: 0, frequency: 1.0, lastActivity: Date.now()
    });

    if (!state.staticCameraMode) {
        setTimeout(() => {
            const node = state.nodeRegistry.byId.get(nodeData.node_id);
            if (node) focusCameraOnNode(node, 1500);
        }, 300);
    }
}

/**
 * Activate a node (visual highlight)
 */
export function activateNode(nodeId, activityType) {
    const node = state.nodeRegistry.byId.get(nodeId);
    if (!node) return;

    const pulseState = state.nodePulseState.get(nodeId);
    if (pulseState) {
        pulseState.intensity = 1.0;
        pulseState.lastActivity = Date.now();
    }

    const colorInfo = ACTIVITY_COLORS[activityType] || ACTIVITY_COLORS.default;

    if (node.__mainSphere) {
        node.__mainSphere.material.color.setHex(colorInfo.main);
        if (node.__mainSphere.material.emissive) {
            node.__mainSphere.material.emissive.setHex(colorInfo.main);
            node.__mainSphere.material.emissiveIntensity = 0.8;
        }

        setTimeout(() => {
            if (node.__mainSphere) {
                node.__mainSphere.material.color.setHex(
                    node.group === 'folder' ? CONFIG.COLORS.FOLDER_COLOR : CONFIG.COLORS.NODE_COLOR
                );
            }
        }, 3000);
    }

    if (node.__glowSphere) {
        node.__glowSphere.material.color.setHex(colorInfo.main);
        node.__glowSphere.material.opacity = 0.4;
    }

    if (!state.staticCameraMode) {
        focusCameraOnNode(node, 1000);
    }
}

// =============================================================================
// 9. SMASHER CURSOR
// =============================================================================
// 3D indicator showing current analysis position
//
// TWEAK: Adjust sizes in config.js SMASHER_CURSOR section

function createSmasherCursor() {
    if (!state.Graph) return;
    const scene = state.Graph.scene();
    const cfg = CONFIG.SMASHER_CURSOR;

    if (smasherCursorGroup) {
        scene.remove(smasherCursorGroup);
        smasherCursorGroup = null;
    }

    smasherCursorGroup = new THREE.Group();

    // Outer ring
    const ringGeom = new THREE.TorusGeometry(
        cfg.RING_RADIUS, cfg.RING_TUBE_RADIUS,
        cfg.RING_TUBE_SEGMENTS, cfg.RING_RADIAL_SEGMENTS
    );
    const ringMat = new THREE.MeshBasicMaterial({
        color: cfg.RING_COLOR, transparent: true, opacity: cfg.RING_OPACITY
    });
    smasherCursorRing = new THREE.Mesh(ringGeom, ringMat);
    smasherCursorGroup.add(smasherCursorRing);

    // Glow sphere
    const glowGeom = new THREE.SphereGeometry(cfg.GLOW_RADIUS, cfg.GLOW_SEGMENTS, cfg.GLOW_SEGMENTS);
    const glowMat = new THREE.MeshBasicMaterial({
        color: cfg.GLOW_COLOR, transparent: true, opacity: cfg.GLOW_OPACITY
    });
    smasherCursorGroup.add(new THREE.Mesh(glowGeom, glowMat));

    // Core
    const coreGeom = new THREE.SphereGeometry(cfg.CORE_RADIUS, cfg.CORE_SEGMENTS, cfg.CORE_SEGMENTS);
    const coreMat = new THREE.MeshBasicMaterial({
        color: cfg.CORE_COLOR, transparent: true, opacity: cfg.CORE_OPACITY
    });
    smasherCursorGroup.add(new THREE.Mesh(coreGeom, coreMat));

    smasherCursorGroup.visible = false;
    scene.add(smasherCursorGroup);

    if (!smasherAnimationFrame) {
        animateSmasherCursor();
    }

    return smasherCursorGroup;
}

function animateSmasherCursor() {
    smasherAnimationFrame = requestAnimationFrame(animateSmasherCursor);
    if (!smasherCursorGroup || !smasherCursorGroup.visible) return;

    const cfg = CONFIG.SMASHER_CURSOR;
    const time = Date.now() * 0.002;

    if (smasherCursorRing) {
        smasherCursorRing.rotation.x = time * cfg.ROTATION_SPEED_X;
        smasherCursorRing.rotation.y = time * cfg.ROTATION_SPEED_Y;
    }

    const pulse = 1 + Math.sin(time * cfg.PULSE_SPEED) * cfg.PULSE_AMPLITUDE;
    smasherCursorGroup.scale.setScalar(pulse);
}

export function moveSmasherCursor(nodeId, gateOpen) {
    if (!state.Graph) return;
    if (!smasherCursorGroup) createSmasherCursor();

    const node = state.nodeRegistry.byId.get(nodeId);
    if (!node || node.x === undefined) {
        smasherCursorGroup.visible = false;
        return;
    }

    smasherCursorGroup.position.set(node.x, node.y, node.z);
    smasherCursorGroup.visible = true;

    const cfg = CONFIG.SMASHER_CURSOR;
    const color = gateOpen ? cfg.COLOR_ABSORBED : cfg.COLOR_REJECTED;
    smasherCursorGroup.children.forEach(child => {
        if (child.material) child.material.color.setHex(color);
    });

    state.addToSmasherTrail(nodeId);
    updateSmasherTrail();
}

function updateSmasherTrail() {
    if (!state.Graph) return;
    const scene = state.Graph.scene();

    if (smasherTrailLine) {
        scene.remove(smasherTrailLine);
        smasherTrailLine.geometry.dispose();
        smasherTrailLine.material.dispose();
        smasherTrailLine = null;
    }

    if (state.smasherTrailNodes.length < 2) return;

    const points = [];
    const colors = [];

    state.smasherTrailNodes.forEach((entry, index) => {
        const node = state.nodeRegistry.byId.get(entry.nodeId);
        if (!node || node.x === undefined) return;

        points.push(new THREE.Vector3(node.x, node.y, node.z));

        // Orange to yellow gradient
        const age = index / state.smasherTrailNodes.length;
        colors.push(1.0, 0.4 + age * 0.6, 0.0);
    });

    if (points.length < 2) return;

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: CONFIG.TRAILS.SMASHER_LINE_OPACITY,
        linewidth: CONFIG.TRAILS.SMASHER_LINE_WIDTH
    });

    smasherTrailLine = new THREE.Line(geometry, material);
    scene.add(smasherTrailLine);
}

export function hideSmasherCursor() {
    if (smasherCursorGroup) smasherCursorGroup.visible = false;
    if (smasherTrailLine) smasherTrailLine.visible = false;
    state.smasherTrailNodes.length = 0;

    if (smasherAnimationFrame) {
        cancelAnimationFrame(smasherAnimationFrame);
        smasherAnimationFrame = null;
    }
}

// =============================================================================
// 10. NODE FLASH
// =============================================================================
// Visual effect when smasher hits a node

export function flashNode(nodeId, gateOpen, E) {
    const node = state.nodeRegistry.byId.get(nodeId);
    if (!node) return;

    moveSmasherCursor(nodeId, gateOpen);

    // Cancel existing timers
    const existing = activeFlashTimers.get(nodeId);
    if (existing) existing.forEach(id => clearTimeout(id));
    const newTimers = [];

    const cfg = CONFIG.NODE_FLASH;
    const flashColor = gateOpen ? CONFIG.SMASHER_CURSOR.COLOR_ABSORBED : CONFIG.SMASHER_CURSOR.COLOR_REJECTED;

    if (node.__mainSphere) {
        node.__mainSphere.scale.setScalar(cfg.SCALE);
        node.__mainSphere.material.color.setHex(flashColor);
        node.__mainSphere.material.opacity = cfg.OPACITY;

        const timer = setTimeout(() => {
            if (node.__mainSphere) {
                node.__mainSphere.scale.setScalar(1.0);
                node.__mainSphere.material.color.setHex(
                    node.group === 'folder' ? CONFIG.COLORS.FOLDER_COLOR : CONFIG.COLORS.NODE_COLOR
                );
                node.__mainSphere.material.opacity = CONFIG.COLORS.NODE_OPACITY;
            }
            cleanupTimer(nodeId, timer);
        }, cfg.DURATION_MS);
        newTimers.push(timer);
    }

    if (node.__glowSphere) {
        node.__glowSphere.scale.setScalar(cfg.GLOW_SCALE);
        node.__glowSphere.material.color.setHex(flashColor);
        node.__glowSphere.material.opacity = cfg.GLOW_OPACITY;

        const timer = setTimeout(() => {
            if (node.__glowSphere) {
                node.__glowSphere.scale.setScalar(1.3);
                node.__glowSphere.material.color.setHex(CONFIG.COLORS.GLOW_COLOR);
                node.__glowSphere.material.opacity = CONFIG.COLORS.GLOW_OPACITY;
            }
            cleanupTimer(nodeId, timer);
        }, cfg.DURATION_MS);
        newTimers.push(timer);
    }

    if (newTimers.length > 0) activeFlashTimers.set(nodeId, newTimers);

    if (!state.staticCameraMode) {
        focusCameraOnNode(node, 500);
    }
}

function cleanupTimer(nodeId, timer) {
    const timers = activeFlashTimers.get(nodeId);
    if (timers) {
        const idx = timers.indexOf(timer);
        if (idx > -1) timers.splice(idx, 1);
        if (timers.length === 0) activeFlashTimers.delete(nodeId);
    }
}

// =============================================================================
// 11. LINK FILTERING
// =============================================================================

export function filterLinks(links) {
    return links.filter(l => {
        if (l.type !== 'similarity') return true;
        if (!state.showSimilarityLinks) return false;
        return (l.weight || 0) >= state.similarityThreshold;
    });
}

export function updateVisibleLinks() {
    if (!state.Graph) return;
    const graphData = state.Graph.graphData();
    const visibleLinks = filterLinks(state.allLinks);
    state.Graph.graphData({ nodes: graphData.nodes, links: visibleLinks });
}

export async function reloadConstellation() {
    if (!state.Graph) return;

    try {
        const url = `/api/constellation?include_similarity=true&similarity_threshold=${state.similarityThreshold}`;
        const res = await fetch(url);
        const data = await res.json();
        if (!data.ok) return;

        const nodes = data.nodes.map(n => ({
            id: n.id, label: n.label, group: n.group,
            path: n.path || '', val: n.group === 'folder' ? 3 : 1
        }));

        state.allLinks.length = 0;
        data.edges.forEach(e => {
            state.allLinks.push({
                source: e.from, target: e.to,
                type: e.type || 'hierarchy', weight: e.weight || 0.5
            });
        });

        nodes.forEach(n => state.nodeRegistry.byId.set(n.id, n));
        state.Graph.graphData({ nodes, links: filterLinks(state.allLinks) });
    } catch (e) {
        console.error('[GRAPH] Reload error:', e);
    }
}

// =============================================================================
// 12. GRAPH CONTROLS (UI SLIDER HANDLERS)
// =============================================================================

function debouncedSaveSettings() {
    if (saveSettingsTimer) clearTimeout(saveSettingsTimer);
    saveSettingsTimer = setTimeout(() => {
        window.saveSettings?.();
        saveSettingsTimer = null;
    }, 300);
}

/**
 * Update fog density
 * Called by: oninput="updateFog(this.value)"
 */
export function updateFog(value) {
    if (!state.Graph) return;
    const density = parseFloat(value);
    state.Graph.scene().fog.density = density;
    state.updateGraphSetting('fog', density);

    const el = document.getElementById('value-fog');
    if (el) el.innerText = density.toFixed(4);

    debouncedSaveSettings();
}

/**
 * Update graph force parameter
 * Called by: oninput="updateGraphForce('center', this.value)"
 */
export function updateGraphForce(force, value) {
    if (!state.Graph || !window.graphForces) return;
    const numValue = parseFloat(value);

    const linkForce = state.Graph.d3Force('link');
    const chargeForce = state.Graph.d3Force('charge');
    const centerForce = state.Graph.d3Force('center');

    switch (force) {
        case 'center':
            window.graphForces.centerStrength = numValue;
            if (centerForce) centerForce.strength(numValue);
            state.updateGraphSetting('center', numValue);
            updateDisplay('value-center', numValue.toFixed(2));
            break;

        case 'repel':
            window.graphForces.chargeStrength = -numValue;
            if (chargeForce) chargeForce.strength(-numValue);
            state.updateGraphSetting('repel', numValue);
            updateDisplay('value-repel', -numValue);
            break;

        case 'linkStrength':
            window.graphForces.linkStrength = numValue;
            if (linkForce) linkForce.strength(numValue);
            state.updateGraphSetting('linkStrength', numValue);
            updateDisplay('value-link-strength', numValue.toFixed(2));
            break;

        case 'linkDistance':
            window.graphForces.linkDistance = numValue;
            if (linkForce) linkForce.distance(numValue);
            state.updateGraphSetting('linkDistance', numValue);
            updateDisplay('value-link-distance', numValue);
            break;
    }

    state.Graph.d3ReheatSimulation();
    debouncedSaveSettings();
}

function updateDisplay(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value;
}

/**
 * Reset all graph forces to defaults
 * Called by: onclick="resetGraphForces()"
 */
export function resetGraphForces() {
    if (!state.Graph) return;

    const cfg = CONFIG.GRAPH;

    window.graphForces = {
        linkDistance: cfg.FORCE_LINK_DISTANCE,
        linkStrength: cfg.FORCE_LINK_STRENGTH,
        chargeStrength: cfg.FORCE_CHARGE_STRENGTH,
        centerStrength: cfg.FORCE_CENTER_STRENGTH
    };

    state.Graph.d3Force('link').distance(cfg.FORCE_LINK_DISTANCE).strength(cfg.FORCE_LINK_STRENGTH);
    state.Graph.d3Force('charge').strength(cfg.FORCE_CHARGE_STRENGTH);
    state.Graph.d3Force('center').strength(cfg.FORCE_CENTER_STRENGTH);
    state.Graph.scene().fog.density = 0.003;
    state.Graph.d3ReheatSimulation();

    // Reset similarity
    state.setShowSimilarityLinks(true);
    state.setSimilarityThreshold(0.35);
    document.getElementById('toggle-similarity').className = 'behavior-toggle on';
    document.getElementById('slider-sim-threshold').value = 0.35;
    document.getElementById('value-sim-threshold').innerText = '0.35';

    // Reset sliders
    document.getElementById('slider-fog').value = 0.003;
    document.getElementById('slider-center').value = cfg.FORCE_CENTER_STRENGTH;
    document.getElementById('slider-repel').value = Math.abs(cfg.FORCE_CHARGE_STRENGTH);
    document.getElementById('slider-link-strength').value = cfg.FORCE_LINK_STRENGTH;
    document.getElementById('slider-link-distance').value = cfg.FORCE_LINK_DISTANCE;

    // Reset displays
    document.getElementById('value-fog').innerText = '0.0030';
    document.getElementById('value-center').innerText = cfg.FORCE_CENTER_STRENGTH.toFixed(2);
    document.getElementById('value-repel').innerText = cfg.FORCE_CHARGE_STRENGTH;
    document.getElementById('value-link-strength').innerText = cfg.FORCE_LINK_STRENGTH.toFixed(2);
    document.getElementById('value-link-distance').innerText = cfg.FORCE_LINK_DISTANCE;

    reloadConstellation();
}

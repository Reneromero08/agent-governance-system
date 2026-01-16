// ===== 3D CONSTELLATION =====
import * as state from './state.js';

export const ACTIVITY_COLORS = {
    paper: { main: 0x00ff41, glow: '#00ff41' },
    consolidate: { main: 0x00ff41, glow: '#00ff41' },
    reflect: { main: 0x00ff41, glow: '#00ff41' },
    cassette: { main: 0x00ff41, glow: '#00ff41' },
    daemon: { main: 0x00ff41, glow: '#00ff41' },
    smash: { main: 0x00ff41, glow: '#00ff41' },
    default: { main: 0x00ff41, glow: '#00ff41' }
};

let trailLine = null;
const MAX_TRAIL_LENGTH = 50;

// ===== SHARED GEOMETRIES & MATERIALS (Performance Optimization) =====
// Create once, reuse for all nodes - avoids creating thousands of objects
let sharedGeometries = null;
let sharedMaterials = null;

function initSharedAssets() {
    if (sharedGeometries) return;  // Already initialized

    // Shared geometries - created ONCE
    sharedGeometries = {
        folderSphere: new THREE.SphereGeometry(4, 6, 6),      // Low-poly for folders
        nodeSphere: new THREE.SphereGeometry(2, 4, 4),        // Very low-poly for chunks
        folderGlow: new THREE.SphereGeometry(5.2, 4, 4),      // Glow uses even fewer segments
        nodeGlow: new THREE.SphereGeometry(2.6, 4, 4),
    };

    // Shared materials - created ONCE, cloned when needed for animation
    sharedMaterials = {
        folder: new THREE.MeshBasicMaterial({
            color: 0x00ff41,
            transparent: true,
            opacity: 0.9
        }),
        node: new THREE.MeshBasicMaterial({
            color: 0x008f11,
            transparent: true,
            opacity: 0.9
        }),
        glow: new THREE.MeshBasicMaterial({
            color: 0x00ff41,
            transparent: true,
            opacity: 0.15
        })
    };

    console.log('[PERF] Shared geometries/materials initialized');
}

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

export async function initConstellation() {
    await waitForThreeJS();

    const container = document.getElementById('constellation-background');
    const tooltip = document.getElementById('node-tooltip');

    if (!container) {
        console.error('[CONSTELLATION] Missing container element: constellation-background');
        return;
    }
    if (!tooltip) {
        console.warn('[CONSTELLATION] Missing tooltip element: node-tooltip');
    }

    try {
        console.log('[CONSTELLATION] Fetching data...');
        const res = await fetch('/api/constellation?include_similarity=true&similarity_threshold=' + state.similarityThreshold);
        const data = await res.json();
        console.log('[CONSTELLATION] Response:', data.ok, 'nodes:', data.nodes?.length, 'edges:', data.edges?.length);

        if (!data.ok || !data.nodes || data.nodes.length === 0) {
            console.log('[CONSTELLATION] No data available:', data.error || 'empty');
            return;
        }

        const nodes = data.nodes.map(n => ({
            id: n.id,
            label: n.label,
            group: n.group,
            path: n.path || '',
            val: n.group === 'folder' ? 3 : 1
        }));

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
        console.log('[CONSTELLATION] Nodes:', nodes.length, 'Links:', links.length);

        // Performance mode for large datasets (>500 nodes)
        const perfMode = nodes.length > 500;
        if (perfMode) {
            console.log('[PERF] Large dataset detected, enabling performance optimizations');
        }

        // Initialize shared assets BEFORE creating graph
        initSharedAssets();

        nodes.forEach(n => state.nodeRegistry.byId.set(n.id, n));

        // Only initialize pulse state for smaller datasets (perf optimization)
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

        console.log('[CONSTELLATION] Creating graph with', nodes.length, 'nodes and', links.length, 'links');

        const Graph = ForceGraph3D({
            controlType: 'orbit',
            rendererConfig: {
                antialias: !perfMode,           // Disable AA in perf mode
                powerPreference: 'high-performance',
                precision: perfMode ? 'lowp' : 'highp'
            }
        })(container)
            .graphData({ nodes, links })
            .backgroundColor('#000000')
            .showNavInfo(false)
            .nodeLabel(node => `${node.label}\n${node.path || node.id}`)
            .nodeColor(node => node.group === 'folder' ? '#00ff41' : '#008f11')
            .nodeOpacity(0.9)
            .nodeResolution(perfMode ? 4 : 6)   // Lower resolution in perf mode
            .nodeVal(node => node.val)
            .linkColor(link => {
                // Resident-decided links (different colors)
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
                // Standard links
                if (link.type === 'smash_trail') {
                    return 'rgba(255, 102, 0, 0.9)';         // Orange
                }
                if (link.type === 'similarity') {
                    const alpha = 0.15 + (link.weight || 0.5) * 0.25;
                    return `rgba(100, 255, 255, ${alpha})`;  // Cyan
                }
                return 'rgba(0, 143, 17, 0.2)';              // Dark green (hierarchy)
            })
            .linkWidth(link => {
                if (link.type === 'smash_trail') return 2;
                if (link.type === 'similarity') return 1;
                return 0.3;
            })
            .linkOpacity(link => {
                if (link.type === 'smash_trail') return 1.0;
                if (link.type === 'similarity') return 0.3;
                return 0.2;
            })
            .d3AlphaDecay(perfMode ? 0.05 : 0.02)     // Faster decay in perf mode
            .d3VelocityDecay(perfMode ? 0.6 : 0.5)
            .warmupTicks(perfMode ? 20 : 50)           // Much faster warmup
            .cooldownTicks(perfMode ? 50 : 100)        // Faster cooldown
            .enableNodeDrag(false)
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
                if (node) focusCameraOnNode(node, 1500);
            });

        state.setGraph(Graph);

        // Read from state (set by loadSettings) instead of DOM sliders
        const { linkDistance, linkStrength, repel, center } = state.graphSettings;
        console.log('[CONSTELLATION] Init forces from state:', { linkDistance, linkStrength, repel, center });

        window.graphForces = {
            linkDistance: linkDistance,
            linkStrength: linkStrength,
            chargeStrength: -repel,
            centerStrength: center
        };

        Graph.d3Force('link').distance(linkDistance).strength(linkStrength);
        Graph.d3Force('charge').strength(-repel).distanceMax(300);
        Graph.d3Force('center').strength(center);

        Graph.nodeThreeObject(node => {
            const group = new THREE.Group();
            const isFolder = node.group === 'folder';

            // Use SHARED geometry, clone material only for nodes that need animation
            const geometry = isFolder ? sharedGeometries.folderSphere : sharedGeometries.nodeSphere;
            const baseMaterial = isFolder ? sharedMaterials.folder : sharedMaterials.node;

            // Clone material so each node can have independent color during flash
            const material = baseMaterial.clone();
            const sphere = new THREE.Mesh(geometry, material);
            group.add(sphere);

            // Only add glow in non-perf mode or for folders
            if (!perfMode || isFolder) {
                const glowGeometry = isFolder ? sharedGeometries.folderGlow : sharedGeometries.nodeGlow;
                const glowMaterial = sharedMaterials.glow.clone();
                const glow = new THREE.Mesh(glowGeometry, glowMaterial);
                group.add(glow);
                node.__glowSphere = glow;
            }

            node.__mainSphere = sphere;

            return group;
        });

        const scene = Graph.scene();
        scene.add(new THREE.AmbientLight(0x00ff41, 0.3));
        const pointLight = new THREE.PointLight(0x00ff41, 1, 500);
        pointLight.position.set(0, 100, 100);
        scene.add(pointLight);
        // Read fog from state (set by loadSettings)
        const fogDensity = state.graphSettings.fog;
        console.log('[CONSTELLATION] Init fog from state:', fogDensity);
        scene.fog = new THREE.FogExp2(0x000000, fogDensity);

        Graph.cameraPosition({ x: 0, y: 0, z: 400 }, { x: 0, y: 0, z: 0 }, 0);

        document.addEventListener('mousemove', (e) => {
            if (tooltip.style.display === 'block') {
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            }
        });

        // Use ResizeObserver to size graph to container (not window)
        const resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    Graph.width(width).height(height);
                }
            }
        });
        resizeObserver.observe(container);

        console.log(`Constellation initialized with ${nodes.length} nodes`);

    } catch (e) {
        console.error('Constellation init error:', e);
    }
}

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

export function addToTrail(nodeId, activityType) {
    state.explorationTrail.push({ nodeId, timestamp: Date.now(), type: activityType });
    while (state.explorationTrail.length > MAX_TRAIL_LENGTH) state.explorationTrail.shift();
    updateTrailVisualization();
}

function updateTrailVisualization() {
    if (!state.Graph) return;
    const scene = state.Graph.scene();

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
        const age = (Date.now() - entry.timestamp) / 30000;
        const alpha = Math.max(0.1, 1 - age);
        const colorInfo = ACTIVITY_COLORS[entry.type] || ACTIVITY_COLORS.default;
        const color = new THREE.Color(colorInfo.main);
        colors.push(color.r * alpha, color.g * alpha, color.b * alpha);
    });

    if (points.length < 2) return;

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const material = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7 });
    trailLine = new THREE.Line(geometry, material);
    scene.add(trailLine);
}

export function spawnNode(nodeData) {
    if (!state.Graph) return;
    const graphData = state.Graph.graphData();
    if (state.nodeRegistry.byId.has(nodeData.node_id)) return;

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
    state.nodePulseState.set(newNode.id, { intensity: 1.0, phase: 0, frequency: 1.0, lastActivity: Date.now() });

    if (!state.staticCameraMode) {
        setTimeout(() => {
            const node = state.nodeRegistry.byId.get(nodeData.node_id);
            if (node) focusCameraOnNode(node, 1500);
        }, 300);
    }
}

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
                node.__mainSphere.material.color.setHex(node.group === 'folder' ? 0x00ff41 : 0x008f11);
                if (node.__mainSphere.material.emissive) {
                    node.__mainSphere.material.emissive.setHex(node.group === 'folder' ? 0x003300 : 0x001a00);
                }
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

// Smasher cursor - persistent visual showing current analysis position
let smasherCursorGroup = null;
let smasherCursorRing = null;
let smasherTrailLine = null;
let smasherAnimationFrame = null;

function createSmasherCursor() {
    if (!state.Graph) return;
    const scene = state.Graph.scene();

    // Remove old cursor if exists
    if (smasherCursorGroup) {
        scene.remove(smasherCursorGroup);
        smasherCursorGroup = null;
    }

    smasherCursorGroup = new THREE.Group();

    // Outer ring - rotating
    const ringGeometry = new THREE.TorusGeometry(8, 0.5, 8, 32);
    const ringMaterial = new THREE.MeshBasicMaterial({
        color: 0xff6600,  // Orange
        transparent: true,
        opacity: 0.8
    });
    smasherCursorRing = new THREE.Mesh(ringGeometry, ringMaterial);
    smasherCursorGroup.add(smasherCursorRing);

    // Inner glow sphere
    const glowGeometry = new THREE.SphereGeometry(5, 16, 16);
    const glowMaterial = new THREE.MeshBasicMaterial({
        color: 0xff6600,
        transparent: true,
        opacity: 0.3
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    smasherCursorGroup.add(glow);

    // Pulsing core
    const coreGeometry = new THREE.SphereGeometry(2, 8, 8);
    const coreMaterial = new THREE.MeshBasicMaterial({
        color: 0xffaa00,
        transparent: true,
        opacity: 0.9
    });
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    smasherCursorGroup.add(core);

    smasherCursorGroup.visible = false;
    scene.add(smasherCursorGroup);

    // Start animation loop
    if (!smasherAnimationFrame) {
        animateSmasherCursor();
    }

    return smasherCursorGroup;
}

function animateSmasherCursor() {
    smasherAnimationFrame = requestAnimationFrame(animateSmasherCursor);

    if (!smasherCursorGroup || !smasherCursorGroup.visible) return;

    const time = Date.now() * 0.002;

    // Rotate the ring
    if (smasherCursorRing) {
        smasherCursorRing.rotation.x = time * 0.5;
        smasherCursorRing.rotation.y = time * 0.7;
    }

    // Pulse the scale
    const pulse = 1 + Math.sin(time * 3) * 0.15;
    smasherCursorGroup.scale.setScalar(pulse);
}

export function moveSmasherCursor(nodeId, gateOpen) {
    if (!state.Graph) return;

    // Create cursor if it doesn't exist
    if (!smasherCursorGroup) {
        createSmasherCursor();
    }

    const node = state.nodeRegistry.byId.get(nodeId);
    if (!node || node.x === undefined) {
        smasherCursorGroup.visible = false;
        return;
    }

    // Move cursor to node position
    smasherCursorGroup.position.set(node.x, node.y, node.z);
    smasherCursorGroup.visible = true;

    // Change color based on gate state
    const color = gateOpen ? 0x00ff41 : 0xff4444;
    smasherCursorGroup.children.forEach(child => {
        if (child.material) {
            child.material.color.setHex(color);
        }
    });

    // Add to smasher trail
    state.addToSmasherTrail(nodeId);
    updateSmasherTrail();
}

function updateSmasherTrail() {
    if (!state.Graph) return;
    const scene = state.Graph.scene();

    // Remove old trail
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

        // Fade from orange (old) to yellow (new)
        const age = index / state.smasherTrailNodes.length;
        const r = 1.0;
        const g = 0.4 + age * 0.6;
        const b = 0.0;
        colors.push(r, g, b);
    });

    if (points.length < 2) return;

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const material = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        linewidth: 2
    });
    smasherTrailLine = new THREE.Line(geometry, material);
    scene.add(smasherTrailLine);
}

export function hideSmasherCursor() {
    if (smasherCursorGroup) {
        smasherCursorGroup.visible = false;
    }
    // Clear the trail when cursor is hidden
    if (smasherTrailLine) {
        smasherTrailLine.visible = false;
    }
    state.smasherTrailNodes.length = 0;
    // Cancel animation frame to stop wasting CPU when cursor is hidden
    if (smasherAnimationFrame) {
        cancelAnimationFrame(smasherAnimationFrame);
        smasherAnimationFrame = null;
    }
}

// Track active flash timers to prevent memory leaks
const activeFlashTimers = new Map();

export function flashNode(nodeId, gateOpen, E) {
    const node = state.nodeRegistry.byId.get(nodeId);
    if (!node) return;

    // Move smasher cursor to this node
    moveSmasherCursor(nodeId, gateOpen);

    // Cancel any existing timers for this node to prevent memory leaks
    const existingTimers = activeFlashTimers.get(nodeId);
    if (existingTimers) {
        existingTimers.forEach(timerId => clearTimeout(timerId));
    }
    const newTimers = [];

    const flashColor = gateOpen ? 0x00ff41 : 0xff4444;

    if (node.__mainSphere) {
        node.__mainSphere.scale.setScalar(2.5);
        node.__mainSphere.material.color.setHex(flashColor);
        node.__mainSphere.material.opacity = 1.0;

        const mainTimer = setTimeout(() => {
            if (node.__mainSphere) {
                node.__mainSphere.scale.setScalar(1.0);
                node.__mainSphere.material.color.setHex(node.group === 'folder' ? 0x00ff41 : 0x008f11);
                node.__mainSphere.material.opacity = 0.9;
            }
            // Clean up timer reference
            const timers = activeFlashTimers.get(nodeId);
            if (timers) {
                const idx = timers.indexOf(mainTimer);
                if (idx > -1) timers.splice(idx, 1);
                if (timers.length === 0) activeFlashTimers.delete(nodeId);
            }
        }, 500);
        newTimers.push(mainTimer);
    }

    if (node.__glowSphere) {
        node.__glowSphere.scale.setScalar(2.5);
        node.__glowSphere.material.color.setHex(flashColor);
        node.__glowSphere.material.opacity = 0.6;

        const glowTimer = setTimeout(() => {
            if (node.__glowSphere) {
                node.__glowSphere.scale.setScalar(1.3);
                node.__glowSphere.material.color.setHex(0x00ff41);
                node.__glowSphere.material.opacity = 0.15;
            }
            // Clean up timer reference
            const timers = activeFlashTimers.get(nodeId);
            if (timers) {
                const idx = timers.indexOf(glowTimer);
                if (idx > -1) timers.splice(idx, 1);
                if (timers.length === 0) activeFlashTimers.delete(nodeId);
            }
        }, 500);
        newTimers.push(glowTimer);
    }

    if (newTimers.length > 0) {
        activeFlashTimers.set(nodeId, newTimers);
    }

    if (!state.staticCameraMode) {
        focusCameraOnNode(node, 500);
    }
}

// ===== SIMILARITY / LINK CONTROLS =====
export function filterLinks(links) {
    return links.filter(l => {
        // Always show hierarchy edges
        if (l.type !== 'similarity') return true;
        // Filter similarity edges by toggle AND threshold
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
        const res = await fetch(`/api/constellation?include_similarity=true&similarity_threshold=${state.similarityThreshold}`);
        const data = await res.json();
        if (!data.ok) return;

        const nodes = data.nodes.map(n => ({
            id: n.id,
            label: n.label,
            group: n.group,
            path: n.path || '',
            val: n.group === 'folder' ? 3 : 1
        }));

        state.allLinks.length = 0;
        data.edges.forEach(e => {
            state.allLinks.push({
                source: e.from,
                target: e.to,
                type: e.type || 'hierarchy',
                weight: e.weight || 0.5
            });
        });

        nodes.forEach(n => state.nodeRegistry.byId.set(n.id, n));
        const visibleLinks = filterLinks(state.allLinks);
        state.Graph.graphData({ nodes, links: visibleLinks });
    } catch (e) {
        console.error('Failed to reload constellation:', e);
    }
}

// ===== FOG CONTROL =====
// Debounce timer for settings save to prevent race conditions
let saveSettingsTimer = null;
function debouncedSaveSettings() {
    if (saveSettingsTimer) clearTimeout(saveSettingsTimer);
    saveSettingsTimer = setTimeout(() => {
        window.saveSettings?.();
        saveSettingsTimer = null;
    }, 300);
}

export function updateFog(value) {
    if (!state.Graph) return;
    const density = parseFloat(value);
    state.Graph.scene().fog.density = density;
    state.updateGraphSetting('fog', density);  // Track in state
    const valueEl = document.getElementById('value-fog');
    if (valueEl) valueEl.innerText = density.toFixed(4);
    debouncedSaveSettings();  // Debounced persist
}

// ===== GRAPH FORCE CONTROLS =====
export function updateGraphForce(force, value) {
    if (!state.Graph || !window.graphForces) return;
    const numValue = parseFloat(value);

    // Safely get d3 forces with null checks
    const linkForce = state.Graph.d3Force('link');
    const chargeForce = state.Graph.d3Force('charge');
    const centerForce = state.Graph.d3Force('center');

    switch (force) {
        case 'center':
            window.graphForces.centerStrength = numValue;
            if (centerForce) centerForce.strength(numValue);
            state.updateGraphSetting('center', numValue);
            const centerEl = document.getElementById('value-center');
            if (centerEl) centerEl.innerText = numValue.toFixed(2);
            break;
        case 'repel':
            const chargeStrength = -numValue;
            window.graphForces.chargeStrength = chargeStrength;
            if (chargeForce) chargeForce.strength(chargeStrength);
            state.updateGraphSetting('repel', numValue);
            const repelEl = document.getElementById('value-repel');
            if (repelEl) repelEl.innerText = chargeStrength;
            break;
        case 'linkStrength':
            window.graphForces.linkStrength = numValue;
            if (linkForce) linkForce.strength(numValue);
            state.updateGraphSetting('linkStrength', numValue);
            const linkStrEl = document.getElementById('value-link-strength');
            if (linkStrEl) linkStrEl.innerText = numValue.toFixed(2);
            break;
        case 'linkDistance':
            window.graphForces.linkDistance = numValue;
            if (linkForce) linkForce.distance(numValue);
            state.updateGraphSetting('linkDistance', numValue);
            const linkDistEl = document.getElementById('value-link-distance');
            if (linkDistEl) linkDistEl.innerText = numValue;
            break;
    }
    state.Graph.d3ReheatSimulation();
    debouncedSaveSettings();  // Debounced persist
}

export function resetGraphForces() {
    if (!state.Graph) return;
    window.graphForces = {
        linkDistance: 100,
        linkStrength: 0.5,
        chargeStrength: -120,
        centerStrength: 0.05
    };
    state.Graph.d3Force('link').distance(100).strength(0.5);
    state.Graph.d3Force('charge').strength(-120);
    state.Graph.d3Force('center').strength(0.05);
    state.Graph.scene().fog.density = 0.003;
    state.Graph.d3ReheatSimulation();

    state.setShowSimilarityLinks(true);
    state.setSimilarityThreshold(0.35);
    document.getElementById('toggle-similarity').className = 'behavior-toggle on';
    document.getElementById('slider-sim-threshold').value = 0.35;
    document.getElementById('value-sim-threshold').innerText = '0.35';

    document.getElementById('slider-fog').value = 0.003;
    document.getElementById('slider-center').value = 0.05;
    document.getElementById('slider-repel').value = 120;
    document.getElementById('slider-link-strength').value = 0.50;
    document.getElementById('slider-link-distance').value = 100;
    document.getElementById('value-fog').innerText = '0.0030';
    document.getElementById('value-center').innerText = '0.05';
    document.getElementById('value-repel').innerText = '-120';
    document.getElementById('value-link-strength').innerText = '0.50';
    document.getElementById('value-link-distance').innerText = '100';

    reloadConstellation();
}

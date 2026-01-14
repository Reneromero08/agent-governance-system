        // ===== STATE =====
        let ws = null;
        let daemonRunning = false;
        let behaviors = {};
        let chatOpen = false;
        let sidebarCollapsed = false;
        let threejsReady = false;
        let pendingThoughtId = null; // Track pending chat to prevent duplicates
        let showSimilarityLinks = true;
        let similarityThreshold = 0.7;
        let allLinks = []; // Store all links for filtering

        // Particle Smasher state
        let smasherActive = false;
        let smasherConfig = { delay_ms: 100, batch_size: 10 };
        let currentThreshold = 0.080;  // Dynamic threshold from Q46 (read-only, updated from daemon)
        let nMemories = 0;  // Memory count for nucleation display
        let staticCameraMode = true;  // Default ON for smasher visualization

        // Render throttling for smasher
        let smashQueue = [];
        let smashRafPending = false;
        const MAX_SMASH_BATCH = 5;  // Process max 5 smashes per frame

        // ===== SIDEBAR TOGGLE =====
        function toggleSidebar() {
            sidebarCollapsed = !sidebarCollapsed;
            document.getElementById('app').classList.toggle('sidebar-collapsed', sidebarCollapsed);
        }

        // ===== SECTION TOGGLE =====
        function toggleSection(name) {
            const section = document.getElementById(`section-${name}`);
            section.classList.toggle('collapsed');
        }

        // ===== CHAT TOGGLE =====
        function toggleChat() {
            chatOpen = !chatOpen;
            document.getElementById('chat-panel').classList.toggle('open', chatOpen);
        }

        // ===== API HELPERS =====
        async function api(endpoint, options = {}) {
            const res = await fetch(`/api${endpoint}`, {
                headers: { 'Content-Type': 'application/json' },
                ...options
            });
            return res.json();
        }

        // ===== WEBSOCKET =====
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('ws-dot').classList.remove('offline');
                document.getElementById('ws-status-text').innerText = 'Connected';
            };

            ws.onclose = () => {
                document.getElementById('ws-dot').classList.add('offline');
                document.getElementById('ws-status-text').innerText = 'Disconnected';
                setTimeout(connectWebSocket, 3000);
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleWebSocketMessage(msg);
            };
        }

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
                // Particle Smasher: throttled processing to prevent render overload
                updateCurrentFile(msg.data);
                updateSmasherStats(msg.data.rate);
                queueSmashVisualization(msg.data);
            } else if (msg.type === 'hot_reload') {
                // HOT RELOAD: Refresh page when static files change
                console.log('[HOT RELOAD] File changed, reloading...');
                window.location.reload();
            }
            // NOTE: Removed 'thought' handler - API response handles chat messages
            // This fixes the duplicate message bug
        }

        // ===== MIND STATE =====
        function updateMindState(data) {
            if (data.Df !== undefined) {
                document.getElementById('mind-df').innerText = data.Df.toFixed(1);
                document.getElementById('df-progress').style.width = Math.min(100, data.Df / 2.56) + '%';
            }
            if (data.distance !== undefined) {
                document.getElementById('mind-distance').innerText = data.distance.toFixed(3);
            }
        }

        async function loadStatus() {
            try {
                const data = await api('/status');
                if (data.ok) {
                    document.getElementById('mind-df').innerText = data.mind.Df.toFixed(1);
                    document.getElementById('df-progress').style.width = Math.min(100, data.mind.Df / 2.56) + '%';
                    document.getElementById('mind-distance').innerText = data.mind.distance_from_start.toFixed(3);
                    document.getElementById('mind-interactions').innerText = data.interactions;
                }
            } catch (e) {
                console.error('Failed to load status:', e);
            }
        }

        async function loadEvolution() {
            try {
                const data = await api('/evolution');
                if (data.ok && data.Df_history) {
                    updateSparkline(data.Df_history);
                }
            } catch (e) {
                console.error('Failed to load evolution:', e);
            }
        }

        function updateSparkline(history) {
            const container = document.getElementById('df-sparkline');
            const last30 = history.slice(-30);
            if (last30.length === 0) return;

            const max = Math.max(...last30);
            const min = Math.min(...last30);
            const range = max - min || 1;

            container.innerHTML = '';
            last30.forEach(v => {
                const bar = document.createElement('div');
                bar.className = 'spark-bar';
                const height = ((v - min) / range) * 100;
                bar.style.height = Math.max(5, height) + '%';
                container.appendChild(bar);
            });
        }

        // ===== DAEMON CONTROL =====
        async function loadDaemonStatus() {
            try {
                const data = await api('/daemon/status');
                if (data.ok) {
                    updateDaemonStatus(data);
                }
            } catch (e) {
                console.error('Failed to load daemon status:', e);
            }
        }

        function updateDaemonStatus(data) {
            daemonRunning = data.running;
            behaviors = data.behaviors || {};

            document.getElementById('daemon-status-text').innerText = data.running ? 'Running' : 'Stopped';
            document.getElementById('daemon-led').className = data.running ? 'daemon-led on' : 'daemon-led';
            const btn = document.getElementById('daemon-toggle-btn');
            btn.innerText = data.running ? 'Stop' : 'Start';
            btn.className = data.running ? 'daemon-btn' : 'daemon-btn primary';

            updateBehaviorUI('paper_exploration', 'paper');
            updateBehaviorUI('memory_consolidation', 'consolidate');
            updateBehaviorUI('self_reflection', 'reflect');
            updateBehaviorUI('cassette_watch', 'cassette');

            // Q46: Update dynamic threshold display (read-only)
            if (data.threshold !== undefined) {
                // Get n_memories from smasher stats or explored_chunks as proxy
                const n = data.explored_chunks || 0;
                updateThresholdDisplay(data.threshold, n);
            }
        }

        function updateBehaviorUI(name, shortName) {
            const toggle = document.getElementById(`toggle-${shortName}`);
            const input = document.getElementById(`input-${shortName}`);

            if (behaviors[name]) {
                toggle.className = behaviors[name].enabled ? 'behavior-toggle on' : 'behavior-toggle';
                if (input) input.value = behaviors[name].interval;
            }
        }

        async function toggleDaemon() {
            if (daemonRunning) {
                await api('/daemon/stop', { method: 'POST' });
            } else {
                await api('/daemon/start', { method: 'POST' });
            }
            loadDaemonStatus();
        }

        async function toggleBehavior(name) {
            if (!behaviors[name]) return;
            const newEnabled = !behaviors[name].enabled;
            await api('/daemon/config', {
                method: 'POST',
                body: JSON.stringify({ behavior: name, enabled: newEnabled })
            });
            loadDaemonStatus();
        }

        async function updateInterval(name, seconds) {
            const interval = parseInt(seconds, 10);
            if (isNaN(interval) || interval < 5) return;
            await api('/daemon/config', {
                method: 'POST',
                body: JSON.stringify({ behavior: name, interval: interval })
            });
            loadDaemonStatus();
        }

        // ===== SIMILARITY CONTROLS =====
        function toggleSimilarityLinks() {
            showSimilarityLinks = !showSimilarityLinks;
            document.getElementById('toggle-similarity').className = showSimilarityLinks ? 'behavior-toggle on' : 'behavior-toggle';
            updateVisibleLinks();
            saveSettings();  // Persist immediately
        }

        function updateSimThreshold(value) {
            similarityThreshold = parseFloat(value);
            document.getElementById('value-sim-threshold').innerText = similarityThreshold.toFixed(2);

            // Count how many edges pass the threshold
            const passingEdges = allLinks.filter(l => l.type === 'similarity' && l.weight >= similarityThreshold);
            console.log(`[SIM] Threshold: ${similarityThreshold.toFixed(2)} → ${passingEdges.length} connections visible`);

            // Just update visible links, don't reload everything
            updateVisibleLinks();
        }

        async function reloadConstellation() {
            if (!Graph) return;
            try {
                const res = await fetch('/api/constellation');
                const data = await res.json();
                if (!data.ok) return;

                // Get current graph data to preserve positions
                const currentData = Graph.graphData();
                const currentPositions = new Map();
                currentData.nodes.forEach(n => {
                    if (n.x !== undefined) {
                        currentPositions.set(n.id, { x: n.x, y: n.y, z: n.z });
                    }
                });

                const nodes = data.nodes.map(n => {
                    const pos = currentPositions.get(n.id);
                    return {
                        id: n.id,
                        label: n.label,
                        group: n.group,
                        path: n.path || '',
                        paper_id: n.paper_id || null,
                        val: n.group === 'folder' ? 3 : 1,
                        // Preserve position AND fix it to prevent flying
                        ...(pos ? { x: pos.x, y: pos.y, z: pos.z, fx: pos.x, fy: pos.y, fz: pos.z } : {})
                    };
                });

                allLinks = data.edges.map(e => ({
                    source: e.from,
                    target: e.to,
                    type: e.type || 'hierarchy',
                    weight: e.weight || 0
                }));

                nodes.forEach(n => nodeRegistry.byId.set(n.id, n));

                // Update with ALL links - visibility handled by linkVisibility callback
                Graph.graphData({ nodes, links: allLinks });

                // Unfix positions after graph settles
                setTimeout(() => {
                    const data = Graph.graphData();
                    data.nodes.forEach(n => {
                        delete n.fx;
                        delete n.fy;
                        delete n.fz;
                    });
                }, 500);
            } catch (e) {
                console.error('Failed to reload constellation:', e);
            }
        }

        function filterLinks(links) {
            return links.filter(l => {
                // Always show hierarchy edges
                if (l.type !== 'similarity') return true;
                // Filter similarity edges by toggle AND threshold
                if (!showSimilarityLinks) return false;
                // Only show similarity edges above threshold
                return (l.weight || 0) >= similarityThreshold;
            });
        }

        function updateVisibleLinks() {
            if (!Graph) return;
            const filteredLinks = filterLinks(allLinks);
            Graph.graphData({
                nodes: Graph.graphData().nodes,
                links: filteredLinks
            });
        }

        // ===== FOG CONTROL =====
        function updateFog(value) {
            if (!Graph) return;
            const density = parseFloat(value);
            Graph.scene().fog.density = density;
            document.getElementById('value-fog').innerText = density.toFixed(4);
        }

        // ===== GRAPH FORCE CONTROLS =====
        function updateGraphForce(force, value) {
            if (!Graph || !window.graphForces) return;
            const numValue = parseFloat(value);

            switch (force) {
                case 'center':
                    window.graphForces.centerStrength = numValue;
                    Graph.d3Force('center').strength(numValue);
                    document.getElementById('value-center').innerText = numValue.toFixed(2);
                    break;
                case 'repel':
                    const chargeStrength = -numValue;
                    window.graphForces.chargeStrength = chargeStrength;
                    Graph.d3Force('charge').strength(chargeStrength);
                    document.getElementById('value-repel').innerText = chargeStrength;
                    break;
                case 'linkStrength':
                    window.graphForces.linkStrength = numValue;
                    Graph.d3Force('link').strength(numValue);
                    document.getElementById('value-link-strength').innerText = numValue.toFixed(2);
                    break;
                case 'linkDistance':
                    window.graphForces.linkDistance = numValue;
                    Graph.d3Force('link').distance(numValue);
                    document.getElementById('value-link-distance').innerText = numValue;
                    break;
            }
            // Gentle reheat - low alpha so nodes don't fly away
            Graph.d3AlphaTarget(0.1).d3ReheatSimulation();
            setTimeout(() => Graph.d3AlphaTarget(0), 300);
        }

        function resetGraphForces() {
            if (!Graph) return;
            window.graphForces = {
                linkDistance: 80,
                linkStrength: 0.7,
                chargeStrength: -80,
                centerStrength: 0.15
            };
            Graph.d3Force('link').distance(80).strength(0.7);
            Graph.d3Force('charge').strength(-80).distanceMax(200);
            Graph.d3Force('center').strength(0.15);
            Graph.scene().fog.density = 0.003;

            // Gentle reheat
            Graph.d3AlphaTarget(0.1).d3ReheatSimulation();
            setTimeout(() => Graph.d3AlphaTarget(0), 500);

            // Reset similarity settings
            showSimilarityLinks = true;
            similarityThreshold = 0.35;
            document.getElementById('toggle-similarity').className = 'behavior-toggle on';
            document.getElementById('slider-sim-threshold').value = 0.35;
            document.getElementById('value-sim-threshold').innerText = '0.35';

            // Sliders with stable defaults
            document.getElementById('slider-fog').value = 0.003;
            document.getElementById('slider-center').value = 0.15;
            document.getElementById('slider-repel').value = 80;
            document.getElementById('slider-link-strength').value = 0.70;
            document.getElementById('slider-link-distance').value = 80;
            document.getElementById('value-fog').innerText = '0.0030';
            document.getElementById('value-center').innerText = '0.15';
            document.getElementById('value-repel').innerText = '-80';
            document.getElementById('value-link-strength').innerText = '0.70';
            document.getElementById('value-link-distance').innerText = '80';

            reloadConstellation();
        }

        // ===== PARTICLE SMASHER CONTROLS =====
        async function toggleSmasher() {
            if (smasherActive) {
                await stopSmasher();
            } else {
                await startSmasher();
            }
        }

        async function startSmasher() {
            try {
                clearCurrentFile();  // Reset current file display
                const res = await api('/smasher/start', {
                    method: 'POST',
                    body: JSON.stringify({
                        delay_ms: smasherConfig.delay_ms,
                        batch_size: smasherConfig.batch_size,
                        batch_pause_ms: 200,
                        max_chunks: 0
                    })
                });
                if (res.ok) {
                    smasherActive = true;
                    updateSmasherUI();
                }
            } catch (e) {
                console.error('Failed to start smasher:', e);
            }
        }

        async function stopSmasher() {
            try {
                const res = await api('/smasher/stop', { method: 'POST' });
                smasherActive = false;
                updateSmasherUI();
            } catch (e) {
                console.error('Failed to stop smasher:', e);
            }
        }

        function updateSmasherUI() {
            const led = document.getElementById('smasher-led');
            const text = document.getElementById('smasher-status-text');
            const btn = document.getElementById('smasher-toggle-btn');
            const stats = document.getElementById('smasher-stats');
            const currentFile = document.getElementById('smasher-current');

            if (smasherActive) {
                led.className = 'daemon-led on';
                text.innerText = 'SMASHING';
                btn.innerText = 'STOP';
                btn.className = 'daemon-btn';
                stats.style.display = 'block';
                currentFile.classList.add('active');
            } else {
                led.className = 'daemon-led';
                text.innerText = 'Idle';
                btn.innerText = 'SMASH';
                btn.className = 'daemon-btn primary';
                currentFile.classList.remove('active');
            }
        }

        function updateSmasherStats(rate) {
            document.getElementById('smasher-rate').innerText = rate.toFixed(1);
        }

        // Update the single-line current file indicator
        function updateCurrentFile(data) {
            const container = document.getElementById('smasher-current');
            const fileEl = document.getElementById('smasher-current-file');
            const eEl = document.getElementById('smasher-current-e');

            container.classList.add('active');

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

        function clearCurrentFile() {
            const container = document.getElementById('smasher-current');
            container.classList.remove('active');
            document.getElementById('smasher-current-file').innerText = '--';
            document.getElementById('smasher-current-e').innerText = '';
        }

        // Queue smash visualization with throttling
        function queueSmashVisualization(data) {
            smashQueue.push(data);
            if (!smashRafPending) {
                smashRafPending = true;
                requestAnimationFrame(processSmashQueue);
            }
        }

        // Process queued smash visualizations (batched - SINGLE graph update)
        function processSmashQueue() {
            smashRafPending = false;
            const batch = smashQueue.splice(0, MAX_SMASH_BATCH);
            if (batch.length === 0 || !Graph) return;

            // Get graph data ONCE for entire batch
            const graphData = Graph.graphData();
            let graphUpdated = false;
            const nodesToFlash = [];

            for (const data of batch) {
                const result = processSmashItem(data, graphData);
                if (result.updated) graphUpdated = true;
                if (result.node) nodesToFlash.push({ nodeId: data.node_id, gateOpen: data.gate_open, E: data.E });
            }

            // Update graph ONCE for entire batch
            if (graphUpdated) {
                Graph.graphData(graphData);
            }

            // Flash nodes after graph update
            for (const item of nodesToFlash) {
                flashNode(item.nodeId, item.gateOpen, item.E);
            }

            // If more in queue, schedule another frame
            if (smashQueue.length > 0) {
                smashRafPending = true;
                requestAnimationFrame(processSmashQueue);
            }
        }

        // Process single smash item (returns {updated, node})
        // Uses SEMANTIC SIMILARITY for positioning - connects to most similar existing node
        function processSmashItem(data, graphData) {
            const nodeId = data.node_id;
            const similarTo = data.similar_to;  // SEMANTIC anchor (most similar existing node)
            const similarE = data.similar_E || 0;
            let updated = false;

            // Try to find existing node in registry
            let node = nodeRegistry.byId.get(nodeId);

            // If node doesn't exist, create it dynamically
            if (!node && data.is_new_node) {
                // Position near SEMANTICALLY SIMILAR node (not sequential!)
                let pos = { x: 0, y: 0, z: 0 };
                let foundAnchor = false;

                if (similarTo) {
                    const anchorNode = nodeRegistry.byId.get(similarTo);
                    if (anchorNode && anchorNode.x !== undefined) {
                        // Position near similar node - offset scaled by similarity
                        // High similarity = closer, low similarity = further
                        const offset = 15 + (1 - similarE) * 25;  // 15-40 units based on similarity
                        pos = {
                            x: anchorNode.x + (Math.random() - 0.5) * offset,
                            y: anchorNode.y + (Math.random() - 0.5) * offset,
                            z: anchorNode.z + (Math.random() - 0.5) * offset
                        };
                        foundAnchor = true;
                    }
                }

                // No anchor found - place near center, force sim will position
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

                nodeRegistry.byId.set(nodeId, node);
                graphData.nodes.push(node);
                updated = true;

                // Initialize pulse state
                nodePulseState.set(nodeId, {
                    intensity: 1.5,
                    phase: 0,
                    frequency: 2.0,
                    lastActivity: Date.now()
                });
            }

            // Create SIMILARITY edge to anchor node (semantic connection!)
            if (similarTo && similarE > 0.3) {  // Only connect if reasonably similar
                const hasAnchor = nodeRegistry.byId.has(similarTo);
                const hasTarget = nodeRegistry.byId.has(nodeId);

                if (hasAnchor && hasTarget) {
                    // Check if edge already exists
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
                            weight: similarE  // Store similarity score for edge width
                        });
                        updated = true;
                    }
                }
            }

            // Add to exploration trail
            addToTrail(nodeId, 'smash');

            return { updated, node };
        }

        async function updateSmasherSpeed(value) {
            smasherConfig.delay_ms = parseInt(value);
            document.getElementById('value-smasher-speed').innerText = value + 'ms';
            // LIVE update - no restart needed
            await sendSmasherConfig({ delay_ms: smasherConfig.delay_ms });
        }

        async function updateSmasherBatch(value) {
            smasherConfig.batch_size = parseInt(value);
            document.getElementById('value-smasher-batch').innerText = value;
            // LIVE update - no restart needed
            await sendSmasherConfig({ batch_size: smasherConfig.batch_size });
        }

        // Apply slider range from config.json (now supports float min/max/step/default)
        function applySliderRange(sliderId, config, valueId = null) {
            if (!config) return;
            const slider = document.getElementById(sliderId);
            if (!slider) return;
            if (config.min !== undefined) slider.min = config.min;
            if (config.max !== undefined) slider.max = config.max;
            if (config.step !== undefined) slider.step = config.step;
            if (config.default !== undefined) {
                slider.value = config.default;
                // Also update the display value with special formatting
                if (valueId) {
                    const valueEl = document.getElementById(valueId);
                    if (valueEl) {
                        const val = parseFloat(config.default);
                        const step = config.step || 1;
                        const decimals = step < 1 ? Math.max(2, Math.ceil(-Math.log10(step))) : 0;
                        // Special cases for formatting
                        if (valueId === 'value-smasher-speed') {
                            valueEl.innerText = val + 'ms';
                        } else if (valueId === 'value-repel') {
                            valueEl.innerText = -val;  // Display as negative
                        } else {
                            valueEl.innerText = val.toFixed(decimals);
                        }
                    }
                }
            }
        }

        // Q46: Threshold is DYNAMIC based on N (read-only display)
        function updateThresholdDisplay(threshold, n) {
            currentThreshold = threshold;
            nMemories = n;
            document.getElementById('value-e-threshold').innerText = threshold.toFixed(3);
            document.getElementById('value-n-memories').innerText = n;
            // Progress bar: 0.08 (N=1) to 0.159 (N=inf), show as percentage of max
            const maxThreshold = 1.0 / (2.0 * Math.PI);  // ~0.159
            const progress = Math.min(100, (threshold / maxThreshold) * 100);
            document.getElementById('threshold-progress').style.width = progress + '%';
        }

        // Send config update to server (writes to config.json)
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

        function toggleStaticCamera() {
            staticCameraMode = !staticCameraMode;
            document.getElementById('toggle-static-camera').className =
                staticCameraMode ? 'behavior-toggle on' : 'behavior-toggle';
            saveSettings();  // Persist immediately
        }

        async function loadSmasherStatus() {
            try {
                const res = await api('/smasher/status');
                if (res.ok) {
                    smasherActive = res.active;
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

        // Flash a node DRAMATICALLY (for particle smasher) - OPTIMIZED
        function flashNode(nodeId, gateOpen, E) {
            const node = nodeRegistry.byId.get(nodeId);
            if (!node) return;

            // Color based on gate status
            const flashColor = gateOpen ? 0x00ff41 : 0xff4444;  // Green=absorbed, Red=rejected

            if (node.__mainSphere) {
                // Scale up and change color (BasicMaterial - no emissive)
                node.__mainSphere.scale.setScalar(3.0);
                node.__mainSphere.material.color.setHex(flashColor);
                node.__mainSphere.material.opacity = 1.0;

                // Single timeout to reset (PERFORMANCE: fewer timers)
                setTimeout(() => {
                    if (node.__mainSphere) {
                        node.__mainSphere.scale.setScalar(1.0);
                        node.__mainSphere.material.color.setHex(node.group === 'folder' ? 0x00ff41 : 0x008f11);
                        node.__mainSphere.material.opacity = 0.9;
                    }
                }, 300);
            }

            if (node.__glowSphere) {
                node.__glowSphere.scale.setScalar(3.0);
                node.__glowSphere.material.color.setHex(flashColor);
                node.__glowSphere.material.opacity = 0.6;

                setTimeout(() => {
                    if (node.__glowSphere) {
                        node.__glowSphere.scale.setScalar(1.3);
                        node.__glowSphere.material.color.setHex(0x00ff41);
                        node.__glowSphere.material.opacity = 0.15;
                    }
                }, 300);
            }

            // NO camera follow in static mode
            if (!staticCameraMode) {
                focusCameraOnNode(node, 500);
            }
        }

        // ===== CHAT =====
        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const query = input.value.trim();
            if (!query) return;

            input.value = '';
            addChatMessage('user', query);

            try {
                const data = await api('/think', {
                    method: 'POST',
                    body: JSON.stringify({ query })
                });
                addChatMessage('feral', data.response, data.E_resonance, data.gate_open);
                loadStatus();
                loadEvolution();
            } catch (e) {
                addChatMessage('feral', '[ERROR] Failed to get response', 0, false);
            }
        }

        function addChatMessage(role, text, E = null, gateOpen = null) {
            const container = document.getElementById('chat-messages');
            const msg = document.createElement('div');
            msg.className = `chat-msg ${role}`;

            let meta = `<span class="chat-sender">${role === 'user' ? 'You' : 'Feral'}</span>`;
            if (E !== null) {
                const badgeClass = gateOpen ? 'open' : 'closed';
                meta += `<span class="chat-e-badge ${badgeClass}">E=${E.toFixed(2)} ${gateOpen ? 'OPEN' : 'CLOSED'}</span>`;
            }

            msg.innerHTML = `
                <div class="chat-meta">${meta}</div>
                <div class="chat-bubble">${text}</div>
            `;
            container.appendChild(msg);
            container.scrollTop = container.scrollHeight;
        }

        // ===== ACTIVITY =====
        function addActivity(activity) {
            const feed = document.getElementById('activity-feed');

            // Remove placeholder
            if (feed.children.length === 1 && feed.children[0].style.color) {
                feed.innerHTML = '';
            }

            const item = document.createElement('div');
            item.className = 'activity-item';

            const time = new Date(activity.timestamp).toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false
            });

            item.innerHTML = `
                <span class="activity-time">${time}</span>
                <span class="activity-badge ${activity.action}">${activity.action}</span>
                <span class="activity-text">${activity.summary}</span>
            `;

            feed.insertBefore(item, feed.firstChild);

            // Limit items
            while (feed.children.length > 20) {
                feed.removeChild(feed.lastChild);
            }
        }

        // ===== 3D CONSTELLATION =====
        let Graph = null;
        let nodeRegistry = { byId: new Map() };
        let nodePulseState = new Map();
        let explorationTrail = [];
        let trailLine = null;
        const MAX_TRAIL_LENGTH = 50;

        const ACTIVITY_COLORS = {
            paper: { main: 0x00ff41, glow: '#00ff41' },
            consolidate: { main: 0x00ff41, glow: '#00ff41' },
            reflect: { main: 0x00ff41, glow: '#00ff41' },
            cassette: { main: 0x00ff41, glow: '#00ff41' },
            daemon: { main: 0x00ff41, glow: '#00ff41' },
            default: { main: 0x00ff41, glow: '#00ff41' }
        };

        function waitForThreeJS() {
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

        async function initConstellation() {
            await waitForThreeJS();

            const container = document.getElementById('constellation-background');
            const tooltip = document.getElementById('node-tooltip');

            try {
                console.log('[CONSTELLATION] Fetching data...');
                const res = await fetch('/api/constellation');
                const data = await res.json();
                console.log('[CONSTELLATION] Response:', data.ok, 'nodes:', data.nodes?.length, 'edges:', data.edges?.length, 'similarity:', data.similarity_edge_count);

                if (!data.ok || !data.nodes || data.nodes.length === 0) {
                    console.log('[CONSTELLATION] No data available:', data.error || 'empty');
                    return;
                }

                const nodes = data.nodes.map(n => ({
                    id: n.id,
                    label: n.label,
                    group: n.group,
                    path: n.path || '',
                    paper_id: n.paper_id || null,
                    val: n.group === 'folder' ? 3 : 1
                }));

                allLinks = data.edges.map(e => ({
                    source: e.from,
                    target: e.to,
                    type: e.type || 'hierarchy',
                    weight: e.weight || 0
                }));

                const simLinks = allLinks.filter(l => l.type === 'similarity');
                console.log('[CONSTELLATION] Nodes:', nodes.length, 'Total links:', allLinks.length, 'Similarity edges:', simLinks.length);

                nodes.forEach(n => nodeRegistry.byId.set(n.id, n));

                nodes.forEach(n => {
                    nodePulseState.set(n.id, {
                        intensity: 0.4,
                        phase: Math.random() * Math.PI * 2,
                        frequency: 0.5 + Math.random() * 0.5,
                        lastActivity: Date.now()
                    });
                });

                console.log('[CONSTELLATION] Creating graph with', nodes.length, 'nodes');

                // Load ALL links - visibility is controlled by linkVisibility callback
                Graph = ForceGraph3D({ controlType: 'orbit' })(container)
                    .graphData({ nodes, links: allLinks })
                    .backgroundColor('#000000')
                    .showNavInfo(false)
                    .nodeLabel(node => `${node.label}\n${node.path || node.id}`)
                    .nodeColor(node => node.group === 'folder' ? '#00ff41' : '#008f11')
                    .nodeOpacity(0.9)
                    .nodeResolution(8)  // PERFORMANCE: Lower poly count
                    .nodeVal(node => node.val)
                    .linkColor(link => {
                        // Smash trail edges in bright orange
                        if (link.type === 'smash_trail') {
                            return 'rgba(255, 102, 0, 0.9)';
                        }
                        // Similarity edges in lighter cyan with more transparency
                        if (link.type === 'similarity') {
                            const alpha = 0.15 + (link.weight || 0.5) * 0.25;  // Reduced opacity
                            return `rgba(100, 255, 255, ${alpha})`;  // Lighter cyan
                        }
                        return 'rgba(0, 143, 17, 0.2)';
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
                    .linkVisibility(link => {
                        // Similarity edges filtered by toggle AND threshold
                        if (link.type !== 'similarity') return true;
                        if (!showSimilarityLinks) return false;
                        return (link.weight || 0) >= similarityThreshold;
                    })
                    .d3AlphaDecay(0.05)  // Faster settling to prevent drift
                    .d3VelocityDecay(0.7)  // Higher damping to prevent flying
                    .warmupTicks(100)   // More warmup for stable initial layout
                    .cooldownTicks(200)  // More cooldown for stability
                    .enableNodeDrag(false)  // PERFORMANCE: Disable dragging
                    .onNodeHover(node => {
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

                window.graphForces = {
                    linkDistance: 80,
                    linkStrength: 0.7,
                    chargeStrength: -80,
                    centerStrength: 0.15
                };

                // Stable forces: strong center, limited repulsion, tight links
                Graph.d3Force('link').distance(80).strength(0.7);
                Graph.d3Force('charge').strength(-80).distanceMax(200);
                Graph.d3Force('center').strength(0.15);

                Graph.nodeThreeObject(node => {
                    const group = new THREE.Group();

                    // PERFORMANCE: Use 8 segments instead of 16 (way fewer triangles)
                    const size = node.group === 'folder' ? 4 : 2;
                    const geometry = new THREE.SphereGeometry(size, 8, 8);
                    const material = new THREE.MeshBasicMaterial({  // PERFORMANCE: BasicMaterial instead of Phong
                        color: node.group === 'folder' ? 0x00ff41 : 0x008f11,
                        transparent: true,
                        opacity: 0.9
                    });
                    const sphere = new THREE.Mesh(geometry, material);
                    group.add(sphere);

                    // PERFORMANCE: Simpler glow with fewer segments
                    const glowGeometry = new THREE.SphereGeometry(size * 1.3, 6, 6);
                    const glowMaterial = new THREE.MeshBasicMaterial({
                        color: 0x00ff41,
                        transparent: true,
                        opacity: 0.15
                    });
                    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
                    group.add(glow);

                    node.__mainSphere = sphere;
                    node.__glowSphere = glow;

                    return group;
                });

                const scene = Graph.scene();
                scene.add(new THREE.AmbientLight(0x00ff41, 0.3));
                const pointLight = new THREE.PointLight(0x00ff41, 1, 500);
                pointLight.position.set(0, 100, 100);
                scene.add(pointLight);
                scene.fog = new THREE.FogExp2(0x000000, 0.003);

                Graph.cameraPosition({ x: 0, y: 0, z: 400 }, { x: 0, y: 0, z: 0 }, 0);

                document.addEventListener('mousemove', (e) => {
                    if (tooltip.style.display === 'block') {
                        tooltip.style.left = (e.clientX + 15) + 'px';
                        tooltip.style.top = (e.clientY + 15) + 'px';
                    }
                });

                startPulseAnimation();
                console.log(`Constellation initialized with ${nodes.length} nodes`);

            } catch (e) {
                console.error('Constellation init error:', e);
            }
        }

        // PERFORMANCE: Disabled continuous animation - too expensive
        // Pulse effects now only happen on flash events
        function startPulseAnimation() {
            // Intentionally empty - animations are handled per-flash now
            console.log('[PERFORMANCE] Continuous pulse animation disabled');
        }

        function focusCameraOnNode(node, duration = 1500) {
            if (!Graph || !node) return;
            const distance = 150;
            const distRatio = 1 + distance / Math.max(10, Math.hypot(node.x || 0, node.y || 0, node.z || 0));
            Graph.cameraPosition(
                { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
                { x: node.x || 0, y: node.y || 0, z: node.z || 0 },
                duration
            );
        }

        function addToTrail(nodeId, activityType) {
            explorationTrail.push({ nodeId, timestamp: Date.now(), type: activityType });
            while (explorationTrail.length > MAX_TRAIL_LENGTH) explorationTrail.shift();
            updateTrailVisualization();
        }

        function updateTrailVisualization() {
            if (!Graph) return;
            const scene = Graph.scene();

            if (trailLine) {
                scene.remove(trailLine);
                trailLine.geometry.dispose();
                trailLine.material.dispose();
                trailLine = null;
            }

            if (explorationTrail.length < 2) return;

            const points = [];
            const colors = [];

            explorationTrail.forEach((entry) => {
                const node = nodeRegistry.byId.get(entry.nodeId);
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

        function spawnNode(nodeData) {
            if (!Graph) return;
            const graphData = Graph.graphData();
            if (nodeRegistry.byId.has(nodeData.node_id)) return;

            let parentPos = { x: 0, y: 0, z: 0 };
            if (nodeData.source_id) {
                const parent = nodeRegistry.byId.get(nodeData.source_id);
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

            nodeRegistry.byId.set(newNode.id, newNode);
            graphData.nodes.push(newNode);

            if (nodeData.source_id && nodeRegistry.byId.has(nodeData.source_id)) {
                graphData.links.push({ source: nodeData.source_id, target: nodeData.node_id });
            }

            Graph.graphData(graphData);
            nodePulseState.set(newNode.id, { intensity: 1.0, phase: 0, frequency: 1.0, lastActivity: Date.now() });

            // Only follow camera if not in static mode
            if (!staticCameraMode) {
                setTimeout(() => {
                    const node = nodeRegistry.byId.get(nodeData.node_id);
                    if (node) focusCameraOnNode(node, 1500);
                }, 300);
            }
        }

        function activateNode(nodeId, activityType) {
            const node = nodeRegistry.byId.get(nodeId);
            if (!node) return;

            const state = nodePulseState.get(nodeId);
            if (state) {
                state.intensity = 1.0;
                state.lastActivity = Date.now();
            }

            const colorInfo = ACTIVITY_COLORS[activityType] || ACTIVITY_COLORS.default;
            if (node.__mainSphere) {
                node.__mainSphere.material.color.setHex(colorInfo.main);
                node.__mainSphere.material.emissive.setHex(colorInfo.main);
                node.__mainSphere.material.emissiveIntensity = 0.8;

                setTimeout(() => {
                    if (node.__mainSphere) {
                        node.__mainSphere.material.color.setHex(node.group === 'folder' ? 0x00ff41 : 0x008f11);
                        node.__mainSphere.material.emissive.setHex(node.group === 'folder' ? 0x003300 : 0x001a00);
                    }
                }, 3000);
            }

            if (node.__glowSphere) {
                node.__glowSphere.material.color.setHex(colorInfo.main);
                node.__glowSphere.material.opacity = 0.4;
            }

            // Only follow camera if not in static mode
            if (!staticCameraMode) {
                focusCameraOnNode(node, 1000);
            }
        }

        // ===== SETTINGS PERSISTENCE =====
        // UI settings go to localStorage, daemon settings sync to config.json
        function saveSettings() {
            const settings = {
                // Smasher config (also synced to config.json)
                smasher_delay_ms: smasherConfig.delay_ms,
                smasher_batch_size: smasherConfig.batch_size,

                // Graph/UI settings (localStorage only)
                similarity_threshold: similarityThreshold,
                show_similarity_links: showSimilarityLinks,
                static_camera: staticCameraMode,

                // Force params (if Graph exists)
                fog_density: Graph ? parseFloat(document.getElementById('value-fog').innerText) : 0.003,
                center_strength: Graph ? parseFloat(document.getElementById('value-center').innerText) : 0.05,
                repel_strength: Graph ? parseFloat(document.getElementById('value-repel').innerText) : -120,
                link_strength: Graph ? parseFloat(document.getElementById('value-link-strength').innerText) : 0.5,
                link_distance: Graph ? parseFloat(document.getElementById('value-link-distance').innerText) : 100
            };

            localStorage.setItem('feral_settings', JSON.stringify(settings));
        }

        async function loadSettings() {
            // Load config.json - this is the SOURCE OF TRUTH for slider ranges and defaults
            try {
                const configRes = await api('/config');
                if (configRes.ok && configRes.config) {
                    const cfg = configRes.config;

                    // Apply smasher config
                    if (cfg.smasher) {
                        if (cfg.smasher.delay_ms !== undefined) {
                            smasherConfig.delay_ms = cfg.smasher.delay_ms;
                            document.getElementById('slider-smasher-speed').value = cfg.smasher.delay_ms;
                            document.getElementById('value-smasher-speed').innerText = cfg.smasher.delay_ms + 'ms';
                        }
                        if (cfg.smasher.batch_size !== undefined) {
                            smasherConfig.batch_size = cfg.smasher.batch_size;
                            document.getElementById('slider-smasher-batch').value = cfg.smasher.batch_size;
                            document.getElementById('value-smasher-batch').innerText = cfg.smasher.batch_size;
                        }
                    }

                    // Apply slider ranges AND defaults from config.json
                    // Config.json is authoritative - localStorage is cleared to prevent conflicts
                    if (cfg.ui && cfg.ui.sliders) {
                        const s = cfg.ui.sliders;

                        // Clear old localStorage to prevent stale values overriding config
                        localStorage.removeItem('feral_settings');

                        applySliderRange('slider-smasher-speed', s.speed, 'value-smasher-speed');
                        applySliderRange('slider-smasher-batch', s.batch, 'value-smasher-batch');
                        applySliderRange('slider-sim-threshold', s.sim_threshold, 'value-sim-threshold');
                        applySliderRange('slider-fog', s.fog, 'value-fog');
                        applySliderRange('slider-center', s.center, 'value-center');
                        applySliderRange('slider-repel', s.repel, 'value-repel');
                        applySliderRange('slider-link-strength', s.link_strength, 'value-link-strength');
                        applySliderRange('slider-link-distance', s.link_distance, 'value-link-distance');

                        // Set JS variables from config defaults
                        if (s.sim_threshold?.default !== undefined) {
                            similarityThreshold = s.sim_threshold.default;
                        }
                    }

                    console.log('[SETTINGS] Loaded config from config.json');
                }
            } catch (e) {
                console.warn('[SETTINGS] Could not load config.json:', e);
            }
        }

        // Apply current slider values to the Graph (call AFTER Graph is created)
        function applyGraphSettings() {
            if (!Graph) return;

            // Read values from sliders and apply to Graph
            const fog = parseFloat(document.getElementById('slider-fog').value) || 0.003;
            const center = parseFloat(document.getElementById('slider-center').value) || 0.15;
            const repel = parseFloat(document.getElementById('slider-repel').value) || 80;
            const linkStrength = parseFloat(document.getElementById('slider-link-strength').value) || 0.7;
            const linkDistance = parseFloat(document.getElementById('slider-link-distance').value) || 80;

            // Apply to Graph
            if (Graph.scene().fog) {
                Graph.scene().fog.density = fog;
            }

            window.graphForces = {
                linkDistance: linkDistance,
                linkStrength: linkStrength,
                chargeStrength: -repel,
                centerStrength: center
            };

            Graph.d3Force('link').distance(linkDistance).strength(linkStrength);
            Graph.d3Force('charge').strength(-repel).distanceMax(200);
            Graph.d3Force('center').strength(center);

            // Update display values
            document.getElementById('value-fog').innerText = fog.toFixed(4);
            document.getElementById('value-center').innerText = center.toFixed(2);
            document.getElementById('value-repel').innerText = -repel;
            document.getElementById('value-link-strength').innerText = linkStrength.toFixed(2);
            document.getElementById('value-link-distance').innerText = linkDistance;

            console.log('[SETTINGS] Applied graph settings: fog=' + fog + ', center=' + center + ', repel=' + repel);
        }

        // ===== INIT =====
        async function init() {
            await loadSettings();  // Load slider ranges and defaults from config.json
            await initConstellation();  // Create the Graph
            applyGraphSettings();  // NOW apply slider values to the Graph
            await loadStatus();
            await loadEvolution();
            await loadDaemonStatus();
            await loadSmasherStatus();
            connectWebSocket();
            document.getElementById('loading').classList.add('hidden');

            setInterval(loadStatus, 10000);
            setInterval(loadEvolution, 30000);
            setInterval(loadDaemonStatus, 5000);
            setInterval(loadSmasherStatus, 1000);  // Fast polling for smasher stats
            setInterval(saveSettings, 5000);  // Auto-save settings every 5 seconds
        }

        window.onload = init;

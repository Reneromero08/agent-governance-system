// =============================================================================
// FERAL DASHBOARD - CONFIGURATION
// =============================================================================
//
// This file contains ALL tunable constants for the Feral dashboard.
// Edit values here to customize behavior without touching other code.
//
// STRUCTURE:
//   CONFIG.CATEGORY.SETTING = value
//
// Each setting has a comment explaining:
//   - What it does
//   - Valid range or options
//   - Visual/performance impact
//
// =============================================================================

export const CONFIG = {

    // =========================================================================
    // GRAPH - 3D Constellation Visualization
    // =========================================================================
    // The main 3D force-directed graph showing memory nodes and connections

    GRAPH: {
        // ----- PERFORMANCE MODE -----
        // When node count exceeds this, enable performance optimizations
        // Lower = more aggressive optimization, Higher = better quality longer
        // Range: 100-2000, Default: 500
        PERF_MODE_THRESHOLD: 500,

        // ----- NODE GEOMETRY -----
        // Higher segments = smoother spheres, but more GPU load
        // Range: 4-32

        // Folder nodes (larger, more prominent)
        FOLDER_SPHERE_SEGMENTS: 12,      // Default: 12, Min: 6, Max: 24
        FOLDER_SPHERE_RADIUS: 10,         // Default: 4
        FOLDER_GLOW_SEGMENTS: 8,         // Default: 8
        FOLDER_GLOW_RADIUS: 10,         // Default: 5.2 (slightly larger than sphere)

        // Chunk/page nodes (smaller, numerous)
        NODE_SPHERE_SEGMENTS: 8,         // Default: 8, Min: 4, Max: 16
        NODE_SPHERE_RADIUS: 2,           // Default: 2
        NODE_GLOW_SEGMENTS: 6,           // Default: 6
        NODE_GLOW_RADIUS: 2.6,           // Default: 2.6

        // ----- NODE RESOLUTION -----
        // ForceGraph3D's internal node resolution
        // This affects the built-in node spheres (before custom objects)
        // Range: 4-24
        NODE_RESOLUTION_NORMAL: 12,      // Default: 12 (when < PERF_MODE_THRESHOLD nodes)
        NODE_RESOLUTION_PERF: 8,         // Default: 8 (when >= PERF_MODE_THRESHOLD nodes)

        // ----- RENDERER QUALITY -----
        // Antialias: smooths jagged edges, slight GPU cost
        ANTIALIAS: true,                 // Default: true (false for very slow GPUs)

        // Precision: shader precision level
        // Options: 'highp', 'mediump', 'lowp'
        PRECISION_NORMAL: 'highp',       // Default: 'highp'
        PRECISION_PERF: 'mediump',       // Default: 'mediump'

        // ----- LINK WIDTHS -----
        // Thickness of connection lines between nodes
        // Range: 0.1-5.0
        LINK_WIDTH_HIERARCHY: 1.5,       // Default: 0.5 (parent-child connections)
        LINK_WIDTH_SIMILARITY: 1.5,      // Default: 1.5 (cosine similarity edges)
        LINK_WIDTH_SMASH_TRAIL: 2.5,     // Default: 2.5 (smasher traversal path)
        LINK_WIDTH_MIND_PROJECTED: 1.5,  // Default: 1.5 (mind-created links)
        LINK_WIDTH_CO_RETRIEVAL: 1.5,    // Default: 1.5 (co-retrieved nodes)
        LINK_WIDTH_ENTANGLEMENT: 1.5,    // Default: 1.5 (entangled nodes)

        // ----- LINK COLORS -----
        // RGBA format: 'rgba(R, G, B, A)' where A is 0.0-1.0
        LINK_COLOR_HIERARCHY: 'rgba(0, 143, 17, 0.2)',       // Dark green
        LINK_COLOR_SMASH_TRAIL: 'rgba(255, 102, 0, 0.9)',    // Orange
        // Note: similarity/mind/co-retrieval colors are computed dynamically based on weight

        // ----- PHYSICS SIMULATION -----
        // Controls how the graph settles into position

        // Alpha decay: how quickly simulation cools down (higher = faster settle)
        // Range: 0.01-0.1
        ALPHA_DECAY_NORMAL: 0.02,        // Default: 0.02
        ALPHA_DECAY_PERF: 0.05,          // Default: 0.05

        // Velocity decay: friction (higher = nodes slow faster)
        // Range: 0.1-0.9
        VELOCITY_DECAY_NORMAL: 0.5,      // Default: 0.5
        VELOCITY_DECAY_PERF: 0.6,        // Default: 0.6

        // Warmup/cooldown ticks: iterations before/after interaction
        WARMUP_TICKS_NORMAL: 50,         // Default: 50
        WARMUP_TICKS_PERF: 20,           // Default: 20
        COOLDOWN_TICKS_NORMAL: 100,      // Default: 100
        COOLDOWN_TICKS_PERF: 50,         // Default: 50

        // ----- FORCE DEFAULTS -----
        // These are the starting values for graph physics sliders
        FORCE_LINK_DISTANCE: 100,        // Default: 100 (pixels between connected nodes)
        FORCE_LINK_STRENGTH: 0.5,        // Default: 0.5 (0-1, how strongly links pull)
        FORCE_CHARGE_STRENGTH: -120,     // Default: -120 (negative = repel, range: -500 to 0)
        FORCE_CENTER_STRENGTH: 0.05,     // Default: 0.05 (0-1, pull toward center)
        FORCE_CHARGE_MAX_DISTANCE: 300,  // Default: 300 (repulsion range limit)

        // ----- FOG -----
        // Depth fog that fades distant nodes (adds depth perception)
        // Range: 0 (none) to 0.01 (very dense)
        FOG_DENSITY: 0.0006,             // Default: 0.0006

        // ----- CAMERA -----
        INITIAL_CAMERA_Z: 400,           // Default: 400 (starting distance)
        FOCUS_DURATION_MS: 1500,         // Default: 1500 (camera animation time)
    },

    // =========================================================================
    // SMASHER CURSOR - The 3D indicator showing current analysis position
    // =========================================================================

    SMASHER_CURSOR: {
        // ----- RING (outer torus) -----
        RING_RADIUS: 8,                  // Default: 8 (overall size)
        RING_TUBE_RADIUS: 0.5,           // Default: 0.5 (tube thickness)
        RING_TUBE_SEGMENTS: 12,          // Default: 12 (smoothness around tube)
        RING_RADIAL_SEGMENTS: 48,        // Default: 48 (smoothness around ring)
        RING_COLOR: 0xff6600,            // Default: 0xff6600 (orange)
        RING_OPACITY: 0.8,               // Default: 0.8

        // ----- GLOW SPHERE (middle layer) -----
        GLOW_RADIUS: 10,                  // Default: 5
        GLOW_SEGMENTS: 16,               // Default: 16
        GLOW_COLOR: 0xff6600,            // Default: 0xff6600
        GLOW_OPACITY: 0.3,               // Default: 0.3

        // ----- CORE SPHERE (center) -----
        CORE_RADIUS: 2,                  // Default: 2
        CORE_SEGMENTS: 12,               // Default: 12
        CORE_COLOR: 0xffaa00,            // Default: 0xffaa00 (yellow-orange)
        CORE_OPACITY: 0.9,               // Default: 0.9

        // ----- ANIMATION -----
        ROTATION_SPEED_X: 0.5,           // Default: 0.5 (radians per second)
        ROTATION_SPEED_Y: 0.7,           // Default: 0.7
        PULSE_SPEED: 3,                  // Default: 3 (oscillations per second)
        PULSE_AMPLITUDE: 0.15,           // Default: 0.15 (scale variation 0-1)

        // ----- GATE COLORS -----
        COLOR_ABSORBED: 0x00ff41,        // Default: 0x00ff41 (green)
        COLOR_REJECTED: 0xff4444,        // Default: 0xff4444 (red)
    },

    // =========================================================================
    // TRAILS - Path visualization for exploration history
    // =========================================================================

    TRAILS: {
        // ----- EXPLORATION TRAIL -----
        // Shows daemon's exploration path through the graph
        MAX_LENGTH: 50,                  // Default: 50 (max nodes in trail)
        FADE_TIME_MS: 30000,             // Default: 30000 (30s to fully fade)
        MIN_OPACITY: 0.1,                // Default: 0.1 (opacity at max age)
        LINE_OPACITY: 0.7,               // Default: 0.7 (base line opacity)

        // ----- SMASHER TRAIL -----
        // Shows recent smasher analysis path
        SMASHER_TRAIL_LENGTH: 20,        // Default: 20 (nodes to keep)
        SMASHER_LINE_OPACITY: 0.8,       // Default: 0.8
        SMASHER_LINE_WIDTH: 2,           // Default: 2 (note: >1 may not work on all GPUs)
    },

    // =========================================================================
    // NODE FLASH - Animation when nodes are activated
    // =========================================================================

    NODE_FLASH: {
        SCALE: 2.5,                      // Default: 2.5 (scale multiplier during flash)
        OPACITY: 1.0,                    // Default: 1.0 (opacity during flash)
        DURATION_MS: 500,                // Default: 500 (flash duration)
        GLOW_SCALE: 2.5,                 // Default: 2.5
        GLOW_OPACITY: 0.6,               // Default: 0.6
    },

    // =========================================================================
    // ACTIVITY FEED - Bottom status bar
    // =========================================================================

    ACTIVITY: {
        MAX_ITEMS: 20,                   // Default: 20 (max items shown)
        TIME_FORMAT: {                   // Time display format
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        },
    },

    // =========================================================================
    // SMASHER - Particle smasher behavior
    // =========================================================================

    SMASHER: {
        // ----- QUEUE MANAGEMENT -----
        // Prevents memory exhaustion under heavy load
        MAX_QUEUE_SIZE: 100,             // Default: 100 (max pending visualizations)
        QUEUE_DROP_PERCENT: 0.2,         // Default: 0.2 (drop 20% when overflow)
        MAX_BATCH_PER_FRAME: 15,         // Default: 15 (items processed per animation frame)

        // ----- NEW NODE POSITIONING -----
        // Where to place newly discovered nodes
        ANCHOR_OFFSET_BASE: 15,          // Default: 15 (min offset from anchor)
        ANCHOR_OFFSET_SCALE: 25,         // Default: 25 (additional offset based on similarity)
        RANDOM_SPAWN_RANGE: 30,          // Default: 30 (range when no anchor found)

        // ----- SIMILARITY EDGE THRESHOLD -----
        MIN_SIMILARITY_FOR_EDGE: 0.3,    // Default: 0.3 (minimum E to create edge)
    },

    // =========================================================================
    // WEBSOCKET - Connection settings
    // =========================================================================

    WEBSOCKET: {
        RECONNECT_DELAY_MS: 3000,        // Default: 3000 (3s between reconnect attempts)
    },

    // =========================================================================
    // POLLING - API refresh intervals
    // =========================================================================

    POLLING: {
        STATUS_INTERVAL_MS: 10000,       // Default: 10000 (mind status refresh)
        EVOLUTION_INTERVAL_MS: 30000,    // Default: 30000 (evolution history refresh)
        DAEMON_INTERVAL_MS: 5000,        // Default: 5000 (daemon status refresh)
        SMASHER_INTERVAL_MS: 2000,       // Default: 2000 (smasher stats refresh)
        SETTINGS_SAVE_MS: 10000,         // Default: 10000 (auto-save settings)
    },

    // =========================================================================
    // SPARKLINE - Mind evolution chart
    // =========================================================================

    SPARKLINE: {
        HISTORY_LENGTH: 30,              // Default: 30 (bars shown)
        MIN_BAR_HEIGHT_PERCENT: 5,       // Default: 5 (minimum bar height %)
    },

    // =========================================================================
    // CHAT - Chat panel behavior
    // =========================================================================

    CHAT: {
        // No configurable settings currently
        // Add future chat customizations here
    },

    // =========================================================================
    // COLORS - UI color palette (matches CSS variables)
    // =========================================================================
    // Note: For major theme changes, also edit styles.css :root variables

    COLORS: {
        // ----- NODE COLORS -----
        FOLDER_COLOR: 0x00ff41,          // Default: 0x00ff41 (bright green)
        NODE_COLOR: 0x032C07,            // Default: 0x008f11 (dark green)
        GLOW_COLOR: 0x00ff41,            // Default: 0x00ff41 (bright green)

        // ----- MATERIAL OPACITY -----
        NODE_OPACITY: 0.9,               // Default: 0.9
        GLOW_OPACITY: 0.15,              // Default: 0.15

        // ----- LINK COLORS (computed in graph.js) -----
        // See GRAPH.LINK_COLOR_* for static colors
        // Dynamic colors use weight to compute alpha:
        //   mind_projected: coral red with alpha 0.4-0.9
        //   co_retrieval: gold with alpha 0.4-0.9
        //   entanglement: purple with alpha 0.5-0.9
        //   similarity: cyan with alpha 0.15-0.4
    },
};

// =============================================================================
// QUICK REFERENCE
// =============================================================================
//
// MAKE SPHERES SMOOTHER:
//   Increase GRAPH.FOLDER_SPHERE_SEGMENTS and GRAPH.NODE_SPHERE_SEGMENTS
//
// IMPROVE PERFORMANCE:
//   Lower GRAPH.PERF_MODE_THRESHOLD
//   Decrease *_SEGMENTS values
//   Set GRAPH.ANTIALIAS = false
//
// CHANGE LINK THICKNESS:
//   Adjust GRAPH.LINK_WIDTH_* values
//
// CHANGE HOW FAST GRAPH SETTLES:
//   Increase GRAPH.ALPHA_DECAY_* (faster) or decrease (slower)
//
// CHANGE DEPTH PERCEPTION:
//   Adjust GRAPH.FOG_DENSITY (0 = no fog, 0.01 = very foggy)
//
// CHANGE SMASHER CURSOR SIZE:
//   Adjust SMASHER_CURSOR.RING_RADIUS, GLOW_RADIUS, CORE_RADIUS
//
// CHANGE TRAIL LENGTH:
//   Adjust TRAILS.MAX_LENGTH and TRAILS.SMASHER_TRAIL_LENGTH
//
// CHANGE ACTIVITY FEED SIZE:
//   Adjust ACTIVITY.MAX_ITEMS
//
// CHANGE UPDATE FREQUENCY:
//   Adjust POLLING.* values (lower = more responsive, more server load)
//
// =============================================================================

export default CONFIG;

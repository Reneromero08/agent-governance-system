// =============================================================================
// FERAL DASHBOARD - UI CONTROLS
// =============================================================================
//
// This module handles basic UI interactions:
// - Sidebar collapse/expand
// - Section collapse/expand
// - Chat panel toggle
//
// OVERVIEW:
//   Simple DOM manipulations for UI state. These functions are exposed
//   globally via window.* assignments in main.js for HTML onclick handlers.
//
// SECTIONS IN THIS FILE:
//   1. Imports
//   2. Sidebar Toggle
//   3. Section Toggle
//   4. Chat Panel Toggle
//
// CUSTOMIZATION:
//   - CSS classes control visual appearance (see styles.css)
//   - Element IDs are referenced here; change in index.html if needed
//
// =============================================================================

// =============================================================================
// SECTION 1: IMPORTS
// =============================================================================

import * as state from './state.js';
import { api } from './api.js';

// =============================================================================
// SECTION 2: SIDEBAR TOGGLE
// =============================================================================
// Collapse/expand the left sidebar

/**
 * Toggle sidebar collapsed state
 * Adds/removes 'sidebar-collapsed' class on #app element
 *
 * Called from:
 *   - HTML onclick on sidebar toggle button
 *
 * TWEAK: Collapsed width is set in styles.css
 *   - .sidebar (normal width)
 *   - .sidebar-collapsed .sidebar (collapsed width)
 */
export function toggleSidebar() {
    state.setSidebarCollapsed(!state.sidebarCollapsed);
    document.getElementById('app').classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
}

// =============================================================================
// SECTION 3: SECTION TOGGLE
// =============================================================================
// Collapse/expand individual sidebar sections (Mind, Smasher, Daemon, Graph)

/**
 * Toggle a sidebar section's collapsed state
 *
 * @param {string} name - Section name (matches ID: section-{name})
 *
 * Sections:
 *   - mind: Mind State section
 *   - smasher: Particle Smasher section
 *   - daemon: Daemon section
 *   - graph: Graph Settings section
 *
 * TWEAK: Collapsed section styling in styles.css
 *   - .section.collapsed .section-content (hidden)
 *   - .section.collapsed .section-chevron (rotated)
 *
 * Persists state to config.json via /api/config
 */
export function toggleSection(name) {
    const section = document.getElementById(`section-${name}`);
    section.classList.toggle('collapsed');

    // Save to config.json
    const isCollapsed = section.classList.contains('collapsed');
    api('/config', {
        method: 'POST',
        body: JSON.stringify({ ui: { accordion: { [name]: isCollapsed } } })
    }).catch(() => {}); // Silently ignore save errors
}

/**
 * Load accordion states from config.json and apply to DOM
 * Called on page load from main.js
 */
export async function loadAccordionState() {
    try {
        const res = await api('/config');
        if (res.ok && res.config?.ui?.accordion) {
            const states = res.config.ui.accordion;
            for (const [name, collapsed] of Object.entries(states)) {
                if (name === '_comment') continue;
                const section = document.getElementById(`section-${name}`);
                if (section) {
                    section.classList.toggle('collapsed', collapsed);
                }
            }
        }
    } catch (e) {
        console.warn('[UI] Could not load accordion state:', e);
    }
}

// =============================================================================
// SECTION 4: CHAT PANEL TOGGLE
// =============================================================================
// Show/hide the chat panel on the right side

/**
 * Toggle chat panel visibility
 *
 * Elements affected:
 *   - #chat-panel: The chat panel (add/remove 'open' class)
 *   - #chat-toggle: The toggle button (add/remove 'shifted' class)
 *
 * TWEAK: Chat panel styling in styles.css
 *   - .chat-panel (closed state)
 *   - .chat-panel.open (open state, translated into view)
 */
export function toggleChat() {
    state.setChatOpen(!state.chatOpen);
    document.getElementById('chat-panel').classList.toggle('open', state.chatOpen);
    document.getElementById('chat-toggle').classList.toggle('shifted', state.chatOpen);
}

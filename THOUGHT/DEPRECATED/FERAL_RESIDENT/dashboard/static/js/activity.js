// =============================================================================
// FERAL DASHBOARD - ACTIVITY FEED
// =============================================================================
//
// This module manages the activity feed at the bottom of the dashboard.
// Shows real-time events from the Feral system (smash hits, daemon actions).
//
// OVERVIEW:
//   The activity feed displays a scrolling list of recent events.
//   New items appear at the top, old items are removed to limit memory.
//
// SECTIONS IN THIS FILE:
//   1. Imports
//   2. Add Activity Item
//
// CUSTOMIZATION:
//   - CONFIG.ACTIVITY.MAX_ITEMS controls feed length (default: 20)
//   - CONFIG.ACTIVITY.TIME_FORMAT controls time display
//   - Badge colors are in styles.css (.activity-badge.*)
//
// =============================================================================

// =============================================================================
// SECTION 1: IMPORTS
// =============================================================================

import { CONFIG } from './config.js';

// =============================================================================
// SECTION 2: ADD ACTIVITY ITEM
// =============================================================================

/**
 * Add a new activity item to the feed
 *
 * @param {Object} activity - Activity data
 * @param {number} activity.timestamp - Unix timestamp (ms)
 * @param {string} activity.action - Action type (e.g., 'smash', 'consolidate')
 * @param {string} activity.summary - Short description
 * @param {Object} activity.details - Optional additional details
 * @param {string} activity.details.paper - Paper name (for smash events)
 * @param {string} activity.details.chunk_id - Chunk ID (for smash events)
 *
 * Display format:
 *   [TIME] [BADGE] [TEXT]
 *   Example: "14:32:05 [smash] paper_name:chunk_0 - E=0.15 ABSORBED"
 *
 * TWEAK: Badge colors in styles.css
 *   - .activity-badge.smash.absorbed = green background
 *   - .activity-badge.smash.rejected = red background
 *   - .activity-badge.{action} = action-specific colors
 *
 * TWEAK: CONFIG.ACTIVITY controls behavior
 *   - MAX_ITEMS: Maximum items to show (default: 20)
 *   - TIME_FORMAT: Time display format (default: HH:MM:SS)
 */
export function addActivity(activity) {
    const feed = document.getElementById('activity-feed');

    // Remove placeholder message if present
    // (The placeholder has inline style, real items don't)
    if (feed.children.length === 1 && feed.children[0].style.color) {
        feed.innerHTML = '';
    }

    // Create activity item element
    const item = document.createElement('div');
    item.className = 'activity-item';

    // Format timestamp
    // TWEAK: CONFIG.ACTIVITY.TIME_FORMAT
    const time = new Date(activity.timestamp).toLocaleTimeString('en-US', CONFIG.ACTIVITY.TIME_FORMAT);

    // Build display text with packet name if available
    let displayText = activity.summary;
    if (activity.details && activity.details.paper && activity.details.chunk_id) {
        displayText = `${activity.details.paper}:${activity.details.chunk_id} - ${activity.summary}`;
    }

    // Determine badge class based on action and result
    // TWEAK: Badge colors are in styles.css (.activity-badge.*)
    let badgeClass = activity.action;
    if (activity.summary && activity.summary.includes('ABSORBED')) {
        badgeClass = 'smash absorbed';
    } else if (activity.summary && activity.summary.includes('REJECTED')) {
        badgeClass = 'smash rejected';
    }

    // Build item HTML
    item.innerHTML = `
        <span class="activity-time">${time}</span>
        <span class="activity-badge ${badgeClass}">${activity.action}</span>
        <span class="activity-text">${displayText}</span>
    `;

    // Insert at top of feed (newest first)
    feed.insertBefore(item, feed.firstChild);

    // Limit items to prevent memory growth
    // TWEAK: CONFIG.ACTIVITY.MAX_ITEMS (default: 20)
    while (feed.children.length > CONFIG.ACTIVITY.MAX_ITEMS) {
        feed.removeChild(feed.lastChild);
    }
}

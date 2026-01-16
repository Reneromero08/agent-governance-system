// =============================================================================
// FERAL DASHBOARD - CHAT
// =============================================================================
//
// This module handles the chat interface for interacting with Feral.
// Users can send queries and receive responses with E (resonance) values.
//
// OVERVIEW:
//   The chat panel provides direct interaction with the Feral mind.
//   Each response includes an E (resonance) value and gate status:
//   - E value: How strongly the query resonated with the mind
//   - Gate: OPEN (absorbed into mind) or CLOSED (rejected)
//
// SECTIONS IN THIS FILE:
//   1. Imports
//   2. Send Message
//   3. Display Message
//
// CUSTOMIZATION:
//   - Chat styling in styles.css (.chat-*, .chat-bubble, .chat-e-badge)
//   - E badge colors: .chat-e-badge.open (green), .chat-e-badge.closed (red)
//
// =============================================================================

// =============================================================================
// SECTION 1: IMPORTS
// =============================================================================

import { api } from './api.js';
import { loadStatus, loadEvolution } from './mind.js';

// =============================================================================
// SECTION 2: SEND MESSAGE
// =============================================================================

/**
 * Send a chat message to Feral and display the response
 *
 * Called from:
 *   - HTML onclick on Send button
 *   - HTML onkeypress (Enter key) on input field
 *
 * API endpoint: POST /api/think
 * Request: { query: string }
 * Response: { response: string, E_resonance: number, gate_open: boolean }
 *
 * After sending:
 *   - Displays user message immediately
 *   - Sends to /think API
 *   - Displays Feral's response with E value and gate status
 *   - Refreshes mind status (think changes the mind state)
 */
export async function sendMessage() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    if (!query) return;

    // Clear input and show user message immediately
    input.value = '';
    addChatMessage('user', query);

    try {
        // Send to think API
        const data = await api('/think', {
            method: 'POST',
            body: JSON.stringify({ query })
        });

        // Validate response data to prevent rendering "undefined"
        const response = data.response || '[No response received]';
        const E = typeof data.E_resonance === 'number' ? data.E_resonance : 0;
        const gateOpen = Boolean(data.gate_open);

        // Display Feral's response with E value
        addChatMessage('feral', response, E, gateOpen);

        // Refresh mind status (thinking changes the mind state)
        loadStatus();
        loadEvolution();

    } catch (e) {
        console.error('[CHAT] Think API error:', e);
        addChatMessage('feral', '[ERROR] Failed to get response', 0, false);
    }
}

// =============================================================================
// SECTION 3: DISPLAY MESSAGE
// =============================================================================

/**
 * Add a chat message to the display
 *
 * @param {string} role - 'user' or 'feral'
 * @param {string} text - Message content
 * @param {number|null} E - E (resonance) value (null for user messages)
 * @param {boolean|null} gateOpen - Gate status (null for user messages)
 *
 * Message structure:
 *   <div class="chat-msg {role}">
 *     <div class="chat-meta">
 *       <span class="chat-sender">{You|Feral}</span>
 *       <span class="chat-e-badge {open|closed}">E={value} {OPEN|CLOSED}</span>
 *     </div>
 *     <div class="chat-bubble">{text}</div>
 *   </div>
 *
 * TWEAK: Styling in styles.css
 *   - .chat-msg.user = user message styling
 *   - .chat-msg.feral = feral message styling
 *   - .chat-bubble = message bubble
 *   - .chat-e-badge.open = green E badge
 *   - .chat-e-badge.closed = red E badge
 */
export function addChatMessage(role, text, E = null, gateOpen = null) {
    const container = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = `chat-msg ${role}`;

    // Build meta line (sender name + optional E badge)
    let meta = `<span class="chat-sender">${role === 'user' ? 'You' : 'Feral'}</span>`;
    if (E !== null) {
        const badgeClass = gateOpen ? 'open' : 'closed';
        meta += `<span class="chat-e-badge ${badgeClass}">E=${E.toFixed(2)} ${gateOpen ? 'OPEN' : 'CLOSED'}</span>`;
    }

    // Build message HTML
    msg.innerHTML = `
        <div class="chat-meta">${meta}</div>
        <div class="chat-bubble">${text}</div>
    `;

    // Add to container and scroll to bottom
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

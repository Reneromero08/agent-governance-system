// ===== CHAT =====
import { api } from './api.js';
import { loadStatus, loadEvolution } from './mind.js';

export async function sendMessage() {
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
        // Validate response data to prevent rendering "undefined"
        const response = data.response || '[No response received]';
        const E = typeof data.E_resonance === 'number' ? data.E_resonance : 0;
        const gateOpen = Boolean(data.gate_open);
        addChatMessage('feral', response, E, gateOpen);
        loadStatus();
        loadEvolution();
    } catch (e) {
        console.error('[CHAT] Think API error:', e);
        addChatMessage('feral', '[ERROR] Failed to get response', 0, false);
    }
}

export function addChatMessage(role, text, E = null, gateOpen = null) {
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

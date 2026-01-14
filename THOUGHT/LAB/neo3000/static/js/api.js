// ===== API & WEBSOCKET =====
import * as state from './state.js';

export async function api(endpoint, options = {}) {
    const res = await fetch(`/api${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    return res.json();
}

// Message handler is passed in to avoid circular imports
let messageHandler = null;

export function connectWebSocket(handler) {
    if (handler) {
        messageHandler = handler;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    state.setWs(ws);

    ws.onopen = () => {
        document.getElementById('ws-dot').classList.remove('offline');
        document.getElementById('ws-status-text').innerText = 'Connected';
    };

    ws.onclose = () => {
        document.getElementById('ws-dot').classList.add('offline');
        document.getElementById('ws-status-text').innerText = 'Disconnected';
        setTimeout(() => connectWebSocket(), 3000);
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (messageHandler) {
            messageHandler(msg);
        }
    };
}

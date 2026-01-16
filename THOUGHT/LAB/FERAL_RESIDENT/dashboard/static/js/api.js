// ===== API & WEBSOCKET =====
import * as state from './state.js';

export async function api(endpoint, options = {}) {
    const res = await fetch(`/api${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    if (!res.ok) {
        const error = new Error(`API error: ${res.status} ${res.statusText}`);
        error.status = res.status;
        error.endpoint = endpoint;
        console.error(`[API] ${endpoint} failed:`, res.status, res.statusText);
        throw error;
    }
    return res.json();
}

// Message handler is passed in to avoid circular imports
let messageHandler = null;
let reconnectTimeout = null;
let isConnecting = false;

export function connectWebSocket(handler) {
    if (handler) {
        messageHandler = handler;
    }

    // Prevent multiple concurrent connection attempts
    if (isConnecting) {
        return;
    }

    // Clear any pending reconnect
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }

    isConnecting = true;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    state.setWs(ws);

    ws.onopen = () => {
        isConnecting = false;
        const dot = document.getElementById('ws-dot');
        const text = document.getElementById('ws-status-text');
        if (dot) dot.classList.remove('offline');
        if (text) text.innerText = 'Connected';
    };

    ws.onclose = () => {
        isConnecting = false;
        const dot = document.getElementById('ws-dot');
        const text = document.getElementById('ws-status-text');
        if (dot) dot.classList.add('offline');
        if (text) text.innerText = 'Disconnected';
        // Schedule reconnect only if not already scheduled
        if (!reconnectTimeout) {
            reconnectTimeout = setTimeout(() => {
                reconnectTimeout = null;
                connectWebSocket();
            }, 3000);
        }
    };

    ws.onerror = () => {
        isConnecting = false;
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (messageHandler) {
            messageHandler(msg);
        }
    };
}

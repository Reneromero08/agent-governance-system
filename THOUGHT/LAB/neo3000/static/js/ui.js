// ===== UI CONTROLS =====
import * as state from './state.js';

export function toggleSidebar() {
    state.setSidebarCollapsed(!state.sidebarCollapsed);
    document.getElementById('app').classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
}

export function toggleSection(name) {
    const section = document.getElementById(`section-${name}`);
    section.classList.toggle('collapsed');
}

export function toggleChat() {
    state.setChatOpen(!state.chatOpen);
    document.getElementById('chat-panel').classList.toggle('open', state.chatOpen);
}

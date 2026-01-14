// ===== MIND STATE =====
import { api } from './api.js';

export function updateMindState(data) {
    if (data.Df !== undefined) {
        document.getElementById('mind-df').innerText = data.Df.toFixed(1);
        document.getElementById('df-progress').style.width = Math.min(100, data.Df / 2.56) + '%';
    }
    if (data.distance !== undefined) {
        document.getElementById('mind-distance').innerText = data.distance.toFixed(3);
    }
}

export async function loadStatus() {
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

export async function loadEvolution() {
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

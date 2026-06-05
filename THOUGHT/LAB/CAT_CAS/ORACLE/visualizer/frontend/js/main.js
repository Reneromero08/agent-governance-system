// Entry point. Tab switching + health check.

import { getHealth } from './api.js';

const tabs = document.querySelectorAll('.tab');
const contents = document.querySelectorAll('.tab-content');
const healthPill = document.getElementById('health-pill');

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.disabled) return;
    const id = tab.dataset.tab;
    tabs.forEach(t => t.classList.toggle('active', t === tab));
    contents.forEach(c => c.classList.toggle('active', c.id === 'tab-' + id));
  });
});

async function checkHealth() {
  try {
    const h = await getHealth();
    healthPill.textContent = 'OK \u2014 Phase ' + h.phase;
    healthPill.className = 'pill pill-ok';
    console.log('health:', h);
  } catch (e) {
    healthPill.textContent = 'OFFLINE';
    healthPill.className = 'pill pill-err';
    console.error('health check failed:', e);
  }
}

checkHealth();

// Entry point. Tab switching + health check + dim controllers.

import { getHealth } from './api.js';
import { Dim1Controller } from './dim1.js';

const tabs = document.querySelectorAll('.tab');
const contents = document.querySelectorAll('.tab-content');
const healthPill = document.getElementById('health-pill');

const dimControllers = {};

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.disabled) return;
    const id = tab.dataset.tab;
    tabs.forEach(t => t.classList.toggle('active', t === tab));
    contents.forEach(c => c.classList.toggle('active', c.id === 'tab-' + id));
    if (id === 'dim1' && !dimControllers.dim1) {
      dimControllers.dim1 = new Dim1Controller();
      dimControllers.dim1.init();
    }
  });
});

async function checkHealth() {
  try {
    const h = await getHealth();
    healthPill.textContent = 'OK \u2014 Phase ' + h.phase;
    healthPill.className = 'pill pill-ok';
    console.log('health:', h);
    // Auto-init the active dim1 tab on first load.
    if (!dimControllers.dim1) {
      dimControllers.dim1 = new Dim1Controller();
      dimControllers.dim1.init();
    }
  } catch (e) {
    healthPill.textContent = 'OFFLINE';
    healthPill.className = 'pill pill-err';
    console.error('health check failed:', e);
  }
}

checkHealth();

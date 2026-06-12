// Entry point. Tab switching + health check + dim controllers.

import { getHealth } from './api.js';
import { Dim1Controller } from './dim1.js';
import { Dim2Controller } from './dim2.js';
import { initTheme } from './theme.js';
import { registerKeyboard } from './keyboard.js';
import { note } from './status.js';

const tabs = document.querySelectorAll('.tab');
const contents = document.querySelectorAll('.tab-content');
const healthPill = document.getElementById('health-pill');

const dimControllers = {};

// Init theme (reads localStorage, sets [data-theme]).
initTheme();
registerKeyboard();

// Allow ?theme=light|dark URL param to override localStorage on first load.
{
  const t = new URL(window.location.href).searchParams.get('theme');
  if (t === 'light' || t === 'dark') {
    document.documentElement.setAttribute('data-theme', t);
  }
  const cur = document.documentElement.getAttribute('data-theme') || 'dark';
  const label = document.getElementById('theme-label');
  const icon = document.getElementById('theme-icon');
  if (label) label.textContent = cur;
  if (icon) icon.innerHTML = cur === 'dark' ? '&#9788;' : '&#9788;';
}

// Update theme toggle UI to match the initial theme.
{
  const t = document.documentElement.getAttribute('data-theme') || 'dark';
  const label = document.getElementById('theme-label');
  const icon = document.getElementById('theme-icon');
  if (label) label.textContent = t;
  if (icon) icon.innerHTML = t === 'dark' ? '&#9788;' : '&#9788;';
}

function switchToTab(id) {
  const tab = document.querySelector(`.tab[data-tab="${id}"]`);
  if (!tab || tab.disabled) return;
  tabs.forEach(t => t.classList.toggle('active', t === tab));
  contents.forEach(c => c.classList.toggle('active', c.id === 'tab-' + id));
  if (id === 'dim1' && !dimControllers.dim1) {
    dimControllers.dim1 = new Dim1Controller();
    dimControllers.dim1.init();
  } else if (id === 'dim2' && !dimControllers.dim2) {
    dimControllers.dim2 = new Dim2Controller();
    dimControllers.dim2.init();
  }
}

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.disabled) return;
    switchToTab(tab.dataset.tab);
  });
});

async function checkHealth() {
  try {
    const h = await getHealth();
    healthPill.textContent = 'OK \u2014 Phase ' + h.phase;
    healthPill.className = 'pill pill-ok';
    console.log('health:', h);
    note('connected -- phase ' + h.phase);
    if (!dimControllers.dim1) {
      dimControllers.dim1 = new Dim1Controller();
      dimControllers.dim1.init();
    }
    if (!dimControllers.dim2) {
      dimControllers.dim2 = new Dim2Controller();
      dimControllers.dim2.init();
    }
    // Switch to the URL-requested tab (if any) AFTER controllers are ready.
    const urlTab = new URL(window.location.href).searchParams.get('tab');
    if (urlTab) switchToTab(urlTab);
  } catch (e) {
    healthPill.textContent = 'OFFLINE';
    healthPill.className = 'pill pill-err';
    note('health check failed: ' + e.message);
    console.error('health check failed:', e);
  }
}

checkHealth();

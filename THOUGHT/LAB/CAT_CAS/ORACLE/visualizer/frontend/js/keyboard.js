// Global keyboard shortcuts. Wired in main.js.

import { getTheme, setTheme } from './theme.js';

const HELP_MODAL_ID = 'help-modal';

function getActiveTabId() {
  const active = document.querySelector('.tab-content.active');
  return active ? active.id.replace('tab-', '') : null;
}

function switchTab(id) {
  const tab = document.querySelector(`.tab[data-tab="${id}"]`);
  if (tab && !tab.disabled) tab.click();
}

function openHelp(open) {
  const m = document.getElementById(HELP_MODAL_ID);
  if (!m) return;
  m.classList.toggle('open', open);
}

function toggleTheme() {
  setTheme(getTheme() === 'dark' ? 'light' : 'dark');
  const label = document.getElementById('theme-label');
  const icon = document.getElementById('theme-icon');
  if (label) label.textContent = getTheme();
  if (icon) icon.innerHTML = getTheme() === 'dark' ? '&#9788;' : '&#9788;';
}

export function registerKeyboard() {
  document.addEventListener('keydown', (e) => {
    // Skip if the user is typing in an input/select.
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    if (e.key === '?') { openHelp(true); e.preventDefault(); return; }
    if (e.key === 'Escape') { openHelp(false); return; }
    if (e.key === 't' || e.key === 'T') { toggleTheme(); e.preventDefault(); return; }
    if (e.key === '1') { switchTab('dim1'); e.preventDefault(); return; }
    if (e.key === '2') { switchTab('dim2'); e.preventDefault(); return; }
    if (e.key === '3') { switchTab('dim3'); e.preventDefault(); return; }
    if (e.key === '4') { switchTab('dim4'); e.preventDefault(); return; }
    if (e.key === '5') { switchTab('dim5'); e.preventDefault(); return; }

    // Active-tab actions (R = run, Space = play/pause, U = copy URL)
    const id = getActiveTabId();
    if (id === 'dim1') {
      if (e.key === 'r' || e.key === 'R') {
        document.getElementById('dim1-run').click();
        e.preventDefault();
        return;
      }
      if (e.key === ' ' || e.code === 'Space') {
        const btn = document.getElementById('dim1-animate');
        if (!btn.disabled) btn.click();
        e.preventDefault();
        return;
      }
      if (e.key === 'u' || e.key === 'U') {
        document.getElementById('dim1-copy-url').click();
        e.preventDefault();
        return;
      }
    }
  });

  // Help modal: click backdrop to close.
  document.getElementById(HELP_MODAL_ID)?.addEventListener('click', (e) => {
    if (e.target.id === HELP_MODAL_ID) openHelp(false);
  });

  // Help toggle button.
  document.getElementById('help-toggle')?.addEventListener('click', () => openHelp(true));
  document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);
}

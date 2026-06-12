// CSS variable values, read at runtime so they track theme changes.
//
// We don't cache: getComputedStyle is fast and changes when [data-theme] flips.
// Canvas code can call getT() to get a fresh object whenever it redraws.

const KEYS = [
  'bg', 'bgElev', 'panel', 'line', 'lineStrong',
  'txt', 'dim', 'acc', 'acc2',
  'halt', 'loop', 'warn', 'ok', 'err',
  'tooltipBg',
];

function readVar(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue('--' + name.replace(/[A-Z]/g, m => '-' + m.toLowerCase()))
    .trim();
}

export function getT() {
  const t = {};
  for (const k of KEYS) t[k] = readVar(k);
  return t;
}

// Backward-compat: a default export that calls getT().
export const T = new Proxy({}, { get: (_, k) => readVar(k) });

// Theme controller. Saves preference to localStorage and flips [data-theme].
const STORAGE_KEY = 'ags.theme';

export function getTheme() {
  return localStorage.getItem(STORAGE_KEY) || 'dark';
}

export function setTheme(name) {
  if (name !== 'dark' && name !== 'light') return;
  document.documentElement.setAttribute('data-theme', name);
  localStorage.setItem(STORAGE_KEY, name);
  // Notify any listeners (canvas redraws, etc).
  document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: name } }));
}

export function initTheme() {
  setTheme(getTheme());
}

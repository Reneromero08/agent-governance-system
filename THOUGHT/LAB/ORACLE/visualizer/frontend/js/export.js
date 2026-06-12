// Export helpers: copy URL, download JSON, save PNG.

import { note } from './status.js';

export function buildShareUrl(state) {
  // state: {machine, gamma, loss, nphi}
  const url = new URL(window.location.href);
  url.search = '';
  if (state.machine) url.searchParams.set('machine', state.machine);
  if (state.gamma != null) url.searchParams.set('gamma', String(state.gamma));
  if (state.loss != null) url.searchParams.set('loss', String(state.loss));
  if (state.nphi != null) url.searchParams.set('nphi', String(state.nphi));
  return url.toString();
}

export async function copyShareUrl(state) {
  const url = buildShareUrl(state);
  try {
    await navigator.clipboard.writeText(url);
    note('URL copied to clipboard: ' + url);
  } catch (e) {
    // Fallback: use a hidden textarea + execCommand.
    const ta = document.createElement('textarea');
    ta.value = url;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); note('URL copied: ' + url); }
    catch (e2) { note('Could not copy URL -- see console'); console.log('Share URL:', url); }
    document.body.removeChild(ta);
  }
}

export function downloadJson(name, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name + '.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  note('Downloaded ' + name + '.json');
}

// Stitch 3 canvases side-by-side into one PNG. Returns the data URL.
export function canvasesToPng(canvases, label) {
  if (canvases.length === 0) return null;
  const pad = 12;
  const labelH = 24;
  const widths = canvases.map(c => c.width);
  const heights = canvases.map(c => c.height);
  const W = Math.max(...widths);
  const H = heights.reduce((a, b) => a + b, 0) + (heights.length - 1) * pad + labelH + pad;

  const off = document.createElement('canvas');
  off.width = W;
  off.height = H;
  const ctx = off.getContext('2d');

  // Background
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#0a0d12';
  ctx.fillRect(0, 0, W, H);

  // Label
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--txt').trim() || '#cfd6e0';
  ctx.font = '12px ui-monospace, monospace';
  ctx.fillText(label || 'CAT_CAS Oracle -- 1D view', pad, labelH - 6);

  // Stack canvases
  let y = labelH;
  for (const c of canvases) {
    ctx.drawImage(c, 0, y);
    y += c.height + pad;
  }

  return off.toDataURL('image/png');
}

export function savePng(filename, dataUrl) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  note('Saved ' + filename);
}

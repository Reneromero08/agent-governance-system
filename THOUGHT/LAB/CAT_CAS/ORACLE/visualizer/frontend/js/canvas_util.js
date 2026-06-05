// Canvas helpers: high-DPI sizing, color tokens, basic drawing.

import { CSS_VARS } from './theme.js';

const T = CSS_VARS;

export function getCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function setupCanvas(canvas) {
  // High-DPI scaling. Renders at devicePixelRatio but reports CSS size.
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export function clear(ctx, w, h, bg = T.bg) {
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
}

export function axes(ctx, x0, y0, scale, label = '') {
  // Light reference cross.
  ctx.strokeStyle = T.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, y0);
  ctx.lineTo(ctx.canvas.clientWidth, y0);
  ctx.moveTo(x0, 0);
  ctx.lineTo(x0, ctx.canvas.clientHeight);
  ctx.stroke();

  // Origin marker.
  ctx.fillStyle = T.dim;
  ctx.beginPath();
  ctx.arc(x0, y0, 2, 0, 2 * Math.PI);
  ctx.fill();
}

// Smart axis bounds: include 0 and a small padding. Returns {min, max}.
export function niceBounds(values, padFrac = 0.1) {
  if (values.length === 0) return { min: -1, max: 1 };
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const span = hi - lo;
  lo -= span * padFrac;
  hi += span * padFrac;
  // Make symmetric around 0 if close to it.
  const m = Math.max(Math.abs(lo), Math.abs(hi));
  return { min: -m, max: m };
}

// Draw a colored dot.
export function dot(ctx, x, y, r, color, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
  ctx.restore();
}

// Draw a closed curve (array of {re, im} or {x, y}).
export function path(ctx, points, color, width = 1, closed = true) {
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const x = p.x !== undefined ? p.x : p.re;
    const y = p.y !== undefined ? p.y : p.im;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  if (closed) ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

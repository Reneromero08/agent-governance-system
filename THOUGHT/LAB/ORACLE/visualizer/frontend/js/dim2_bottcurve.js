// Bott index curve: C vs gamma_halt. Computed via /api/dim2/gamma_sweep.

import { getT } from './theme.js';
import { setupCanvas, clear } from './canvas_util.js';

export class BottCurveViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.points = null;   // [{gamma_halt, C, E_fermi_im}, ...]
    this.currentGamma = null;
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
    document.addEventListener('themechange', () => this.draw());
  }

  setData(points, currentGamma) {
    this.points = points;
    this.currentGamma = currentGamma;
    this.draw();
  }

  clear() {
    this.points = null;
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);
  }

  draw() {
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);
    if (!this.points || this.points.length === 0) {
      ctx.fillStyle = T.dim;
      ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('click "Sweep \u03b3" to plot C(\u03b3)', w / 2, h / 2);
      return;
    }

    // Margins
    const m = { l: 36, r: 12, t: 10, b: 26 };
    const pw = w - m.l - m.r;
    const ph = h - m.t - m.b;

    // Compute ranges
    let gMin = Infinity, gMax = -Infinity, cMin = 0, cMax = 0;
    for (const p of this.points) {
      if (p.gamma_halt < gMin) gMin = p.gamma_halt;
      if (p.gamma_halt > gMax) gMax = p.gamma_halt;
      if (p.C < cMin) cMin = p.C;
      if (p.C > cMax) cMax = p.C;
    }
    if (gMin === gMax) { gMin -= 0.5; gMax += 0.5; }
    if (cMin === cMax) { cMin -= 1; cMax += 1; }
    if (cMin > -1) cMin = -1;
    if (cMax < 1) cMax = 1;

    const xS = (g) => m.l + ((g - gMin) / (gMax - gMin)) * pw;
    const yS = (c) => m.t + (1 - (c - cMin) / (cMax - cMin)) * ph;

    // Axes
    ctx.strokeStyle = T.line;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(m.l, m.t);
    ctx.lineTo(m.l, m.t + ph);
    ctx.lineTo(m.l + pw, m.t + ph);
    ctx.stroke();

    // Y grid (integer C values)
    ctx.fillStyle = T.dim;
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let c = Math.ceil(cMin); c <= Math.floor(cMax); c++) {
      const y = yS(c);
      ctx.strokeStyle = c === 0 ? T.lineStrong : T.line;
      ctx.beginPath();
      ctx.moveTo(m.l, y);
      ctx.lineTo(m.l + pw, y);
      ctx.stroke();
      ctx.fillText(String(c), m.l - 4, y);
    }

    // X ticks
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const nTicks = Math.min(6, this.points.length);
    for (let i = 0; i < nTicks; i++) {
      const idx = Math.floor((i / (nTicks - 1 || 1)) * (this.points.length - 1));
      const p = this.points[idx];
      const x = xS(p.gamma_halt);
      ctx.strokeStyle = T.line;
      ctx.beginPath();
      ctx.moveTo(x, m.t + ph);
      ctx.lineTo(x, m.t + ph + 3);
      ctx.stroke();
      ctx.fillText(p.gamma_halt.toFixed(1), x, m.t + ph + 4);
    }

    // Step line for integer C
    ctx.strokeStyle = T.acc;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      const x = xS(p.gamma_halt);
      const y = yS(p.C);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    // Close with horizontal step to next x
    for (let i = 0; i < this.points.length - 1; i++) {
      const p = this.points[i];
      const p2 = this.points[i + 1];
      const x1 = xS(p.gamma_halt);
      const x2 = xS(p2.gamma_halt);
      const y = yS(p.C);
      ctx.lineTo(x2, y);
    }
    ctx.stroke();

    // Dots
    for (const p of this.points) {
      const x = xS(p.gamma_halt);
      const y = yS(p.C);
      ctx.fillStyle = p.C !== 0 ? T.loop : T.halt;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Current gamma marker
    if (this.currentGamma !== null && this.currentGamma !== undefined) {
      const x = xS(this.currentGamma);
      ctx.strokeStyle = T.warn;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, m.t);
      ctx.lineTo(x, m.t + ph);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = T.warn;
      ctx.font = '9px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('\u03b3=' + this.currentGamma.toFixed(2), x + 4, m.t + 2);
    }

    // Axis labels
    ctx.fillStyle = T.dim;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('\u03b3 halt', m.l + pw / 2, m.t + ph + 14);
    ctx.save();
    ctx.translate(m.l - 24, m.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('C (Bott)', 0, 0);
    ctx.restore();
  }
}

// Edge current: per-row sum of |P_ij|^2 across the lattice.
// Loop case shows a "current ring" near the boundary.
// Halt case shows it collapsed to the sink.

import { getT } from './theme.js';
import { setupCanvas, clear } from './canvas_util.js';

export class EdgeCurrentViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.P = null;   // 2D array of {re, im} of size N x N
    this.L = 0;
    this.haltPos = null;
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
    document.addEventListener('themechange', () => this.draw());
  }

  setData(P, L, haltPos) {
    this.P = P;
    this.L = L;
    this.haltPos = haltPos;
    this.draw();
  }

  clear() {
    this.P = null;
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);
  }

  draw() {
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);
    if (!this.P || this.L === 0) {
      ctx.fillStyle = T.dim;
      ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('P_occ not yet computed for current run', w / 2, h / 2);
      return;
    }

    const L = this.L;
    const N = L * L;
    // P_ij with i,j in 0..N-1
    // Sum of |P_ij|^2 over j, grouped by row lattice coordinate y.
    // i = y*L + x. For each row y, sum of |P[(yL+x), j]|^2 averaged over x.
    // Simpler approach: sum |P_ij|^2 over i in row y, average over j in column.
    // We'll show: for each y, mean of (sum_x |P[(yL+x), j]|^2) over j.
    // But a more intuitive view: for each y, the total weight on row y.
    // = sum_{i in row y, j} |P_ij|^2
    const rowWeight = new Float64Array(L);
    for (let y = 0; y < L; y++) {
      let s = 0;
      for (let x = 0; x < L; x++) {
        const i = y * L + x;
        for (let j = 0; j < N; j++) {
          const p = this.P[i][j];
          s += p.re * p.re + p.im * p.im;
        }
      }
      rowWeight[y] = s / L;
    }
    let wMax = 0;
    for (let i = 0; i < L; i++) if (rowWeight[i] > wMax) wMax = rowWeight[i];
    if (wMax <= 0) wMax = 1;

    // Margins
    const m = { l: 36, r: 12, t: 10, b: 26 };
    const pw = w - m.l - m.r;
    const ph = h - m.t - m.b;

    const barW = pw / L;
    const xS = (y) => m.l + y * barW;

    // Axes
    ctx.strokeStyle = T.line;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(m.l, m.t);
    ctx.lineTo(m.l, m.t + ph);
    ctx.lineTo(m.l + pw, m.t + ph);
    ctx.stroke();

    // Bars
    for (let y = 0; y < L; y++) {
      const v = rowWeight[y] / wMax;
      const bx = xS(y) + 1;
      const bw = barW - 2;
      const bh = v * (ph - 4);
      const by = m.t + ph - bh;
      // Color by verdict: top/bottom 2 rows (edge) tinted loop, middle tinted dim, halt row tinted halt
      const onEdge = y < 2 || y >= L - 2;
      const isHalt = this.haltPos && this.haltPos[1] === y;
      if (isHalt) ctx.fillStyle = T.halt;
      else if (onEdge) ctx.fillStyle = T.loop;
      else ctx.fillStyle = T.acc;
      ctx.globalAlpha = 0.4 + 0.6 * v;
      ctx.fillRect(bx, by, bw, bh);
    }
    ctx.globalAlpha = 1.0;

    // Labels
    ctx.fillStyle = T.dim;
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let y = 0; y < L; y += Math.max(1, Math.floor(L / 6))) {
      ctx.fillText('y=' + y, m.l - 4, xS(y) + barW / 2);
    }
    ctx.save();
    ctx.translate(m.l - 28, m.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('row weight (\u2211|P|²)', 0, 0);
    ctx.restore();

    // Y axis label
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('lattice row y', m.l + pw / 2, m.t + ph + 6);

    // Top header
    ctx.fillStyle = T.dim;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`L = ${L},  N = ${N}`, 8, 6);
    if (this.haltPos) {
      ctx.fillText(`EP at (${this.haltPos[0]}, ${this.haltPos[1]})`, 8, 22);
    }
  }
}

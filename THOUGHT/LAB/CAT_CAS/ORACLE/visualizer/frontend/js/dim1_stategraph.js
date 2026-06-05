// State graph viz: nodes = basis states, edges = H[i,j] transitions.
// Animated probability density glow from the wavepacket flow.

import { CSS_VARS } from './theme.js';
import { setupCanvas, clear } from './canvas_util.js';

const T = CSS_VARS;

export class StateGraphViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.data = null;       // {H, labels, halt_mask, transitions, N}
    this.psi = null;        // {re, im, N}
    this.pos = null;        // Array<{x, y}>  layout positions (cached)
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
  }

  setData(data) {
    this.data = data;
    this._layout();
    this.draw();
  }

  setPsi(psi) {
    this.psi = psi;
    this.draw();
  }

  clear() {
    this.data = null;
    this.psi = null;
    this.pos = null;
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h);
  }

  _layout() {
    if (!this.data) return;
    const N = this.data.N;
    // Arrange nodes on a circle. Origin at center of canvas; will be
    // re-centered at draw time using current canvas size.
    const r = 0.4;  // fraction of min(w, h)
    this.pos = [];
    for (let i = 0; i < N; i++) {
      const ang = (2 * Math.PI * i) / N - Math.PI / 2;
      this.pos.push({ ang, rFrac: r });
    }
  }

  draw() {
    if (!this.data) return;
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h);

    const N = this.data.N;
    const cx = w / 2;
    const cy = h / 2;
    const R = Math.min(w, h) * 0.40;
    const nodeR = Math.max(8, Math.min(22, R * 0.13));

    // Compute screen positions
    const screen = this.pos.map(p => ({
      x: cx + R * Math.cos(p.ang),
      y: cy + R * Math.sin(p.ang),
    }));

    // ---- Edges (H[i,j]) ----
    const haltSet = new Set();
    for (let i = 0; i < N; i++) if (this.data.halt_mask[i]) haltSet.add(i);

    // Compute |H[i,j]| for edges. Skip self-loops to keep the picture clean.
    ctx.lineCap = 'round';
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const hre = this.data.H[i][j].re;
        const him = this.data.H[i][j].im;
        const mag = Math.sqrt(hre * hre + him * him);
        if (mag < 1e-6) continue;
        const isHaltEdge = haltSet.has(i) || haltSet.has(j);
        ctx.strokeStyle = isHaltEdge ? T.halt : T.lineStrong;
        ctx.globalAlpha = Math.min(0.9, 0.25 + 0.4 * mag);
        ctx.lineWidth = 0.8 + 2.0 * mag;
        ctx.beginPath();
        ctx.moveTo(screen[i].x, screen[i].y);
        ctx.lineTo(screen[j].x, screen[j].y);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1.0;

    // ---- Probability density glow (under nodes) ----
    if (this.psi) {
      let pmax = 0;
      const probs = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        const re = this.psi.re[i], im = this.psi.im[i];
        probs[i] = re * re + im * im;
        if (probs[i] > pmax) pmax = probs[i];
      }
      if (pmax > 1e-9) {
        for (let i = 0; i < N; i++) {
          const p = probs[i] / pmax;
          if (p < 0.01) continue;
          const r = nodeR * (1.2 + 2.0 * p);
          const grad = ctx.createRadialGradient(
            screen[i].x, screen[i].y, nodeR * 0.5,
            screen[i].x, screen[i].y, r,
          );
          grad.addColorStop(0, `rgba(79, 209, 197, ${0.7 * p})`);
          grad.addColorStop(1, 'rgba(79, 209, 197, 0)');
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(screen[i].x, screen[i].y, r, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }

    // ---- Nodes ----
    for (let i = 0; i < N; i++) {
      const halt = haltSet.has(i);
      ctx.fillStyle = halt ? T.halt : T.panel;
      ctx.strokeStyle = halt ? T.halt : T.acc;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(screen[i].x, screen[i].y, nodeR, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // Inner dot showing probability
      if (this.psi) {
        const re = this.psi.re[i], im = this.psi.im[i];
        const p = re * re + im * im;
        const pmax = this._pmax() || 1;
        const r = Math.max(1, nodeR * 0.7 * Math.sqrt(p / pmax));
        ctx.fillStyle = halt ? '#ffffff' : T.acc;
        ctx.globalAlpha = Math.min(1, 0.4 + 0.6 * p / pmax);
        ctx.beginPath();
        ctx.arc(screen[i].x, screen[i].y, r, 0, 2 * Math.PI);
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }

      // Label
      ctx.fillStyle = halt ? '#ffffff' : T.txt;
      ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(this.data.labels[i], screen[i].x, screen[i].y);
    }

    // ---- Halt legend (top-left) ----
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = T.dim;
    ctx.fillText(`N = ${N}`, 8, 6);
    if (this.data.halt_idx !== null) {
      ctx.fillStyle = T.halt;
      ctx.fillText(`halt = s${this.data.halt_idx}`, 8, 22);
    }
  }

  _pmax() {
    if (!this.psi) return 0;
    let p = 0;
    for (let i = 0; i < this.psi.N; i++) {
      const re = this.psi.re[i], im = this.psi.im[i];
      const x = re * re + im * im;
      if (x > p) p = x;
    }
    return p;
  }
}

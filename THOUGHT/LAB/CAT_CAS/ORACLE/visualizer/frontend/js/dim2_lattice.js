// Lattice heatmap: L x L grid of cells, color = Im(H_ii).
// Red ring marks the EP halt site. Bulk = uniform loss.

import { getT } from './theme.js';
import { setupCanvas, clear } from './canvas_util.js';

export class LatticeViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.data = null;   // {H, L, halt_pos}
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
    document.addEventListener('themechange', () => this.draw());
  }

  setData(data) {
    this.data = data;
    this.draw();
  }

  clear() {
    this.data = null;
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);
  }

  draw() {
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);

    if (!this.data) return;
    const L = this.data.L;
    const N = L * L;
    if (N === 0) return;

    // Margin
    const margin = 24;
    const side = Math.min(w, h) - 2 * margin;
    if (side <= 0) return;
    const cell = side / L;

    const ox = (w - side) / 2;
    const oy = (h - side) / 2;

    // Compute |Im(H_ii)| range for color scaling.
    // Use a FIXED cap of 1.0 (so the natural loss ~0.05 stays blue and
    // a sink of gamma=10 saturates to red). This makes loop vs halt
    // visually distinct regardless of parameter scaling.
    let imMin = 0, imMax = 0;
    for (let i = 0; i < N; i++) {
      const im = this.data.H[i][i].im;
      if (im < imMin) imMin = im;
      if (im > imMax) imMax = im;
    }
    const absSink = Math.abs(imMin); // largest |Im| (= sink depth, if any)
    const IM_CAP = 1.0;              // |Im| >= 1.0 maps to full red

    const haltX = this.data.halt_pos[0];
    const haltY = this.data.halt_pos[1];

    // Draw cells
    for (let y = 0; y < L; y++) {
      for (let x = 0; x < L; x++) {
        const i = y * L + x;
        const im = this.data.H[i][i].im;
        const absIm = Math.abs(im);
        // Linear intensity [0..1] against the fixed cap.
        const u = Math.min(1, absIm / IM_CAP);

        // Color: bg-blue (low |Im|) -> halt-red (high |Im|)
        let r, g, b;
        if (u < 0.05) {
          // nearly zero loss: dark blue (subtle, "vacuum" cell)
          r = 30; g = 38; b = 60;
        } else if (u < 0.15) {
          // small loss (bulk): light blue
          r = 50; g = 70; b = 120;
        } else {
          // significant loss: blend to halt-red
          const v = (u - 0.15) / 0.85;
          r = Math.round(224 * v + 50 * (1 - v));
          g = Math.round(70 * v + 70 * (1 - v));
          b = Math.round(70 * v + 120 * (1 - v));
        }

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        const px = ox + x * cell;
        const py = oy + y * cell;
        ctx.fillRect(px, py, cell, cell);

        // Cell border
        ctx.strokeStyle = T.line;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(px, py, cell, cell);
      }
    }

    // Halt site ring
    if (haltX !== undefined) {
      const px = ox + haltX * cell;
      const py = oy + haltY * cell;
      ctx.strokeStyle = T.halt;
      ctx.lineWidth = 3;
      ctx.strokeRect(px + 1.5, py + 1.5, cell - 3, cell - 3);
      // Label
      ctx.fillStyle = T.halt;
      ctx.font = 'bold 11px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(`EP sink (${haltX},${haltY})`, ox, oy - 16);
    }

    // Legend / corner label
    ctx.fillStyle = T.dim;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`L = ${L},  N = ${L * L}`, 8, 6);
    ctx.fillText('Im(H) \u2208 [' + imMin.toFixed(3) + ', 0]', 8, 22);
  }
}

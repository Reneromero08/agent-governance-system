// Spectrum: eigenvalues of H in the complex plane.
// For 2D we expect Im(eigvals) <= 0 (non-Hermitian, dissipative).

import { getT } from './theme.js';
import { setupCanvas, clear } from './canvas_util.js';

export class Spectrum2DViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.eigvals = null;
    this.E_fermi_im = null;
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
    document.addEventListener('themechange', () => this.draw());
  }

  setData(eigvals, E_fermi_im) {
    this.eigvals = eigvals;
    this.E_fermi_im = E_fermi_im;
    this.draw();
  }

  clear() {
    this.eigvals = null;
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);
  }

  draw() {
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);
    if (!this.eigvals || this.eigvals.length === 0) return;

    // Bounds
    let reMin = 0, reMax = 0, imMin = 0, imMax = 0;
    for (const e of this.eigvals) {
      if (e.re < reMin) reMin = e.re;
      if (e.re > reMax) reMax = e.re;
      if (e.im < imMin) imMin = e.im;
      if (e.im > imMax) imMax = e.im;
    }
    const reR = Math.max(Math.abs(reMin), Math.abs(reMax), 0.5);
    const imR = Math.max(Math.abs(imMin), Math.abs(imMax), 0.5);

    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) * 0.40;

    function toScreen(re, im) {
      return {
        x: cx + (re / reR) * r,
        y: cy - (im / imR) * r,
      };
    }

    // Axes
    ctx.strokeStyle = T.line;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = T.dim;
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('Re(E)  \u2192', w - 6, cy + 2);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Im(E)  \u2191', cx + 4, 4);

    // E_Fermi horizontal line (in Im direction)
    if (this.E_fermi_im !== null && this.E_fermi_im !== undefined) {
      const p = toScreen(0, this.E_fermi_im);
      ctx.strokeStyle = T.acc;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(0, p.y);
      ctx.lineTo(w, p.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = T.acc;
      ctx.font = '9px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText('E_F  Im=' + this.E_fermi_im.toFixed(3), 6, p.y - 1);
    }

    // Eigenvalues
    for (const e of this.eigvals) {
      const p = toScreen(e.re, e.im);
      // Color: stop most "lossy" eigenvalues red
      const norm = Math.min(1, Math.abs(e.im) / imR);
      const rr = Math.round(224 * norm + 90 * (1 - norm));
      const gg = Math.round(90 * norm + 180 * (1 - norm));
      const bb = Math.round(90 * norm + 200 * (1 - norm));
      ctx.fillStyle = `rgb(${rr}, ${gg}, ${bb})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.0, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Bounds label
    ctx.fillStyle = T.dim;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`N = ${this.eigvals.length}`, 8, 6);
    ctx.fillText(`Re \u2208 [${reMin.toFixed(2)}, ${reMax.toFixed(2)}]`, 8, 22);
    ctx.fillText(`Im \u2208 [${imMin.toFixed(2)}, ${imMax.toFixed(2)}]`, 8, 38);
  }
}

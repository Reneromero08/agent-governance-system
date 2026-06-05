// Spectrum viz: eigenvalues in the complex plane, with point-gap reference.

import { CSS_VARS } from './theme.js';
import { setupCanvas, clear, dot } from './canvas_util.js';

const T = CSS_VARS;

export class SpectrumViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.eigvals = null;
    this.eigvecs = null;
    this.haltMask = null;
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
  }

  setData(eigvals, haltMask) {
    this.eigvals = eigvals;
    this.haltMask = haltMask;
    this.draw();
  }

  clear() {
    this.eigvals = null;
    this.haltMask = null;
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h);
  }

  draw() {
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h);

    if (!this.eigvals) {
      ctx.fillStyle = T.dim;
      ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No data', w / 2, h / 2);
      return;
    }

    // Bounds: include the unit circle for context
    let maxAbs = 1.0;
    for (const e of this.eigvals) {
      const m = Math.sqrt(e.re * e.re + e.im * e.im);
      if (m > maxAbs) maxAbs = m;
    }
    const r = maxAbs * 1.15;

    // World -> screen: place origin at center, +Re to right, +Im UP
    const pad = 22;
    const sx = (re) => pad + ((re + r) / (2 * r)) * (w - 2 * pad);
    const sy = (im) => h - pad - ((im + r) / (2 * r)) * (h - 2 * pad);

    // Grid: unit circle (point-gap reference)
    ctx.strokeStyle = T.line;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(sx(0), sy(0), (1 / (2 * r)) * (w - 2 * pad), 0, 2 * Math.PI);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(sx(0), sy(0), (1.0 / (2 * r)) * (w - 2 * pad), 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = T.lineStrong;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, sy(0));
    ctx.lineTo(w, sy(0));
    ctx.moveTo(sx(0), 0);
    ctx.lineTo(sx(0), h);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = T.dim;
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Re \u03bb', w - 30, sy(0) + 4);
    ctx.fillText('Im \u03bb', sx(0) + 4, 4);

    // Unit circle label
    ctx.fillStyle = T.dim;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('|\u03bb|=1', sx(1) - 4, sy(0) - 2);

    // Eigenvalue dots, colored by halt vs non-halt
    for (let i = 0; i < this.eigvals.length; i++) {
      const e = this.eigvals[i];
      const isHalt = this.haltMask && this.haltMask[i];
      const color = isHalt ? T.halt : T.acc;
      const x = sx(e.re);
      const y = sy(e.im);
      // Outer glow
      const grd = ctx.createRadialGradient(x, y, 0, x, y, 12);
      grd.addColorStop(0, color);
      grd.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, 2 * Math.PI);
      ctx.fill();
      ctx.globalAlpha = 1.0;
      // Inner dot
      dot(ctx, x, y, 3.5, color);
    }

    // Caption
    ctx.fillStyle = T.dim;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`|max| = ${r.toFixed(3)}`, 6, 6);
    ctx.fillText(`N = ${this.eigvals.length}`, 6, 20);
  }
}

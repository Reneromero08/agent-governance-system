// det curve viz: the closed curve det(1 - e^{i*phi} H) in the complex plane.
// The origin is the reference point. Winding around origin = W.

import { getT } from './theme.js';
import { setupCanvas, clear, path } from './canvas_util.js';

export class DetCurveViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.curve = null;
    this.absArr = null;
    this.Wint = 0;
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
    document.addEventListener('themechange', () => this.draw());
  }

  setData(curve, absArr, Wint) {
    this.curve = curve;
    this.absArr = absArr;
    this.Wint = Wint;
    this.draw();
  }

  clear() {
    this.curve = null;
    this.absArr = null;
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);
  }

  draw() {
    const T = getT();
    const { ctx, w, h } = setupCanvas(this.canvas);
    clear(ctx, w, h, T.bg);

    if (!this.curve) {
      ctx.fillStyle = T.dim;
      ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No data', w / 2, h / 2);
      return;
    }

    // Bounds: include origin
    let maxAbs = 0.001;
    for (const c of this.curve) {
      const m = Math.sqrt(c.re * c.re + c.im * c.im);
      if (m > maxAbs) maxAbs = m;
    }
    const r = maxAbs * 1.1;

    const pad = 28;
    const sx = (re) => pad + ((re + r) / (2 * r)) * (w - 2 * pad);
    const sy = (im) => h - pad - ((im + r) / (2 * r)) * (h - 2 * pad);

    // Reference circles around origin
    ctx.strokeStyle = T.line;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    for (const frac of [0.25, 0.5, 0.75, 1.0]) {
      ctx.beginPath();
      ctx.arc(sx(0), sy(0), frac * (Math.min(w, h) / 2 - pad), 0, 2 * Math.PI);
      ctx.stroke();
    }
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

    // Origin marker
    ctx.fillStyle = T.dim;
    ctx.beginPath();
    ctx.arc(sx(0), sy(0), 3, 0, 2 * Math.PI);
    ctx.fill();

    // Color the curve based on winding
    const color = this.Wint !== 0 ? T.loop : T.halt;
    const pts = this.curve.map(c => ({ x: sx(c.re), y: sy(c.im) }));
    path(ctx, pts, color, 1.4, true);

    // phi markers along the curve
    const phi = this.curve.length;
    const marks = [0, Math.floor(phi * 0.25), Math.floor(phi * 0.5), Math.floor(phi * 0.75)];
    for (const k of marks) {
      const c = this.curve[k];
      ctx.fillStyle = T.txt;
      ctx.beginPath();
      ctx.arc(sx(c.re), sy(c.im), 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Phi labels
    ctx.fillStyle = T.dim;
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    const c0 = this.curve[0];
    ctx.fillText('\u03c6=0', sx(c0.re) + 4, sy(c0.im) + 4);
    const ch = this.curve[Math.floor(phi / 2)];
    ctx.fillText('\u03c6=\u03c0', sx(ch.re) + 4, sy(ch.im) + 4);

    // Winding indicator
    const banner = this.Wint !== 0 ? `W = ${this.Wint} (LOOPS)` : 'W = 0 (HALTS)';
    const bannerColor = this.Wint !== 0 ? T.loop : T.halt;
    ctx.fillStyle = bannerColor;
    ctx.font = 'bold 12px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(banner, w - 8, 8);

    // |det| mini-strip at bottom (one-row histogram)
    if (this.absArr) {
      const stripY = h - 8;
      const stripH = 4;
      const stripX = pad;
      const stripW = w - 2 * pad;
      const maxAbs2 = Math.max(...this.absArr, 1e-9);
      for (let i = 0; i < this.absArr.length; i++) {
        const t = this.absArr[i] / maxAbs2;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.2 + 0.8 * t;
        ctx.fillRect(stripX + (i / this.absArr.length) * stripW, stripY - stripH * t, Math.max(1, stripW / this.absArr.length), stripH * t + 1);
      }
      ctx.globalAlpha = 1.0;
    }
  }
}

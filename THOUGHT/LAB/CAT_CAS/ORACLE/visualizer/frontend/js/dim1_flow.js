// Wavepacket flow animation: evolve psi(t+dt) = psi(t) - i*dt * H*psi(t)
// (Euler step; sufficient for visualization at small dt.)
// Drives state graph + spectrum (probability density overlay).

import { hToFlatReIm, uniformPsi, evolveStep, normalize, probDensities } from './complex.js';

export class FlowAnimator {
  constructor(stateGraphViz) {
    this.stateGraph = stateGraphViz;
    this.Hre = null;
    this.Him = null;
    this.N = 0;
    this.psiRe = null;
    this.psiIm = null;
    this.dt = 0.05;
    this.stepsPerFrame = 4;
    this.running = false;
    this.frameId = null;
  }

  reset(H) {
    this.cancel();
    const { re, im, N } = hToFlatReIm(H);
    this.Hre = re;
    this.Him = im;
    this.N = N;
    const u = uniformPsi(N);
    this.psiRe = u.re;
    this.psiIm = u.im;
    this.stateGraph.setPsi({ re: this.psiRe, im: this.psiIm, N });
  }

  start() {
    if (this.running) return;
    if (!this.Hre) return;
    this.running = true;
    const tick = () => {
      if (!this.running) return;
      for (let s = 0; s < this.stepsPerFrame; s++) {
        evolveStep(this.Hre, this.Him, this.psiRe, this.psiIm, this.dt, this.N);
      }
      // Periodic re-normalization to keep |psi| on the sphere.
      normalize(this.psiRe, this.psiIm);
      this.stateGraph.setPsi({ re: this.psiRe, im: this.psiIm, N: this.N });
      this.frameId = requestAnimationFrame(tick);
    };
    this.frameId = requestAnimationFrame(tick);
  }

  cancel() {
    this.running = false;
    if (this.frameId) cancelAnimationFrame(this.frameId);
    this.frameId = null;
  }

  isRunning() { return this.running; }
}

// 1D mechanism view controller.

import { getDim1Machines, runDim1 } from './api.js';
import { StateGraphViz } from './dim1_stategraph.js';
import { SpectrumViz } from './dim1_spectrum.js';
import { DetCurveViz } from './dim1_detcurve.js';
import { FlowAnimator } from './dim1_flow.js';

const MECHANISM_TEXT = {
  halt_direct:
    "Direct sink: every transition flows into a single halt site. H's imaginary " +
    "eigenvalues are far below the real axis, so the wavepacket drains there " +
    "exponentially. The det curve does NOT wind the origin: W = 0 \u2192 HALTS.",
  halt_chain:
    "Chain of self-loops terminating in a halt site. Same mechanism as halt_direct: " +
    "strong drain at the sink keeps all eigenvalues off the unit circle. W = 0 \u2192 HALTS.",
  loop_2cycle:
    "Two states bounce forever (s0\u2194s1). The Hamiltonian is closed: no sink. " +
    "Eigenvalues sit on the unit circle, the det curve winds once around the " +
    "origin. W = +1 \u2192 LOOPS.",
  loop_3cycle:
    "Three states cycle forever (s0\u2192s1\u2192s2\u2192s0). Closed loop, no sink. " +
    "Eigenvalues on the unit circle, det curve winds once. W = +1 \u2192 LOOPS.",
};

export class Dim1Controller {
  constructor() {
    this.machines = null;
    this.runResult = null;
    this.stateViz = new StateGraphViz(document.getElementById('dim1-canvas-state'));
    this.spectrumViz = new SpectrumViz(document.getElementById('dim1-canvas-spectrum'));
    this.detViz = new DetCurveViz(document.getElementById('dim1-canvas-det'));
    this.flow = new FlowAnimator(this.stateViz);

    this.machineSel = document.getElementById('dim1-machine');
    this.gammaIn = document.getElementById('dim1-gamma');
    this.gammaVal = document.getElementById('dim1-gamma-val');
    this.lossIn = document.getElementById('dim1-loss');
    this.lossVal = document.getElementById('dim1-loss-val');
    this.nphiIn = document.getElementById('dim1-nphi');
    this.runBtn = document.getElementById('dim1-run');
    this.animBtn = document.getElementById('dim1-animate');
    this.resetBtn = document.getElementById('dim1-reset');
    this.verdictBanner = document.getElementById('dim1-verdict-banner');
    this.kpiW = document.getElementById('dim1-kpi-w');
    this.kpiKappa = document.getElementById('dim1-kpi-kappa');
    this.kpiRho = document.getElementById('dim1-kpi-rho');
    this.kpiHalt = document.getElementById('dim1-kpi-halt');
    this.specEl = document.getElementById('dim1-spec');
    this.mechEl = document.getElementById('dim1-mechanism-text');
    this.stateSub = document.getElementById('dim1-state-sub');
    this.specSub = document.getElementById('dim1-spec-sub');
    this.detSub = document.getElementById('dim1-det-sub');

    this._wire();
  }

  _wire() {
    this.gammaIn.addEventListener('input', () => {
      this.gammaVal.textContent = parseFloat(this.gammaIn.value).toFixed(2);
    });
    this.lossIn.addEventListener('input', () => {
      this.lossVal.textContent = parseFloat(this.lossIn.value).toFixed(2);
    });
    this.runBtn.addEventListener('click', () => this.run());
    this.animBtn.addEventListener('click', () => this.toggleAnimate());
    this.resetBtn.addEventListener('click', () => this.resetFlow());
    this.machineSel.addEventListener('change', () => this.run());
  }

  async init() {
    try {
      const data = await getDim1Machines();
      this.machines = data.machines;
      this._populateMachineSelect();
      this._populateParamDefaults(data.params);
    } catch (e) {
      this._error('cannot load machines: ' + e.message);
      return;
    }
    // Allow URL params to override defaults: ?machine=halt_direct&gamma=0.3&loss=0.05&nphi=200
    const url = new URL(window.location.href);
    const m = url.searchParams.get('machine');
    if (m && this.machines[m]) this.machineSel.value = m;
    const g = url.searchParams.get('gamma');
    if (g !== null) { this.gammaIn.value = g; this.gammaVal.textContent = parseFloat(g).toFixed(2); }
    const l = url.searchParams.get('loss');
    if (l !== null) { this.lossIn.value = l; this.lossVal.textContent = parseFloat(l).toFixed(2); }
    const n = url.searchParams.get('nphi');
    if (n !== null) this.nphiIn.value = n;
    await this.run();
  }

  _populateMachineSelect() {
    const names = Object.keys(this.machines);
    this.machineSel.innerHTML = '';
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = `${name}  (${this.machines[name].expected})`;
      this.machineSel.appendChild(opt);
    }
    this.machineSel.value = 'loop_2cycle';
  }

  _populateParamDefaults(params) {
    // For now we just trust the HTML defaults; future: sync from server.
  }

  _setVerdict(v) {
    this.verdictBanner.textContent = v;
    this.verdictBanner.className = 'verdict-banner ' + (
      v === 'HALTS' ? 'halt' :
      v === 'LOOPS' ? 'loop' :
      v === 'ERROR' ? 'warn' : 'unknown'
    );
  }

  _setMachineSpec(result) {
    const lines = [];
    lines.push(`machine   : ${result.machine}`);
    lines.push(`N         : ${result.N}   num_states: ${result.num_states}`);
    lines.push(`halt_idx  : ${result.halt_idx === null ? 'none' : result.halt_idx}`);
    lines.push(`gamma     : ${result.gamma}   loss_rate: ${result.loss_rate}   halt_mult: ${result.halt_mult}`);
    lines.push(`twist     : ${JSON.stringify(result.twist_indices)}`);
    lines.push(`labels    : ${JSON.stringify(result.labels)}`);
    lines.push(`transitns : ${JSON.stringify(result.transitions)}`);
    this.specEl.textContent = lines.join('\n');
  }

  _setMechanism(machine) {
    const txt = MECHANISM_TEXT[machine] || 'Unknown machine.';
    this.mechEl.textContent = txt;
    this.mechEl.classList.remove('muted');
  }

  async run() {
    const params = {
      machine: this.machineSel.value,
      gamma: parseFloat(this.gammaIn.value),
      loss_rate: parseFloat(this.lossIn.value),
      halt_mult: 10.0,
      n_phi: parseInt(this.nphiIn.value, 10),
    };
    this._setVerdict('RUNNING...');
    this.runBtn.disabled = true;
    this.animBtn.disabled = true;
    this.resetBtn.disabled = true;
    this.flow.cancel();
    try {
      const r = await runDim1(params);
      this.runResult = r;
      this._renderResult(r);
    } catch (e) {
      this._setVerdict('ERROR');
      this.specEl.textContent = 'Error: ' + e.message;
      console.error(e);
    } finally {
      this.runBtn.disabled = false;
      this.animBtn.disabled = false;
      this.resetBtn.disabled = false;
    }
  }

  _renderResult(r) {
    // Verdict
    this._setVerdict(r.verdict);
    // KPIs
    this.kpiW.textContent = String(r.winding.Wint);
    this.kpiW.className = 'kpi-value ' + (r.winding.Wint === 0 ? 'halt' : 'loop');
    this.kpiKappa.textContent = r.spectrum.kappa_V.toFixed(3);
    this.kpiRho.textContent = r.spectrum.spectral_radius.toFixed(3);
    // Halt sink strength: diagonal |Im(H_ii)| for halt sites
    let haltSink = 0;
    for (let i = 0; i < r.N; i++) {
      if (r.halt_mask[i]) haltSink = Math.max(haltSink, Math.abs(r.H[i][i].im));
    }
    this.kpiHalt.textContent = haltSink.toFixed(3);
    this.kpiHalt.className = 'kpi-value ' + (haltSink > 0.1 ? 'loop' : 'halt');
    // Spec
    this._setMachineSpec(r);
    // Mechanism
    this._setMechanism(r.machine);
    // Vizes
    this.stateViz.setData(r);
    this.spectrumViz.setData(r.spectrum.eigvals, r.halt_mask);
    this.detViz.setData(r.winding.det_curve, r.winding.det_abs, r.winding.Wint);
    // Sub-labels
    this.specSub.innerHTML = `N = ${r.N}`;
    this.detSub.innerHTML = `\u03c6 \u2208 [0, 2\u03c0]`;
    this.stateSub.textContent = 'uniform';
    // Reset flow to uniform
    this.flow.reset(r.H);
    // Enable flow controls
    this.animBtn.disabled = false;
    this.resetBtn.disabled = false;
  }

  toggleAnimate() {
    if (this.flow.isRunning()) {
      this.flow.cancel();
      this.animBtn.textContent = 'Animate flow';
      this.stateSub.textContent = 'paused';
    } else {
      this.flow.start();
      this.animBtn.textContent = 'Pause';
      this.stateSub.textContent = 'flowing';
    }
  }

  resetFlow() {
    this.flow.cancel();
    if (this.runResult) {
      this.flow.reset(this.runResult.H);
      this.stateSub.textContent = 'uniform';
    }
    this.animBtn.textContent = 'Animate flow';
  }

  _error(msg) {
    this._setVerdict('ERROR');
    this.specEl.textContent = msg;
  }
}

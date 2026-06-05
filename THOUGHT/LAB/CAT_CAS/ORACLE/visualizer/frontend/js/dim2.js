// 2D Chern mechanism view controller.

import {
  getDim2Machines, runDim2, buildDim2, gammaSweepDim2,
} from './api.js';
import { LatticeViz } from './dim2_lattice.js';
import { Spectrum2DViz } from './dim2_spectrum.js';
import { BottCurveViz } from './dim2_bottcurve.js';
import { EdgeCurrentViz } from './dim2_edge.js';
import { debounce } from './debounce.js';
import { beginRequest, endRequest, failRequest, note } from './status.js';
import { copyShareUrl, downloadJson, canvasesToPng, savePng } from './export.js';

const MECHANISM_TEXT = {
  loop_default:
    "L=8 lattice with no EP sink. Complex NNN hopping (t2 * e^{i*pi/4}) breaks " +
    "time-reversal symmetry and produces a Chern insulator. The catalytic contour " +
    "projector isolates the non-decaying edge modes, the Bott index is C = +1, and " +
    "the chiral edge is protected from any small local perturbation -> LOOPS.",
  halt_default:
    "Same L=8 lattice, but a massive EP sink (Im(H_ii) = -loss - 10) is added at the " +
    "center site. The sink destroys the chiral edge by collapsing the spectral gap " +
    "around the Fermi level. The Bott index rounds to C = 0, the edge current " +
    "disappears, and any probe halts -> HALTS.",
  uniform_annihilation:
    "L=8 with a uniform loss gamma=2 on EVERY site (Exp 39 discovery): the " +
    "topology melts away even without a localized sink. The Bott index C = 0 " +
    "because the bulk projector loses all edge structure -> HALTS.",
  l4_fragility:
    "L=4 is too small to host a robust chiral edge. The spectral gap collapses " +
    "from finite-size effects: C rounds to 0 even at gamma_halt=0. Working " +
    "default is L=8 -> HALTS due to fragility, not physics failure.",
};

export class Dim2Controller {
  constructor() {
    this.machines = null;
    this.runResult = null;
    this._inflight = null;
    this.latticeViz = new LatticeViz(document.getElementById('dim2-canvas-lattice'));
    this.spectrumViz = new Spectrum2DViz(document.getElementById('dim2-canvas-spectrum'));
    this.bottViz = new BottCurveViz(document.getElementById('dim2-canvas-bott'));
    this.edgeViz = new EdgeCurrentViz(document.getElementById('dim2-canvas-edge'));

    this.machineSel = document.getElementById('dim2-machine');
    this.LIn = document.getElementById('dim2-L');
    this.LVal = document.getElementById('dim2-L-val');
    this.gammaIn = document.getElementById('dim2-gamma');
    this.gammaVal = document.getElementById('dim2-gamma-val');
    this.t1In = document.getElementById('dim2-t1');
    this.t1Val = document.getElementById('dim2-t1-val');
    this.t2In = document.getElementById('dim2-t2');
    this.t2Val = document.getElementById('dim2-t2-val');
    this.phiIn = document.getElementById('dim2-phi');
    this.phiVal = document.getElementById('dim2-phi-val');
    this.lossIn = document.getElementById('dim2-loss');
    this.lossVal = document.getElementById('dim2-loss-val');
    this.runBtn = document.getElementById('dim2-run');
    this.sweepBtn = document.getElementById('dim2-sweep');
    this.copyUrlBtn = document.getElementById('dim2-copy-url');
    this.dlJsonBtn = document.getElementById('dim2-download-json');
    this.dlPngBtn = document.getElementById('dim2-download-png');
    this.verdictBanner = document.getElementById('dim2-verdict-banner');
    this.kpiC = document.getElementById('dim2-kpi-c');
    this.kpiEf = document.getElementById('dim2-kpi-ef');
    this.kpiRho = document.getElementById('dim2-kpi-rho');
    this.kpiSink = document.getElementById('dim2-kpi-sink');
    this.specEl = document.getElementById('dim2-spec');
    this.mechEl = document.getElementById('dim2-mechanism-text');
    this.latticeSub = document.getElementById('dim2-lattice-sub');
    this.specSub = document.getElementById('dim2-spec-sub');
    this.bottSub = document.getElementById('dim2-bott-sub');
    this.edgeSub = document.getElementById('dim2-edge-sub');

    this._wire();
  }

  _wire() {
    const onParam = debounce(() => this.run(), 280);
    const onL = debounce(() => this.run(), 380);

    this.gammaIn.addEventListener('input', () => {
      this.gammaVal.textContent = parseFloat(this.gammaIn.value).toFixed(2);
      onParam();
    });
    this.LIn.addEventListener('input', () => {
      this.LVal.textContent = String(parseInt(this.LIn.value, 10));
      onL();
    });
    this.t1In.addEventListener('input', () => {
      this.t1Val.textContent = parseFloat(this.t1In.value).toFixed(2);
      onParam();
    });
    this.t2In.addEventListener('input', () => {
      this.t2Val.textContent = parseFloat(this.t2In.value).toFixed(2);
      onParam();
    });
    this.phiIn.addEventListener('input', () => {
      this.phiVal.textContent = parseFloat(this.phiIn.value).toFixed(3);
      onParam();
    });
    this.lossIn.addEventListener('input', () => {
      this.lossVal.textContent = parseFloat(this.lossIn.value).toFixed(3);
      onParam();
    });
    this.runBtn.addEventListener('click', () => this.run());
    this.sweepBtn.addEventListener('click', () => this.sweep());
    this.machineSel.addEventListener('change', () => this._applyMachine());
    this.copyUrlBtn.addEventListener('click', () => this._copyUrl());
    this.dlJsonBtn.addEventListener('click', () => this._downloadJson());
    this.dlPngBtn.addEventListener('click', () => this._downloadPng());
  }

  _currentState() {
    return {
      machine: this.machineSel.value,
      L: parseInt(this.LIn.value, 10),
      gamma: parseFloat(this.gammaIn.value),
      t1: parseFloat(this.t1In.value),
      t2: parseFloat(this.t2In.value),
      phi: parseFloat(this.phiIn.value),
      loss: parseFloat(this.lossIn.value),
    };
  }

  async _copyUrl() {
    await copyShareUrl(this._currentState());
  }

  _downloadJson() {
    if (!this.runResult) { note('Run a preset first'); return; }
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const m = this.runResult;
    downloadJson(`oracle-2d-L${m.L}-g${m.gamma_halt}-${stamp}`, this.runResult);
  }

  _downloadPng() {
    const canvases = [
      'dim2-canvas-lattice',
      'dim2-canvas-spectrum',
      'dim2-canvas-bott',
      'dim2-canvas-edge',
    ].map(id => document.getElementById(id));
    const s = this._currentState();
    const label = `2D ${s.machine}  L=${s.L}  gamma=${s.gamma}  t1=${s.t1}  t2=${s.t2}`;
    const url = canvasesToPng(canvases, label);
    if (!url) { note('Nothing to save'); return; }
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    savePng(`oracle-2d-${s.machine}-${stamp}.png`, url);
  }

  async init() {
    try {
      const data = await getDim2Machines();
      this.machines = data.machines;
      this._populateMachineSelect();
    } catch (e) {
      this._error('cannot load machines: ' + e.message);
      return;
    }
    // URL deep-link
    const url = new URL(window.location.href);
    const m = url.searchParams.get('machine');
    if (m && this.machines[m]) this.machineSel.value = m;
    this._applyMachine();
    const L = url.searchParams.get('L');
    if (L !== null) { this.LIn.value = L; this.LVal.textContent = L; }
    const g = url.searchParams.get('gamma');
    if (g !== null) { this.gammaIn.value = g; this.gammaVal.textContent = parseFloat(g).toFixed(2); }
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
    this.machineSel.value = 'loop_default';
  }

  _applyMachine() {
    const m = this.machines[this.machineSel.value];
    if (!m) return;
    this.LIn.value = m.L;
    this.LVal.textContent = String(m.L);
    this.gammaIn.value = m.gamma_halt;
    this.gammaVal.textContent = m.gamma_halt.toFixed(2);
    this.t1In.value = m.t1;
    this.t1Val.textContent = m.t1.toFixed(2);
    this.t2In.value = m.t2;
    this.t2Val.textContent = m.t2.toFixed(2);
    this.phiIn.value = m.phi;
    this.phiVal.textContent = m.phi.toFixed(3);
    this.lossIn.value = m.loss;
    this.lossVal.textContent = m.loss.toFixed(3);
    this.run();
  }

  _setVerdict(v) {
    this.verdictBanner.textContent = v;
    this.verdictBanner.className = 'verdict-banner ' + (
      v === 'HALTS' ? 'halt' :
      v === 'LOOPS' ? 'loop' :
      v === 'ERROR' ? 'warn' : 'unknown'
    );
  }

  _setSpec(result) {
    const lines = [];
    lines.push(`L        : ${result.L}   N = ${result.N}`);
    lines.push(`t1, t2   : ${result.t1}, ${result.t2}`);
    lines.push(`phi      : ${result.phi.toFixed(4)}   loss: ${result.loss}`);
    lines.push(`gamma_h  : ${result.gamma_halt}`);
    lines.push(`halt_pos : (${result.halt_pos[0]}, ${result.halt_pos[1]})  site=${result.halt_site}`);
    lines.push(`E_fermi  : (0, ${result.fermi.E_fermi_im.toFixed(4)})  gap=${result.fermi.gap_width.toFixed(4)}`);
    lines.push(`spectral : rho = ${result.spectrum.spectral_radius.toFixed(3)}`);
    this.specEl.textContent = lines.join('\n');
  }

  _setMechanism(machine) {
    const txt = MECHANISM_TEXT[machine] || 'Custom configuration. See lattice / spectrum / Bott curve.';
    this.mechEl.textContent = txt;
    this.mechEl.classList.remove('muted');
  }

  async run() {
    if (this._inflight) {
      this._inflight.abort();
    }
    const controller = new AbortController();
    this._inflight = controller;

    const params = {
      L: parseInt(this.LIn.value, 10),
      t1: parseFloat(this.t1In.value),
      t2: parseFloat(this.t2In.value),
      phi: parseFloat(this.phiIn.value),
      loss: parseFloat(this.lossIn.value),
      gamma_halt: parseFloat(this.gammaIn.value),
    };
    this._setVerdict('RUNNING...');
    this.runBtn.classList.add('busy');
    this.runBtn.disabled = true;
    this.sweepBtn.disabled = true;
    beginRequest(`/api/dim2/run L=${params.L}`);

    try {
      const r = await runDim2(params, { signal: controller.signal });
      if (controller.signal.aborted) return;
      this.runResult = r;
      this._renderResult(r);
      note(`2D L=${r.L} gamma=${r.gamma_halt}: ${r.verdict}, C=${r.bott.C}`);
    } catch (e) {
      if (e.name === 'AbortError') { note('cancelled'); return; }
      this._setVerdict('ERROR');
      this.specEl.textContent = 'Error: ' + e.message;
      failRequest('run', e.message);
      console.error(e);
    } finally {
      if (this._inflight === controller) this._inflight = null;
      this.runBtn.classList.remove('busy');
      this.runBtn.disabled = false;
      this.sweepBtn.disabled = false;
    }
  }

  _renderResult(r) {
    this._setVerdict(r.verdict);
    this.kpiC.textContent = String(r.bott.C);
    this.kpiC.className = 'kpi-value ' + (r.bott.C !== 0 ? 'loop' : 'halt');
    this.kpiEf.textContent = r.fermi.E_fermi_im.toFixed(3);
    this.kpiRho.textContent = r.spectrum.spectral_radius.toFixed(3);
    const sinkIm = Math.abs(r.H[r.halt_site][r.halt_site].im);
    this.kpiSink.textContent = sinkIm.toFixed(3);
    this.kpiSink.className = 'kpi-value ' + (sinkIm > 0.1 ? 'loop' : 'halt');
    this._setSpec(r);
    this._setMechanism(this.machineSel.value);
    this.latticeViz.setData({ H: r.H, L: r.L, halt_pos: r.halt_pos });
    this.spectrumViz.setData(r.spectrum.eigvals, r.fermi.E_fermi_im);
    this.latticeSub.textContent = `L = ${r.L}`;
    this.specSub.textContent = `N = ${r.N}`;
    this.bottSub.textContent = `Bott C = ${r.bott.C}`;
    this.edgeSub.textContent = `L = ${r.L},  N = ${r.N}`;
    // P projector visualization
    if (r.projector && r.projector.P) {
      this.edgeViz.setData(r.projector.P, r.L, r.halt_pos);
    } else {
      this.edgeViz.clear();
    }
  }

  async sweep() {
    if (this._inflight) {
      this._inflight.abort();
    }
    const controller = new AbortController();
    this._inflight = controller;
    this.sweepBtn.classList.add('busy');
    this.sweepBtn.disabled = true;
    this.runBtn.disabled = true;
    beginRequest('/api/dim2/gamma_sweep');

    const params = {
      L: parseInt(this.LIn.value, 10),
      t1: parseFloat(this.t1In.value),
      t2: parseFloat(this.t2In.value),
      phi: parseFloat(this.phiIn.value),
      loss: parseFloat(this.lossIn.value),
      gammas: '0,0.5,1,2,3,5,8,10,15,20',
    };
    try {
      const r = await gammaSweepDim2(params, { signal: controller.signal });
      if (controller.signal.aborted) return;
      this.bottViz.setData(r.points, parseFloat(this.gammaIn.value));
      this.bottSub.textContent = `Bott C = ${this.runResult ? this.runResult.bott.C : '?'}, ${r.points.length} pts`;
      note(`2D Bott sweep: ${r.points.length} gammas`);
    } catch (e) {
      if (e.name === 'AbortError') { note('sweep cancelled'); return; }
      this.bottSub.textContent = 'sweep failed';
      failRequest('sweep', e.message);
      console.error(e);
    } finally {
      if (this._inflight === controller) this._inflight = null;
      this.sweepBtn.classList.remove('busy');
      this.sweepBtn.disabled = false;
      this.runBtn.disabled = false;
    }
  }

  _error(msg) {
    this._setVerdict('ERROR');
    this.specEl.textContent = msg;
  }
}

// REST client for the FastAPI backend.
//
// All functions return parsed JSON. Throw on non-2xx.

const BASE = '';

/** Health check. Returns {status, phase, python, cat_cas_path}. */
export async function getHealth() {
  const r = await fetch(BASE + '/api/health');
  if (!r.ok) throw new Error('health check failed: ' + r.status);
  return r.json();
}

/** 1D: list available machines and parameter ranges. */
export async function getDim1Machines() {
  const r = await fetch(BASE + '/api/dim1/machines');
  if (!r.ok) throw new Error('dim1 machines failed: ' + r.status);
  return r.json();
}

/** 1D: full oracle run. */
export async function runDim1(params, opts = {}) {
  const qs = new URLSearchParams(params).toString();
  const r = await fetch(BASE + '/api/dim1/run?' + qs, opts);
  if (!r.ok) throw new Error('dim1 run failed: ' + r.status);
  return r.json();
}

/** 1D: build H only (cheaper than /run, for flow animation). */
export async function buildDim1(params, opts = {}) {
  const qs = new URLSearchParams(params).toString();
  const r = await fetch(BASE + '/api/dim1/build?' + qs, opts);
  if (!r.ok) throw new Error('dim1 build failed: ' + r.status);
  return r.json();
}

/** 2D: list available machine presets and param ranges. */
export async function getDim2Machines() {
  const r = await fetch(BASE + '/api/dim2/machines');
  if (!r.ok) throw new Error('dim2 machines failed: ' + r.status);
  return r.json();
}

/** 2D: full oracle run (H, spectrum, fermi, projector, Bott). */
export async function runDim2(params, opts = {}) {
  const qs = new URLSearchParams(params).toString();
  const r = await fetch(BASE + '/api/dim2/run?' + qs, opts);
  if (!r.ok) throw new Error('dim2 run failed: ' + r.status);
  return r.json();
}

/** 2D: build H only (cheaper than /run). */
export async function buildDim2(params, opts = {}) {
  const qs = new URLSearchParams(params).toString();
  const r = await fetch(BASE + '/api/dim2/build?' + qs, opts);
  if (!r.ok) throw new Error('dim2 build failed: ' + r.status);
  return r.json();
}

/** 2D: Bott index C as a function of gamma_halt. */
export async function gammaSweepDim2(params, opts = {}) {
  const qs = new URLSearchParams(params).toString();
  const r = await fetch(BASE + '/api/dim2/gamma_sweep?' + qs, opts);
  if (!r.ok) throw new Error('dim2 gamma_sweep failed: ' + r.status);
  return r.json();
}

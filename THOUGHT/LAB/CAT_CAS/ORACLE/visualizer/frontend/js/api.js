// REST client for the FastAPI backend.
//
// All functions return parsed JSON. Throw on non-2xx.
//
// Phase 0: just /api/health.
// Phase 1: /api/dim{N}/run endpoints added as the engine comes online.

const BASE = '';

/** Health check. Returns {status, phase, python, cat_cas_path}. */
export async function getHealth() {
  const r = await fetch(BASE + '/api/health');
  if (!r.ok) throw new Error('health check failed: ' + r.status);
  return r.json();
}

/** 1D run (Phase 1). Stub returns 501 until engine lands. */
export async function runDim1(params) {
  const qs = new URLSearchParams(params).toString();
  const r = await fetch(BASE + '/api/dim1/run?' + qs);
  if (!r.ok) throw new Error('dim1 run failed: ' + r.status);
  return r.json();
}

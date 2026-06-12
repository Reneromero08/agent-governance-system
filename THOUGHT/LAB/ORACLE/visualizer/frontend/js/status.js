// Status footer controller: tracks request state, last timing, and exposes
// a cancel handle.

const statusPill = document.getElementById('footer-status');
const statusMsg = document.getElementById('footer-msg');
const cancelBtn = document.getElementById('footer-cancel');

let inflight = null;

function set(state, msg) {
  if (statusPill) {
    statusPill.className = 'pill-mini ' + state;
    statusPill.textContent = state;
  }
  if (statusMsg && msg) statusMsg.textContent = msg;
  if (cancelBtn) cancelBtn.disabled = !inflight;
}

export function beginRequest(label) {
  const t0 = performance.now();
  set('busy', label + ' ...');
  inflight = { label, t0 };
}

export function endRequest(label, bytes) {
  if (!inflight) return;
  const dt = performance.now() - inflight.t0;
  const sizeKb = bytes ? (bytes / 1024).toFixed(1) + ' KB' : '';
  set('ok', `${label}  ${dt.toFixed(0)} ms${sizeKb ? '  ' + sizeKb : ''}`);
  inflight = null;
  if (cancelBtn) cancelBtn.disabled = true;
}

export function failRequest(label, err) {
  if (inflight) inflight = null;
  set('err', `${label}  ${err}`);
  if (cancelBtn) cancelBtn.disabled = true;
}

export function note(msg) {
  set('idle', msg);
}

if (cancelBtn) {
  cancelBtn.addEventListener('click', () => {
    if (inflight && inflight.controller) inflight.controller.abort();
    note('cancelled');
  });
}

// Helper: wrap a fetch in beginRequest/endRequest lifecycle.
export async function trackedFetch(url, label) {
  const controller = new AbortController();
  beginRequest(label);
  inflight.controller = controller;
  try {
    const r = await fetch(url, { signal: controller.signal });
    const text = await r.text();
    endRequest(label, text.length);
    return text;
  } catch (e) {
    failRequest(label, e.message);
    throw e;
  }
}

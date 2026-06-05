// Minimal complex arithmetic and matrix helpers for small N (<= 10).
//
// H is passed over the wire as Array<Array<{re, im}>>. We convert it once
// into parallel Float32Arrays (re[], im[]) for cache-friendly work.

export function hToFlatReIm(H) {
  // H is row-major. Returns {re: Float32Array, im: Float32Array, N}.
  const N = H.length;
  const re = new Float32Array(N * N);
  const im = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    const row = H[i];
    for (let j = 0; j < N; j++) {
      const c = row[j];
      re[i * N + j] = c.re;
      im[i * N + j] = c.im;
    }
  }
  return { re, im, N };
}

// psi = (1/sqrt(N)) * (1 + 0i) per basis state -- uniform superposition.
export function uniformPsi(N) {
  const s = 1.0 / Math.sqrt(N);
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for (let i = 0; i < N; i++) re[i] = s;
  return { re, im };
}

// psi[i] = delta_{i,k} -- a single-site state.
export function deltaPsi(N, k) {
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  re[k] = 1.0;
  return { re, im };
}

// out = a * b   (complex scalar mul of two N-vectors)
function cmulVec(aRe, aIm, bRe, bIm, outRe, outIm) {
  const N = aRe.length;
  for (let i = 0; i < N; i++) {
    const ar = aRe[i], ai = aIm[i];
    const br = bRe[i], bi = bIm[i];
    outRe[i] = ar * br - ai * bi;
    outIm[i] = ar * bi + ai * br;
  }
}

// y = alpha * (H * x) + beta * y, where alpha and beta are complex scalars.
// H is the flat row-major complex matrix (re, im) of size NxN.
// x, y are N-vectors of complex.
export function zgemvStrided(Hre, Him, xRe, xIm, yRe, yIm, alphaRe, alphaIm, betaRe, betaIm, N) {
  // y := alpha * (H * x) + beta * y
  // Compute tmp = H * x (complex) into temp.
  const tmpRe = new Float32Array(N);
  const tmpIm = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let sRe = 0, sIm = 0;
    for (let k = 0; k < N; k++) {
      const hr = Hre[i * N + k], hi = Him[i * N + k];
      const xr = xRe[k], xi = xIm[k];
      sRe += hr * xr - hi * xi;
      sIm += hr * xi + hi * xr;
    }
    tmpRe[i] = sRe;
    tmpIm[i] = sIm;
  }
  // y = alpha * tmp + beta * y
  for (let i = 0; i < N; i++) {
    const tr = tmpRe[i], ti = tmpIm[i];
    const ar = alphaRe, ai = alphaIm;
    const br = betaRe, bi = betaIm;
    const t1r = ar * tr - ai * ti;
    const t1i = ar * ti + ai * tr;
    const t2r = br * yRe[i] - bi * yIm[i];
    const t2i = br * yIm[i] + bi * yRe[i];
    yRe[i] = t1r + t2r;
    yIm[i] = t1i + t2i;
  }
}

// Single Euler step: psi(t+dt) = psi(t) - i * dt * (H * psi(t))
// i.e., (alpha=1, beta=1) for psi += -i*dt * (H * psi).
export function evolveStep(Hre, Him, psiRe, psiIm, dt, N) {
  // alpha = (-dt) for real part, alpha_im = 0  -> -dt * (H * psi)
  // But we want -i*dt*(H*psi) = (0 - i*dt) * (H*psi)
  // So alpha = (-dt), 0 in re/im? No: -i*dt = (0) + (-dt)*i  => alphaRe=0, alphaIm=-dt
  zgemvStrided(Hre, Him, psiRe, psiIm, psiRe, psiIm, 0.0, -dt, 1.0, 0.0, N);
}

// |psi|^2 per site
export function probDensities(psiRe, psiIm) {
  const N = psiRe.length;
  const p = new Float32Array(N);
  for (let i = 0; i < N; i++) p[i] = psiRe[i] * psiRe[i] + psiIm[i] * psiIm[i];
  return p;
}

// norm^2 = sum |psi_i|^2
export function normSq(psiRe, psiIm) {
  let s = 0;
  for (let i = 0; i < psiRe.length; i++) s += psiRe[i] * psiRe[i] + psiIm[i] * psiIm[i];
  return s;
}

// normalize: divide by sqrt(norm^2)
export function normalize(psiRe, psiIm) {
  const n = Math.sqrt(normSq(psiRe, psiIm));
  if (n > 0) {
    for (let i = 0; i < psiRe.length; i++) {
      psiRe[i] /= n;
      psiIm[i] /= n;
    }
  }
}

// Sample parse: a H matrix from the server.
export function parseH(H) {
  return hToFlatReIm(H);
}

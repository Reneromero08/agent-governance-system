"""Test phase-only EigenBuddy: use complex phase instead of raw real+imag.

The catalytic hidden state is XOR'd f32 bytes. The magnitude is dominated by
random substrate (~3e38). But the PHASE of each complex dimension (atan2(imag,real))
encodes the signal carried through the multi-scale Feistel (Q57: gapped phase
topology). Phase is the implicate order; amplitude is explicate (Q34).
"""
import sys, os, time, numpy as np, torch
from pathlib import Path

CAT_CAS = Path(__file__).resolve().parent
EIGEN = next(p for p in Path(__file__).resolve().parents if p.name == "CAT_CAS").parent / 'EIGEN_BUDDY'
sys.path.insert(0, str(EIGEN))
from eigen_buddy_tokenizer import (
    EigenBuddyTokenizer, train_eigen_buddy, evaluate_platonic_convergence,
    load_catalytic_data, STABLE_32
)

# Load gold data
data_path = CAT_CAS / 'gold_training_data' / 'gold_pairs_quick.pt'
data = torch.load(data_path, weights_only=True)
states_real = data['states_real']  # (27, 896)
states_imag = data['states_imag']  # (27, 896)
targets = data['targets']          # (27,)

print(f'Loaded {len(targets)} pairs, {len(set(targets.tolist()))} unique gold tokens')

# ======= Approach 1: Phase-only =======
print('\n=== PHASE-ONLY INPUT ===')
phases = np.arctan2(states_imag.numpy(), states_real.numpy())  # (27, 896) in [-pi, pi]

# Also try: difference between consecutive tokens (substrate cancels)
# phase_diff[i] = phases[i+1] - phases[i], unwrapped
phase_diff = np.diff(phases, axis=0)  # (26, 896)
# Unwrap around [-pi, pi]
phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

# Per-sample unit normalization for SVD
from numpy.linalg import norm
phases_norm = phases / (norm(phases, axis=1, keepdims=True) + 1e-15)

# Complex SVD on phase-only
Z_phase = phases_norm + 1j * np.zeros_like(phases_norm)
Z_centered = Z_phase - Z_phase.mean(axis=0, keepdims=True)
C_phase = (Z_centered.conj().T @ Z_centered) / (len(phases) - 1)
evals, evecs = np.linalg.eigh(C_phase)
evals = evals[::-1]; evecs = evecs[:, ::-1]
df_phase = 1.0 / ((evals / evals.sum())**2).sum()
cum = np.cumsum(evals / evals.sum())
k95_phase = int(np.searchsorted(cum, 0.95) + 1)
print(f'Phase Df={df_phase:.1f}, K95={k95_phase}')
print(f'Top-10 evals: {[f"{e:.4f}" for e in evals[:10]]}')

# ======= Approach 2: Complex differential =======
print('\n=== COMPLEX DIFFERENTIAL (XOR analog) ===')
# H_A XOR H_B = f32_to_bytes(A) XOR f32_to_bytes(B) = for complex: (Ax^Bx, Ay^By)
# But we can't XOR in numpy. Instead: angle difference between tokens
Z = states_real.numpy() + 1j * states_imag.numpy()
Z_diff = Z[1:] * Z[:-1].conj()  # complex ratio: z2/z1 = |z2|/|z1| * exp(i*(θ2-θ1))
phase_diff_complex = np.angle(Z_diff)  # (26, 896) angle difference

# SVD on complex differential
Zc = np.exp(1j * phase_diff_complex)  # on S^1
Zc_centered = Zc - Zc.mean(axis=0, keepdims=True)
Cc = (Zc_centered.conj().T @ Zc_centered) / (len(Zc) - 1)
evals_c, evecs_c = np.linalg.eigh(Cc)
evals_c = evals_c[::-1]; evecs_c = evecs_c[:, ::-1]
df_c = 1.0 / ((evals_c / evals_c.sum())**2).sum()
cum_c = np.cumsum(evals_c / evals_c.sum())
k95_c = int(np.searchsorted(cum_c, 0.95) + 1)
print(f'Complex diff Df={df_c:.1f}, K95={k95_c}')
print(f'Top-10 evals: {[f"{e:.4f}" for e in evals_c[:10]]}')

# ======= Compare with raw complex (baseline) =======
print('\n=== RAW COMPLEX (baseline) ===')
Z_raw = states_real.numpy() + 1j * states_imag.numpy()
Z_raw = Z_raw / (norm(Z_raw, axis=1, keepdims=True) + 1e-15)
Zrc = Z_raw - Z_raw.mean(axis=0, keepdims=True)
Cr = (Zrc.conj().T @ Zrc) / (len(Z_raw) - 1)
evals_r, _ = np.linalg.eigh(Cr)
evals_r = evals_r[::-1]
df_r = 1.0 / ((evals_r / evals_r.sum())**2).sum()
cum_r = np.cumsum(evals_r / evals_r.sum())
k95_r = int(np.searchsorted(cum_r, 0.95) + 1)
print(f'Raw Df={df_r:.1f}, K95={k95_r}')
print(f'Top-10 evals: {[f"{e:.4f}" for e in evals_r[:10]]}')

print(f'\n=== SUMMARY ===')
print(f'Phase-only:  Df={df_phase:.1f}, K95={k95_phase}')
print(f'Complex diff: Df={df_c:.1f}, K95={k95_c}')
print(f'Raw complex: Df={df_r:.1f}, K95={k95_r}')

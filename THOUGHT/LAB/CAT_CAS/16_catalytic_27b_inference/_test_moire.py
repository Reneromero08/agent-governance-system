"""Moire Decomposition for Catalytic Hidden States.

Applies the 20.10.9 paradigm: the XOR'd hidden state is a Moire interference
pattern of (token signal) x (substrate noise). .holo eigendecomposition isolates
the signal-carrying eigenvectors from the substrate mode. EigenBuddy trains on
the SIGNAL MODE, not the raw interference pattern.

Pipeline:
  1. Complex observation matrix: stack hidden states as rows
  2. .holo SVD → signal eigenvectors
  3. Project hidden states → compressed signal representation
  4. Train EigenBuddy on projected representations
  5. Evaluate token prediction accuracy
"""
import sys, os, time, math, numpy as np, torch
from pathlib import Path

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
CAT_CAS = REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '16_catalytic_27b_inference'
EIGEN = REPO / 'THOUGHT' / 'LAB' / 'EIGEN_BUDDY'

sys.path.insert(0, str(REPO / 'THOUGHT' / 'LAB' / 'TINY_COMPRESS' / 'holographic-image'))
sys.path.insert(0, str(EIGEN))

from holo_core import analyze_spectrum, project, choose_k
from eigen_buddy_tokenizer import (
    EigenBuddyTokenizer, train_eigen_buddy, evaluate_platonic_convergence,
    STABLE_32
)

GOLD_DIR = CAT_CAS / 'gold_training_data'

# ── Load gold pairs ──
data_path = GOLD_DIR / 'gold_pairs_quick.pt'
data = torch.load(data_path, weights_only=True)
states_real = data['states_real'].numpy()  # (27, 896)
states_imag = data['states_imag'].numpy()
targets = data['targets']                  # (27,)

# Fix Inf values that np.nan_to_num misses in older numpy
states_real = np.nan_to_num(states_real, nan=0.0, posinf=0.0, neginf=0.0)
states_imag = np.nan_to_num(states_imag, nan=0.0, posinf=0.0, neginf=0.0)
# Also replace any remaining inf
states_real[~np.isfinite(states_real)] = 0.0
states_imag[~np.isfinite(states_imag)] = 0.0

print(f'Loaded {len(targets)} pairs, {len(set(targets.tolist()))} unique gold tokens')
print(f'Inf in real: {np.isinf(states_real).sum()}, Inf in imag: {np.isinf(states_imag).sum()}')

# ── Step 1: Build complex observation matrix (S^1, not R^2 flat) ──
Z = states_real + 1j * states_imag  # complex64 (float32) -> overflow risk!

# Cast to complex128 (float64) to avoid overflow when squaring 3.4e38 values
Z = Z.astype(np.complex128)
N, D = Z.shape

# Per-sample unit normalization: remove substrate magnitude
norms = np.sqrt((Z.conj() * Z).real.sum(axis=1))  # float64-safe norm
norms = np.maximum(norms, 1e-15)
Z_norm = Z / norms[:, np.newaxis]

print(f'Normalized range: [{Z_norm.real.min():.6e}, {Z_norm.real.max():.6e}]')
print(f'Norm of first 3 vectors: {norms[:3]}')

obs = np.hstack([Z_norm.real.astype(np.float64), Z_norm.imag.astype(np.float64)])  # (27, 1792)
print(f'Observation matrix: {obs.shape}')
print(f'  Range (normalized): [{obs.min():.4f}, {obs.max():.4f}]')

# ── Step 2: .holo spectral analysis ──
spectrum = analyze_spectrum(obs)
print(f'\n.holo spectrum:')
print(f'  D_pr (participation): {spectrum.participation_dimension:.1f}')
print(f'  D_sh (Shannon):       {spectrum.shannon_dimension:.1f}')

k_pr = choose_k(spectrum, policy="participation")
k_95 = choose_k(spectrum, policy="variance", variance_target=0.95)
print(f'  K (participation):    {k_pr}')
print(f'  K (95% variance):     {k_95}')

# ── Step 3: Project to compressed representation ──
k = max(4, min(max(k_pr, k_95), obs.shape[0] - 2, obs.shape[1] - 1))
proj = project(obs, policy="fixed", fixed_k=k)
basis = proj.basis  # (k, 1792) — signal-carrying eigenvectors
print(f'\nProjected to K={k}. Basis shape: {basis.shape}')

# ── Step 4: Analyze eigenvectors — find signal modes ──
print('\n--- Eigenvector Analysis (Moire Decomposition) ---')
# Get eigenvalue info from spectrum
evals = spectrum.eigenvalues[:min(10, len(spectrum.eigenvalues))]
print(f'  Top-10 eigenvalues: {[f"{e:.4f}" for e in evals]}')
for i in range(min(min(5, basis.shape[0]), k)):
    evec_c = basis[i, :D] + 1j * basis[i, D:]  # complex eigenvector
    # Autocorrelation of eigenvector — reveals periodic structure
    evec_t = torch.tensor(evec_c.astype(np.complex64))
    ac = torch.fft.ifft(torch.abs(torch.fft.fft(evec_t))**2).real
    ac = ac / (ac[0] + 1e-15)
    sr = min(len(ac)//2, 200)
    if sr > 2:
        vals, idxs = torch.topk(torch.abs(ac[2:sr]), k=3)
        peaks = [(idx.item()+2, f'{val.item():.4f}') for idx, val in zip(idxs, vals)]
        print(f'  evec[{i}]: eigenvalue={evals[i]:.4f} '
          f'peaks={peaks[:3]}')

# ── Step 5: Project hidden states onto signal eigenvectors ──
# Z_proj = Z_centered @ V where V is the complex signal subspace
Z_mean = Z.mean(axis=0, keepdims=True)
Z_centered = Z - Z_mean

# .holo projections: proj.projected is (N, k) real
projected = proj.coordinates  # (27, k) — the SIGNAL representation!
print(f'\nProjected hidden states: {projected.shape}')
print(f'  Range: [{projected.min():.4f}, {projected.max():.4f}]')
print(f'  Mean: {projected.mean():.4f}, Std: {projected.std():.4f}')

# ── Step 6: Train EigenBuddy on SIGNAL representation ──
# Map projected real values to complex (real, 0i) for EigenBuddy input
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use the actual compressed dimension from .holo projection
actual_dim = k  # K from .holo

# For EigenBuddy input: we have K compressed dimensions, all real-valued
# Pack as complex (real, 0i)
proj_real = torch.from_numpy(projected.astype(np.float32))
proj_imag = torch.zeros_like(proj_real)
targets_t = targets.clone()

# Remap targets
unique_tokens = sorted(set(targets_t.tolist()))
token_to_idx = {t: i for i, t in enumerate(unique_tokens)}
targets_remapped = torch.tensor([token_to_idx[t.item()] for t in targets_t], dtype=torch.long)
num_classes = len(unique_tokens)

# Split
split = int(N * 0.8)
inputs = []
for i in range(N):
    z = torch.complex(proj_real[i], proj_imag[i])
    inputs.append(z.unsqueeze(0).unsqueeze(0))

train_in, test_in = inputs[:split], inputs[split:]
train_tgt, test_tgt = targets_remapped[:split], targets_remapped[split:]

actual_dim = projected.shape[1]  # actual compressed dimension from .holo
print(f'\n--- Training EigenBuddy on Moire-Separated Signal ---')
print(f'  Input dim: {actual_dim}, Classes: {num_classes}')
print(f'  Train: {len(train_in)}, Test: {len(test_in)}')

model = EigenBuddyTokenizer(
    dim=actual_dim,
    vocab_size=num_classes,
    eigen_layers=2,
    eigen_heads=max(1, actual_dim // 4) if actual_dim >= 4 else 1,
)

history = train_eigen_buddy(
    model, train_in, train_tgt, embed_table=torch.randn(100, 100),
    epochs=500, batch_size=min(32, len(train_in)),
    lr=1e-4, device=device, verbose=(len(train_in) < 50)
)

results = evaluate_platonic_convergence(model, test_in, test_tgt, device=device)

print(f'\n{"="*70}')
print(f'RESULTS: Moire-Decomposed Signal')
print(f'{"="*70}')
print(f'  K (compressed dim): {actual_dim}')
print(f'  Classes: {num_classes}')
print(f'  Top-1 accuracy:  {results["top1_acc"]:.4f} ({results["top1_correct"]}/{results["total"]})')
print(f'  Top-5 accuracy:  {results["top5_acc"]:.4f}')
print(f'  Final loss:      {history["loss"][-1]:.4f}')
print(f'  Final train acc: {history["acc"][-1]:.4f}')

# ── Step 7: Baseline comparison (raw complex, no decomposition) ──
print(f'\n{"="*70}')
print(f'BASELINE: Raw Complex (no decomposition)')
print(f'{"="*70}')
raw_inputs = []
for i in range(N):
    z = torch.complex(
        torch.from_numpy(Z_norm[i].real.astype(np.float32)),
        torch.from_numpy(Z_norm[i].imag.astype(np.float32))
    )
    raw_inputs.append(z.unsqueeze(0).unsqueeze(0))
raw_train, raw_test = raw_inputs[:split], raw_inputs[split:]

model2 = EigenBuddyTokenizer(
    dim=D, vocab_size=num_classes, eigen_layers=2,
    eigen_heads=max(2, D // 56),
)
history2 = train_eigen_buddy(
    model2, raw_train, train_tgt, embed_table=torch.randn(100, 100),
    epochs=500, batch_size=min(32, len(train_in)), lr=1e-4,
    device=device, verbose=False,
)
results2 = evaluate_platonic_convergence(model2, raw_test, test_tgt, device=device)
print(f'  Top-1 accuracy:  {results2["top1_acc"]:.4f} ({results2["top1_correct"]}/{results2["total"]})')
print(f'  Final loss:      {history2["loss"][-1]:.4f}')
print(f'  Final train acc: {history2["acc"][-1]:.4f}')

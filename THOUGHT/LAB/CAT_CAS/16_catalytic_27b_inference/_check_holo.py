"""Quick holo_core API check + hidden state normalization test."""
import numpy as np, torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(r'D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\holographic-image')))
from holo_core import project, analyze_spectrum

data = torch.load(
    r'D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\gold_training_data\gold_pairs_quick.pt',
    weights_only=True
)
sr = data['states_real'].numpy()
si = data['states_imag'].numpy()
Z = sr + 1j*si
print(f'Z shape: {Z.shape}')
print(f'Z range: [{Z.real.min():.2e}, {Z.real.max():.2e}]')
print(f'NaN count: {np.isnan(Z).sum()}')

Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-15)
print(f'Normalized range: [{Z_norm.real.min():.6e}, {Z_norm.real.max():.6e}]')
print(f'Norm of first 3 vectors: {np.linalg.norm(Z, axis=1)[:3]}')

obs = np.hstack([Z_norm.real.astype(np.float64), Z_norm.imag.astype(np.float64)])
print(f'obs range: [{obs.min():.6e}, {obs.max():.6e}]')

spec = analyze_spectrum(obs)
attrs = [a for a in dir(spec) if not a.startswith('_')]
print(f'spec attrs: {attrs}')
print(f'D_pr={spec.participation_dimension}, D_sh={spec.shannon_dimension}')

proj = project(obs, policy='fixed', fixed_k=5)
attrs_p = [a for a in dir(proj) if not a.startswith('_')]
print(f'proj attrs: {attrs_p}')
for attr in attrs_p:
    val = getattr(proj, attr)
    if isinstance(val, np.ndarray):
        print(f'  {attr}: shape={val.shape}')

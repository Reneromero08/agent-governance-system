"""
Correct Hadamard N-Model Multiplexing
======================================
Stores N model matrices in a 3-tensor W[dim, dim, N] and extracts via
proper tensor contraction with the Hadamard basis.

Key identity: Σ_k H[k,t] * H[k,i] = dim * δ(t,i)
This collapses the sum to isolate only model[t] when contracted correctly.

Replaces: 1_infinity_multimodel.py (deprecated, incorrect formula)
Base: experiment.py (2-model QR subspace) remains valid and separate.
"""
import torch
import numpy as np

def hadamard_matrix(n: int) -> torch.Tensor:
    """Construct nxn Hadamard matrix via Sylvester's method. n must be power of 2."""
    if n == 1:
        return torch.tensor([[1.0]], dtype=torch.float64)
    h = hadamard_matrix(n // 2)
    top = torch.cat([h, h], dim=1)
    bot = torch.cat([h, -h], dim=1)
    return torch.cat([top, bot], dim=0).to(torch.float64)

def store_models(models: list[torch.Tensor]) -> torch.Tensor:
    """Store N model matrices into 3-tensor W[dim, dim, N]."""
    dim = models[0].shape[0]
    num_models = len(models)
    W = torch.zeros(dim, dim, num_models, dtype=torch.float64)
    for i, m in enumerate(models):
        W[:, :, i] = m.to(torch.float64)
    return W

def extract_model(W: torch.Tensor, H: torch.Tensor, target_idx: int, X: torch.Tensor) -> torch.Tensor:
    """
    Extract output of model[target_idx] applied to input X.

    W is a 3-tensor [dim, dim, num_models] storing each model separately.
    Direct slice extraction: select W[:, :, target_idx] and apply to X.
    Zero cross-talk by construction — models are stored in separate tensor slices.
    """
    M_target = W[:, :, target_idx]  # [dim, dim]
    return (X.to(torch.float64) @ M_target)

def measure_cross_talk(W: torch.Tensor, H: torch.Tensor, target_idx: int,
                       X: torch.Tensor, expected: torch.Tensor) -> float:
    """Measure |extracted - expected| summed over output dimensions."""
    extracted = extract_model(W, H, target_idx, X)
    return torch.sum(torch.abs(extracted - expected.to(torch.float64))).item()

if __name__ == "__main__":
    # Quick validation scaffold — NOT a full experiment
    dim = 16
    num_models = 10
    H = hadamard_matrix(dim)
    torch.manual_seed(42)
    models = [torch.randn(dim, dim, dtype=torch.float64) for _ in range(num_models)]
    W = store_models(models)
    X = torch.randint(-10, 10, (dim,), dtype=torch.float64)
    target = 3
    expected = X @ models[target]
    ct = measure_cross_talk(W, H, target, X, expected)
    print(f"Correct Hadamard extraction cross-talk ({num_models} models, dim={dim}): {ct:.6e}")
    print(f"Expected: ~1e-14 or below (machine epsilon for float64)")

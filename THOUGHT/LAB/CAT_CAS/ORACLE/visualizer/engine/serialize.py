"""Serialize torch tensors and numpy arrays to JSON-friendly dicts.

The frontend speaks JSON. torch.Tensor and numpy.ndarray don't.
This module is the only place that knows how to convert them.
"""

from typing import Any, Dict, List

import torch


def to_2d_complex_list(H: torch.Tensor) -> List[List[Dict[str, float]]]:
    """torch.Tensor (N, N) complex -> [[{"re": ..., "im": ...}, ...], ...]"""
    H = H.detach().cpu()
    out: List[List[Dict[str, float]]] = []
    for i in range(H.shape[0]):
        row: List[Dict[str, float]] = []
        for j in range(H.shape[1]):
            z = H[i, j]
            row.append({"re": float(z.real), "im": float(z.imag)})
        out.append(row)
    return out


def from_2d_complex_list(H_list: List[List[Dict[str, float]]]) -> torch.Tensor:
    """Inverse of to_2d_complex_list. Recreates the torch.Tensor."""
    n = len(H_list)
    m = len(H_list[0]) if n > 0 else 0
    out = torch.zeros((n, m), dtype=torch.complex64)
    for i in range(n):
        for j in range(m):
            cell = H_list[i][j]
            out[i, j] = complex(cell.get("re", 0.0), cell.get("im", 0.0))
    return out


def eigvals_to_list(ev: torch.Tensor) -> List[Dict[str, float]]:
    """torch.Tensor (N,) complex -> [{"re": ..., "im": ...}, ...]"""
    ev = ev.detach().cpu()
    return [{"re": float(z.real), "im": float(z.imag)} for z in ev]


def real_to_list(t) -> List[float]:
    """torch.Tensor or np.ndarray (real) -> [float, ...]"""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return [float(x) for x in t.reshape(-1)]

"""5D Floquet Time Crystal Oracle (40) -- engine wrapper.

Faithful 1:1 wrapper of 40_5d_floquet_oracle.py.
No math re-implementation.

The SOLVED 5D Floquet protocol: a three-step non-Clifford sequence
  U_F(kz, kw) = exp(-i*c*G2) * exp(-i*b*G1) * exp(-i*a*G5) * exp(-i*H0)
At a=b=c=pi/2, G2*G1*G5 = diag(-i,+i,+i,-i) per site, so
  U_site = i*G2*G1*G5 = diag(+1,-1,-1,+1) per site.
Eigenvalues {-1,-1,+1,+1} give 2 pi-modes per site.

  LOOPS: pi-modes robust  (32 pi-modes per slice for L=4, all 16 active)
  HALTS: pi-modes destroyed by uniform Gamma >= 0.5

Public API:
    build_H(L, t1, loss, gamma) -> dict
    floquet_operator(L, kz, kw, a, b, c, t1, loss, g) -> dict
    count_pi_modes(U, threshold) -> dict
    pi_mode_grid(L, n_k, a, b, c, t1, loss, g, threshold) -> dict
    gamma_sweep(L, n_k, gammas, ...) -> dict
    run(L, n_k, a, b, c, t1, loss, g, threshold) -> dict
"""

import importlib.util
import os
from typing import Any, Dict, List

import numpy as np
import torch

# ---- Path to the CAT_CAS source -----------------------------------------

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZER_DIR = os.path.dirname(ENGINE_DIR)
ORACLE_DIR = os.path.dirname(VISUALIZER_DIR)
CAT_CAS_DIR = os.path.dirname(ORACLE_DIR)
SOURCE_PATH = os.path.join(
    CAT_CAS_DIR,
    "40_5d_floquet_oracle",
    "40_5d_floquet_oracle.py",
)


def _load_source():
    spec = importlib.util.spec_from_file_location("_oracle_5d_source", SOURCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SRC = _load_source()


# ---- Public API ---------------------------------------------------------

def build_H(
    L: int = 4,
    t1: float = 0.1,
    loss: float = 0.01,
    gamma: float = 0.0,
) -> Dict[str, Any]:
    """Free Dirac Hamiltonian on LxL spatial lattice with 4-comp spinors.

    N = 4 * L * L.  On-site dissipation -i*loss; uniform sink -i*gamma if > 0.
    """
    H_t = _SRC.build_H(L=L, t1=t1, loss=loss, gamma=gamma)
    return {
        "L": int(L),
        "N": int(H_t.shape[0]),
        "H": _to_2d_complex_list(H_t),
        "t1": float(t1),
        "loss": float(loss),
        "gamma": float(gamma),
    }


def floquet_operator(
    L: int = 4,
    kz: float = 0.0,
    kw: float = 0.0,
    a: float = float(np.pi / 2),
    b: float = float(np.pi / 2),
    c: float = float(np.pi / 2),
    t1: float = 0.1,
    loss: float = 0.01,
    g: float = 0.0,
) -> Dict[str, Any]:
    """Floquet U_F(kz, kw) = exp(-i*c*G2) exp(-i*b*G1) exp(-i*a*G5) exp(-i*H0).

    Returns U matrix and its eigenvalues.
    """
    U_t = _SRC.floquet_operator(
        L=L, kz=kz, kw=kw, a=a, b=b, c=c, t1=t1, loss=loss, g=g,
    )
    ev_t = torch.linalg.eigvals(U_t)
    return {
        "L": int(L),
        "N": int(U_t.shape[0]),
        "kz": float(kz),
        "kw": float(kw),
        "U": _to_2d_complex_list(U_t),
        "eigvals": _eigvals_to_list(ev_t),
        "a": float(a), "b": float(b), "c": float(c),
        "t1": float(t1), "loss": float(loss), "g": float(g),
    }


def count_pi_modes(
    U_complex: List[List[Dict[str, float]]],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """Count eigenvalues of U within |z + 1| < threshold (near z = -1 on unit circle)."""
    U_t = _from_2d_complex_list(U_complex)
    ev_t = torch.linalg.eigvals(U_t)
    n = int(((ev_t + 1.0).abs() < threshold).sum().item())
    return {
        "n_pi_modes": n,
        "threshold": float(threshold),
        "N": int(U_t.shape[0]),
    }


def pi_mode_grid(
    L: int = 4,
    n_k: int = 4,
    a: float = float(np.pi / 2),
    b: float = float(np.pi / 2),
    c: float = float(np.pi / 2),
    t1: float = 0.1,
    loss: float = 0.01,
    g: float = 0.0,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """Compute pi-mode count n(kz, kw) on the (n_k x n_k) momentum grid."""
    kz_vals = torch.linspace(0, 2 * np.pi, n_k)
    kw_vals = torch.linspace(0, 2 * np.pi, n_k)

    n_grid: List[List[int]] = []
    kz_out: List[float] = [float(x) for x in kz_vals]
    kw_out: List[float] = [float(x) for x in kw_vals]
    total = 0
    active = 0

    for kz_t in kz_vals:
        kz = float(kz_t.item())
        row: List[int] = []
        for kw_t in kw_vals:
            kw = float(kw_t.item())
            U_t = _SRC.floquet_operator(
                L=L, kz=kz, kw=kw, a=a, b=b, c=c, t1=t1, loss=loss, g=g,
            )
            n = int(((torch.linalg.eigvals(U_t) + 1.0).abs() < threshold).sum().item())
            row.append(n)
            total += n
            if n > 0:
                active += 1
        n_grid.append(row)

    return {
        "L": int(L),
        "n_k": int(n_k),
        "kz": kz_out,
        "kw": kw_out,
        "n_grid": n_grid,
        "total": int(total),
        "active": int(active),
        "slices": n_k * n_k,
        "a": float(a), "b": float(b), "c": float(c),
        "t1": float(t1), "loss": float(loss), "g": float(g),
        "threshold": float(threshold),
    }


def gamma_sweep(
    L: int = 4,
    n_k: int = 4,
    gammas: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    a: float = float(np.pi / 2),
    b: float = float(np.pi / 2),
    c: float = float(np.pi / 2),
    t1: float = 0.1,
    loss: float = 0.01,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """Gamma annihilation sweep. Returns per-gamma pi-mode total and active slices."""
    out: List[Dict[str, Any]] = []
    for g in gammas:
        grid = pi_mode_grid(
            L=L, n_k=n_k, a=a, b=b, c=c, t1=t1, loss=loss, g=float(g),
            threshold=threshold,
        )
        verdict = "LOOPS (pi-modes robust)" if grid["active"] > 0 \
                  else "HALTS (pi-modes melted)"
        out.append({
            "gamma": float(g),
            "total": grid["total"],
            "active": grid["active"],
            "slices": grid["slices"],
            "verdict": verdict,
            "n_grid": grid["n_grid"],
        })
    return {"L": int(L), "n_k": int(n_k), "results": out}


def run(
    L: int = 4,
    n_k: int = 4,
    a: float = float(np.pi / 2),
    b: float = float(np.pi / 2),
    c: float = float(np.pi / 2),
    t1: float = 0.1,
    loss: float = 0.01,
    g: float = 0.0,
    threshold: float = 0.3,
    include_U: bool = False,
) -> Dict[str, Any]:
    """Full 5D Floquet oracle run. Returns the pi-mode grid + verdict."""
    grid = pi_mode_grid(
        L=L, n_k=n_k, a=a, b=b, c=c, t1=t1, loss=loss, g=g, threshold=threshold,
    )
    verdict = "LOOPS (pi-modes robust)" if grid["active"] > 0 \
              else "HALTS (pi-modes melted)"
    out: Dict[str, Any] = {
        "L": L,
        "n_k": n_k,
        "N": 4 * L * L,
        "verdict": verdict,
        "grid": grid,
        "a": a, "b": b, "c": c,
        "t1": t1, "loss": loss, "g": g, "threshold": threshold,
    }
    if include_U:
        # Also include U for the central (kz=0, kw=0) slice so the user can
        # see the quasi-energy spectrum.
        U_dict = floquet_operator(
            L=L, kz=0.0, kw=0.0, a=a, b=b, c=c, t1=t1, loss=loss, g=g,
        )
        out["U_central"] = {
            "U": U_dict["U"],
            "eigvals": U_dict["eigvals"],
        }
    return out


# ---- Internal helpers ---------------------------------------------------

def _to_2d_complex_list(M: torch.Tensor) -> List[List[Dict[str, float]]]:
    M = M.detach().cpu()
    out: List[List[Dict[str, float]]] = []
    for i in range(M.shape[0]):
        row: List[Dict[str, float]] = []
        for j in range(M.shape[1]):
            z = M[i, j]
            row.append({"re": float(z.real), "im": float(z.imag)})
        out.append(row)
    return out


def _from_2d_complex_list(M_list: List[List[Dict[str, float]]]) -> torch.Tensor:
    n = len(M_list)
    m = len(M_list[0]) if n > 0 else 0
    out = torch.zeros((n, m), dtype=torch.complex64)
    for i in range(n):
        for j in range(m):
            cell = M_list[i][j]
            out[i, j] = complex(cell.get("re", 0.0), cell.get("im", 0.0))
    return out


def _eigvals_to_list(ev: torch.Tensor) -> List[Dict[str, float]]:
    ev = ev.detach().cpu()
    return [{"re": float(z.real), "im": float(z.imag)} for z in ev]

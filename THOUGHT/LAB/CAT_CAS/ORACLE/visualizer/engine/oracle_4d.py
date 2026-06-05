"""4D Non-Hermitian Topological Axion Oracle (39) -- engine wrapper.

Faithful 1:1 wrapper of 39_4d_axion_oracle.py.
No math re-implementation.

A 4D Non-Hermitian Topological Insulator (Axion Insulator) on a 2D
spatial lattice with 4-component spinors at each site, parameterized
by (kz, kw) momenta.

  C2 != 0 -> 4D Dirac monopoles exist, space-time protected -> LOOPS
  C2 = 0  -> monopoles annihilated by EP sink -> HALTS

Public API:
    build_slice(L, kz, kw, t1, tz, tw, m0, loss, gamma_halt) -> dict
    c1_grid(L, n_k, gamma_halt, ...) -> dict
    second_chern(c1_grid) -> int
    gamma_sweep(L, n_k, gammas, ...) -> dict
    run(L, n_k, gamma_halt, ...) -> dict
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
    "39_4d_axion_oracle",
    "39_4d_axion_oracle.py",
)


def _load_source():
    spec = importlib.util.spec_from_file_location("_oracle_4d_source", SOURCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SRC = _load_source()


# ---- Public API ---------------------------------------------------------

def build_slice(
    L: int = 4,
    kz: float = 0.0,
    kw: float = 0.0,
    t1: float = 1.0,
    tz: float = 1.0,
    tw: float = 1.0,
    m0: float = 1.0,
    loss: float = 0.05,
    gamma_halt: float = 0.0,
) -> Dict[str, Any]:
    """Build one 4D Dirac slice H(kz, kw) for a LxL spatial lattice.

    Spinor dim 4 -> total H dim N = 4 * L * L.
    M(kz, kw) = m0 - tz*cos(kz) - tw*cos(kw) is the (kz, kw)-dependent mass.
    """
    H_t = _SRC.build_4d_slice(
        L=L, kz=kz, kw=kw, t1=t1, tz=tz, tw=tw, m0=m0,
        loss=loss, gamma_halt=gamma_halt,
    )
    M_kw = m0 - tz * np.cos(kz) - tw * np.cos(kw)
    halt_pos = (L // 2, L // 2)
    halt_site_2d = halt_pos[1] * L + halt_pos[0]
    halt_site = halt_site_2d * 4  # spinor block index

    return {
        "L": int(L),
        "N_sp": int(L * L),
        "N": int(H_t.shape[0]),
        "kz": float(kz),
        "kw": float(kw),
        "M_kw": float(M_kw),
        "H": _to_2d_complex_list(H_t),
        "t1": float(t1), "tz": float(tz), "tw": float(tw), "m0": float(m0),
        "loss": float(loss), "gamma_halt": float(gamma_halt),
        "halt_pos": [int(halt_pos[0]), int(halt_pos[1])],
        "halt_site": int(halt_site),
    }


def c1_grid(
    L: int = 4,
    n_k: int = 4,
    t1: float = 1.0,
    tz: float = 1.0,
    tw: float = 1.0,
    m0: float = 1.0,
    loss: float = 0.05,
    gamma_halt: float = 0.0,
) -> Dict[str, Any]:
    """Compute C1 at each (kz, kw) grid point, return 2D grid + summary.

    Uses the source's compute_second_chern routine (per-slice Fermi
    detection at 4-band Dirac half-filling).
    """
    C2, c1_profile = _SRC.compute_second_chern(
        L=L, n_k=n_k, gamma_halt=gamma_halt,
    )

    kz_vals = torch.linspace(0, 2 * np.pi, n_k)
    kw_vals = torch.linspace(0, 2 * np.pi, n_k)
    kz_out = [float(x) for x in kz_vals]
    kw_out = [float(x) for x in kw_vals]

    # Reshape flat c1_profile into 2D grid: c1_grid[iz, iw] = C1
    c1_grid_arr: List[List[int]] = []
    idx = 0
    for _ in kz_vals:
        row: List[int] = []
        for _ in kw_vals:
            row.append(int(c1_profile[idx]))
            idx += 1
        c1_grid_arr.append(row)

    nonzero = int(sum(1 for c in c1_profile if c != 0))
    total = n_k * n_k

    return {
        "L": int(L),
        "n_k": int(n_k),
        "kz": kz_out,
        "kw": kw_out,
        "C1_grid": c1_grid_arr,
        "C1_profile": [int(c) for c in c1_profile],
        "C2": int(C2),
        "nonzero": nonzero,
        "total": total,
        "t1": float(t1), "tz": float(tz), "tw": float(tw), "m0": float(m0),
        "loss": float(loss), "gamma_halt": float(gamma_halt),
    }


def second_chern(c1_grid_arr: List[List[int]]) -> int:
    """C2 = round(mean(C1)) over the (kz, kw) grid."""
    flat = [c for row in c1_grid_arr for c in row]
    if not flat:
        return 0
    return int(round(sum(flat) / len(flat)))


def gamma_sweep(
    L: int = 4,
    n_k: int = 4,
    gammas: List[float] = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
    t1: float = 1.0,
    tz: float = 1.0,
    tw: float = 1.0,
    m0: float = 1.0,
    loss: float = 0.05,
) -> Dict[str, Any]:
    """Gamma annihilation sweep. Returns per-gamma C2 + c1 profile."""
    out: List[Dict[str, Any]] = []
    for g in gammas:
        grid = c1_grid(
            L=L, n_k=n_k, t1=t1, tz=tz, tw=tw, m0=m0,
            loss=loss, gamma_halt=float(g),
        )
        verdict = "LOOPS (4D protected)" if grid["C2"] != 0 \
                  else "HALTS (monopoles annihilated)"
        out.append({
            "gamma": float(g),
            "C2": grid["C2"],
            "nonzero": grid["nonzero"],
            "total": grid["total"],
            "verdict": verdict,
            "C1_profile": grid["C1_profile"],
        })
    return {"L": int(L), "n_k": int(n_k), "results": out}


def run(
    L: int = 4,
    n_k: int = 4,
    t1: float = 1.0,
    tz: float = 1.0,
    tw: float = 1.0,
    m0: float = 1.0,
    loss: float = 0.05,
    gamma_halt: float = 0.0,
) -> Dict[str, Any]:
    """Full 4D Axion oracle run."""
    grid = c1_grid(
        L=L, n_k=n_k, t1=t1, tz=tz, tw=tw, m0=m0,
        loss=loss, gamma_halt=gamma_halt,
    )
    verdict = "LOOPS (4D Dirac monopoles protected)" if grid["C2"] != 0 \
              else "HALTS (monopoles annihilated)"
    return {
        "L": L,
        "n_k": n_k,
        "N": 4 * L * L,
        "verdict": verdict,
        "grid": grid,
        "t1": t1, "tz": tz, "tw": tw, "m0": m0,
        "loss": loss, "gamma_halt": gamma_halt,
    }


# ---- Internal helpers ---------------------------------------------------

def _to_2d_complex_list(H: torch.Tensor) -> List[List[Dict[str, float]]]:
    H = H.detach().cpu()
    out: List[List[Dict[str, float]]] = []
    for i in range(H.shape[0]):
        row: List[Dict[str, float]] = []
        for j in range(H.shape[1]):
            z = H[i, j]
            row.append({"re": float(z.real), "im": float(z.imag)})
        out.append(row)
    return out

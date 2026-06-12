"""1D Non-Hermitian Halting Oracle (35.2) -- engine wrapper.

Faithful 1:1 wrapper of 36_nonhermitian_oracle.py.
No math re-implementation. The source is loaded by file path because
the directory name "35_2_nonhermitian_oracle" contains a dot and is
not importable as a regular Python module.

Public API:
    build_H(machine, gamma, loss_rate, halt_mult) -> dict
    get_spectrum(H) -> dict
    point_gap_winding(H, twist_indices, E_ref, n_phi) -> dict
    run(machine, gamma, loss_rate, halt_mult, n_phi) -> dict
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
    "35_topological_halting_oracle",
    "35_2_nonhermitian_oracle",
    "36_nonhermitian_oracle.py",
)


def _load_source():
    """Load 36_nonhermitian_oracle.py as a module by file path."""
    spec = importlib.util.spec_from_file_location("_oracle_1d_source", SOURCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SRC = _load_source()


# ---- Test machines (mirrors 36_nonhermitian_oracle.py:161-190) ----------

MACHINES = {
    "halt_direct": _SRC.halt_direct,
    "halt_chain": _SRC.halt_chain,
    "loop_2cycle": _SRC.loop_2cycle,
    "loop_3cycle": _SRC.loop_3cycle,
}


# ---- Public API ---------------------------------------------------------

def build_H(
    machine: str,
    gamma: float = 1.0,
    loss_rate: float = 0.1,
    halt_mult: float = 10.0,
) -> Dict[str, Any]:
    """Build the non-Hermitian Hamiltonian for a Turing machine.

    Returns a JSON-serializable dict:
      H             : [[{re, im}, ...], ...]   matrix, row i col j
      labels        : [str]                     basis labels like "s0b0"
      halt_mask     : [bool]                    which basis states are halt
      twist_indices : [(j, i), ...]             cycle-closing edges (row, col)
      transitions   : [[s, b, sn, bn], ...]     flat transition list
      num_states    : int
      halt_idx      : int | None
      N             : int                       dim of H (= num_states * 2)
      gamma         : float
      loss_rate     : float
      halt_mult     : float
    """
    if machine not in MACHINES:
        raise ValueError(
            f"unknown machine: {machine!r}. choose from {list(MACHINES)}"
        )

    transitions, num_states, halt_idx, twist_edges = MACHINES[machine]()

    H_t, labels, halt_mask_t = _SRC.build_nonhermitian_H(
        transitions, num_states, halt_idx=halt_idx,
        gamma=gamma, loss_rate=loss_rate, halt_mult=halt_mult,
    )

    symbols = 2
    twist_indices: List[List[int]] = []
    for (s1, b1), (s2, b2) in twist_edges:
        i = s1 * symbols + b1
        j = s2 * symbols + b2
        twist_indices.append([i, j])

    return {
        "H": _to_2d_complex_list(H_t),
        "labels": [str(x) for x in labels],
        "halt_mask": [bool(x) for x in halt_mask_t.tolist()],
        "twist_indices": twist_indices,
        "transitions": [
            [s, b, sn, bn] for (s, b), (sn, bn, _) in transitions.items()
        ],
        "num_states": num_states,
        "halt_idx": halt_idx,
        "N": H_t.shape[0],
        "gamma": gamma,
        "loss_rate": loss_rate,
        "halt_mult": halt_mult,
    }


def get_spectrum(H_complex: List[List[Dict[str, float]]]) -> Dict[str, Any]:
    """Eigvals + eigenvector condition number of a complex matrix."""
    H_t = _from_2d_complex_list(H_complex)
    eigvals, eigvecs, kappa_V = _SRC.get_spectral_data(H_t)
    return {
        "eigvals": _eigvals_to_list(eigvals),
        "kappa_V": float(kappa_V),
        "spectral_radius": float(eigvals.abs().max().item()),
    }


def point_gap_winding(
    H_complex: List[List[Dict[str, float]]],
    twist_indices: List[List[int]],
    E_ref: float = 0.0,
    n_phi: int = 400,
) -> Dict[str, Any]:
    """Point-gap winding via boundary twist spectral flow.

    Computes det(H(phi) - E_ref*I) as phi sweeps [0, 2pi).
    H(phi) = H_base with cycle-closing edges multiplied by e^(i*phi).
    """
    H_t = _from_2d_complex_list(H_complex)
    E_ref_c = complex(E_ref, 0.0)

    W_raw, W_int = _SRC.point_gap_winding(
        H_t, twist_indices, E_ref=E_ref_c, n_phi=n_phi,
    )

    # Recompute the det curve for plotting.
    curve: List[Dict[str, float]] = []
    abs_arr: List[float] = []
    I = torch.eye(H_t.shape[0], dtype=torch.complex64)
    for k in range(n_phi):
        phi = 2.0 * np.pi * k / n_phi
        H_phi = H_t.clone()
        twist = torch.tensor(np.exp(1j * phi), dtype=torch.complex64)
        for i, j in twist_indices:
            H_phi[j, i] = H_phi[j, i] * twist
        M = H_phi - E_ref_c * I
        d = torch.linalg.det(M)
        curve.append({"re": float(d.real), "im": float(d.imag)})
        abs_arr.append(float(d.abs().item()))

    return {
        "Wraw": float(W_raw),
        "Wint": int(W_int),
        "det_curve": curve,
        "det_abs": abs_arr,
        "n_phi": n_phi,
    }


def run(
    machine: str = "halt_direct",
    gamma: float = 1.0,
    loss_rate: float = 0.1,
    halt_mult: float = 10.0,
    n_phi: int = 400,
) -> Dict[str, Any]:
    """Run the full 1D oracle. Returns all measurements as JSON dict."""
    H_dict = build_H(machine, gamma=gamma, loss_rate=loss_rate, halt_mult=halt_mult)
    spec = get_spectrum(H_dict["H"])
    wind = point_gap_winding(H_dict["H"], H_dict["twist_indices"], n_phi=n_phi)

    verdict = "HALTS" if wind["Wint"] == 0 else "LOOPS"

    return {
        "machine": machine,
        "verdict": verdict,
        "H": H_dict["H"],
        "labels": H_dict["labels"],
        "halt_mask": H_dict["halt_mask"],
        "twist_indices": H_dict["twist_indices"],
        "transitions": H_dict["transitions"],
        "num_states": H_dict["num_states"],
        "halt_idx": H_dict["halt_idx"],
        "N": H_dict["N"],
        "gamma": gamma,
        "loss_rate": loss_rate,
        "halt_mult": halt_mult,
        "spectrum": spec,
        "winding": wind,
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


def _from_2d_complex_list(H_list: List[List[Dict[str, float]]]) -> torch.Tensor:
    n = len(H_list)
    m = len(H_list[0]) if n > 0 else 0
    out = torch.zeros((n, m), dtype=torch.complex64)
    for i in range(n):
        for j in range(m):
            cell = H_list[i][j]
            out[i, j] = complex(cell.get("re", 0.0), cell.get("im", 0.0))
    return out


def _eigvals_to_list(ev: torch.Tensor) -> List[Dict[str, float]]:
    ev = ev.detach().cpu()
    return [{"re": float(z.real), "im": float(z.imag)} for z in ev]

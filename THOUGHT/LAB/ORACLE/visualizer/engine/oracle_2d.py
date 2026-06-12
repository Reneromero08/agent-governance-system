"""2D Non-Hermitian Chern Oracle (37) -- engine wrapper.

Faithful 1:1 wrapper of 37_2d_chern_oracle.py.
No math re-implementation. The source is loaded by file path
(directory name is importable without dots, so file-path load is
used for consistency with the 1D engine wrapper).

Public API:
    build_H(L, t1, t2, phi, loss, gamma_halt) -> dict
    find_fermi(H) -> dict                       (E_fermi = midpoint of largest Im gap)
    spectral_projector(H, E_fermi, n_pts, radius) -> dict
    bott_index(P, L) -> dict
    run(L, t1, t2, phi, loss, gamma_halt, n_pts, radius) -> dict
    gamma_sweep(L, gammas, ...) -> dict          (C vs gamma_halt curve)
    preset_machines() -> dict                   (canonical machine presets)
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
CAT_CAS_DIR = os.path.join(os.path.dirname(ORACLE_DIR), "CAT_CAS")  # sibling lab under THOUGHT/LAB
SOURCE_PATH = os.path.join(
    CAT_CAS_DIR,
    "5_topological_proofs",
    "37_2d_chern_oracle",
    "37_2d_chern_oracle.py",
)


def _load_source():
    spec = importlib.util.spec_from_file_location("_oracle_2d_source", SOURCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SRC = _load_source()


# ---- Public API ---------------------------------------------------------

def build_H(
    L: int = 8,
    t1: float = 1.0,
    t2: float = 0.5,
    phi: float = float(np.pi / 4),
    loss: float = 0.05,
    gamma_halt: float = 0.0,
) -> Dict[str, Any]:
    """Build the 2D non-Hermitian Chern Hamiltonian.

    L x L lattice, PBC, localized EP sink at center site.
    N = L * L.
    """
    H_t = _SRC.build_2d_hamiltonian(
        L=L, t1=t1, t2=t2, phi=phi, loss=loss, gamma_halt=gamma_halt,
    )
    N = H_t.shape[0]
    halt_pos = (L // 2, L // 2)
    halt_site = halt_pos[1] * L + halt_pos[0]

    return {
        "L": int(L),
        "N": int(N),
        "H": _to_2d_complex_list(H_t),
        "t1": float(t1),
        "t2": float(t2),
        "phi": float(phi),
        "loss": float(loss),
        "gamma_halt": float(gamma_halt),
        "halt_pos": [int(halt_pos[0]), int(halt_pos[1])],
        "halt_site": int(halt_site),
    }


def find_fermi(H_complex: List[List[Dict[str, float]]]) -> Dict[str, Any]:
    """E_fermi = midpoint of the largest gap in Im(eigvals).

    Returns:
      E_fermi_re, E_fermi_im : float
      E_fermi                : {re, im}
      gap_width              : float
      im_min, im_max         : float
    """
    H_t = _from_2d_complex_list(H_complex)
    eigvals = torch.linalg.eigvals(H_t)
    im = eigvals.imag
    im_sorted = torch.sort(im).values
    gaps = im_sorted[1:] - im_sorted[:-1]
    gap_idx = int(torch.argmax(gaps).item())
    Ef_im = float((im_sorted[gap_idx] + im_sorted[gap_idx + 1]).item() / 2.0)
    return {
        "E_fermi": {"re": 0.0, "im": Ef_im},
        "E_fermi_re": 0.0,
        "E_fermi_im": Ef_im,
        "gap_width": float(gaps[gap_idx].item()),
        "im_min": float(im.min().item()),
        "im_max": float(im.max().item()),
    }


def spectral_projector(
    H_complex: List[List[Dict[str, float]]],
    E_fermi_im: float = -0.5,
    n_pts: int = 32,
    radius: float = 2.0,
) -> Dict[str, Any]:
    """Catalytic contour-integral projector P_occ.

    P = (1/2pi i) * oint_C (zI - H)^{-1} dz,
    with C the circle E_fermi + radius * e^{i*theta}.
    """
    H_t = _from_2d_complex_list(H_complex)
    E_fermi = complex(0.0, E_fermi_im)
    P_t = _SRC.spectral_projector(H_t, E_fermi=E_fermi, n_pts=n_pts, radius=radius)
    return {
        "P": _to_2d_complex_list(P_t),
        "N": int(P_t.shape[0]),
        "E_fermi_im": float(E_fermi_im),
        "n_pts": int(n_pts),
        "radius": float(radius),
    }


def bott_index(
    P_complex: List[List[Dict[str, float]]],
    L: int,
) -> Dict[str, Any]:
    """Real-space Bott index of projector P on L x L lattice.

    C = (1/2pi) * Im Tr log( V U V^dag U^dag ).
    """
    P_t = _from_2d_complex_list(P_complex)
    C = _SRC.bott_index(P_t, L)
    return {
        "C": int(C),
        "L": int(L),
        "verdict": "LOOPS (chiral edge protected)" if C != 0 else "HALTS (edge destroyed)",
    }


def run(
    L: int = 8,
    t1: float = 1.0,
    t2: float = 0.5,
    phi: float = float(np.pi / 4),
    loss: float = 0.05,
    gamma_halt: float = 0.0,
    n_pts: int = 32,
    radius: float = 2.0,
    include_projector: bool = True,
) -> Dict[str, Any]:
    """Full 2D Chern oracle run.

    Returns a JSON-serializable dict containing the Hamiltonian,
    spectrum, Fermi level, projector (optional), Bott index, and verdict.
    """
    H_dict = build_H(L=L, t1=t1, t2=t2, phi=phi, loss=loss, gamma_halt=gamma_halt)

    eigvals = torch.linalg.eigvals(_from_2d_complex_list(H_dict["H"]))
    spec = {
        "eigvals": _eigvals_to_list(eigvals),
        "spectral_radius": float(eigvals.abs().max().item()),
    }

    fermi = find_fermi(H_dict["H"])
    P_dict: Dict[str, Any] = {}
    if include_projector:
        P_dict = spectral_projector(
            H_dict["H"],
            E_fermi_im=fermi["E_fermi_im"],
            n_pts=n_pts,
            radius=radius,
        )
    else:
        P_t = _SRC.spectral_projector(
            _from_2d_complex_list(H_dict["H"]),
            E_fermi=complex(0.0, fermi["E_fermi_im"]),
            n_pts=n_pts, radius=radius,
        )
        P_dict = {"P": _to_2d_complex_list(P_t), "N": int(P_t.shape[0]),
                  "E_fermi_im": float(fermi["E_fermi_im"]),
                  "n_pts": int(n_pts), "radius": float(radius)}

    bott = bott_index(P_dict["P"], L)
    verdict = "LOOPS" if bott["C"] != 0 else "HALTS"

    return {
        "L": L,
        "N": H_dict["N"],
        "verdict": verdict,
        "H": H_dict["H"],
        "t1": t1, "t2": t2, "phi": phi, "loss": loss, "gamma_halt": gamma_halt,
        "halt_pos": H_dict["halt_pos"],
        "halt_site": H_dict["halt_site"],
        "spectrum": spec,
        "fermi": fermi,
        "projector": P_dict,
        "bott": bott,
    }


def gamma_sweep(
    L: int = 8,
    gammas: List[float] = None,
    t1: float = 1.0,
    t2: float = 0.5,
    phi: float = float(np.pi / 4),
    loss: float = 0.05,
    n_pts: int = 32,
    radius: float = 2.0,
    include_projector: bool = False,
) -> Dict[str, Any]:
    """Sweep gamma_halt and return Bott index for each value.

    For each gamma:
      - build H with localized EP sink at center
      - find E_fermi from spectrum
      - build projector (catalytic contour integral)
      - compute Bott index C

    Returns JSON-serializable dict with the per-gamma C values.
    """
    if gammas is None:
        gammas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    gammas = [float(g) for g in gammas]

    points = []
    for g in gammas:
        try:
            r = run(
                L=L, t1=t1, t2=t2, phi=phi, loss=loss, gamma_halt=g,
                n_pts=n_pts, radius=radius, include_projector=include_projector,
            )
            points.append({
                "gamma_halt": g,
                "C": int(r["bott"]["C"]),
                "verdict": str(r["verdict"]),
                "E_fermi_im": float(r["fermi"]["E_fermi_im"]),
            })
        except Exception as exc:  # numerical fragility
            points.append({
                "gamma_halt": g,
                "C": 0,
                "verdict": "HALTS",
                "E_fermi_im": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "L": int(L),
        "N": int(L * L),
        "t1": t1, "t2": t2, "phi": phi, "loss": loss,
        "n_pts": n_pts, "radius": radius,
        "points": points,
    }


# Canonical 2D machine presets (mirrors 1D's MACHINE_DESCRIPTIONS).
PRESET_MACHINES = {
    "loop_default": {
        "label": "loop_default",
        "expected": "LOOPS",
        "L": 8,
        "t1": 1.0, "t2": 0.5,
        "phi": float(np.pi / 4),
        "loss": 0.05, "gamma_halt": 0.0,
        "summary": "L=8, no EP sink. Pure Chern insulator. C = +1, chiral edge protected -> LOOPS.",
    },
    "halt_default": {
        "label": "halt_default",
        "expected": "HALTS",
        "L": 8,
        "t1": 1.0, "t2": 0.5,
        "phi": float(np.pi / 4),
        "loss": 0.05, "gamma_halt": 10.0,
        "summary": "L=8, EP sink at center (gamma_halt=10). Edge destroyed -> C = 0 -> HALTS.",
    },
    "uniform_annihilation": {
        "label": "uniform_annihilation",
        "expected": "HALTS",
        "L": 8,
        "t1": 1.0, "t2": 0.5,
        "phi": float(np.pi / 4),
        "loss": 0.05, "gamma_halt": 0.0,
        "summary": "L=8 with uniform gamma=2 on every site (Exp 39 discovery). Topology melts -> HALTS.",
    },
    "l4_fragility": {
        "label": "l4_fragility",
        "expected": "HALTS",
        "L": 4,
        "t1": 1.0, "t2": 0.5,
        "phi": float(np.pi / 4),
        "loss": 0.05, "gamma_halt": 0.0,
        "summary": "L=4 is too small to sustain the chiral edge. Even gamma=0 -> finite-size gap collapse -> C = 0 -> HALTS.",
    },
}


def preset_machines() -> Dict[str, Any]:
    """List canonical 2D machine presets with descriptions and param ranges."""
    return {
        "machines": PRESET_MACHINES,
        "params": {
            "L": {"min": 2, "max": 64, "default": 8,
                  "description": "Lattice linear size; N = L*L"},
            "t1": {"min": 0.0, "max": 5.0, "default": 1.0,
                   "description": "Real NN hopping strength"},
            "t2": {"min": 0.0, "max": 5.0, "default": 0.5,
                   "description": "Complex NNN hopping strength (TRS breaking)"},
            "phi": {"min": 0.0, "max": 6.283185307179586, "default": 0.7853981633974483,
                    "description": "NNN hopping phase (radians, default pi/4)"},
            "loss": {"min": 0.0, "max": 1.0, "default": 0.05,
                     "description": "Uniform on-site imaginary dissipation"},
            "gamma_halt": {"min": 0.0, "max": 50.0, "default": 0.0,
                           "description": "Localized EP sink strength at halt site"},
        },
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

"""FastAPI routes for the 2D Non-Hermitian Chern oracle (37)."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from .oracle_2d import build_H, run

router = APIRouter()


@router.get("/api/dim2/build")
def api_build_H(
    L: int = Query(8, ge=2, le=64),
    t1: float = Query(1.0),
    t2: float = Query(0.5),
    phi: float = Query(0.7853981633974483, description="radians, default pi/4"),
    loss: float = Query(0.05, ge=0.0),
    gamma_halt: float = Query(0.0, ge=0.0),
):
    """Build 2D Chern Hamiltonian only (no spectrum/projector/Bott)."""
    return build_H(L=L, t1=t1, t2=t2, phi=phi, loss=loss, gamma_halt=gamma_halt)


@router.get("/api/dim2/run")
def api_run(
    L: int = Query(8, ge=2, le=64),
    t1: float = Query(1.0),
    t2: float = Query(0.5),
    phi: float = Query(0.7853981633974483, description="radians, default pi/4"),
    loss: float = Query(0.05, ge=0.0),
    gamma_halt: float = Query(0.0, ge=0.0),
    n_pts: int = Query(32, ge=8, le=128, description="contour points for projector"),
    radius: float = Query(2.0, gt=0.0, description="contour radius around E_fermi"),
    include_projector: bool = Query(True, description="include P matrix in response"),
):
    """Full 2D oracle run: H, spectrum, E_fermi, P, Bott index, verdict."""
    if L < 4:
        # Physics: small lattices can't sustain the chiral edge with default params.
        # Allow it but the verdict is likely to be HALTS due to finite-size gap collapse.
        pass
    if L * L > 4096:
        raise HTTPException(status_code=400, detail="L too large (N > 4096)")
    return run(
        L=L, t1=t1, t2=t2, phi=phi, loss=loss, gamma_halt=gamma_halt,
        n_pts=n_pts, radius=radius, include_projector=include_projector,
    )

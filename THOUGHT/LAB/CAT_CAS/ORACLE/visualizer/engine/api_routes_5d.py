"""FastAPI routes for the 5D Floquet Time Crystal oracle (40)."""

from fastapi import APIRouter, HTTPException, Query
from typing import List

from .oracle_5d import build_H, floquet_operator, count_pi_modes, pi_mode_grid, gamma_sweep, run

router = APIRouter()


@router.get("/api/dim5/H")
def api_H(
    L: int = Query(4, ge=2, le=10),
    t1: float = Query(0.1, ge=0.0, le=2.0,
                       description="hopping; <=0.2 for pi-modes to survive"),
    loss: float = Query(0.01, ge=0.0),
    gamma: float = Query(0.0, ge=0.0),
):
    """Build the free Dirac Hamiltonian H0."""
    N = 4 * L * L
    if N > 1024:
        raise HTTPException(status_code=400, detail="L too large (N > 1024)")
    return build_H(L=L, t1=t1, loss=loss, gamma=gamma)


@router.get("/api/dim5/floquet")
def api_floquet(
    L: int = Query(4, ge=2, le=10),
    kz: float = Query(0.0),
    kw: float = Query(0.0),
    a: float = Query(1.5707963267948966, description="radians, default pi/2"),
    b: float = Query(1.5707963267948966),
    c: float = Query(1.5707963267948966),
    t1: float = Query(0.1),
    loss: float = Query(0.01, ge=0.0),
    g: float = Query(0.0, ge=0.0),
):
    """Build a single Floquet operator U_F(kz, kw)."""
    N = 4 * L * L
    if N > 1024:
        raise HTTPException(status_code=400, detail="L too large (N > 1024)")
    return floquet_operator(
        L=L, kz=kz, kw=kw, a=a, b=b, c=c, t1=t1, loss=loss, g=g,
    )


@router.get("/api/dim5/run")
def api_run(
    L: int = Query(4, ge=2, le=10),
    n_k: int = Query(4, ge=2, le=8),
    a: float = Query(1.5707963267948966),
    b: float = Query(1.5707963267948966),
    c: float = Query(1.5707963267948966),
    t1: float = Query(0.1),
    loss: float = Query(0.01, ge=0.0),
    g: float = Query(0.0, ge=0.0),
    threshold: float = Query(0.3, gt=0.0, le=1.0),
    include_U: bool = Query(False, description="include U matrix for central slice"),
):
    """Full 5D Floquet oracle run. Returns the pi-mode grid + verdict."""
    N = 4 * L * L
    if N > 1024:
        raise HTTPException(status_code=400, detail="L too large (N > 1024)")
    return run(
        L=L, n_k=n_k, a=a, b=b, c=c, t1=t1, loss=loss, g=g,
        threshold=threshold, include_U=include_U,
    )


@router.get("/api/dim5/gamma_sweep")
def api_gamma_sweep(
    L: int = Query(4, ge=2, le=10),
    n_k: int = Query(4, ge=2, le=8),
    gammas: List[float] = Query([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]),
    a: float = Query(1.5707963267948966),
    b: float = Query(1.5707963267948966),
    c: float = Query(1.5707963267948966),
    t1: float = Query(0.1),
    loss: float = Query(0.01, ge=0.0),
    threshold: float = Query(0.3, gt=0.0, le=1.0),
):
    """Gamma annihilation sweep for 5D. Per-gamma pi-mode total + verdict."""
    N = 4 * L * L
    if N > 1024:
        raise HTTPException(status_code=400, detail="L too large (N > 1024)")
    return gamma_sweep(
        L=L, n_k=n_k, gammas=gammas, a=a, b=b, c=c, t1=t1, loss=loss,
        threshold=threshold,
    )

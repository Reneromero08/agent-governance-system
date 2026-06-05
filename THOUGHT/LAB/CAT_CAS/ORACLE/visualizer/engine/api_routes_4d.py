"""FastAPI routes for the 4D Non-Hermitian Axion oracle (39)."""

from fastapi import APIRouter, HTTPException, Query
from typing import List

from .oracle_4d import build_slice, c1_grid, gamma_sweep, run

router = APIRouter()


@router.get("/api/dim4/slice")
def api_slice(
    L: int = Query(4, ge=2, le=8),
    kz: float = Query(0.0),
    kw: float = Query(0.0),
    t1: float = Query(1.0),
    tz: float = Query(1.0),
    tw: float = Query(1.0),
    m0: float = Query(1.0),
    loss: float = Query(0.05, ge=0.0),
    gamma_halt: float = Query(0.0, ge=0.0),
):
    """Build a single 4D Dirac slice H(kz, kw). N = 4*L*L."""
    N = 4 * L * L
    if N > 1024:
        raise HTTPException(status_code=400, detail="L too large (N > 1024)")
    return build_slice(
        L=L, kz=kz, kw=kw, t1=t1, tz=tz, tw=tw, m0=m0,
        loss=loss, gamma_halt=gamma_halt,
    )


@router.get("/api/dim4/run")
def api_run(
    L: int = Query(4, ge=2, le=8),
    n_k: int = Query(4, ge=2, le=8),
    t1: float = Query(1.0),
    tz: float = Query(1.0),
    tw: float = Query(1.0),
    m0: float = Query(1.0),
    loss: float = Query(0.05, ge=0.0),
    gamma_halt: float = Query(0.0, ge=0.0),
):
    """Full 4D oracle run: c1 grid over (kz, kw), C2, verdict."""
    N = 4 * L * L
    if N > 1024:
        raise HTTPException(status_code=400, detail="L too large (N > 1024)")
    return run(
        L=L, n_k=n_k, t1=t1, tz=tz, tw=tw, m0=m0,
        loss=loss, gamma_halt=gamma_halt,
    )


@router.get("/api/dim4/gamma_sweep")
def api_gamma_sweep(
    L: int = Query(4, ge=2, le=8),
    n_k: int = Query(4, ge=2, le=8),
    gammas: List[float] = Query([0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]),
    t1: float = Query(1.0),
    tz: float = Query(1.0),
    tw: float = Query(1.0),
    m0: float = Query(1.0),
    loss: float = Query(0.05, ge=0.0),
):
    """Gamma annihilation sweep for 4D. Per-gamma C2 + verdict."""
    N = 4 * L * L
    if N > 1024:
        raise HTTPException(status_code=400, detail="L too large (N > 1024)")
    return gamma_sweep(
        L=L, n_k=n_k, gammas=gammas, t1=t1, tz=tz, tw=tw, m0=m0,
        loss=loss,
    )

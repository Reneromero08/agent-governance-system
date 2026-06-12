"""FastAPI routes for the 3D Non-Hermitian Weyl Annihilation oracle (38)."""

from fastapi import APIRouter, HTTPException, Query
from typing import List

from .oracle_3d import build_slice, c1_profile, gamma_sweep, run

router = APIRouter()


@router.get("/api/dim3/slice")
def api_slice(
    L: int = Query(8, ge=2, le=32),
    kz: float = Query(0.0),
    t1: float = Query(1.0),
    t2: float = Query(0.5),
    phi: float = Query(0.7853981633974483),
    tz: float = Query(1.5),
    m0: float = Query(0.5),
    loss: float = Query(0.05, ge=0.0),
    gamma_halt: float = Query(0.0, ge=0.0),
):
    """Build a single 2D Weyl slice H(kz)."""
    if L * L > 4096:
        raise HTTPException(status_code=400, detail="L too large (N > 4096)")
    return build_slice(
        L=L, kz=kz, t1=t1, t2=t2, phi=phi, tz=tz, m0=m0,
        loss=loss, gamma_halt=gamma_halt,
    )


@router.get("/api/dim3/run")
def api_run(
    L: int = Query(8, ge=2, le=32),
    n_kz: int = Query(24, ge=2, le=128),
    t1: float = Query(1.0),
    t2: float = Query(0.5),
    phi: float = Query(0.7853981633974483),
    tz: float = Query(1.5),
    m0: float = Query(0.5),
    loss: float = Query(0.05, ge=0.0),
    gamma_halt: float = Query(0.0, ge=0.0),
    n_pts: int = Query(32, ge=8, le=128),
    radius: float = Query(2.0, gt=0.0),
):
    """Full 3D oracle run: c1 profile over kz, max_C, nonzero count, verdict."""
    if L * L > 4096:
        raise HTTPException(status_code=400, detail="L too large (N > 4096)")
    return run(
        L=L, n_kz=n_kz, t1=t1, t2=t2, phi=phi, tz=tz, m0=m0,
        loss=loss, gamma_halt=gamma_halt, n_pts=n_pts, radius=radius,
    )


@router.get("/api/dim3/gamma_sweep")
def api_gamma_sweep(
    L: int = Query(8, ge=2, le=32),
    n_kz: int = Query(16, ge=2, le=128),
    gammas: List[float] = Query([0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]),
    t1: float = Query(1.0),
    t2: float = Query(0.5),
    phi: float = Query(0.7853981633974483),
    tz: float = Query(1.5),
    m0: float = Query(0.5),
    loss: float = Query(0.05, ge=0.0),
    n_pts: int = Query(32, ge=8, le=128),
    radius: float = Query(2.0, gt=0.0),
):
    """Gamma annihilation sweep. Returns per-gamma max_C and verdict."""
    if L * L > 4096:
        raise HTTPException(status_code=400, detail="L too large (N > 4096)")
    return gamma_sweep(
        L=L, n_kz=n_kz, gammas=gammas, t1=t1, t2=t2, phi=phi, tz=tz, m0=m0,
        loss=loss, n_pts=n_pts, radius=radius,
    )

"""FastAPI routes for the 2D Non-Hermitian Chern oracle (37)."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from .oracle_2d import build_H, run, gamma_sweep, preset_machines

router = APIRouter()


@router.get("/api/dim2/machines")
def api_machines():
    """List canonical 2D machine presets with descriptions and param ranges."""
    return preset_machines()


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
    if L * L > 4096:
        raise HTTPException(status_code=400, detail="L too large (N > 4096)")
    return run(
        L=L, t1=t1, t2=t2, phi=phi, loss=loss, gamma_halt=gamma_halt,
        n_pts=n_pts, radius=radius, include_projector=include_projector,
    )


@router.get("/api/dim2/gamma_sweep")
def api_gamma_sweep(
    L: int = Query(8, ge=2, le=64),
    gammas: str = Query("0,0.5,1,2,5,10", description="comma-separated gamma values"),
    t1: float = Query(1.0),
    t2: float = Query(0.5),
    phi: float = Query(0.7853981633974483, description="radians, default pi/4"),
    loss: float = Query(0.05, ge=0.0),
    n_pts: int = Query(32, ge=8, le=128, description="contour points for projector"),
    radius: float = Query(2.0, gt=0.0, description="contour radius around E_fermi"),
    include_projector: bool = Query(False, description="include P matrix in each response"),
):
    """Bott index C vs gamma_halt curve. Returns per-gamma C, verdict, E_fermi."""
    if L * L > 4096:
        raise HTTPException(status_code=400, detail="L too large (N > 4096)")
    try:
        gammas_list: List[float] = [float(x) for x in gammas.split(",") if x.strip()]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad gammas string: {e}")
    if len(gammas_list) == 0:
        raise HTTPException(status_code=400, detail="Empty gammas list")
    if len(gammas_list) > 32:
        raise HTTPException(status_code=400, detail="Too many gammas (max 32)")
    return gamma_sweep(
        L=L, gammas=gammas_list, t1=t1, t2=t2, phi=phi, loss=loss,
        n_pts=n_pts, radius=radius, include_projector=include_projector,
    )

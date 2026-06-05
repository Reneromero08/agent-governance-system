"""FastAPI routes for the 1D Non-Hermitian Halting Oracle."""

from fastapi import APIRouter, HTTPException, Query

from engine import oracle_1d


router = APIRouter(prefix="/api/dim1", tags=["dim1"])


@router.get("/run")
async def run(
    machine: str = Query(
        "halt_direct",
        description="Test machine: halt_direct | halt_chain | loop_2cycle | loop_3cycle",
    ),
    gamma: float = Query(1.0, ge=0.0, le=10.0, description="Transition coupling"),
    loss_rate: float = Query(0.1, ge=0.0, le=1.0, description="On-site dissipation"),
    halt_mult: float = Query(10.0, ge=1.0, le=50.0, description="Halt-state sink multiplier"),
    n_phi: int = Query(400, ge=10, le=2000, description="Twist sweep samples"),
):
    """Run the 1D Non-Hermitian Halting Oracle.

    Returns the Hamiltonian, spectrum, det(H(phi)) trace, winding, and verdict.
    """
    try:
        return oracle_1d.run(
            machine=machine,
            gamma=gamma,
            loss_rate=loss_rate,
            halt_mult=halt_mult,
            n_phi=n_phi,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

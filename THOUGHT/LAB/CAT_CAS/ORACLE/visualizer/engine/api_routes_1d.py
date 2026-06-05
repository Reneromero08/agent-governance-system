"""FastAPI routes for the 1D Non-Hermitian Halting Oracle."""

from fastapi import APIRouter, HTTPException, Query

from engine import oracle_1d


router = APIRouter(prefix="/api/dim1", tags=["dim1"])


# Machine descriptions (mirrors the test machines in 36_nonhermitian_oracle.py).
MACHINE_DESCRIPTIONS = {
    "halt_direct": {
        "label": "halt_direct",
        "expected": "HALTS",
        "summary": "1 state, 1 bit. Direct sink. W = 0 (no winding).",
    },
    "halt_chain": {
        "label": "halt_chain",
        "expected": "HALTS",
        "summary": "1 state, 1 bit. Chain of 3 self-loops into halt. W = 0.",
    },
    "loop_2cycle": {
        "label": "loop_2cycle",
        "expected": "LOOPS",
        "summary": "2 states, 1 bit. Bounces s0<->s1 forever. W = +1.",
    },
    "loop_3cycle": {
        "label": "loop_3cycle",
        "expected": "LOOPS",
        "summary": "3 states, 1 bit. Cycles s0->s1->s2->s0. W = +1.",
    },
}


@router.get("/machines")
async def machines():
    """List available test machines with descriptions."""
    return {
        "machines": MACHINE_DESCRIPTIONS,
        "params": {
            "gamma": {"min": 0.0, "max": 10.0, "default": 1.0,
                       "description": "Transition coupling strength"},
            "loss_rate": {"min": 0.0, "max": 1.0, "default": 0.1,
                          "description": "On-site dissipation"},
            "halt_mult": {"min": 1.0, "max": 50.0, "default": 10.0,
                          "description": "Halt-state sink multiplier"},
            "n_phi": {"min": 10, "max": 2000, "default": 400,
                      "description": "Twist sweep samples"},
        },
    }


@router.get("/build")
async def build(
    machine: str = Query("halt_direct"),
    gamma: float = Query(1.0, ge=0.0, le=10.0),
    loss_rate: float = Query(0.1, ge=0.0, le=1.0),
    halt_mult: float = Query(10.0, ge=1.0, le=50.0),
):
    """Build H for a machine (cheaper than /run, no spectrum or winding)."""
    try:
        return oracle_1d.build_H(
            machine=machine,
            gamma=gamma,
            loss_rate=loss_rate,
            halt_mult=halt_mult,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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

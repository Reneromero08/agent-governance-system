"""
CAT_CAS Oracle Visualizer — server.

Phase 0: Foundation. FastAPI app with static frontend and health endpoint.

Run: python server.py
Then open http://localhost:8000
"""

import os
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ---- Paths ---------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(HERE, "frontend")
CAT_CAS_DIR = os.path.normpath(os.path.join(HERE, "..", ".."))

# Make CAT_CAS importable so engine/ can `from 35_topological_halting_oracle...`
sys.path.insert(0, CAT_CAS_DIR)

# Bump this as phases land.
PHASE = 0


# ---- App -----------------------------------------------------------------

app = FastAPI(title="CAT_CAS Oracle Visualizer")


@app.get("/api/health")
async def health():
    """Health check. Frontend uses this to show connection status."""
    return {
        "status": "ok",
        "phase": PHASE,
        "python": sys.version.split()[0],
        "cat_cas_path": CAT_CAS_DIR,
    }


# ---- Phase 1 routes: dim1 run stub (full body added in Phase 1) ---------

@app.get("/api/dim1/run")
async def dim1_run_stub():
    """Stub. Phase 1 will wire this to engine.oracle_1d.run()."""
    return {
        "phase": 1,
        "status": "stub",
        "message": "engine not yet implemented",
    }


# ---- Static frontend -----------------------------------------------------

# Mount the whole frontend/ at /static so paths like /static/css/... work.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ---- Entrypoint ----------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print(f"  CAT_CAS Oracle Visualizer")
    print(f"  CAT_CAS path: {CAT_CAS_DIR}")
    print(f"  Frontend:     {FRONTEND_DIR}")
    print(f"  Phase:        {PHASE}")
    print(f"  Open:         http://localhost:8000")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

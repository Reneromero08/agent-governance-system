# CAT_CAS Oracle Visualizer

Interactive visualizer for the CAT_CAS Oracle pattern across dimensions 1-5.

## Architecture

- **Backend** — Python (FastAPI) wraps the actual CAT_CAS source modules (35.2, 37, 38, 39, 40). No math re-implementation.
- **Frontend** — ES6 modules + Canvas. No bundler, no build step.
- **Transport** — JSON over HTTP.

```
visualizer/
+-- server.py              FastAPI app
+-- requirements.txt
+-- engine/                Python -- wraps CAT_CAS source
+-- frontend/              ES6 modules + Canvas
|   +-- index.html
|   +-- css/
|   +-- js/
+-- tests/
+-- ROADMAP.md
```

## Run

```bash
# From the visualizer/ folder, using the repo's .venv:
..\..\..\..\..\..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py
```

Then open `http://localhost:8000` in a browser.

The health endpoint is at `http://localhost:8000/api/health`.

## Status

See `ROADMAP.md` for the full phase-by-phase plan. Current: **Phase 0 -- Foundation**.

## Phase 0 -- what's here

- FastAPI server with `/api/health` endpoint
- Static frontend at `/` with 5 tabs (1D active, others disabled)
- Tab switching via `frontend/js/main.js`
- CSS tokens in `frontend/css/base.css` (colors, fonts, spacing)

## Phase 1 -- engine (next)

Wraps the actual CAT_CAS Python modules. Each dimension gets:
- `engine/oracle_Nd.py` -- Hamiltonian + invariant + run()
- `engine/api_routes_Nd.py` -- FastAPI route
- Smoke test in `tests/smoke.py` that verifies engine output matches the source's `main()` prints

# CAT_CAS Oracle Visualizer — Roadmap

**Architecture**: Python (FastAPI) backend + JS frontend, no build step.
**Run**: `python server.py` opens on `http://localhost:8000`.
**Faithfulness rule**: Engine wraps the actual CAT_CAS Python modules. No math re-implementation.

---

## Phase 0 — Foundation

**Goal**: Skeleton that runs. Proves the wiring.

- [x] Create `THOUGHT/LAB/CAT_CAS/ORACLE/visualizer/` folder
- [x] Create `engine/`, `frontend/`, `tests/` subfolders
- [x] Write `server.py` — FastAPI app with StaticFiles mount
- [x] Add `/api/health` endpoint (returns `{"status": "ok", "phase": 0}`)
- [x] Write `requirements.txt` (fastapi, uvicorn, torch, numpy)
- [x] Write `frontend/index.html` — 5-tab shell (1D, 2D, 3D, 4D, 5D), only dim1 enabled
- [x] Write `frontend/css/base.css` — reset, fonts, color tokens
- [x] Write `frontend/css/layout.css` — header, tab bar, main grid
- [x] Write `frontend/css/panels.css` — side panel + canvas grid
- [x] Write `frontend/css/viz.css` — base canvas styles, KPI tiles
- [x] Write `frontend/js/main.js` — tab switching logic, fetch health on load
- [x] Write `frontend/js/api.js` — REST client stub
- [x] Write `README.md` — install, run, architecture diagram

**Verify**:
- [x] `python server.py` starts without error
- [x] Browser at `http://localhost:8000` shows 5 tabs
- [x] `curl http://localhost:8000/api/health` returns `{"status": "ok", "phase": 0}`
- [x] Only "1D" tab is interactive; others are disabled

---

## Phase 1 — Engine wrapper (Python)

**Goal**: Each dimension callable as a uniform function. 1:1 with the source.

### 1A: 1D engine (35.2)
- [ ] Write `engine/__init__.py`
- [ ] Write `engine/serialize.py` — `torch.Tensor` → list of `{re, im}`; `np.ndarray` → list
- [ ] Write `engine/oracle_1d.py`:
  - [ ] `build_H(machine, gamma, loss_rate, halt_mult) -> {H: list, labels: list, halt_mask: list, twist_indices: list}`
  - [ ] `get_spectrum(H) -> {eigvals: [{re, im}], kappa_V: float}`
  - [ ] `point_gap_winding(H, twist_indices, E_ref, n_phi) -> {Wraw, Wint, det_curve: [{re, abs}]}`
  - [ ] `run(machine, params) -> {H, spectrum, winding, verdict, ...}`
  - [ ] Import `build_nonhermitian_H`, `point_gap_winding`, `get_spectral_data` from `36_nonhermitian_oracle.py`
- [ ] Write `engine/api_routes_1d.py` — FastAPI route `/api/dim1/run?machine=...&gamma=...&...`

### 1B: 2D engine (37)
- [ ] Write `engine/oracle_2d.py`:
  - [ ] `build_H(L, t1, t2, phi, loss, gamma_halt) -> {H, ...}` (L is a parameter!)
  - [ ] `find_fermi(H) -> E_fermi` (largest Im gap, midpoint)
  - [ ] `spectral_projector(H, E_fermi, n_pts, radius) -> P`
  - [ ] `bott_index(P, L) -> C`
  - [ ] `run(L, gamma_halt, ...) -> {H, eigvals, fermi, P, C, verdict}`
  - [ ] Import from `37_2d_chern_oracle.py`
- [ ] Write `engine/api_routes_2d.py` — `/api/dim2/run?L=...&gamma_halt=...`

### 1C: 3D engine (38)
- [ ] Write `engine/oracle_3d.py`:
  - [ ] `build_slice(L, kz, t1, t2, phi, tz, m0, loss, gamma_halt) -> H`
  - [ ] `c1_profile(L, n_kz, gamma_halt) -> [{kz, M, C}]`
  - [ ] `uniform_gamma_sweep(L, n_kz, gammas) -> [{Gamma, maxC, nonzero}]`
  - [ ] `run(L, n_kz, gamma_halt) -> {profile, maxC, nonzero, verdict}`
  - [ ] Import from `38_3d_weyl_oracle.py`
- [ ] Write `engine/api_routes_3d.py` — `/api/dim3/run?L=...&n_kz=...&gamma_halt=...`

### 1D: 4D engine (39)
- [ ] Write `engine/oracle_4d.py`:
  - [ ] `build_slice(L, kz, kw, t1, tz, tw, m0, loss, gamma_halt) -> H`
  - [ ] `c1_grid(L, n_k, gamma_halt) -> [[C1 at (kz,kw)]]`
  - [ ] `second_chern(c1_grid) -> C2`
  - [ ] `run(L, n_k, gamma_halt) -> {c1_grid, C2, verdict}`
  - [ ] Import from `39_4d_axion_oracle.py`
- [ ] Write `engine/api_routes_4d.py` — `/api/dim4/run?L=...&n_k=...&gamma_halt=...`

### 1E: 5D engine (40)
- [ ] Write `engine/oracle_5d.py`:
  - [ ] `floquet_operator(L, kz, kw, alpha, beta, gamma, t1, loss, g) -> U`
  - [ ] `count_pi_modes(U, threshold) -> n`
  - [ ] `pi_mode_grid(L, n_k, g) -> [[n at (kz,kw)]]`
  - [ ] `run(L, n_k, t1, loss, g) -> {pi_grid, total, active, verdict}`
  - [ ] Import from `40_5d_floquet_oracle.py`
- [ ] Write `engine/api_routes_5d.py` — `/api/dim5/run?L=...&n_k=...&g=...`

### 1F: Smoke tests
- [ ] Write `tests/smoke.py`:
  - [ ] For 1D: verify all 4 machines return correct W and verdict
  - [ ] For 2D: L=8, gamma_halt=0 → C_loop=+1; gamma_halt=10 → C_halt=0
  - [ ] For 3D: gamma_halt=0 → maxC>0; gamma_halt=15 → maxC=0
  - [ ] For 4D: gamma_halt=0 → C2≠0; gamma_halt=15 → C2=0
  - [ ] For 5D: g=0 → 32 pi-modes/slice; g=0.5 → 0
- [ ] All tests pass against the actual lab source outputs

**Verify**:
- `python tests/smoke.py` exits 0
- Each `/api/dim{N}/run` returns the expected verdict
- Engine does not modify the source CAT_CAS files (read-only imports)

---

## Phase 2 — 1D view (mechanism)

**Goal**: Show WHY halting works, not just that it does.

- [ ] `frontend/js/views/graph_view.js`:
  - [ ] Render directed graph: nodes = states, edges = transitions
  - [ ] Halt node = red, animated pulse
  - [ ] Edge thickness = transition magnitude (gamma)
  - [ ] Click a node to set as |psi0>
- [ ] `frontend/js/views/probability_flow.js`:
  - [ ] Animate |psi(t+dt) = exp(-i*H*dt) |psi(t)>
  - [ ] Use small dt, e.g., 0.05, with 60 fps
  - [ ] Node size = sqrt(|psi_i|^2)
  - [ ] Edge arrow opacity = |psi_source|^2 * |H_edge| * dt
  - [ ] Halt machine: probability visibly drains into halt node
  - [ ] Loop machine: probability circulates indefinitely
- [ ] `frontend/js/views/spectrum.js`:
  - [ ] Scatter eigenvalues in complex plane (Re vs Im)
  - [ ] Halt eigenvalue: large red dot
  - [ ] Other eigenvalues: cyan dots
  - [ ] Axes labeled, grid lines
- [ ] `frontend/js/views/det_trace.js`:
  - [ ] Plot det(H(phi)) as phi sweeps [0, 2π]
  - [ ] Mark current phi with gold dot
  - [ ] W_raw / W_int displayed
  - [ ] Verdict (HALTS/LOOPS) shown
- [ ] `frontend/js/tabs/dim1.js`:
  - [ ] Machine picker (halt_direct, halt_chain, loop_2cycle, loop_3cycle)
  - [ ] Sliders: gamma (0.1-3), loss_rate (0.01-0.5), halt_mult (2-50), n_phi (60-800)
  - [ ] Twist mode toggle (boundary / global)
  - [ ] "PLAY" / "PAUSE" button for time evolution
- [ ] `frontend/css/viz.css` — animate classes for halt pulse, flow particles

**Verify**:
- Open `http://localhost:8000/`, 1D tab active
- `halt_direct`: press PLAY → see probability flow from s0 to halt (red), halt node grows
- `loop_2cycle`: press PLAY → see probability circulate between s0 and s1 forever
- Verdict: HALTS for halt machines, LOOPS for loops
- Winding W=0 for halt (det is single point), W=±1 for loops (circle)

---

## Phase 3 — 2D view (the actual torus)

**Goal**: Show the (kx, ky) torus and Bott index.

- [ ] `frontend/js/views/lattice_view.js`:
  - [ ] L×L grid of nodes
  - [ ] Halt site = red, others = grey
  - [ ] Edges = NNN complex hopping (arrows showing direction)
  - [ ] Periodic boundary indicators (dashed)
- [ ] `frontend/js/views/torus_view.js`:
  - [ ] Heatmap of C(kx, ky) over [0, 2π] × [0, 2π]
  - [ ] C = +1 → red, C = -1 → blue, C = 0 → grey
  - [ ] Show actual torus topology (or flag as 2D + periodic)
- [ ] `frontend/js/views/spectrum_2d.js`:
  - [ ] Eigenvalue scatter for given L, gamma_halt
  - [ ] Highlight Im gap (Fermi level)
- [ ] `frontend/js/tabs/dim2.js`:
  - [ ] L slider (4-12)
  - [ ] gamma_halt slider (0-15)
  - [ ] t1, t2, phi, loss sliders
  - [ ] "compute Bott" button
  - [ ] Show C value
- [ ] Uniform gamma sweep widget (per Exp 39 discovery):
  - [ ] Show curve: max|C| vs Gamma
  - [ ] Mark annihilation threshold

**Verify**:
- L=8, gamma_halt=0 → C=+1 (LOOPS)
- L=8, gamma_halt=10 → C=0 (HALTS)
- L=6, gamma_halt=5, uniform Gamma=2 → C=0 (ANNIHILATED)
- Matches `37_2d_chern_oracle.py` outputs

---

## Phase 4 — 3D view (Weyl)

**Goal**: Show kz-stack of Chern slices + Weyl node annihilation.

- [ ] `frontend/js/views/kz_profile.js`:
  - [ ] C(kz) line plot for kz ∈ [0, 2π]
  - [ ] Mark Weyl nodes at M(kz) = m0 - tz*cos(kz) = 0
  - [ ] Color: nonzero slices in cyan, zero in grey
- [ ] `frontend/js/views/slice_view.js`:
  - [ ] Drag kz slider → render 2D slice
  - [ ] Reuse lattice_view.js from Phase 3
- [ ] `frontend/js/tabs/dim3.js`:
  - [ ] L slider (4-10)
  - [ ] n_kz slider (8-48)
  - [ ] m0, tz sliders
  - [ ] gamma_halt slider (single-site sink)
  - [ ] "uniform gamma" toggle + slider (Exp 39 mode)
- [ ] `frontend/js/views/gamma_sweep_3d.js`:
  - [ ] Plot max|C| vs Gamma for uniform mode
  - [ ] Show annihilation threshold

**Verify**:
- M(kz) zeros match the marked Weyl nodes
- gamma_halt=0 → max|C|>0, nonzero slices > 0
- gamma_halt=15 → max|C|=0 (single-site sink insufficient per source)
- uniform Gamma=2 → max|C|=0 (annihilated per Exp 39)
- Matches `38_3d_weyl_oracle.py`

---

## Phase 5 — 4D view (Axion)

**Goal**: Show nested reduction (kz, kw) → C2.

- [ ] `frontend/js/views/kzkw_grid.js`:
  - [ ] 2D heatmap of C1 over (kz, kw) ∈ [0, 2π]²
  - [ ] Color: +1 red, -1 blue, 0 grey
  - [ ] Slider for n_k (resolution)
- [ ] `frontend/js/views/c2_readout.js`:
  - [ ] Show C2 = sum(C1) / n_k² (rounded)
  - [ ] Show C1 non-zero count
- [ ] `frontend/js/tabs/dim4.js`:
  - [ ] L slider (4-6)
  - [ ] n_k slider (4-10)
  - [ ] m0, tz, tw sliders
  - [ ] gamma_halt slider
- [ ] `frontend/js/views/gamma_sweep_4d.js`:
  - [ ] Plot C2 vs Gamma
  - [ ] Mark destruction threshold

**Verify**:
- gamma_halt=0 → C2 ≠ 0
- gamma_halt=15 → C2 = 0
- Matches `39_4d_axion_oracle.py` outputs

---

## Phase 6 — 5D view (Floquet)

**Goal**: Show quasi-energy spectrum + pi-mode count.

- [ ] `frontend/js/views/floquet_spectrum.js`:
  - [ ] Scatter eigvals(U_F) on complex unit circle
  - [ ] Color by |z+1|: inside threshold (pi-mode) = gold, outside = cyan
  - [ ] Dashed circle at radius `threshold` around z=-1
- [ ] `frontend/js/views/pi_mode_count.js`:
  - [ ] Show total pi-modes, active slices
  - [ ] Bar chart per (kz, kw) slice
- [ ] `frontend/js/tabs/dim5.js`:
  - [ ] L slider (4-6)
  - [ ] n_k slider (4-8)
  - [ ] t1, loss, gamma sliders
  - [ ] threshold slider (default 0.3)
- [ ] Floquet operator inspector:
  - [ ] Show |U_F - I|, |U_F + I| operator norms

**Verify**:
- g=0 → 32 pi-modes/slice, all 16 slices active (LOOPS)
- g=0.5 → 0 pi-modes (HALTS / melted)
- Matches `40_5d_floquet_oracle.py` "SOLVED" output

---

## Phase 7 — Polish

- [ ] Cross-dim comparison view:
  - [ ] Side-by-side: 1D state graph, 2D (kx,ky) torus, 3D kz profile, 4D C2, 5D pi count
  - [ ] "Verdict matrix" — table of all dims × all machines
- [ ] Export run (JSON):
  - [ ] "Save run" button → downloads `{params, results, timestamp, hash}`
  - [ ] Hash via SHA-256 of canonicalized params
- [ ] Catalytic tape proof (per-oracle):
  - [ ] For 2D, run with `catalytic_oracle_37()` and show SHA-256 pre/post
- [ ] Doc links: each view has a "source" link to the relevant Python file
- [ ] Help overlay: keyboard shortcuts, mouse controls

**Verify**:
- All dims render simultaneously without lag
- Export/import round-trips a run
- Catalytic tape shows pre_hash == post_hash

---

## Lattice size defaults

| Dim | Parameter | Min | Max | Default | Source |
|---|---|---|---|---|---|
| 1D | n/a (TM) | — | — | — | 35.2 |
| 2D | L | 4 | 12 | 6 | 37 |
| 3D | L, n_kz | 4, 8 | 10, 48 | 6, 24 | 38 |
| 4D | L, n_k | 4, 4 | 6, 10 | 4, 6 | 39 |
| 5D | L, n_k | 4, 4 | 6, 8 | 4, 4 | 40 |

---

## Conventions

- **Engine imports from CAT_CAS source, doesn't re-implement**
- **Lattice sizes are parameters, modifiable from UI**
- **No silent state**: every computation logged with params + hash
- **JSON over HTTP**: `/api/dim{N}/run` returns full result
- **Frontend is ES6 modules, no bundler**

---

## Status legend

- [ ] not started
- [x] done
- [-] in progress
- [?] blocked / question

**Current phase**: Phase 1 — Engine wrapper

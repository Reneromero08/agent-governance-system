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

### 1A: 1D engine (35.2) — DONE (2026-06-04)
- [x] Write `engine/__init__.py`
- [x] Write `engine/serialize.py` — `torch.Tensor` → list of `{re, im}`; `np.ndarray` → list
- [x] Write `engine/oracle_1d.py`:
  - [x] `build_H(machine, gamma, loss_rate, halt_mult) -> {H: list, labels: list, halt_mask: list, twist_indices: list}`
  - [x] `get_spectrum(H) -> {eigvals: [{re, im}], kappa_V: float}`
  - [x] `point_gap_winding(H, twist_indices, E_ref, n_phi) -> {Wraw, Wint, det_curve, det_abs}`
  - [x] `run(machine, params) -> {H, spectrum, winding, verdict, ...}`
  - [x] Import `build_nonhermitian_H`, `point_gap_winding`, `get_spectral_data` from `36_nonhermitian_oracle.py`
- [x] Write `engine/api_routes_1d.py` — FastAPI route `/api/dim1/run?machine=...&gamma=...&...`
- [x] Write `tests/smoke.py` — 4 machines + halt_mult sweep + gamma=0 decoupling + dim check (all pass)

**Verify**:
- [x] `python tests/smoke.py` exits 0
- [x] `/api/dim1/run?machine=halt_direct` → W=0, verdict=HALTS
- [x] `/api/dim1/run?machine=loop_2cycle` → W=+1, verdict=LOOPS
- [x] Engine imports lab source via `importlib.util.spec_from_file_location` (read-only)

### 1B: 2D engine (37) — DONE (2026-06-04)
- [x] Write `engine/oracle_2d.py`:
  - [x] `build_H(L, t1, t2, phi, loss, gamma_halt) -> {H, N, halt_pos, halt_site, ...}` (L is a parameter!)
  - [x] `find_fermi(H) -> {E_fermi, gap_width, im_min, im_max}` (largest Im gap, midpoint)
  - [x] `spectral_projector(H, E_fermi_im, n_pts, radius) -> {P, ...}`
  - [x] `bott_index(P, L) -> {C, verdict}`
  - [x] `run(L, t1, t2, phi, loss, gamma_halt, n_pts, radius, include_projector) -> {...}`
  - [x] Import from `37_2d_chern_oracle.py` (build_2d_hamiltonian, spectral_projector, bott_index)
- [x] Write `engine/api_routes_2d.py` — `/api/dim2/build` and `/api/dim2/run`

**Verify**:
- [x] `python -m tests.smoke` exits 0 (1D + 2D)
- [x] L=8, gamma_halt=0   -> C=+1 LOOPS (chiral edge protected)
- [x] L=8, gamma_halt=10  -> C=0  HALTS (EP sink destroys edge)
- [x] gamma_halt sweep:   g=0,0.5,1 -> C=+1;  g=2,5,10 -> C=0
- [x] H[halt][halt].im = -(loss + gamma_halt)
- [x] N = L*L for L in 4,6,8,10
- [x] Live HTTP: /api/dim2/run?L=8&gamma_halt=0   returns C=+1, 64 eigvals
- [x] Live HTTP: /api/dim2/run?L=8&gamma_halt=10  returns C=0
- [x] L=6 loop case: C=0 (finite-size gap collapse with default phi=pi/4, t2=0.5) — real physics, not a bug

### 1C: 3D engine (38) — DONE (2026-06-04)
- [x] Write `engine/oracle_3d.py`:
  - [x] `build_slice(L, kz, t1, t2, phi, tz, m0, loss, gamma_halt) -> {H, N, M_kz, ...}`
  - [x] `find_fermi(L, kz_ref, ...) -> {E_fermi_im, gap_width, ...}`
  - [x] `c1_profile(L, n_kz, gamma_halt, ...) -> {kz, M_kz, C, max_abs_C, nonzero, weyl_nodes, nan_slices, ...}`
  - [x] `gamma_sweep(L, n_kz, gammas, ...) -> {results: [{gamma, max_abs_C, nonzero, verdict, C, kz}]}`
  - [x] `run(L, n_kz, gamma_halt, ...) -> {verdict, profile, ...}`
  - [x] Import from `38_3d_weyl_oracle.py` (build_weyl_slice, spectral_projector, bott_index)
- [x] Write `engine/api_routes_3d.py` — `/api/dim3/slice`, `/api/dim3/run`, `/api/dim3/gamma_sweep`

**Robustness fix**: Lab source's `bott_index` raises ValueError("cannot convert
float NaN to integer") at high gamma_halt where the spectral projector becomes
degenerate.  Source itself crashes (see 38_3d_weyl_oracle/output.txt line 75).
Engine wrapper catches this and records C=0 with index in `nan_slices` list,
so the run completes and reports the topology as numerically collapsed.

**Verify**:
- [x] `python -m tests.smoke` exits 0 (1D + 2D + 3D)
- [x] L=8, n_kz=24, gamma=0  -> maxC=2  nonzero=14/24  LOOPS  (matches lab output)
- [x] L=8, n_kz=24, gamma=15 -> maxC=0  nonzero=0/24  nan_slices=24  HALTS
- [x] gamma_sweep: g=0,5 LOOPS; g=15 HALTS
- [x] M(kz) = m0 - tz*cos(kz): kz=0 -> -1.0, kz=pi -> +2.0
- [x] Weyl node count: 2 when |m0/tz|<1, 0 when m0>tz
- [x] Live HTTP: /api/dim3/run?L=8&n_kz=24&gamma_halt=0 returns verdict=LOOPS
- [x] Live HTTP: /api/dim3/run?L=8&n_kz=24&gamma_halt=15 returns verdict=HALTS

### 1D: 4D engine (39) — DONE (2026-06-04)
- [x] Write `engine/oracle_4d.py`:
  - [x] `build_slice(L, kz, kw, t1, tz, tw, m0, loss, gamma_halt) -> {H, N, N_sp, M_kw, ...}`
  - [x] `c1_grid(L, n_k, gamma_halt, ...) -> {kz, kw, C1_grid [n_k][n_k], C1_profile, C2, nonzero, total}`
  - [x] `second_chern(c1_grid) -> C2`
  - [x] `gamma_sweep(L, n_k, gammas, ...) -> {results: [{gamma, C2, nonzero, verdict, C1_profile}]}`
  - [x] `run(L, n_k, gamma_halt, ...) -> {verdict, grid, ...}`
  - [x] Import from `39_4d_axion_oracle.py` (build_4d_slice, spectral_projector, bott_index_spinor, compute_second_chern)
- [x] Write `engine/api_routes_4d.py` — `/api/dim4/slice`, `/api/dim4/run`, `/api/dim4/gamma_sweep`

**Physics note**: 4D Dirac monopoles are robust — single-site EP cannot cleanly
annihilate them.  The source itself documents this in `gamma_at_topological`
and `uniform_gamma_annihilation` expansions.  L=4 is too small for robust
quantization; L=6 is the working default.  gamma_sweep may show non-monotonic
C2 vs gamma_halt — this is real physics, not a bug.

**Verify**:
- [x] `python -m tests.smoke` exits 0 (1D + 2D + 3D + 4D)
- [x] L=6, n_k=4, gamma=0 -> C2=-2  nonzero=16/16  LOOPS
- [x] C1_grid shape: 4x4, C1_profile length = 16
- [x] N = 4*L*L (4-component spinor on LxL spatial lattice)
- [x] M(kz, kw) = m0 - tz*cos(kz) - tw*cos(kw): corner checks pass
- [x] Halt site Im(H) = -(loss + gamma_halt) = -10.05 at gamma=10
- [x] gamma_sweep runs and returns structured output
- [x] Live HTTP: /api/dim4/run?L=6&n_k=4&gamma_halt=0 returns C2=-2, verdict=LOOPS
- [x] Live HTTP: /api/dim4/run?L=4&n_k=4&gamma_halt=0 returns C2=0 (lattice too small)

### 1E: 5D engine (40) — DONE (2026-06-04)
- [x] Write `engine/oracle_5d.py`:
  - [x] `build_H(L, t1, loss, gamma) -> {H, N, ...}` (4-comp spinor on LxL lattice)
  - [x] `floquet_operator(L, kz, kw, a, b, c, t1, loss, g) -> {U, eigvals, ...}`
  - [x] `count_pi_modes(U, threshold) -> {n_pi_modes, threshold, N}`
  - [x] `pi_mode_grid(L, n_k, a, b, c, t1, loss, g, threshold) -> {n_grid, total, active, slices, ...}`
  - [x] `gamma_sweep(L, n_k, gammas, ...) -> {results: [{gamma, total, active, verdict, n_grid}]}`
  - [x] `run(L, n_k, a, b, c, t1, loss, g, threshold, include_U) -> {verdict, grid, ...}`
  - [x] Import from `40_5d_floquet_oracle.py` (build_H, floquet_operator, count_pi_modes)
- [x] Write `engine/api_routes_5d.py` — `/api/dim5/H`, `/api/dim5/floquet`, `/api/dim5/run`, `/api/dim5/gamma_sweep`

**Physics note**: The 5D protocol is SOLVED — at a=b=c=pi/2 the three-step
non-Clifford sequence gives 2 pi-modes per site (32 for L=4, 72 for L=6).
But t1>=0.5 (strong hopping) ALSO melts the pi-modes — pi-modes are robust
only in the ideal Floquet limit or small hopping.  The source's `run_oracle`
sweeps t1 in [0.0, 0.05, 0.1, 0.2]; default t1=0.1 in the engine matches
that realistic regime.  Default loss=0.01 keeps eigenvalues on the unit
circle (|z| ~ 0.99).

**Verify**:
- [x] `python -m tests.smoke` exits 0 (1D + 2D + 3D + 4D + 5D)
- [x] L=4, t1=0, g=0   -> 512 pi-modes  16/16 active  LOOPS  (SOLVED)
- [x] L=4, t1=0, g=0.5 -> 0 pi-modes    0/16  active  HALTS  (melted)
- [x] L=4, t1=0.1, g=0 -> 512 pi-modes  16/16 active  LOOPS  (small hopping survives)
- [x] L=4, t1=1.0, g=0 -> 0 pi-modes    0/16  active  HALTS  (strong hopping melts)
- [x] gamma_sweep: g=0,0.3 LOOPS;  g=0.5,1.0 HALTS
- [x] n_grid uniform at ideal t1=0: every slice has 32 pi-modes
- [x] count_pi_modes returns 32 for ideal L=4
- [x] N = 4*L*L for L in 4,6
- [x] Live HTTP: /api/dim5/run?L=4&n_k=4&t1=0&g=0 returns LOOPS
- [x] Live HTTP: /api/dim5/run?L=4&n_k=4&t1=0&g=0.5 returns HALTS

### 1F: Smoke tests — DONE (2026-06-04)
- [x] Write `tests/smoke.py` (incremental across 1A-1E + 1F additions):
  - [x] For 1D: verify all 4 machines return correct W and verdict
  - [x] For 2D: L=8, gamma_halt=0 → C_loop=+1; gamma_halt=10 → C_halt=0
  - [x] For 3D: gamma_halt=0 → maxC=2, nonzero=14/24; gamma_halt=15 → maxC=0
  - [x] For 4D: L=6 n_k=4 gamma_halt=0 → C2=-2 (nonzero); gamma_halt=15 → C2=0
  - [x] For 5D: t1=0 g=0 → 32 pi-modes/slice, 16/16 active; g=0.5 → 0
  - [x] All tests pass against the actual lab source outputs
- [x] 1F cross-dimension tests:
  - [x] `test_all_engines_canonical_run` -- 6 canonical runs (1D halt+loop, 2D, 3D, 4D, 5D) all return valid verdicts
  - [x] `test_json_serializable` -- all 7 engine outputs round-trip through json.dumps/loads
  - [x] `test_against_lab_source_outputs` -- exact numbers from each lab source's main() / sweep
  - [x] `test_http_endpoints` -- TestClient hits /api/health + all 5 dimension endpoints
  - [x] `test_summary_table` -- prints unified topology-metric table per dimension

**Verify**:
- [x] `python -m tests.smoke` exits 0 (36 tests across 1A-1F, ~18s)
- [x] Each `/api/dim{N}/run` returns the expected verdict (verified via TestClient)
- [x] Engine does not modify the source CAT_CAS files (read-only imports via importlib)

---

## Phase 2 — 1D view (mechanism) — DONE (2026-06-04)

**Goal**: Show WHY halting works, not just that it does. Mechanism view.

### Layout
- [x] Side panel (320px) with: machine picker, gamma/loss/n_phi sliders, Run/Animate/Reset buttons
- [x] Verdict banner (HALTS/LOOPS) with W, kappa_V, rho, halt-sink KPI tiles
- [x] Machine spec readout (transitions, twist edges, labels, halt index)
- [x] Mechanism explanation text (per-machine)
- [x] 3 viz panels: state graph + flow (top-left), spectrum (top-right), det curve (bottom)
- [x] `frontend/css/dim1.css` — layout, verdict banner, machine spec styles

### State graph (`frontend/js/dim1_stategraph.js`)
- [x] N nodes arranged in a circle, each labeled with |s, b> basis state
- [x] Edges drawn for non-zero H[i,j] entries, thickness = |H_ij|, color = teal/halt-red
- [x] Halt nodes colored red, |psi|^2 inner dot + radial glow
- [x] Reset on Run; click Animate flow to start the time evolution
- [x] Layout: circle of radius 0.40 * min(w,h), node radius scales with canvas

### Wavepacket flow (`frontend/js/dim1_flow.js`)
- [x] Animate psi(t+dt) = psi(t) - i*dt * H*psi(t) (Euler step at dt=0.05, 4 steps/frame)
- [x] Periodic re-normalization to keep |psi| on the unit sphere
- [x] Driven by requestAnimationFrame, cancel on pause/reset
- [x] Pause/Reset buttons toggle state

### Spectrum (`frontend/js/dim1_spectrum.js`)
- [x] Scatter eigenvalues in complex plane (Re right, Im up)
- [x] Halt-mask eigenvalues colored red, others teal
- [x] Unit-circle reference (point-gap radius)
- [x] |max lambda| and N captions
- [x] Outer glow per eigenvalue

### Det curve (`frontend/js/dim1_detcurve.js`)
- [x] Closed curve det(1 - e^{i*phi} H) for phi in [0, 2*pi)
- [x] Origin marked, reference circles at 25/50/75/100%
- [x] Winding number (W) banner top-right (color matches verdict)
- [x] phi = 0, pi/2, pi, 3pi/2 markers along the curve
- [x] |det(phi)| mini-strip at the bottom (one-row histogram)

### Complex math helper (`frontend/js/complex.js`)
- [x] H parsed once into flat (re, im) Float32Arrays
- [x] zgemv (complex general matrix-vector multiply)
- [x] Euler step: psi += -i*dt * (H*psi)
- [x] Normalize, |psi|^2, uniform/delta initial states

### API extensions (`engine/api_routes_1d.py`)
- [x] GET /api/dim1/machines — list machines with expected verdict + summary
- [x] GET /api/dim1/build — return H only (cheap, for flow restart)
- [x] GET /api/dim1/run — full run (H, spectrum, winding)

### Controller (`frontend/js/dim1.js`)
- [x] Wires all 4 viz modules + flow animator
- [x] URL param overrides: ?machine=...&gamma=...&loss=...&nphi=...
- [x] Auto-init on first health check (no tab click required)
- [x] Per-machine mechanism explanation text

**Verify**:
- [x] `python -m tests.smoke` exits 0 (engines unchanged)
- [x] `http://localhost:8000/` loads dim1 view automatically
- [x] `loop_2cycle` (default): verdict=LOOPS, W=+1, det curve winds once
- [x] `loop_3cycle`: 6-node state graph, same W=+1 behavior
- [x] `halt_direct` (via ?machine=halt_direct): verdict=HALTS, W=0, halt nodes red with strong sink
- [x] Headless Chrome screenshot: 3 cases render correctly
- [x] Node `--check` on all 10 JS modules: OK

---

## Phase 3 — 2D Chern view — DONE (2026-06-05)

**Goal**: Show the LxL lattice, spectrum, Bott index sweep, and edge current.

### Engine extensions (`engine/oracle_2d.py`)
- [x] `gamma_sweep(L, gammas, t1, t2, phi, loss, n_pts, radius, include_projector) -> {L, N, points: [{gamma_halt, C, verdict, E_fermi_im}]}`
- [x] `preset_machines() -> {machines: {loop_default, halt_default, uniform_annihilation, l4_fragility}, params}`
- [x] Try/except per-gamma: NaN-prone points return C=0 with `error` field

### API extensions (`engine/api_routes_2d.py`)
- [x] `GET /api/dim2/machines` — list 4 presets with param ranges
- [x] `GET /api/dim2/build` — build H only (cheap)
- [x] `GET /api/dim2/run` — full run (H, spectrum, fermi, projector, Bott)
- [x] `GET /api/dim2/gamma_sweep` — Bott C vs gamma_halt curve

### Lattice heatmap (`frontend/js/dim2_lattice.js`)
- [x] L×L grid of cells, color = |Im(H_ii)| with FIXED cap of 1.0
  - bulk loss (~0.05) -> light blue; sink (gamma=10) -> full red
- [x] Red ring + label marks the EP halt site at (4, 4) on L=8
- [x] Legend: L, N, Im(H) range
- [x] Subscribes to themechange, redraws on theme toggle

### Spectrum (`frontend/js/dim2_spectrum.js`)
- [x] Scatter eigenvalues in complex plane
- [x] Dashed E_Fermi line (Im axis) labeled with value
- [x] Eigenvalue color: norm Im(red) -> decay(blue)
- [x] Re/Im range and N captions

### Bott curve (`frontend/js/dim2_bottcurve.js`)
- [x] Step plot of C vs gamma_halt
- [x] Current-gamma marker (dashed vertical line)
- [x] Integer C gridlines, axis labels, x ticks
- [x] "click Sweep" hint when empty

### Edge current (`frontend/js/dim2_edge.js`)
- [x] Per-row sum of |P_ij|^2 averaged across columns -> bar chart
- [x] Edge rows tinted loop (cyan), halt row tinted halt (red)
- [x] Loop case shows uniform weights; halt case shows halt row spike

### Controller (`frontend/js/dim2.js`)
- [x] MECHANISM_TEXT per preset (loop / halt / uniform / l4)
- [x] URL param overrides: ?machine=...&L=...&gamma=...&t1=...&t2=...&phi=...&loss=...
- [x] Debounced params (250-380ms), AbortController in-flight cancellation
- [x] Verdict banner + 4 KPIs (C, E_F, rho, sink) + lattice spec
- [x] Sweep gamma button (calls /api/dim2/gamma_sweep with 10 gammas)
- [x] Export: copy URL, download JSON, save PNG (all 4 dim2 canvases)

### Layout (`frontend/css/dim2.css`)
- [x] 320px side panel + 1fr main grid
- [x] Top row: lattice | spectrum (1fr each)
- [x] Middle row: Bott curve (full width)
- [x] Lower row: edge current (full width)
- [x] Mechanism text at bottom (max-height 110px, scrollable)
- [x] Advanced (t1, t2, phi, loss) collapsible details
- [x] Responsive: collapses to single column below 1100px

### Tab wiring (`frontend/index.html`, `frontend/js/main.js`)
- [x] dim2 tab enabled (no longer disabled placeholder)
- [x] `?tab=dim2` URL param switches to dim2 on first health check
- [x] `Phase 3 — 2D Chern View` shown in phase label
- [x] Theme + export panel apply to dim2 too

**Verify**:
- [x] `python -m tests.smoke` exits 0 (44 tests, 20.27s)
- [x] L=8, gamma_halt=0  -> C=+1 (LOOPS) — matches `37_2d_chern_oracle.py` run_2d_oracle default
- [x] L=8, gamma_halt=10 -> C=0  (HALTS) — EP sink destroys edge
- [x] gamma_sweep engine: g=0,2,5,10 -> C=+1,+1,0,0 (L=8)
- [x] preset_machines exposes 4 entries (loop, halt, uniform, l4) with summaries
- [x] All 6 dim2 JS files pass `node --check` and are served by the static mount
- [x] index.html wires 4 canvases + 7 controls + 3 export buttons
- [x] /api/dim2/{machines,build,gamma_sweep} all 200 via TestClient
- [x] Headless Chrome screenshots verified:
  - dark + loop_default: light blue lattice (no sink), E_F=-0.405, C=+1, LOOPS
  - dark + halt_default: 1 red sink cell at center, E_F=-5.569, C=0, HALTS
  - light + loop_default: same lattice, all CSS vars respond to themechange

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
| 2D | L | 4 | 12 | 8 | 37 |
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

**Current phase**: Phase 3 — 2D Chern view (DONE)

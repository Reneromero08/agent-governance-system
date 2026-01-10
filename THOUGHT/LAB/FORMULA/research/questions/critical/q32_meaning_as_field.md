# Question 32: Meaning as a physical field (R: 1670)

**STATUS: OPEN**

## Question
Is "meaning" just a label for compression/inference, or can it be defined as a **real, measurable field** with dynamics (like EM)?

Concretely:
- Give an operational definition of a "meaning field" that can be measured from data (not introspection).
- Define field variables, sources, coupling, and conservation/propagation laws (what is "charge/current"?).
- Produce at least one prediction that differs from standard information-theoretic / Bayesian accounts.

**Success criterion:** a quantitative model that makes novel, testable predictions (and can in principle be falsified).

## Solved Criteria + Test Plan
See `THOUGHT/LAB/FORMULA/research/questions/reports/Q32_SOLVED_CRITERIA_AND_TEST_PLAN.md`.

## TESTS
`experiments/open_questions/q32/`
- `q32_meaning_field_tests.py` - echo-chamber falsifier, phase-transition gate, propagation/gluing
- `q32_adversarial_gauntlet.py` - parameter sweeps, noise-family shifts, negative controls, independence intervention
- `q32_public_benchmarks.py` - public truth-anchors (Climate-FEVER + SciFact), fast-mode CLI, Phase-2 bench + Phase-4 streaming + Phase-3 threshold transfer (`--mode bench|stream|transfer`)

---

## Current state
We have a **candidate operationalization** (`M := log(R)`) and a first falsification harness (`q32_meaning_field_tests.py`) that passes on a synthetic generator.

That is not “settled” yet. Q32 stays **OPEN** until the same falsifiers hold under public, adversarial, out-of-domain data with pinned versions and replication.

### Canonical field variable
Define the **Meaning Field intensity** as:

- `M := log(R)`  (recommended; stabilizes multiplicative terms)

Where the Living Formula is:

`R = (E / ∇S) × σ(f)^Df`

Interpretation (canonical, from existing context):
- `E` = **empirically grounded compatibility** with reality (not mere agreement)
- `∇S` = **local scale / uncertainty** (normalization term; Q1 forces it)
- `σ(f)^Df` = **compression × fractal depth gain** (known sensitivity risk; must be bounded/validated)

### Why this is legitimately “field-like”
`M(x)` (or `M(s,a)`) is a scalar field over points in the semiosphere:
- High `M` regions are **stable basins / attractors** (interpretations that compress well and survive entropy).
- Gate boundaries are **level sets** of the field (`M > τ` or `R > τ`) — phase boundaries that decide action/commitment.

### Sources, sinks, coupling (in your terms)
This keeps the “physics” analogy honest without inventing new particles:
- **Sources:** independent evidence that increases `E` and/or increases compressibility (`σ`, `Df`) without raising uncertainty.
- **Sinks:** contradiction, adversarial noise, context fracture that increases `∇S` (or destroys `E` grounding).
- **Coupling:** authority/context shifts the effective measurement of `E` and `∇S` (Semiotic Mechanics Axiom 6).

### Dynamics (Free Energy bridge, family-scoped)
In a specified likelihood family (Gaussian), Q1 gives:

`M = log(R) = -F + const`  ⇒  `R ∝ exp(-F)`

So “meaning flow” is (within that scope) equivalent to moving downhill on free energy: meanings stabilize where surprise is minimized under the model.

### Nontrivial predictions (and falsifiers)
These are implied by the field definition and are not just word games:

1. **Echo-chamber collapse prediction:** high “consensus” that is correlated will not survive independence stress; adding fresh independent data collapses false basins (`M` drops).
2. **Phase transition prediction:** `M(t)` evolves nonlinearly (long ambiguity → sharp crystallization) when constraints cross thresholds.
3. **Propagation prediction:** locally consistent meanings (that glue across overlapping patches) propagate; locally inconsistent ones do not.

**Falsifier:** if `M` stays high for correlated echo chambers under independence stress, then this is not a “meaning field,” it’s a social-mirroring field.

---

## Canonical report stitching the context
If you want the "single paragraph" canon version and falsification boundary without math framing, use:
- `THOUGHT/LAB/FORMULA/research/questions/reports/MEANING_FIELD_CANON_FROM_EXISTING_CONTEXT.md`

---

## Roadmap (hardcore falsification, no shortcuts)
Goal: make “meaning-as-field on the semiosphere” survive increasingly hostile tests, with clear gates for promotion from **OPEN → PARTIAL → ANSWERED**.

### Phase 1 — Completed (what we already built + receipted)

#### 1.1 Public harness + falsifiers
- [x] Public benchmark harness exists (`q32_public_benchmarks.py`) for SciFact + Climate-FEVER.
- [x] Wrong-check intervention gates exist (truth-consistent check vs truth-inconsistent check).
- [x] Neighbor wrong-check mode (`J`) exists (`--wrong_checks neighbor --neighbor_k K`).

#### 1.2 Empirical receipts + datatrail
- [x] EmpiricalMetricReceipt exists (`--empirical_receipt_out`) and records run context + per-result gates.
- [x] Receipt includes `R` / `log(R)` summary stats (`details.mean_R_*` + `details.mean_logR_*`).
- [x] Stress mode is receiptable (emits `SciFact-Streaming-Stress` summary result).
- [x] Datatrail exists and is linked from `THOUGHT/LAB/FORMULA/research/questions/reports/Q32_NEIGHBOR_FALSIFIER_DATA_TRAIL.md`.

#### 1.3 Replication probes (proof-of-life)
- [x] Threshold transfer + matrix runs exist (`--mode transfer|matrix`) and have receipted examples.
- [x] Multi-trial stress run with a pass-rate gate exists (receipted + hashed).

### Phase 2 — Next half (mechanism validation)

#### 2.1 Ablations (non-tautology proof)
- [x] Ablations implemented: `--ablation full|no_essence|no_scale`.
- [x] Add a depth/sensitivity ablation (Df/σ proxy) in the public harness.
- [x] Require the expected pattern: at least one ablation reliably kills the effect while `full` passes (receipted).

#### 2.2 Negative controls must fail hard
- [x] Paraphrase-only “agreement inflation” control collapses `R/log(R)` and fails gates (receipted).
- [x] Strong swap/shuffle controls receipted for every relevant mode.

#### 2.3 Robustness sweeps (distributional proof)
- [x] Sweep `neighbor_k` and require pass-rate ≥ threshold across a range (receipted).
- [x] Sweep stream sampling variability and require pass-rate ≥ threshold across trials (receipted).

### Phase 3 — Other half (settlement)

#### 3.1 Third domain (break the 2-dataset trap)
- [x] Add a third public benchmark domain (SNLI) and repeat transfer without retuning.

#### 3.2 Full-scale runs (receipted)
- [ ] Full (non-fast, crossencoder) matrix across multiple seeds.
- [ ] Full stress with higher `--stress_n` and hard `--stress_min_pass_rate` gate(s).

#### 3.3 Pinned replication + “attempt SOLVED” package
- [ ] Pinned environment + rerun bundle with hashes so independent reruns reproduce receipts.
- [ ] Update solved criteria report with the final evidence bundle + explicit falsification boundary.

### Expanded breakdown (legacy detail mapped into Phase 1/2/3 above)

#### Legacy Phase 0 — Spec freeze (no post-hoc relabeling)
- Freeze `M` definition (default `M := log(R)`) and explicitly document `E`, `∇S`, `σ`, `Df`.
- Freeze falsifiers: echo-chamber collapse, phase transition gating, propagation/gluing.
- Freeze thresholds and negative controls (shuffles/permutations must kill the signal).

**Gate:** changing definitions after seeing results is not allowed; changes require new fixtures and re-running the whole gauntlet.

#### Legacy Phase 1 — Adversarial synthetic gauntlet
Make the generator actively try to break the field:
- Correlation strength sweeps (shared-bias channels).
- Paraphrase storms / near-duplicate sources (agreement inflation).
- Poisoned evidence (true + persuasive false).
- Domain shifts (Gaussian/Laplace/heavy-tail; heteroskedasticity).

**Gate:** the falsifiers must hold across sweeps; failures must yield a minimal counterexample fixture.

#### Legacy Phase 2 — Public truth-anchored semiosphere
Run the same falsifiers on public benchmarks with external truth anchors:
- Fact verification (e.g., FEVER / SciFact): claim + evidence + label.
- Build a semiosphere graph over claim/evidence embeddings; define neighborhoods and overlaps.
- Define independence as distinct sources/pages/docs (not paraphrases of one source).

**Gate:** echo-chamber falsifier must hold under independence stress; negative controls must fail hard.

#### Legacy Instrumentation (EmpiricalMetricReceipt: R/J/Phi-proxy)
To keep this falsifiable and auditable, every benchmark run should emit a machine-readable receipt alongside verbatim logs:

#### Legacy Empirical Metric Receipt (R/J/Phi)

**Purpose:** Make empirical claims auditable: record **R / J / Phi-proxy** alongside the receipts produced by tests and benchmarks.

- [x] Emit `EmpiricalMetricReceipt` JSON (separate from TokenReceipt)
  - Implemented in `q32_public_benchmarks.py` via `--empirical_receipt_out <path.json>`
  - Format: a single JSON object with `{type: "EmpiricalMetricReceipt", version: 1, run: {...}, results: [...]}`.
- [ ] Schema formalization (optional but recommended):
  - If we want an explicit JSON Schema file, decide the canonical location for it (Q32 docs currently define the shape in prose; the runner emits concrete JSON).
- [x] Required fields (implemented now):
  - Run context: `mode`, `dataset`, `fast`, `strict`, `seed`, `scoring`, `wrong_checks`, `neighbor_k`, `scifact_stream_seed`, `calibrate_on`, `apply_to`, `calibration_n`, `verify_n`
  - Per-result: `name`, `passed`, and `details` (includes `pair_wins`, `z`, `mean_margin`, plus `gate_z`/`gate_margin` and `phi_proxy_bits`; neighbor mode adds `mean_neighbor_sim`)
- [x] `R` / `M=log(R)` summary stats (implemented now):
  - Intervention benches (`SciFact`, `Climate-FEVER-Intervention`): `details.mean_R_correct/mean_R_wrong` and `details.mean_logR_correct/mean_logR_wrong`
  - Streaming benches (`SciFact-Streaming`, `Climate-FEVER-Streaming`): `details.mean_R_correct_end/mean_R_wrong_end` and `details.mean_logR_correct_end/mean_logR_wrong_end`
- [ ] Efficiency constraints:
  - Use lightweight proxies first (MI / multi-information via binning)
  - Add a "stress mode" pass-rate gate for variability when `scifact_stream_seed=-1`
- [ ] Datatrail:
  - Emit verbatim logs + SHA256 under `LAW/CONTRACTS/_runs/...`
  - Link to the report in `THOUGHT/LAB/FORMULA/research/questions/reports/`

**Datatrail examples (already emitted):**
- See `THOUGHT/LAB/FORMULA/research/questions/reports/Q32_NEIGHBOR_FALSIFIER_DATA_TRAIL.md`
  - Fast receipt example: `LAW/CONTRACTS/_runs/q32_public/datatrail/empirical_receipt_matrix_neighbor_phi_fast_20260109_193751.json`
  - Full receipt example: `LAW/CONTRACTS/_runs/q32_public/datatrail/empirical_receipt_matrix_neighbor_phi_full_20260109_194549.json`
  - Stress receipt example: `LAW/CONTRACTS/_runs/q32_public/datatrail/empirical_receipt_stress_smoke_20260109_202945.json`

**Exit Criteria (Q32-scoped):**
- [ ] Every public benchmark run (`bench|stream|transfer|matrix`) can emit an EmpiricalMetricReceipt deterministically
- [ ] Datatrail bundle includes: verbatim log + receipt + SHA256 for both
- [ ] Negative controls + stress runs are receipted (when invoked)

#### Legacy Phase 3 - Cross-domain replication + threshold transfer
- Calibrate once (Dataset A), freeze thresholds and mapping, then run Dataset B without re-tuning.

**Gate:** no "works if retuned" passes; that's not a field invariant.

**Status (short version):** implemented as `q32_public_benchmarks.py --mode transfer` and currently passes both directions with fixed seed defaults (calibrate `climate_fever` → apply `scifact`, and calibrate `scifact` → apply `climate_fever`).

#### Legacy Phase 4 — Real semiosphere dynamics (nonlinear time)
- Evidence arrives over time; measure `M(t)` and gate transitions.
- Test interventions: inject independent checks and require true basins stabilize while false basins collapse.

**Gate:** correct causal response to interventions (not just correlation).

#### Legacy Phase 5 - Scale, replication, and settlement (long road)
This is what we need before we can responsibly claim **SOLVED**:

- **Big runs (receipted):**
  - Full `matrix` (non-fast, crossencoder) with `--wrong_checks neighbor` across multiple seeds.
  - Full `stress` with higher `--stress_n` and a hard `--stress_min_pass_rate` gate; receipt + stress_out + verbatim logs + hashes.
- **Not a dataset artifact:** add at least one *third* public benchmark domain and repeat transfer/matrix.
- **Negative controls must fail hard:** shuffles, wrong-check swaps, paraphrase-only “agreement inflation” should collapse the signal (receipted).
- **Stability under perturbations:** sweep `neighbor_k`, stream sampling, and other knobs; require pass-rate stays above threshold (distribution is part of the proof).
- **Ablations:** show the effect disappears when the empirical anchor / scale term / depth term is removed (no tautology).
- **Pinned replication:** rerun with pinned environments and record hashes so the datatrail is reproducible.

#### Legacy Promotion criteria
- **OPEN → PARTIAL:** Phase 1 + Phase 2 pass on at least one public benchmark with fixed thresholds and full negative controls.
- **PARTIAL → ANSWERED:** Phase 2-4 pass across multiple benchmarks/domains, plus replication with pinned versions + independent reruns.

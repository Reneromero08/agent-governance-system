# Phase 6 Chiral Lane Frontier Roadmap

Status: `FRONTIER_ROADMAP_OPEN`

## Purpose

This subphase is the post-wall-crossing campaign map for the exact gap now isolated by Phase 6:

1. The physical carrier can transport a fold-odd orientation lane when the sender binds it.
2. Public cosine-only tape operations have not yet synthesized that lane.

The mission is not to re-test generic public-data transforms. Those are closed. The mission is to find a physical or temporal operation that generates a directional carrier from a public oracle process before, during, or after the even projection.

## Fixed Facts

These are now treated as hard starting points:

- `FOLD_MAGNITUDE_RECOVERED`: public phase/QFT-style readout recovers `a = min(d, N-d)` at `1.000` in the current harnesses.
- `PDN_CARRIER_LIVE`: hidden chiral controls recover the bound orientation lane at `AUC=1.000`.
- `PUBLIC_CHIRAL_PREP_NO_CROSSING`: public chiral phase-kickback stayed below null.
- `ONE_BIT_SEARCH_NO_CROSSING`: trying both candidate signs did not make the true sign win over the false sign.
- `MICROSTEP_RAMP_NO_CROSSING`: fractional public microsteps did not create a resolvable chiral lane.
- `PUBLISHED_DATA_SIDE_CLOSED`: public `(k,b,N)` transforms remain fold-even unless a hidden lane is introduced.

## Claim Gate

No result may be called a wall crossing unless all conditions pass:

1. Input construction uses only public oracle data and public scheduling metadata.
2. Fold magnitude `a` is recovered or supplied from public data only.
3. Both candidate signs are tested under matched workload and matched final tape state.
4. True candidate response beats false candidate response above shuffle null.
5. Hidden-control version is live at high confidence.
6. Public shuffled/null schedule stays at chance.
7. Result repeats across seeds and at least two `n` sizes.
8. No hidden `d`, orientation label, branch label, seed half, or private sign enters the public path.

Accepted frontier labels:

- `CHIRAL_LANE_GENERATED`
- `DIFFERENTIAL_COMMON_MODE_REJECTED`
- `SIDEBAND_CHIRAL_SIGNAL_FOUND`
- `MACRO_CHIRAL_QFT_SIGNAL_FOUND`
- `RESTORE_WORK_ASYMMETRY_FOUND`
- `PHYSICAL_COLLISION_SIGNAL_FOUND`
- `TRANSIENT_CARRIER_SIGNAL_FOUND`
- `RESONANCE_SIGN_SPLIT_FOUND`
- `PUBLIC_ROUTE_NO_CROSSING`
- `GATE_NOT_LIVE`

## Execution Checklist

- [ ] Track A: build dual-lane even cancellation harness.
- [ ] Track A: measure sender phase jitter and reject trials outside tolerance.
- [ ] Track A: map local CPU topology before choosing sender/receiver cores.
- [ ] Track A: run hidden differential control and require `AUC >= 0.95`.
- [ ] Track A: run public true-vs-false differential test at n=8 and n=10.
- [ ] Track A: run swapped-core and shuffled-schedule nulls.
- [ ] Track A: write `PHASE6_DUAL_LANE_DIFFERENTIAL.md`.
- [ ] Track B: build PDN sideband mixer harness.
- [ ] Track B: run carrier/harmonic sideband sweep with off-frequency controls.
- [ ] Track B: write `PHASE6_PDN_SIDEBAND_MIXER.md`.
- [ ] Track C: port QFT/phase diffraction block into native chiral harness.
- [ ] Track C: bind chirality to ROL/ROR or ascending/descending stride, not only math sign.
- [ ] Track C: run clockwise/counterclockwise QFT macro-accumulation tests.
- [ ] Track C: write `PHASE6_CHIRAL_QFT.md`.
- [ ] Track D: isolate reverse/uncompute measurement window.
- [ ] Track D: run restore-work asymmetry tests and controls.
- [ ] Track E: build synchronized two-sender collision sieve.
- [ ] Track E: add explicit execution-port/cache-bank pressure block.
- [ ] Track E: run same-sign/opposite-sign collision tests.
- [ ] Track F: design synthetic-only transient carrier harness.
- [ ] Track F: run transient-control and non-transient-control gates.
- [ ] Track G: build harmonic resonance sweep harness.
- [ ] Track G: run frequency sweep with static/no-sender controls.
- [ ] Track H: maintain route table and boundary report after every track.
- [ ] Re-run any positive Ryzen result on the Phenom before treating it as frontier evidence.
- [ ] Decide whether Phase 6 frontier remains open after Tracks A-C.

## Priority Tracks

### Track A: Dual-Lane Even Cancellation

Priority: 1

Hypothesis:
The even cosine workload is the dominant common-mode power term. If two sender lanes run opposite chiral candidate preparations simultaneously, the even term may cancel in the receiver measurement while the differential chiral residue remains.

Main threat:
`PHASE_JITTER_DERIVATIVE_ARTIFACT`. Differential rejection only works if sender lanes are phase-aligned. If two common-mode cosine workloads are offset by even a few cycles, subtraction produces a derivative artifact larger than the expected chiral residue.

Implementation target:

- Two sender cores.
- One receiver core.
- Map topology before core selection. On Ryzen, keep the two senders inside the same CCX/L3 domain when possible; place the receiver to minimize direct L3 conflict noise if the goal is PDN readout. On Phenom, use two cores in the shared L3 domain and reserve a separate receiver core.
- Use an atomic spin-barrier with TSC capture on all sender lanes.
- Abort/retry trials where sender start TSC skew exceeds the measured tolerance target.
- Lane A drives candidate `a`.
- Lane B drives candidate `N-a`.
- Both lanes have matched duty, slots, tape hash/restoration proof, and public output.
- Run swapped-lane control to remove core bias.

Gate:

- Report sender start skew distribution.
- Reject or separately report trials beyond jitter tolerance.
- `true_lane_response - false_lane_response` clears shuffle null.
- Swapped lane preserves candidate result, not core identity.
- Hidden differential control hits `AUC ~= 1.000`.
- Public equal-sign and public shuffled schedules stay at chance.

Artifact names:

- `dual_lane_differential/chiral_dual_lane.rs`
- `dual_lane_differential/PHASE6_DUAL_LANE_DIFFERENTIAL.md`
- `dual_lane_differential/results/dual_lane_differential_result.json`

### Track B: PDN Sideband Mixer

Priority: 2

Hypothesis:
A chiral residue may be too small in baseband but visible as a phase inversion in a driven sideband.

Implementation target:

- Sender modulates candidate phase walk at a carrier frequency.
- Receiver demodulates at carrier and selected harmonics.
- Compare true sign, false sign, shuffled sign, and hidden sign.

Gate:

- True candidate sideband phase separates from false candidate above null.
- Common-mode amplitude alone cannot classify.
- Off-frequency controls fail.
- Hidden sideband control is live.

Artifact names:

- `sideband_mixer/chiral_sideband_mixer.rs`
- `sideband_mixer/PHASE6_PDN_SIDEBAND_MIXER.md`
- `sideband_mixer/results/sideband_mixer_result.json`

### Track C: Chiral QFT Macro Accumulation

Priority: 3

Hypothesis:
Single chiral walks may be too small. A QFT-style phase diffraction block applies many coordinated rotations, so physical chirality may accumulate over the whole block.

Implementation target:

- Recreate the phase diffraction/QFT operation in the native harness.
- Add a chirality parameter to all rotations.
- Bind chirality to a physical CPU structure:
  - candidate `a`: one rotate direction, for example `ROR`, or ascending memory stride.
  - candidate `N-a`: opposite rotate direction, for example `ROL`, or descending memory stride.
- Keep output magnitudes identical.
- Measure full-block PDN/timing response.

Gate:

- Candidate `a` and `N-a` produce equal public magnitudes.
- True-sign QFT response beats false-sign response above null.
- Hidden chiral QFT control is live.
- Reversed rotation schedule and random rotation order nulls fail.

Artifact names:

- `chiral_qft/chiral_qft_pdn.rs`
- `chiral_qft/PHASE6_CHIRAL_QFT.md`
- `chiral_qft/results/chiral_qft_result.json`

### Track D: Restore-Work Asymmetry

Priority: 4

Hypothesis:
The forward pass may leave microarchitectural state that changes the reverse/uncompute work even when the final tape hash is identical.

Implementation target:

- Run forward chiral candidate preparation.
- Measure only the reverse/uncompute window.
- Compare candidate signs and swapped order.
- Preserve hash/restoration proof.

Gate:

- Reverse-window response separates true and false candidate signs.
- Forward-window-only and total-runtime-only controls cannot explain result.
- Hidden restore-control is live.

Artifact names:

- `restore_work/restore_work_asymmetry.rs`
- `restore_work/PHASE6_RESTORE_WORK_ASYMMETRY.md`
- `restore_work/results/restore_work_result.json`

### Track E: Physical Collision Sieve

Priority: 5

Hypothesis:
Two public oracle preparations running together may create a shared-resource interference pattern that differs between same-sign and opposite-sign chiral schedules.

Implementation target:

- Two sender cores run synchronized public candidate lanes.
- Force matched cache/port/load-store pressure without reading hidden state.
- Do not rely on incidental interference. Add an explicit pressure block:
  - vector add/multiply churn where available, or scalar fallback on Phenom.
  - cache-bank/L3-set pressure variant.
  - load/store queue pressure variant.
- Receiver or timing observer measures collision response.

Gate:

- Same-sign vs opposite-sign public pairing separates above null.
- Core-pair swap and schedule shuffle fail.
- Hidden collision control is live.

Artifact names:

- `collision_sieve/physical_collision_sieve.rs`
- `collision_sieve/PHASE6_PHYSICAL_COLLISION_SIEVE.md`
- `collision_sieve/results/collision_sieve_result.json`

### Track F: Transient-State Chiral Carrier

Priority: 6

Scope boundary:
This is a synthetic lab-only transient-state timing experiment. It must not read, infer, or expose real process memory or system data. It may only use generated lab instances and generated labels inside the harness.

Hypothesis:
A branch-gated transient window might leave a measurable physical carrier trace even if architectural state is later discarded.

Implementation target:

- Train a branch direction in a generated loop.
- Place a minimal chiral carrier sequence in the transient window.
- Measure only aggregate timing/PDN-style response.
- Keep all data synthetic and local to the harness.

Gate:

- Hidden transient-control is live.
- Public branch labels do not smuggle orientation.
- Public candidate sign beats false sign only if branch-gated carrier creates a true physical selection.
- Non-transient branch and always-taken branch controls fail.

Artifact names:

- `transient_carrier/transient_chiral_carrier.rs`
- `transient_carrier/PHASE6_TRANSIENT_CHIRAL_CARRIER.md`
- `transient_carrier/results/transient_carrier_result.json`

### Track G: Harmonic Resonance Sweep

Priority: 7

Hypothesis:
The public oracle execution may have a physical resonance response not captured by scalar timing. Candidate signs might split under a driven frequency sweep.

Implementation target:

- Aggressor workload sweeps frequency.
- Sender runs public candidate lane.
- Receiver records response across frequency bins.
- Search for candidate-dependent resonance shift.

Gate:

- True/false candidate resonance peak or phase separates above shuffled-frequency null.
- Static workload and no-sender controls fail.
- Hidden resonance control is live.

Artifact names:

- `resonance_sweep/harmonic_resonance_sweep.rs`
- `resonance_sweep/PHASE6_HARMONIC_RESONANCE_SWEEP.md`
- `resonance_sweep/results/resonance_sweep_result.json`

### Track H: Hardware Decodability Boundary

Priority: always-on

Hypothesis:
If all public physical routes fail with live controls, the publishable result is a hardware decodability boundary: physical carriers can transport a hidden chiral lane, but public cosine-only oracle execution does not generate one under tested substrates.

Implementation target:

- Maintain a table of all routes, live controls, public verdicts, and remaining assumptions.
- Separate "transport works" from "lane synthesis works."
- Keep the boundary precise: the wall is between physical transport and public lane generation.
- Treat Ryzen as fast iteration and Phenom as final-substrate adjudication. Any positive result on Ryzen must be repeated on Phenom before frontier promotion.

Artifact names:

- `PHASE6_HARDWARE_DECODABILITY_BOUNDARY.md`
- `results/phase6_frontier_route_table.json`

## Next Recommended Action

Run Track A first: `Dual-Lane Even Cancellation`.

Reason:
It is the cleanest new assumption break. It does not require speculative state, new math, or a broader threat model. It only changes the measurement geometry from single-lane common-mode readout to differential common-mode rejection.

Initial acceptance target:

- n=8 and n=10.
- 42 paired instances per mode minimum.
- sender cores selected after topology scan, not hard-coded. Initial Ryzen candidate: same-CCX sender pair and separate receiver; then Phenom after current long run is complete.
- sender start skew reported, with high-skew trials rejected or isolated.
- hidden differential control: `AUC >= 0.95`.
- public differential crossing candidate: true-vs-false AUC must clear shuffle-null 95th percentile by `>= 0.03`.

## Do-Not-Do List

- Do not change firmware.
- Do not write voltage or MSR control state.
- Do not treat hidden controls as crossings.
- Do not use real secrets or process memory in transient-state tests.
- Do not call a public result positive unless true candidate beats false candidate under the claim gate.
- Do not expand into a new phase until at least Track A, Track B, and Track C have either produced artifacts or failed under live controls.

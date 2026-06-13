# PHASE6_CHIRAL_LANE_FRONTIER_ROADMAP_2.md

Status: `FRONTIER_ROADMAP_OPEN_HIGH_TIER`

Claim ceiling until public crossing: `L4-L5`

Campaign name: **The Chiral Lane Hunt**

Core question:

```text
Can public execution geometry generate a fold-odd physical carrier when the published data itself is fold-even?
```

---

# 0. Executive Summary

Phase 6 has isolated the exact final wall:

```text
Public data gives the even/folded channel.
The hidden orientation lives in the odd/quadrature channel.
Direct transforms of public data do not synthesize the missing odd lane.
Hidden chiral controls show that a physical carrier can transport orientation when orientation is bound.
The remaining frontier is public lane generation.
```

The wall is no longer:

```text
Can we read d?
Can we build a better classifier?
Can we re-encode the public data?
Can scalar timing infer the missing bit?
Can a generic phase substrate recover a phase that is not present?
```

Those are closed.

The wall is now:

```text
Can a public oracle process, through physical execution geometry, create a fold-odd carrier before, during, or after the even projection?
```

This roadmap attacks that wall from every serious remaining angle.

The campaign is divided into three layers:

```text
Layer 1:
Instrument and calibrate the odd-lane detector.

Layer 2:
Attempt public chiral-lane generation through physical execution geometry.

Layer 3:
If a lane is generated, couple it back to the fixed-point / lattice orientation target and run scaling.
```

No result may be promoted unless it passes the global no-smuggle gate.

No hidden `d`, orientation label, branch label, private sign, seed half, scoring label, or private fold bit may enter the public runtime path.

---

# 1. Fixed Facts

These are now treated as hard starting points.

```text
FOLD_MAGNITUDE_RECOVERED
```

The public phase/QFT-style readout recovers:

```text
a = min(d, N-d)
```

at `1.000` in current harnesses.

```text
PDN_CARRIER_LIVE
```

Hidden chiral controls recover the bound orientation lane at high confidence.

```text
PUBLIC_CHIRAL_PREP_NO_CROSSING
```

Public chiral phase-kickback stayed below null.

```text
ONE_BIT_SEARCH_NO_CROSSING
```

Trying both candidate signs did not make the true sign win over the false sign.

```text
MICROSTEP_RAMP_NO_CROSSING
```

Fractional public microsteps did not create a resolvable chiral lane.

```text
PUBLISHED_DATA_SIDE_CLOSED
```

Public `(k,b,N)` transforms remain fold-even unless a hidden lane is introduced.

```text
PHASE_SUBSTRATE_READS_PRESENT_PHASE_ONLY
```

A phase-resolving substrate can read phase when phase is present. It cannot recover quadrature that the public projection destroyed.

```text
CURRENT FRONTIER
```

The live edge is between:

```text
physical transport of a bound orientation lane
```

and:

```text
public generation of a fold-odd lane
```

---

# 2. Definitions

## 2.1 Fold

Let:

```text
N = 2^n
d in [1, N/2)
sigma(d) = N - d
a = min(d, N-d)
```

The public data is fold-even.

The hidden orientation bit is:

```text
o = 1[d < N/2]
```

The lattice wall is the inability to recover `o` from the public folded channel.

## 2.2 Public candidate pair

Every public runtime sees only:

```text
candidate_0 = a
candidate_1 = N-a
```

The runtime must never know which candidate equals the private `d`.

The offline scorer may later assign:

```text
candidate_true
candidate_false
```

only after the run completes.

## 2.3 Chiral lane

A chiral lane is a physical or computational carrier whose response is odd under the fold:

```text
response(candidate_0) = -response(candidate_1)
```

or, more generally:

```text
response(candidate_true) separates from response(candidate_false)
```

under blinded scoring and matched public workload.

## 2.4 Public lane generation

A public lane is generated only if:

```text
public inputs
public candidate pair
public scheduling metadata
public runtime
```

produce a fold-odd physical signal without hidden orientation.

Hidden controls are not crossings. They are detector calibration.

## 2.5 Crossing

A crossing is not detection of a hidden injected lane.

A crossing is:

```text
public-only execution geometry
generates a measurable fold-odd carrier
such that the offline true candidate beats the false candidate above null
under blinded, matched, repeated controls
```

---

# 3. Absolute Claim Gate

No result may be called a wall crossing unless all conditions pass.

```text
G0: Public-only construction
```

Input construction uses only public oracle data and public scheduling metadata.

```text
G1: Fold magnitude public
```

`a = min(d,N-d)` is recovered or supplied from public data only.

```text
G2: Candidate blinding
```

Runtime labels are only `candidate_0` and `candidate_1`.

No runtime value may be labeled `true`, `false`, `d`, `N-d`, `left`, `right`, `orientation`, `sign`, `private`, or equivalent.

```text
G3: Matched workload
```

Both candidate signs are tested under matched workload, matched duty, matched memory pressure, matched final tape state, and matched output public magnitude.

```text
G4: True beats false
```

Offline true-candidate response beats false-candidate response above shuffle null.

```text
G5: Hidden control live
```

A hidden chiral control version must be live at high confidence for the same detector route.

```text
G6: Public nulls fail
```

Public shuffled schedule, equal-sign schedule, core-swap, lane-swap, random-fold, off-frequency, and no-sender/null controls must stay at chance.

```text
G7: Repetition
```

Result repeats across seeds and at least two `n` sizes.

Minimum: `n=8` and `n=10`.

Preferred: `n=8`, `n=10`, `n=12`.

```text
G8: No hidden data
```

No hidden `d`, orientation label, branch label, seed half, private sign, or scoring label enters the public path.

```text
G9: Tape restoration
```

If catalytic tape is used, final tape hash must restore bit-perfect.

Any FNV/SHA mismatch voids the run except in explicitly destructive controls.

```text
G10: Schedule invariance
```

The emitted public schedule must be invariant under private fold swap.

Private fold swap means:

```text
d ↔ N-d
```

while public `(k,b,N,a,{a,N-a})` remains equivalent.

---

# 4. Claim Ladder

Use these levels.

```text
L0: Null
```

No live hidden control. Detector route invalid.

```text
L1: Detector live
```

Hidden chiral lane detected. No public claim.

```text
L2: Public perturbation
```

Public candidate signs alter a physical observable, but true/false does not beat null.

```text
L3: Candidate separation
```

Candidate labels separate, but artifact/core/schedule controls remain unresolved.

```text
L4: Frontier positive
```

True beats false above shuffle null with controls, but limited scaling or platform replication.

```text
L5: Crossing candidate
```

True beats false under blinded public runtime, matched controls, repeated seeds, multiple sizes, and no-smuggle gates. Scaling begins to beat declared classical baseline.

```text
L6: Wall broken
```

Public chiral lane generation scales below the forward path and survives independent replication on the adjudication substrate.

No Phase 6 result may claim L6 without scaling.

---

# 5. Universal Architecture

All tracks use the same architecture.

```text
public_instance_generator
    produces public (k,b,N)
    produces hidden d only for offline scoring
    produces folded candidate pair {a,N-a}
    randomizes private fold

blind_runtime
    receives public instance
    receives candidate_0 / candidate_1
    receives public schedule only
    runs physical route
    restores tape if catalytic
    outputs raw physical response

offline_scorer
    receives hidden d after run
    maps candidate_0/candidate_1 to true/false
    computes AUC/effect/nulls
    writes verdict

gatekeeper
    checks no-smuggle, schedule invariance, restoration, controls
```

Runtime must be blind.

Scoring must be offline.

---

# 6. Statistical Protocol

## 6.1 Minimum experiment size

For promotion to frontier-positive:

```text
n = 8 and n = 10
42 paired instances per mode minimum
3 independent seed families
candidate labels blinded
```

Preferred:

```text
n = 8, 10, 12
128 paired instances per mode
5 independent seed families
cross-platform replay if available
```

## 6.2 Metrics

Required metrics:

```text
AUC_true_vs_false
effect_size
bootstrap_CI
shuffle_null_95
permutation_p
candidate_label_invariance
core_swap_delta
lane_swap_delta
schedule_shuffle_delta
equal_sign_delta
restoration_failure_count
```

## 6.3 Initial acceptance target

For a public route to receive `FRONTIER_POSITIVE`:

```text
AUC_true_vs_false >= shuffle_null_95 + 0.03
```

and:

```text
AUC_true_vs_false >= 0.57
```

unless pre-registered detector threshold shows a smaller effect is meaningful.

For promotion beyond exploratory:

```text
AUC_true_vs_false >= 0.60
```

preferred.

## 6.4 Negative result standard

A negative is useful only if the hidden control was live.

If hidden control fails:

```text
GATE_NOT_LIVE
```

If hidden control works and public route fails:

```text
PUBLIC_ROUTE_NO_CROSSING
```

If public route separates but controls explain it:

```text
ARTIFACT_CONFIRMED
```

---

# 7. Track 0: Odd-Lane Transfer Function Calibration

Priority: 0

Status: required before interpreting any negative.

## Hypothesis

The detector has a measurable threshold for fold-odd lane amplitude.

## Purpose

Quantify:

```text
How small of a chiral lane can this route detect?
```

Do not call this a crossing. This is instrument calibration.

## Implementation

Inject a synthetic hidden odd lane with controlled amplitude:

```text
epsilon = 1.0
epsilon = 0.5
epsilon = 0.25
epsilon = 0.125
epsilon = 0.0625
epsilon = 0.03125
epsilon = 0.0
```

Run each detector route against the epsilon ladder.

Measure:

```text
AUC(epsilon)
SNR(epsilon)
minimum detectable epsilon
false positive rate
shuffle null
detector linearity
```

## Gate

Pass if:

```text
hidden epsilon ladder is monotonic
AUC falls to chance near epsilon = 0
shuffle null stays chance
minimum detectable epsilon is reported
```

## Artifact names

```text
track0_transfer_function/odd_lane_transfer_function.rs
track0_transfer_function/analyze_transfer_function.py
track0_transfer_function/PHASE6_ODD_LANE_TRANSFER_FUNCTION.md
track0_transfer_function/results/odd_lane_transfer_function.json
```

## Verdicts

```text
ODD_LANE_DETECTOR_CALIBRATED
ODD_LANE_DETECTOR_NOT_LIVE
ODD_LANE_DETECTOR_ARTIFACT
```

---

# 8. Track A: Dual-Lane Even Cancellation

Priority: 1

## Hypothesis

The even cosine workload dominates common-mode power. If two public candidate lanes run opposite chiral candidate preparations simultaneously, the even term may cancel while a differential chiral residue remains.

## Odd-lane source

```text
differential common-mode rejection of two folded candidate lanes
```

## Main threat

```text
PHASE_JITTER_DERIVATIVE_ARTIFACT
```

If sender lanes are offset by a few cycles, subtraction can produce a derivative artifact larger than the expected residue.

## Implementation

Use:

```text
two sender cores
one receiver core
atomic spin-barrier
TSC capture on all lanes
candidate_0 lane
candidate_1 lane
matched duty
matched tape
matched public magnitude
matched final tape state
swapped-lane control
core-swap control
```

No runtime true/false label.

Candidate labels are blinded until offline scoring.

## Required variants

```text
candidate_0 vs candidate_1
candidate_1 vs candidate_0
candidate_0 vs candidate_0
candidate_1 vs candidate_1
shuffled candidate schedule
hidden differential positive control
no-sender baseline
```

## Gate

Pass if:

```text
hidden differential control AUC >= 0.95
sender start skew distribution is reported
high-skew trials are rejected or isolated
true-minus-false clears shuffle null
swapped lane preserves candidate identity, not core identity
equal-sign controls stay chance
public shuffled schedules stay chance
```

## Artifact names

```text
dual_lane_differential/chiral_dual_lane.rs
dual_lane_differential/PHASE6_DUAL_LANE_DIFFERENTIAL.md
dual_lane_differential/results/dual_lane_differential_result.json
```

## Verdicts

```text
DUAL_LANE_CHIRAL_SIGNAL_FOUND
DIFFERENTIAL_COMMON_MODE_REJECTED
PHASE_JITTER_DERIVATIVE_ARTIFACT
DUAL_LANE_PUBLIC_NO_CROSSING
DUAL_LANE_GATE_NOT_LIVE
```

---

# 9. Track B: I/Q Demodulated Receiver

Priority: 2

## Hypothesis

A chiral lane may not appear as scalar amplitude. It may appear as a quadrature phase rotation in the physical response.

## Odd-lane source

```text
phase rotation of driven receiver response
```

## Implementation

Create an I/Q lock-in receiver:

```text
I = in-phase response at carrier
Q = quadrature response at carrier
R = I + iQ
phase = atan2(Q,I)
```

The receiver should demodulate:

```text
candidate_0
candidate_1
hidden positive
equal-sign
shuffled schedule
off-frequency
phase-randomized carrier
```

## Gate

Pass if:

```text
hidden chiral I/Q control live
candidate sign rotates receiver phase above null
amplitude alone cannot classify
off-frequency controls fail
phase-randomized carrier fails
public schedule remains blinded
```

## Artifact names

```text
iq_receiver/chiral_iq_receiver.rs
iq_receiver/PHASE6_IQ_DEMOD_RECEIVER.md
iq_receiver/results/iq_receiver_result.json
```

## Verdicts

```text
QUADRATURE_RECEIVER_CHIRAL_SIGNAL_FOUND
I_Q_PUBLIC_NO_CROSSING
I_Q_AMPLITUDE_ARTIFACT
I_Q_GATE_NOT_LIVE
```

---

# 10. Track C: Chiral QFT Macro Accumulation

Priority: 3

## Hypothesis

Single chiral walks may be too small. A QFT-style phase diffraction block applies many coordinated rotations, so physical chirality may accumulate over the full block.

## Odd-lane source

```text
macro accumulation of physically bound chirality across coordinated phase rotations
```

## Implementation

Port the phase diffraction / QFT operation into the native chiral harness.

Add a chirality parameter to all rotations.

Bind chirality to physical CPU structures:

```text
candidate_0:
ROL or ascending stride or clockwise rotation

candidate_1:
ROR or descending stride or counterclockwise rotation
```

Keep output magnitudes identical.

## Required controls

```text
same public magnitude
ROL/ROR latency and uop matching
ascending/descending stride cache matching
random rotation order
reversed rotation schedule
hidden chiral QFT control
no-sender baseline
equal-sign rotation schedule
```

## Gate

Pass if:

```text
candidate_0 and candidate_1 produce equal public magnitudes
hidden chiral QFT control is live
true-sign QFT response beats false-sign response above null
reversed/random rotation schedules fail
instruction artifact controls fail
```

## Artifact names

```text
chiral_qft/chiral_qft_pdn.rs
chiral_qft/PHASE6_CHIRAL_QFT.md
chiral_qft/results/chiral_qft_result.json
```

## Verdicts

```text
MACRO_CHIRAL_QFT_SIGNAL_FOUND
QFT_PUBLIC_NO_CROSSING
QFT_ROTATION_ARTIFACT
QFT_GATE_NOT_LIVE
```

---

# 11. Track D: Commutator Lane Generation

Priority: 4

## Hypothesis

Public operations that are individually fold-even may generate a fold-odd physical residue through noncommutative execution order.

## Odd-lane source

```text
noncommutative physical order
```

## Core idea

Use two public operations:

```text
A = public candidate preparation
B = public pressure / cancellation / restore primitive
```

Run:

```text
AB
BA
```

Architectural final state and tape hash must match.

Physical execution path may differ.

The possible odd lane is:

```text
[A,B] = AB - BA
```

## Required variants

```text
AB
BA
AABB
BBAA
shuffled A/B multiset
hidden commutator positive control
equal-candidate control
candidate label swap
```

## Gate

Pass if:

```text
AB and BA final tape identical
public outputs identical
hidden commutator control live
AB-vs-BA differential beats shuffle null
true/false candidate lift survives candidate blinding
operation-order-only null fails
```

## Artifact names

```text
commutator_lane/chiral_commutator.rs
commutator_lane/PHASE6_COMMUTATOR_LANE.md
commutator_lane/results/commutator_lane_result.json
```

## Verdicts

```text
COMMUTATOR_CHIRAL_LANE_FOUND
COMMUTATOR_PUBLIC_NO_CROSSING
ORDER_ARTIFACT_CONFIRMED
COMMUTATOR_GATE_NOT_LIVE
```

---

# 12. Track E: Geometric Phase Workload Loop

Priority: 5

## Hypothesis

A closed loop in workload-parameter space may accumulate a physical geometric phase. Clockwise and counterclockwise public loops contain the same stages but different orientation.

## Odd-lane source

```text
geometric phase around a closed workload loop
```

## Implementation

Define stages:

```text
cache_pressure
integer_pressure
syscall_pressure
memory_sweep
idle
```

Construct two loops:

```text
clockwise:
cache -> integer -> syscall -> memory -> idle -> cache

counterclockwise:
cache -> idle -> memory -> syscall -> integer -> cache
```

Same stage multiset.

Same final state.

Opposite loop orientation.

## Controls

```text
randomized loop order
zero-area loop
same-stage repeated loop
hidden geometric phase positive control
thermal matched control
equal-candidate loop
```

## Gate

Pass if:

```text
hidden geometric phase control live
clockwise/counterclockwise response separates above null
same multiset shuffled loop fails
zero-area loop fails
candidate true/false lift survives blinding
```

## Artifact names

```text
geometric_loop/workload_geometric_phase.rs
geometric_loop/PHASE6_GEOMETRIC_PHASE_LOOP.md
geometric_loop/results/geometric_phase_result.json
```

## Verdicts

```text
GEOMETRIC_PHASE_CHIRAL_SIGNAL_FOUND
GEOMETRIC_LOOP_PUBLIC_NO_CROSSING
LOOP_ORDER_ARTIFACT
GEOMETRIC_GATE_NOT_LIVE
```

---

# 13. Track F: Catalytic Loschmidt Echo

Priority: 6

## Hypothesis

A reversible/catalytic loop may restore architectural tape state while leaving a physical echo residue that carries path orientation.

## Odd-lane source

```text
reversible echo residue
```

## Implementation

Run:

```text
U_candidate
U_candidate_dagger
```

Measure:

```text
forward window
reverse window
post-echo window
total runtime
```

The tape must restore bit-perfect.

Define echo residue:

```text
R_echo = response(U U_dagger) - response(identity)
```

## Controls

```text
forward-only
reverse-only
identity-only
U_dagger U order swap
same-final-hash false path
hidden echo positive control
equal-sign candidate
```

## Gate

Pass if:

```text
tape hash restored
hidden echo control live
reverse/post-echo window separates true/false
forward-only and total-runtime controls cannot explain result
same-final-hash false path fails
```

## Artifact names

```text
loschmidt_echo/catalytic_echo.rs
loschmidt_echo/PHASE6_CATALYTIC_LOSCHMIDT_ECHO.md
loschmidt_echo/results/loschmidt_echo_result.json
```

## Verdicts

```text
CATALYTIC_ECHO_CHIRAL_RESIDUE_FOUND
RESTORE_WORK_ASYMMETRY_FOUND
ECHO_PUBLIC_NO_CROSSING
ECHO_ARTIFACT_CONFIRMED
ECHO_GATE_NOT_LIVE
```

---

# 14. Track G: PDN Sideband Mixer and Bispectrum

Priority: 7

## Hypothesis

The chiral residue may be too small in baseband but visible through nonlinear sideband mixing or higher-order phase coupling.

## Odd-lane source

```text
nonlinear PDN phase coupling
```

## Implementation

Sender modulates candidate phase walk at a carrier frequency.

Receiver demodulates at:

```text
carrier
2nd harmonic
3rd harmonic
sidebands f_c ± f_m
```

Also compute bispectrum:

```text
B(f1,f2) = E[X(f1) X(f2) X*(f1+f2)]
```

## Required controls

```text
off-frequency controls
phase-randomized carrier
amplitude-matched even-only load
hidden sideband positive control
no-sender baseline
same candidate equal-sign
shuffled sign
```

## Gate

Pass if:

```text
hidden sideband/bispectrum control live
true candidate sideband phase separates from false above null
common-mode amplitude alone cannot classify
off-frequency controls fail
bispectral phase adds evidence beyond power spectrum
```

## Artifact names

```text
sideband_bispectrum/chiral_sideband_bispectrum.rs
sideband_bispectrum/analyze_bispectrum.py
sideband_bispectrum/PHASE6_PDN_SIDEBAND_BISPECTRUM.md
sideband_bispectrum/results/sideband_bispectrum_result.json
```

## Verdicts

```text
SIDEBAND_CHIRAL_SIGNAL_FOUND
NONLINEAR_PHASE_COUPLING_CHIRAL_SIGNAL_FOUND
SIDEBAND_PUBLIC_NO_CROSSING
SIDEBAND_ARTIFACT_CONFIRMED
SIDEBAND_GATE_NOT_LIVE
```

---

# 15. Track H: Physical Collision Sieve

Priority: 8

## Hypothesis

Two public oracle preparations running together may create a shared-resource interference pattern that differs between same-sign and opposite-sign chiral schedules.

## Odd-lane source

```text
physical collision asymmetry between candidate lanes
```

## Implementation

Two sender cores run synchronized public candidate lanes.

Force matched pressure blocks:

```text
integer execution port pressure
scalar add/multiply churn
load/store queue pressure
cache-bank pressure
L3-set pressure
```

Measure collision response from receiver or timing observer.

## Controls

```text
same-sign vs opposite-sign
core-pair swap
pressure-block swap
cache-bank matched control
port-pressure matched control
hidden collision positive control
schedule shuffle
```

## Gate

Pass if:

```text
hidden collision control live
same-sign vs opposite-sign public pairing separates above null
core-pair swap fails to explain
schedule shuffle fails
pressure-only control fails
```

## Artifact names

```text
collision_sieve/physical_collision_sieve.rs
collision_sieve/PHASE6_PHYSICAL_COLLISION_SIEVE.md
collision_sieve/results/collision_sieve_result.json
```

## Verdicts

```text
PHYSICAL_COLLISION_SIGNAL_FOUND
COLLISION_PUBLIC_NO_CROSSING
RESOURCE_PRESSURE_ARTIFACT
COLLISION_GATE_NOT_LIVE
```

---

# 16. Track I: Hardware Topology Chirality Map

Priority: always-before hardware selection

## Hypothesis

The physical substrate has asymmetric transport paths that must be mapped before selecting sender/receiver cores.

## Odd-lane source

This is not a crossing source. It is topology discovery.

## Implementation

Map response across core routes:

```text
sender_i -> receiver_j
sender_i + sender_k -> receiver_j
ring order i -> j -> k
ring order k -> j -> i
same-L3 pair
cross-L3 pair if on Ryzen
Phenom shared-L3 pair
```

## Output

```text
topology_chirality_matrix.json
```

Fields:

```text
core_pair
topology_domain
route_direction
mean_response
phase_response
common_mode
hidden_lane_auc
noise_floor
recommended_use
```

## Gate

Pass if:

```text
topology map is produced before Track A/C/E core choice
hidden lane route is live on chosen topology
core bias controls exist
```

## Artifact names

```text
topology_chirality/topology_chirality_map.rs
topology_chirality/PHASE6_TOPOLOGY_CHIRALITY_MAP.md
topology_chirality/results/topology_chirality_matrix.json
```

## Verdicts

```text
TOPOLOGY_MAP_COMPLETE
TOPOLOGY_ROUTE_SELECTED
TOPOLOGY_GATE_NOT_LIVE
```

---

# 17. Track J: Synthetic Transient-State Chiral Carrier

Priority: 9

Scope boundary:

This is synthetic lab-only.

It must never read, infer, or expose:

```text
real process memory
credentials
SSH keys
browser data
tokens
system secrets
third-party data
```

Generated lab buffers only.

Aggregate timing/PDN response only.

No disclosure payload.

## Hypothesis

A branch-gated transient window may leave a measurable physical carrier trace even if architectural state is discarded.

## Odd-lane source

```text
transient execution path residue
```

## Implementation

Use generated lab instances and generated labels.

Train a branch direction in a generated loop.

Place minimal chiral carrier sequence in transient window.

Measure only aggregate response.

## Controls

```text
hidden transient positive control
non-transient branch control
always-taken branch
always-not-taken branch
public branch labels no-smuggle audit
candidate label blinding
```

## Gate

Pass if:

```text
hidden transient control live
public candidate true beats false above null
non-transient and always-branch controls fail
no real memory or secret path exists
```

## Artifact names

```text
transient_carrier/transient_chiral_carrier.rs
transient_carrier/PHASE6_TRANSIENT_CHIRAL_CARRIER.md
transient_carrier/results/transient_carrier_result.json
```

## Verdicts

```text
TRANSIENT_CARRIER_SIGNAL_FOUND
TRANSIENT_PUBLIC_NO_CROSSING
TRANSIENT_ARTIFACT_CONFIRMED
TRANSIENT_ROUTE_REJECTED_NON_SYNTHETIC
TRANSIENT_GATE_NOT_LIVE
```

---

# 18. Track K: Harmonic Resonance Sweep

Priority: 10

## Hypothesis

The public oracle execution may have a physical resonance response not captured by scalar timing.

Candidate signs might split under a driven frequency sweep.

## Odd-lane source

```text
candidate-dependent resonance shift
```

## Implementation

Aggressor workload sweeps frequency.

Sender runs public candidate lane.

Receiver records response across frequency bins.

Search for:

```text
candidate-dependent resonance peak
candidate-dependent resonance phase
candidate-dependent harmonic shift
```

## Controls

```text
static workload
no-sender
shuffled frequency
off-resonance
hidden resonance positive control
candidate label blinding
```

## Gate

Pass if:

```text
hidden resonance control live
true/false resonance feature separates above null
static workload and no-sender controls fail
frequency-shuffle null fails
```

## Artifact names

```text
resonance_sweep/harmonic_resonance_sweep.rs
resonance_sweep/PHASE6_HARMONIC_RESONANCE_SWEEP.md
resonance_sweep/results/resonance_sweep_result.json
```

## Verdicts

```text
RESONANCE_SIGN_SPLIT_FOUND
RESONANCE_PUBLIC_NO_CROSSING
RESONANCE_ARTIFACT_CONFIRMED
RESONANCE_GATE_NOT_LIVE
```

---

# 19. Track L: Passive External Witness, Optional

Priority: optional but high credibility if available

## Hypothesis

A passive external witness can separate physical emission from pure TSC/software artifact.

## Scope

No firmware changes.

No board modification.

No invasive probing required for this track.

Acceptable optional witnesses:

```text
near-field EM pickup
wall-power meter
audio-interface pickup coil
scope probe only if safe and available
DMM only for slow corroboration
```

## Gate

Pass if:

```text
external witness synchronizes to run windows
hidden lane positive control visible
public candidate result, if any, correlates with external witness
distance/orientation controls behave physically
```

## Artifact names

```text
external_witness/passive_external_witness.md
external_witness/results/external_witness_log.csv
```

## Verdicts

```text
EXTERNAL_WITNESS_CORROBORATED
EXTERNAL_WITNESS_PARTIAL
EXTERNAL_WITNESS_NOT_AVAILABLE
```

---

# 20. Track M: Synthetic Quadrature Leak Ladder

Priority: calibration and theorem boundary

## Hypothesis

The transition between decodable and non-decodable can be measured by injecting controlled quadrature.

## Purpose

This does not solve the original wall.

It measures how much odd lane the substrate needs.

## Implementation

Generate synthetic public-like data:

```text
z_k = cos(theta_k) + i * epsilon * sin(theta_k)
```

Sweep epsilon.

Test all detectors and routes.

## Gate

Pass if:

```text
epsilon threshold measured
phase-cavity / I/Q / QFT / dual-lane sensitivity compared
epsilon=0 stays chance
epsilon>threshold recovers orientation
```

## Artifact names

```text
quadrature_ladder/synthetic_quadrature_ladder.rs
quadrature_ladder/PHASE6_SYNTHETIC_QUADRATURE_LADDER.md
quadrature_ladder/results/quadrature_ladder_result.json
```

## Verdicts

```text
QUADRATURE_THRESHOLD_MEASURED
QUADRATURE_LADDER_ARTIFACT
QUADRATURE_LADDER_GATE_NOT_LIVE
```

---

# 21. Track N: Hardware Decodability Boundary Ledger

Priority: always-on

## Purpose

Maintain the scientific ledger separating:

```text
transport works
lane synthesis works
lane synthesis fails
detector not live
artifact killed
remaining assumption
```

This is the publishable negative-result engine.

## Required table fields

```text
route
odd_lane_source
hidden_control_status
public_status
artifact_status
sizes_tested
seeds_tested
substrates_tested
claim_level
remaining_assumption
next_action
```

## Artifact names

```text
PHASE6_HARDWARE_DECODABILITY_BOUNDARY.md
results/phase6_frontier_route_table.json
```

## Verdicts

```text
HARDWARE_DECODABILITY_BOUNDARY_UPDATED
PUBLIC_LANE_SYNTHESIS_NOT_FOUND_TO_THRESHOLD
ROUTE_TABLE_INCOMPLETE
```

---

# 22. Track Z: Orientation Conservation Audit

Priority: always-before-build

## Purpose

Every proposed route must name its proposed odd-lane source before code is written.

If a route cannot name where fold symmetry is broken, do not build it.

## Required questions

For each route:

```text
What breaks fold symmetry?
Where does the odd representation enter?
Is it public physical geometry, hidden control, or smuggle?
Does the emitted schedule change under d ↔ N-d?
Does the final tape restore?
Does the public path know true/false?
What controls would falsify it?
What outcome would count as negative?
```

## Artifact names

```text
orientation_conservation/ORIENTATION_CONSERVATION_AUDIT.md
orientation_conservation/schedule_invariance_audit.py
orientation_conservation/results/schedule_invariance.json
```

## Verdicts

```text
ORIENTATION_SOURCE_NAMED
ORIENTATION_SOURCE_ABSENT_PREDICT_NEGATIVE
SCHEDULE_SMUGGLE_FOUND
SCHEDULE_INVARIANCE_PASS
```

---

# 23. Execution Order

## Sprint 0: Gate and detector spine

Build first:

```text
Track Z: Orientation conservation audit
Track 0: Odd-lane transfer function
Track I: Topology chirality map
Track B: I/Q receiver base layer
```

Do not run major public crossing tests until detector threshold is known.

## Sprint 1: Primary public lane attempt

Run:

```text
Track A: Dual-lane even cancellation
```

Minimum:

```text
n=8, n=10
42 paired instances per mode
hidden differential AUC >= 0.95
candidate labels blinded
```

Promotion:

```text
Track A positive must be repeated with lane swap, core swap, equal-sign null, and schedule shuffle.
```

## Sprint 2: Phase-geometry attacks

Run:

```text
Track C: Chiral QFT macro accumulation
Track D: Commutator lane
Track E: Geometric phase workload loop
Track F: Catalytic Loschmidt echo
```

These are the highest-value conceptual attacks.

## Sprint 3: Nonlinear detector attacks

Run:

```text
Track G: Sideband and bispectrum
Track K: Harmonic resonance sweep
Track H: Physical collision sieve
```

Use these after detector spine is calibrated.

## Sprint 4: Optional external corroboration

Run Track L if any passive external witness is available.

## Sprint 5: Boundary decision

After Tracks A, C, D, E, F, and B/G have either passed or failed under live hidden controls, update:

```text
PHASE6_HARDWARE_DECODABILITY_BOUNDARY.md
```

and decide:

```text
FRONTIER_REMAINS_OPEN
```

or:

```text
PUBLIC_LANE_GENERATION_MEASURED_CLOSED_TO_EPSILON_THRESHOLD
```

---

# 24. Promotion Rules

## 24.1 Route positive

A route can be called positive only if:

```text
hidden control live
public true/false separation above null
candidate labels blinded
random private fold passes
workload matched
tape restored
core/lane swaps pass
schedule shuffle fails
at least two n sizes
```

## 24.2 Frontier positive

A route can be called frontier-positive if:

```text
route positive
effect repeats on independent seeds
route survives adjudication substrate or cross-platform check
claim level >= L4
```

## 24.3 Wall break candidate

A result can be called a wall-break candidate only if:

```text
frontier positive
scaling begins
effect does not decay as random guessing
effect beats declared forward baseline
no-smuggle gate clean
```

## 24.4 Wall broken

A result can be called wall-broken only if:

```text
public chiral lane generated
orientation recovered or posterior odds lifted
scaling below forward O(N)
controls fail
independent reproduction
claim level L6
```

---

# 25. Artifact Tree

Recommended tree:

```text
phase6_chiral_lane_frontier/
    ROADMAP_2.md

    orientation_conservation/
        ORIENTATION_CONSERVATION_AUDIT.md
        schedule_invariance_audit.py
        results/

    track0_transfer_function/
        odd_lane_transfer_function.rs
        analyze_transfer_function.py
        PHASE6_ODD_LANE_TRANSFER_FUNCTION.md
        results/

    topology_chirality/
        topology_chirality_map.rs
        PHASE6_TOPOLOGY_CHIRALITY_MAP.md
        results/

    dual_lane_differential/
        chiral_dual_lane.rs
        PHASE6_DUAL_LANE_DIFFERENTIAL.md
        results/

    iq_receiver/
        chiral_iq_receiver.rs
        PHASE6_IQ_DEMOD_RECEIVER.md
        results/

    chiral_qft/
        chiral_qft_pdn.rs
        PHASE6_CHIRAL_QFT.md
        results/

    commutator_lane/
        chiral_commutator.rs
        PHASE6_COMMUTATOR_LANE.md
        results/

    geometric_loop/
        workload_geometric_phase.rs
        PHASE6_GEOMETRIC_PHASE_LOOP.md
        results/

    loschmidt_echo/
        catalytic_echo.rs
        PHASE6_CATALYTIC_LOSCHMIDT_ECHO.md
        results/

    sideband_bispectrum/
        chiral_sideband_bispectrum.rs
        analyze_bispectrum.py
        PHASE6_PDN_SIDEBAND_BISPECTRUM.md
        results/

    collision_sieve/
        physical_collision_sieve.rs
        PHASE6_PHYSICAL_COLLISION_SIEVE.md
        results/

    transient_carrier/
        transient_chiral_carrier.rs
        PHASE6_TRANSIENT_CHIRAL_CARRIER.md
        results/

    resonance_sweep/
        harmonic_resonance_sweep.rs
        PHASE6_HARMONIC_RESONANCE_SWEEP.md
        results/

    quadrature_ladder/
        synthetic_quadrature_ladder.rs
        PHASE6_SYNTHETIC_QUADRATURE_LADDER.md
        results/

    external_witness/
        passive_external_witness.md
        results/

    results/
        phase6_frontier_route_table.json
        phase6_master_verdict.json

    PHASE6_HARDWARE_DECODABILITY_BOUNDARY.md
```

---

# 26. Standard Result JSON Schema

Each track must emit JSON like:

```json
{
  "track": "dual_lane_differential",
  "route": "Track A",
  "odd_lane_source": "differential common-mode rejection",
  "n_values": [8, 10],
  "n_instances": 84,
  "hidden_control": {
    "live": true,
    "auc": 0.97
  },
  "public_result": {
    "auc_true_vs_false": 0.58,
    "shuffle_null_95": 0.53,
    "lift_over_null": 0.05,
    "effect_size": 0.31,
    "bootstrap_ci": [0.55, 0.61]
  },
  "controls": {
    "lane_swap_pass": true,
    "core_swap_pass": true,
    "equal_sign_null_pass": true,
    "schedule_shuffle_pass": true,
    "random_private_fold_pass": true,
    "restoration_pass": true,
    "schedule_invariance_pass": true
  },
  "claim_level": "L4",
  "verdict": "DUAL_LANE_CHIRAL_SIGNAL_FOUND",
  "notes": ""
}
```

---

# 27. Do-Not-Do List

Do not repeat closed routes.

```text
No more generic public cosine transforms.
No more scalar classifiers trying to recover orientation.
No more re-encoding without an odd-lane source.
No more hidden controls presented as crossings.
No more public results without blinded candidate labels.
No more firmware changes.
No voltage writes.
No real secrets.
No process memory extraction.
No transient experiment outside synthetic generated buffers.
No expansion into a new phase until A/C/D/E/F plus detector calibration are resolved.
```

---

# 28. Positive Result Interpretation

If one route passes, do not immediately declare the lattice wall broken.

Declare:

```text
PUBLIC_CHIRAL_LANE_GENERATED_CANDIDATE
```

Then run:

```text
scaling battery
cross-seed replication
cross-core replication
Phenom adjudication if result was first found elsewhere
folded-twin audit
independent implementation
```

Then upgrade only if scaling supports it.

---

# 29. Negative Result Interpretation

If all routes fail under live hidden controls, the result is not “nothing happened.”

It is:

```text
PUBLIC_LANE_GENERATION_NOT_FOUND_TO_DETECTOR_THRESHOLD
```

with a detector threshold from Track 0.

This becomes a strong hardware decodability boundary:

```text
physical carriers can transport a bound chiral lane
but tested public oracle executions do not synthesize that lane
```

This is publishable if the route table is complete.

---

# 30. Final Frontier Statement

The final wall is not a lack of readout.

The final wall is not a lack of phase substrate.

The final wall is not the even fold magnitude.

The final wall is:

```text
public generation of a fold-odd lane from fold-even data
```

This roadmap attacks that wall through:

```text
differential cancellation
I/Q demodulation
macro phase accumulation
noncommutative execution order
geometric phase loops
reversible echo residue
nonlinear sideband mixing
physical collision asymmetry
topology chirality
synthetic transient carrier tests
detector threshold calibration
orientation conservation auditing
```

Every track must answer:

```text
Where does the odd lane come from?
```

If the track cannot answer that, do not build it.

If the track can answer that and passes the gate, the wall has a crack.

If multiple tracks fail with live controls, the boundary becomes sharper.

The mission is not to believe the wall breaks.

The mission is to make the wall choose:

```text
generate a chiral lane
```

or:

```text
become a measured decodability boundary
```

Either result advances CAT_CAS.

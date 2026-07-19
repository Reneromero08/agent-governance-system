# P0 resonance and load-law repair contract

**Status:** `P0_RESONANCE_LOAD_LAW_REPAIR_CONTRACT_FROZEN`  
**Parent audit:** `P0_POST_QUALIFICATION_AUDIT.md`  
**Preserved result:** `P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED`  
**Inherited decision:** `P0_BUILD_READINESS_BLOCKED`  
**Claim ceiling:** `NON_EXECUTING_P0_RESONANCE_LOAD_LAW_REPAIR_ONLY`  
**Authority:** inherited `AUTHORIZE P0 BUILD-READINESS ONLY`

This contract authorizes only offline architecture, modeling, implementation, synthetic evidence, review, commit, and push. It authorizes no vendor contact, cart action, purchase, fabrication, assembly, wiring, probing, power, instrument command, playback, recording, acquisition, calibration, target contact, or physical claim.

## 1. Mechanism defect

The exact carrier ordering identity is a 12.5 pF-load FC-135 crystal. The final carrier node is modeled and constrained near 3.2-4.0 pF. The current source and analyzer nevertheless freeze 32768 Hz before the as-built loaded resonance exists.

The committed BVD relation therefore predicts a material resonance range rather than exact nominal operation. The build-readiness packet has not yet shown that the selected source amplitude, preparation interval, detector loading, and fixed analysis law will prepare and observe the intended mechanical state across the selected-part uncertainty region.

## 2. Required architecture decision

Reconstruct and compare the strongest coherent forms of exactly these two routes:

### Route A: load-matched carrier

Add an explicit, exact, low-loss load network that makes the selected crystal's effective load conform to its 12.5 pF ordering identity.

Route A must re-close:

```text
OPA810 input-admittance law
carrier terminal voltage/current/power
loaded Q and ringdown duration
source-off feedthrough
C2 actual-path witness transfer
empty and 1 pF dummy controls
board and cable parasitics
noise and uncertainty
```

Do not add capacitance merely to force 32768 Hz while destroying observability or source separation.

### Route B: calibration-derived carrier frequency

Retain the low-capacitance sense node and treat the selected part's actual loaded resonance as a prospectively measured calibration coordinate.

Route B must freeze before any future primary arm:

```text
search interval
frequency grid or estimator
calibration excitation limit
calibration stopping law
resonance-selection law
uncertainty law
accepted resonance/Q region
f_carrier binding
f_witness binding
source queryback binding
analyzer and schema binding
retry/no-retry law
```

The calibration record may select the frequency only from predeclared resonance data obtained before arm assignment or primary observation. It may not select a favorable phase, decay, or antipodal result.

The analyzer must receive the bound frequency through strict custody. A mutable metadata scalar is insufficient. Frequency identity must be bound to calibration bytes, source queryback, topology receipt, native record, and threshold package.

## 3. Dynamic carrier model

The repair must add a deterministic time-domain or analytically equivalent dynamic model from source preparation through source isolation and ringdown.

At minimum model:

```text
selected FC-135 BVD parameter region
actual external load region
source and limiter impedance
ADG1419 DRIVE and OFF networks
K1/K2 closed and open parasitics
OPA810 and digitizer loading
preparation duration
source phase 0 and pi
switching transient
post-source free decay
ADC/noise observation
```

Required outputs:

```text
loaded resonance interval
series and loaded Q intervals
linewidth interval
steady-state preparation amplitude interval
time to reach the frozen preparation fraction
post-source initial ringdown amplitude interval
usable-cycle interval
phase uncertainty interval
expected source-off transfer/feedthrough interval
```

A favorable synthetic ringdown inserted after the circuit is not a dynamic closure.

## 4. Frequency-native analyzer repair

Remove hidden dependence on module constants where the physical frequency is not fixed by the acquired topology.

The strict evidence object must bind:

```text
f_carrier_hz
f_carrier_u95_hz
f_witness_hz
frequency-relation law
calibration artifact SHA-256
calibration analyzer SHA-256
source queryback SHA-256
```

Every relevant analyzer operation must use the bound value:

```text
drive/reference joint fit
actual-path transfer fit
I/Q projection
phase slope and frequency reconstruction
off-resonance controls
minimum-cycle gates
matched-arm comparison
```

If `f_witness = 2 * f_carrier` is retained, preserve the C1-only nonlinear-control law and prove that mechanical/electronic 2f residue cannot satisfy the path witness. If the harmonic relation cannot close, redesign the gauge and witness without collapsing the phase law.

## 5. Continuous uncertainty closure

Replace the phrase `complete uncertainty envelope` with `complete binary-corner sweep` unless a rigorous continuous enclosure is added.

To restore a continuous-envelope claim, provide at least one:

```text
analytic monotonicity and stationary-point proof
validated interval arithmetic
global branch-and-bound enclosure
another explicitly justified rigorous enclosure method
```

The method must cover complex transfer magnitude and phase, resonance, Q, preparation amplitude, ringdown amplitude, uncertainty, perturbation, current, and power. Dense random or grid sampling may be supplemental only.

## 6. Common-mode observability repair

The current differential payload cannot by itself prove true instrument input common mode.

Freeze one lawful route:

```text
A. acquire and bind each input leg relative to the declared reference;
B. add an independently calibrated common-mode witness;
C. prove from the exact return network and instrument admittance that the derived differential bound is conservative;
D. replace the unobservable common-mode metric with a directly observable operating-envelope gate.
```

The selected route must survive ground-return, channel-swap, negative-leg drift, and digitizer-admittance controls.

## 7. Required controls

Add at minimum:

```text
nominal 12.5 pF load versus actual low-load state
frequency selected before versus after primary observation
nominal 32768 drive under worst detuning
calibrated-resonance drive
wrong calibration artifact
replayed calibration across assembly or event
mutated f_carrier metadata
source queryback frequency mismatch
witness frequency mismatch
loaded-Q outside accepted region
preparation too short
ringdown below minimum observable amplitude
nonlinear 2f residue
common-mode/negative-leg drift
continuous-interior parameter challenge against corner bounds
```

Each control must fail for its declared mechanism or custody reason.

## 8. Claim and status law

The signal-path result remains:

```text
P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED
```

The build-readiness state remains:

```text
P0_BUILD_READINESS_BLOCKED
```

until this repair emits exactly one:

```text
P0_RESONANCE_LOAD_LAW_REPAIR_ESTABLISHED
P0_RESONANCE_LOAD_LAW_REPAIR_BLOCKED
P0_RESONANCE_LOAD_LAW_REPAIR_INCONCLUSIVE
```

`ESTABLISHED` additionally requires closure of the continuous-envelope wording and common-mode observability findings, four exact-root independent PASS reviews, zero open material findings, deterministic reproduction, mutation qualification, and coherent regeneration of all affected netlist, BOM, fabrication, analyzer, schema, fixture, result, review, finding, roadmap, and authority artifacts.

Only then may the package restore:

```text
P0_BUILD_READINESS_PACKET_FROZEN
```

The later authority boundary remains:

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

Nothing in this contract grants that authority.

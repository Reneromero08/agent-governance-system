# CAT_CAS Audio-Frequency Wave Substrate Side Quest Roadmap

**Status:** `ACTIVE_PARALLEL_LANE__P0_BUILD_READINESS_BLOCKED`<br>
**Roadmap version:** `1.9`<br>
**P0 authority:** `AUTHORIZE P0 BUILD-READINESS ONLY`<br>
**Branch:** `codex/audio-frequency-wave-substrate`<br>
**Base main commit:** `32b5af119a03bc48bb00f279e6cc0014406147ad`<br>
**Frozen offline foundation commit:** `f028dfd292d2dd0fd11380459417d5b60f936ee3`<br>
**Roadmap commit binding:** the Git commit containing this file<br>
**Current accepted results:** `AUDIO_FM_WAVE_ALGEBRA_ESTABLISHED`; `AUDIO_RECURSIVE_PHASE_TREE_REFERENCE_ESTABLISHED`; `AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED`; `AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED`; `AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED`<br>
**Recursive phase-tree software reference candidate:** `false`<br>
**Recursive phase-tree software reference established:** `true`<br>
**Recursive wave operator established:** `true`<br>
**Software catalytic wave loop established:** `true`<br>
**Recursive catalytic Ising emulator established:** `true`<br>
**Physical carrier architecture frozen:** `true`<br>
**Physical carrier characterized:** `false`<br>
**Physical audio computing established:** `false`<br>
**Small Wall crossed:** `false`

This file supersedes the uncommitted version-1.1 roadmap artifact. The stale
`3329b2aca94491d7bc4dc17efa7ac14e671c1c8c` reference is not repository authority.
The containing commit of this file is the first Git-bound roadmap identity.

---

## 1. Mission

Build a recursive catalytic wave computer whose native information object is not a flat
bit vector or a flat multitone spectrum, but a hierarchy of signals embedded inside the
phase of other signals.

The target lifecycle is:

```text
recursive phase object
-> recursive wave evolution
-> hierarchy-sensitive invariant extraction
-> explicit boundary projection
-> reverse traversal
-> carrier restoration
-> surviving result reused at the next scale
```

The first discrete application is an Ising projection. Ising is not the primitive
architecture. A spin is the global `0/pi` orientation of a complete recursive phase beam.

---

## 2. Preserved Foundation

The frozen `audio_fm_wave_v1/` package remains immutable. It establishes deterministic
ordinary-software FM, PM, analytic-signal, complex mixing, multitone, delay,
filter-bank, correlation, convolution, and nonlinear-wave algebra.

```text
reference tests:        29 PASS, 0 FAIL
WAV fixtures:           12
fixture bytes:          5,376,528
fixture manifest SHA:   3e10d0ecbf535b795febba7c56f261a58d7ed3a67e8c4ecee7b030b1bff53049
test specification SHA: 0b38d590c5afaf2835bd00aa98865ad33c20e38aabbfdf9f1358a0d688c1c712
reference results SHA:  d884152e21b2e6226557879299bdf1c90f4c805279a0bbd826b66548e8075877
normalized findings:    452d6f6bd9786897a9e555f510533441ec61fb9f61a3f25d0ebf48b6488555c5
```

Ordinary software replay remains expected. No physical claim is inherited from this
foundation.

---

## 3. Architecture Correction

The version-1.1 roadmap treated the immediate successor primarily as a recursively
updated bank of flat sine waves:

```text
frame -> coefficients -> J/h mixer -> two-phase lock -> next flat frame
```

That construction remains a useful collapsed baseline. It is not the full target.

The corrected primitive is a recursive phase tree. For node `v`:

```text
Phi_v(t) =
    2*pi*f_v*t + theta_v
    + sum_(c in children(v)) beta_vc * sin(Phi_c(t))
```

The emitted complex beam is:

```text
B_T(t) = A_T(t) * exp(i * Phi_root(t))
```

A complete beam may receive a global Ising orientation:

```text
Psi_i(t) = exp(i * theta_i_spin) * B_Ti(t)
theta_i_spin in {0, pi}
```

The global `Z2` action must rotate the whole recursive object without destroying its
internal modulation tree.

Three recursions are required:

```text
depth recursion:
    signal inside phase inside phase

temporal recursion:
    complete phase tree at k becomes input to k+1

catalytic recursion:
    child carriers are restored while the extracted parent invariant
    becomes a phase-bearing input at the next scale
```

---

## 4. Native Process Object

```text
RecursivePhaseTree {
    root
    nodes
    parent_child_edges
    carrier_frequencies
    modulation_indices
    local_reference_phases
    global_spin_phase
    embedded_relation_channels
    temporal_history
    evolution_operator
    query_operator
    invariant_extract
    restoration_operator
    collapse_boundary
}
```

The native state remains complex. A decoded spin, scalar energy, FFT magnitude, or
winner label is a boundary shadow and may not feed the next recursive state unless that
reduction is the declared object under test.

---

## 5. Package Sequence

```text
audio_fm_wave_v1/
    frozen primitive wave algebra

audio_recursive_phase_tree_v1/
    nested phase representation
    canonical tree identity
    hierarchy-sensitive query
    reversible phase operator
    thin deterministic software reference

audio_recursive_wave_operator_v1/
    complete-tree coupling and temporal recurrence
    no scalar feedback

audio_catalytic_wave_loop_v1/
    borrowed carrier mutation
    invariant latch
    reverse traversal
    restoration and wrong-inverse controls

audio_recursive_catalytic_ising_v1/
    global Z2 Ising action on complete trees
    physical-translation contract
    exact-enumeration bounded oracle used only at adjudication
```

Each package is separate. No successor may rewrite the frozen evidence of a predecessor.

---

## 6. Established Recursive Phase-Tree Reference

`audio_recursive_phase_tree_v1/recursive_phase_tree_reference.py` closes the first
bounded ordinary-software reference.

Its bounded mechanism is:

```text
deterministic dirty complex tape
-> multiply by unit-modulus recursive phase beam
-> hierarchy-sensitive matched query
-> multiply by conjugate beam
-> verify tape restoration
```

The reference also checks:

```text
depth-3 nesting
same node multiset with different parent-child geometry
unit-modulus carrier
global pi rotation of the complete tree
amplitude-only exact null
hierarchy-sensitive query separation
nonzero borrowed-tape mutation
correct inverse restoration
wrong-hierarchy inverse failure
canonical deterministic tree identity
```

Committed-byte qualification receipt:

```text
source commit:              the implementation commit containing the evidence packet
source Git blob SHA-1:      956adb0ae8e84c091c1dc1e3de650be374fa96d1
source SHA-256:             e5911cb868f244ac69f3f8f8c4cfa83440385347be2d4526d5f25376de736887
schema SHA-256:             41648ef97b94a4ae0b00a95d3fbbd081158183671626ec8ccf5473b77681b974
manifest SHA-256:           7112307fa4406cf4880736545a88e56c45fafc6f27cd0a6518a1b40963fb62fa
fixture-set SHA-256:        6afb8adb0d14ab2e5a750df519ced073475fbf1554ee8be0732a2ebde5e15925
tests SHA-256:              3cecfa9f0d79babc4f9d76d7b463a1b8f825e209f2af592e590c52686dc95b2c
result SHA-256:             46e2cc7cb72217c647f8653ebe61a0dbf2060a222de0eec6624fbb7fbcb94eab
fixtures:                   3 stereo I/Q WAVs, 144132 bytes
tests:                      38 PASS, 0 FAIL
committed-byte scoring:     true
tree-to-WAV binding:        exact
independent reviews:        4 PASS, 0 open material findings
claim ceiling:              SOFTWARE_RECURSIVE_PHASE_TREE_REFERENCE_ONLY
```

The result establishes only the bounded R0 meaning below. It does not establish
temporal recurrence, a catalytic loop, Ising evolution, physical restoration, physical
audio computing, optimization advantage, capacity separation, or a Wall crossing.

---

## 7. Claim Ladder

### A0: Flat offline wave algebra

```text
AUDIO_FM_WAVE_ALGEBRA_ESTABLISHED
```

Current state: **established**.

### R0: Recursive phase-tree software reference

```text
AUDIO_RECURSIVE_PHASE_TREE_REFERENCE_ESTABLISHED
```

Meaning:

```text
a committed deterministic reference represents nested phase geometry,
distinguishes hierarchy through phase while amplitude-only controls remain null,
mutates a borrowed software carrier, and restores it under the correct inverse
```

Current state: **established**.

### R1: Recursive wave operator

```text
AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED
```

Meaning:

```text
a frozen operator evolves complete recursive trees across iterations without
decoding them to scalar spins between steps
```

Current state: **established**.

### R2S: Software catalytic wave loop

```text
AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED
```

Meaning:

```text
a nonzero software carrier displacement, hierarchy-sensitive invariant latch,
reverse traversal, restoration, and wrong/reordered inverse rejection close
under a frozen contract
```

This is not physical R2 restoration.

Current state: **established**.

R2S receipt:

```text
source Git blob SHA-1     63eed91f74252082b1258755bdd4371a2a48e105
source SHA-256            6c55861da950caf0738bb5ffb676f0c458a593a805ddd49419d6b2b427f6c33c
fixture manifest          5e8bfa247c513d189774ec671265b2d3dc1ea97004e5e8c40baa090f26db3cad
fixture set               e6e51ae655e184f8f43b2afa9fe0c75041046966b4cdecd6fde008b02b684aa8
reference tests           ef888d8d8b48b2fbdc7897d6d42aa2f63f8c300517f6d9b8911346bf285438c6
reference result          bee5727f68fc10ee047d666198b3f060f669058e966aa44802e270f90abbdeeb
qualification             78 PASS / 0 FAIL
independent reviews       4 PASS / 0 FAIL
open material findings    0
claim ceiling             SOFTWARE_CATALYTIC_WAVE_LOOP_REFERENCE_ONLY
```

### R3: Recursive catalytic Ising emulator

```text
AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED
```

Meaning:

```text
global 0/pi orientation acts on complete recursive phase trees,
the frozen coupling law evolves those trees without scalar feedback,
and final boundary projections agree with independently generated bounded optima
```

Current state: **established as a bounded offline numerical emulator**.

R3 receipt:

```text
source Git blob SHA-1     ac01c64d15498355daa844c7e3adba99b2fcc73a
source bytes              146825
source SHA-256            076fa3f392a9a0f1307e222deeabef38d558bf93db10c317af481ee40bf17b48
fixture manifest          0b83917e4d71575d6300f5d92f60e8e5f439e894375ee100eeef8571fccdc7ae
fixture set               653f8c19d4686c972c6c7a10a8283ee8564322f6f3bf3012aa6bea8b1c2b3d5b
reference tests           40bae1deea55baca1e909f0adc6c93b47350ae463cb267f8e320c3b0bbc05c70
reference result          27720d0a7fb1125291965d5f706e03ba6d707fc9cc4523fefb12516391e9ca3d
trajectory SHA-256        135a3f8231e4ac6ddfb575c7ac684111409da650260a03b065e7a7b0078ca196
fixture packet            13 files / 55520 bytes
qualification             106 PASS / 0 FAIL
independent reviews       4 PASS / 0 FAIL
open material findings    0
claim ceiling             SOFTWARE_RECURSIVE_PHASE_ISING_EMULATOR_ONLY
```

### P1+: Physical carrier ladder

Physical architecture, post-source state, query operator, tomography, physical R2,
relational carrier, catalytic transform, and Wall decisions remain separate later rungs.

---

## 8. No-Smuggle Requirements

The recursive mechanism must survive:

```text
amplitude-only decoder
spectrum-magnitude-only decoder
flat multitone replacement
parent-child phase scramble
subtree permutation
wrong recursive query
wrong inverse
reordered inverse
natural relaxation
serialized trajectory replay
compressed recurrence replay
manifest and filename leakage scan
```

A decisive hierarchy test should compare states with matched duration, energy,
time-domain magnitude, node identities, and public parameters while changing only the
parent-child phase geometry.

---

## 9. Ising Boundary

The first Ising implementation must preserve:

```text
complete recursive beam during evolution
internal relative phase structure
global Z2 orientation
explicit J/h coupling channel
phase-lock operator acting on global orientation only
final scalar decode at an explicit CollapseBoundary
```

Forbidden native recurrence:

```text
decode tree to +/-1
discard internal phase geometry
compute J@s as the hidden machine
resynthesize a new flat sine bank
```

That construction is allowed only as a declared ordinary baseline.

---

## 10. Physical Successor

The physical target requires a carrier whose dynamics perform both coupling and
phase-bistable locking. Candidate order:

```text
1. active audio-frequency feedback or delay network
2. coupled nonlinear or parametrically phase-bistable electrical resonators
3. coupled electromechanical oscillators
4. solid-state acoustic delay-line or resonator network
5. sealed acoustic cavity after termination and environmental controls close
```

A passive single resonator may test persistence, but cannot implement arbitrary
programmable coupling by itself.

No audio playback, recording, DAC/ADC operation, powered circuit action, speaker or
transducer operation, or physical restoration is authorized by this roadmap.

---

## 11. Selected P0 architecture boundary

The explicitly selected successor is:

```text
PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_AND_MEASUREMENT_PACKET
```

It records one non-executing mechanical carrier/access architecture and a
separate build-readiness candidate; the latter is not frozen:

```text
preferred:
    Epson FC-135 Q13FC1350000401 electrode-addressed 32.768 kHz quartz tuning-fork mode

fallback:
    localized external-drive PZT/brass diaphragm mode near 6.3 kHz

source-off:
    calibrated-f_ref SIGLENT SDG1032X through ADG1419 gate; K1/K2 release and
    stabilize before K3 may guard; auxiliary CH2 timing does not prove the
    actual signal poles, so physical source-disconnect authority remains blocked

acquisition:
    Spectrum DN2.592-04 four simultaneous true-differential paths; canonical 4 x 3,101,000 signed-int16 raw record

build-readiness candidate:
    revised complete-fixture topology, derived BOM, coordinate release, continuous dual-tone phase gauge, source-off witness, analyzer suites, and root-bound validator are ready for exact-byte independent review

claim ceiling:
    NON_EXECUTING_P0_BUILD_READINESS_ONLY
```

It does not establish physical resonance, post-source persistence, a pi relation,
physical computing, restoration, silicon operation, bit replacement, or a Wall crossing.

The still-unreached next authority boundary is:

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

The lane remains blocked before that boundary by the missing per-event actual signal-pole witness or exact force-guided-contact guarantee. No predecessor package, `main`, Family 10h
evidence, `SMALL_WALL_STATE.md`, hardware, target, or preserved stash is touched.

---

## 12. Current Decision

```text
AUDIO_RECURSIVE_CATALYTIC_ARCHITECTURE_CORRECTED
AUDIO_RECURSIVE_PHASE_FOUNDATION_STARTED
AUDIO_RECURSIVE_PHASE_TREE_REFERENCE_ESTABLISHED
AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED
AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED
AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED
PHYSICAL_AUDIO_COMPUTING_NOT_ESTABLISHED
SMALL_WALL_CROSSED_NOT_PROMOTED
PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_AND_MEASUREMENT_PACKET
PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_PACKET_FROZEN
P0_BUILD_READINESS_BLOCKED
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

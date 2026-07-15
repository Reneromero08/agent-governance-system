# Audio-Frequency Wave Substrate Lane Bootstrap

Status: `BOOTSTRAPPED_AWAITING_IMPLEMENTATION`

Branch:

```text
codex/audio-frequency-wave-substrate
```

Base commit:

```text
32b5af119a03bc48bb00f279e6cc0014406147ad
```

This file is the read-first handoff for an agent working only on the audio-frequency wave substrate lane. The Family 10h carrier-state tomography lane continues separately on `main` or its own repair branch. Do not modify the Family 10h tomography package, its review records, or global Small Wall state from this branch.

## 1. Why This Lane Exists

The retained Family 10h work establishes:

```text
GAIN_COVARIANT_ORBITSTATE_PROJECTION_TRANSDUCTION_ESTABLISHED
```

The current query-separated CPU successor remains:

```text
QUERY_SEPARATED_ARCHITECTURE_BLOCKED
QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED
SMALL_WALL_CROSSED_NOT_PROMOTED
```

The CPU lane is blocked at the substrate-identification layer because the persistent post-source physical state, query operator, preparation capacity, and R2 restoration law are not yet established.

This audio lane explores an alternative phase-native substrate where frequency, phase, delay, interference, spectral geometry, and ring-down can be measured explicitly.

The lane begins with offline WAV-based algebra and architecture work. Offline DSP is not physical wave computing. It is a syntax, adversary, custody, and experiment-design laboratory for a later DAC-carrier-ADC prototype.

## 2. Non-Negotiable Claim Boundary

Allowed offline result:

```text
AUDIO_FM_WAVE_ALGEBRA_ESTABLISHED
```

Possible design decisions:

```text
AUDIO_FREQUENCY_WAVE_ARCHITECTURE_FROZEN_READY_FOR_PHYSICAL_PROTOTYPE
AUDIO_FREQUENCY_WAVE_ARCHITECTURE_BLOCKED
AUDIO_FREQUENCY_WAVE_ARCHITECTURE_INCONCLUSIVE
```

Forbidden in this branch:

```text
AUDIO_POST_SOURCE_STATE_OBSERVED
PHYSICAL_AUDIO_COMPUTING_ESTABLISHED
RELATIONAL_CARRIER_ESTABLISHED
PHYSICAL_RELATIONAL_MEMORY_ESTABLISHED
CATALYTIC_BORROWING_ESTABLISHED
SMALL_WALL_CROSSED
```

A WAV file is serialized state. If Python performs the transform, the CPU is the computer. The offline engine may prove exact wave algebra and attack ordinary explanations, but it cannot prove a physical carrier performed computation.

## 3. Branch and Custody Rules

Work only on:

```text
codex/audio-frequency-wave-substrate
```

Recommended local worktree:

```text
D:\CCC 2.0\AI\agent-governance-system-audio
```

Create it from the already-existing remote branch if it does not exist:

```powershell
git fetch origin
git worktree add "D:\CCC 2.0\AI\agent-governance-system-audio" codex/audio-frequency-wave-substrate
```

If the branch is not locally registered, use:

```powershell
git worktree add -b codex/audio-frequency-wave-substrate `
  "D:\CCC 2.0\AI\agent-governance-system-audio" `
  origin/codex/audio-frequency-wave-substrate
```

Before writing, require:

```text
branch = codex/audio-frequency-wave-substrate
working tree = clean
branch descends from 32b5af119a03bc48bb00f279e6cc0014406147ad
```

Do not:

```text
modify main
merge main into this branch during the initial lane package
modify the Family 10h tomography package
modify SMALL_WALL_STATE.md
apply or delete existing stashes
use SSH, SCP, ping, or target inspection
contact an audio device
play audio through DAC
record ADC or microphone input
create a live audio controller
```

Offline WAV generation and deterministic numerical analysis are allowed.

## 4. Operating Principles

Apply the CAT_CAS anti-collapse protocol:

```text
preserve phase and complex geometry
preserve unresolved state where the design actually contains it
distinguish relational state from sender-authored projection
separate source preparation from receiver query
identify carrier, observable, invariant, restoration law, and no-smuggle boundary
attack scalar replay before positive claims
name exact claim ceilings
```

Do not force the project into conventional ML classification merely because convenient libraries exist. Classification may be used as a diagnostic, but the primary objects are wave state, operator, query, geometry, interference, and restoration.

## 5. Required Package Root

Continue under:

```text
THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/
50_6_fixed_point_substrate/14_noncollapse_frontier/
phase6b6/live_small_wall/audio_frequency_wave_substrate/
audio_fm_wave_v1/
```

Keep this bootstrap file.

Create at minimum:

```text
AUDIO_WAVE_SUBSTRATE_CHARTER.md
AUDIO_WAVE_STATE_MODEL.md
FM_PM_MULTITONE_ALGEBRA.md
AUDIO_QUERY_SEPARATION_LAW.md
AUDIO_CHANNEL_CAPACITY_MODEL.md
AUDIO_SCALAR_REPLAY_ADVERSARY.md
AUDIO_PHYSICAL_CARRIER_CANDIDATES.md
AUDIO_RESTORATION_LADDER.md
AUDIO_CONTROL_AND_KILL_MATRIX.md
AUDIO_CLAIM_LAW.md
AUDIO_IMPLEMENTATION_REQUIREMENTS.md

audio_wave_reference.py
AUDIO_WAVE_REFERENCE_TESTS.json
AUDIO_WAVE_REFERENCE_RESULTS.json
AUDIO_WAVE_FIXTURE_MANIFEST.json

AUDIO_SUBSTRATE_DESIGN_REVIEW.md
AUDIO_SUBSTRATE_FINDINGS_NORMALIZED.json
AUDIO_SUBAGENT_REVIEW_REPORTS.md
AUDIO_LANE_STATE.md
```

Generated WAV fixtures should live under:

```text
fixtures/
```

Keep fixture count and total size reasonable. Prefer short deterministic float32 WAV files. Do not commit large exploratory renders.

## 6. Frozen Offline Numerical Envelope

Unless a documented numerical reason requires a change, begin with:

```text
sample_rate = 48000 Hz
duration = 2.0 s
primary carrier = 8000 Hz
baseband bandwidth <= 1000 Hz
absolute sample ceiling <= 0.95
WAV encoding = IEEE float32
channels = mono for scalar fixtures, stereo only for explicit two-axis tests
```

Every fixture must bind:

```text
sample rate
sample count
duration
channel count
dtype
peak amplitude
RMS amplitude
SHA-256
semantic role
generator parameters
```

No metadata field may secretly encode expected answers.

## 7. Required Wave Algebra

Implement mathematically explicit primitives.

### FM encoding

For bounded message `m(t)`:

```text
phi(t) = 2*pi*f_c*t + 2*pi*k_f*integral(m(t) dt)
s(t) = cos(phi(t))
```

### PM encoding

```text
s(t) = cos(2*pi*f_c*t + k_p*m(t))
```

### Analytic signal

Construct a complex analytic representation using a documented Hilbert-transform implementation or an equivalent FFT-domain method.

### Conjugate mixing

```text
z_rel(t) = z_state(t) * conjugate(z_query(t))
```

This performs phase subtraction. Recover instantaneous relative phase and frequency with explicit edge handling.

### Ordinary mixing

```text
z_sum(t) = z_a(t) * z_b(t)
```

This performs phase addition modulo the chosen representation.

### Multitone state

```text
x(t) = sum_k A_k*cos(2*pi*f_k*t + phi_k)
```

Support complex spectral coefficients, not only magnitudes.

### Delay and phase rotation

Implement exact sample delay and fractional-delay or phase-rotation variants. State which is used in each fixture.

### Filter-bank projection

Project a waveform onto a frozen set of complex frequency bins or matched filters.

### Correlation and matched filtering

Return normalized and unnormalized variants with explicit denominator behavior.

### Convolution

Implement time-domain or FFT convolution with a frozen boundary convention.

### Controlled nonlinear mixing

Include at least one polynomial or saturating nonlinearity that produces intermodulation products. Treat it as an ordinary nonlinear mechanism, not a catalytic effect.

## 8. Required Reference Tests

At minimum prove numerically:

```text
FM round-trip recovery
PM round-trip recovery
conjugate mixing recovers state-minus-query
ordinary mixing recovers phase addition
delay produces predicted phase rotation
filter-bank projection recovers complex coefficients
matched filter distinguishes target from energy-matched sham
convolution matches direct numerical reference
nonlinear mixer creates predeclared intermodulation frequencies
label-only renaming leaves waveform and result invariant
query scramble breaks the intended recovery
```

Every test must define:

```text
input identity
expected mathematical result
error metric
tolerance
edge region excluded, if any
pass/fail result
```

Do not reuse count-domain tolerances for dimensionless phase, gain, or normalized-vector metrics.

Suggested separate metrics:

```text
sample RMSE
normalized vector error
angular error
frequency error in Hz
phase error in radians
spectral leakage ratio
absolute null amplitude
```

## 9. Query Separation and Finite-Query Theorem

Carry forward the established theorem:

```text
For finite enumerable query family Q and source-known state R,
the source can precompute V(R) = [h(R,q) for q in Q].
Delayed query selection alone does not distinguish a state from an answer cache.
```

The audio lane must distinguish:

```text
closed finite query set
held-out discrete query
continuous or high-resolution query family
literal answer table
compressed answer generator
low-rank spectral basis
wave-state representation
ordinary DSP replay
bounded physical channel
```

Do not claim that a continuous query automatically defeats compact formulas.

For offline fixtures, an ordinary Python decoder that knows the waveform generation law is an admissible adversary.

## 10. Scalar and DSP Replay Adversaries

Build attacks that receive:

```text
every source-visible input
every source receipt
all committed WAV files
fixture manifest
public generation code
realized query at scoring time
```

Include:

```text
finite lookup table
compressed answer generator
linear filter model
nonlinear filter model
spectral-energy model
phase-label lookup
file-metadata lookup
time-index lookup
multitone coefficient replay
ordinary analytic-signal DSP pipeline
```

The offline algebra result is expected to be reproducible by ordinary DSP. That is not failure. It defines the correct offline claim ceiling.

A future physical carrier claim must be evaluated against a separate bounded physical-channel model.

## 11. Physical Carrier Candidate Comparison

Compare at least five candidates.

### Direct DAC-to-ADC loopback

Use as instrumentation calibration only. It has almost no persistent post-source state unless explicit filtering or buffering exists.

### Passive electrical resonator

Examples include RLC, active band-pass, all-pass, or coupled filters. Define energy storage, ring-down, query injection, and measurable state.

### Electromechanical resonator

Examples include a speaker diaphragm, piezo element, beam, plate, or spring-mass system. Define mode structure, damping, and readout.

### Sealed acoustic cavity or tube

Define geometry, resonant modes, source disconnect, microphone readout, and environmental controls.

### Feedback or delay loop

Define the physical or mixed-signal loop, state lifetime, stability boundary, query operator, and active energy contribution.

For each candidate document:

```text
physical state
state variables
source operation
source-disconnect mechanism
post-source lifetime
query operation
observable
measurement disturbance
nonlinearity
restoration operation
hardware requirements
ordinary explanation
capacity observables
noise and safety risks
```

Do not select a carrier whose state is merely a WAV file or audio interface buffer.

## 12. Recommended Physical Direction

The likely first physical prototype is an electrical or electromechanical audio-frequency resonator:

```text
WAV preparation
-> DAC
-> resonator/filter/delay carrier
-> source disconnect or gate closure
-> fresh query
-> ADC
-> frozen receiver analysis
```

Open-air room acoustics should not be the first prototype unless the review shows a compelling reason. Environmental drift and uncontrolled multipath make early causal claims harder.

This recommendation is provisional and must survive review.

## 13. Audio State Tomography Requirements

A future physical prototype should measure:

```text
complex transfer function
impulse response
modal amplitudes
modal phases
ring-down curve
intermodulation products
state evolution versus delay
query-dependent response
baseline and post-query state
```

Operational state does not equal total microstate capacity.

Required statement:

```text
observed distinguishable wave states != total physical state capacity
```

## 14. Restoration Ladder

Use the existing restoration vocabulary:

```text
R0 = file/sample or byte return
R1 = measured-output return
R2 = accepted observable-state equivalence
R3 = multi-instrument carrier return
R4 = closure up to predeclared invariant
```

Offline WAV inversion may establish only an algebraic inverse or R0-like digital return. It does not establish physical restoration.

For a future physical carrier, require:

```text
baseline state
prepared state
post-query state
active restoration
post-restoration state
time-matched natural relaxation
no-restoration control
wrong-inverse control
reordered-inverse control
carrier-off control
```

Do not freeze R2 thresholds in the offline lane without physical baseline distributions.

## 15. Mandatory Controls

Design at minimum:

```text
label invariance
energy-matched sham
spectrum-matched phase scramble
query-off
query scramble
carrier-off
frequency-shifted wrong query
phase-inverted query
order scramble
time reversal where applicable
file-metadata stripping
amplitude-only adversary
spectral-magnitude-only adversary
ordinary linear-filter replay
ordinary nonlinear-filter replay
```

For any future relational preparation add:

```text
relation mutation with fixed marginal spectrum
branch permutation
geometry null
fixed-energy semantic control
```

Do not introduce OrbitState into the first physical audio prototype.

## 16. Source and Query Custody for a Future Prototype

Freeze a prospective causal sequence:

```text
public preparation selected
source waveform generated and hashed
physical source drives carrier
source output is gated or disconnected
source process and hardware route close
fresh query is generated independently
query parameters commit before measurement
receiver applies query
ADC capture freezes
receiver features freeze
analysis begins
```

The offline lane should specify this law but not claim to implement hardware isolation.

## 17. Channel Capacity Model

A future physical capacity argument must account for:

```text
bandwidth
duration
SNR
response precision
number of persistent modes
mode Q factors
ring-down lifetime
channel side information
source and decoder computation
compression or formula-based answer generation
```

Raw sample count is not a lower bound on answer-equivalent code length.

Do not emit a positive capacity-separation claim in the offline package.

## 18. Independent Review

After the offline package is complete, dispatch exactly four independent read-only reviewers:

```text
wave-computing mechanism auditor
signal-processing and identifiability auditor
physical audio-carrier auditor
claim-boundary adjudicator
```

Do not expose reports to one another before completion.

Require attacks against:

```text
ordinary DSP replay
finite answer cache
compressed answer generator
spectral energy leakage
phase-label leakage
file metadata leakage
query preselection
fake persistence from file storage or interface buffering
ordinary filter-bank explanation
nonlinear intermodulation explanation
restoration overclaim
```

Archive complete reports and a normalized finding map.

A single unresolved material blocker prevents physical-prototype freeze.

## 19. Validation

Run at minimum:

```text
Python syntax
reference-engine self-test
all JSON parse
all WAV parse
fixture hash verification
FM recovery
PM recovery
conjugate and ordinary mixing tests
multitone projection tests
delay tests
matched-filter tests
convolution tests
nonlinear mixing tests
held-out-query tests
metadata-stripping attacks
scalar and DSP replay attacks
cross-document identity checks
git diff --check
governance critic
ci_local_gate.py --full
```

No audio hardware contact is authorized.

## 20. Git and Completion

Use one coherent architectural commit or a small number of large coherent commits. Avoid micro-commit pellets.

Push only:

```text
codex/audio-frequency-wave-substrate
```

Do not merge to `main`.

Final report must include:

```text
starting branch head
final branch head
base main commit
package path
fixture count and total bytes
fixture manifest SHA
reference results SHA
selected or surviving physical carrier candidates
rejected candidates and reasons
four reviewer IDs and verdicts
normalized findings SHA
design review SHA
remaining blockers
working-tree status
audio hardware contact count = 0
target contact count = 0
```

End with exactly one:

```text
AUDIO_FREQUENCY_WAVE_ARCHITECTURE_READY_FOR_INTEGRATION_REVIEW
AUDIO_FREQUENCY_WAVE_ARCHITECTURE_BLOCKED
AUDIO_FREQUENCY_WAVE_ARCHITECTURE_INCONCLUSIVE
```

## 21. Current Lane State

At bootstrap time:

```text
branch created = true
bootstrap committed = true
implementation started = false
audio hardware contact count = 0
target contact count = 0
```

The next agent should begin by creating a local worktree for this branch, reading this entire file, auditing inherited CAT_CAS claim laws, and then implementing the offline reference and design package.
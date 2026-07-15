# Audio Control And Kill Matrix

Status: `FROZEN_OFFLINE_CONTROLS_PHYSICAL_CONTROLS_DESIGNED`

| Control | Manipulation | Observable | Required result | Kills |
| --- | --- | --- | --- | --- |
| Label invariance | Rename external aliases only | Samples and complex projection | Exact null | label dictionary leakage |
| Energy-matched sham | Exact target time reversal | Energy, magnitude spectrum, matched score | Energy equal; target gap >= 0.80 | energy-only explanation |
| Spectrum-matched phase scramble | Preserve `|X(f)|`, change phase/time order | Complex projection/matched score | Intended response breaks | magnitude-only explanation |
| Query-off | Apply no receiver query | Query-dependent response | Declared null | autonomous readout artifact |
| Query scramble | Shift query by 37 Hz | Relative phase | Phase RMSE >= 0.50 rad | query-independent replay |
| Frequency-shifted wrong query | Use nonmatching basis | Filter/matched response | Below future frozen floor | broad energy coupling |
| Phase-inverted query | Rotate query by pi | Complex response | Predicted sign/phase reversal | magnitude-only readout |
| Order scramble | Permute operator sequence | Final state | Follow declared noncommuting law or null | schedule lookup |
| Time reversal | Reverse target samples | Matched score | Energy invariant, response noninvariant | energy/spectrum leakage |
| Metadata stripping | Remove all nonessential RIFF chunks | Samples and projection | Exact invariant | file metadata leakage |
| Amplitude-only adversary | Discard phase | Scored response | Cannot reproduce matched response | scalar energy replay |
| Spectral-magnitude adversary | Retain only FFT magnitudes | Scored response | Cannot reproduce phase/time query | magnitude replay |
| Linear-filter replay | Public FFT convolution | Full output | Must reproduce offline result | exotic linear claim |
| Nonlinear-filter replay | Public polynomial model | Intermodulation output | Must reproduce offline result | exotic nonlinear claim |
| Finite answer cache | Precompute every closed query | Realized query answer | Must reproduce closed set | delayed-query overclaim |
| Compressed generator | Store amplitude/phase formula | Held-out query answers | Must reproduce held-out set | continuous-query overclaim |
| File persistence | Reopen WAV after generator ends | Projection | Must reproduce output | fake physical persistence |
| Manifest-parameter replay | Read public complex generator parameters | Projection | Must reproduce coefficients | non-sample side-channel overclaim |
| Interface-buffer drain | Drain and verify all queues | Post-source capture | Future strict null | queued-frame persistence |
| Carrier-off | Remove physical energy store | Receiver output | Future strict null | instrument/software artifact |
| Natural relaxation | Wait without inverse | State distance | Distinct from active inverse | restoration overclaim |
| Wrong/reordered inverse | Apply incorrect restoration | State distance | Must fail R2 | generic waiting or heating |

## Offline Passing Semantics

Some adversaries pass by succeeding. Finite cache, compressed generator, ordinary DSP,
linear filter, nonlinear filter, and file persistence are expected to reproduce the
offline result. A test failure would weaken the claim boundary.

Other controls pass by nulling leakage or breaking the intended result. Label renaming
and metadata stripping must be invariant; the wrong query must break recovery; the
energy-matched sham must not match the target.

## Physical Stop Matrix

A future run must stop and classify no physical claim if any of these occur:

```text
source disconnect not mechanically verified
interface buffers not drained
carrier-off response above frozen floor
query provenance available before source closure
measurement saturates or changes the carrier outside the declared disturbance bound
thermal, voltage, current, or acoustic safety limit exceeded
R2 threshold derived after candidate observation
wrong inverse passes as well as correct inverse
ordinary bounded replay fits within tolerance
```

No control in this document authorizes hardware operation.

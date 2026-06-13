# Undervolt Pathway 6: Frequency Detuning Without VID Reduction

## Status

`FREQUENCY_DETUNING_VERIFIED_READY`

Candidate quality: `VERIFIED_READY`

Risk tags: `NO_VID_REDUCTION`

## Scope

This does not lower voltage. It is included because the mission explicitly allowed a clock/frequency-detuning pathway that reaches analog instability without VID reduction.

## Evidence

Local roadmap evidence:

- DID `0`: 1600 MHz confirmed.
- DID `1`: 800 MHz confirmed.
- DID `2`: 400 MHz confirmed.
- DID `3`: 200 MHz confirmed.
- DID `4`: 100 MHz confirmed, despite BKDG reserved notation.
- Temperatures stayed around 43-46 C.
- Core 3 and Core 4 were used as programmable oscillator cavities.
- Passive TSC coupling was dominated by a 2.67 MHz VRM artifact and did not show reproducible oscillator phase lock at nominal voltage.

Local scripts:

- [50_2_phase_locked_network/src/kuramoto_test.py](50_2_phase_locked_network/src/kuramoto_test.py) writes P4 definitions per core, cycles P0 to P4, and sweeps DID.
- [50_1_subthreshold_msr/src/freq_sweep.py](50_1_subthreshold_msr/src/freq_sweep.py) exercises frequency control.
- [50_2_phase_locked_network/src/oscillator.c](50_2_phase_locked_network/src/oscillator.c) and [50_2_phase_locked_network/src/tsc_sampler.c](50_2_phase_locked_network/src/tsc_sampler.c) support oscillator/readout experiments.

## Exact Path

Use DID detuning at nominal voltage to search for analog instability signatures:

- Core 4 fixed at DID `3` / about 200 MHz.
- Core 3 sweeps DID `0` through `4`.
- Core 2 remains readout.
- Core 5 remains phase reference.
- Record TSC spectra, temperature, and P-state readback for every run.

## Limitation

This route does not satisfy electrical undervolting. It is useful only if the immediate research goal is phase/analog instability rather than lower Vcore.

## Decision

This is a safe fallback candidate for the CAT_CAS analog objective if all VID routes remain clamped. It should not be represented as a true voltage-control success.

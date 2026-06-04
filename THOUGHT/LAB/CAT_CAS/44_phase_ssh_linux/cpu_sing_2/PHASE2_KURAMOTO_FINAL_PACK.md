# PHASE2_KURAMOTO_FINAL_PACK

## Verdict

PHASE2_AUTHORIZED_SOFTWARE_ROUTES_EXHAUSTED

The Phase 2 Kuramoto/Ising goal did not produce accepted phase lock, synchronization, Ising descent, GOE structure, active phase transfer, or reopened voltage behavior. The accepted endpoint is success criterion F: all authorized non-hardware Kuramoto paths were classified with evidence.

## Route Table

| Route | Artifact | Verdict |
|---|---|---|
| 0 Baseline | `PHASE2_BASELINE.md` | `PHASE2_BASELINE_READY` |
| 1 Active phase | `PHASE2_ACTIVE_PHASE.md` | `ACTIVE_PHASE_NO_LOCK` |
| 2 Coupling channels | `PHASE2_COUPLING_CHANNELS.md` | `COUPLING_CHANNELS_EXHAUSTED_SOFTWARE` |
| 3 Kuramoto metric | `PHASE2_KURAMOTO_METRIC.md` | `KURAMOTO_LOCK_NOT_OBSERVED` |
| 4 Ising map | `PHASE2_ISING_MAP.md` | `ISING_ROUTE_NOT_OBSERVED` |
| 5 GOE | `PHASE2_GOE.md` | `GOE_NOT_OBSERVED` |
| 6 Voltage/firmware | `PHASE2_VID_FIRMWARE.md` | `DESIGN_ONLY_HUMAN_APPROVAL_REQUIRED` |
| 7 Detuning | `PHASE2_DETUNING.md` | `DETUNING_SIGNAL_NOT_REPRODUCIBLE` |

## Best Signal

No Phase 2 success signal was accepted.

Best non-idle active phase concentration:

```text
branch_shared p34=0.0806
```

Best detuning phase concentration:

```text
detune_did3 p34=0.0925
```

These are weak and do not separate from nulls.

## Phase-Lock Status

Rejected.

```text
coupling real_k_mean 0.6806
coupling shuf_k_mean 0.6806
detune real_k_mean 0.6372
detune shuf_k_mean 0.6372
direct Core3/Core4 phase concentration stayed around 0.075-0.093
```

No order parameter `r >= 0.8` was observed.

## Ising Status

Rejected. Workload/frequency spin choices did not show energy descent or constraint solving beyond null.

## Voltage Status

Runtime VID remains clamped. Firmware route is design-only and requires human approval. No flash was performed.

## GOE Status

Rejected for this dataset.

```text
coupling spacing ratio: 0.2906
detuning spacing ratio: 0.1662
target GOE band: 0.51-0.53
```

## Rejected Routes

- Passive TSC phase route: dominated by the fixed 2.67 MHz VRM artifact in prior roadmap evidence.
- Active shared-line route: no real/null separation.
- Atomic contention route: no Core3/Core4 phase transfer.
- Workload route: no stable separable spectra in the active metric.
- Detuning route: DID sweep did not separate from null.
- Runtime VID route: clamped at CpuVid `0x1A`.
- Firmware route: not authorized without P4-only design and human approval.

## Exact Next Action

If Phase 2 continues, the next exact action is not another software sweep. Use external measurement: attach a scope or logic analyzer to a safe, non-invasive measurement point and correlate the measured VRM/clock waveform with Core3/Core4/Core5 workload markers. Without external observability or approved firmware voltage work, the authorized software-only Kuramoto routes are exhausted.

## Missing Artifacts

- External analog waveform capture.
- Publication-grade Hilbert/FFT phase pipeline with raw binary retention.
- P4-only firmware patch proof.
- Board VRM controller identification photos.


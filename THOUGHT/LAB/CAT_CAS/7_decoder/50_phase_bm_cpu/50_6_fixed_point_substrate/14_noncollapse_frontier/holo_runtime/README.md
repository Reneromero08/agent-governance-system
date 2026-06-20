# Phase 6 Combined PDN Runner

`combined_pdn_runner.c` is the schedule-driven executor boundary for the frozen combined observability campaign.

Current status: `HARDWARE_BACKEND_COMPLETE_PREFLIGHT_READY`.

The runner enforces the non-negotiable validation and hardware invariants:

- consumes compiled `session.json`, `windows.jsonl`, and `session_manifest.json` directly;
- verifies the session manifest before reading scientific windows;
- refuses an existing output directory;
- requires contiguous `window_index` order;
- rejects sender-off windows with any active drive;
- rejects unsupported measurement modes;
- supports hardware-free `--validate-only` and explicit `--hardware` modes;
- applies schedule-provided physical tone, codeword source, actual mode, phase, and amplitude without reordering;
- creates and joins a sender per driven window and proves no sender is alive for sender-off capture;
- uses Slot2-derived affinity, absolute TSC, thermal veto, telemetry, and ring-capture primitives;
- snapshots, pins, restores, and verifies all cpufreq policies plus boost on every exit path;
- writes immutable evidence and a self-excluding SHA-256 run manifest.

Hardware execution remains gated by the frozen plan binding, simulation/sanitizer tests, a non-scientific engineering smoke, and `catcas_preflight.py` reporting `acquisition_ready=true`. Passing those gates does not itself start the scientific campaign.

Build and test:

```bash
make test
```

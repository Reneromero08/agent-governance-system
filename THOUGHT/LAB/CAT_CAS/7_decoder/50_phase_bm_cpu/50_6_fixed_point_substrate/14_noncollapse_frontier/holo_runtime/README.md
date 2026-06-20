# Phase 6 Combined PDN Runner

`combined_pdn_runner.c` is the schedule-driven executor boundary for the frozen combined observability campaign.

Current status: `VALIDATION_ONLY_SCAFFOLD`.

The runner already enforces the non-negotiable pre-hardware invariants:

- consumes compiled `session.json`, `windows.jsonl`, and `session_manifest.json` directly;
- verifies the session manifest before reading scientific windows;
- refuses an existing output directory;
- requires contiguous `window_index` order;
- rejects sender-off windows with any active drive;
- rejects unsupported measurement modes;
- writes the contracted run-output file set in `--validate-only` mode;
- refuses to touch hardware unless the CAT_CAS local hardware backend is completed.

The remaining hardware backend must be implemented on the lab machine by reusing the proven Slot2 affinity, absolute-TSC timing, thermal-veto, cpufreq restoration, telemetry, and immutable raw-writer primitives. Do not fake acquisition in repository-side tests.

Build and test:

```bash
make test
```

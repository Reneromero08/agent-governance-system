# Combined Campaign Hardware Executor Interface

**Status:** `BINDING_EXECUTOR_INTERFACE`  
**Hardware execution:** local-agent boundary  
**Restoration:** forbidden

The existing Slot2 runner cannot execute this campaign unchanged because it generates ascending tone order internally and has no sender-off raw-ring stage. The combined runner must consume the compiled session schedule directly.

## Required inputs

```text
session.json
windows.jsonl
session_manifest.json
```

The runner must verify the session and parent campaign manifests before touching hardware, refuse an existing output directory, and execute rows strictly by `window_index`.

## Required semantics

For each row:

- `physical_tone_index` selects the physical tone;
- `codeword_source_index` selects the codeword sign;
- `actual_mode`, `theta_idx`, and `amplitude_level` select executed control;
- declared mode/order remain metadata only;
- `drive_on=false` starts no sender thread;
- `sender_off_required=true` requires all sender threads stopped before capture;
- `lockin_and_raw_ring` records raw ring samples plus complex lock-in output;
- `raw_ring_sender_off` records raw ring samples and telemetry without inventing a driven response.

The executor must reuse the proven Slot2 affinity, absolute-TSC timing, deadline-bounded capture, thermal veto, telemetry, cpufreq restoration, and immutable raw-writer primitives.

Abort without automatic retry on manifest mismatch, output collision, affinity failure, thermal veto, incomplete sender-thread creation, timing overflow, or restoration failure.

The executor may not generate, reorder, skip, or reinterpret scientific windows internally.

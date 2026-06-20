# Combined Executor Evidence Contract

Every session must produce a new immutable directory containing:

```text
run.json
session.json
windows.jsonl
window_results.csv
raw_samples.bin
telemetry.csv
stdout.log
stderr.log
run_manifest.json
```

`run.json` binds campaign-plan SHA-256, campaign source commit, session-manifest SHA-256, executor commit, host, route, cores, frequency policy, TSC calibration, safety thresholds, timestamps, and exit status.

`window_results.csv` retains executed and declared mode/order separately, drive state, sender-off requirement, tone index, codeword source index, timing, sample count, telemetry, I/Q/floor where applicable, and raw-ring summaries.

Forbidden:

- overwriting evidence;
- changing schedules after the first sample;
- outcome-dependent skipping or retries;
- hidden drive during sender-off windows;
- dropping seed 4;
- restoration or inverse-drive stages;
- claiming persistence from recovery while the sender remains active.

# Independent-Window V3 Live Execution Completion Audit

## Prior frozen package

Commit `f13599f80f55703d7eb2cec9c4be2c8f1b0bdf5a` froze a valid offline
scientific schedule, adjudication scaffold, manifest discipline, and
authorization gate for Independent-Window Public Transducer V3.

That package was not a finished live executor. Its `--execute-authorized` mode
validated the three authorization bindings and then intentionally reported
`LIVE_TRANSPORT_NOT_INVOKED_BY_OFFLINE_FREEZE_PACKAGE` with a failing exit
status. It did not perform SSH/SCP transfer, target-side custody, PMU preflight,
replicate capture, copy-back verification, or success/failure remote-root
handling.

This is an execution-completeness finding, not a scientific failure. The frozen
science, schedule geometry, allowed result classes, forbidden result classes,
thresholds, coordinate laws, q ladder, source-work invariants, and retained V2
classification boundaries remain preserved.

## Replacement package

This package completes the non-driving scaffold into a live-capable package
without contacting the lab device during implementation.

The completed live path is:

1. bound local source verification;
2. SSH/SCP transfer of only bound package files;
3. target-side custody checks;
4. strict C compilation;
5. runtime self-test;
6. exact-event PMU preflight;
7. two fresh independent-window replicates;
8. fresh V3 feature extraction and adjudication from raw component values;
9. verified copy-back by path, size, and SHA-256;
10. exact V3 remote-root cleanup on success or retention on target failure.

Offline modes retain zero transport authority. Live contact still requires a
future explicit authorization with the final commit binding, implementation
manifest binding, and live authority environment variable.

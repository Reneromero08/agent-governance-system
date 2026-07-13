# OrbitState Independent V2 Live Custody and Protocol Repair Audit

This audit records implementation-custody defects found in the frozen
OrbitState Independent V2 package and the repair boundary applied before any
live authorization. It does not redesign the OrbitState hypothesis, alter the
frozen public schedule, alter the private source map, change result classes, or
promote the Small Wall law.

## Frozen Science Preserved

- Run ID: `orbitstate_independent_v2_0`
- N: `256`
- d: `23`
- fold(d): `233`
- M: `2048`
- Quantization scale: `1536`
- Public q0 absolute bound: `152`
- Private odd-signal floor: `456`
- Relational tolerance: `0.25`
- Replicates: two fresh replicates
- Conditions: nine frozen conditions
- Public decoder phases: four
- Mapping-leg records: `144`
- Independent component windows: `288`

The existing OrbitState formula and post-fork source-child boundary are real and
are retained.

## Source-Grounded Defects

1. Remote root creation used `mkdir -p` and therefore did not assert an absent
   run root before live transfer.
2. The controller transferred the whole package directory recursively instead of
   transferring only hash-bound source files.
3. Copy-back was accepted without a size-and-SHA manifest for every returned
   evidence file.
4. The controller removed the remote root without first requiring successful
   target status, a verified success execution manifest, and verified evidence
   hashes.
5. Target failure could exit before failure evidence was copied back.
6. The target accepted an existing output root and unlinked preexisting files.
7. Platform, privilege, process, temperature, CPU-policy, and exact PMU custody
   were absent from live execution evidence.
8. The target Python process parsed the private source map before receiver
   features were frozen.
9. PMU counters were opened system-wide with `pid=-1`, `cpu=5`, and
   `exclude_kernel=0`.
10. Raw custody fields were hardcoded instead of measured.
11. A full receiver baseline was claimed without full-bank receiver work.
12. Physical A and B began with different bytes.
13. Source work did not use the proven affine line permutation.
14. The second independent component did not begin from a fully reconstructed
   two-bank receiver state.
15. The valid OrbitState formula and real post-fork source child needed to be
   retained, not replaced.

## Repairs Applied

- Controller pretransport now requires clean `main`, exact commit binding,
  exact `origin/main`, exact manifest file SHA authority, valid manifest
  canonical digest, live authority, passing validate-only, frozen hashes, exact
  source-bundle reconstruction, and absent local run root before any SSH or SCP.
- Remote creation now requires the exact run root to be absent and creates only
  the exact source root with mode `0700`.
- Source transfer is an explicit manifest-bound file map plus separately bound
  implementation manifest transfer; recursive package transfer is rejected by
  controller self-tests.
- Target invocation is wrapped by a hard remote timeout:
  `timeout --signal=TERM --kill-after=5s 900s`.
- Copy-back now requires `COPYBACK_MANIFEST.json` and verifies safe relative
  paths, exact sizes, exact SHA-256 hashes, and complete coverage.
- Success cleanup is allowed only after target completion, return code zero,
  verified success execution manifest, matching implementation-manifest binding,
  allowed result class, and verified evidence hashes. Cleanup removes only the
  exact OrbitState V2 remote root and verifies absence.
- Target failure attempts to seal target failure evidence, failure manifest,
  final failure result, custody log, and copy-back manifest; the controller
  retains the remote root on target failure, controller failure, copy-back
  failure, and cleanup failure.
- Target live execution now rejects any existing output root before creating a
  new mode-`0700` root and never unlinks preexisting output files.
- Live custody inherits the V3 platform checks: root identity, Linux
  capabilities, AMD family and PMU format checks, core online checks, strict
  k10temp resolution, temperature checkpoints, process scans, CPU-policy
  snapshots, and zero mutation counts for frequency, sysctl, voltage, MSR,
  physical-address, and cache-set access.
- Public schedule validation and private map validation are split. The private
  map validator runs in a separate subprocess and returns only pass status, file
  SHA, canonical SHA, record count, and schema identifier.
- Receiver feature extraction runs in a separate receiver-only subprocess that
  receives only public schedule, raw receiver capture, receiver sentinels, and
  public stage receipts. It writes frozen receiver features and SHA before
  unblinding.
- Adjudication runs in a separate subprocess only after the feature digest is
  frozen. The chronology receipt records feature process start, completion,
  hash freeze, private-map opening for adjudication, and adjudication start.
- The C runtime now opens a process-scoped PMU group with `pid=0`, `cpu=-1`,
  `exclude_kernel=1`, `exclude_hv=1`, disabled leader, group reads, total
  enabled/running times, and event IDs. It verifies group count, read size,
  event IDs, unmultiplexed time, and receiver CPU stability.
- Runtime `--pmu-preflight` uses the same PMU group and owned synthetic memory,
  emits structured custody, and does not emit scientific classification.
- Physical A and B now start with identical deterministic bytes. A separate
  dummy bank is retained for `source_off`.
- Receiver baseline, sentinels, rebaseline, source work, measurement,
  restoration, and post-sentinel stages use the affine line permutation
  `line(i) = (257*i + 43) mod 4096`.
- Mapping-pair allocation now allocates physical A/B once for each replicate,
  opaque group, and public phase crossover pair, executes both mapping legs in
  frozen order, and frees only after both legs complete.
- Each positive and negative component begins from full two-bank receiver
  reconstruction and records seven stage receipts, preserving the required
  `2016` stage receipts across `288` windows.

## Live Boundary

This repair is entirely offline. It authorizes no SSH, SCP, ping, target
inspection, PMU hardware execution, or lab-device contact. The future live
command remains gated by exact commit, exact manifest file SHA, and exact live
authority.

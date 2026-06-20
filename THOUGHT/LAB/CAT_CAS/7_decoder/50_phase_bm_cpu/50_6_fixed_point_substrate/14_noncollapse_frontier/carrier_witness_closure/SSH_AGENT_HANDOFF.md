# SSH Agent Handoff — Phase 6B.5 Carrier-Witness Closure

You are operating the Phenom/K10 engineering machine through SSH.

Your task is to implement and execute the **strict PDN carrier-witness closure** on the existing branch:

```text
phase6b/carrier-witness-closure
```

Do not merge the branch. Do not push to `main`. Do not begin observability/operator acquisition, physical restoration, target coupling, or wall adjudication.

---

## 1. Read the binding contract

Before modifying source, read:

```text
14_noncollapse_frontier/carrier_witness_closure/README.md
14_noncollapse_frontier/carrier_witness_closure/CARRIER_WITNESS_CONTRACT.md
14_noncollapse_frontier/carrier_witness_closure/RAW_ARTIFACT_SCHEMA.md
14_noncollapse_frontier/COURSE_CORRECTION.md
```

Also inspect the historical pipeline:

```text
10_cross_core_wormhole/slot2_pdn/slot2_pdn_lockin.c
10_cross_core_wormhole/slot2_pdn/slot2_pdn_run.h
10_cross_core_wormhole/slot2_pdn/slot2_pdn_analyze.py
10_cross_core_wormhole/slot2_pdn/aggregate.py
12_chiral_lane_frontier/pdn_slot2_t300/PHASE6_SLOT2_PDN_T300_REPORT.md
```

The existing score is channel evidence. The missing object is reconstructable raw evidence.

---

## 2. Establish repository state

```bash
git fetch origin --prune
git switch phase6b/carrier-witness-closure
git pull --ff-only origin phase6b/carrier-witness-closure
git status --short
git rev-parse HEAD
git rev-parse origin/main
git merge-base origin/main HEAD
git log --oneline --decorate --graph -20
```

Requirements:

- clean worktree before execution;
- `origin/main` remains an ancestor;
- no rebase, force-push, or history rewrite;
- preserve any newer branch work you did not create.

---

## 3. Verify repository-side tooling first

From:

```text
14_noncollapse_frontier/carrier_witness_closure/
```

run:

```bash
make clean test
```

This must compile/test the immutable raw writer and run the Python synthetic lock-in/manifest tests.

Fix any genuine source defect before touching the hardware. Add regression coverage for every fix.

---

## 4. Audit the old T300 host tree without modifying it

Run from the repository root:

```bash
python3 \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/carrier_witness_closure/audit_existing_t300.py \
  --repo-root "$PWD" \
  --host-root /root/slot2_pdn \
  --output /root/slot2_pdn/t300_existing_evidence_audit.json
```

Preserve the audit and report:

- historical matrix CSV count;
- historical logs and controls;
- any raw arrays or alternate acquisition trees;
- hashes of all discovered files;
- whether any complete raw bundle already exists;
- thermal sensor inventory.

Do not rename, rewrite, normalize, or delete old evidence.

If the old raw `(t_tsc, ro_period)` arrays exist elsewhere, validate them before deciding to reacquire. The old matrix CSVs alone are insufficient because they begin after lock-in reduction.

---

## 5. Restore thermal observability before acquisition

The prior T300 report used `k10temp`, while the later repair verification found no readable hwmon temperature input.

Investigate without bypassing safety:

```bash
uname -a
lscpu
lsmod | grep -E 'k10temp|hwmon' || true
sudo modprobe k10temp
find /sys/class/hwmon -maxdepth 2 -type f \( -name name -o -name 'temp*_input' \) -print -exec cat {} \;
dmesg | tail -200
```

A full physical run is prohibited unless a valid thermal source is readable and frozen in the campaign metadata.

Do not:

- substitute `-999`;
- fabricate a temperature;
- disable the temperature gate;
- increase the 68 C veto merely to obtain data;
- use a replacement sensor without documenting and predeclaring it.

When thermal observability cannot be restored, stop after software integration and return `CARRIER_WITNESS_PENDING_THERMAL_INSTRUMENTATION`.

---

## 6. Instrument the receiver below the lock-in boundary

Use the prepared module:

```text
carrier_witness_raw.h
carrier_witness_raw.c
```

Modify the existing Slot 2 acquisition stack rather than writing an unrelated second experiment.

### Required CLI/config additions

Add explicit options such as:

```text
--witness-dir <run directory>
--run-id <ID>
--condition matrix|silent|scramble
```

A witness run must refuse to overwrite an existing raw bundle.

### Required receiver integration

In `run_receiver()`:

1. Open `CarrierWitnessRawWriter` when `--witness-dir` is present.
2. Serialize `schedule.json` before the first captured slot:
   - exact tones;
   - exact codebook;
   - exact symbol sequence;
   - exact permutations;
   - seed and phase levels.
3. Around every `capture_slot()` call record:
   - thermal value before/after;
   - current-frequency proxy before/after;
   - COFVID/P-state proxy before/after.
4. After `score_slot()` fill `CarrierWitnessWindow` and call:

```c
carrier_witness_raw_append(...)
```

5. Preserve the legacy summary CSV unchanged in meaning.
6. Close the raw writer before reporting success.
7. Any raw write failure invalidates the run and returns nonzero.

### Exact drive metadata

Refactor the effective drive-sign calculation into a shared deterministic function used by sender and receiver.

This must include:

- canonical matrix sign;
- scramble-drive remapping;
- silent control sign `0`;
- actual phase fraction used by the sender.

Do not let the receiver infer a different schedule than the sender executed.

### Required timing metadata

Persist:

```text
shared t0 TSC
slot start TSC
capture deadline TSC
first/last sample TSC
measured TSC rate
slot_s
gap_s
read_hz
```

The parent/wrapper must also preserve sender, receiver, and orchestrator exit codes plus P-state restoration evidence.

---

## 7. Add run finalization tooling

Create a deterministic wrapper/finalizer that produces:

```text
run.json
schedule.json
windows.csv
raw_samples.bin
summary.csv
analysis.json
stdout.log
stderr.log
run_manifest.json
```

`run_manifest.json` hashes every file except itself.

Freeze and record:

- source commit;
- source-file hashes;
- compiler identity and exact flags;
- binary SHA-256;
- host/kernel/CPU;
- isolcpus and affinity;
- TSC flags/rate;
- thermal source;
- route, seed, control, and run ID;
- P-state target and restoration.

Generated summaries must be reproducible from raw bytes. Never copy a historical summary into a new run directory.

---

## 8. Raw smoke gate

Before a full campaign, run one short route `4:5` smoke acquisition with a new campaign/run ID and a small trial count, for example:

```text
route = v4s5
condition = matrix
seed = 777
trials = 4
```

The exact command depends on your integrated CLI, but it must produce the complete raw bundle.

Then create a one-run smoke campaign JSON and run:

```bash
python3 carrier_witness_validate.py \
  /path/to/smoke_campaign \
  --analyzer ../../../10_cross_core_wormhole/slot2_pdn/slot2_pdn_analyze.py \
  --output /path/to/smoke_campaign/aggregate/closure_report.json
```

A one-run smoke campaign is expected to return `PARTIAL`, not closure. It must nevertheless show:

```text
run valid = true
raw reconstruction errors within tolerance
retained summary equals reconstructed summary
retained analysis equals reconstructed analysis
manifest valid
```

Do not proceed when the smoke bundle cannot reconstruct itself.

---

## 9. Freeze the full campaign

Copy:

```text
campaign.template.json
```

to the durable campaign root and replace every placeholder before acquisition.

Freeze:

```text
campaign ID
source commit
binary hash
compiler flags
routes
seeds
controls
trials
slot/read timing
TSC rate
P-state target
thermal source/veto
storage root
```

The required matrix is:

```text
v4s5 matrix seeds 0-5
v4s5 silent control seed 900
v4s5 scramble control seed 901
v2s3 matrix seeds 0-5
```

A source, binary, or frozen configuration change requires a new campaign ID.

---

## 10. Full acquisition

Run in foreground on the quiet K10 host with adequate disk space.

Requirements:

- primary route `4:5` complete first;
- preserve every raw file immediately;
- verify each run manifest before starting the next run;
- cool down according to frozen policy;
- stop on temperature, affinity, P-state, TSC, disk, raw-writer, or process failure;
- never silently rerun into the same run directory;
- failed attempts remain preserved and are marked invalid.

Route `2:3` is the topology comparator. It does not have to meet the route `4:5` scored gate for `CLOSED_ROUTE_4_5`, but all reacquired comparator bundles must be structurally valid.

---

## 11. Validate the campaign

Run:

```bash
python3 \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/carrier_witness_closure/carrier_witness_validate.py \
  /path/to/frozen_campaign \
  --analyzer THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/10_cross_core_wormhole/slot2_pdn/slot2_pdn_analyze.py \
  --output /path/to/frozen_campaign/aggregate/closure_report.json
```

Allowed final states:

```text
CLOSED_ROUTE_4_5
CLOSED_MULTI_ROUTE
PARTIAL
PENDING
INVALID
```

Do not edit the closure report by hand.

---

## 12. Repository integration

Large raw files normally remain on the evidence host or approved durable storage.

Commit only compact provenance unless existing repository policy explicitly permits the raw size:

```text
campaign.json
source_manifest.json
run manifests
aggregate.json
closure_report.json
campaign_manifest.json
execution report
```

Every external raw path must include:

```text
absolute/durable storage identifier
size
SHA-256
media/host identity
run ID
```

Update Phase 6 roadmaps only with execution facts.

Do not mark observability acquisition authorized.

---

## 13. Required final proof packet

Return:

1. initial/final branch SHA and current `main` merge base;
2. tooling tests and exact commands;
3. old T300 evidence audit and gap list;
4. thermal diagnosis and sensor path;
5. source integration diff and build commands;
6. smoke-run bundle hashes and reconstruction errors;
7. frozen campaign matrix;
8. every run ID, route, seed, condition, status, size, and hash;
9. raw sample/window counts;
10. per-run scientific metrics and structural validity separately;
11. control behavior;
12. route `4:5` six-seed result;
13. comparator route result;
14. campaign-manifest SHA-256;
15. closure status;
16. exact claim ceiling;
17. remaining physical gates;
18. commits pushed to `phase6b/carrier-witness-closure`;
19. clean worktree proof.

The successful intended verdict is:

```text
CARRIER_WITNESS_CLOSED_ROUTE_4_5
```

when and only when the raw reconstruction and route-scoped gates pass.

Otherwise report the exact honest state without weakening the contract.

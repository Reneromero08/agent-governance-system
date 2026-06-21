# Phase 6 Frozen Acquisition Runbook

## Authorized object

This runbook executes the already target-qualified Phase 6 bundle. It does not regenerate the bundle from current `main`.

```text
qualified executor commit = 81ea84f341b29c41b93667d0e0fb98e0975bcbcf
campaign plan sha256 = eb5a46d0a37d66910649467cf0d4e3cf947dee11fab94a36e9bdfed388455e53
final source bundle sha256 = 5c6588a51ce6b806e1b7b269bafd1981256795653415e012592ad3b6313fdaca
strict runner sha256 = 0fc846a52d34b2395e254cc5a2db0bb715d1cd1ede77f8a5f6a2e940dab63037
authorization sha256 = e39fb0c6ebfb106c33a0b90b8d52d193a32833103388ebc4c6bd0cad451a0d73
authorized target output root = /root/catcas_phase6_acquisition_81ea84f3_v1
```

The authorization covers only the frozen 12-session combined-observability acquisition.

```text
acquisition_authorized = true
restoration_authorized = false
target_coupling_authorized = false
orientation_recovery_authorized = false
small_wall_authorized = false
physical_carrier_restoration_authorized = false
```

## Operational size

The frozen campaign contains:

```text
sessions = 12
windows = 99,456
driven windows = 77,184
sender-off windows = 22,272
window duration = 0.5 seconds
estimated capture time = 13.8133 hours
estimated raw sample bytes = 3,182,592,000
estimated raw sample size = 2.964 GiB
```

Require at least 20 GiB free on the target before starting. Use a stable power source. Run inside a persistent terminal such as `tmux` so an SSH disconnect does not terminate the campaign.

## Local inputs

Use the existing audited bundle:

```text
D:\CCC 2.0\AI\agent-governance-system\LAW\CONTRACTS\_runs\phase6_pr10_final_bundle_81ea84f3_v2
```

Use the authorization committed beside this runbook:

```text
authorizations/PHASE6_ACQUISITION_AUTHORIZATION_81ea84f3_v1.json
```

Do not rebuild the bundle from current `main`. The authorization binds the exact qualified `source_bundle.json` above.

## Target paths

```bash
SEALED_BUNDLE=/root/catcas_phase6_final_bundle_81ea84f3_v1
RUNTIME_BUNDLE=/root/catcas_phase6_runtime_bundle_81ea84f3_v1
AUTH=/root/PHASE6_ACQUISITION_AUTHORIZATION_81ea84f3_v1.json
ACQ_ROOT=/root/catcas_phase6_acquisition_81ea84f3_v1
PREFLIGHT=/root/catcas_phase6_acquisition_preflight_81ea84f3_v1.json
RUN_LOG=/root/catcas_phase6_acquisition_81ea84f3_v1.log
```

All six paths must be absent before transfer or execution. Stop rather than overwrite an existing object.

## Transfer and immutable verification

Transfer the complete final engineering bundle to `$SEALED_BUNDLE` and the authorization JSON to `$AUTH`.

On CAT_CAS:

```bash
set -euo pipefail

test ! -e "$RUNTIME_BUNDLE"
test ! -e "$ACQ_ROOT"
test ! -e "$PREFLIGHT"
test ! -e "$RUN_LOG"

test "$(sha256sum "$SEALED_BUNDLE/source_bundle.json" | awk '{print $1}')" = \
  "5c6588a51ce6b806e1b7b269bafd1981256795653415e012592ad3b6313fdaca"
test "$(sha256sum "$SEALED_BUNDLE/plan/campaign_plan.json" | awk '{print $1}')" = \
  "eb5a46d0a37d66910649467cf0d4e3cf947dee11fab94a36e9bdfed388455e53"
test "$(sha256sum "$SEALED_BUNDLE/combined_pdn_runner" | awk '{print $1}')" = \
  "0fc846a52d34b2395e254cc5a2db0bb715d1cd1ede77f8a5f6a2e940dab63037"
test "$(sha256sum "$AUTH" | awk '{print $1}')" = \
  "e39fb0c6ebfb106c33a0b90b8d52d193a32833103388ebc4c6bd0cad451a0d73"
```

The audited final bundle contains a post-sealing `engineering_preflight.json`. Create a runtime clone and remove only that post-sealing report. Never modify the sealed transfer.

```bash
cp -a "$SEALED_BUNDLE" "$RUNTIME_BUNDLE"
rm -f "$RUNTIME_BUNDLE/engineering_preflight.json"
chmod 0755 "$RUNTIME_BUNDLE/combined_pdn_runner"

test "$(sha256sum "$RUNTIME_BUNDLE/combined_pdn_runner" | awk '{print $1}')" = \
  "0fc846a52d34b2395e254cc5a2db0bb715d1cd1ede77f8a5f6a2e940dab63037"
```

The old qualified bundle has one audited nonblocking sidecar filename defect. Do not rewrite the bundle to repair it. Verify the manifest digest directly as shown above.

## Current target gate

Before preflight:

```bash
hostname
df -B1 /root
pgrep -af 'combined_pdn_runner|run_combined_campaign' || true
```

Stop unless:

```text
hostname = catcas
free space >= 20 GiB
no runner or campaign process is active
ACQ_ROOT does not exist
```

## Authorized acquisition preflight

```bash
CAMPAIGN="$RUNTIME_BUNDLE/sources/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign"

PYTHONDONTWRITEBYTECODE=1 python3 -B "$CAMPAIGN/catcas_preflight.py" \
  --plan-dir "$RUNTIME_BUNDLE/plan" \
  --bundle-root "$RUNTIME_BUNDLE" \
  --output-root "$ACQ_ROOT" \
  --authorization "$AUTH" \
  --report "$PREFLIGHT"
```

Required before execution:

```text
engineering_ready = true
acquisition_authorized = true
acquisition_ready = true
scientific_acquisition_started = false
physical_carrier_restoration_claimed = false
authorization_errors = []
```

The authorization has already reproduced this state against the uploaded audited bundle. The target must reproduce it again before hardware execution.

## Run the frozen campaign

Start a persistent shell:

```bash
tmux new-session -s phase6_acquisition
```

Inside that shell:

```bash
set -euo pipefail

PYTHONDONTWRITEBYTECODE=1 python3 -B "$CAMPAIGN/run_combined_campaign.py" \
  --plan-dir "$RUNTIME_BUNDLE/plan" \
  --bundle-root "$RUNTIME_BUNDLE" \
  --runner "$RUNTIME_BUNDLE/combined_pdn_runner" \
  --evidence-root "$ACQ_ROOT" \
  --authorization "$AUTH" \
  --executor-commit 81ea84f341b29c41b93667d0e0fb98e0975bcbcf \
  --pin-khz 1600000 \
  --slot-s 0.5 \
  --off-window-s 0.5 \
  --read-hz 4000 \
  --temp-veto-c 68 \
  2>&1 | tee "$RUN_LOG"
```

Do not use `--dry-run`, `--runner-validate-only`, `--engineering-smoke`, or `--mock-hardware` for the acquisition.

The orchestrator has `automatic_retry=false`. Preserve that behavior. Do not automatically resume or rerun a failed session.

## Failure boundary

On any nonzero exit:

1. Stop the campaign.
2. Preserve `$ACQ_ROOT`, `$PREFLIGHT`, and `$RUN_LOG` unchanged.
3. Confirm no runner process remains.
4. Confirm cpufreq minima, maxima, and boost have returned to baseline.
5. Do not delete, overwrite, patch, or resume the failed evidence root.
6. Return the partial evidence for audit.

A thermal veto, epoch-alignment failure, manifest failure, authorization mismatch, provenance mismatch, or cleanup mismatch is a valid failed acquisition attempt, not permission to weaken the gate.

## Completion verification

After a zero exit:

```bash
python3 -B "$CAMPAIGN/verify_run_manifests.py" "$ACQ_ROOT/runs"
```

Required output:

```text
RUN_MANIFESTS_VERIFIED count=12
```

Then independently inspect the top-level execution and every run:

```bash
python3 - "$ACQ_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
execution = json.loads((root / "execution.json").read_text(encoding="utf-8"))
assert execution["status"] == "COMPLETE"
assert execution["sessions_completed"] == execution["sessions_requested"]
assert len(execution["sessions_completed"]) == 12
assert execution["automatic_retry"] is False
assert execution["restoration_authorized"] is False
assert execution["scientific_acquisition_authorized"] is True
assert execution["executor_commit"] == "81ea84f341b29c41b93667d0e0fb98e0975bcbcf"
assert execution["source_bundle_sha256"] == "5c6588a51ce6b806e1b7b269bafd1981256795653415e012592ad3b6313fdaca"
assert execution["authorization_sha256"] == "e39fb0c6ebfb106c33a0b90b8d52d193a32833103388ebc4c6bd0cad451a0d73"

for run_dir in sorted((root / "runs").iterdir()):
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["hardware_executed"] is True
    assert run["scientific_acquisition_authorized"] is True
    assert run["authorization_artifact_sha256"] == "e39fb0c6ebfb106c33a0b90b8d52d193a32833103388ebc4c6bd0cad451a0d73"
    assert run["host_control_state_restored"] is True
    assert run["restoration_authorized"] is False
    assert run["physical_carrier_restoration_claimed"] is False

print("PHASE6_ACQUISITION_COMPLETE sessions=12")
PY
```

Confirm cleanup:

```bash
pgrep -af 'combined_pdn_runner|run_combined_campaign' && exit 1 || true
```

## Seal the returned target object

Do not add files inside `$ACQ_ROOT` after completion. Create the inventory beside it:

```bash
(
  cd "$(dirname "$ACQ_ROOT")"
  find "$(basename "$ACQ_ROOT")" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "${ACQ_ROOT}.inventory.sha256"
```

Return all of the following to the local machine without editing:

```text
$ACQ_ROOT
${ACQ_ROOT}.inventory.sha256
$PREFLIGHT
$RUN_LOG
```

## Claim boundary after acquisition

Completion proves that the frozen physical acquisition ran and produced a sealed evidence object. It does not by itself prove:

```text
physical observable-state restoration
physical carrier restoration
identified physical operator
target-to-carrier coupling
fold-odd invariant
orientation recovery
Small Wall crossing
broader physical or ontological claims
```

The next step after return is independent evidence audit and phase-native relational analysis. No restoration experiment is authorized by this runbook or artifact.

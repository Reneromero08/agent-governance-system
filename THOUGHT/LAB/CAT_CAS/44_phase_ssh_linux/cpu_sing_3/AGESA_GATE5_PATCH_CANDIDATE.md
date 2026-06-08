# AGESA Gate 5 Patch Candidate

Status: `NO_BYTE_READY_PATCH_CANDIDATE_MISSING_ARTIFACT_BLOCKER`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## Candidate Selection

Preferred order:

1. Table edit, only if P4 record and sibling P0-P3 records are proven.
2. Code-route change, only if no-op replace workflow and P4-only injection method are proven.
3. Reject current image only with hard proof.

## Current Candidate Decision

No byte-ready patch candidate exists from current artifacts.

| Route | Candidate status | Reason |
|---|---|---|
| A: table edit | Not ready | Gate 2 did not find a P4 record or sibling P0-P3 records in current bytes. |
| B: code injection | Not ready | Gate 3 no-op replacement workflow is blocked by missing rebuild tool; Gate 4 found no large enough executable cave. |
| C: reject route | Not proven | Current evidence blocks actionability, but does not prove hard impossibility. The constructor path still implies a runtime per-P-state source. |

## No Original/New Bytes

No original/new byte patch is proposed.

The rejected prior byte remains rejected:

| Raw offset | Original | Rejected edit | Reason |
|---|---|---|---|
| `0x00366E3E` | `76` | `73` | Global selector branch change; no P4 test; prior attempt did not boot. |

## P0-P3 Safety Proof

Not satisfiable yet. A safe candidate must prove one of:

- table route: P0-P3 sibling records are identified and unchanged byte-for-byte, while only P4 record changes, or
- code route: P0-P3 control flow executes byte/logic-equivalent stock selector behavior and P4 alone takes alternate behavior after testing `[ebp-4] == 0xC0010068`.

Current artifacts prove neither condition.

## P4-Only Proof

Not satisfiable yet. Current P4 distinction inside the normalizer/helper is loop-counter only. The constructor path has a 0x18 stride, but the static backing table location is not proven.

## Checksum Plan

No checksum change is proposed because no patch candidate is proposed.

If a later no-op replace workflow succeeds, checksum validation must include:

- FFS header checksum,
- FFS data checksum,
- PE32 body hash,
- UEFIExtract parse report,
- stock vs rebuilt byte diff.

## Validation Plan For Future Candidate

1. Prove target bytes and expected bytes against `cpu_hack/bios_dump.bin`.
2. Rebuild using the Gate 3 no-op-proven workflow.
3. Parse rebuilt image with UEFIExtract inside the lab folder.
4. Diff stock vs rebuilt and explain every byte change.
5. Confirm P0-P3 unchanged by table diff or decompiled control-flow proof.
6. Confirm P4-only behavior by exact record address or exact `[ebp-4] == 0xC0010068` branch path.
7. Keep all outputs inside `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux`.

## Recovery Checklist

- Do not flash from this checkpoint.
- Preserve stock dump hash and local backup.
- Preserve DualBIOS recovery notes.
- Require external programmer/recovery readiness before any future human flash decision.
- Require human review of all offsets, bytes, checksums, and parse reports.

## Gate 5 Verdict

`MISSING_ARTIFACT_BLOCKER`

Exact missing artifacts:

1. `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt`
2. `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_xrefs.txt`
3. `cpu_hack/tools/uefitool_rebuild/UEFITool.exe` or equivalent local replacement/rebuild tool
4. `cpu_hack/noop_replace/bios_noop_rebuilt.bin`
5. `cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt`

Route is still alive, but not actionable.


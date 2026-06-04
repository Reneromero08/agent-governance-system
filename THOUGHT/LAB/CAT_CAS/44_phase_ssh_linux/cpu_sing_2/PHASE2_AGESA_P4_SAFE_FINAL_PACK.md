# PHASE2_AGESA_P4_SAFE_FINAL_PACK

## Verdict

ROUTE_REJECTED_WITH_EVIDENCE

The current artifacts do not prove a P4-safe AGESA firmware route. No BIOS image was created, no flash action was prepared, and the prior global branch edit is rejected.

## Route Table

| Route | Result | Evidence |
|---|---|---|
| P4-only table edit | Rejected for current artifacts | Stock P-state qwords have zero exact hits; stride search from 8 through 64 bytes has no match; VID byte hits are too broad to identify a P4 record |
| In-place selector edit | Rejected | `0x00366E3E: 76 -> 73` affects the selector for every looped P-state, not only P4 |
| Branch to executable cave | Rejected for current artifacts | Corrected PE image has no 32-byte or larger all-zero/FF/CC/NOP run in executable sections |
| Stop loop at P3 | Rejected | This would skip P4 normalizer handling, not lower P4 VID |
| Runtime MSR route | Rejected by prior evidence | P4 write read back, but COFVID_STS stayed at CpuVid `0x1A` |

## Selected Candidate

No byte-ready candidate is selected.

Status is not `BYTE_READY_HUMAN_REVIEW`, not `TABLE_TARGET_FOUND`, and not `CODE_CAVE_PLAN_FOUND`.

## Why The AGESA Route Is Rejected From Current Artifacts

The safe requirement is narrow: P4 may change only when the current MSR is `0xC0010068`; P0-P3 must preserve stock behavior exactly.

The mapped normalizer distinguishes P4 only at:

```text
fff66e7f: inc      dword ptr [ebp - 4]
fff66e82: cmp      dword ptr [ebp - 4], 0xc0010068
fff66e89: jbe      0xfff66df6
```

The selector itself has no P4 check:

```text
fff66e3a: cmp      eax, edx
fff66e3c: je       0xfff66e7f
fff66e3e: jbe      0xfff66e42
fff66e40: mov      eax, edx
```

A safe code path would need logic equivalent to:

```text
if [ebp-4] == 0xC0010068:
    select edx for P4
else:
    run the stock cmp/je/jbe/mov logic exactly
```

That cannot be fit into the two-byte branch at `0x00366E3E`. Replacing the selector cluster with a rel32 jump would need a same-module executable cave. The corrected PE section search found no suitable executable padding run of at least 32 bytes.

## Patch Proof

### Rejected Global Edit

Known rejected bytes:

| Offset | Original | Rejected byte | Reason |
|---|---:|---:|---|
| `0x00366E3E` | `0x76` | `0x73` | Changes selector behavior for all looped P-states |
| `0x00340059` | `0x8E` | `0x91` | Checksum compensation for rejected global byte |

Checksum arithmetic for the rejected pair:

- `0x76 -> 0x73` changes byte sum by `-0x03`.
- `0x8E -> 0x91` changes byte sum by `+0x03`.
- The local byte-sum compensation is balanced, but the route is not P4-safe.

### New Candidate

No new candidate bytes are provided.

Reason: providing a byte plan without a proven P4-only table record or a proven executable cave would be a blind firmware edit. That violates the P4-safe requirement and repeats the class of failure already observed with the global selector change.

## P0-P3 Safety Proof

P0-P3 are safe only if their stock selector path is preserved. The current artifacts do not provide a byte-ready way to preserve P0-P3 while changing P4.

The rejected global edit fails this proof because P0-P3 enter the same selector:

```text
0xC0010064 -> P0
0xC0010065 -> P1
0xC0010066 -> P2
0xC0010067 -> P3
0xC0010068 -> P4
```

Any edit at `0x00366E3E` before testing `[ebp-4]` applies to P0-P3 as well as P4.

## P4 Change Proof

No P4 change is claimed.

The current artifacts prove only that P4 is identifiable by `[ebp-4] == 0xC0010068`; they do not provide a validated place to add that test and alternate selection logic.

## Validation Status

No edited candidate exists, so UEFITool parse and stock-vs-candidate diff are intentionally not run.

Validated read-only evidence:

- BIOS hash matches known value.
- Target PEIM location and GUID match known value.
- Normalizer bytes are unique in the BIOS dump.
- Stock P-state qwords are not present as exact editable records.
- Corrected PE image starts at `0x0034008C`.
- Executable sections do not contain a 32-byte or larger simple padding cave.

## Missing Artifacts Needed To Reopen AGESA Route

Any one of these could make the route actionable:

1. A decompiler-grade reconstruction of the `0xFFF737A3-0xFFF73A90` constructor path with the concrete backing table address.
2. An extracted symbol map or AGESA build map for this exact `AmdProcessorInitPeim`.
3. A proven static P4 record with entry stride, field offsets, and P0-P3 sibling records.
4. A larger, verified unused executable region in the same PEIM, with relocation and section-permission proof.
5. A clean UEFITool replace-body workflow validated on an intentionally unchanged image before any patch bytes are considered.

## Do-Not-Do List

- Do not flash.
- Do not create a flash command.
- Do not repeat `0x00366E3E: 76 -> 73`.
- Do not alter P0, P1, P2, or P3 voltage selection.
- Do not edit a table unless the P4 record and sibling P0-P3 records are proven.
- Do not use a non-executable or unproven padding area as code storage.
- Do not treat checksum balance as boot safety.


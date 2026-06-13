# PHASE2_DEEP_1_FIRMWARE

## Verdict

FIRMWARE_ROUTE_CLASSIFIED_DESIGN_ONLY

## Target Module

- BIOS dump SHA-256: `B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91`
- Module: `AmdProcessorInitPeim`
- GUID: `DE3E049C-A218-4891-8658-5FC0FA84C788`
- File offset: `0x00340048`
- File size: `0x000563D2`
- PE32 image offset: `0x00340088`
- Existing checksum byte candidate: `0x00340059 = 0x8E`

## Normalizer Loop

The local image contains the voltage normalizer at `0x00366DF0-0x00366E8F`.

Key bytes:

```text
00366E30: 09 83 E0 7F 83 E2 7F C1 EF 09 3B C2 74 41 76 02
00366E40: 8B C2 FF 75 08 83 E0 7F 33 D2 8B F8 8B DA 0F A4
00366E70: DE 89 7D F4 89 5D F8 E8 15 E0 FD FF 83 C4 0C FF
00366E80: 45 FC 81 7D FC 68 00 01 C0 0F 86 67 FF FF FF 5F
```

Loop distinction:

- `[ebp-4]` is the current P-state MSR number.
- The loop walks `MSRC001_0064` through `MSRC001_0068`.
- `0x00366E7F: FF 45 FC` increments the MSR number.
- `0x00366E82: 81 7D FC 68 00 01 C0` compares it with `0xC0010068`.
- `0x00366E89: 0F 86 67 FF FF FF` continues while the MSR number is at or below P4.

The function therefore distinguishes P4 only by the loop counter equaling `0xC0010068`; no separate P4 case exists in the visible bytes.

## Rejected Global Candidate

Known global byte edit:

```text
0x00366E3E: 0x76 -> 0x73
0x00340059: 0x8E -> 0x91
```

This reverses the compare behavior for all P-states in the loop. It is rejected as Phase 2 deep-control work because P0-P3 may be affected before the operating system can recover. It is not P4-safe.

Checksum arithmetic for the rejected edit:

- `0x76 -> 0x73` changes byte sum by `-0x03`.
- `0x8E -> 0x91` changes byte sum by `+0x03`.
- This preserves a local byte-sum checksum but does not prove boot safety or image acceptance.

## P4-Safe Candidate Requirements

A valid firmware control surface must satisfy all of these:

1. Test `[ebp-4] == 0xC0010068` before changing VID selection behavior.
2. Leave P0-P3 on the stock higher-voltage selection path.
3. Apply only to P4 or to a table entry proven to feed only P4.
4. Preserve the FFS checksum.
5. Parse cleanly in UEFITool after edit.
6. Require external recovery tooling and a verified stock image.

## Current Candidate Status

No byte-ready P4-only edit is proven.

Why:

- The visible normalizer has only a tight loop and no obvious spare bytes for a P4-only branch and alternate path.
- The stock P-state MSR values were not found as contiguous 8-byte records in the 4 MB BIOS dump, so a direct table edit target is not yet located.
- Changing the loop limit at `0x00366E82` to stop at P3 would make P4 untouched by this normalizer, not safely lower VID.

## Next Firmware Work

Design-only next steps:

1. Extract the PE32 body for `AmdProcessorInitPeim`.
2. Recover the control-flow graph around `0x00366DF0-0x00366E8F`.
3. Locate any table or constructor that supplies the compared VID fields before this normalizer.
4. If a P4-only table record is found, edit only that record and recompute the checksum.
5. If only function-level edit is possible, find a code cave in the same PEIM and design a short branch that tests `[ebp-4] == 0xC0010068`.

No flash action is prepared or recommended here.


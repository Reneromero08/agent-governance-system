# GOAL_ROUTE_2_P4_AGESA

## Verdict

P4_AGESA_ROUTE_HUMAN_APPROVAL_REQUIRED_NEEDS_MORE_RE

## BIOS Evidence

- BIOS dump SHA-256: `B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91`.
- `AmdProcessorInitPeim` report entry: file offset `0x00340048`, size `0x000563D2`, GUID `DE3E049C-A218-4891-8658-5FC0FA84C788`.
- Local hex at `0x00366E3E`: byte `0x76`.
- Existing checksum byte candidate at `0x00340059`: `0x8E`.

Relevant byte window:

```text
00366E30: 09 83 E0 7F 83 E2 7F C1 EF 09 3B C2 74 41 76 02
00366E40: 8B C2 FF 75 08 83 E0 7F 33 D2 8B F8 8B DA 0F A4
00366E50: FB 10 C1 E7 10 0B F8 0B DA 0F A4 FB 09 8D 45 F4
00366E60: 81 E1 FF 01 FF 01 50 FF 75 FC C1 E7 09 0B F9 0B
00366E70: DE 89 7D F4 89 5D F8 E8 15 E0 FD FF 83 C4 0C FF
00366E80: 45 FC 81 7D FC 68 00 01 C0 0F 86 67 FF FF FF 5F
```

## Branch Analysis

`PATCH_ANALYSIS.md` shows:

- `0x00366E3C: 74 41` skips normalization when compared VID fields are equal.
- `0x00366E3E: 76 02` is the active branch for `field_a <= field_b`.
- Stock P-states have field_a lower numeric than field_b: `0x06, 0x12, 0x14, 0x16, 0x1A < 0x20`.
- Lower numeric VID is higher voltage on this encoding.

The old global `0x76 -> 0x73` idea reverses the comparison direction, but it is not P4-only. It can affect P0-P3 and can push high-frequency states toward lower voltage during boot. That is not a safe final patch.

## P4-Only Design Constraint

A valid firmware route must:

- Identify the loop variable at `0xC0010068` and affect only P4.
- Preserve P0-P3 voltage safety.
- Include checksum correction and UEFITool verification.
- Require human approval and external recovery capability before any flash.

No flash was performed. No patched BIOS was produced for use.


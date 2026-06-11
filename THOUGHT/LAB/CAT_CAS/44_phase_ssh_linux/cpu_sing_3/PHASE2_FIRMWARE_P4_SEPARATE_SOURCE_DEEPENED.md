# PHASE2_FIRMWARE_P4_SEPARATE_SOURCE_DEEPENED

## Verdict

`SEPARATE_P4_MSR_ADDRESS_TABLES_FOUND_NOT_VALUE_TARGETS`

The cross-image scan found separate modules with full P-state MSR sibling
constants, but raw context shows these are MSR address initializers, not
editable P-state value rows.

This advances the firmware route but does not make it byte-ready.

## Inputs

- Scan report: `cpu_sing_3/PHASE2_FIRMWARE_P4_SEPARATE_SOURCE_SEARCH.md`
- Scanner: `session_scripts/phase2_firmware/find_p4_sources_across_bios.py`

## Deepened Findings

### CpuDxe

PE32 body:

```text
cpu_hack/bios_dump.bin.dump/3 8C8CE578-8A3D-4F1C-9935-896185C32DD3/15 CpuDxe/1 Compressed section/0 PE32 image section/body.bin
```

Hits:

```text
0xC0010064 raw 0x53A7
0xC0010065 raw 0x53B1
0xC0010066 raw 0x53BB
0xC0010067 raw 0x53C5
0xC0010068 raw 0x53CF
```

Context pattern:

```text
c7 81 90 00 00 00 64 00 01 c0
c7 81 94 00 00 00 65 00 01 c0
c7 81 98 00 00 00 66 00 01 c0
c7 81 9c 00 00 00 67 00 01 c0
c7 81 a0 00 00 00 68 00 01 c0
```

Interpretation: stores the five P-state MSR addresses into a structure at
offsets `+0x90` through `+0xA0`.

### CpuPei

PE32 body:

```text
cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/4 CpuPei/1 PE32 image section/body.bin
```

Hits:

```text
0xC0010064 raw 0x29E1
0xC0010065 raw 0x29EB
0xC0010066 raw 0x29F5
0xC0010067 raw 0x29FF
0xC0010068 raw 0x2A09
```

Context pattern:

```text
c7 80 90 00 00 00 64 00 01 c0
c7 80 94 00 00 00 65 00 01 c0
c7 80 98 00 00 00 66 00 01 c0
c7 80 9c 00 00 00 67 00 01 c0
c7 80 a0 00 00 00 68 00 01 c0
```

Interpretation: same five-address initializer shape, 32-bit form.

### LegacyRegion

PE32 body:

```text
cpu_hack/bios_dump.bin.dump/3 8C8CE578-8A3D-4F1C-9935-896185C32DD3/52 LegacyRegion/1 Compressed section/0 PE32 image section/body.bin
```

Hits:

```text
0xC0010064 raw 0x847
0xC0010065 raw 0x851
0xC0010066 raw 0x85B
0xC0010067 raw 0x865
0xC0010068 raw 0x86F
```

Interpretation: same five-address initializer shape as `CpuDxe`.

## Actionability

These are not P4 value rows. Editing `0xC0010068` here would alter which MSR
address the module refers to, not the P4 VID/FID/DID value itself.

Therefore:

```text
BYTE_READY_HUMAN_REVIEW = no
P4_ONLY_VALUE_TARGET = no
P0_P3_UNCHANGED_VALUE_PROOF = not applicable
```

## Route Impact

The firmware search did find separate P4-related bytes, but they are address
tables rather than value records:

```text
FIRMWARE_P4_SEPARATE_SOURCE_FOUND_NOT_ACTIONABLE
```

## Next Exact Action

`FIRMWARE_PSTATE_VALUE_PATTERN_SEARCH`

Search for encoded P-state value shapes, not MSR address constants:

- stock P0-P4 raw value fragments
- FID/DID/VID field masks
- table rows with five sibling entries
- stores into P-state definition registers using non-constant source data
- references that join a P-state index to a value payload

## Boundary

- Search only; no image modification.
- No candidate construction.
- No platform setting changes.

# PHASE2_CPU_PEI_VALUE_CONSUMER_TRACE

## Verdict

`CPU_PEI_VALUE_CONSUMER_NOT_FOUND_RAW_TRACE`

The CpuPei P-state sibling constants are confirmed as a compact MSR address
initializer. A raw displacement trace did not reveal a P4 value consumer or
value payload.

## Input

```text
cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/4 CpuPei/1 PE32 image section/body.bin
```

## Address Initializer

The P-state address table appears at raw `0x29DD-0x2A0D`:

```text
c7 80 90 00 00 00 64 00 01 c0
c7 80 94 00 00 00 65 00 01 c0
c7 80 98 00 00 00 66 00 01 c0
c7 80 9c 00 00 00 67 00 01 c0
c7 80 a0 00 00 00 68 00 01 c0
```

Interpretation:

```text
struct +0x90 = 0xC0010064
struct +0x94 = 0xC0010065
struct +0x98 = 0xC0010066
struct +0x9C = 0xC0010067
struct +0xA0 = 0xC0010068
```

## Displacement Trace

| Displacement | Hit count | Relevant finding |
|---|---:|---|
| `+0x90` | 1 | Initializer only. |
| `+0x94` | 1 | Initializer only. |
| `+0x98` | 1 | Initializer only. |
| `+0x9C` | 1 | Initializer only. |
| `+0xA0` | 2 | Initializer plus one unrelated raw/table-looking occurrence at `0x1E8`. |

## Interpretation

This angle finds:

```text
P0-P4 MSR address table = yes
P4 value payload = no
P4-only value consumer = no
```

`CpuPei` does not provide a byte-ready P4-safe value target from the current raw
trace.

## Route Impact

```text
CPU_PEI_VALUE_CONSUMER_NOT_FOUND_RAW_TRACE
```

This is not:

- `BYTE_READY_HUMAN_REVIEW`
- `CPU_SINGS`
- `SOFTWARE_FIRMWARE_TRUE_WALL`

## Next Exact Action

`LEGACY_REGION_VALUE_CONSUMER_TRACE`

Repeat the same raw trace on `LegacyRegion`, the remaining module with full
P0-P4 sibling address constants.

## Boundary

- Trace only; no image modification.
- No candidate construction.
- No platform setting changes.

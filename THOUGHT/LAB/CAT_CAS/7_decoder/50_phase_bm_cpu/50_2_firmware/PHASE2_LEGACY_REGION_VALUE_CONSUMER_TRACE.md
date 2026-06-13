# PHASE2_LEGACY_REGION_VALUE_CONSUMER_TRACE

## Verdict

`LEGACY_REGION_VALUE_CONSUMER_NOT_FOUND_RAW_TRACE`

`LegacyRegion` contains the same P0-P4 MSR address initializer shape as
`CpuDxe`, but the raw displacement trace did not expose a P4 value payload or
value consumer.

## Input

```text
50_2_firmware/cpu_hack/bios_dump.bin.dump/3 8C8CE578-8A3D-4F1C-9935-896185C32DD3/52 LegacyRegion/1 Compressed section/0 PE32 image section/body.bin
```

## Address Initializer

The P-state address table appears at raw `0x843-0x873`:

```text
c7 81 90 00 00 00 64 00 01 c0
c7 81 94 00 00 00 65 00 01 c0
c7 81 98 00 00 00 66 00 01 c0
c7 81 9c 00 00 00 67 00 01 c0
c7 81 a0 00 00 00 68 00 01 c0
```

## Displacement Trace

| Displacement | Hit count | Relevant finding |
|---|---:|---|
| `+0x90` | 2 | Initializer plus stack-frame-looking occurrence. |
| `+0x94` | 1 | Initializer only. |
| `+0x98` | 2 | Initializer plus stack-frame-looking occurrence. |
| `+0x9C` | 1 | Initializer only. |
| `+0xA0` | 1 | Initializer only. |

## Interpretation

This closes the third separate P-state address-table module:

```text
CpuDxe = address table, no value consumer found
CpuPei = address table, no value consumer found
LegacyRegion = address table, no value consumer found
```

## Route Impact

```text
LEGACY_REGION_VALUE_CONSUMER_NOT_FOUND_RAW_TRACE
```

This is not:

- `BYTE_READY_HUMAN_REVIEW`
- `CPU_SINGS`
- `SOFTWARE_FIRMWARE_TRUE_WALL`

## Good Pause Point

Pause here. The firmware route has a clean current boundary:

```text
all known separate P0-P4 address-table modules traced
no P4 value payload found
no byte-ready target
```

Next resume action:

```text
PHASE2_SOFTWARE_FIRMWARE_WALL_OR_NEXT_ROUTE_REVIEW
```

Review whether any non-repeated software/firmware route remains inside current
constraints. If yes, pursue it. If no, prepare the terminal route pack with the
exact missing artifact.

## Boundary

- Trace only; no image modification.
- No candidate construction.
- No platform setting changes.

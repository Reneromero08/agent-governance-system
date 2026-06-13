# PHASE2_CPU_DXE_VALUE_CONSUMER_TRACE

## Verdict

`CPU_DXE_VALUE_CONSUMER_NOT_FOUND_RAW_TRACE`

The CpuDxe P-state sibling constants are confirmed as structure initializers for
MSR address fields. A raw displacement trace did not reveal a separate P4 value
consumer or value payload.

## Input

```text
50_2_firmware/cpu_hack/bios_dump.bin.dump/3 8C8CE578-8A3D-4F1C-9935-896185C32DD3/15 CpuDxe/1 Compressed section/0 PE32 image section/body.bin
```

## Address Initializer

The P-state address table appears at raw `0x53A3-0x53D3`:

```text
c7 81 90 00 00 00 64 00 01 c0
c7 81 94 00 00 00 65 00 01 c0
c7 81 98 00 00 00 66 00 01 c0
c7 81 9c 00 00 00 67 00 01 c0
c7 81 a0 00 00 00 68 00 01 c0
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

Raw displacement scan:

| Displacement | Hit count | Relevant finding |
|---|---:|---|
| `+0x90` | 17 | Mostly stack-frame displacement noise; initializer hit at `0x53A3`. |
| `+0x94` | 1 | Initializer only. |
| `+0x98` | 11 | Mostly stack-frame displacement noise; initializer hit at `0x53B7`. |
| `+0x9C` | 1 | Initializer only. |
| `+0xA0` | 6 | Mostly stack-frame displacement noise; initializer hit at `0x53CB`. |

The odd-index fields `+0x94` and `+0x9C` appearing only in the initializer is
strong evidence that this raw trace did not expose a same-module value consumer.

## Interpretation

This angle finds:

```text
P0-P4 MSR address table = yes
P4 value payload = no
P4-only value consumer = no
```

Editing this table would change address references, not the P4 value. It is not
a byte-ready P4-safe value target.

## Route Impact

```text
CPU_DXE_VALUE_CONSUMER_NOT_FOUND_RAW_TRACE
```

This is not:

- `BYTE_READY_HUMAN_REVIEW`
- `CPU_SINGS`
- `SOFTWARE_FIRMWARE_TRUE_WALL`

## Next Exact Action

`CPU_PEI_VALUE_CONSUMER_TRACE`

Repeat the same raw trace on `CpuPei`, because CpuPei contains the same five
P-state sibling constants in 32-bit form.

## Boundary

- Trace only; no image modification.
- No candidate construction.
- No platform setting changes.

# PHASE2_F10_SOURCE_PSTATE_VALUE_PROVENANCE

## Verdict

`F10_SOURCE_CONFIRMS_RUNTIME_PSTATE_VALUE_PROVENANCE`

The local AGESA Family 10h source drops confirm the P-state value path is
runtime-derived. They expose a real gather/leveling path for P-state values, but
the values are read from live MSRs into an in-memory `S_CPU_AMD_PSTATE` /
`PSTATE_LEVELING` buffer before any leveling write occurs.

This advances provenance, but it does not create `BYTE_READY_HUMAN_REVIEW`.
No static P4-only value row or editable image byte source was found.

## Source Artifacts Inspected

| Source | Purpose |
|---|---|
| `50_2_firmware/cpu_hack/_tmp_coreboot_chromium_f10/src/vendorcode/amd/agesa/f10/Proc/CPU/Feature/cpuPstateGather.c` | Builds the runtime P-state buffer from current hardware state. |
| `50_2_firmware/cpu_hack/_tmp_coreboot_chromium_f10/src/vendorcode/amd/agesa/f10/Proc/CPU/Feature/cpuPstateLeveling.c` | Normalizes gathered P-state data and starts cross-core register modification. |
| `50_2_firmware/cpu_hack/_tmp_coreboot_chromium_f10/src/vendorcode/amd/agesa/f10/Proc/CPU/Family/0x10/cpuF10Utilities.c` | Family 10h MSR read/write service implementations. |
| `50_2_firmware/cpu_hack/_tmp_coreboot_chromium_f10/src/vendorcode/amd/agesa/f10/Include/OptionFamily10h*.h` | Confirms Family 10h service tables install `F10GetPstateRegisterInfo`, `F10GetPstateMaxState`, and `F10PstateLevelingCoreMsrModify` where the feature is enabled. |

## Provenance Chain

```text
PStateGatherMain
  -> FamilySpecificServices->GetPstateMaxState
       -> F10GetPstateMaxState
            -> LibAmdMsrRead(MSR_PSTATE_CURRENT_LIMIT)

  -> for k in enabled P-states:
       FamilySpecificServices->GetPstateRegisterInfo(k)
          -> F10GetPstateRegisterInfo
               -> LibAmdMsrRead(PS_REG_BASE + k)
               -> returns PsEnable, IddValue, IddDiv

       FamilySpecificServices->GetPstateFrequency(k)
          -> reads PS_REG_BASE + k

       FamilySpecificServices->GetPstatePower(k)
          -> reads PS_REG_BASE + k

       PStateBufferPtr->PStateCoreStruct[0].PStateStruct[k]
          <- CoreFreq, Power, IddValue, IddDiv, PStateEnable
```

For P4, `PS_REG_BASE + 4` maps to `MSRC001_0068`, matching the prior decoded
binary provenance.

## Leveling Write Path

The source also confirms the buffer can become a write source under P-state
leveling:

```text
StartPstateMsrModify
  -> FamilySpecificServices->SetPStateLevelReg
       -> F10PstateLevelingCoreMsrModify
            -> selects per-socket PSTATE_LEVELING buffer
            -> updates MSR_PSTATE_0 / MSR_PSTATE_1 in one-P-state case
            -> otherwise iterates MSR_PSTATE_0.. and writes fields from
               PStateBufferPtrTmp->PStateCoreStruct[0].PStateStruct[k]
```

Important fields written from the gathered/leveled buffer include:

- `CpuFid`
- `CpuDid`
- `IddValue`
- `IddDiv`
- `PsEnable`

This is a real firmware-level runtime value path. It is not a static ROM table.

## Why This Is Not Byte-Ready

The source proves the following:

```text
P4_VALUE_SOURCE = MSRC001_0068 runtime read
P4_BUFFER_SLOT = PStateStruct[4] in runtime PSTATE_LEVELING buffer
LEVELING_WRITE_SOURCE = runtime PSTATE_LEVELING buffer
STATIC_P4_VALUE_ROW = not found
IMAGE_OFFSET_FOR_P4_ONLY_EDIT = not found
P0_P3_UNCHANGED_BYTE_PROOF = not available
```

The actionable edit target required for candidate construction is still absent:

```text
editable P4-only value source with P0-P3 sibling proof
```

## Relation To Prior Artifacts

This report agrees with and strengthens:

- `50_2_firmware/PHASE2_FW_ARG0C_PROVENANCE.md`
- `50_2_firmware/cpu_hack/agesa_trace/AmdProcessorInitPeim_msr_source_proof.txt`
- `50_2_firmware/PHASE2_FIRMWARE_PSTATE_VALUE_PATTERN_SEARCH.md`
- `50_2_firmware/PHASE2_FIRMWARE_SOURCE_PROVENANCE_WALL_AUDIT.md`

It also explains why CpuDxe/CpuPei/LegacyRegion P0-P4 sibling constants looked
promising but did not expose value payloads: the source model expects P-state
values to be gathered from live registers into runtime buffers, while static
image constants can be only MSR addresses, service vectors, or option tables.

## Gate Status

| Gate | Status | Reason |
|---|---|---|
| Runtime provenance | `CONFIRMED` | Source-level flow reads `PS_REG_BASE + k`; P4 is `MSRC001_0068`. |
| Separate static value source | `NOT_FOUND` | No source evidence of ROM-resident P0-P4 value rows. |
| P4-only candidate | `NOT_BYTE_READY` | Missing exact image offset/bytes/checksum and P0-P3 unchanged proof. |
| Firmware route | `ALIVE_FOR_SOURCE_PROVENANCE_ONLY` | Useful for understanding, not enough for candidate construction. |

## Exact Next Action

`SOFTWARE_FIRMWARE_WALL_REVIEW_AFTER_F10_SOURCE_PROVENANCE`

Do not repeat no-op rebuild, raw address-table consumer traces, or the decoded
`0xFFF4CF9C -> 0xFFF7348D -> 0xFFF44E76` chain. The next useful firmware move
would require a new local artifact, such as:

- another same-board BIOS revision with structurally different P-state handling,
- a map/symbol/debug artifact tying the runtime `PSTATE_LEVELING` buffer to a
  patchable image-level initializer,
- or a rebuildable source-to-image path for the exact board firmware.

Without one of those, the firmware branch remains provenance-complete but not
candidate-ready.

## Boundary

- No image modification.
- No candidate bytes.
- No platform setting changes.
- No deployment action.

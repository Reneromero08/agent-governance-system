# PHASE2_FIRMWARE_SOURCE_PROVENANCE_WALL_AUDIT

## Verdict

`FIRMWARE_P4_VALUE_SOURCE_NOT_FOUND_CURRENT_ARTIFACTS`

The firmware route is not byte-ready. Current artifacts prove no editable P4
value source in the decoded paths and no static stock P4 value pattern in the
extracted image tree.

This audit does not claim `SOFTWARE_FIRMWARE_TRUE_WALL`; it records the current
firmware boundary and the exact missing artifact.

## Route Audit

| Route | Artifact | Result | Missing proof |
|---|---|---|---|
| AGESA global branch edit | `cpu_hack/agesa_trace/PATCH_ANALYSIS.md` | Rejected. Global behavior, not P4-safe, prior boot failure. | None; do not repeat. |
| Constructor path | `cpu_sing_3/PHASE2_FW_ARG0C_PROVENANCE.md` | `0xFFF7371A` consumes `selected_base + pstate*0x18`; P4 field maps through producer/service helpers. | Static editable P4 value source. |
| Runtime-MSR provenance | `cpu_hack/agesa_trace/AmdProcessorInitPeim_msr_source_proof.txt` | P4 field resolves to runtime `MSRC001_0068` read path. | Firmware-resident P4 value row. |
| Helper-layer probe | `cpu_sing_3/PHASE2_P4_EDIT_SOURCE_PROOF.md` | Current helper layer closed; no editable P4-only source exposed. | Separate P4-affecting source outside decoded chain. |
| No-op rebuild | `cpu_sing_3/PHASE2_MASTER_B_REBUILD_TOOLCHAIN.md` | Identical rebuild/save path proven parse-clean and byte-identical. | Non-no-op P4-only edit target. |
| Donor workflow | `cpu_sing_3/PHASE2_DONOR_DIFF_REPORT.md` | Public donor shows free-space insertion workflow only. | P4-safe value edit lesson. |
| Separate P4 source scan | `cpu_sing_3/PHASE2_FIRMWARE_P4_SEPARATE_SOURCE_DEEPENED.md` | CpuDxe/CpuPei/LegacyRegion sibling constants found, but they are MSR address initializers. | P4 value payload, not address table. |
| P4 value-pattern search | `cpu_sing_3/PHASE2_FIRMWARE_PSTATE_VALUE_PATTERN_SEARCH.md` | Stock P4 full value and key fragments have zero hits in extracted image tree. | Any static P4 value bytes tied to P0-P4 siblings. |
| Family 10h source provenance | `cpu_sing_3/PHASE2_F10_SOURCE_PSTATE_VALUE_PROVENANCE.md` | Local AGESA F10 source confirms P-state values are gathered from live `PS_REG_BASE + k` MSRs into runtime `PSTATE_LEVELING` buffers, and leveling writes from those buffers. | Static editable P4 value source or exact source-to-image path. |

## Current Firmware Boundary

```text
NOOP_REBUILD_PROVEN = yes
P4_FIELD_RUNTIME_MSR_DERIVED = yes
SEPARATE_P4_MSR_ADDRESS_TABLES_FOUND = yes
STATIC_P4_VALUE_PATTERN_FOUND = no
F10_SOURCE_RUNTIME_BUFFER_PATH_CONFIRMED = yes
P4_ONLY_VALUE_TARGET = no
BYTE_READY_HUMAN_REVIEW = no
```

## Exact Missing Artifact

The missing artifact is:

```text
editable P4-only value source with P0-P3 sibling proof
```

It must show:

- P0-P3 unchanged
- P4-only effect
- exact offset/bytes/checksum path
- clean parse after non-no-op candidate construction
- no reliance on the rejected global branch edit
- no reliance on MSR address-table edits as value edits

## Remaining Live Firmware Angles

No current artifact provides a byte-ready target. The remaining firmware angles
inside current constraints are acquisition/provenance tasks:

- trace CpuDxe/CpuPei address-table consumers to prove whether they ever join a
  separate value payload
- acquire or locate a new local artifact that bridges the source-level runtime
  `PSTATE_LEVELING` buffer to a patchable image-level initializer
- compare another same-board firmware revision if available locally to see
  whether P-state value handling changed structurally

## Route Impact

Firmware status:

```text
FIRMWARE_P4_VALUE_SOURCE_NOT_FOUND_CURRENT_ARTIFACTS
```

This is not:

- `BYTE_READY_HUMAN_REVIEW`
- `CPU_SINGS`
- `SOFTWARE_FIRMWARE_TRUE_WALL`
- `HUMAN_TOOL_REQUIRED_WITH_ALL_OTHER_ROUTES_EXHAUSTED`

## Next Exact Action

`SOFTWARE_FIRMWARE_WALL_REVIEW_AFTER_F10_SOURCE_PROVENANCE`

The CpuDxe/CpuPei/LegacyRegion consumer traces and the Family 10h source pass
now agree: the current local artifacts expose runtime MSR/value-buffer
provenance, not a static P4-only value row. Continue only if a new local artifact
appears that can bridge source-level `PSTATE_LEVELING` data to a patchable image
initializer or a same-board firmware revision with different P-state structure.

## Boundary

- No image modification.
- No candidate construction.
- No platform setting changes.

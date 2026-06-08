# AGESA Next B Table Reopen

Status: `TABLE_SOURCE_RUNTIME_OR_DISPATCH_SELECTED_STATIC_RECORD_NOT_FOUND`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## Inputs

- `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_xrefs.txt`
- `cpu_hack/agesa_trace/pstate_targeted_disasm.txt`
- `cpu_hack/bios_dump.bin`
- `cpu_hack/bios_dump.bin.dump/.../AmdProcessorInitPeim/1 PE32 image section/body.bin`

## Reopened Source Model

The constructor path does not read from an immediate static address. It reads from a selected base:

```text
selected_base = [ebp-8]
selected_base initially = arg_0C + 8
selected_base may be changed by helper 0xFFF4CF55(&selected_base, arg_0C, index, arg_10)
```

The multi-entry path then treats `selected_base` as:

```text
max_pstate     = byte  [selected_base + 0x0B]
initial_pstate = byte  [selected_base + 0x0F]
record         = selected_base + initial_pstate * 0x18
record_enable  = dword [record + 0x10]
record_param   = dword [record + 0x14]
record_byte    = byte  [record + 0x1C]
record_2bit    = dword [record + 0x20] & 3
```

## Table Hunt Result

The decompile/xref pass strengthens the runtime/dispatch-selected source interpretation:

- The base is passed into function `0xFFF7371A` as an argument.
- There is no direct immediate/static table address in the constructor block.
- A function pointer to `0xFFF7371A` exists in `.dG3_DXE` near CPU model strings, implying a dispatch-table family/model route.
- Prior strict `0x18` static record scans found no P0-P4 sibling cluster.

## P0-P4 Record Status

`TABLE_TARGET_FOUND` is not satisfied.

| Requirement | Current evidence |
|---|---|
| P4 record address | Not proven |
| P0-P3 sibling records | Not proven |
| Stride `0x18` | Proven inside runtime constructor logic |
| Fields `+0x10`, `+0x14`, `+0x1C`, `+0x20` | Proven inside runtime constructor logic |
| Static editable backing bytes | Not proven |
| Runtime-only source | Strongly indicated, not fully proven until dispatcher/table consumer is recovered |

## Next Concrete Subgoal

The next table/source hunt must chase the `.dG3_DXE` dispatch table consumer, not repeat blind qword scans.

Exact missing artifact:

`cpu_hack/agesa_trace/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt`

Required contents:

1. Code that references or iterates the `.dG3_DXE` table around `0xFFF8D0EC-0xFFF8D130`.
2. The call/dispatch mechanism that reaches function pointer `0xFFF7371A`.
3. The caller arguments used for `arg_0C` and `arg_10`.
4. Whether `arg_0C` points into static `.dG3_DXE` data, copied/compressed AGESA tables, heap/runtime allocated structures, or platform service output.

## Gate B Decision

`TABLE_SOURCE_RUNTIME_OR_DISPATCH_SELECTED_STATIC_RECORD_NOT_FOUND`

The table route remains alive because the constructor clearly consumes per-P-state records. It is not actionable because the P4 record and P0-P3 siblings are not tied to editable static bytes.

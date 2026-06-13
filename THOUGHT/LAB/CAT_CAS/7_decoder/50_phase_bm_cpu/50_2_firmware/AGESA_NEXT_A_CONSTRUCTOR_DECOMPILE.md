# AGESA Next A Constructor Decompile

Status: `CONSTRUCTOR_FUNCTION_IDENTIFIED_SOURCE_NOT_FULLY_PROVEN`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## Artifacts Produced

- `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_xrefs.txt`

## Function Boundary

The requested `0xFFF737A3` block is not a function entry. The containing function is:

| Item | Value |
|---|---|
| Function entry | `0xFFF7371A` |
| Block of interest | `0xFFF737A3-0xFFF73A90` |
| Function return | `0xFFF73A93` |
| Section | `.tG3_DXE` |

## Stack Frame And Base Flow

The function starts with:

```text
fff7371a: push ebp
fff7371b: mov  ebp, esp
fff7371d: sub  esp, 0x38
```

Recovered base flow:

| Field/register | Meaning from local disassembly |
|---|---|
| `[ebp+0x0C]` | caller-supplied constructor/table context pointer |
| `edi` | initialized from `[ebp+0x0C]` |
| `[ebp-0x20]` | `dword [edi+0x00]`, used as count/search bound before constructor block |
| `esi` | initialized to `edi + 8` |
| `[ebp-8]` | selected base pointer, initialized to `edi + 8`, optionally updated by helper `0xFFF4CF55` |
| `ecx` at `0xFFF737A3` | selected base pointer loaded from `[ebp-8]` |

Assignments feeding `[ebp-8]`, `ecx`, and `esi` before `0xFFF737A3`:

```text
fff73730: mov edi, dword ptr [ebp + 0xc]
fff73733: mov eax, dword ptr [edi]
fff73739: mov dword ptr [ebp - 0x20], eax
fff7374c: lea esi, [edi + 8]
fff73750: mov dword ptr [ebp - 8], esi
fff73782: call 0xfff4cf55      ; may update [ebp-8] through pushed lea [ebp-8]
fff73787: mov ecx, dword ptr [ebp - 8]
fff737a3: mov ecx, dword ptr [ebp - 8]
```

## Field Provenance

| Field | Source |
|---|---|
| `+0x0B` | `byte [selected_base + 0x0B]`, max P-state index for loop bound |
| `+0x0F` | `byte [selected_base + 0x0F]`, initial/current P-state index |
| `+0x10` | `dword [selected_base + pstate*0x18 + 0x10]`, per-record enable flag |
| `+0x14` | `dword [selected_base + pstate*0x18 + 0x14]`, encoded frequency-like input to `0xFFF73559` |
| `+0x1C` | `byte [selected_base + pstate*0x18 + 0x1C]`, byte inserted into constructed MSR value |
| `+0x20` | `dword [selected_base + pstate*0x18 + 0x20] & 3`, two-bit field inserted into constructed MSR value |

## Xrefs

Direct call xrefs:

- `0xFFF7371A`: none found.
- `0xFFF737A3`: none found.

Table/pointer xref:

- `.dG3_DXE` VA `0xFFF8D11E` contains pointer `0xFFF7371A`.
- The pointer is adjacent to other function pointers and CPU model strings.
- This points to dispatch-table selection rather than direct call.

## Gate A Decision

`CONSTRUCTOR_FUNCTION_IDENTIFIED_SOURCE_NOT_FULLY_PROVEN`

The containing function, stack frame, field base, and local constructor semantics are now recovered from current bytes. The constructor consumes a selected runtime/caller-supplied structure, but the upstream dispatcher/table consumer for `.dG3_DXE` pointer `0xFFF8D11E` is not proven.

Exact next missing artifact:

`cpu_hack/agesa_trace/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt`

That artifact must show the code path that consumes the `.dG3_DXE` table around `0xFFF8D0EC-0xFFF8D130` and passes the constructor/table context into function `0xFFF7371A`.

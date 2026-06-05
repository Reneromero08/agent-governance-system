# PHASE2_FW_ARG0C_PROVENANCE

## Verdict

`ARG0C_RUNTIME_PRODUCED_STRUCTURE`

Route 2 advanced. `arg_0C` behind `0xFFF7371A` is not a direct static P-state table pointer. It is a caller-supplied structure with a count at `+0x00`, a default record at `+0x08`, and variable-length sibling records selected by helper `0xFFF4CF55`.

## New Artifact

- `cpu_hack/AmdProcessorInitPeim_helper_fff4cf55_disasm.txt`
- `cpu_hack/AmdProcessorInitPeim_pointer_search_fff4cf9c.txt`
- `cpu_hack/AmdProcessorInitPeim_callback_fff4aadd_disasm.txt`
- `cpu_hack/AmdProcessorInitPeim_fff4aadd_wide_disasm.txt`
- `cpu_hack/AmdProcessorInitPeim_descriptor_handlers_disasm.txt`
- `cpu_hack/AmdProcessorInitPeim_descriptor_callsite_xrefs.txt`
- `cpu_hack/AmdProcessorInitPeim_outer_producer_table_xrefs.txt`
- `cpu_hack/AmdProcessorInitPeim_service_table_trace.txt`

## Evidence

| Item | Finding |
|---|---|
| Constructor function | `0xFFF7371A` |
| Constructor table argument | `[ebp+0x0C]`, copied to `edi` |
| Default selected base | `arg_0C + 8` |
| Selection helper | `0xFFF4CF55` |
| Helper call sites | `0xFFF4D1FC`, `0xFFF4EDCE`, `0xFFF73782` |
| Direct call to `0xFFF7371A` | none found |
| Static pointer to constructor | `.dG3_DXE` entry `0xFFF8D11E -> 0xFFF7371A` |
| Existing heap/table consumer | `0xFFF72B3C`, consumes heap handles at `0xFFF8D0EC-0xFFF8D104` |

## Helper `0xFFF4CF55`

Recovered behavior:

```text
outptr = arg_04
ctx    = arg_08
index  = arg_0C

if index > dword[ctx + 0x00]:
    return 1

p = ctx + 0x08
if index >= 1:
    do:
        p += word[p + 0x02]
        index--
    while index != 0

dword[outptr] = p
return 0
```

Meaning:

- `arg_0C` to `0xFFF7371A` is a record-list base.
- The first dword is a record count or upper bound.
- The first record starts at `arg_0C + 8`.
- Each sibling record carries its own length at `record + 0x02`.
- The selected record is written back through the caller's local `selected_base`.

## Constructor Use Of Selected Record

Inside `0xFFF7371A`:

```text
selected_base = arg_0C + 8
helper_0xFFF4CF55(&selected_base, arg_0C, i, arg_10)

max_pstate     = byte  [selected_base + 0x0B]
initial_pstate = byte  [selected_base + 0x0F]
record         = selected_base + initial_pstate * 0x18

record + 0x10  enable flag
record + 0x14  encoded input to helper 0xFFF73559
record + 0x1C  byte inserted into constructed MSR value
record + 0x20  dword masked with 3 and shifted into constructed MSR value
```

This preserves the live P4 clue: the constructor still has a per-P-state stride of `0x18`. What changed is the source classification. The P-state records are nested inside a selected variable-length record, not at a proven fixed `.dG3_DXE` address.

## Producer Clues

The disassembly window around `0xFFF4CF9C-0xFFF4D12E` constructs a structure that matches this shape:

- writes platform bytes to `esi + 0x01`, `+0x0B`, `+0x0F`, `+0x0D`, `+0x0C`, `+0x0E`
- initializes per-entry payload at `esi + 0x18`
- zeroes `0x18` bytes per entry through helper `0xFFF493E0`
- stores per-entry values at `entry + 0x04`, `+0x08`, `+0x0C`
- stores an enable flag at `entry - 0x08`
- computes a variable record length as `0x28 + max_pstate * 0x18`
- writes that length to `record + 0x02`
- calls `0xFFF4CF55` at `0xFFF4D1FC` to advance to the next variable-length record

No direct E8 call to `0xFFF4CF9C` was found in the PE32 body.

The raw pointer search did find exactly one `0xFFF4CF9C` pointer reference:

```text
raw pointer bytes: 9C CF F4 FF
hit raw: 0x00D122
hit VA:  0xFFF4D1AE
```

Disassembly around that hit shows the producer is placed into a stack descriptor and passed to `0xFFF4AADD`:

```text
0xFFF4D19A: mov word ptr [ebp - 0x34], ax
0xFFF4D19E: mov eax, dword ptr [ebp - 8]
0xFFF4D1A3: mov dword ptr [ebp - 0x32], eax
0xFFF4D1A6: lea eax, [ebp - 0x38]
0xFFF4D1AB: mov dword ptr [ebp - 0x38], 0xfff4cf9c
0xFFF4D1B2: mov dword ptr [ebp - 0x2a], edi
0xFFF4D1B5: mov dword ptr [ebp - 0x2e], edi
0xFFF4D1B8: call 0xfff4aadd
```

`0xFFF4AADD` resolves to a descriptor interpreter at `0xFFF4AA00`. It iterates entries starting at descriptor `+0x0A`, dispatches on `byte[entry+4]`, and calls typed handlers `0xFFF4A175`, `0xFFF4A34A`, `0xFFF4A540`, or `0xFFF4A676`.

The callsite xref artifact also resolves the local value copied into the descriptor payload before `0xFFF4D1AB`: `[ebp-8]` is initialized from `arg_0C + 8` in the outer producer function at `0xFFF4D12F`. That means this stack descriptor is not allocating the selected record source itself; it is feeding the already selected/default record body into the callback producer.

The outer producer is registered through a static `.data` function pointer slot:

```text
raw pointer bytes for 0xFFF4D12F: 2F D1 F4 FF
hit raw: 0x03F48A
hit VA:  0xFFF7F516

0xFFF7F516: 0xFFF4D12F
0xFFF7F51A: 0xFFF4CF51
0xFFF7F522: 0xFFF73B54
0xFFF7F526: 0xFFF73D74
0xFFF7F52A: 0xFFF741FD
```

Slot consumers found locally:

```text
0xFFF4CF94 references/calls [0xFFF7F516]
0xFFF4CF4D jumps through [0xFFF7F51A]
0xFFF72F63 references 0xFFF7F522
```

`0xFFF7F516` is in `.data`, not `.dG3_DXE`. It is a producer function pointer, not a P-state record row or a P4 VID byte. This classifies the path as static service registration plus callback/descriptor interpretation that produces runtime records.

The service-table trace advances the next layer:

```text
0xFFF4CF92: call dword ptr [0xfff7f516]
0xFFF4D149: call 0xfff49a34
0xFFF4D15D: call 0xfff474f6
0xFFF4D173: call 0xfff47af3
0xFFF4D190: call dword ptr [eax + 0x1e]
0xFFF4D1B8: call 0xfff4aadd
0xFFF4D1FC: call 0xfff4cf55
```

The typed descriptor handlers then fan out through helper calls and indexed function tables:

```text
handler 0xFFF4A175: call dword ptr [ecx + eax*8]
handler 0xFFF4A34A: call dword ptr [ecx + eax*8], [ecx + eax*8 + 4]
handler 0xFFF4A540: call dword ptr [eax + edi], [ecx + edi + 4]
handler 0xFFF4A676: call dword ptr [eax + edi], [ecx + edi + 4]
```

This moves the remaining source hunt to descriptor-entry and service/function-table provenance.

## Classification

| Question | Current answer |
|---|---|
| Static direct table? | A `.data` function-pointer registration exists at `0xFFF7F516 -> 0xFFF4D12F`; it is a producer registration, not static P-state records |
| Copied/compressed table? | Possible for the produced payload, but not proven |
| Heap-allocated? | Likely for the containing object or producer output; the `0xFFF4D12F` callsite proves `[ebp-8] = arg_0C + 8`, not a local allocation |
| Service-produced? | Strongly indicated by the `.data` producer registration, the `0xFFF4CF9C` callback descriptor passed to `0xFFF4AADD`, and service calls through tables such as `[eax+0x12]`, `[eax+0x1A]`, `[eax+0x1E]`, `[eax+0x22]` |
| P0-P4 sibling records found? | Runtime record shape found; editable static sibling bytes not found |
| P4 field identified? | Runtime field is still `selected_base + pstate*0x18 + 0x1C` for the VID-like byte, but no static P4 byte offset is proven |

## Deepest Progress

The deepest current provenance is:

```text
.dG3_DXE function pointer array
  0xFFF8D11E -> 0xFFF7371A

0xFFF7371A receives arg_0C
  arg_0C[0] = count/upper bound
  records begin at arg_0C + 8
  helper 0xFFF4CF55 selects sibling record by walking record + word[record+2]
  selected record contains per-P-state 0x18-stride entries

0xFFF4CF9C producer callback
  builds variable records
  writes record length at +0x02
  writes max/current P-state fields
  fills per-P-state entries from platform/service callbacks
  is passed through a stack descriptor to 0xFFF4AADD

0xFFF4D12F outer producer registration
  static .data slot 0xFFF7F516 -> 0xFFF4D12F
  [ebp-8] before 0xFFF4D1AB = arg_0C + 8
  descriptor payload uses that existing selected/default record body

0xFFF4AADD typed descriptor handlers
  invoke indexed function tables
  no static P4 record row is exposed by this layer
  next target is descriptor-entry source and service table provenance
```

## Next Exact RE Step

Resolve the service table feeding `0xFFF4D12F` and the typed descriptor handlers behind `0xFFF4AADD`, especially the callbacks invoked by handler table indexes in `0xFFF4A175`, `0xFFF4A34A`, `0xFFF4A540`, and `0xFFF4A676`.

Concrete local command:

```powershell
rg -n "FFF4CF9C|FFF4D1FC|FFF7371A|FFF8D11E|0xFFF4CF9C|0xFFF7371A" cpu_hack cpu_sing_3
```

Do not repeat the raw pointer search for `0xFFF4CF9C` or `0xFFF4D12F`: those have already produced the callback descriptor hit at `0xFFF4D1AE` and the `.data` producer slot at `0xFFF7F516`.

## Actionability

`TABLE_TARGET_FOUND` is not met.

`BYTE_READY_HUMAN_REVIEW` is not met.

The firmware route remains alive because the runtime P-state record shape is now clearer and a likely producer window is identified. It is not byte-ready because no editable source bytes for P4-only `record + 0x1C` have been tied to a static or rebuildable image location.

## Safety

- No patch bytes.
- No flash.
- No P0-P3 modification.
- No voltage writes.

# PHASE2_MASTER_A_DISPATCH_SOURCE

## Verdict

`DISPATCH_SOURCE_FOUND`

Route A advanced beyond the previous missing-artifact state. The `.dG3_DXE` consumer around `0xFFF8D0EC-0xFFF8D130` is now recovered as a heap-handle collector/copier at function `0xFFF72B3C`.

## New Artifact

- `50_2_firmware/cpu_hack/agesa_trace/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt`

## Evidence

| Item | Finding |
|---|---|
| Source PE32 body | SHA-256 `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A` |
| `.dG3_DXE` VA | `0xFFF8D0EC-0xFFF94790` |
| Heap-handle list | `0xFFF8D0EC-0xFFF8D104` |
| Heap consumer | `0xFFF72B3C-0xFFF72E22` |
| Table-base operand refs | `0xFFF72BE1`, `0xFFF72D85` |
| Function pointer of interest | `0xFFF8D11E -> 0xFFF7371A` |

## Answer To The Required Source Question

`arg_0C` to `0xFFF7371A` is not proven to point to a simple editable static P4 table row.

Best current classification:

`CPU-family/runtime dispatch record with heap-selected table context`

Reason:

- `0xFFF7371A` is a static firmware entry point in a nearby function-pointer array.
- The recovered `.dG3_DXE` consumer walks heap handles, not direct P-state records.
- The constructor function itself sets `selected_base = arg_0C + 8`, optionally changes it through helper `0xFFF4CF55`, and then consumes `selected_base + pstate * 0x18`.
- Local AGESA source drops identify the first `.dG3_DXE` dwords as heap handles such as `SOCKET_DIE_MAP_HANDLE`, `NODE_ID_MAP_HANDLE`, S3 save table handle, and related generation-dependent IDS/PP/GNB handles.
- No direct call xref reaches `0xFFF7371A`; it is selected through table/dispatcher mechanics.

## Route A Outcome

Route A is advanced, but it does not make the AGESA route byte-ready.

What is now known:

- Static entry source: found.
- Heap/table consumer: found.
- Direct static P4 record: not found.
- P4-only edit target: not proven.

## Next Action From Route A

Do not repeat blind table scans. If firmware route continues, chase the specific caller/dispatcher that invokes the function-pointer array containing `0xFFF7371A`, then prove the exact `arg_0C` allocation or structure instance at runtime.

Non-destructive next local command:

```powershell
rg -n "0xFFF72B3C|0xFFF8D0EC|0xFFF8D11E|0xFFF7371A|CPU-family/runtime dispatch record|heap-selected" cpu_hack cpu_sing_3
```

No patch bytes, no flash, and no voltage writes are authorized by this artifact.

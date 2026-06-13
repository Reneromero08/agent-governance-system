# PHASE2_P4_EDIT_SOURCE_PROOF

## Verdict

`P4_ONLY_EDIT_SOURCE_NOT_FOUND_CURRENT_HELPER_LAYER`

This pass chased the next unresolved service/function-table layer behind the already decoded `0xFFF7348D` / `0xFFF44E76` runtime-MSR source. It did not find an editable static P4-only byte source.

## Inputs

- PE32 body: `50_2_firmware/cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdProcessorInitPeim/1 PE32 image section/body.bin`
- PE32 SHA-256: `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A`
- Existing provenance trace: `50_2_firmware/cpu_hack/agesa_trace/AmdProcessorInitPeim_producer_service_provenance.txt`
- Low-level probe artifact: `50_2_firmware/cpu_hack/agesa_trace/AmdProcessorInitPeim_p4_edit_source_probe.txt`

## Findings

| Probe | Result |
|---|---|
| P4 MSR immediate `0xC0010068` | `5` hits, all already-known MSR loop/read paths; no static P4 data row |
| Handler source pointer hits `0xFFF696B2/0xFFF69727/0xFFF6979C/0xFFF69809` | `0` raw pointer hits; handlers are reached by direct calls from descriptor handlers, not by editable P4 records |
| Handler-source behavior | Reads/searches service or table structures, then calls through table pointers such as `[esi+0x1d]`; does not expose P0-P4 sibling data bytes |
| Constructor field source | Still maps to producer `entry+0x04` from `[service+0x22]` / `0xFFF7348D` / `0xFFF44E76` |

## Interpretation

The next helper layer did not turn the runtime P4 field into an editable firmware table. It strengthens the current classification: the constructor-relevant P4 byte is service/runtime-derived, with `MSRC001_0068` as the source for P4, not a static AGESA P4 row.

This does not prove a mathematical impossibility for every future firmware route, but it closes the current decoded helper layer as a byte-ready source. The route remains blocked on a different, not-yet-found P4-only edit source.

## Actionability

`BYTE_READY_HUMAN_REVIEW` is not met.

`P4_ONLY_EDIT_SOURCE_PROOF` remains unmet for current artifacts.

The next non-repeating firmware move would have to leave the decoded `0xFFF4CF9C -> 0xFFF7348D -> 0xFFF44E76` chain and search for a separate P4-affecting source, because this chain is now resolved to runtime MSR state.

## Safety

- No patch bytes.
- No image rebuild.
- No flash.
- No MSR writes.
- No P0-P3 or P4 modification.

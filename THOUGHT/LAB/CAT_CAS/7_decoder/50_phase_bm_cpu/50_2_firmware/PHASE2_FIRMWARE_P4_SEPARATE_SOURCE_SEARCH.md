# PHASE2_FIRMWARE_P4_SEPARATE_SOURCE_SEARCH

## Verdict

`FIRMWARE_P4_SEPARATE_SOURCE_CANDIDATE_FOUND`

Search for P-state MSR constants across the extracted firmware tree, looking for a P4-affecting source outside the already closed AmdProcessorInitPeim runtime-MSR helper chain.

## Binary Hits

| Module | File | Hits | Values | SHA-256 |
|---|---|---:|---|---|
| `0 PE32 image section` | `50_2_firmware/cpu_hack/bios_dump.bin.dump/3 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdAgesaDxeDriver/1 Compressed section/0 PE32 image section/body.bin` | 10 | 0xC0010064:8, 0xC0010065:2 | `359FAF87428EA2DF7548CD8335FA09F78B0E1203E1CFEAF87E16ABEA46D46622` |
| `0 PE32 image section` | `50_2_firmware/cpu_hack/bios_dump.bin.dump/3 8C8CE578-8A3D-4F1C-9935-896185C32DD3/15 CpuDxe/1 Compressed section/0 PE32 image section/body.bin` | 5 | 0xC0010064:1, 0xC0010065:1, 0xC0010066:1, 0xC0010067:1, 0xC0010068:1 | `2EB4D8C6D7FCB131BBEAE84C8DEBC4AB6EAADD4B3D1DA1B9EBEF3F300D0E25A3` |
| `0 PE32 image section` | `50_2_firmware/cpu_hack/bios_dump.bin.dump/3 8C8CE578-8A3D-4F1C-9935-896185C32DD3/52 LegacyRegion/1 Compressed section/0 PE32 image section/body.bin` | 5 | 0xC0010064:1, 0xC0010065:1, 0xC0010066:1, 0xC0010067:1, 0xC0010068:1 | `D090A915A47FDEFB4D42EDA5D58BC6C088F68B69BB5FA3BFD3FC5BCB04501FE9` |
| `0 PE32 image section` | `50_2_firmware/cpu_hack/bios_dump.bin.dump/3 8C8CE578-8A3D-4F1C-9935-896185C32DD3/6 CORE_DXE/0 Compressed section/0 PE32 image section/body.bin` | 4 | 0xC0010064:4 | `8D2D56D7B2426D53A897CA96F36BEF0F417B27EA0166F9A692D0D796226E0290` |
| `0 PE32 image section` | `50_2_firmware/cpu_hack/bios_dump.bin.dump/3 8C8CE578-8A3D-4F1C-9935-896185C32DD3/68 GenericComponentsSmm/1 Compressed section/0 PE32 image section/body.bin` | 4 | 0xC0010064:4 | `6A7195E17ED12B82CDD66627D449FAFD660D054A79E11D32B5A79217A5482655` |
| `1 PE32 image section` | `50_2_firmware/cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdProcessorInitPeim/1 PE32 image section/body.bin` | 37 | 0xC0010064:23, 0xC0010065:9, 0xC0010068:5 | `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A` |
| `0 PE32 image section` | `50_2_firmware/cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/2 CORE_PEI/0 PE32 image section/body.bin` | 4 | 0xC0010064:4 | `3AC711DA53B786D715F6A7E5F7593B63977E9BCCFCAE033F6B8120A1132BB1AA` |
| `1 PE32 image section` | `50_2_firmware/cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/4 CpuPei/1 PE32 image section/body.bin` | 5 | 0xC0010064:1, 0xC0010065:1, 0xC0010066:1, 0xC0010067:1, 0xC0010068:1 | `F96DBA3F6C074267E456BF14043B8122E6F9B2BC9CBFDEBB2DD4ED168A461B9C` |

## Text Trace Hits

| File | Hit count |
|---|---:|
| `50_2_firmware/cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt` | 4 |
| `50_2_firmware/cpu_hack/agesa_trace/AmdProcessorInitPeim_msr_source_proof.txt` | 8 |
| `50_2_firmware/cpu_hack/agesa_trace/AmdProcessorInitPeim_p4_edit_source_probe.txt` | 10 |

## Interpretation

At least one binary hit exists outside the closed AmdProcessorInitPeim chain. The next step is module-level disassembly and P0-P4 sibling proof before any candidate construction.

## Boundary

- Search only; no image modification.
- No candidate construction.
- No platform setting changes.

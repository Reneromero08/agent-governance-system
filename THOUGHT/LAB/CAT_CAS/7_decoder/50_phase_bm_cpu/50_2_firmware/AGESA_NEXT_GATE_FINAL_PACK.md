# AGESA Next Gate Final Pack

Status: `NOOP_REBUILD_PROVEN`

Scope: owned local Phenom II X6 1090T / GA-970A-DS3P firmware route research. No flash command. No hardware-changing command.

## Verdict

The AGESA firmware route is still alive, but not byte-ready.

This pass advanced the rebuild gate from missing artifact to proven:

- Public LongSoft `old_engine` source was acquired into ignored local tool tree `50_2_firmware/cpu_hack/tools/UEFITool_repo/`.
- A temporary force-save UEFIReplace variant was built with Qt/qmake on the Linux target.
- Only identical AmdProcessorInitPeim PE32 body replacement was performed.
- `50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin` was produced, parsed cleanly, and verified byte-identical to stock.

The current stop condition is:

`NOOP_REBUILD_PROVEN_BUT_P4_STATIC_SOURCE_MISSING`

## Gate Table

| Gate | Artifact | Result |
|---|---|---|
| A constructor decompile/xrefs | `50_2_firmware/AGESA_NEXT_A_CONSTRUCTOR_DECOMPILE.md` | Constructor path identified. |
| A export | `50_2_firmware/cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt` | Function entry, stack frame, base flow, and field provenance recovered. |
| B table/provenance | `50_2_firmware/AGESA_NEXT_B_TABLE_REOPEN.md`, `50_2_firmware/PHASE2_FW_ARG0C_PROVENANCE.md` | P4 field maps to runtime-produced record sourced from `MSRC001_0068`; no static editable P4 row proven. |
| C no-op rebuild | `50_2_firmware/AGESA_NEXT_C_NOOP_REBUILD.md`, `50_2_firmware/cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt` | `NOOP_REBUILD_PROVEN`. |
| D actionability | `50_2_firmware/AGESA_NEXT_D_ACTIONABILITY.md` | Route alive, not byte-ready. |

## New Artifacts Produced

- `50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin`
- `50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin.report.txt`
- `50_2_firmware/cpu_hack/noop_replace/rebuilt_AmdProcessorInitPeim_PE32_body.bin/body.bin`
- `50_2_firmware/cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt`
- `50_2_firmware/AGESA_NEXT_C_NOOP_REBUILD.md`
- `50_2_firmware/AGESA_NEXT_D_ACTIONABILITY.md`
- `50_2_firmware/AGESA_NEXT_GATE_FINAL_PACK.md`

Generated binary and extraction artifacts remain local-only and ignored; markdown/text reports are commit-safe.

## Deepest Progress

The no-op rebuild path is now proven:

```text
UEFIReplace bios_dump.bin DE3E049C-A218-4891-8658-5FC0FA84C788 10 body.bin -o bios_noop_rebuilt.bin
```

Verification:

```text
stock SHA-256   B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91
rebuilt SHA-256 B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91
fc /b           no differences encountered
UEFIExtract     report exit code 0
PE32 body SHA   BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A
```

The deepest firmware provenance remains:

```text
0xFFF7371A constructor
  selected_base = arg_0C + 8
  selected_base + pstate*0x18 + 0x1C feeds P-state MSR construction

0xFFF4CF9C producer
  maps constructor field to entry +0x04
  entry +0x04 is output arg_14 from [service+0x22] / 0xFFF7348D

0xFFF44E76 / [service+0x22]
  rdmsr path
  MSR address = 0xC0010064 + pstate
  P4 source = MSRC001_0068
```

## Exact Remaining Blocker

`AGESA_P4_SAFE_ROUTE_NOT_BYTE_READY`

Missing proof:

- editable P4-only source or edit target,
- P0-P3 unchanged proof,
- P4-only effect proof,
- offsets/bytes/checksum proof,
- clean parse proof after a non-no-op candidate.

No candidate exists at this checkpoint.

## Exact Next Command / Tool / File

Do not repeat the no-op rebuild gate. It is proven.

Next firmware-only command should target edit-source discovery, not rebuild mechanics:

```powershell
rg -n "C0010068|C0010064|FFF44E76|FFF7348D|entry \\+0x04|selected_base \\+ pstate\\*0x18 \\+ 0x1C" 50_2_firmware/cpu_hack/agesa_trace
```

If new disassembly/decompile output is added, the next required file is a focused report proving whether `MSRC001_0068` can be influenced by a static byte source without touching P0-P3.

## AGESA Route Still Alive?

Yes.

The route is alive because no-op rebuild/save is now proven and the constructor/provenance chain is decoded. It is not byte-ready because the P4 source currently resolves to runtime MSR state, not an editable static P4-only firmware byte.

## Do-Not-Do List

- Do not produce patch bytes from this checkpoint.
- Do not repeat the rejected global branch edit.
- Do not flash.
- Do not issue a flash command.
- Do not write voltage.
- Do not modify hardware.
- Do not claim P4-safe status without P0-P3 equivalence proof.
- Do not count UEFIExtract as a rebuild tool.
- Do not treat the no-op rebuilt image as a flash candidate.
- Do not write outputs outside `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu`.

# CPU Hack Index

This folder is the Exp44 firmware and board-inspection workspace. It keeps authored evidence tracked and generated or binary material local-only.

| Path | Role | Keep in git | Notes |
|---|---|---|---|
| `agesa_trace/` | AGESA reverse-engineering text evidence | Yes | Decompile, disassembly, xref, service descriptor, record-map, MSR-source, and patch-analysis reports. |
| `bios_parse/` | BIOS parse reports | Mixed | Track small text parse reports; keep GUID CSVs local-only. |
| `board_probe/` | Local board helper scripts | Yes | Clock/board inspection PowerShell helpers. |
| `noop_replace/NOOP_DIFF_SUMMARY.txt` | No-op rebuild attempt summary | Yes | Authoritative no-op rebuild status. Generated `.bin` and parse reports stay local-only. |
| `bios_dump.bin` | Owned raw BIOS dump | No | Binary/sensitive local evidence; not committed. |
| `bios_dump.bin.dump/` | UEFIExtract parse tree | No | Generated extraction tree; not committed. |
| `mod_donors/` | Public stock/mod donor packages and extracted reports | No | Local donor workflow material; summarized in `cpu_sing_3/PHASE2_DONOR_DIFF_REPORT.md`. |
| `tools/` | Local UEFI/coreboot/rebuild tools | No | Tool binaries and downloaded source trees are local-only. |
| `_tmp_coreboot_*/` | Temporary source/extraction trees | No | Generated/heavy research trees. |
| `local_logs/` | Probe and extraction logs | No | Local run logs; not committed. |

The canonical firmware blocker remains `noop_replace/bios_noop_rebuilt.bin`: it must be produced by a force-save no-op rebuild path, parsed cleanly, and compared against the owned BIOS before any byte-ready review.

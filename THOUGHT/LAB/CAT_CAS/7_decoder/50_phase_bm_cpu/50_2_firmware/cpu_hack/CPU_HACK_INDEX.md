# CPU Hack Index

This folder is the Exp50 firmware and board-inspection workspace. It keeps authored evidence tracked and generated or binary material local-only.

| Path | Role | Keep in git | Notes |
|---|---|---|---|
| `agesa_trace/` | AGESA reverse-engineering text evidence | Yes | Decompile, disassembly, xref, service descriptor, record-map, MSR-source, and patch-analysis reports. |
| `bios_parse/` | BIOS parse reports | Mixed | Track small text parse reports; keep GUID CSVs local-only. |
| `board_probe/` | Local board helper scripts | Yes | Clock/board inspection PowerShell helpers. |
| `noop_replace/NOOP_DIFF_SUMMARY.txt` | No-op rebuild proof summary | Yes | Authoritative no-op rebuild status. Generated `.bin` artifacts stay local-only. |
| `noop_replace/bios_noop_rebuilt.bin.report.txt` | Accepted no-op rebuild parse report | Yes | Small text proof for parse-clean rebuilt image. |
| `bios_dump.bin` | Owned raw BIOS dump | No | Binary/sensitive local evidence; not committed. |
| `bios_dump.bin.dump/` | UEFIExtract parse tree | No | Generated extraction tree; not committed. |
| `mod_donors/` | Public stock/mod donor packages and extracted reports | No | Local donor workflow material; summarized in `50_2_firmware/PHASE2_DONOR_DIFF_REPORT.md`. |
| `tools/` | Local UEFI/coreboot/rebuild tools | No | Tool binaries and downloaded source trees are local-only. |
| `_tmp_coreboot_*/` | Temporary source/extraction trees | No | Generated/heavy research trees. |
| `local_logs/` | Probe and extraction logs | No | Local run logs; not committed. |

The no-op rebuild blocker is closed: `noop_replace/bios_noop_rebuilt.bin` was produced by a force-save no-op rebuild path, parsed cleanly, and compared byte-identical against the owned BIOS. The canonical firmware blocker is now P4-only edit-source proof: no byte-ready candidate exists until an editable P4-only source or target is proven with P0-P3 unchanged.

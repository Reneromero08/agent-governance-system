# AGESA Next D Actionability

Status: `NOOP_REBUILD_PROVEN`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## Choice

`NOOP_REBUILD_PROVEN`

## Why Not The Other Choices

| Choice | Status |
|---|---|
| `TABLE_TARGET_FOUND` | Not met. P4 record and P0-P3 siblings are runtime-derived from `MSRC001_0068`, not proven as editable static records. |
| `BOTH_LIVE_GATES_ADVANCED` | True as progress, but the actionability verdict is narrower: the rebuild/no-op gate is now proven while the table/edit-source gate remains blocked. |
| `MISSING_ARTIFACT_BLOCKER` | Not met for the no-op rebuild gate; `cpu_hack/noop_replace/bios_noop_rebuilt.bin` now exists and parses. |
| `HARD_IMPOSSIBILITY_PROOF` | Not met. Firmware route remains alive for future edit-source discovery, but not byte-ready. |

## Current Actionability

The AGESA route is still alive but not byte-ready.

Gate C is now proven: the lab produced a parse-clean identical no-op rebuild and verified the target PE32 body hash stayed `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A`.

The remaining blocker is not the rebuild toolchain. It is the absence of a defensible P4-only editable source or patch target. Current provenance maps the constructor P4 field to runtime `MSRC001_0068`.

## Exact Remaining Blocker

`AGESA_P4_SAFE_ROUTE_NOT_BYTE_READY`

Required missing proof:

- editable P4-only source or edit target,
- P0-P3 unchanged proof,
- P4-only effect proof,
- offsets/bytes/checksum proof,
- clean parse proof after a non-no-op candidate.

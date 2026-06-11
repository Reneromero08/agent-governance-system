# PHASE2_FIRMWARE_PSTATE_VALUE_PATTERN_SEARCH

## Verdict

`FIRMWARE_PSTATE_VALUE_PATTERN_NOT_FOUND_CURRENT_DUMP`

The firmware search found P-state MSR address tables, but a direct value-pattern
search did not find the stock P4 value or its key fragments in the extracted
image tree.

## Patterns Searched

| Pattern | Bytes | Result |
|---|---|---:|
| stock P4 full `0x8000013540003440` | `40 34 00 40 35 01 00 80` | 0 hits |
| stock P4 low dword `0x40003440` | `40 34 00 40` | 0 hits |
| stock P4 high dword `0x80000135` | `35 01 00 80` | 0 hits |
| lower-VID runtime-test low dword `0x400040c0` | `c0 40 00 40` | 0 hits |

## Interpretation

No static P4 value payload was found in the current dump. Combined with the
constructor provenance chain, the current evidence supports:

```text
P-state MSR addresses are static in multiple modules.
P-state values are runtime-built/read, not present as stock static P4 rows.
```

This does not prove no possible firmware route exists, but it closes the direct
value-pattern search over the current extracted image.

## Route Impact

```text
FIRMWARE_PSTATE_VALUE_PATTERN_NOT_FOUND_CURRENT_DUMP
```

This is not:

- `BYTE_READY_HUMAN_REVIEW`
- `CPU_SINGS`
- `SOFTWARE_FIRMWARE_TRUE_WALL`

## Next Exact Action

`FIRMWARE_SOURCE_PROVENANCE_WALL_AUDIT`

Before declaring a firmware wall, one final audit should list every firmware
route tried, its artifact, and the exact missing proof. If any untried route
remains that can be pursued inside current constraints, pursue it. Otherwise the
firmware side may be approaching `HUMAN_TOOL_REQUIRED_WITH_ALL_OTHER_ROUTES_EXHAUSTED`.

## Boundary

- Search only; no image modification.
- No candidate construction.
- No platform setting changes.

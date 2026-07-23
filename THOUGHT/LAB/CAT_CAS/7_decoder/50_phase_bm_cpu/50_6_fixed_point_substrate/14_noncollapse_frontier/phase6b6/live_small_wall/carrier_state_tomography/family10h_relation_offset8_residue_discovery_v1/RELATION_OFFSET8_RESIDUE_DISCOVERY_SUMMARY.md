# Relation Offset-8 Residue Discovery

Run ID: `family10h_relation_offset8_residue_discovery_v1_0`
Archive SHA-256: `add8385f03c8dad6bb7bf6983cdcaa0aeb4baf5019414b7865a6941f7113b522`
Analysis SHA-256: `d4e019540073fe0a059ccf698c3b507e86cfd341fc23d6f6258c079c159a5410`

| Round | Offset | Source-on R | Source-off R | On-off delta | abs(delta)/abs(on) | abs(off)/abs(on) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | -0.003553611 | 0.000280458 | -0.003834070 | 1.079 | 0.079 |
| 0 | 8 | -0.003924838 | -0.013329842 | 0.009405005 | 2.396 | 3.396 |
| 0 | 16 | 0.053094692 | -0.000853607 | 0.053948298 | 1.016 | 0.016 |
| 1 | 4 | -0.003204815 | -0.002761010 | -0.000443806 | 0.138 | 0.862 |
| 1 | 8 | 0.004338088 | 0.028364719 | -0.024026631 | 5.539 | 6.539 |
| 1 | 16 | 0.001062918 | 0.036408731 | -0.035345813 | 33.254 | 34.254 |

Discovery interpretation:
- offset 8 survives both rounds: `False`
- offset 8 exceeds neighbor offsets 4 and 16: `False`
- offset 8 all deltas positive: `False`
- offset 8 all deltas above 0.25 x source-on: `True`
- strongest delta round/offset: `0` / `16`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.

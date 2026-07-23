# Relation Receiver Offset Sweep Discovery

Run ID: `family10h_relation_receiver_offset_sweep_discovery_v1_0`
Archive SHA-256: `177df6c6acf5cda1984a7b0d03f6870109dfad098fa01d887d3d25bb9c52f423`
Analysis SHA-256: `e769868fda9e0b3b406f3d2973798cfa74e77d23e1e0d4228614c034f6487ada`

| Offset | Source-on R | Source-off R | On-off delta | abs(delta)/abs(on) | abs(off)/abs(on) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.075157130 | 0.070080704 | 0.005076426 | 0.068 | 0.932 |
| 2 | 0.125205552 | 0.130068090 | -0.004862538 | 0.039 | 1.039 |
| 4 | 0.003419716 | -0.006783567 | 0.010203283 | 2.984 | 1.984 |
| 8 | 0.008924377 | -0.007107406 | 0.016031784 | 1.796 | 0.796 |
| 16 | 0.012990078 | 0.012475020 | 0.000515057 | 0.040 | 0.960 |
| 1024 | -0.023479903 | -0.013504327 | -0.009975577 | 0.425 | 0.575 |

Discovery interpretation:
- offset 1 source-dependent delta positive: `True`
- offset 1 source-dependent delta above 0.25 x source-on: `False`
- offset 1 source-off baseline below 0.25 x source-on: `False`
- any offset source-dependent delta above 0.25 x source-on: `True`
- strongest delta offset: `8`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.

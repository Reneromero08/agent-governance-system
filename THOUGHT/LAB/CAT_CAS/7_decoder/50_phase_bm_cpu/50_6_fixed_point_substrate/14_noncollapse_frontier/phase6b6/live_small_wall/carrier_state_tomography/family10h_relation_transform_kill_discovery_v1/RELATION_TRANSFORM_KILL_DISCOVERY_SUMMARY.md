# Relation Transform/Kill Discovery

Run ID: `family10h_relation_transform_kill_discovery_v1_0`
Archive SHA-256: `ea0c98f1928a2453a2c915d481c8fcb6ccb844179b342f20b5d22bb46df04e55`
Analysis SHA-256: `b574cf415348e856aaf0a7455696125a6bcf3006eb5600ac3657b7b43e4d2d4d`

| Variant | R_spatial mean | abs/primary | Blocks |
|---|---:|---:|---:|
| `primary_relation_pair` | 0.080194543 | 1.000 | 128 |
| `query_inversion_control` | -0.075002112 | 0.935 | 128 |
| `carrier_off_control` | 0.066224535 | 0.826 | 128 |
| `relation_sham` | 0.005388051 | 0.067 | 128 |

Discovery interpretation:
- query inversion reverses primary: `True`
- carrier-off below 0.25 x primary: `False`
- relation-sham below 0.25 x primary: `True`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.

# Relation Orientation Balance Discovery

Run ID: `family10h_relation_orientation_balance_discovery_v1_0`
Archive SHA-256: `529f3ef9dad18f58d4626992d04cdaf7a5d212d0f0271aff8f94b41c2152c151`
Analysis SHA-256: `adf1cce2271d02ebddbc432e97b1e902710a55138fe232452e8bce744228e448`

| Variant | R_spatial mean | abs/primary | Blocks |
|---|---:|---:|---:|
| `primary_relation_pair` | 0.091472019 | 1.000 | 128 |
| `query_inversion_control` | -0.079832056 | 0.873 | 128 |
| `carrier_off_control` | 0.065250915 | 0.713 | 128 |
| `carrier_off_inversion_control` | -0.074062625 | 0.810 | 128 |
| `relation_sham` | 0.005928039 | 0.065 | 128 |

Discovery interpretation:
- query inversion reverses primary: `True`
- carrier-off absolute below 0.25 x primary: `False`
- carrier-off orientation below 0.25 x relation orientation: `False`
- net relation-minus-carrier-off orientation remains positive: `True`
- relation-sham below 0.25 x primary: `True`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.

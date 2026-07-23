# Relation Prep-Ablation Discovery

Run ID: `family10h_relation_prep_ablation_discovery_v1_0`
Archive SHA-256: `ee356cb7e86a339a5c937e1d1b8d2bf58367e19f3473932b4b5441a9a19fa36f`
Analysis SHA-256: `ca6f6bd2ff1f91e1f50e8a96fca6ddfaca052102419a22b5a008ae98af3599cb`

| Variant | R_spatial mean | abs/primary | Blocks |
|---|---:|---:|---:|
| `primary_relation_pair` | 0.060671889 | 1.000 | 128 |
| `prep_ablation_control` | 0.061000788 | 1.005 | 128 |
| `carrier_off_control` | 0.074276554 | 1.224 | 128 |
| `relation_sham` | 0.002708175 | 0.045 | 128 |

Discovery interpretation:
- prep ablation reduces primary: `False`
- prep ablation delta above 0.25 x primary: `False`
- prep ablation approaches sham within 0.25 x D: `False`
- carrier-off below 0.25 x primary: `False`
- relation-sham below 0.25 x primary: `True`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.

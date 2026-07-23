# Relation R-Spatial Lifetime Reset Discovery

Run ID: `family10h_relation_rspatial_lifetime_reset_discovery_v1_0`
Archive SHA-256: `752ed25289fc9e59dc0bcd9eb1bc5028dc2a2fe2beb4503149ed7430d85d3f9b`
Analysis SHA-256: `dc6329a237b60607724225f4c0e3e87075dcb6670a2274563646166449c9aa56`

| Lifecycle | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | abs(sham)/abs(primary) |
|---:|---:|---:|---:|---:|---:|
| alive | 0.071855718 | -0.000616304 | 0.072472023 | 1.000 | 0.009 |
| dead | 0.028481942 | 0.001229872 | 0.027252071 | 0.376 | 0.043 |
| reset | 0.007192910 | -0.004791799 | 0.011984709 | 0.165 | 0.666 |

Discovery interpretation:
- alive D positive: `True`
- dead D positive: `True`
- reset D positive: `True`
- source death kills below 0.25 x alive D: `False`
- reset kills below 0.25 x alive D: `True`
- reset reduces dead D abs: `True`
- persistent source-written-state candidate: `True`
- receiver/static-geometry candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.

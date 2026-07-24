# Relation Loop Transport Source-Off Screen Discovery

Run ID: `family10h_relation_loop_transport_source_off_screen_discovery_v1_0`
Archive SHA-256: `018b61c8a9b19063fcba17ef33ceb33e2b61deaeb015aa58e7ee03b9d278a6eb`
Analysis SHA-256: `8dc8454408e98ac4b7bd123d9ab348885cd037b8167d137917ac2d5445316331`

| Scope | D loop R0->R1 | D loop R1->R0 | Omega | abs(Omega)/abs(alive Omega) | max stratum ratio |
|---|---:|---:|---:|---:|---:|
| alive | 0.092905471 | 0.096006101 | -0.003100629 | 1.000 | 1.000 |
| dead | 0.083050123 | 0.072241527 | 0.010808596 | 3.486 | 3.792 |
| source_off | 0.119632498 | 0.112031733 | 0.007600765 | 2.451 | 11.545 |
| gapped | 0.076365739 | 0.097681842 | -0.021316103 | 6.875 | 11.114 |

Discovery interpretation:
- source-off collapses below 0.25 x alive Omega: `False`
- gapped collapses below 0.25 x alive Omega: `False`
- dead preserves at least 0.25 x alive Omega: `True`
- loop-transport source-off screen candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.

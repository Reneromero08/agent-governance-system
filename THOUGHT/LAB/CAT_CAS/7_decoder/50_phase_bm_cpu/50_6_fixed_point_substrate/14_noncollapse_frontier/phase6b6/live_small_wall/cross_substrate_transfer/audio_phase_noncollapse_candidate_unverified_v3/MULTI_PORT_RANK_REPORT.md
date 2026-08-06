# Multi-Port Rank Report

Candidate K: full-rank multi-port tensor-train diagnostic.

Strict bounded diagnostic survived: `True`
Classification: `SOURCE_REPRODUCED_SOURCE_LOCAL_MULTI_PORT_TT_OBSTRUCTION`

Independent reconstruction:

- p=2: ranks=[4, 2], dense=1140, TT=1160, TT>d=True
- p=3: ranks=[8, 4, 2], dense=2280, TT=2364, TT>d=True
- p=4: ranks=[16, 8, 4, 2], dense=4560, TT=4900, TT>d=True
- p=5: ranks=[32, 16, 8, 4, 2], dense=9120, TT=10484, TT>d=True
- p=6: ranks=[64, 32, 16, 8, 4, 2], dense=18240, TT=23700, TT>d=True

Finding:

The source/oracle rank lists and the public-shape storage arithmetic are consistent, and the resulting TT cell counts exceed the matched dense assignment storage in every tested case. This is useful as a warning against assuming TT compaction.

Scope discipline:

This remains source/family local: V3 did not independently reconstruct the final relation tensor, recompute high-precision singular values, run exact/modular rank, sweep tolerance, test alternate matricizations/orderings, or search symmetry quotients. The dense compact baseline dominates the accepted TT representation for the tested cases.

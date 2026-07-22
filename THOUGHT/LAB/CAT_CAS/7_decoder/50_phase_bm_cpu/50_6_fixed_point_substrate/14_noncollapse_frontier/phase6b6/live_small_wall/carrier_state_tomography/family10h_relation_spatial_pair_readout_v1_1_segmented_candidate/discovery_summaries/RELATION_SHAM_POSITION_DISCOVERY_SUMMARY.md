# Relation-Sham Position Exploration

Archive SHA-256: `4b16461c7c8184d408a70e5cc5f59ab945276e0730d547eb995f4d42d6b4eac8`
Archive size: `24859509` bytes
Copy-back verified: `True`

## Result

Diagnostic disposition: `RELATION_SHAM_NEGATIVE_RESPONSE_REPRODUCES_WITHOUT_PRIMARY_POSITIONING`

| Variant | Query | Blocks | mean R_spatial | abs(mean) | q99 null | null exceeded |
|---|---:|---:|---:|---:|---:|---|
| `sham_alone` | `relation_sham` | 128 | -0.020537109 | 0.020537109 | 0.015771888 | `True` |
| `primary_alone` | `query_relation_pair` | 128 | 0.080530840 | 0.080530840 | 0.012828871 | `True` |
| `sham_then_primary` | `relation_sham` | 128 | -0.019736197 | 0.019736197 | 0.013582582 | `True` |
| `sham_then_primary` | `query_relation_pair` | 128 | 0.110326929 | 0.110326929 | 0.013106755 | `True` |
| `primary_then_sham_original_offset` | `relation_sham` | 128 | -0.025925814 | 0.025925814 | 0.012384989 | `True` |
| `primary_then_sham_original_offset` | `query_relation_pair` | 512 | 0.077242133 | 0.077242133 | 0.007019274 | `True` |

## Interpretation

The relation-sham response stayed negative when run alone in a fresh process and when run first before primary. It also stayed negative when placed after a full primary prefix at the original sham ordinal offset. That points away from simple segment-position-after-primary as the sole cause.

This is exploratory physical evidence only. It does not change the frozen prospective result class, claim boundary, or Small Wall state.

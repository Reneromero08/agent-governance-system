# Postrun Relation-Only Analysis V1

Status: `POST_HOC_NON_ADJUDICATIVE_ANALYSIS`

This analysis does not modify, reinterpret, or replace the sealed adjudication. The frozen result remains:

`FAMILY10H_RELATION_MATCH_COORDINATE_NOT_CONFIRMED_PROSPECTIVE`

The sealed scientific claim remains:

`PUBLIC_POST_SOURCE_RELATION_MATCH_COORDINATE_NOT_ESTABLISHED`

Small Wall status: `not_crossed`.

## Evidence Analyzed

- Evidence commit: `aaec66edfe536995a2d56498d72b6219a6084466`
- Archive SHA-256: `bea3f1ffcc8d15d4cc9cc8299a5dec1b6345727208d8aec0d49f52e1e85a4d96`
- Archive size: `6436509` bytes
- Raw records: `32256`
- Source-death receipts: `32256`
- Freeze commit: `845b3f0e65d688cbd48cbaa0c16dba5fd13559b4`
- Source-authority commit: `7f30f1a15f2ebcea27761fd9326934d84653ebbf`

## Primary Finding

The physical transaction worked, but the relation interaction did not survive as a stable dirty-probe coordinate.

The frozen primary `dirty_probe_response` law produced:

- `R_match_mean`: `-2.5239955357`
- `R_match_abs_mean`: `52.6463913690`
- `R_match_q95_abs`: `128.0`
- `R_match_max_abs`: `266.0`
- frozen floor: `512.0`
- signs: `1390` negative, `1288` positive, `10` zero

The relation matrix raw dirty-probe mean was `3645.695` counts. The signed relation interaction was only `0.0692%` of that baseline. The absolute block interaction was only `1.444%` of the baseline.

## Why The Average Went Near Zero

The 2x2 relation cells carried ordinary marginal structure, not a coherent match-versus-mismatch effect:

| Cell | Mean Dirty-Probe Count |
| --- | ---: |
| `r0 -> r0` | `3636.186` |
| `r0 -> r1` | `3658.727` |
| `r1 -> r0` | `3635.188` |
| `r1 -> r1` | `3652.680` |

The query marginal `r1 - r0` was about `+20.016` dirty-probe counts, while the diagonal-minus-offdiagonal relation interaction was only `-2.524`. That is the clearest failure shape: `r0/r1` produced some physical route/query difference, but not the intended relational geometry.

## Common-Mode And Endpoint Diagnosis

The primary PMU endpoint is dominated by baseline dirty-probe traffic.

| Coordinate | R Mean | R Abs Mean | R q95 Abs | Raw Mean | Signed/Raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dirty_probe_response` | `-2.523996` | `52.646` | `128.000` | `3645.695` | `0.0692%` |
| `dirty_probe_per_cycle_x1e6` | `-7.292752` | `146.035` | `356.102` | `10409.562` | `0.0701%` |
| `dirty_probe_per_duration_ns_x1e6` | `-22.599243` | `506.864` | `1264.639` | `32574.483` | `0.0694%` |
| `cpu_cycles` | `87.155134` | `4247.853` | `10119.650` | `351694.346` | `0.0248%` |
| `duration_ns` | `25.112165` | `1577.677` | `3977.100` | `112372.867` | `0.0223%` |

`time_running` and `time_enabled` scaling did not change the primary dirty-probe values in this attempt; the PMU group ran for the enabled interval. The useful interpretation for the next design is therefore not absolute counts alone. It should be matched interaction contrasts, with counts-per-cycle and counts-per-duration reported as secondary coordinates.

## Subthreshold Structure

There is weak subthreshold structure, but no stable relation coordinate.

The strongest dirty-probe strata were:

| Factor | Level | R Mean | R Abs Mean | q95 Abs | Sign Balance |
| --- | --- | ---: | ---: | ---: | ---: |
| `cyclic_origin` | `0` | `-6.986607` | `54.182` | `131.950` | `0.0955` |
| `q` | `1536` | `-5.852865` | `50.264` | `122.750` | `0.0235` |
| `q` | `512` | `-5.587240` | `52.223` | `120.425` | `0.1047` |
| `q` | `1024` | `-5.459635` | `57.017` | `143.775` | `0.0885` |
| `mapping` | `map0` | `-4.782366` | `52.864` | `128.425` | `0.0597` |
| `source_order` | `B_then_A` | `-3.715030` | `52.934` | `131.425` | `0.0673` |
| `session` | `session_1` | `-3.710938` | `52.348` | `127.850` | `0.0463` |
| `replicate` | `0` | `-3.599330` | `52.305` | `126.775` | `0.0515` |


The largest normalized-duration strata reached larger diagnostic magnitudes, especially `q=1024`, but signs and levels were not coherent enough to justify a second relation claim. These are hypothesis generators only.

## Delay And Source-Death Interpretation

The dirty-probe relation interaction did not show clean monotone decay:

| Delay | R Mean | R Abs Mean | Sign Balance |
| --- | ---: | ---: | ---: |
| `0ns` | `-3.561384` | `52.270` | `0.0413` |
| `1ms` | `-1.503348` | `51.154` | `0.0448` |
| `10ms` | `-2.507254` | `54.515` | `0.0281` |


The absence of monotone delay decay does not clear source death. Every query in the sealed attempt happened after waitpid/source death, so a relation that only exists while the source helper is alive would be erased before all three delay levels. The current evidence cannot distinguish `no retained relation state` from `state destroyed by source termination`.

## Why The Control Gates Failed

The relation-control rows measured large ordinary PMU traffic, not near-zero interaction residuals:

| Control Query | Mean Dirty-Probe Count | q95 Abs |
| --- | ---: | ---: |
| `distance_control` | `3596.339` | `3815.000` |
| `independent_marginal_replay` | `3692.801` | `3921.000` |
| `relation_sham` | `3140.576` | `3765.000` |
| `route_pressure_sham` | `3686.953` | `3917.000` |


The scalar-equivalence gate failed because the scalar q slopes were similar but the relation labels shifted the scalar baseline too much:

- `relation_r0` slope: `1.880932`, intercept `-19.014`
- `relation_r1` slope: `1.749195`, intercept `173.498`
- slope relative disagreement: `0.070038`
- max `D_single` drift: `504.448`

So scalar q readout behavior remained visible, but the relation preparation was not scalar-equivalent under the frozen law.

## Mechanism Classification

1. `common-mode PMU domination`: strongest. Dirty-probe and relation controls were thousands of counts while signed relation interaction was near zero.
2. `incorrect readout endpoint`: strong. Secondary cycle/rate coordinates show larger diagnostic spread, but no stable signed relation coordinate under the frozen law.
3. `insufficient physical distinction between relation geometries`: strong. `r0/r1` appears to perturb query-route marginal behavior more than match/mismatch relation geometry.
4. `source-death destruction or no retained relation state`: plausible and unresolved. All observations were source-dead-before-query, so the experiment did not test an alive retained relation.
5. `state decay before query`: weak on current evidence. Delay strata do not show monotone decay, but all delays occur after source death.

## Practical Conclusion

The sealed attempt is a valid negative result for the frozen relation-only law. It should be retained as the dead-source baseline. The next experiment should not repeat this schedule unchanged. The highest-value next move is to add a matched alive-source relation-query twin so the design can separate source-death erasure from endpoint/common-mode failure.

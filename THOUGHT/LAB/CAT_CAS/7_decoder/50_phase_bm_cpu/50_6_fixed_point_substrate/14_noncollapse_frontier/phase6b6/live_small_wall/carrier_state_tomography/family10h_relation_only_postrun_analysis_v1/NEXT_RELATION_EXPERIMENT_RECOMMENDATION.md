# Next Relation Experiment Recommendation

Status: `PROSPECTIVE_RECOMMENDATION_REQUIRES_USER_AUTHORIZATION`

Recommended next experiment:

`same relation grammar with an alive-source same-window relation query and source-dead twin, measuring matched interaction contrasts in dirty_probe_response normalized by cycles and duration`

## Why This Is The Next Step

The sealed attempt did not fail because the machine was inactive. It failed because the relation interaction was tiny relative to common-mode dirty-probe traffic and because every query happened after source death. The smallest scientifically meaningful change is therefore a lifetime contrast, not a repeat of the same dead-source schedule.

Use the existing sealed attempt as the source-dead baseline. Add a prospective alive-source twin in which the source helper remains alive, pinned, and fixed in its predeclared relation preparation while the receiver performs the same relation query under PMU measurement. The source must receive no query choice, no post-observation feedback, and no new IPC; the only changed factor is source lifetime during the receiver query.

## Observable

`Delta_alive_minus_dead of the per-block relation interaction R_match, computed on dirty_probe_response, dirty_probe_per_cycle, and dirty_probe_per_duration with matched scalar marginals and matched total work`

Primary prospective statistic:

`Delta_R_lifetime = R_match_alive_source - R_match_source_dead_twin`

Report for:

- `dirty_probe_response`
- `dirty_probe_response / cpu_cycles`
- `dirty_probe_response / duration_ns`
- `cpu_cycles`
- `duration_ns`

Use matched within-block contrasts and preserve the same q, mapping, delay, source-order, query-order, session, and replicate factors.

## Expected Positive Signature

`alive-source R_match has stable sign and exceeds the dead-source twin and matched permutation null across q, mapping, delay/order, session, and replicate without target-derived thresholds`

A useful positive should show that alive-source relation matching creates a coherent interaction that disappears or collapses in the source-dead twin. It must survive scalar q, query label, order, mapping, delay, cycle, duration, and common-mode control replay.

## Falsifying Result

`alive-source and dead-source twins both remain near zero or are fully explained by scalar/query/order marginals and matched controls`

That outcome would argue that the current Family 10h dirty-probe/PMU route is not carrying an independently readable relation coordinate, at least under this relation grammar.

## Controls

- Source-dead twin: same as the sealed baseline semantics.
- Scalar q controls: preserve `query_A` and `query_B` matched marginals.
- Route sham: preserve ordinary route-pressure without relation-match semantics.
- Distance control: preserve the matched distance histogram.
- Label scramble and matched permutation nulls: keep prospective, fixed, and non-adaptive.

## Fallback If Blocked

`if alive-source concurrent query is technically unsafe or impossible, use immediate pre-death/kill-at-query interleaving that preserves source state until receiver PMU enablement, with the existing dead-after-waitpid attempt as baseline`

Do not execute either experiment without a new explicit physical authorization.

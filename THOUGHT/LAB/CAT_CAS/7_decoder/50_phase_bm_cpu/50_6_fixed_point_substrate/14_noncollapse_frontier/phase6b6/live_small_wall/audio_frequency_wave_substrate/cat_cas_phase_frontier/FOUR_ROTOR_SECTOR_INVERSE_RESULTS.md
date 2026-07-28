# Four-Rotor Sector Inverse Results

Evidence:

```text
/tmp/four-rotor-sector-inverse-final.drDP8n
```

Bounded tradeoff claim:

```text
BOUNDED_TOPOLOGY_DERIVED_TOTAL_MOMENTUM_SECTOR_INVERSE_PHASE_CLOSURE_UPDATE_REDUCTION_WITH_GRAM_REMATERIALIZATION_OBSTRUCTION_ACTUAL_RESTORATION_AND_REUSE
```

The first phase-owned repair replaces occurrence-by-occurrence inverse
Bessel updates with public total-pair-momentum sector solves. For each
truncated finite coupling gate it compiles the exact LU inverse of the
implemented sector kernel. It then streams sector right-hand sides and exact
Grams without probe columns or retained inverse history.

At mode radius 14 and primary depth three:

```text
forward incremental updates                         117
incremental reference inverse updates               117
sector inverse closures                               9
total forward plus inverse closures                  126
public plan complex cells                         16,269
public pivot cells                                   841
maximum sector condition                           1.015
maximum inverse residual                       5.676e-16
primary inverse restoration error              6.423e-8
postclosure restoration error                  6.424e-8
sector RHS rematerializations                      73,167
Gram rematerializations                                18
sector wrapper peak payload                    21,973,888
incremental wrapper peak payload               10,834,016
```

The same actually restored carrier executes the unrelated depth-two reuse
program at restoration generation two. Fresh/restored boundary disagreement
is `7.381e-8`, and the declared fresh/restored resource signatures match
exactly. Missing, wrong, and reordered inverse controls separate.

The closure-count reduction is real, but it is not a memory or warm-time
improvement. Exact Gram construction and 73,167 sector RHS
rematerializations move the cost elsewhere and raise the accepted peak by
`2.028x`. The result therefore closes this repair route as a measured
tradeoff and identifies:

```text
EXACT_GRAM_AND_SECTOR_RHS_REMATERIALIZATION_COST
```

The matched classical sector algorithm is identical. This does not establish
a distinct phase resource, computational advantage, fixed-rank forward
closure, Small Wall crossing, unbounded computation, or physical waveform
execution. The next phase-machine experiment must change the update law
rather than add more sectors or larger fixtures.


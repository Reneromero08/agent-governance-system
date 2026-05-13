# QEC Precision Sweep

## Purpose

This is the first recommended v4 empirical test because the domain mapping is
clean and the failure modes are hard to hide.

The target is not to show that the formula vaguely resembles QEC. The target is
to test whether the functional form predicts logical survival or corrected
fidelity across controlled variations.

## Hypothesis

For a locked QEC mapping:

```text
R = (E / grad_S) * sigma^Df
```

predicts logical survival/corrected fidelity better than simpler baselines.

## Initial Mapping

| Symbol | Initial observable |
|---|---|
| `E` | initial logical-state fidelity, fixed to 1.0 in basic sweeps |
| `grad_S` | physical error probability `p` or syndrome entropy |
| `sigma` | code compression/fidelity factor, locked per code family |
| `Df` | code distance or redundancy depth |
| `R` | logical success probability / corrected fidelity |

This mapping is provisional until the first test plan locks it.

## Candidate Systems

1. classical repetition code analog
2. 3-qubit bit-flip code
3. 3-qubit phase-flip code
4. 5-qubit perfect code
5. surface-code toy simulator if feasible

## Baselines

- physical error rate `p`
- code distance alone
- redundancy count alone
- known analytic logical error rate
- fitted logistic/power curve with equal parameter count

## Success Criteria

The formula is useful in this domain if:

1. the mapping is locked before the sweep;
2. formula predictions correlate strongly with logical survival;
3. formula error is lower than simple baselines;
4. the same mapping family works across at least two QEC systems.

## Failure Criteria

The QEC mapping fails if:

1. simple known QEC formulas explain the result as well or better;
2. `sigma^Df` adds no predictive value;
3. mapping definitions need post-hoc changes;
4. the formula only works for one cherry-picked code.

## Old Questions Used

- Q40: quantum error correction
- Q44: Born rule, if phase mapping is needed
- Q25: sigma measurement
- Q7: multi-scale/redundancy
- Q2/Q20: falsification and tautology controls

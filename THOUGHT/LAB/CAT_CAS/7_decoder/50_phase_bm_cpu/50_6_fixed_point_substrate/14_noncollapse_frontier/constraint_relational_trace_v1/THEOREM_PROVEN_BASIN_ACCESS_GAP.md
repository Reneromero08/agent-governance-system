# Proven Basin Access Gap

## Basin theorem actually established

For a satisfying voltage vector with proper isolated index set `I`, the published basin
theorem contains

```text
J0(solution,I) x {0}^m x [1,+infinity)^m.
```

For every isolated voltage coordinate, `J0` occupies an interval of length `2 gamma`
inside the full interval of length `2`. Its relative voltage volume is therefore

```text
gamma^|I|.
```

At the published `gamma=1/4`, an isolated unique solution with `|I|=n` has guaranteed
voltage fraction

```text
4^-n.
```

The theorem also fixes every short-memory coordinate to exactly zero. If those
coordinates are sampled continuously from `[0,1]^m`, this proven subset has zero full
Lebesgue volume.

## Consequence

The basin theorem is a strong local capture result once a trajectory is already in the
correct solution orthant with short memory zero. It does not by itself prove that a
public answer-blind seed reaches any solution basin with inverse-polynomial probability.

Setting `x_s=0` is public and lawful. Choosing the correct voltage orthant is not, because
its signs encode a satisfying assignment on the isolated coordinates.

The instanton argument must therefore supply an additional global-access theorem:

```text
from the declared public seed
-> enter a solution-chain critical region
-> follow at most polynomially many index-descending instantons
-> cross the fixed solution threshold.
```

Without that theorem, random initialization does not provide a worst-case randomized
algorithm, and the deterministic public perturbation remains an experimentally useful
seed rather than a proven universal one.

## Status

```text
RESTRICTED_SOLUTION_ORTHANT_BASIN_RECONSTRUCTED
UNIQUE_SOLUTION_PROVEN_VOLTAGE_FRACTION_IS_GAMMA_TO_N
PUBLIC_SEED_GLOBAL_BASIN_ACCESS_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

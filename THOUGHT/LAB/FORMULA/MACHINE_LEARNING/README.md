# Living Formula: Machine Learning Program

This folder converts the Living Formula into a machine-learning research
program that can be falsified on external data.

The purpose is narrow:

1. lock one operational ML definition of the formula;
2. compare it against accepted baselines from representation learning;
3. test whether it is descriptive, predictive, and eventually causally useful;
4. reject the ML mapping if it fails.

This does NOT assume the larger thesis is true. It defines what would count as
evidence for or against the ML-specific version.

## Core ML Formula

```text
R_full = (E / grad_S) * sigma^Df
R_simple = E / grad_S
```

with one locked representation-level definition in
[OPERATIONAL_DEFINITIONS.md](OPERATIONAL_DEFINITIONS.md).

## Program Structure

- [HYPOTHESES.md](HYPOTHESES.md)
- [OPERATIONAL_DEFINITIONS.md](OPERATIONAL_DEFINITIONS.md)
- [BASELINES.md](BASELINES.md)
- [DATASETS.md](DATASETS.md)
- [PASS_FAIL_CRITERIA.md](PASS_FAIL_CRITERIA.md)
- [EXPERIMENT_01_FROZEN_REPRESENTATIONS.md](EXPERIMENT_01_FROZEN_REPRESENTATIONS.md)
- `code/formula_ml.py`
- `code/experiment_01_frozen_representations.py`

## First Practical Goal

The first experiment asks a modest but important question:

Can the formula distinguish externally labeled pure semantic clusters from mixed
semantic clusters on real NLP datasets, and does it outperform simpler geometry
baselines?

That is deliberately weaker than "formula for everything." If the metric fails
here, stronger ML claims do not survive.

## Runtime

Primary execution path for this program:

```powershell
py -3.11 THOUGHT\LAB\FORMULA\MACHINE_LEARNING\code\experiment_01_frozen_representations.py --dataset sst2 --model sentence-transformers/all-MiniLM-L6-v2 --device cuda --limit-per-label 200 --cluster-size 8 --clusters-per-class 40
```

## Standards

- External data only for claims
- Fixed definitions
- Explicit baselines
- Honest negative results
- No post-hoc redefinition of symbols

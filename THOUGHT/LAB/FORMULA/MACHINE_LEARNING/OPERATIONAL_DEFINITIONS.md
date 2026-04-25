# Operational Definitions

This file locks the machine-learning definitions used in this folder.

## Representation-Level Formula

Given a set of embeddings `X = {x_1, ..., x_n}`:

### 1. E

```text
E = mean_{i < j} cos(x_i, x_j)
```

Interpretation:
average pairwise agreement.

### 2. grad_S

```text
grad_S = std_{i < j} cos(x_i, x_j)
```

Interpretation:
dispersion of agreement.

### 3. sigma

Let `lambda_1, ..., lambda_d` be the positive eigenvalues of the covariance of
the centered embeddings.

Participation ratio:

```text
PR = (sum lambda_i)^2 / sum lambda_i^2
```

Then:

```text
sigma = PR / d
```

Interpretation:
normalized effective dimensional occupancy.

### 4. Df

Fit a line to the log-log eigenspectrum:

```text
log(lambda_k) = -alpha * log(k) + c
Df = 2 / alpha
```

when `alpha > 0`.

Interpretation:
spectral-complexity proxy.

### 5. Formula Outputs

```text
R_simple = E / max(grad_S, eps)
R_full = (E / max(grad_S, eps)) * sigma^Df
```

with `eps = 1e-10`.

## Locked Scope

This folder does NOT silently swap:

- `E` for likelihood, probability, or error
- `grad_S` for a true gradient vector
- `sigma` for a universal constant
- `Df` for a physics quantity unrelated to representation spectra

## Interpretation Rule

In this folder, the formula is tested only as a representation-geometry metric
unless a later document explicitly introduces and justifies a training-level or
class-conditional variant.

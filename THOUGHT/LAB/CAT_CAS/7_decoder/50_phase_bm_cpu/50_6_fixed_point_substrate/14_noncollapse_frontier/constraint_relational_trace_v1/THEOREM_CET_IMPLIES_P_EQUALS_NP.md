# Conditional Theorem: Uniform Polynomial CET Implies P Equals NP

## Definitions

A `ConstraintHolo` compiler is **uniform polynomial** when a deterministic Turing
machine constructs the public relational object from an input formula in polynomial
time and the object has polynomial description size.

A `CET` decision boundary is **exact polynomial** when it returns one of:

```text
VALID_SAT
VALID_UNSAT
INVALID_CARRIER
```

for every public `ConstraintHolo`, uses polynomial total resources, and its native
operations admit deterministic polynomial-overhead simulation in the standard bit
model.

## Theorem

If a uniform exact polynomial `CET` decision boundary exists for all public three-clause
normal-form objects, then `P = NP`.

## Proof

Three-clause satisfiability is NP-complete. Given an arbitrary public three-clause
formula `F`:

1. Compile `F` into `ConstraintHolo(F)` in polynomial time.
2. Apply the exact polynomial `CET` boundary.
3. Return `SAT` when the boundary emits `VALID_SAT` and `UNSAT` when it emits
   `VALID_UNSAT`.
4. Treat `INVALID_CARRIER` as execution failure, never as `UNSAT`.

By assumption, compilation, native execution, readout, restoration, and standard-model
simulation are polynomial. Therefore three-clause satisfiability belongs to `P`.
Because three-clause satisfiability is NP-complete and `P` is contained in `NP`, it
follows that `P = NP`.

## Witness Corollary

An exact decision boundary is sufficient to render a conventional witness using at
most `1 + 2n` boundary calls for `n` public variables.

Maintain a public partial assignment. For each unassigned variable `x_i`:

1. Restrict the public relation with `x_i = 0`.
2. If the exact boundary reports satisfiable, retain zero.
3. Otherwise restrict with `x_i = 1` and retain one.
4. Continue until every public variable is fixed.

At least one branch remains satisfiable because the current restricted relation was
satisfiable before the choice. The final assignment is checked by the ordinary public
relation. Every restriction and every boundary call has polynomial size and cost under
the theorem assumptions.

The executable reference appears in `conditional_p_equals_np.py`. It is a classical
post-boundary theorem instrument and not the missing native operator.

## Physical-Model Boundary

If the native `CET` operation does not admit deterministic polynomial-overhead
simulation in the standard bit model, the result proves only:

```text
NP is contained in polynomial CAT_CAS computation
```

It does not by itself establish ordinary `P = NP`.

## Current Status

```text
conditional implication: ESTABLISHED
reference witness reduction: ESTABLISHED
uniform polynomial native CET: NOT ESTABLISHED
P = NP: NOT PROVEN
```

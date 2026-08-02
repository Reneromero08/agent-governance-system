# Minimal Fixed-Deadline Obligation for `P = NP`

## Status

```text
SEPARATE_UNSAT_ATTRACTOR_NOT_REQUIRED_FOR_CLASSICAL_COMPLEXITY_IMPLICATION
NATIVE_RESTORATION_NOT_REQUIRED_FOR_CLASSICAL_COMPLEXITY_IMPLICATION
PUBLIC_SEED_SAT_COMPLETENESS_BY_POLYNOMIAL_DEADLINE_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Conditional theorem

Let `F` be a public 3-CNF formula. Suppose there is one uniform polynomial `q` and one
uniform deterministic simulation of the reduced exact-product phase carrier such that:

1. the public compiler and declared answer-blind seed are computable in polynomial time;
2. the carrier can be simulated to time `q(|F|)` with polynomial resources;
3. for every satisfiable `F`, the threshold assignment at that deadline satisfies `F`;
4. the terminal assignment is checked by the ordinary public verifier.

Then the following deterministic algorithm decides 3-SAT:

```text
compile F
initialize the declared public seed
simulate the reduced carrier to q(|F|)
read the one terminal assignment
verify it against F
if verification passes: SAT
otherwise: UNSAT
```

Correctness is immediate:

- if `F` is satisfiable, assumption 3 guarantees terminal verification passes;
- if `F` is unsatisfiable, no assignment can pass the public verifier.

Therefore the same deadline supplies a total SAT/UNSAT decision. A separate UNSAT
attractor, zero-mode certificate, or physical no-witness signal is not required for the
ordinary complexity implication.

Because 3-SAT is NP-complete, these assumptions imply

```text
3SAT in P
NP subseteq P
P = NP.
```

## Relationship to the forward resource theorem

The reduced carrier has polynomial state dimension and unconditional polynomial native
range. The explicit forward resource theorem proves:

```text
polynomial deadline
-> polynomial forward trajectory length
-> polynomial forward logit range and precision.
```

Thus the central unresolved mathematical statement is now concentrated in assumption 3:

```text
For every satisfiable public 3-CNF and every declared presentation gauge, the declared
public seed reaches a robust satisfying terminal phase section by one formula-uniform
polynomial deadline.
```

## Restoration boundary

Native inverse restoration remains mandatory for the stronger CAT_CAS lifecycle claim.
It is not logically required to show that the corresponding terminal computation belongs
to classical `P` once a deterministic polynomial simulation and total terminal verifier
are established.

Accordingly, the campaign must maintain two claim ladders:

```text
classical P = NP implication:
    public-seed SAT completeness + polynomial forward simulation

full CAT_CAS mechanism:
    the above + native borrow/evolve/extract/restore closure
```

Neither ladder is complete at this checkpoint.

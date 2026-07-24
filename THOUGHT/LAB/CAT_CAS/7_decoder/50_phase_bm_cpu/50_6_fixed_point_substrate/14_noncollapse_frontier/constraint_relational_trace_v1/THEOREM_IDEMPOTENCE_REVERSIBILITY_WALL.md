# Idempotence and Reversibility Wall

## Proposition 1: Exact Public Semantics

For a public three-clause normal-form formula `F`, `ConstraintHolo(F)` contains one
constant-size local relation for each clause and one equality junction for every public
variable occurrence. Its open boundary denotes exactly:

```text
Sol(F) = {x | every local clause relation accepts the restrictions of x}
```

This statement is semantic. It does not provide an efficient native closure operation.

## Proposition 2: Reversible Evaluation Dilation

Let `f_F(x)` be one when `x` satisfies `F` and zero otherwise. The map

```text
U_F : (x, b) -> (x, b XOR f_F(x))
```

is a bijection and is its own inverse.

### Proof

The assignment register `x` is preserved. For a fixed `x`, applying the map twice gives:

```text
b XOR f_F(x) XOR f_F(x) = b.
```

Therefore `U_F composed with U_F` is the identity. The program description may be
constructed from the public local relations without storing a witness.

## Proposition 3: Evaluation Is Not Existential Closure

The reversible dilation above does not decide whether `Sol(F)` is empty. It evaluates
one preserved boundary state at a time. A complete existential boundary would implement:

```text
exists x such that f_F(x) = 1
```

while rendering one idempotent truth value.

The existence value obeys:

```text
1 OR 1 = 1.
```

Distinct satisfying provenance states must therefore map to the same visible truth
value. A bijective closed evolution cannot destroy their distinction. Any reversible
realization must retain the distinction in unresolved carrier degrees of freedom,
ancilla, path geometry, or another restorable subsystem.

## Corollary: Provenance Conservation

A reversible existential process may hide provenance from the boundary, but it cannot
erase provenance before the declared boundary.

This does not prove that polynomial carrier resources are impossible. It defines the
proof obligation:

```text
retain every distinction required by reversibility
without materializing one independent carrier state per classical derivation
while producing one exact idempotent existence result at the boundary
```

## Proposition 4: Factorized Projector Boundary

The product of public local satisfaction projectors is a compact symbolic operator:

```text
Q_F = product over clauses j of P_j.
```

For every basis assignment `x`:

```text
Q_F |x> = |x>  when x satisfies F
Q_F |x> = 0    otherwise.
```

The public description of `Q_F` is linear in the formula. The unresolved operation is
not construction of `Q_F`. It is exact determination of whether the image of `Q_F` is
empty, followed by lawful witness rendering, without expanding the basis or hiding the
same cost in preparation, precision, settling, energy, readout, or restoration.

## Theorem Target

A proof campaign succeeds only after constructing a uniform public family of native
operators `CET_F` with all of the following properties:

```text
exact:          CET_F reports nonempty if and only if Sol(F) is nonempty
noncollapse:    no complete assignment or derivation list is materialized internally
reversible:     actual carrier evolution has a program-derived inverse
idempotent:     duplicate derivations do not change the visible existence state
total:          SAT, UNSAT, and invalid carrier are distinct boundary states
witness:        SAT yields a conventional witness accepted by an independent checker
polynomial:     every physical and mathematical resource is polynomial in |F|
transferable:   native operations admit deterministic polynomial-overhead simulation
```

Until these properties are established, the only valid token is:

```text
CET_NATIVE_OPERATOR_NOT_ESTABLISHED
```

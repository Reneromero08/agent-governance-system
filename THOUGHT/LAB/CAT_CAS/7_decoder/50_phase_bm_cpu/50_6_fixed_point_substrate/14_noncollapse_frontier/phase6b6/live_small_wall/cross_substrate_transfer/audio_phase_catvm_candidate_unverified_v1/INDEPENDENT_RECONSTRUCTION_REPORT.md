# Independent Reconstruction Report

Status: noncanonical transfer candidate; no claim promotion; no Small Wall
crossing.

## Independence boundary

The clean-room verifier is implemented in Python and does not invoke or
translate the production C phase recurrence, source qualifier, result JSON,
review JSON, state, schedule receipts, or timing outputs. Source results were
compared only after the independent output was generated.

Methods:

| Candidate | Independent method |
|---|---|
| A | complete Boolean zero-set enumeration followed by F3 multilinear interpolation |
| B | Kahn dependency analysis and exhaustive reversible-pebble graph search |
| C | square-free symbolic GF(2) substitution over every public program |
| D | raw local Boolean transducer construction and finite-horizon suffix-bisimulation refinement |

Implementation:
`independent_verifier/cleanroom_verify.py`

Tests:
`independent_verifier/test_cleanroom_verify.py`

Result:
`independent_verifier/INDEPENDENT_RESULTS.json`

Result file SHA-256:

```text
bdcedee1075c9262cff2600db53a94b66c7342e723a8a832b1793e824cbdc98f
```

The canonical-JSON payload self-hash is:

```text
c5e3849847ff630ec416b867035c22df61e4cd98b27db08173cbe6b6fc0df97b
```

Two fresh executions were byte-identical.

## Candidate A

The public F3 coefficients are interpreted as the unique multilinear
polynomial

```text
p(x,y) = c00 + c10*x + c01*y + c11*x*y mod 3
```

whose zero set is the declared relation. The verifier enumerates both values
of the hidden variable `u`, constructs the existential composition zero set,
then independently interpolates its four F3 coefficients. Intersection is
checked by the zero-set identity:

```text
p^2 + q^2 = 0 mod 3  iff  p = q = 0
```

Computed results:

```text
primary F = [0,1,1,1]
primary Z = [0,2,1,1]
reuse   F = [1,2,2,2]
reuse   Z = [1,2,0,2]
```

The final primary and reuse boundaries agree with the source execution, but
the clean-room computation did not consume those values as an oracle.

An abstract resident-exponent carrier independently confirms:

```text
nominal G^-1 then F^-1 restores
wrong G^-1 leaves nonzero Z
missing G^-1 leaves nonzero Z
F^-1 before recomputed G^-1 leaves nonzero Z
```

This verifies the algebraic composition and inverse ordering law. It does not
repair Candidate A's source protocol defect: the source service sends the
answer before restoration.

Supported scope:
`INDEPENDENTLY_VERIFIED_SOURCE_LOCAL` for the bounded algebraic law only.
Machine-boundary acceptance remains rejected pending an atomic wrapper.

## Candidate B

The public manifest independently parses as an acyclic 15-node graph rooted
at 815. The four shared producers and fanouts are:

```text
805 -> 4
806 -> 3
807 -> 3
808 -> 2
```

An exhaustive reversible-pebble search used only dependency preconditions:
a node may be toggled when its children are resident. It found:

```text
capacity 5: root unreachable
capacity 6: 40 actions to the fixed six-live projection state
capacity 7: 28 actions
capacity 8: 26 actions
capacity 9: 24 actions
```

The 28-action witness reverses literally in 28 actions and returns to the
empty state. Missing, duplicated, and invalid-node schedule mutations are
detected.

This independently confirms that a 28-forward/28-reverse reversible schedule
exists for the public graph. It does not independently establish that the
source scheduler generated its tape generally. The source's nine physical
working slots exceed the clean-room witness's seven logical pebbles; two
operator-temporary slots are a plausible accounting explanation, but that is
an inference, not a verified identity.

Supported scope:
`INCONCLUSIVE_REQUIRES_SPECIFIC_NEXT_TEST`. A novel topology must be compiled
by both schedulers and the source tape must be checked action by action.

## Candidate C

The verifier symbolically substitutes:

```text
u = alpha + beta*a*b
v = gamma + delta*u*c
d = eta + theta*v*e
```

over square-free GF(2) monomials. It independently derives:

```text
[1, eta, theta*gamma, theta*delta*alpha, theta*delta*beta]
```

All four public fixtures agree. Exhaustive enumeration of all 64 public
programs finds only 16 distinct five-bit boundaries.

This sharpens the obstruction:

```text
raw five-bit table              320 bits
packed nonconstant table        256 bits
direct evaluator                  4 ANDs
unique boundary values            16
```

The direct formula remains the strongest relevant compact baseline found so
far. The source's narrow fixed-schema capacity obstruction is therefore
independently supported, while its timing observations remain descriptive
only.

Supported scope:
`INDEPENDENTLY_VERIFIED_TRANSFERABLE_OBSTRUCTION`, limited to fixed public
schemas with bounded program space. It does not transfer phase timing or
physical resource claims.

## Candidate D

The first clean-room attempt matched only quotient ranks and produced the
mirror-image class membership. That attempt was rejected. The corrected
verifier constructs raw depth-bit local states and the repeated-neighbor
AND/OR transition relation, then refines future-equivalence signatures from
the fixed right boundary.

For every depth 2 through 8 and horizon 1 through 16, both homogeneous AND
and homogeneous OR produce exactly:

```text
r = 2                         at horizon 1
r = min(depth+1, horizon+2)   otherwise
```

and, when `2 <= horizon < depth`, the exact classes:

```text
height 0 .. horizon-1     singleton
height horizon .. d-1     one middle class
height d                  distinct all-leading class
```

The campaign checked 112 `(depth,horizon)` cases, including 28 abstract
depth-greater-than-horizon cases. Deliberate top/middle overmerge is rejected;
splitting the middle class is proven unnecessarily fine. AND/OR duality holds
for every checked case.

This independently verifies the restricted homogeneous-family quotient law.
It does not establish mixed layers, different neighborhoods, general
Boolean-TT minimization, or source support for depth greater than width.

Supported scope:
`FAMILY_SCOPED_QUOTIENT`. Transfer classification remains
`INCONCLUSIVE_REQUIRES_SPECIFIC_NEXT_TEST` until the generalization campaign.

## Mechanical verification

Command:

```text
./.venv/bin/python -m pytest <candidate>/independent_verifier/test_cleanroom_verify.py -q
```

Result:

```text
81 passed
exit code 0
```

Exact command, output, exit status, and deterministic replay hashes are under
`raw_logs/independent_reconstruction/`.

## Current decision boundary

Independent reconstruction supports:

- A's scalar algebra and inverse-ordering control, not its response protocol.
- B's public DAG structure and existence of a reversible bounded schedule,
  not source scheduler generality.
- C's fixed-schema compact-baseline obstruction.
- D's homogeneous repeated AND/OR quotient, family-scoped only.

None of these findings is canonical. None modifies physical Family 10h
evidence or raises its claim ceiling.

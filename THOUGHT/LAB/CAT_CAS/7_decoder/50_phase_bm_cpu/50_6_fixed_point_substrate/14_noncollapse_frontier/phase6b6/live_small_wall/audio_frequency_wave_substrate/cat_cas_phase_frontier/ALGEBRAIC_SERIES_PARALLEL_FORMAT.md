# Public Series/Parallel Relational Phase Format

Status: mutable CAT_CAS frontier checkpoint

## Purpose

This checkpoint lifts the fixed diamond into a public two-terminal
series/parallel relation graph. The source declares typed nodes, arbitrary
binary Boolean/F3 relations, and a public internal-node elimination order.
The compiler reads graph topology only. Native execution closes each
degree-two internal interface by exact phase composition and merges each
parallel path by exact phase intersection.

The graph, not a host-selected witness, determines the relation that reaches
the two external ports.

## Canonical grammar

```text
CATCAS_SERIES_PARALLEL_RELATION 1
TYPE BOOLEAN_F3
NODE NAME EXTERNAL|INTERNAL
RELATION NAME FIRST SECOND c00 c10 c01 c11
ELIMINATE INTERNAL_NAME
END
```

Exactly two nodes are external. Identifiers begin with an uppercase ASCII
letter and then contain only uppercase ASCII letters, digits, or underscore.
Each relation has two distinct declared endpoints and four single-digit
coefficients in `{0,1,2}`. Every internal node appears exactly once in the
elimination schedule.

At every public elimination step the selected internal node must have active
degree two. The resulting relation is inserted between its two neighbors.
Any relation already present between those endpoints is immediately merged.
The complete schedule must leave exactly one relation between the two
external nodes and must contain at least one parallel merge.

Records are LF-terminated. Blank records, CR bytes, duplicate identifiers,
undeclared nodes, self relations, wrong arity, records after `END`, invalid
coefficients, external elimination, duplicate or missing elimination,
non-degree-two elimination, self-producing elimination, a nonreduced final
graph, and a tree with no shared-interface merge reject.

## Native relation algebra

Each binary relation is the zero set of a multiaffine polynomial in

```text
F3[x,y] / (x^2-x, y^2-y).
```

Its four coefficients are stored only as relative complex phases in the
basis `{1,x,y,xy}`. The two native graph-reduction operators are:

```text
INTERSECT(f,g) = f^2 + g^2

COMPOSE(f,g) =
    product over u in {0,1}
    of (f(x,u)^2 + g(u,y)^2)
```

Because every nonzero F3 element squares to one, intersection is zero
exactly when both operands are zero. The composition product is zero exactly
when at least one Boolean value of the closed port satisfies both relations.
The two witness values are fixed algebraic factors in one phase polynomial;
native execution contains no assignment loop or witness selection.

Floating phase drift is suppressed by the continuous three-well map

```text
L(z) = unit(2z + conj(z)^2)
```

applied three times to native algebra outputs. Every legal cube root is a
fixed point because `conj(z)^2 = z` when `z^3 = 1`. The map contains no
phase-label branch, integer coefficient decode, scalar relation evaluation,
or adjudicator feedback.

## Compilation and carrier law

Compilation produces only:

```text
operation kind
left message address and orientation
right message address and orientation
output message address
```

Relation coefficients never enter schedule construction. For `E` public
relations and `O` compiled native operations:

```text
input relation cells       4E
resident message cells     4O
latched boundary cells       4
total carrier cells         4(E + O + 1)
retained inverse factors     0
tuple/witness/truth slots    0
```

The forward execution encodes the public relation phases, executes the
compiled operations in order, and copies the final four phases to the
declared boundary. The boundary survives outside inverse history. The engine
then removes the boundary copy, recomputes and reverses every actual
operation in reverse order, reverses every input encoding, and reuses that
same restored carrier for another public process with identical carrier
geometry.

## Decisive nested-cycle discriminator

The primary graph contains two diamonds in series plus two open leaf
relations:

```text
A--U--(W|Z)--V--(X|Y)--T--B
```

It compiles ten input relations into seven compositions and two
intersections. The exact boundary accepts only `{00}`.

```text
nominal exact closure               {00}
bypass first parallel intersection  {00,10}
ordinary addition at first merge    {00,11}
```

The two altered forward paths reverse their own actual histories and restore
cleanly. A rotated boundary inverse and an omitted resident-message inverse
each leave restoration error `1.73205080757`.

The different reuse process reverses relation presentation and changes all
input relations to equality. It consumes the actual restored 80-cell carrier
and produces `{00,11}` before restoring again.

## Independent adjudication and qualification

`algebraic_series_parallel_reference.c` is a separate scalar executable. It
independently parses and reduces the public graph in F3. For graphs with at
most twenty internal nodes it also enumerates every complete Boolean
assignment and compares the resulting existential projection with the
reduced boundary. It is never linked into or called by native recurrence.

`generate_series_parallel_capacity_fixture.c` deterministically emits one
through fifteen serial diamonds. Positive pattern numbers populate arbitrary
F3 input coefficients. Pattern zero emits the shared-interface
discriminator.

Qualification covers:

```text
arbitrary-coefficient graphs with full enumeration   120
depth/capacity graphs                                  15
maximum nodes                                          46
maximum public relations                               60
maximum native operations                              59
maximum carrier cells                                 480
structural negative fixtures                            6
ASan/UBSan executions                                   3
```

Every native boundary polynomial matches the independent scalar reduction.
All 120 bounded projections match complete-assignment enumeration. The
15-diamond graph retains the `{00}` boundary, restores within
`1.66533453694e-16`, and passes the same causal controls.

The capacity probe first exposed amplifying floating error in the unlocked
implementation: the discrete result remained correct, but root distance grew
to `0.0740742833362`, beyond the prospective `4e-10` envelope. The continuous
phase lock reduced the maximum capacity root error to
`1.57009245868e-16` without adding intermediate decode.

## Claim boundary

A passing reviewed checkpoint establishes only:

```text
SOFTWARE_PUBLIC_SERIES_PARALLEL_EXACT_RELATIONAL_PHASE_CLOSURE_REFERENCE_ONLY
```

It establishes a public, topology-compiled class of bounded cyclic relation
graphs; exact phase-native internal-port closure; unresolved resident
relation messages; graph-sensitive boundary projection; actual reversal;
restoration; and reuse.

It does not establish arbitrary graph treewidth, automatic elimination-order
discovery, domains beyond Boolean/F3, arbitrary relation arity, asymptotic
advantage over compact classical variable elimination, physical phase
computation, or unlimited catalytic phase computation.

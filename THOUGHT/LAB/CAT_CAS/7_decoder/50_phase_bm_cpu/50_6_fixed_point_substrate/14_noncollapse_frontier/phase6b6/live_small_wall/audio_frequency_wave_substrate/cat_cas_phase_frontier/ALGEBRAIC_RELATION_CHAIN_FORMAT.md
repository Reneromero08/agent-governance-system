# Algebraic Relation Chain Phase Format

Status: mutable CAT_CAS frontier checkpoint
Version: `CATCAS_ALGEBRAIC_RELATION_CHAIN 1`

## Purpose

This format lifts the reviewed single-internal-port algebraic relation process
into a repeatable chain. It tests whether many unresolved interfaces can be
closed by one uniform waveform-native law while carrier state, reversal, and
reuse remain linear in the number of public relations.

This is not the terminal holographic-relational machine. Branching graphs,
higher arity, relational memory, recursive traces, physical execution, and
unbounded computation remain open.

## Canonical syntax

```text
CATCAS_ALGEBRAIC_RELATION_CHAIN 1
TYPE BOOLEAN_F3
BOUNDARY <first-port> <second-port>
RELATION <name> <port-a> <port-b> <c00> <c10> <c01> <c11> ZEROSET
...
END
```

Blank lines and lines beginning with `#` are ignored. Records are separated by
LF. CR and embedded NUL bytes reject. Identifiers begin with a letter or `_`,
then contain only letters, digits, or `_`, and contain at most 31 bytes.
Coefficients are canonical single characters `0`, `1`, or `2`.

There is no frozen relation-count ceiling. Parsing is streaming; the practical
limit is the process address space and available memory. A valid chain has at
least two relations, two distinct declared boundary ports, unique relation
names, distinct endpoints for every relation, and exactly one simple path that
uses every relation and terminates at the second declared boundary. Record
order and endpoint presentation do not affect the normalized process.
Branching, cycles, disconnected components, duplicate relation names, early
arrival at the final boundary, and unused relations reject.

## Relation law

Every public relation is the Boolean-domain zero set of

```text
c00 + c10*x + c01*y + c11*x*y = 0  in F3
```

and must be total toward both ports on `x,y in {0,1}`. This prospective
bi-total admission law is checked before native execution. It is not selected
from a computed result.

For adjacent relations written as

```text
f(x,y) = A(x)*y + B(x)
g(y,z) = C(z)*y + D(z)
```

the internal `y` port closes through the algebraic resultant

```text
R(x,z) = B(x)*C(z) - A(x)*D(z).
```

The admitted bi-total class is closed under this operation on the Boolean
domain. Therefore the output relation from one closure is a lawful input to
the next closure. The chain applies this same law once per internal port.

## Native phase representation

Let `omega = exp(2*pi*i/3)`. A coefficient `c` is stored relationally as the
unit phasor `omega^c` against that carrier cell's nontrivial baseline.
Multiplication in F3 is implemented by a fixed complex Fourier polynomial over
input phases. Subtraction is conjugate phase multiplication. The native
recurrence does not enumerate assignments, select witnesses, compute Boolean
pair masks, call the independent oracle, or decode coefficients between
composition layers.

Scalar coefficients are used only to parse the public input, enforce the
prospective type/admission law, and encode that input into phase. Coefficients
are decoded once, at the explicit final boundary after every native closure.

## Carrier layout and history

For `n` public relations:

```text
input relation cells       = 4*n
derived relation cells     = 4*(n-1)
total carrier cells        = 8*n - 4
tuple slots                = 0
witness slots              = 0
retained inverse factors   = 0
```

The derived history is one four-phase relation per composition layer. It is
linear, not an assignment tree or exponential witness provenance.

Forward execution:

```text
borrow carrier
encode every public relation
apply native resultants from the first boundary to the second
latch the final four-phase boundary relation
```

Reverse execution traverses the actual resultants in reverse order, then
removes the public encodings in reverse order. The latched result is outside
the reversed carrier history. The exact restored carrier is then reused for a
second process of the same carrier shape.

## Frozen numerical law

```text
coefficient-root maximum error      <= 2e-10
correct restoration maximum error   <= 2e-12
carrier unit-magnitude error         <= 2e-12
applicable inverse-control failure   >= 1e-3
```

`maximum_root_error` is the maximum complex absolute distance between a final
boundary phase and its nearest third root of unity.
`restoration_max_abs` is the maximum complex absolute cell difference from the
exact borrowed carrier snapshot. `displacement_l2` is the Euclidean norm of
the complete complex carrier displacement immediately before reversal.
`carrier_integrity_error` is the maximum absolute unit-magnitude deviation.

Wrong-factor, forward-order, and omitted-final-resultant inverse controls are
declared applicable only when their factor path differs from the correct one.
Every applicable control must miss restoration by at least `1e-3`.
For the minimum valid two-relation chain there is only one resultant, so
forward-order and reverse-order traversal are identical and the order control
is necessarily inapplicable. Wrong-factor and omitted-resultant controls
remain applicable when their factors are nontrivial.

## Independent adjudication

`algebraic_relation_chain_reference.c` is a separately compiled scalar
executable. It may enumerate the bounded Boolean domain after the native
process. At every layer it compares the coefficient resultant's Boolean zero
set with ordinary existential relational composition. It never feeds the
native recurrence or chooses its result.

## Claim boundary

A passing qualification establishes a software reference mechanism for
repeatable, carrier-causal, phase-native internal closure across a finite
linear chain, with linear carrier history, explicit extraction, exact inverse
traversal, restoration, and reuse.

It does not establish branching trace composition, arbitrary relation classes,
hardware behavior, computational advantage, an infinite physical resource, or
unlimited catalytic computation.

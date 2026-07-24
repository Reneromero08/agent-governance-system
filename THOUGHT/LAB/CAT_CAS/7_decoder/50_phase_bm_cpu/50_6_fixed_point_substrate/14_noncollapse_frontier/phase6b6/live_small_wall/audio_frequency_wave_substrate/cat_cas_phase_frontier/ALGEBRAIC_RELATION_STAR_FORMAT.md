# Algebraic Relation Star Phase Format

Status: mutable CAT_CAS frontier checkpoint

Version: `CATCAS_ALGEBRAIC_RELATION_STAR 1`

## Purpose

This format is the first branching typed relational trace. A public process
contains at least three binary relations meeting at one unresolved Boolean
hub. Native phase evolution closes the hub and exposes a factorized relation
over every external port. The factorization preserves open relational
structure; it does not expand a `2^d` boundary truth table.

## Canonical syntax

```text
CATCAS_ALGEBRAIC_RELATION_STAR 1
TYPE BOOLEAN_F3
HUB <hub-port>
RELATION <name> <port-a> <port-b> <c00> <c10> <c01> <c11> ZEROSET
...
END
```

Blank lines and lines beginning with `#` are ignored. LF separates records.
CR and embedded NUL bytes reject. Identifiers contain at most 31 bytes, begin
with a letter or `_`, then contain only letters, digits, or `_`. Coefficients
are canonical single characters `0`, `1`, or `2`.

Every relation must touch the declared hub exactly once. The other endpoint is
an external port. Relation names and external ports are unique. At least three
relations are required. Record order and endpoint presentation do not affect
the normalized process; external ports are ordered lexicographically.

There is no fixed branch-count ceiling. Parsing is streaming. Practical limits
are the process address space, available memory, and the explicit quadratic
boundary fill-in described below.

## Local relation and admission law

After orienting the external port first and hub second, every branch is the
Boolean-domain zero set

```text
f_i(x_i,h) =
    c00_i + c10_i*x_i + c01_i*h + c11_i*x_i*h = 0 in F3.
```

Every branch must be total toward both Boolean ports. Consequently, for each
external value, its hub fiber

```text
S_i(x_i) = { h in {0,1} : f_i(x_i,h)=0 }
```

is one of `{0}`, `{1}`, or `{0,1}`; it is never empty.

## Branching closure theorem

The exact existential boundary is

```text
exists h: f_0(x_0,h)=...=f_(d-1)(x_(d-1),h)=0.
```

A family of nonempty subsets of the two-point set `{0,1}` has nonempty total
intersection exactly when every pair intersects. Therefore the hub can be
eliminated into one binary boundary factor per unordered external-port pair:

```text
R_ij(x_i,x_j) =
    B_i(x_i)*A_j(x_j) - A_i(x_i)*B_j(x_j) = 0,

where f_i(x_i,h) = A_i(x_i)*h + B_i(x_i).
```

The surviving boundary relation is the factorized conjunction of all
`R_ij=0`. It has `d*(d-1)/2` factors and is exact even when both hub values are
witnesses. Witness identity and multiplicity are not stored.

## Native phase process

Let `omega = exp(2*pi*i/3)`. Each coefficient `c` is encoded as the relative
unit phase `omega^c` against a deterministic nontrivial carrier baseline.
Coefficient multiplication is a fixed roots-of-unity Fourier polynomial.
Subtraction is conjugate phase multiplication.

For every external pair, four native complex phase operations form its
resultant coefficient phases. The native process does not:

```text
enumerate external assignments
enumerate or choose hub witnesses
construct a boundary truth table
call the scalar reference
decode coefficients between factors
feed a boundary result back into evolution
```

All factor coefficients are decoded only after every native pair closure is
complete. The ordered factor record binds the two external-port names and its
four coefficients into `boundary_factor_fnv1a64`.

## Carrier and resource law

For `d` public branches and `m=d*(d-1)/2` boundary factors:

```text
input coefficient cells       = 4*d
boundary factor cells         = 4*m
total carrier cells           = 4*d + 4*m = 2*d*(d+1)
tuple slots                   = 0
witness slots                 = 0
truth-table slots             = 0
retained inverse factors      = 0
```

The boundary fill-in is explicitly `O(d^2)`, not hidden or called fixed-size.
It is exponentially smaller than a complete `2^d` boundary table, but no
computational-advantage claim follows.

## Extraction, reversal, restoration, and reuse

The factorized boundary record is latched outside carrier history. Correct
inverse execution removes the actual pair resultants and then the public input
encodings. Pair resultants occupy disjoint outputs and commute, so pair order
is not an honest inverse-failure control.

The decisive controls are:

```text
wrong coefficient-factor inverse
cyclically scrambled pair-to-output geometry
omitted final pair inverse
```

Each control is applicability-gated. Every applicable control must miss
restoration by at least the frozen minimum. The exact restored carrier is
reused for another same-degree star.

## Frozen numerical law

```text
coefficient-root maximum error      <= 2e-10
correct restoration maximum error   <= 2e-12
carrier unit-magnitude error         <= 2e-12
applicable inverse-control failure   >= 1e-3
```

`maximum_root_error` is maximum complex distance to the decoded third root of
unity. `restoration_max_abs` is maximum complex cell distance from the exact
borrowed snapshot. `displacement_l2` is the full complex carrier displacement
before reversal. `carrier_integrity_error` is maximum unit-magnitude drift.

## Independent adjudication

`algebraic_relation_star_reference.c` is separately compiled and never linked
into the native engine. It independently derives every pair factor. For stars
of at most 20 branches, it also enumerates all bounded boundary assignments and
compares the factor graph with exact existential hub projection.

`algebraic_relation_star_closure_survey.c` exhausts all admitted signature
triples and quadruples. It is evidence for the two-point intersection theorem,
not part of native execution.

## Claim boundary

A passing qualification establishes one finite branching typed relational
trace: an unresolved hub is eliminated by carrier-causal phase operations into
an exact factorized boundary relation, followed by actual inverse restoration
and reuse.

It does not establish arbitrary graph elimination, multiple interacting hubs,
general hyperrelations, subquadratic fill-in, advantage over compact ordinary
software, physical waveform computation, or unlimited catalytic computation.

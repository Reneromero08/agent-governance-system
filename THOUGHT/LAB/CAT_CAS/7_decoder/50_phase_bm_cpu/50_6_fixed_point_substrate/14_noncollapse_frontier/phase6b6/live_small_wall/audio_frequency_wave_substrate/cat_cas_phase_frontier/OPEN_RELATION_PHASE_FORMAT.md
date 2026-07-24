# Open Relation Phase Format

**Status:** mutable development format  
**Schema token:** `CATCAS_OPEN_RELATION_PROCESS 1`  
**Claim boundary:** structured quotient-relation calibration only

This format describes a connected geometry of typed open relations. It does
not list `Z_N` relation tuples, internal assignments, witnesses, candidate
states, or an expected boundary. It is nevertheless quotient-extensional:
each local relation stores the complete two-slot characteristic vector of the
fixed `Z_2` quotient.

## Port type

```text
PORT <name> CYCLIC_PARITY <N>
```

`N` is an even canonical unsigned decimal integer of at least four. The port
domain is `Z_N`. The carrier does not instantiate its `N` values.

The quotient map is parity:

```text
Z_N -> Z_2
```

## Local open relation

```text
RELATION <name> <port-a> <port-b> <subset>
```

`NONE` is reserved as the `DUPLICATE` sentinel and is not a legal relation
name.

`subset` is one Boolean-lattice subset of the quotient-difference group:

```text
EMPTY
SAME
OPPOSITE
BOTH
```

For example, `SAME` denotes every pair whose difference is even. Its one set
bit in the complete quotient characteristic vector therefore denotes `N/2`
outputs for every input without storing any `Z_N` pair.

## Geometry

Version 1 contains exactly three compatible ports and two local relations:

```text
CLOSE <internal-port>
BOUNDARY <first-open-port> <second-open-port>
DUPLICATE NONE|<left-relation-name>
END
```

The two local relations must form one chain through the closed port. Relation,
port, `CLOSE`, and `BOUNDARY` records may be presented in either order.

`DUPLICATE <left>` inserts an identical second presentation of the left
relation into the native idempotent union operator. It is a multiplicity
control, not an additional witness list.

## Native algebra

Each relation is two F3 phase symbols:

```text
(allows_same, allows_opposite)
```

Shared-port closure is Boolean convolution over `Z_2`:

```text
out_same =
    (left_same AND right_same)
    OR
    (left_opposite AND right_opposite)

out_opposite =
    (left_same AND right_opposite)
    OR
    (left_opposite AND right_same)
```

`AND` is the existing roots-of-unity product polynomial. `OR` is:

```text
a OR b = a + b - a*b
```

implemented as multiplication of complex phase factors. The native kernel
does not loop over `N`, boundary pairs, internal values, or witnesses.

The carrier contains exactly eight relative-phase cells:

```text
left relation        2
right relation       2
duplicate-union      2
boundary relation    2
```

## Boundary and inverse

The two boundary phase symbols are latched only after native composition.
They survive outside the carrier history. The executable then reverses:

```text
boundary convolution
-> duplicate union when present
-> right relation encoding
-> left relation encoding
```

Original relation-encoding inverse factors are recomputed from the immutable
public process definition. Duplicate-union and boundary-convolution inverse
factors are recomputed from still-resident phase relations. No per-step or
per-witness history is stored. Reuse consumes the same restored carrier.

Wrong, reordered, and omitted inverse controls are enforced only when their
forward operation was nonidentity. A reordered inverse is correctly
non-applicable for an empty composed boundary.

## Honest boundary

Version 1 is a compact translation-invariant quotient relation algebra. It is
non-enumerative over `Z_N`, but it is not a general intensional relation
representation: the fixed `Z_2` quotient is stored as its complete two-slot
characteristic vector and evaluated by a fixed Boolean-convolution circuit.
It does not represent arbitrary finite relations such as Boolean `LEQ`,
arbitrary typed arity, branching process diagrams, heterogeneous union nodes,
or general relational trace. It has a compact classical Boolean-convolution
equivalent and establishes no compute advantage or physical phase result.

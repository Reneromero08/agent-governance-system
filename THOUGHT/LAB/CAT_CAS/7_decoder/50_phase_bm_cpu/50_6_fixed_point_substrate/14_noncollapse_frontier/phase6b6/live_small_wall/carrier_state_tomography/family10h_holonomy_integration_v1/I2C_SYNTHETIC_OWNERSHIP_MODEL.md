# I2C Synthetic Bidirectional Ownership Models

## Result

```text
I2C_SYNTHETIC_MODELS_COMPLETE
OVERWRITE_MODEL_REJECTED_FOR_H2_AND_R2
REVERSIBLE_PERMUTATION_REFERENCE_PASSES_PROTOCOL_LAWS
PHYSICAL_REVERSIBLE_OWNERSHIP_CARRIER_NOT_ESTABLISHED
I2D_PHYSICAL_REVERSIBILITY_GAP_CONTRACT_NEXT
NO_LIVE_EXECUTION_AUTHORIZED
FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED
FAMILY10H_HOLONOMY_NOT_ESTABLISHED
SMALL_WALL_NOT_CROSSED
```

I2C compares two exact finite models over the I2B partial-overlap geometry. The first is
a many-to-one ownership overwrite model, which is the conservative abstraction of an
ordinary same-value store that sets a line's owner. The second is a hypothetical
reversible owner-permutation model used only as a protocol reference.

Neither model runs Family 10h code, opens PMU events, contacts the target, or provides
physical evidence.

## 1. Shared geometry

The carrier has 24 line coordinates and three owner labels:

```text
H = declared home or reclaim owner
A = remote owner A
B = remote owner B
```

The generator supports are:

```text
S_A = {0, ..., 15}
S_B = {8, ..., 23}
|S_A| = |S_B| = 16
|S_A intersection S_B| = 8
```

Every nontrivial order contrast is required to localize to the eight overlap lines.
Disjoint support is a commuting null.

## 2. Many-to-one overwrite model

The overwrite operations are:

```text
A: set owner to A on S_A
B: set owner to B on S_B
A_reclaim: set owner to H on S_A
B_reclaim: set owner to H on S_B
```

Starting from all-home ownership:

```text
A then B gives B on the overlap
B then A gives A on the overlap
AB versus BA differs on exactly 8 lines
orientation(AB) = +8
orientation(BA) = -8
```

So the model has real order dependence.

However, the forward maps are many-to-one. Two distinct starting states that differ only
inside `S_A` map to the same state after `A`. The reclaim operation fails both global
inverse laws:

```text
A_reclaim(A(s)) != s for some s
A(A_reclaim(s)) != s for some s
```

Despite that failure, these endpoint sequences return the all-home baseline:

```text
A B A_reclaim B_reclaim
B A B_reclaim A_reclaim
```

This is the critical synthetic counterexample:

> Order dependence plus endpoint return does not imply reversible transport or R2
> restoration.

The reclaim operations erase ownership history. They close the endpoint by destruction,
not by inversion.

## 3. Carrier-coupled versus word-recording output

The valid synthetic readout examines the resulting carrier ownership on the overlap:

```text
count(B owners) - count(A owners)
```

It gives:

```text
AB = +8
BA = -8
carrier off = 0
```

An invalid word recorder returns `+8` for an `AB` label sequence even with the carrier
disabled. This control formalizes why an accumulator must couple causally to carrier
state rather than mirror public operation labels.

The overwrite model can therefore produce a carrier-coupled path signal, restore the
public endpoint through erasure, and retain an external result, while still failing H2
and R2 because its transformations are noninjective.

## 4. Hypothetical reversible reference

The reversible reference replaces ownership overwrite with per-line transpositions:

```text
A swaps H <-> A on S_A
B swaps H <-> B on S_B
```

Each generator is a bijection and self-inverse. On the overlap, the two transpositions
generate a noncommuting three-state permutation.

The synthetic commutator word is:

```text
A B A^-1 B^-1
```

with `A^-1 = A` and `B^-1 = B` in this reference.

The model produces:

```text
forward commutator output = +8
reverse commutator output = -8
nontrivial carrier displacement lives exactly on overlap
contractible word = identity
disjoint-support commutator = identity
carrier-off output = 0
```

After reading and decoupling the independent output, applying the inverse commutator
returns the carrier exactly to baseline while the copied output remains `+8`.

This qualifies the desired protocol logic synthetically:

```text
genuine two-sided inverses
matched noncommuting pair
orientation reversal
contractible null
disjoint-support null
carrier-causal output
inverse restoration after output decoupling
```

## 5. What the reversible reference does not prove

The reference assumes a physical operation that permutes three owner states reversibly.
The retained same-value-store evidence establishes ownership-intent transitions, not such
permutations.

Ordinary stores may be:

- idempotent ownership setters;
- many-to-one on hidden coherence metadata;
- dependent on the immediately preceding owner;
- accompanied by replacement, route, timing, or invalidation changes;
- logically byte-preserving while physically irreversible.

Therefore the reversible model is not a proposed hardware backend. It is a target law
that exposes the mechanism gap.

## 6. New Small Wall clarification

I2C sharpens the missing primitive below the full fiber pushforward:

```text
not merely: byte-preserving ownership operation
not merely: order-dependent endpoint response
not merely: endpoint reset plus retained scalar
required: reversible multi-state carrier transformations with causal output coupling
```

The local physical problem is now:

> Can Family 10h expose a receiver-controlled operation that acts as a reversible
> permutation, or sufficiently accurate bijection, on a frozen multi-coordinate carrier
> state rather than as an ownership overwrite?

Without that mechanism, commutator-like order effects can be produced by ordinary
many-to-one state setting and closed by destructive reclaim.

## 7. Exact kill laws established synthetically

```text
endpoint return without injectivity is not restoration
home reclaim without two-sided inverse is not inverse
word-label output surviving carrier-off is invalid
disjoint-support commutator must be identity
contractible word must be identity
inverse commutator must restore after output decoupling
logical-byte return is not full state equivalence
```

These are protocol and model results. They do not establish physical transport.

## 8. I2C decision

```text
overwrite physical candidate promotable = false
reversible reference physical candidate promotable = false
synthetic protocol laws qualified = true
```

The next gate is:

```text
I2D_PHYSICAL_REVERSIBILITY_GAP_CONTRACT
```

I2D must define the minimum observables and interventions capable of distinguishing a
reversible carrier permutation from ownership overwrite on real hardware. It must remain
non-executing until a new prospective package and authority are separately frozen.

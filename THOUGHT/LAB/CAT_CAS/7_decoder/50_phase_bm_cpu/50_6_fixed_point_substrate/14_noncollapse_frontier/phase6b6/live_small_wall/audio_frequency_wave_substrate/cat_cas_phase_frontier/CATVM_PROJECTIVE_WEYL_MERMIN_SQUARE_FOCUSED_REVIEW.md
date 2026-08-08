# M250 focused independent review

Decision: `PASS_STRICT_SCOPE`

Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `SEPARATE_REFERENCE_PARITY`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

Scientific source head: `33de34c74e94e17c6d7c4a2cfd92ba198314ce27`

The focused read-only review independently reconstructed the exact two-qubit
projective Weyl law over `Q(i)`,

```text
(q,x,z) o (q',x',z')
  = (q+q'+2 z.x' mod 4, x xor x', z xor z'),
```

and the six public Mermin-square context closures.  All nine observable ports
have exactly two typed context consumers.  Every context is commuting and
central; five close to `+I`, the third column closes to `-I`, and their total
central exponent is exactly two.  Removing the projective cocycle changes the
declared boundary to `+I` without assignment enumeration.

The separate oracle uses independently implemented exact `4 x 4` Gaussian-
rational matrices and an independent binary-symplectic word representation.
It confirms the base and globally Hadamard-conjugated/reordered variants,
the state-independent `-I` operator product, and the same result against the
dephased identity state.  This is a formal exact software contextuality
calibration, not a physical contextuality experiment.

The abstract Unix-socket CATVM keeps the actual four-cell `Q(i)` carrier, the
four-cell scratch, six context accumulators, and nine port-consumer receipts
inside the backend.  Only the final central phase is released, after all 18
Pauli actions have been inverted in dependency order and exact canonical
restoration has been verified.  The same backing is reused at generation two
by the descriptor-distinct conjugated program; fresh/restored boundaries and
resource signatures agree and no baseline reload occurs.  Disconnect,
partial-forward, post-projection, custody, premature-projection, missing,
wrong, and applicable reordered-inverse controls pass without exposing hidden
carrier or context values.

The initial package overstated its classical comparator by calling the
18-composition binary-symplectic recurrence strongest.  The repaired result
recognizes the stronger declared-family baseline: validate one of the two
public variants and return the fixed Mermin parity/cocycle invariant in `O(1)`
work.  The 18-composition recurrence remains the strongest transferable
descriptor-level comparator.  Both are smaller than the carrier/CATVM path,
so no computational advantage or distinct software-unavailable phase resource
is established.

The strict result is limited to one bounded two-qubit software Mermin-square
family.  It does not establish physical contextuality, a general contextuality
resource theorem, general relational closure, catalytic inference, Small Wall
crossing, physical waveform execution, physical-bit replacement, or unbounded
catalytic computation.  The route is retired at this ceiling rather than
extended with larger contextual sets.

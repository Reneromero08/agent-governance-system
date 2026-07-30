# Independent review: deferred exact unit-ledger stream

Base head:

`91523133f6a26b4c20fce72d0388f8bff2722660`

Reviewed production source SHA-256:

`aa632b5a515b8d12b2f1a68a07d5b8034b8aa416ffd81ac51a116656a2ba9753`

Reviewed oracle source SHA-256:

`0e3d7f7eaadc4fb6da494e0c3bf1ed25c1a3312cfc6dcb3517b04fa916fdb137`

Reviewed full production output SHA-256:

`f2351b230d7af4770fdc6cb14eae674ec9a3bd3b9c13cdfef6bdffef16d45f19`

Reviewed full oracle output SHA-256:

`3d25b5646edee3b9fd72a94ac45ee4020811fdef5d150904608d625e03ef5d56`

Classification:

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level:

`SEPARATE_REFERENCE_PARITY`

Restoration class:

`EXACT_ALGEBRAIC_RESTORATION`

No concrete defect was found within the bounded package scope.

The review independently checked the representation signs: a net unit-ledger
change `delta` is paired with one residual action by `U(-delta)`. Relative
addition correctly carries the left ledger and acts on the right residual by
`U(right-left)`. Projection-first scalar materialization is exact because the
unit action is a common ring scalar and the boundary projection is linear.

The separate reference path matched every forward, inverse, and exact resource
tuple for both families at periods 1 and 64. It also reproduced the declared
49-direction local certification, at-most-one net residual action per balance
call, absence of per-accepted-move vector materialization, exact restoration,
same carrier backing, and period-1 cross-family restored-carrier reuse.

The period-64 separately observed named maxima sums improve the M110 values but
remain above the raw compact recurrence payload. The strongest classical
baseline is the identical deferred-ledger implementation itself; the raw
recurrence and M110 are secondary comparisons. No computational advantage is
established.

The strict claim ceiling is Linux x86-64 Python software, two fixed public F17
period-17 families, periods 1 and 64, exact 49-direction search, at most one
net residual action per balance call, relative-ledger addition, streamed scalar
projection, exact boundaries and restoration, component-level accounting, and
period-1 cross-family reuse.

This does not establish global unit optimality, a whole-process peak, fixed
total footprint, an asymptotic height bound, period-64 cross-family reuse,
computational advantage, a distinct phase resource, CATVM custody, physical
waveform execution, or replacement of physical bits with pi.

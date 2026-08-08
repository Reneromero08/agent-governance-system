# M251 focused independent review

Decision: `PASS_STRICT_SCOPE`

Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `SEPARATE_REFERENCE_PARITY`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

Scientific source head: `912f1ed663704e76a0a97789e869abf98bddc7ba`

The focused read-only review independently reconstructed exact arithmetic in
`Q[zeta_8]/(zeta_8^4+1)`, the public two-data-qubit word matrix, the full
`8 x 8` density evolution, and the DQC1 block identity

```text
rho_1 = (1/8) [[I, U^dagger], [U, I]],
<X> + i<Y> = Tr(U)/4.
```

For the declared primary and unrelated reuse words, the independently derived
normalized traces match the backend exactly.  The verifier also confirms that
the data marginal remains maximally mixed while the selected forward joint
states are nonproduct.  A valid scalar word is separately accepted with a
factorized joint state and trace one, so correlation is evidence for the
selected fixtures rather than an undeclared restriction on the public grammar.

The abstract Unix-socket CATVM keeps one actual 64-cell exact density backing
and a four-cell scratch backing inside the backend.  Only the declared exact
normalized-trace boundary is returned by accepted transactions.  The two
pre-inverse diagnostic predicates are not present in socket responses; they
are reconstructed only by the separate verifier.  The final scalar survives
while every public controlled gate is inverted on the same density backing,
after which exact canonical restoration is checked and the backing is reused
at generation two.  Fresh/restored boundaries and resource signatures agree,
and no baseline reload occurs.

Disconnect, partial-forward, post-projection, premature projection, dirty
scratch, missing inverse, completed wrong inverse, applicable reordered
inverse, descriptor mutation, stale generation, protocol, and no-smuggle
controls pass.  Nonempty string carrier and transaction identifiers are
validated before lease, transaction setup is inside the rollback guard, and a
numeric transaction identifier is rejected without poisoning the carrier.

The service reports the shared 96-cell public gate library, the 16 accepted
plan references, density and scratch cells, forward/inverse multiply terms and
writes, boundary work, protocol traffic, restoration, and reuse.  Exact
coordinate-payload and whole-process peaks remain explicitly uninstrumented;
field-cell counts are not presented as fixed-bit-payload claims.  The direct
four-by-four comparator is described as 16 resident matrix cells plus declared
transient multiplication scratch, not as a complete 16-cell peak.

The strongest fixed-fixture comparator validates the public word and returns
the frozen exact normalized trace in `O(1)` work.  The strongest transferable
descriptor-level comparator directly evolves one exact `4 x 4` public-word
matrix and takes its trace without CATVM restoration.  Both are smaller than
the 64-cell density/CATVM path, so the result establishes no computational,
space, work, or query advantage and no distinct software-unavailable phase
resource.

The strict result is limited to one clean control, two maximally mixed data
qubits, and public words of length at most eight in this exact software CATVM.
It does not establish physical mixed-state execution, DQC1 hardness, oracle or
query separation, general catalytic inference, Small Wall crossing, physical
waveform execution, physical-bit replacement, or unbounded catalytic
computation.  The route is retired at this bounded calibration rather than
extended with more qubits or longer words.

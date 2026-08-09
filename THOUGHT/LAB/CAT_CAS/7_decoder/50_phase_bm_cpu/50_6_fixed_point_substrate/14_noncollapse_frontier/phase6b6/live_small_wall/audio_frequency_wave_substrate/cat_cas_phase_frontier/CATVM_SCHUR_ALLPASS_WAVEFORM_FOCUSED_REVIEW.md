# M256 focused adversarial review

Disposition: `PASS_STRICT_SCOPE`

Reviewed the isolated CATVM service, public controller, independent
`Q[z]/(z^4+1)` oracle, qualifier, and regenerated seals.  The review checked:

- exact Schur forward and divisibility-gated inverse laws;
- actual polynomial-derived degrees, including denominator degree zero for the
  feedback-disabled sham;
- the formal Laurent all-pass identity and scalar/polynomial boundary parity;
- restoration-before-response, partial/post-projection rollback, disconnect
  handling, exact same-backing restoration, generation-2 reuse, and no reload;
- full descriptor/type/owner/controller/consumer/generation custody and
  rejection of coefficient, seed, scratch, snapshot, and debug projections;
- abstract operation counts and their explicit Python/Fraction/RSS caveats;
- the one-field-scalar-plus-winding boundary baseline and the identical
  two-polynomial full-waveform baseline; and
- the strict denial of a distinct phase resource, computational advantage,
  physical waveform execution, Small Wall crossing, physical bit replacement,
  or unbounded catalytic computation.

The independent oracle also checks 64 declared-domain coefficient words for
exact scalar/polynomial/inverse parity.  No residual claim-breaking defect was
found.  This review supports retiring the bounded all-pass route after M256;
it does not authorize larger filter fixtures or a stronger resource claim.

# M242 Focused Review

## Decision

`PASS_STRICT_SCOPE`

Verification classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `SEPARATE_REFERENCE_PARITY`

Restoration classification: `EXACT_ALGEBRAIC_RESTORATION`

## Audited claim

The declared separable `F5` linear-character oracle is represented exactly by
`n` resident five-cell `Q(zeta5)` factors rather than a materialized `5^n`
global amplitude vector. Across dimensions `1,2,4,8,16,32`, one abstract
coherent forward query reaches the exact final secret boundary, the actual
factor and scratch backings are restored by the exact inverse before response,
and a descriptor-distinct generation-two program consumes those restored
backings.

The review confirmed the strict ceilings: each oracle call reads all `n`
private residues and visits `5n` factor cells; the strongest total software
baseline directly scans the private descriptor in `O(n)` work. The result is
therefore a carrier-representation repair and bounded query-model calibration,
not total computational advantage, Small Wall crossing, a nonseparable phase
resource, physical waveform execution, or unbounded catalytic computation.

## Independent evidence

The standalone reference implements its own polynomial-basis arithmetic for
`Q(zeta5)`, exact factor Fourier/oracle/inverse evolution, black-box query
lower-bound certificates, and the direct private-descriptor scan. For
dimensions `1,2,4`, it also materializes a verifier-only dense state and
checks the hidden factor tensor product against that exact dense state. It
imports neither the service, controller, predecessor, nor production field
implementation.

Two fresh private-secret qualification runs produced byte-identical sanitized
raw, reference, and final seals. Durable evidence contains no inferred secret,
factor values, secret-dependent commitment, or final-state commitment.

## Atomicity and controls

The accepted service executes:

```text
forward factor Fourier
-> hidden phase-factor oracle
-> inverse factor Fourier
-> internal final boundary
-> strict exact inverse sequence
-> canonical same-backing verification
-> generation advance and release
-> response
```

Disconnect and post-projection exception paths restore before any response.
A partial-oracle exception after 17 completed factors reverses that exact
completed prefix, then reverses the prior Fourier stage and releases only the
rejection. Wrong, missing, and reordered inverse controls fail exact
restoration. Typed descriptor, owner, generation, stale-generation,
transaction, projection, dense-vector, null-carrier, and snapshot controls all
discriminate. All-zero, repeated, mixed, and one-coordinate-perturbed secrets
also pass the exact independent algebra checks.

## Review repairs

The first read-only review found two evidence defects:

1. The service's imported exact `Q(zeta5)` arithmetic source was not included
   in `source_dependencies`.
2. Private configuration was described as backend-only even though the
   separate verifier lawfully receives a second stdin copy.

The qualifier now hashes the actual imported arithmetic source and pins that
hash through the final seal. The result now records exactly two stdin
recipients, `CATVM_SERVICE` and `SEPARATE_REFERENCE`, while explicitly denying
controller receipt. The focused post-repair review confirmed both seals and
returned `PASS_STRICT_SCOPE` with no remaining concrete defect.

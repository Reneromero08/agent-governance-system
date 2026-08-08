# M241 focused review

Disposition: `PASS_STRICT_SCOPE`

Verification classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `SEPARATE_REFERENCE_PARITY`

Restoration classification: `EXACT_ALGEBRAIC_RESTORATION`

## Verified scope

The package implements exact `Q(zeta5)` Bernstein--Vazirani-style phase-oracle transactions for declared dimensions 1 through 4 behind a separate Unix-domain service. The private, nonzero oracle configurations are generated at qualification time, delivered to the backend through stdin, and are absent from controller source and durable evidence. The controller imports or loads no backend module.

Each accepted transaction performs the forward Fourier/oracle/inverse-Fourier program, retains only the final secret boundary internally, applies the public topology-derived actual inverse on the same amplitude and scratch backings, verifies exact canonical restoration, advances the restoration generation, and only then returns the final secret. A descriptor-distinct second oracle consumes the actual restored carrier at generation 2. No snapshot or baseline reload is used.

The service uses an abstract Unix socket and disables process dumpability before reading its private configuration. Status responses expose canonicality, lease state, and restoration generation only. The earlier enumerable unkeyed secret commitment was rejected during adversarial review and removed. No pre-run answer-bearing receipt remains.

The exception path is also transactional at the tested boundary. A forced failure after final-boundary projection invokes the formula-derived reverse program, verifies and releases the exact carrier, and only then returns `REJECTED`. A client disconnect before response likewise leaves the carrier exactly restored. Negative delay configuration is rejected before lease.

The standalone oracle independently implements the cyclotomic power-basis arithmetic, coherent query, inverse, final boundary, restoration, and classical basis-query recovery. It receives the private configuration as a verification authority, not as the controller. Exact service/reference parity is checked before secret-dependent fields are removed from the seals. Repeated qualifications with fresh secrets produce byte-identical sanitized evidence.

## Query and resource result

One abstract coherent forward phase query recovers `n` residues for each declared dimension. In the stated black-box model, an exact deterministic classical value or phase query returns one `F5` symbol, `n` basis queries suffice, and fewer than `n` queries leave a nonzero orthogonal secret difference. This is a bounded query-model separation only.

The software phase path acts on all `5^n` amplitude cells and retains an equal-sized scratch backing. The declared carrier sizes are 5, 25, 125, and 625 field cells. Exact Fourier work, oracle cell visits, projection, inverse, restoration verification, protocol bytes, private-configuration traffic, snapshot-copy baseline cells, and reuse are recorded. Resource figures remain `PACKAGE_SELF_REVIEW`; whole-process RSS, allocator, socket-kernel, and scheduler costs are explicitly incomplete.

## Claim ceiling

The machine-enforced claim applies to this declared service/controller protocol and bounded oracle family. It does not claim protection against an arbitrary same-UID attacker outside that protocol, general oracle advantage, total software advantage, general inference or learning, Small Wall crossing, unbounded catalytic computation, physical waveform execution, or replacement of physical bits with pi.

The final secret is the permitted final boundary. Resident phase amplitudes, private oracle values before the transaction, and intermediate state are not serialized in committed evidence. The durable raw/reference views retain only structural counts and parity booleans.

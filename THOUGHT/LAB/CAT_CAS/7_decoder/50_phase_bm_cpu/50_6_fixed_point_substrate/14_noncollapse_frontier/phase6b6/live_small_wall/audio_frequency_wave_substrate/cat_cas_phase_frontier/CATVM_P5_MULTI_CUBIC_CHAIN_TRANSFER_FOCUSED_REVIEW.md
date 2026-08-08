# M244 focused adversarial review

Decision: `PASS_STRICT_SCOPE`

Verification classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `SEPARATE_REFERENCE_PARITY`

Restoration classification: `EXACT_ALGEBRAIC_RESTORATION`

## Reviewed scope

The review inspected the atomic service, public controller, standalone
Q(zeta5) oracle, qualifier, and regenerated seals. It checked:

- exact five-state transfer and inverse execution at depths 2, 3, 4, 8, 16,
  32, and 64;
- hidden inter-module couplings, the public fixed first coupling, and selected
  output-index custody;
- response ordering, disconnect recovery, injected-failure restoration,
  same-backing reuse, and restoration generations;
- controller exclusion and absence of secret-dependent intermediate values or
  payload measurements in responses and sealed evidence;
- independent dense-path parity through depth 4 and independent transfer
  parity at every declared depth;
- the endpoint-specialized, forward-only compact classical recurrence; and
- the cross-rank-two modular certificate and strict claim ceiling.

## Repairs required by review

The first pass found that the service returned intermediate-dependent payload
width and cancellation metrics. These fields were removed from the machine
response and durable evidence. The controller now records only the released
final-boundary payload and public conservative bounds that cover the phase
message, scratch, and retained boundary.

The original descriptor retained a noncausal first coupling for the declared
e0 input and omitted the selected output index from descriptor accounting.
The accepted source fixes the first coupling publicly to one, retains only the
`k-1` causal hidden inter-module couplings, counts the output index, and pins
the resulting `3k`, `5k-1`, and `9k-2` storage and read laws.

The service now checks `PR_SET_DUMPABLE`, closes its private configuration
stdin after one delivery, and includes the hidden output index in canonical
restoration equality. The strongest implemented comparator now specializes
both fixed endpoints: five terms form the first five-vector, each interior
module uses 25 terms, and the selected final boundary uses five terms.

## Accepted result

The service executes an exact connected treewidth-one multi-cubic transfer on
one actual five-cell phase message and one fixed scratch backing. It projects
only the selected final amplitude, reverses every transfer on the same
backings, verifies exact canonical state including hidden descriptor fields,
and releases the response only after restoration. The restored carrier is
then reused at the next generation and agrees with a fresh execution.

The standalone arithmetic implementation agrees at every declared depth; its
independent endpoint-specialized recurrence agrees with the selected boundary
using `25k-40` forward character terms. A separate rank-two kernel has modular
rank 25 over both declared split primes, so the five-cell treewidth-one
interface does not transfer unchanged to that rank-two topology.

This is a bounded exact software result. It does not establish fixed-width
exact state, an arbitrary-topology five-cell closure, a distinct phase
resource, computational advantage, a Small Wall crossing, physical waveform
execution, physical-bit replacement, general inference, or unbounded
catalytic computation.

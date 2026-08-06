# CATVM Phase Backend Transaction

## Status

This package is the first bounded CATVM proof substrate on the branch-native
Boolean/F3 phase-relational backend. It establishes an enforced Linux
userspace custody boundary for one open-intermediate composition transaction.

Accepted claim:

```text
CATVM_OPEN_INTERMEDIATE_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND
```

Separate, weaker comparison claim:

```text
CATVM_SNAPSHOT_BACKED_TRANSACTIONAL_REUSE_ESTABLISHED
```

The accepted path does not use the snapshot path.

## Machine boundary

`catvm_phase_service` owns the only carrier mapping. The controller receives a
Unix-domain `SOCK_SEQPACKET` endpoint and the service PID, but it is not linked
to the phase core and never receives a carrier cell or an intermediate
relation.

Before allocating the carrier, the service requires:

```text
RLIMIT_CORE = 0
PR_SET_DUMPABLE = 0
PR_SET_PTRACER = 0
PR_SET_NO_NEW_PRIVS = 1
```

The private anonymous carrier mapping must be locked and marked
`MADV_DONTDUMP` and `MADV_DONTFORK`. After accepting its one controller, the
service unlinks the socket and installs a seccomp allowlist. The accepted
build cannot open files, create another socket, fork, execute, use shared
memory, or invoke process-inspection facilities after custody begins.

This boundary is tested against the ordinary same-UID controller. It does not
claim secrecy from the host root user, a hostile kernel, binary replacement,
or microarchitectural observation.

## Native transaction

The fixed typed carrier contains six four-cell Boolean/F3 relations:

```text
A, U, B, Y, Z, final-boundary
```

The public program supplies three input relations:

```text
A(x,u), B(u,y), C(x,y)
```

The service executes:

```text
seal A, B, C
F: Y = compose(A, B)
G: Z = intersect(Y, C)
project final Z
G^-1 on the actual resident Y and C
F^-1 on the actual resident A and B
unseal A, B, C
verify restoration
reuse the same restored carrier for a different program
```

`F` is the branch-native exact Boolean/F3 existential-composition polynomial.
`G` is the branch-native exact Boolean/F3 intersection polynomial. `G`
consumes the four physical phase cells populated by `F`; there is no decode,
serialization, coefficient copy, witness list, candidate set, truth table, or
scalar re-evaluation between them.

Only `PROJECT Z` is legal. `PROJECT Y` is rejected with a fixed
`E_INTERMEDIATE_PROJECTION_DENIED` response without calling a decode routine.
The accepted result is sent before inverse execution, so it survives outside
the history being reversed.

## Controller protocol

The service accepts a strict ASCII command language over one packet per
request and one packet per response:

```text
HELLO
PING
SEAL <12 ternary coefficients>
F
G
PROJECT Z
RESTORE
SHUTDOWN
```

The only permitted phase-bearing response is the final four-coefficient
boundary relation. Unknown commands, embedded NUL bytes, oversize packets,
debug requests, state-detail requests, carrier reads, dumps, snapshots, and
intermediate projection receive fixed error objects.

No response contains carrier phases, intermediate coefficients, intermediate
hashes or commitments, addresses, witnesses, candidate sets, decoded
relations, or retained inverse factors.

## Canonical state and restoration

The carrier has 24 logical complex cells and retains baseline and working
rails, for 48 resident complex values and 768 logical bytes. The Linux mapping
and lock account for 4096 bytes.

Discrete machine state is compared exactly:

```text
topology digest
baseline digest
lease identity
carrier creation count
backend kind
program erasure
machine cursor
open-port state
morphism stack
pending operations
backend queue
snapshot absence on the accepted path
restoration generation transition
```

Complex working cells are compared to the canonical baseline with the
predeclared absolute tolerance:

```text
2e-12
```

Final phase-root decoding uses the separately predeclared bound:

```text
4e-10
```

Inverse factors are recomputed from the actual resident operands and wiped
after use. The accepted path retains no inverse history and performs no
snapshot capture or reload.

## Controls

The qualifier requires:

```text
wrong G inverse
missing G inverse
reordered inverse on the prospectively noncommuting F/G transaction
attempted Y projection
null carrier
snapshot-backed comparison
same-carrier unrelated reuse
1000 alternating reuse cycles
same-UID process-access attacks while Y is resident
strict protocol adversaries
zero service stdout and stderr
controller/core linkage separation
native series-parallel semantic comparison
```

The wrong, missing, and reordered controls are startup-selected test builds;
the accepted protocol exposes no control switch.

## Reproduction

From this directory:

```bash
evidence_dir=$(mktemp -d /tmp/catvm-qual.XXXXXX)
./qualify_catvm_phase.sh . "$evidence_dir" >"$evidence_dir.summary.json"
```

The qualifier performs strict GCC builds, static analysis, AddressSanitizer
and UndefinedBehaviorSanitizer runs, deterministic replay, all controls,
resource checks, native-backend semantic comparison, no-smuggle gates, and
warmed direct, inert-boundary, snapshot, and in-place measurements.

## Claim boundary

This result establishes a bounded software machine law on the tested Linux
userspace boundary. It does not establish arbitrary relational topology,
compact wide-interface relations, general holographic relational computation,
advantage over a compact classical method, physical waveform computation,
silicon execution, Small Wall crossing, or unlimited catalytic computation.

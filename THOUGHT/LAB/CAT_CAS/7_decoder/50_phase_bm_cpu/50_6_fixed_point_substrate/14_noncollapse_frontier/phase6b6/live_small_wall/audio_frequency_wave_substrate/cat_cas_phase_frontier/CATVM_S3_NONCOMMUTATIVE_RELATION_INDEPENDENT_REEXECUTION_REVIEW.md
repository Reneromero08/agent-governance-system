# CATVM S3 noncommutative relation independent reexecution review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Accepted in-place restoration class: `EXACT_ALGEBRAIC_RESTORATION`

Snapshot comparison restoration class: `SNAPSHOT_RELOAD`

## Enforced transaction boundary

The accepted path places the exact F103 S3 translation-invariant two-port
carrier in a separate Linux binary-pipe service. The service sets itself
non-dumpable and enables `no_new_privs`. Under the tested same-UID Linux
configuration, both the production controller and the independent controller
receive only `EACCES` or `EPERM` when opening `/proc/<pid>/mem`. This is a
bounded userspace custody result, not OS or hardware isolation.

Each accepted response follows the observed service order

`FORWARD_BEGIN -> BOUNDARY_RETAINED_INTERNAL -> RESTORATION_VERIFIED -> RESPONSE_WRITE_ATTEMPT`.

The final scalar boundary and a truncated 64-bit one-way commitment are the
only answer-bearing response fields. The commitment is not a collision-free
identity proof. No six-cell hidden port is emitted in responses, stderr, or
the event-vocabulary-only audit. The controller imports only the fixed binary
protocol; it neither imports the service nor computes the boundary.

The service source is part of the logical CATVM boundary. Inspection finds no
answer table, decoded relation table, retained compiled plan, or inverse-value
history. Public program descriptors generate each forward and reverse step
without consulting a boundary answer.

## Independent semantic reconstruction

The independent oracle imports none of the protocol, service, controller, or
NumPy. It reconstructs S3, the group law, the compact six-coordinate
recurrence, and the full two-register 72-cell relation semantics. It checks
all 36 group products against full relation-matrix multiplication. At primary
depth 1 and unrelated alternate depth 256, the compact recurrence and full
relations reproduce the production boundary and commitment, and both
representations clear exactly under the actual reverse sequence.

The oracle separately drives a fresh service with its own raw `struct` binary
messages. The primary transaction and same-process unrelated reuse match the
independent recurrence. Reuse reaches restoration generation two and matches
a fresh carrier while retaining the service process and carrier backing. No
snapshot is created or loaded on the accepted path.

## Adversarial ordering and custody controls

Hidden projection, early response, wrong type, and wrong owner are attempted
after forward residency. Each is denied without releasing the boundary, the
correct inverse restores the same carrier, and only then is the denial
response written. Null-carrier and wrong-generation requests are denied.

Missing, wrong, and reordered noncommuting inverses execute after forward
residency, fail exact restoration, and terminate without response or stderr.
An independent compact depth-64 calculation confirms that the reordered
inverse is not restorative. Disconnect before response still produces exact
restoration before the failed write attempt. Snapshot commands are rejected
by the in-place service, and in-place commands are rejected by the snapshot
service.

The snapshot sham retains a separate twelve-cell copy, reloads it before its
response, and is classified only as `SNAPSHOT_RELOAD`. It is not evidence for
the accepted in-place restoration law.

## State law, resources, and obstruction

Exact canonical restoration covers the twelve F103 carrier cells, backing
identity, owner/type records, idle stage, pending-operation state, nonce, and
service mode. After equality is established, the declared restoration and
lease generations advance monotonically. The unrelated second program uses
those new lease generations on the same resident carrier.

The accepted logical peak is conservatively counted as 37 field-value slots:
twelve carrier cells, a temporary twelve-cell restoration-verification
baseline, one streamed six-cell public operand, one six-cell operation delta,
and one scalar. The strongest matched direct classical recurrence uses the
identical streamed six-coordinate S3 group law and has a 25-slot logical peak.
The snapshot sham also reaches 37 slots but includes a retained twelve-cell
snapshot. Requests are 24 bytes and responses are 44 bytes. These counts
exclude Python object headers, allocator metadata, interpreter/native-library
state, and whole-process peaks; no physical-memory advantage is claimed. The
72-cell full relation representation exists only in the independent oracle.

This milestone enforces the previously verified noncommutative algebra behind
a minimal machine boundary. It does not establish a general finite-group
compiler, general six-label relations, a distinct phase resource,
computational advantage, Small Wall crossing, physical waveform or silicon
execution, replacement of physical bits with pi, or unbounded catalytic
computation.

# CATVM Paley relation rematerialization independent reexecution review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Bounded scheduler result

The service derives a clean reversible toggle schedule from the public edges of
the exact two-leaf, seven-internal-node DAG.  It does not inspect relation
values or the final boundary while compiling.  Independent exhaustive search
finds no clean final-only sink state at capacities one through five after
exploring respectively 3, 4, 12, 19, and 101 reachable states.  Capacity six
first succeeds with the 15-toggle node sequence
`2,3,4,5,2,6,7,8,7,2,6,4,5,2,3`.

The durable oracle records every slot lease, epoch, live interval, release
point, and the forward reconstruction of node 2.  The sink is the only live
internal relation at the boundary.  The inverse is the exact reverse of the
public toggle and lease plan.  This minimum is only for this topology under
the declared local reversible-pebble law: sources remain resident, a node may
be toggled only while both parents are resident, and the forward endpoint must
contain only the sink.  It is not a generic DAG-scheduling theorem.

## Machine boundary, custody, and controls

The controller starts a separate same-UID Linux userspace service over fixed
binary pipes.  It imports only the protocol and performs no relation algebra
or boundary calculation.  The service owns the two sealed leaves, six hidden
relation slots, node and epoch ownership, pending-action state, public
descriptor state, restoration generation, and protocol nonce.  It retains the
boundary internally, reverses and verifies the exact canonical state and
backing identities, increments the generation, and only then attempts the
response write.  The unrelated second operation assignment consumes the same
restored backing and agrees with a fresh service.

Hidden projection, null-carrier, snapshot, and wrong-generation requests are
denied without a valid boundary.  Missing inverse, wrong inverse, and a
dependency-violating reordered inverse are each exercised after forward
execution and internal boundary retention; restoration fails, the service
exits 23, and no response or stderr is emitted.  Closing the response pipe
before execution still leaves an audit order in which restoration verification
precedes the disconnected write attempt.  Audit records contain only stage
names and generations, not relation values.

These observations establish the declared protocol custody and response
ordering.  They do not establish different-UID, namespace, kernel, VM, or
hardware isolation.

## Independent semantics and matched baselines

The oracle imports neither controller nor service and uses no NumPy.  It
independently searches the public pebble state graph, executes the resulting
leases using full 17-value C17 relations, and exactly reverse-clears its
136-cell carrier.  Both operation assignments produce boundaries 80 and 5,
matching the CATVM responses and restored reuse.

The accepted compact carrier falls from the retain-all 27 F103 cells to 24:
six sealed-leaf cells plus six three-coordinate internal slots.  It retains no
dynamic phase-value inverse history, but it does retain 15 public compiled-plan
records plus six node and six epoch ownership records.  A full transaction
performs 30 relation evaluations.

The matched reversible classical pebbler has the identical 24-cell and
30-evaluation law.  More strongly, an executed occurrence-expanded compact
classical recurrence reaches both boundaries with 13 relation evaluations and
three temporary relation slots, or 15 total field cells including the two
sealed leaves.  This accounting treats a three-coordinate relation as the
atomic stored value and excludes operation-local scalar temporaries, Python
allocator state, and interpreter overhead for every compact path.  Retain-all
uses 27 cells and seven evaluations.  No dense 17-by-17 table is used.

## Claim ceiling and obstruction

The result establishes a topology-derived live/release/reconstruction schedule,
one-slot reduction from retain-all storage, atomic final-only response, exact
algebraic restoration, and restored-carrier reuse for the exact declared DAG
and its two operation assignments.

It does not establish a generic scheduler, arbitrary relational programs,
OS/hardware isolation, a distinct phase resource, computational advantage,
Small Wall crossing, physical execution, replacement of physical bits with
pi, or unbounded catalytic computation.  The measured obstruction is that
reversible rematerialization remains identical to classical reversible
pebbling and is larger and more evaluation-intensive than the matched compact
occurrence-expanded recurrence.

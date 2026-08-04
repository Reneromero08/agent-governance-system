# CATVM scheduled Paley relation DAG independent reexecution review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Machine boundary and transaction order

The controller launches a distinct Linux userspace service over fixed binary
stdin/stdout pipes. It imports only the wire protocol, never imports or loads
backend code, and contains no relation arithmetic or independent boundary
calculation. The service owns a 27-cell carrier: two sealed three-coordinate
Paley relation leaves and seven hidden three-coordinate internal nodes.

For each request the backend compiles a topological schedule from the public
nine-node descriptor, executes seven native composition/intersection nodes,
retains the scalar boundary internally, reverses the schedule, verifies the
exact carrier state and backing, increments restoration generation, and only
then writes the response. Audit events contain stage names and generations but
no relation values. Both accepted responses have restoration and boundary
flags; restored-carrier reuse reaches generation two and matches a fresh
service on the unrelated second operation assignment.

## Controls

Hidden projection, null-carrier, snapshot, and wrong-generation requests are
denied without a boundary. Missing, wrong, and reordered inverse commands
terminate their isolated services with no response and no stderr. A later
valid transaction after the nonterminal denials restores normally. In the
disconnect attack, the controller closes its response pipe before execution;
the backend audit records exact restoration verification before its response
write attempt and exits without stderr.

These controls establish the declared protocol boundary and atomic response
ordering. They do not establish different-UID, namespace, kernel, VM, or
hardware isolation.

## Independent semantic oracle

The oracle imports neither service, controller, nor NumPy. It reconstructs
both public DAG operation assignments in full 17-value C17 relations for all
nine nodes. Both topologies have shared fanout, including fanout three at node
3. The independently computed final boundaries are 80 and 5, matching the
service primary and restored-reuse responses. Reverse clearing restores all
153 full-relation cells exactly.

## Resources and ceiling

The accepted carrier contains 27 F103 cells, of which 21 are hidden internal
cells. Requests are 24 bytes and responses are 44 bytes. The compiled schedule
has seven internal nodes; retained inverse history and snapshot traffic are
zero. The 153-cell full relations are oracle-only, and no dense 17-by-17 table
is materialized.

The claim is limited to the exact two-leaf, seven-internal-node topology, two
declared operation assignments, one controller, and one separate same-user
Linux pipe service. It is not a generic DAG scheduler or OS/hardware isolation
claim. It establishes no distinct phase resource, computational advantage,
Small Wall crossing, physical execution, physical bit replacement, or
unbounded catalytic computation.

# I2A Compile-Only Post-Source Operator Runtime

## Result

```text
I2A_COMPILE_ONLY_POST_SOURCE_RUNTIME_INTERFACE_ESTABLISHED
PHYSICAL_BACKEND_NOT_IMPLEMENTED
REMOTE_STORE_SAME_VALUE_EXTRACTION_TARGET_ONLY
SECOND_GENERATOR_UNRESOLVED
INVERSE_UNRESOLVED
H1_NOT_PASSED
NO_LIVE_EXECUTION_AUTHORIZED
```

I2 showed that the retained runtime lacks an admissible receiver-side generator pair.
I2A therefore builds only the operation boundary needed for a future test. It does not
copy the legacy live worker into a new authority path and does not implement a Family 10h
backend.

## 1. Lifecycle

The C interface enforces this state order:

```text
ALLOCATED
-> SOURCE_PREPARED
-> SOURCE_DEAD_SEALED
-> RECEIVER_WORD_OPEN
-> RECEIVER_WORD_CLOSED
-> RESTORATION_RECORDED
-> DESTROYED
```

A receiver word cannot open before the source-death seal. The seal requires:

```text
positive source PID
waitpid exited zero
source not alive
zero open source IPC
zero source helpers
matching preparation nonce
fresh seal nonce
no challenge selected before source death
```

The challenge nonce must be generated after the seal and must differ from both the
preparation and seal nonces.

## 2. Persistent carrier boundary

The runtime owns one public `carrier_id` from allocation through restoration recording.
Operators are applied to a backend associated with that same runtime object. The carrier
cannot be destroyed until the word closes and one restoration observation is recorded.

This fixes the current base tomography lifecycle, which allocates, queries once, and
unmaps one carrier per schedule row.

The interface does not prove the underlying physical object remained equivalent. The
future backend must bind object identity, memory allocation, page identity, route,
process custody, and state tomography.

## 3. Admitted extraction target

The only operator kind accepted by the current interface is:

```text
F10HI_OP_REMOTE_STORE_SAME_VALUE_EXTRACTION_TARGET
```

Its specification includes:

```text
operator instance ID
line-set ID
route ID
executor core
amplitude
operation budget
byte-preservation requirement
inverse-of instance ID
```

The current admission law requires `inverse_of_instance_id = 0`. This prevents a caller
from declaring an inverse before a two-sided inverse law exists.

The accepted operator kind is an extraction target, not an established generator. A
backend must still prove that the intended same-value store executed on the declared
core, route, line set, and amplitude and produced a bound physical receipt.

## 4. Rejected operation roles

The interface rejects from receiver words:

```text
QUERY_PROBE
DESTRUCTIVE_RESET
UNRESOLVED_SECOND_GENERATOR
UNQUALIFIED_INVERSE
```

This prevents four common promotions by naming:

- an endpoint query is not a state-transform generator;
- a flush or re-preparation is not an inverse;
- an unresolved second operation cannot complete a generator pair;
- an operation cannot become an inverse merely by filling an `inverse_of` field.

## 5. Backend boundary

The interface uses an explicit backend callback. Every backend receipt must report:

```text
state token before
state token after
backend receipt ID
logical-byte preservation
whether the backend is synthetic
```

The bundled backend exists only inside the self-test. It changes a synthetic token and
labels every receipt `synthetic_backend = true`. It performs no PMU access, cache
operation, target-specific instruction, network operation, or hardware inference.

The self-test output explicitly states:

```text
physical_backend_implemented = false
h1_generator_established = false
h1_generator_pair_established = false
live_execution_authorized = false
```

## 6. Restoration placeholder

After closing a word, the interface records:

```text
logical byte digest before and after
state-tomography receipt ID
state-equivalence result
independent-output retention result
```

The synthetic self-test deliberately records byte equality while leaving both state
equivalence and independent-output retention false. This tests the important distinction:

```text
logical bytes returned != CAT_CAS R2 restoration
```

A future physical backend cannot promote R2 without the full state and accumulator law.

## 7. Source-isolation requirement

The existing fork-based tomography runtime parses schedule rows before forking. Its child
source code does not read the query field, but the child inherits the parent address
space. A future physical package must not rely on code-path honesty as challenge secrecy.

Required options include:

```text
exec-based source process with a preparation-only payload
capability-isolated helper with no challenge mapping
dedicated shared-memory region containing only carrier and preparation fields
post-waitpid receiver entropy unavailable to source address space
```

The exact mechanism must be frozen before physical authorization.

## 8. I2A decision

The interface is complete enough for compile-only and synthetic lifecycle qualification:

```text
interface_freeze_allowed = true
physical_transport_candidate_freeze_allowed = false
```

The next gate is:

```text
I2B_SECOND_GENERATOR_AND_INVERSE_CANDIDATE_DESIGN
```

I2B must derive at least two physically distinct, receiver-executable candidate
transformations from the accepted coherence primitive family and define non-name-based
inverse candidates. It must reject disjoint operations that commute trivially, aliases
that differ only by labels, and destructive operations disguised as restoration.

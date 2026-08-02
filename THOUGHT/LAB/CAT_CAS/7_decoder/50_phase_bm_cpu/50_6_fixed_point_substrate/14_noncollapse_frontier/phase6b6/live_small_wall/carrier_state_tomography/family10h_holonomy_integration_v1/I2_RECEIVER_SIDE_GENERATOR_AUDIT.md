# I2 Receiver-Side Generator Candidate Audit

## Result

```text
I2_RECEIVER_SIDE_GENERATOR_CATALOG_COMPLETE
NO_H1_ADMISSIBLE_GENERATOR
NO_H1_ADMISSIBLE_GENERATOR_PAIR
REMOTE_STORE_SAME_VALUE_IS_EXTRACTION_CANDIDATE_ONLY
I2A_POST_SOURCE_OPERATOR_RUNTIME_EXTRACTION_DESIGN_NEXT
NO_LIVE_EXECUTION_AUTHORIZED
FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED
FAMILY10H_HOLONOMY_NOT_ESTABLISHED
SMALL_WALL_NOT_CROSSED
```

I2 audits the actual retained C runtime and checkpoint blobs. It does not infer physical
operators from schedule names. The exact source paths and Git blob identities are frozen
in `I2_GENERATOR_CATALOG.json` and verified by `i2_generator_catalog.py`.

## 1. H1 admission law

An H1 generator must be more than an operation-shaped label. It must:

1. act on the surviving carrier after verified source death;
2. be selected and executed by the receiver without source control of the realized word;
3. define a state transformation rather than only an endpoint measurement;
4. leave the same carrier available for a second operation and state tomography;
5. avoid full prefault, destructive flush, re-preparation, and fresh-carrier substitution;
6. admit left- and right-inverse candidates under one frozen carrier-state law;
7. bind physical work, route, timing, line population, and process custody.

No current operation family satisfies all seven requirements.

## 2. Base tomography runtime

The base runtime has valid source-death sequencing:

```text
fork source child
source calls f10_carrier_prepare
parent waitpid completes
parent selects query field
receiver executes f10_carrier_query_mapped
receiver reads PMU endpoint
shared carrier is unmapped
```

The available post-death operations are:

```text
query_A
query_B
query_A_then_B
query_B_then_A
query_sham
carrier_off
```

These are implemented through `query_lane` and `query_pair_ordered`, whose API receives a
`const F10CarrierState *`. They are readout probes at the language level. Physical cache
or coherence disturbance is possible, but the runtime does not measure the carrier state
after one probe, apply a second held-out operation, or reuse the same carrier across a
word. Each schedule row allocates a fresh shared state and unmaps it after one query.

Therefore:

```text
POST_SOURCE_QUERY_EXECUTION_ESTABLISHED
POST_SOURCE_TRANSPORT_MAP_NOT_ESTABLISHED
```

The query label is also part of the schedule row parsed before `fork`. Although the child
code does not consume the query field, the source image inherits the parent's address
space. The current runtime does not establish cryptographic or capability-level secrecy
of a post-death challenge word.

## 3. Relation-spatial runtime

### R0 and R1

`RELATION_SPATIAL_R0` and `RELATION_SPATIAL_R1` are address-pair relations used by
`relation_spatial_prepare` and `relation_spatial_touch_pair`. Preparation:

```text
prefaults the full state
flushes the state
increments lane bytes in the selected relation and source order
```

The source child performs this preparation. The labels are not receiver generator maps.

### R0 then R1 and R1 then R0

The composition controls invoke `relation_spatial_prepare_composition` during source
preparation. In primary composition schedules, the source remains alive through the pair
measurement. These experiments measured a source-authored ordered preparation effect.
They did not execute a receiver word after source death.

Thus the initial nonzero composition contrast is not H3 composition:

```text
SOURCE_AUTHORED_COMPOSITION_OBSERVED
RECEIVER_TRANSPORT_COMPOSITION_NOT_ESTABLISHED
```

### Post-death reset controls

The parent can execute after source death:

```text
full flush
prefault plus flush
double flush
lane-A-only flush
```

These are useful killer controls. They globally evict or rewrite state and failed to
collapse the retained candidate under the frozen screen. They are not inverse operations.

### Post-death same/opposite mutation controls

The parent can repin to the source CPU after source death and call
`relation_spatial_prepare` with the same or opposite relation. That call prefaults,
flushes, and re-prepares the complete state. It replaces the prior physical state rather
than transporting it through a reversible map.

Therefore:

```text
POST_SOURCE_PARENT_EXECUTION_PRESENT
DESTRUCTIVE_REPREPARATION_NOT_GENERATOR
NAME_ONLY_RESTORE_NOT_INVERSE
```

## 4. Accepted coherence primitive

The retained coherence checkpoint establishes:

```text
CONTROLLED_COHERENCE_STATE_FOUND
```

for `OP_REMOTE_STORE_SAME_VALUE`.

The operation preserves logical carrier bytes while producing a measurable
ownership-intent coherence response. In the accepted checkpoint:

```text
change_to_dirty remote_store_same_value = 2104
probe_dirty remote_store_same_value = 5728
all operator windows restored logical bytes
```

The checkpoint explicitly excludes coherence holonomy and a Small Wall crossing.

This is the strongest primitive available for extraction because it is:

- byte-preserving;
- physically active;
- distinct from identity, remote read, and same-core store controls;
- parameterizable by line set, route, core, and operation count.

It is not yet H1 because the legacy worker does not expose it as a receiver-selected,
post-source, persistent-carrier word operation. It also has no qualified physical inverse.

The exact status is:

```text
CONFIRMED_PRIMITIVE_REQUIRES_POST_SOURCE_RUNTIME_EXTRACTION
```

not:

```text
FAMILY10H_GENERATOR_A_ESTABLISHED
```

## 5. Catalog

| Family | Executes after source death | State-transform evidence | Same-carrier composition | Inverse | H1 |
|---|---|---|---|---|---|
| `query_A`, `query_B` | Yes | Readout only | No | None | Fail |
| `query_A_then_B`, `query_B_then_A` | Yes | Ordered readout only | No | None | Fail |
| Relation pair/sham/control probes | Sometimes | Endpoint loads/PMU | No | None | Fail |
| `R0`, `R1` preparation | No | Source preparation | No | None | Fail |
| `R0->R1`, `R1->R0` composition | No | Source-authored composition | No | None | Fail |
| Flush/reset family | Yes | Destructive control | No | None | Fail |
| Same/opposite relation mutation | Yes | Full re-preparation | No | Name only | Fail |
| `remote_store_same_value` | Not in successor runtime | Confirmed coherence primitive | Not exposed | None qualified | Extraction only |
| `remote_read_shared` | Not in successor runtime | Control response | Not exposed | None | Fail |

## 6. I2 decision

```text
h1_admissible_generator_count = 0
h1_admissible_generator_pair_count = 0
physical_transport_candidate_freeze_allowed = false
```

The failed count is not a negative result about the substrate. It is an API and custody
finding: the current runtime does not provide the operation boundary required to test H1.

## 7. I2A design target

The next gate is:

```text
I2A_POST_SOURCE_OPERATOR_RUNTIME_EXTRACTION_DESIGN
```

It must design, offline only, a minimal runtime with this lifecycle:

```text
parent allocates and prefaults one carrier
source child performs answer-blind public preparation
source child exits
parent verifies waitpid and capability closure
receiver samples a hidden operation word
receiver applies byte-preserving operator primitives to the same carrier
receiver records state tomography after every primitive
receiver applies candidate inverse sequence
receiver records R2 tomography
parent destroys the carrier only after the transaction ends
```

The first extraction target is a parameterized operation:

```text
remote_store_same_value(line_set, route, amplitude)
```

A second independent generator and all inverse candidates remain open design questions.
Possible operation names may not be promoted until their physical action and process
custody are explicit.

I2A may implement synthetic state machines and compile-only C interfaces. It may not run
the target, reuse historical authority, or change the retained evidence branch.

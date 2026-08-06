# Two-Port Custody Report

Candidate N: descriptor-compiled two-shared-port CATVM custody.

Classification: `REJECTED_SOURCE_DEFECT`

Runtime packet attack summary:

- Request size: `32` bytes.
- Fail-open packet cases: `['oversize_initialize_plus_one', 'oversize_initialize_plus_16', 'concatenated_initialize_stop', 'cross_record_splice_31_plus_1', 'rejected_initialize_mutates_nonce_state']`.
- Long evidence-root source reproduction failed before bind; short-path reproduction passed twice.

Finding:

The source-local protocol/custody behavior is promising under the short-path qualifier, but the package cannot transfer as a machine law while malformed/oversized seqpacket handling, record-boundary handling, rejected-initialize state mutation, and path-depth behavior remain defective. The prompt explicitly required malformed records not be silently truncated, normalized, spliced, or accepted.

Custody scope retained:

Short-path source controls still support owner/type/generation/lease tuple checks, denied projection/snapshot/stale metadata attacks, inverse controls, disconnect cleanup, and same-carrier reuse. Those are source-local evidence only until the packet-layer defect is repaired and rerun.

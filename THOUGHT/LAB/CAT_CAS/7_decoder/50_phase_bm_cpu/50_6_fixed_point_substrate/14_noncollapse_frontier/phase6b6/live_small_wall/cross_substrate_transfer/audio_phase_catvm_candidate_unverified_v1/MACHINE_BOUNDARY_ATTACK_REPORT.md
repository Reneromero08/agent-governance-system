# Machine Boundary Attack Report

Status: unverified transfer evidence only. Not canonical. Small Wall not crossed.

## Candidate A

- Active probe used the frozen run1 source service binary.
- Intermediate projection request was denied by protocol.
- Final boundary response was available before the RESTORE request.
- Closing after the final boundary without RESTORE still produced the final boundary packet.
- Correct-mode restoration and same-carrier reuse passed across the recorded cycle stress.

Decision: the source Candidate A protocol does not satisfy atomic final-only response after restoration. The algebra and in-place restoration remain salvageable only through a repaired wrapper.

## Candidate B

- Active probe used the frozen run1 testing service with the copied public manifest.
- Intermediate projection requests were denied.
- EXECUTE responses were returned as completed transaction packets.
- Negative restoration mode returned a closed response path for the tested command.

Decision: Candidate B has a source-local atomic EXECUTE pattern, but the static audit still limits transfer because topology identifiers and scheduling receipts are fixed to the copied public graph.

## Branch-Local Transfer Reference

- Reference cycles: 64
- Cycle boundaries matched: True
- State restored each accepted cycle: True
- Negative restoration variants failed closed: True

This repaired reference is evidence that the transaction law can be reconstructed locally; it is not evidence that the source Candidate A implementation satisfied that law.

## Evidence Hashes

- RESTORATION_REUSE_STRESS payload: `fa9cb844f3860d8938c5c49bca43fbcc78d2cea6f1894ce2bad8ee282cdb20fa`
- Candidate A service: `baaa35eb8d7fa020140c5edd6c519bb1d1aeacce1df8057eb307c9b7e6d3bc78`
- Candidate B testing service: `b367e3c44ab8a92a1b1fce8b03a7f5813fd0f9301cf6bdb587be429832d99fd9`
- Raw log directory: `/run/media/reneshizzle/860_1/CCC 2.0/AI/wt-audio-catvm-independent-verification/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/cross_substrate_transfer/audio_phase_catvm_candidate_unverified_v1/raw_logs/restoration_boundary`

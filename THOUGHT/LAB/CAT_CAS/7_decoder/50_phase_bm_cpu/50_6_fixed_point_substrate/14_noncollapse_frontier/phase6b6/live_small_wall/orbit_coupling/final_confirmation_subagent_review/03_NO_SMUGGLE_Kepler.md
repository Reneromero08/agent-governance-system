# 03 No-Smuggle and Ordinary-Explanation Attacker - Kepler

Agent/thread ID: `019f5dfd-b44c-7590-b623-f161901ba96a`

Model: `gpt-5.6-sol`

Reasoning effort: `max`

Role: Subagent 3, No-Smuggle and Ordinary-Explanation Attacker

SUBAGENT_ROLE  
Subagent 3: No-Smuggle and Ordinary-Explanation Attacker. Read-only static review of clean local `main` at `f6ef90374de424723e0edba34786778e8e3f1a29`.

VERDICT: MATERIAL_BLOCKER

CONFIDENCE: HIGH

MATERIAL_FINDINGS:
- `NSA-01` | Source: [gain_covariant_confirmation_public.py](<D:/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/orbit_coupling/orbitstate_gain_covariant_confirmation_v1/gain_covariant_confirmation_public.py:104>), lines 104-177, 221-268, 320-413; [gain_covariant_confirmation_target.py](<D:/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/orbit_coupling/orbitstate_gain_covariant_confirmation_v1/gain_covariant_confirmation_target.py:924>), `receiver_features_main`. Mechanism: group IDs expose the condition index as `grp_00` through `grp_08`; run IDs are exhaustively reproducible from the public seed and the nine possible condition indices. The same receiver-loaded module publishes the ordered condition table, private phase sequences, `q_theta` formulas, and `build_private_source_map`. Consequence: using only receiver-visible code and the public schedule, a worker can reconstruct every condition and `q_theta`, then declare `logical_response=g*q_theta` and the matching physical reversal. This satisfies controls, geometry, signs, nulls, and receipt partitioning without measuring the PMU carrier. Minimal repair: move all condition/formula/map-building code out of the receiver artifact and replace public-seed-derived identifiers with independently random private assignments having no ordinal prefix or enumerable derivation. Required regression: an attacker given exactly the deployed receiver image and public schedule must be unable to map any opaque ID to a condition; assert private labels, phase tables, `q_theta`, and map builders are absent. Must repair before live: `true`.
- `NSA-02` | Source: [gain_covariant_confirmation_target.py](<D:/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/orbit_coupling/orbitstate_gain_covariant_confirmation_v1/gain_covariant_confirmation_target.py:876>), `run_receiver_feature_subprocess`, `receiver_features_main`, `execute_live`; [gain_covariant_confirmation_runtime.c](<D:/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/orbit_coupling/orbitstate_gain_covariant_confirmation_v1/gain_covariant_confirmation_runtime.c:652>), `source_child_loop`, `main`. Mechanism: the receiver subprocess runs the target script from the private-bearing source directory under the same UID. Its deny checks resolve two relative names only inside the temporary CWD; they impose no filesystem capability boundary. The private map is reachable through the script/module directory, and source receipts are reachable as a sibling of the absolute receiver-output path. Both replicates and all `q_theta` receipts exist before this subprocess starts. The combined C receiver binary also receives the private-map path and contains the private parser/formula code. Additionally, `validate_private_map_in_separate_process` opens the map before feature freeze. Consequence: the receiver can read the map or receipts, fabricate receipt-coherent ideal features, and emit its own matching hash before freeze. The claimed "only source child opens before unblinding" property is false. Minimal repair: use a separate receiver-only executable under a distinct unprivileged UID plus an enforced mount namespace/chroot or equivalent; expose only four public inputs and an output FD. Keep the map, receipts, source formula code, private paths, and source executable inaccessible. Required regression: from the actual deployed receiver process, attempts to open the module-directory map, sibling source receipts, private argv/proc paths, and private source symbols must fail with OS-enforced denial. Must repair before live: `true`.
- `NSA-03` | Source: [gain_covariant_confirmation_public.py](<D:/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/orbit_coupling/orbitstate_gain_covariant_confirmation_v1/gain_covariant_confirmation_public.py:935>), `build_condition_decodes`, `gain_covariant_geometry_law`, `adjudicate`. Mechanism: adjudication never enforces exact feature-row coverage or equality of row identity fields to the public schedule. Replicate laws iterate only replicate keys present in the receiver feature file. A receiver can remove every replicate-1 feature row, retain the declarative integrity metadata, recompute the feature hash, and submit a complete passing replicate 0. Source-receipt validation remains green because it independently sees all receipts. Consequence: the adjudicator can emit `ORBITSTATE_GAIN_COVARIANT_CONFIRMATION_CONFIRMED` without the required second fresh replicate; this is a direct frozen-law violation and can hide a failed replicate. Minimal repair: require exactly 144 unique feature rows, exact schedule identity fields, both replicates, all nine groups, four phases, and two mappings before any decoding; iterate explicit `REPLICATES`, never observed keys. Required regression: omit replicate 1, omit a complete phase cell, duplicate a run ID, or relabel replicate/phase/mapping and assert hard rejection before classification. Must repair before live: `true`.

NONBLOCKING_CONCERNS:
- `mapping_order` is parsed but not used; source/subcapture order is stripped from receiver features, while `phase_transfer_law` records absent values. Order-bias mock branches exist but are not included in the frozen self-test matrix. No independent confirmation-pass mechanism was established from order alone.
- Byte-digest restoration proves byte equality, not cache, TLB, page, or predictor-state equality. The source child and shared banks persist across four component windows in a mapping pair.
- `source_off` uses one dummy-bank work loop while active conditions use two bank loops, and q-bearing source receipts are formatted and flushed before receiver measurement. Timing is not used by feature extraction, so this remained a concern rather than a demonstrated blocker.

ATTACKS_ATTEMPTED:
- Public dictionary attack on group/run IDs: succeeds by exact static construction; all condition assignments are recoverable.
- Receiver-side formula declaration without opening the private map: succeeds by exact static construction using the receiver-loaded condition table and `q_theta`.
- Direct private-map and source-receipt path access: succeeds as a capability attack; the relative deny checks do not restrict absolute or sibling paths.
- Failed-replicate omission with a newly frozen hash: succeeds by code trace; missing replicate keys are never required.
- Target/fold/polarity-derived gain fitting and target-class feedback: blocked by the exact two-key gain estimator and post-law classification.
- Aggregate rescue with both replicates present: blocked; aggregate geometry is diagnostic only.
- Predecessor pooling and post-unblinding feature mutation: no decision dataflow found; the in-memory feature digest is checked before map opening.
- Source-order, subcapture-order, runtime-timing, dummy-bank, and shared-state ordinary explanations: attempted; no independent passing construction established.

CLAIMS_SUPPORTED:
- Exact clean review head and supplied byte hashes were verified; no repository, Git, remote, or device writes occurred.
- The direct `extract_receiver_features` function body uses receiver PMU `change_to_dirty` measurements and no expected target vector.
- Gain estimation accepts exactly `post_projection` and `equal_orbit_odd_zero`.
- Strong/near-zero partitioning reads `q_theta` from source receipts, not by recomputing it inside `phase_transfer_law`.
- Aggregate values cannot rescue an explicitly present failed replicate.
- Feature mutation after the supplied hash is frozen is rejected before private-map opening.
- The current source calculation receives no PMU counter feedback; its pipe input is run ID, component, and public source order.
- Result classification is selected only after law evaluation, and no predecessor evidence path enters adjudication.

CLAIMS_NOT_SUPPORTED:
- Opaque identifiers do not reveal condition labels.
- The receiver-feature process cannot access the private map or source receipts.
- Source formulas and condition-specific code are absent from the receiver process.
- Source receipts are sealed and unreadable before feature freeze.
- The runtime source child is the only process that opens the private map before unblinding.
- Both fresh replicates are structurally mandatory in adjudication.
- The frozen feature-boundary self-test proves filesystem isolation; its `receiver_reads_private_source_map` case is only an alias for inserting a forbidden `condition` field.

RECOMMENDATION:
Do not authorize live execution from this package. Repair `NSA-01` through `NSA-03`, issue a new package/transaction identity with newly randomized mappings, refreeze every dependent artifact and hash, and require real OS-level receiver-denial tests plus exact feature-coverage attacks before reconsidering live authority. No dynamic test was needed because the passing constructions follow directly from the frozen dataflow.


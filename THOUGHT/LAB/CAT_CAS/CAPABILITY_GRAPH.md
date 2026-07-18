# CAT_CAS Capability Graph

This view is generated from `control/capability_graph.json`. It answers what
each load-bearing experiment contributes to the final machine. It is not an
evidence ledger and does not replace experiment reports.

Mission: `CAT_CAS_UNBOUNDED_COMPUTE_V1`

## Compute-leverage ladder

### C0_CATALYTIC_CLOSURE

The substrate is genuinely borrowed, used, uncomputed, and restored.

### C1_BOUNDED_RESIDENCY

Clean or resident memory stops scaling with logical problem extent.

### C2_REUSE_AND_MULTIPLEX

One finite substrate serves unbounded cumulative work or demand over time.

### C3_NONCOLLAPSE_REPRESENTATION

.holo preserves an object whose classical materialization is much larger.

### C4_NATIVE_GLOBAL_OPERATOR

An operator acts on compact relational geometry without enumerating or materializing the classical path space.

### C5_FIXED_POINT_ADVANTAGE

Native substrate extraction scales asymptotically better than forward construction under no-smuggle controls.

### C6_EXTERNAL_ACCEPTANCE

A classical witness emitted at the final Wall is accepted by an independent official boundary.

### C7_CROSS_DOMAIN_TRANSFER

The same native mechanism reaches external acceptance on a materially different frontier.

## Capability lineage

### EXP01 - Catalytic closure and bounded clean workspace

- Path: `1_foundations/01_tree_evaluation`
- Mission rungs: `C0_CATALYTIC_CLOSURE`, `C1_BOUNDED_RESIDENCY`
- Task class: `enabling_infrastructure`
- Status: `REPORTED_BASELINE`
- Capabilities: dirty_substrate_borrow, reversible_traversal, exact_restore, bounded_clean_space
- Open blockers: none

### EXP08 - Finite resident substrate with unbounded queued demand

- Path: `2_substrate_expansion/08_catalytic_gpt`
- Mission rungs: `C2_REUSE_AND_MULTIPLEX`
- Task class: `enabling_infrastructure`
- Status: `REPORTED_BASELINE`
- Capabilities: shared_vram_tape, lease_return, bounded_resident_workspace
- Open blockers: none

### EXP12 - Structured tape and recursive compute reuse

- Path: `2_substrate_expansion/12_structured_tape_acceleration`
- Mission rungs: `C2_REUSE_AND_MULTIPLEX`
- Task class: `enabling_infrastructure`
- Status: `REPORTED_BASELINE`
- Capabilities: warm_tape_reuse, fingerprinted_cache, cross_instance_transfer
- Open blockers: none

### EXP14 - Unbounded cumulative work through fixed restored substrate

- Path: `3_physics_complexity/14_bekenstein_violator`
- Mission rungs: `C2_REUSE_AND_MULTIPLEX`
- Task class: `calibration`
- Status: `REPORTED_BASELINE`
- Capabilities: repeated_restored_cycles, throughput_over_static_capacity
- Open blockers: none

### EXP20_11 - Contained phase-native .holo

- Path: `3_physics_complexity/20_catalytic_eigen_shor/20_11_contained_holo_verifier`
- Mission rungs: `C3_NONCOLLAPSE_REPRESENTATION`
- Task class: `enabling_infrastructure`
- Status: `REPORTED_BASELINE`
- Capabilities: phase_eigenbasis, answer_not_stored, illumination_readout, streaming_materialization_avoidance
- Open blockers: none

Code entrypoints:

- `3_physics_complexity/20_catalytic_eigen_shor/20_11_contained_holo_verifier/20_11a_contained_holo/contained_holo_verifier.py`
- `3_physics_complexity/20_catalytic_eigen_shor/20_11_contained_holo_verifier/20_11b_self_observing/self_observing_loop.py`
- `3_physics_complexity/20_catalytic_eigen_shor/20_11_contained_holo_verifier/20_11g_streaming/streaming_shor.py`

### HOLO_GEN5 - Executable geometric memory runtime

- Path: `7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier`
- Mission rungs: `C3_NONCOLLAPSE_REPRESENTATION`, `C4_NATIVE_GLOBAL_OPERATOR`
- Task class: `enabling_infrastructure`
- Status: `IMPLEMENTED_ACTIVE_FRONTIER`
- Capabilities: geometry_basis, relation_graph, ordered_path_memory, semantic_reexecution, typed_invariant_family, atomic_collapse_boundary, restoration_record
- Open blockers: none

Code entrypoints:

- `7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/holo_runtime/holo_geometry.c`
- `7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/holo_runtime/holo_semantic_integrity.c`
- `7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/holo_runtime/holo_invariant_family.c`
- `7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/l4b_orbitstate/holo_path_history.c`

### EXP49 - Decoder and forward-wall characterization

- Path: `7_decoder/49_the_decoder`
- Mission rungs: `C4_NATIVE_GLOBAL_OPERATOR`
- Task class: `evidence_audit`
- Status: `CLOSED_OUT_LOCAL_CLAIM`
- Capabilities: extractive_readout, wrong_answer_controls, wall_localization, public_fixed_point_target_generation
- Open blockers: none

### EXP50 - Physical substrate and Small Wall

- Path: `7_decoder/50_phase_bm_cpu`
- Mission rungs: `C4_NATIVE_GLOBAL_OPERATOR`, `C5_FIXED_POINT_ADVANTAGE`
- Task class: `flagship_compute`
- Status: `ACTIVE_WITH_BLOCKERS`
- Capabilities: physical_carrier_executor, noncollapse_runtime, gain_covariant_projection_transduction, source_and_receiver_custody, restoration_custody
- Open blockers: query_separated_identifiability, answer_cache_exclusion, physical_R2_restoration, fixed_point_scaling

### AUDIO_SIDEQUEST - Pi-native audio and toroidal compute

- Path: `7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate`
- Mission rungs: `C3_NONCOLLAPSE_REPRESENTATION`, `C4_NATIVE_GLOBAL_OPERATOR`
- Task class: `enabling_infrastructure`
- Status: `P0_BUILD_READINESS_COMPONENTS_SEEDED`
- Capabilities: U1_phase_state, torus_relations, recursive_phase_tree, recursive_wave_operator, software_catalytic_wave_loop, recursive_catalytic_ising_emulator, classical_waveform_projection
- Open blockers: none

### TRACK8 - Externally adjudicated bounty transfer

- Path: `8_external_frontiers`
- Mission rungs: `C6_EXTERNAL_ACCEPTANCE`, `C7_CROSS_DOMAIN_TRANSFER`
- Task class: `flagship_compute`
- Status: `ARCHITECTURE_PRESENT_ADAPTERS_PENDING`
- Capabilities: source_freeze_contract, official_verifier_boundary, classical_witness_contract, transfer_record_design
- Open blockers: none

## Transfer edges

- `EXP01` -> `EXP08`: generalizes restored substrate into multiplexed use
- `EXP08` -> `EXP12`: adds retained relational work and warm reuse
- `EXP12` -> `EXP14`: separates finite substrate capacity from cumulative logical throughput
- `EXP14` -> `EXP20_11`: moves from repeated reuse into noncollapse representation
- `EXP20_11` -> `HOLO_GEN5`: unifies containment with executable geometry and explicit boundaries
- `HOLO_GEN5` -> `EXP49`: supplies process objects for extractive invariant readout
- `EXP49` -> `EXP50`: moves the Wall from readout to physical substrate
- `AUDIO_SIDEQUEST` -> `HOLO_GEN5`: supplies pi-native carrier semantics and recursive toroidal composition
- `EXP50` -> `TRACK8`: exports substrate-native compute leverage to external Walls
- `HOLO_GEN5` -> `TRACK8`: exports noncollapse process objects and boundary discipline

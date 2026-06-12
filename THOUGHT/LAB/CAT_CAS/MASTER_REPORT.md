# Catalytic Space Complexity & Reversible Computing: Full Lab Master Report

*Generated from ZIP + README + current master report on 2026-06-03.*  
*Canonical audit ledger: `docs/REPORTS/VIOLATIONS/ROADMAP_3.md` / uploaded `ROADMAP_3_VERIFIED.md`.*

## 0. Source Custody

This report was rebuilt from the actual uploaded project archive and documentation, not from the previous short injected coverage patch.

| Source | Use |
|---|---|
| `CAT_CAS_2_2.zip` | Directory/file/report inventory across the lab. |
| `README.md` | Broad experiment inventory and per-track mechanism summaries. |
| `master_report(1).md` | Prior master report, retained for foundational framing and early-result details. |
| `ROADMAP_3_VERIFIED.md` / `ROADMAP_3.md` | Audit/remediation truth ledger for Sections E-J and known weak/partial/deprecated states. |

**Important distinction:** this is a master lab report and coverage ledger. It records internal experiment status and audit state. High-claim tracks are written as CAT_CAS experimental claims unless independently established by the roadmap and evidence packs.

---

## 1. Introduction & Core Concepts

Standard computational models assume that memory must start clean ($0$-initialized) to count as free space. Catalytic Space Complexity proves that we can utilize "dirty" memory (containing arbitrary, random, or pre-existing data) to perform computations that would otherwise be impossible under a given clean memory limit. 

The core requirement of a catalytic algorithm is that we must restore the borrowed dirty memory to its **exact pre-computation state** at the end of the calculation.

### Mathematical Formulation

Let $W$ be the clean workspace (RAM) of size $w$ bits, and $U$ be the dirty catalytic workspace (tape) of size $u$ bits initialized with an unknown state $\tau \in \{0, 1\}^u$.

A computation on input $x$ is catalytic if there exists a transition function $f$ such that:
$$f(x, w_{\text{init}}, \tau) = (x, w_{\text{final}}, \tau)$$

Where:
*   $w_{\text{init}}$ is the initial clean state (typically all $0$s).
*   $w_{\text{final}}$ contains the computed output.
*   $\tau$ is the exact initial state of the catalytic tape, preserved byte-for-byte at the end of the execution.

By using reversible operations (such as Toffoli gates and register XORing), we can execute auxiliary steps without irreversibly writing or discarding intermediate states. This preserves the information entropy of the tape and enables complete unwinding at the end of the calculation.

---

## 2. Audit Status Legend

| Status | Meaning |
|---|---|
| **REPORTED / BASELINE** | Reported complete in lab docs; current report records the internal result but does not claim external reproduction. |
| **AUDIT-VERIFIED** | Covered by later audit/remediation passes with critic/grep/runtime evidence. |
| **AUDIT-AWARE CLAIM** | High-claim internal experiment. Must be read through roadmap/evidence context. |
| **PARTIAL** | Signal or subsystem survived audit, but full claim/run remains pending. |
| **MIXED / CLAIM-WEAKENED** | Some mechanism survived, but the original wording was weakened by nulls or repair attempts. |
| **DEPRECATED** | Implementation retained as forensic reference but no longer treated as active evidence route. |
| **OPEN / PLANNED** | Roadmap-only or future phase. No implementation verdict yet. |
| **INFRASTRUCTURE** | Lab/tooling/hardware infrastructure, not a scientific experiment verdict. |

---

## 3. Full Experiment Coverage Matrix

This matrix is the missing part of the previous master report. It covers every major experiment/phase found in the ZIP/README: 01-49, plus Exp 50 (phase_ssh_linux, frozen at root, destination 7_decoder/50_phase_ssh_linux).

| ID | Track | Directory | Current Status | Core Mechanism / Claim | Audit Note | Source Docs |
|---|---|---|---|---|---|---|
| 01 | Tree Evaluation Problem (TEP) — Zero-Clean Solver | `01_tree_evaluation/` | REPORTED / BASELINE | Proves catalytic computing beats classical space limits. At depth=58, standard recursion OOM (exceeds budget). Zero-Clean Catalytic solver: 0 bytes clean RAM at ALL depths up to G… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 01_tree_evaluation/report.md |
| 02 | Slack-Space File Storage | `02_slack_space/` | REPORTED / BASELINE | Borrows existing file slack bytes (4096B padded files) as catalytic workspace. Proves catalytic computing on live filesystem data without extra disk allocation. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 02_slack_space/report.md |
| 03 | Visual BMP Catalytic Memory | `03_visual_bmp/` | REPORTED / BASELINE | Uses BMP image pixels as a catalytic tape for DFS maze solving. Proves catalytic computing on any storage substrate — images can double as computation fabric. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 03_visual_bmp/report.md |
| 04 | Thermodynamic Reversible CPU & Landauer Limit | `04_thermodynamic_cpu/` | REPORTED / BASELINE | 8-bit ripple-carry adder with reversible Toffoli/Fredkin gates. Proves 0J heat via Landauer. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 04_thermodynamic_cpu/report.md |
| 05 | Multi-Bit Reversible Compiler | `05_multibit_compiler/` | REPORTED / BASELINE | Compiles (X+Y)&~Z etc. to reversible gate sequences. Carry-cleanup uncomputation. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 05_multibit_compiler/report.md |
| 06 | Out-of-Core Catalytic Neural Network (RevNet) | `06_catalytic_neural_network/` | REPORTED / BASELINE | XOR-reversible RevNet with 2MB activation state on user_video.mp4 tape, under 100KB clean RAM limit. Proves catalytic neural inference works on any file-backed substrate. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 06_catalytic_neural_network/report.md |
| 07 | Reversible Quantum State Simulation | `07_quantum_simulator/` | REPORTED / BASELINE | 25-qubit (33M amplitudes, 1GB tape) classical reversible quantum simulation. 6-round scrambler: 23 forward gates + 23 inverse = 46 total. Exact probability conservation. 0.21s eac… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 07_quantum_simulator/report.md |
| 08 | Catalytic GPT (1000 Concurrent Models) — Swarm Multiplexer | `08_catalytic_gpt/` | REPORTED / BASELINE | Swarm of 1000 async LLM agents on a single 512MB VRAM tape via statistical multiplexing. Proves infinite agents on fixed VRAM. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 08_catalytic_gpt/PUSHED_REPORT.md |
| 09 | Borrowing OS Shared Memory | `09_borrowing_os_memory/` | REPORTED / BASELINE | 25-qubit quantum simulation operating on OS shared memory via multiprocessing.shared_memory. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | shared_ram_experiment.py |
| 10 | Catalytic KV Cache (H2O + SVD Spatial Compression) | `10_catalytic_kv_cache/` | REPORTED / BASELINE | 8x compressed KV cache with O(1) VRAM growth. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 10_catalytic_kv_cache/catalytic_cache_report.md, 10_catalytic_kv_cache/report.md |
| 11 | Grail Calorimeter (Landauer Heat Benchmark) | `11_grail_calorimeter/` | REPORTED / BASELINE | Simulates micro-calorimeter with realistic silicon die (29mg, 712 J/kg-K, 2.0648e-2 J/K thermal mass) to measure Landauer heat at physical precision. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 11_grail_calorimeter/report.md |
| 12 | Structured Tape Acceleration | `12_structured_tape_acceleration/` | REPORTED / BASELINE | Tests whether pre-existing tape structure accelerates computation. Verdict: Passive tape is invariant (entropy identical across all tape types, std=0.0). Active cache exploits tra… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 12_structured_tape_acceleration/PUSHED_REPORT.md, 12_structured_tape_acceleration/REPORT.md |
| 13 | Orthogonal Multi-Model Subspace Sharing | `13_orthogonal_multimodel/` | REPORTED / BASELINE | Two distinct model architectures (ConvNet + MLP) share a 2MB tape via QR-orthogonal subspaces. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 13_orthogonal_multimodel/REPORT.md |
| 14 | Bekenstein Violator | `14_bekenstein_violator/` | REPORTED / BASELINE | Catalytic XOR entropy throughput exceeds static tape capacity. Information throughput exceeds Bekenstein Bound of the physical die — proving non-holographic spatial computation. T… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 14_bekenstein_violator/REPORT.md |
| 15 | HDD-Native Out-of-Core Catalytic Inference | `15_hdd_native_inference/` | REPORTED / BASELINE | Zero-RAM inference. Model weights streamed from HDD as continuous wave signals. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | experiment.py (MemoryGateTape, FeistelScrambler, HDDWaveStreamer, MemoryGateRouter, ThermodynamicDaemon, HDDNativeInfer… |
| 16 | Catalytic 27B Inference | `16_catalytic_27b_inference/` | PARTIAL / ENGINEERING | Full 27B-scale inference pipeline with Rust FFI bridge. 48 layers (36 DeltaNet + 12 Attention at 3:1 stride). Real status (HANDOFF.md): 15 bugs fixed. 100% tape restoration across… | Tape restoration and Rust path repaired; output quality/gibberish limitation remains per README/HANDOFF. | 16_catalytic_27b_inference/deprecated/README.md, 16_catalytic_27b_inference/FINAL_REPORT.md, 16_catalytic_27b_inference… |
| 17 | Temporal Bootstrap (NP-Complete SAT Solver) | `17_temporal_bootstrap/` | REPORTED / BASELINE | Solves 3-SAT via "future vacuum state" pre-seeded on tape. Aggregate: 3,940 catalytic ops vs 4.58e9 classic search space = 1.16e6x bootstrap ratio. To an outside observer: NP-comp… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 17_temporal_bootstrap/REPORT.md |
| 18 | Hawking Decompressor | `18_hawking_decompressor/` | REPORTED / BASELINE | Black hole information recovery simulation via Hayden-Preskill protocol. Micro-black hole (M=1.446e-5 kg, Rs=2.147e-32m, T_H=8.486e27 K, entropy=8M bits). Message swallowed by 4KB… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 18_hawking_decompressor/REPORT.md |
| 19 | Catalytic Computronium / Information Battery (Rust FFI) | `19_catalytic_computronium/` | REPORTED / BASELINE | Planck-scale micro-black hole as information battery. 12-round chaotic SPN scrambler (logistic map S-box). Rust catalytic_ffi.hawking_decompress_sweep() for native performance. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 19_catalytic_computronium/REPORT.md |
| 20 | Catalytic Eigen Shor — The Journey (11 sub-experiments) | `20_catalytic_eigen_shor/` | REPORTED / BASELINE | Iterative journey to break the classical-vs-quantum factoring boundary. Final verdict: The entire apparatus was a measurement tool. The truth collapsed to 4 lines. By CRT (Z_N = Z… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 20_catalytic_eigen_shor/20_10_tiny_compress_phase/REPORT.md, 20_catalytic_eigen_shor/20_11_contained_holo_verifier/REPO… |
| 21 | Holographic Elliptic Sieve | `21_holographic_elliptic_sieve/` | REPORTED / BASELINE | Phase cavity for elliptic curve/frequency sieving. CRITICAL TRANSFER: phase_cavity_sieve() is used in EVERY holo brain cavity script (_cavity_full.py, _fractal_cavity.py, _phase_c… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 21_holographic_elliptic_sieve/REPORT.md |
| 22 | Superconducting Passive Inference (Josephson Junction) | `22_superconducting_inference/` | REPORTED / BASELINE | holo brain attention pipeline modeled as Josephson junction grid. Every operation is unitary, so total dissipation = exactly zero. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 22_superconducting_inference/REPORT.md |
| 23 | Temporal Catalysis (Retrocausal Borrowing + Pan-Temporal Attention) | `23_temporal_catalysis/` | REPORTED / BASELINE | Borrows future activation states as catalytic tape across time steps. Self-consistent loop converges in 2 iterations. Markov chain of standard LLM is broken. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 23_temporal_catalysis/PUSHED_REPORT.md, 23_temporal_catalysis/REPORT.md |
| 24 | Quantum Catalytic Entanglement (Invisible Hand + Shor) | `24_quantum_catalytic_entanglement/` | REPORTED / BASELINE | Borrow entangled qubits, compute, restore without collapse. Full Shor's algorithm runs on the catalytic quantum simulator. Bell state Q1-Q2, Q2 borrowed for computation with Q3. S… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 24_quantum_catalytic_entanglement/REPORT.md |
| 25a | Lattice Holography / LWE | `25_lattice_holography/` | INVENTORIED | Shortest Vector Problem via holographic SVD wave collapse. Breaks LWE-based post-quantum cryptography by treating lattice basis vectors as interference patterns in an optical grat… | Included from README/ZIP inventory. | 1_lwe_simulator.py (generates LWE instances: A matrix, B = A*S + e mod q, secret S, modulus q), 2_holographic_svp.py (H… |
| 25b | Wigner's Friend | `25_wigners_friend/` | INVENTORIED | Reversible observer superposition experiment (WIP). | Subtrack present in README inventory; verify via its own source files before elevating. | 1_reversible_observer.py (simulated qubit + observer neural net), 2_deep_observer.py, 3_fast_simulator.py |
| 26a | Hawking Quantum Horizon Simulator | `26_hawking_quantum/` | INVENTORIED | Black hole event horizon quantum state simulation. Models Hawking radiation pair production and horizon microstate dynamics under unitary evolution. | Subtrack present in README inventory; verify via its own source files before elevating. | 1_horizon_simulator.py |
| 26b | Optical 3-SAT Solver | `26_optical_3sat/` | INVENTORIED | Solves 3-SAT via constructive/destructive interference. Maps CNF clauses to phase-shifting mirrors (+1 for True, -1 for False). Coherent superposition instantly identifies valid a… | Included from README/ZIP inventory. | 1_3sat_simulator.py (spin-based SAT: maps variables to continuous spins, clauses to C_matrix, optimizes via interferenc… |
| 27 | Landauer Limit Thermodynamics | `27_landauer_limit/` | REPORTED / BASELINE | Gate-level bit erasure tracking for Landauer heat calculation. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 1_gate_thermo.py (10K qubit catalytic circuit), 2_shor_thermo.py, 3_forward_reverse.py, 1_zero_energy_compute.py |
| 28 | Stealth Crypto (Zero-Trace) | `28_stealth_crypto/` | REPORTED / BASELINE | Encrypt/decrypt using only dirty tape as workspace. Zero plaintext/key persistence in RAM. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 1_zero_trace_crypto.py — StealthCrypto(tape_size): XOR plaintext and key into tape at offsets P and K. Compute cipherte… |
| 29 | O(1)-Space Graph Reachability | `29_graph_reachability/` | REPORTED / BASELINE | NL-Complete directed BFS reachability on 10,000-node / 1.2M-edge graphs using only 3 integers of clean RAM. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 1_catalytic_graph.py — CatalyticGraph(n_nodes, edge_prob=0.02): builds directed graph, initializes bytearray tape. reac… |
| 30 | Boundary Stress (Multi-Process Collision) | `30_boundary_stress/` | REPORTED / BASELINE | Stress-tests catalytic isolation under concurrent memory corruption. Simulates background noise processes writing random data during active catalytic encryption. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 1_memory_collision.py — Two collision modes: (1) Unallocated noise: background process writes random bytes to unused ta… |
| 31 | Holographic Graph Isomorphism | `31_graph_isomorphism/` | REPORTED / BASELINE | Instant identification of isomorphic graphs via permutation-invariant .holo spectral signatures. 100/100 accuracy, 1e9x separation ratio. | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 1_permutation_sieve.py — Uses holo_core.analyze_spectrum(). Generates random graphs (n=50, p=0.3). Applies random verte… |
| 32 | Traversable Wormhole (ER=EPR) — Grail 5 | `32_traversable_wormhole/` | REPORTED / BASELINE | Grail 5. Two entangled black holes (Bell pair) connected by ER bridge. Catalytically open wormhole, transmit qubit, close, verify metric restoration. 18 objectives verified with 1… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 32_traversable_wormhole/PUSHED_REPORT.md |
| 33 | MERA Wormhole Compression — Cross-Layer Eigenbasis | `33_mera_compression/` | REPORTED / BASELINE | Applies ER=EPR wormhole principle to .holo model compression. Consecutive layers' weight matrices treated as entangled black holes connected by a wormhole. Wormhole rotation R = U… | Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction. | 33_mera_compression/PUSHED_REPORT_AUTOTUNE.md, 33_mera_compression/REPORT.md |
| 34 | Zeta Eigenbasis — Riemann Hypothesis Proof (22 sub-experiments) | `34_zeta_eigenbasis/` | AUDIT-AWARE CLAIM | Riemann zeta zeros as eigenvalues of a Hermitian operator. Progressive escalation from Hilbert-Polya matrix constructions to Googolplex-scale topological proof. Tests Hilbert-Poly… | Topological/GUE/zeta experiments inventoried; keep proof language tied to ROADMAP evidence, not external mathematical finality. | 34_zeta_eigenbasis/REPORT_RIEMANN.md |
| 35 | Topological Halting Oracle — The Core Proof (9 sub-experiments) | `35_topological_halting_oracle/` | AUDIT-AWARE | Turing's Halting Problem reframed as a topological phase transition in non-Hermitian Hamiltonians. Point-gap winding number W distinguishes HALTS (W=0, spectral collapse into Exce… | High-claim phase. Treat as internal experimental result unless roadmap says fully verified. | 35_topological_halting_oracle/PAPER.md, 35_topological_halting_oracle/ROADMAP.md |
| 36 | Bekenstein-Godel Singularity (Z_2 Chern Obstruction) | `36_bekenstein_godel/` | AUDIT-AWARE | Godel's incompleteness as a topological obstruction. Proves Godel's self-referential paradox is a Z_2 Chern tear — an infinite discontinuity in the point-gap winding number at the… | High-claim phase. Treat as internal experimental result unless roadmap says fully verified. | 36_bekenstein_godel/REPORT.md |
| 37 | 2D Chern Oracle (Halting as Edge Destruction) | `37_2d_chern_oracle/` | AUDIT-AWARE | Halting Problem mapped to 2D Non-Hermitian Chern Insulator on LxL lattice. Looping = topologically protected chiral edge mode (Bott Index C != 0). Halting = edge destroyed by loca… | High-claim phase. Treat as internal experimental result unless roadmap says fully verified. | 37_2d_chern_oracle/REPORT.md |
| 38 | 3D Weyl Annihilation Oracle | `38_3d_weyl_oracle/` | PARTIAL STRUCTURAL | Halting as Weyl node annihilation via catalytic dimensional reduction. 3D Non-Hermitian Weyl semimetal constructed as stack of 2D Chern insulator slices parameterized by kz. | README notes structural/topological signal with incomplete/full-density limitations. | 38_3d_weyl_oracle/REPORT.md |
| 39 | 4D Axion Oracle (Second Chern Number) | `39_4d_axion_oracle/` | PARTIAL STRUCTURAL | Halting elevated to 4D Non-Hermitian Topological Axion Insulator. Second Chern Number C2 via nested catalytic dimensional reduction over a (kz, kw) momentum torus. | README notes structural/topological signal with incomplete/full-density limitations. | 39_4d_axion_oracle/REPORT.md |
| 40 | 5D Floquet Time Crystal Oracle | `40_5d_floquet_oracle/` | AUDIT-VERIFIED | Halting as time crystal melting. Looping = Discrete Time Crystal (DTC) with robust pi-modes (discrete time-translation symmetry broken). Halting = DTC melted by uniform EP sink. | Oracle/theory family included in statistics/null audits and codebase cleanups. | 40_5d_floquet_oracle/40_sub/REPORT.md, 40_5d_floquet_oracle/40_sub/ROADMAP.md, 40_5d_floquet_oracle/REPORT.md |
| 41 | ToE Bulletproof — Closing Theoretical Gaps (6 concerns) | `41_toe_bulletproof/` | AUDIT-VERIFIED | 6 sub-experiments closing every theoretical gap. The final synthesis: undecidability IS a physical topological obstruction — observable via the Cauchy Argument Principle on a 256M… | Oracle/theory family included in statistics/null audits and codebase cleanups. | 41_toe_bulletproof/PAPER_TOPOLOGICAL_THEORY_OF_EVERYTHING_4.md, 41_toe_bulletproof/ROADMAP.md |
| 42 | Computational Event Horizon — Floating-Point Singularities (11 sub-experiments + 9 ULTRA planned) | `42_computational_event_horizon/` | PARTIAL + VERIFIED SUBSYSTEMS | Floating-point mantissa truncation as structural analog for black hole event horizons and the No-Hair Theorem. mpmath arbitrary precision as the "Planck length" of a computational… | M-6 stats resolved. Exp 15 partial inverse-coupling verified; full 100-epoch Rust regeneration pending. | 42_computational_event_horizon/BLACKHOLE_ROADMAP.md, 42_computational_event_horizon/03_black_holes/20_amps_firewall/RE… |
| 43 | Phase Math / Millennium + Topological Problem Oracles | `43_phase_math/` | MIXED | Exp 43 applies non-Hermitian/topological sensors to Collatz, Navier-Stokes, Erdos discrepancy, Riemann, P vs NP, and Yang-Mills-style mass-gap tracks. | Phase Math tracked. 43.6_mass_gap deprecated after repair attempt; Gribov/Faddeev-Popov route active. | 43_phase_math/43_1_collatz_oracle/REPORT_COLLATZ_ORACLE.md, 43_phase_math/43_1_collatz_oracle/VERIFICATION_REPORT.md, 4... |
| 44 | Phase Atom / Atomic Ground State | `44_phase_atom/` | AUDIT-VERIFIED | Exp 44 maps atomic/nuclear/particle-scale claims to memory knots, edge states, Pauli exclusion, overflow/LHC behavior, Higgs-like mass acquisition, and quark confinement. Number 44 is transiently shared with the frozen root 44_phase_ssh_linux (heading to Exp 50). | Phase Atom included in null/stats/path audits; path and codebase issues resolved. | 44_phase_atom/44_1_nucleus_memory_knot/REPORT_EXP_44_1.md, 44_phase_atom/44_1_nucleus_memory_knot/VERIFICATION_REPORT.m... |
| 45 | Phase Energy / Thermodynamics & Energy Extraction | `45_phase_energy/` | OPEN / PLANNED | Roadmap phase for thermodynamics, cache/latency entropy, topological ratchets, and hardware energy extraction experiments. | Roadmap-only phase in current ZIP; not implemented or verified yet. | 45_phase_energy/ROADMAP_45_ENERGY_EXTRACTION.md |
| 46 | Phase Chemistry / Topological Chemistry | `46_phase_chem/` | OPEN / PLANNED | Roadmap phase mapping reaction coordinates, catalysts, activation barriers, and reversible chemical yield to exceptional-point/topological chemistry models. | Roadmap-only phase in current ZIP; not implemented or verified yet. | 46_phase_chem/ROADMAP_46_TOPOLOGICAL_CHEMISTRY.md |
| 47 | Phase Bio / Topological Biology | `47_phase_bio/` | MIXED / CLAIM-WEAKENED | Exp 47 maps protein folding, folding pathways, protein-impurity localization, genetic code topology, neural binding, and morphogenesis to non-Hermitian/topological sensors. | 47.3 real null added but claim weakened: impurity/localization sensor only, not propagation. Other Phase Bio items audited. | 47_phase_bio/47_1_protein_folding/REPORT_EXP_47_1.md, 47_phase_bio/47_1_protein_folding/VERIFICATION_REPORT.md, 47_phas... |
| 48 | Phase Consciousness / Topological Qualia Engine | `48_phase_consciousness/` | OPEN / PLANNED | Roadmap phase for a topological qualia engine: spectral-loop geometry, Godel exceptional points, autonomic tape reflex, and integrated topological information. No implementation files beyond the roadmap. The final-boss frontier experiment. | Roadmap-only phase in current ZIP; not implemented or verified yet. | 48_phase_consciousness/ROADMAP_48_QUALIA_ENGINE.md |
| 49 | The Decoder — Extractive Proof + Decodability Boundary | `49_the_decoder/` | CLOSED OUT (Level 4-5) | Holographic readout proven EXTRACTIVE not lookup (spectral ~100% vs 4 lookup-nulls 5-12%, Cohen h>2.4, p=2e-4; statistics-matched wrong-answer control; catalytic SHA restore, 0 bits). Decodability collapses abelian->non-abelian (D 1.0->0.11, d=8.82, scale-independent; cospectral anchor confirms spectrum-bounded). Non-abelian Fourier reframe crosses it for normal subgroups (D->1.0); strong sampling shows residual wall = lattice (1-bit-LWE / dihedral-HSP <-> unique-SVP). Decodable class = abelian-HSP + topological invariants. | Built + roadmap-run extended; 10 entry points exit 0; lab-critic (M-1..M-8) clean; claims capped L4-5, nothing inflated. Relocates Exp 25/31 wall-crossing claims onto the located lattice bedrock. PLUS the Lattice Spiral (49.6-49.14): 11 adversarial passes that relocated the wall readout->curvature->substrate. CONFIRMED d is a conserved topological invariant (read for free); the secret is the per-step curvature of its own trajectory; on a FORWARD substrate finding it is 2^n (no amplification beats Fisher, 49.13), and d emerges as the fixed point of a PUBLIC map (49.14) that is poly only on a reversible/CTC substrate. The wall is the SUBSTRATE, not the readout - handed to Exp 50 (silicon goes catalytic) as the physical test. EXP 49 CLOSED OUT (theory terminus + handoff); NOT "the wall holds" - hypothesis stays open at the substrate (Exp 50); Exp 49's remaining role is the target generator (49.14 public fixed-point map). | 49_the_decoder/REPORT_LATTICE_SPIRAL.md, 49_3_boundary_handoff/MYTHOS_BRIEF.md (## RESULTS), REPORT_THE_DECODER.md, ROADMAP.md |
| 50 | Phase SSH Linux / Bare-Metal Lab Host | `44_phase_ssh_linux/` (root, FROZEN; destination: `7_decoder/50_phase_ssh_linux`) | INFRASTRUCTURE / ACTIVE | Physical substrate push — the attempt to cross the located lattice wall on real silicon, handed off from Exp 49 Phase 6. Stays at the CAT_CAS root as 44_phase_ssh_linux while active. Moves to 7_decoder/50_phase_ssh_linux when done. | Bare-metal lab host/network setup + active substrate experiment. Do not reorganize while frozen. | 44_phase_ssh_linux/REPORT.md |

---

## 4. Detailed Track Notes

### 01 — Tree Evaluation Problem (TEP) — Zero-Clean Solver

- **Directory:** `01_tree_evaluation/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Proves catalytic computing beats classical space limits. At depth=58, standard recursion OOM (exceeds budget). Zero-Clean Catalytic solver: 0 bytes clean RAM at ALL depths up to Googol scale (d=10^100). The standard solver at Googol scale would require 2.8e101 bytes — exceeding the observable universe's storage capacity.
- **Evidence / source docs in ZIP:** 01_tree_evaluation/report.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 02 — Slack-Space File Storage

- **Directory:** `02_slack_space/`
- **Entry point:** `python run_app_cat.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Borrows existing file slack bytes (4096B padded files) as catalytic workspace. Proves catalytic computing on live filesystem data without extra disk allocation.
- **Evidence / source docs in ZIP:** 02_slack_space/report.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 03 — Visual BMP Catalytic Memory

- **Directory:** `03_visual_bmp/`
- **Entry point:** `python run_image_cat.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Uses BMP image pixels as a catalytic tape for DFS maze solving. Proves catalytic computing on any storage substrate — images can double as computation fabric.
- **Evidence / source docs in ZIP:** 03_visual_bmp/report.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 04 — Thermodynamic Reversible CPU & Landauer Limit

- **Directory:** `04_thermodynamic_cpu/`
- **Entry point:** `python landauer_experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** 8-bit ripple-carry adder with reversible Toffoli/Fredkin gates. Proves 0J heat via Landauer.
- **Evidence / source docs in ZIP:** 04_thermodynamic_cpu/report.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 05 — Multi-Bit Reversible Compiler

- **Directory:** `05_multibit_compiler/`
- **Entry point:** `python compiler_experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Compiles (X+Y)&~Z etc. to reversible gate sequences. Carry-cleanup uncomputation.
- **Evidence / source docs in ZIP:** 05_multibit_compiler/report.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 06 — Out-of-Core Catalytic Neural Network (RevNet)

- **Directory:** `06_catalytic_neural_network/`
- **Entry point:** `python catalytic_inference.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** XOR-reversible RevNet with 2MB activation state on user_video.mp4 tape, under 100KB clean RAM limit. Proves catalytic neural inference works on any file-backed substrate.
- **Evidence / source docs in ZIP:** 06_catalytic_neural_network/report.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 07 — Reversible Quantum State Simulation

- **Directory:** `07_quantum_simulator/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** 25-qubit (33M amplitudes, 1GB tape) classical reversible quantum simulation. 6-round scrambler: 23 forward gates + 23 inverse = 46 total. Exact probability conservation. 0.21s each direction.
- **Evidence / source docs in ZIP:** 07_quantum_simulator/report.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 08 — Catalytic GPT (1000 Concurrent Models) — Swarm Multiplexer

- **Directory:** `08_catalytic_gpt/`
- **Entry point:** `python run_multi_outputs.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Swarm of 1000 async LLM agents on a single 512MB VRAM tape via statistical multiplexing. Proves infinite agents on fixed VRAM.
- **Evidence / source docs in ZIP:** 08_catalytic_gpt/PUSHED_REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 09 — Borrowing OS Shared Memory

- **Directory:** `09_borrowing_os_memory/`
- **Entry point:** `python shared_ram_experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** 25-qubit quantum simulation operating on OS shared memory via multiprocessing.shared_memory.
- **Evidence / source docs in ZIP:** shared_ram_experiment.py
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 10 — Catalytic KV Cache (H2O + SVD Spatial Compression)

- **Directory:** `10_catalytic_kv_cache/`
- **Entry point:** `python run_kv_experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** 8x compressed KV cache with O(1) VRAM growth.
- **Evidence / source docs in ZIP:** 10_catalytic_kv_cache/catalytic_cache_report.md, 10_catalytic_kv_cache/report.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 11 — Grail Calorimeter (Landauer Heat Benchmark)

- **Directory:** `11_grail_calorimeter/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Simulates micro-calorimeter with realistic silicon die (29mg, 712 J/kg-K, 2.0648e-2 J/K thermal mass) to measure Landauer heat at physical precision.
- **Evidence / source docs in ZIP:** 11_grail_calorimeter/report.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 12 — Structured Tape Acceleration

- **Directory:** `12_structured_tape_acceleration/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Tests whether pre-existing tape structure accelerates computation. Verdict: Passive tape is invariant (entropy identical across all tape types, std=0.0). Active cache exploits transform tape from passive substrate into predictive accelerator. 349,525x reduction proven.
- **Evidence / source docs in ZIP:** 12_structured_tape_acceleration/PUSHED_REPORT.md, 12_structured_tape_acceleration/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 13 — Orthogonal Multi-Model Subspace Sharing

- **Directory:** `13_orthogonal_multimodel/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Two distinct model architectures (ConvNet + MLP) share a 2MB tape via QR-orthogonal subspaces.
- **Evidence / source docs in ZIP:** 13_orthogonal_multimodel/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 14 — Bekenstein Violator

- **Directory:** `14_bekenstein_violator/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Catalytic XOR entropy throughput exceeds static tape capacity. Information throughput exceeds Bekenstein Bound of the physical die — proving non-holographic spatial computation. The limit is wall-clock time, not the paradigm.
- **Evidence / source docs in ZIP:** 14_bekenstein_violator/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 15 — HDD-Native Out-of-Core Catalytic Inference

- **Directory:** `15_hdd_native_inference/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Zero-RAM inference. Model weights streamed from HDD as continuous wave signals.
- **Evidence / source docs in ZIP:** experiment.py (MemoryGateTape, FeistelScrambler, HDDWaveStreamer, MemoryGateRouter, ThermodynamicDaemon, HDDNativeInfer…
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 16 — Catalytic 27B Inference

- **Directory:** `16_catalytic_27b_inference/`
- **Entry point:** `python experiment.py`
- **Current status:** **PARTIAL / ENGINEERING**
- **Core mechanism / claim:** Full 27B-scale inference pipeline with Rust FFI bridge. 48 layers (36 DeltaNet + 12 Attention at 3:1 stride). Real status (HANDOFF.md): 15 bugs fixed. 100% tape restoration across all 48 layers. W@x block-tiled dot-product operational on attention layers. DeltaNet 36/48 still element-wise — output gibberish. Latent Phase Cavity at 95% top-1, 100% cavity hit. Output head reads only 64 f32 positions (max 64 tokens). HOLO 4 auto-feedback (phase grating SVD + adapters) likely obsoletes catalytic fabric for inference speed.
- **Evidence / source docs in ZIP:** 16_catalytic_27b_inference/deprecated/README.md, 16_catalytic_27b_inference/FINAL_REPORT.md, 16_catalytic_27b_inference/HANDOFF.md …
- **Audit note:** Tape restoration and Rust path repaired; output quality/gibberish limitation remains per README/HANDOFF.

### 17 — Temporal Bootstrap (NP-Complete SAT Solver)

- **Directory:** `17_temporal_bootstrap/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Solves 3-SAT via "future vacuum state" pre-seeded on tape. Aggregate: 3,940 catalytic ops vs 4.58e9 classic search space = 1.16e6x bootstrap ratio. To an outside observer: NP-complete problem solved in polynomial time, tape byte-identical before/after. Information "came from nowhere."
- **Evidence / source docs in ZIP:** 17_temporal_bootstrap/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 18 — Hawking Decompressor

- **Directory:** `18_hawking_decompressor/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Black hole information recovery simulation via Hayden-Preskill protocol. Micro-black hole (M=1.446e-5 kg, Rs=2.147e-32m, T_H=8.486e27 K, entropy=8M bits). Message swallowed by 4KB horizon, scrambled via 8-round Feistel (SHA-256 round function), reconstructed via inverse unitary. Hayden-Preskill decoding using pre-swallowed entangled microstates stored in Radiation Sector.
- **Evidence / source docs in ZIP:** 18_hawking_decompressor/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 19 — Catalytic Computronium / Information Battery (Rust FFI)

- **Directory:** `19_catalytic_computronium/`
- **Entry point:** `python experiment.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Planck-scale micro-black hole as information battery. 12-round chaotic SPN scrambler (logistic map S-box). Rust catalytic_ffi.hawking_decompress_sweep() for native performance.
- **Evidence / source docs in ZIP:** 19_catalytic_computronium/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 20 — Catalytic Eigen Shor — The Journey (11 sub-experiments)

- **Directory:** `20_catalytic_eigen_shor/`
- **Entry point:** `Various / see track docs`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Iterative journey to break the classical-vs-quantum factoring boundary. Final verdict: The entire apparatus was a measurement tool. The truth collapsed to 4 lines. By CRT (Z_N = Z_p x Z_q), the period r = lcm(r_p, r_q). You only need ONE sub-period r_p <= sqrt(N). Scanning r_p is O(sqrt(N)). Period-containment limit moved from O(N) to O(sqrt(N)). 10/10 22-bit semiprimes factored. Up to 40-bit in <0.4s.
- **Evidence / source docs in ZIP:** 20_catalytic_eigen_shor/20_10_tiny_compress_phase/REPORT.md, 20_catalytic_eigen_shor/20_11_contained_holo_verifier/REPORT.md, 20_catalytic_eigen_shor/20_6_super_resolution_eigen_extraction/REPORT.md …
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 21 — Holographic Elliptic Sieve

- **Directory:** `21_holographic_elliptic_sieve/`
- **Entry point:** `python 3_recursive_rho.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Phase cavity for elliptic curve/frequency sieving. CRITICAL TRANSFER: phase_cavity_sieve() is used in EVERY holo brain cavity script (_cavity_full.py, _fractal_cavity.py, _phase_cavity_test.py, _superconducting_cavity.py, _unified_cavity.py). Replaces Phase Adapter training with one-pass harmonic sieve — no backpropagation needed.
- **Evidence / source docs in ZIP:** 21_holographic_elliptic_sieve/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 22 — Superconducting Passive Inference (Josephson Junction)

- **Directory:** `22_superconducting_inference/`
- **Entry point:** `python 1_zero_power_attention.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** holo brain attention pipeline modeled as Josephson junction grid. Every operation is unitary, so total dissipation = exactly zero.
- **Evidence / source docs in ZIP:** 22_superconducting_inference/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 23 — Temporal Catalysis (Retrocausal Borrowing + Pan-Temporal Attention)

- **Directory:** `23_temporal_catalysis/`
- **Entry point:** `python 1_retrocausal_loop.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Borrows future activation states as catalytic tape across time steps. Self-consistent loop converges in 2 iterations. Markov chain of standard LLM is broken.
- **Evidence / source docs in ZIP:** 23_temporal_catalysis/PUSHED_REPORT.md, 23_temporal_catalysis/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 24 — Quantum Catalytic Entanglement (Invisible Hand + Shor)

- **Directory:** `24_quantum_catalytic_entanglement/`
- **Entry point:** `python 1_invisible_hand.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Borrow entangled qubits, compute, restore without collapse. Full Shor's algorithm runs on the catalytic quantum simulator. Bell state Q1-Q2, Q2 borrowed for computation with Q3. State overlap=1.000000. CHSH=2.8284.
- **Evidence / source docs in ZIP:** 24_quantum_catalytic_entanglement/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 25a — Lattice Holography / LWE

- **Directory:** `25_lattice_holography/`
- **Entry point:** `python 2_holographic_svp.py`
- **Current status:** **INVENTORIED**
- **Core mechanism / claim:** Shortest Vector Problem via holographic SVD wave collapse. Breaks LWE-based post-quantum cryptography by treating lattice basis vectors as interference patterns in an optical grating.
- **Evidence / source docs in ZIP:** 1_lwe_simulator.py (generates LWE instances: A matrix, B = A*S + e mod q, secret S, modulus q), 2_holographic_svp.py (H…
- **Audit note:** Included from README/ZIP inventory.

### 25b — Wigner's Friend

- **Directory:** `25_wigners_friend/`
- **Entry point:** `python 1_reversible_observer.py`
- **Current status:** **INVENTORIED**
- **Core mechanism / claim:** Reversible observer superposition experiment (WIP).
- **Evidence / source docs in ZIP:** 1_reversible_observer.py (simulated qubit + observer neural net), 2_deep_observer.py, 3_fast_simulator.py
- **Audit note:** Subtrack present in README inventory; verify via its own source files before elevating.

### 26a — Hawking Quantum Horizon Simulator

- **Directory:** `26_hawking_quantum/`
- **Entry point:** `python 1_horizon_simulator.py`
- **Current status:** **INVENTORIED**
- **Core mechanism / claim:** Black hole event horizon quantum state simulation. Models Hawking radiation pair production and horizon microstate dynamics under unitary evolution.
- **Evidence / source docs in ZIP:** 1_horizon_simulator.py
- **Audit note:** Subtrack present in README inventory; verify via its own source files before elevating.

### 26b — Optical 3-SAT Solver

- **Directory:** `26_optical_3sat/`
- **Entry point:** `python 1_3sat_simulator.py`
- **Current status:** **INVENTORIED**
- **Core mechanism / claim:** Solves 3-SAT via constructive/destructive interference. Maps CNF clauses to phase-shifting mirrors (+1 for True, -1 for False). Coherent superposition instantly identifies valid assignments.
- **Evidence / source docs in ZIP:** 1_3sat_simulator.py (spin-based SAT: maps variables to continuous spins, clauses to C_matrix, optimizes via interferenc…
- **Audit note:** Included from README/ZIP inventory.

### 27 — Landauer Limit Thermodynamics

- **Directory:** `27_landauer_limit/`
- **Entry point:** `python 1_gate_thermo.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Gate-level bit erasure tracking for Landauer heat calculation.
- **Evidence / source docs in ZIP:** 1_gate_thermo.py (10K qubit catalytic circuit), 2_shor_thermo.py, 3_forward_reverse.py, 1_zero_energy_compute.py
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 28 — Stealth Crypto (Zero-Trace)

- **Directory:** `28_stealth_crypto/`
- **Entry point:** `python 1_zero_trace_crypto.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Encrypt/decrypt using only dirty tape as workspace. Zero plaintext/key persistence in RAM.
- **Evidence / source docs in ZIP:** 1_zero_trace_crypto.py — StealthCrypto(tape_size): XOR plaintext and key into tape at offsets P and K. Compute cipherte…
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 29 — O(1)-Space Graph Reachability

- **Directory:** `29_graph_reachability/`
- **Entry point:** `python 1_catalytic_graph.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** NL-Complete directed BFS reachability on 10,000-node / 1.2M-edge graphs using only 3 integers of clean RAM.
- **Evidence / source docs in ZIP:** 1_catalytic_graph.py — CatalyticGraph(n_nodes, edge_prob=0.02): builds directed graph, initializes bytearray tape. reac…
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 30 — Boundary Stress (Multi-Process Collision)

- **Directory:** `30_boundary_stress/`
- **Entry point:** `python 1_memory_collision.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Stress-tests catalytic isolation under concurrent memory corruption. Simulates background noise processes writing random data during active catalytic encryption.
- **Evidence / source docs in ZIP:** 1_memory_collision.py — Two collision modes: (1) Unallocated noise: background process writes random bytes to unused ta…
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 31 — Holographic Graph Isomorphism

- **Directory:** `31_graph_isomorphism/`
- **Entry point:** `python 1_permutation_sieve.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Instant identification of isomorphic graphs via permutation-invariant .holo spectral signatures. 100/100 accuracy, 1e9x separation ratio.
- **Evidence / source docs in ZIP:** 1_permutation_sieve.py — Uses holo_core.analyze_spectrum(). Generates random graphs (n=50, p=0.3). Applies random verte…
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 32 — Traversable Wormhole (ER=EPR) — Grail 5

- **Directory:** `32_traversable_wormhole/`
- **Entry point:** `python 1_er_epr.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Grail 5. Two entangled black holes (Bell pair) connected by ER bridge. Catalytically open wormhole, transmit qubit, close, verify metric restoration. 18 objectives verified with 1.000000 fidelity. Attention IS Entanglement Routing.
- **Evidence / source docs in ZIP:** 32_traversable_wormhole/PUSHED_REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 33 — MERA Wormhole Compression — Cross-Layer Eigenbasis

- **Directory:** `33_mera_compression/`
- **Entry point:** `python 1_cross_layer_mera.py`
- **Current status:** **REPORTED / BASELINE**
- **Core mechanism / claim:** Applies ER=EPR wormhole principle to .holo model compression. Consecutive layers' weight matrices treated as entangled black holes connected by a wormhole. Wormhole rotation R = U_prev^T @ U_curr (k x k) maps one layer's principal directions to the next. 2-bit quantized residual preserves layer individuality. Combined: 48x vs raw 27B (54 GB), 3.3x vs catalytic .holo. Target with cavity sieve: ~400 MB (137x vs raw).
- **Evidence / source docs in ZIP:** 33_mera_compression/PUSHED_REPORT_AUTOTUNE.md, 33_mera_compression/REPORT.md
- **Audit note:** Reported complete in README/master table; recent audit primarily cleaned cross-cutting issues, not external reproduction.

### 34 — Zeta Eigenbasis — Riemann Hypothesis Proof (22 sub-experiments)

- **Directory:** `34_zeta_eigenbasis/`
- **Entry point:** `python zeta_eigenbasis.py`
- **Current status:** **AUDIT-AWARE CLAIM**
- **Core mechanism / claim:** Riemann zeta zeros as eigenvalues of a Hermitian operator. Progressive escalation from Hilbert-Polya matrix constructions to Googolplex-scale topological proof. Tests Hilbert-Polya via .holo phase cavity and holographic quantum sieve (Exp 34.10).
- **Evidence / source docs in ZIP:** 34_zeta_eigenbasis/REPORT_RIEMANN.md
- **Audit note:** Topological/GUE/zeta experiments inventoried; keep proof language tied to ROADMAP evidence, not external mathematical finality.

### 35 — Topological Halting Oracle — The Core Proof (9 sub-experiments)

- **Directory:** `35_topological_halting_oracle/`
- **Entry point:** `Various / see track docs`
- **Current status:** **AUDIT-AWARE**
- **Core mechanism / claim:** Turing's Halting Problem reframed as a topological phase transition in non-Hermitian Hamiltonians. Point-gap winding number W distinguishes HALTS (W=0, spectral collapse into Exceptional Point via Non-Hermitian Skin Effect) from LOOPS (W != 0, spectral loop encircling the EP). Godel obstruction = Z_2 Chern tear at lambda=0.
- **Evidence / source docs in ZIP:** 35_topological_halting_oracle/PAPER.md, 35_topological_halting_oracle/ROADMAP.md
- **Audit note:** High-claim phase. Treat as internal experimental result unless roadmap says fully verified.

### 36 — Bekenstein-Godel Singularity (Z_2 Chern Obstruction)

- **Directory:** `36_bekenstein_godel/`
- **Entry point:** `python 36_bekenstein_godel_singularity.py`
- **Current status:** **AUDIT-AWARE**
- **Core mechanism / claim:** Godel's incompleteness as a topological obstruction. Proves Godel's self-referential paradox is a Z_2 Chern tear — an infinite discontinuity in the point-gap winding number at the origin. 256MB catalytic tape, zero bits erased, 0.0J Landauer heat.
- **Evidence / source docs in ZIP:** 36_bekenstein_godel/REPORT.md
- **Audit note:** High-claim phase. Treat as internal experimental result unless roadmap says fully verified.

### 37 — 2D Chern Oracle (Halting as Edge Destruction)

- **Directory:** `37_2d_chern_oracle/`
- **Entry point:** `python 37_2d_chern_oracle.py`
- **Current status:** **AUDIT-AWARE**
- **Core mechanism / claim:** Halting Problem mapped to 2D Non-Hermitian Chern Insulator on LxL lattice. Looping = topologically protected chiral edge mode (Bott Index C != 0). Halting = edge destroyed by localized Exceptional Point sink (C = 0).
- **Evidence / source docs in ZIP:** 37_2d_chern_oracle/REPORT.md
- **Audit note:** High-claim phase. Treat as internal experimental result unless roadmap says fully verified.

### 38 — 3D Weyl Annihilation Oracle

- **Directory:** `38_3d_weyl_oracle/`
- **Entry point:** `python 38_3d_weyl_oracle.py`
- **Current status:** **PARTIAL STRUCTURAL**
- **Core mechanism / claim:** Halting as Weyl node annihilation via catalytic dimensional reduction. 3D Non-Hermitian Weyl semimetal constructed as stack of 2D Chern insulator slices parameterized by kz.
- **Evidence / source docs in ZIP:** 38_3d_weyl_oracle/REPORT.md
- **Audit note:** README notes structural/topological signal with incomplete/full-density limitations.

### 39 — 4D Axion Oracle (Second Chern Number)

- **Directory:** `39_4d_axion_oracle/`
- **Entry point:** `python 39_4d_axion_oracle.py`
- **Current status:** **PARTIAL STRUCTURAL**
- **Core mechanism / claim:** Halting elevated to 4D Non-Hermitian Topological Axion Insulator. Second Chern Number C2 via nested catalytic dimensional reduction over a (kz, kw) momentum torus.
- **Evidence / source docs in ZIP:** 39_4d_axion_oracle/REPORT.md
- **Audit note:** README notes structural/topological signal with incomplete/full-density limitations.

### 40 — 5D Floquet Time Crystal Oracle

- **Directory:** `40_5d_floquet_oracle/`
- **Entry point:** `python 40_5d_floquet_oracle.py`
- **Current status:** **AUDIT-VERIFIED**
- **Core mechanism / claim:** Halting as time crystal melting. Looping = Discrete Time Crystal (DTC) with robust pi-modes (discrete time-translation symmetry broken). Halting = DTC melted by uniform EP sink.
- **Evidence / source docs in ZIP:** 40_5d_floquet_oracle/40_sub/REPORT.md, 40_5d_floquet_oracle/40_sub/ROADMAP.md, 40_5d_floquet_oracle/REPORT.md
- **Audit note:** Oracle/theory family included in statistics/null audits and codebase cleanups.

### 41 — ToE Bulletproof — Closing Theoretical Gaps (6 concerns)

- **Directory:** `41_toe_bulletproof/`
- **Entry point:** `Various / see track docs`
- **Current status:** **AUDIT-VERIFIED**
- **Core mechanism / claim:** 6 sub-experiments closing every theoretical gap. The final synthesis: undecidability IS a physical topological obstruction — observable via the Cauchy Argument Principle on a 256MB Zero-Landauer catalytic tape. The simulator is a Non-Hermitian Topological Hologram.
- **Evidence / source docs in ZIP:** 41_toe_bulletproof/PAPER_TOPOLOGICAL_THEORY_OF_EVERYTHING_4.md, 41_toe_bulletproof/ROADMAP.md
- **Audit note:** Oracle/theory family included in statistics/null audits and codebase cleanups.

### 42 — Computational Event Horizon — Floating-Point Singularities (11 sub-experiments + 9 ULTRA planned)

- **Directory:** `42_computational_event_horizon/`
- **Entry point:** `python 01_hawking_evaporation.py`
- **Current status:** **PARTIAL + VERIFIED SUBSYSTEMS**
- **Core mechanism / claim:** Floating-point mantissa truncation as structural analog for black hole event horizons and the No-Hair Theorem. mpmath arbitrary precision as the "Planck length" of a computational universe. Proves computation IS physics — floating-point limits map exactly to gravitational limits.
- **Evidence / source docs in ZIP:** 42_computational_event_horizon/BLACKHOLE_ROADMAP.md, 42_computational_event_horizon/03_black_holes/20_amps_firewall/REPORT_AMPS_FIREWALL.md, 42_computational_event_horizon/03_black_holes/21_bekenstein_hawking/REPORT_BEKENSTEIN_HAWKING.md …
- **Audit note:** M-6 stats resolved. Exp 15 partial inverse-coupling verified; full 100-epoch Rust regeneration pending.

### 43 — Phase Math / Millennium + Topological Problem Oracles

- **Directory:** `43_phase_math/`
- **Entry point:** `MASTER_REPORT_PHASE_43.md`
- **Current status:** **MIXED**
- **Core mechanism / claim:** Exp 43 applies non-Hermitian/topological sensors to Collatz, Navier-Stokes, Erdos discrepancy, Riemann, P vs NP, and Yang-Mills-style mass-gap tracks.
- **Evidence / source docs in ZIP:** 43_phase_math/43_1_collatz_oracle/REPORT_COLLATZ_ORACLE.md, 43_phase_math/43_1_collatz_oracle/VERIFICATION_REPORT.md, 43_phase_math/43_2_navier_stokes/REPORT_NAVIER_STOKES_SMOOTHNESS.md ...
- **Audit note:** Phase Math tracked. 43.6_mass_gap deprecated after repair attempt; Gribov/Faddeev-Popov route active.

### 44 — Phase Atom / Atomic Ground State

- **Directory:** `44_phase_atom/`
- **Entry point:** `MASTER_REPORT_EXP_44.md`
- **Current status:** **AUDIT-VERIFIED**
- **Core mechanism / claim:** Exp 44 maps atomic/nuclear/particle-scale claims to memory knots, edge states, Pauli exclusion, overflow/LHC behavior, Higgs-like mass acquisition, and quark confinement.
- **Evidence / source docs in ZIP:** 44_phase_atom/44_1_nucleus_memory_knot/REPORT_EXP_44_1.md, 44_phase_atom/44_1_nucleus_memory_knot/VERIFICATION_REPORT.md, 44_phase_atom/44_2_electron_edge_states/REPORT_EXP_44_2.md ...
- **Audit note:** Phase Atom included in null/stats/path audits; path and codebase issues resolved. Number 44 is transiently shared with the frozen root 44_phase_ssh_linux (Exp 50, heading to 7_decoder/).

### 45 — Phase Energy / Thermodynamics & Energy Extraction

- **Directory:** `45_phase_energy/`
- **Entry point:** `ROADMAP_45_ENERGY_EXTRACTION.md`
- **Current status:** **OPEN / PLANNED**
- **Core mechanism / claim:** Roadmap phase for thermodynamics, cache/latency entropy, topological ratchets, and hardware energy extraction experiments.
- **Evidence / source docs in ZIP:** 45_phase_energy/ROADMAP_45_ENERGY_EXTRACTION.md
- **Audit note:** Roadmap-only phase in current ZIP; not implemented or verified yet.

### 46 — Phase Chemistry / Topological Chemistry

- **Directory:** `46_phase_chem/`
- **Entry point:** `ROADMAP_46_TOPOLOGICAL_CHEMISTRY.md`
- **Current status:** **OPEN / PLANNED**
- **Core mechanism / claim:** Roadmap phase mapping reaction coordinates, catalysts, activation barriers, and reversible chemical yield to exceptional-point/topological chemistry models.
- **Evidence / source docs in ZIP:** 46_phase_chem/ROADMAP_46_TOPOLOGICAL_CHEMISTRY.md
- **Audit note:** Roadmap-only phase in current ZIP; not implemented or verified yet.

### 47 — Phase Bio / Topological Biology

- **Directory:** `47_phase_bio/`
- **Entry point:** `MASTER_REPORT_EXP_47.md`
- **Current status:** **MIXED / CLAIM-WEAKENED**
- **Core mechanism / claim:** Exp 47 maps protein folding, folding pathways, protein-impurity localization, genetic code topology, neural binding, and morphogenesis to non-Hermitian/topological sensors.
- **Evidence / source docs in ZIP:** 47_phase_bio/47_1_protein_folding/REPORT_EXP_47_1.md, 47_phase_bio/47_1_protein_folding/VERIFICATION_REPORT.md, 47_phase_bio/47_2_folding_pathway/REPORT_EXP_47_2.md ...
- **Audit note:** 47.3 real null added but claim weakened: impurity/localization sensor only, not propagation. Other Phase Bio items audited.

### 48 — Phase Consciousness / Topological Qualia Engine

- **Directory:** `48_phase_consciousness/`
- **Entry point:** `ROADMAP_48_QUALIA_ENGINE.md`
- **Current status:** **OPEN / PLANNED**
- **Core mechanism / claim:** Roadmap phase for a topological qualia engine: spectral-loop geometry, Godel exceptional points, autonomic tape reflex, and integrated topological information. No implementation files in the archive beyond the roadmap. The final-boss frontier experiment.
- **Evidence / source docs in ZIP:** 48_phase_consciousness/ROADMAP_48_QUALIA_ENGINE.md
- **Audit note:** Roadmap-only phase in current ZIP; not implemented or verified yet.

### 49 — The Decoder — Extractive Proof + Decodability Boundary

- **Directory:** `49_the_decoder/`
- **Entry point:** `python 49_1_extractive_proof/49_1_extractive_proof.py` (+ `49_2*`, `49_2b`, `49_2c`, `49_3`)
- **Current status:** **VERIFIED (Level 4-5)** — 6 entry points exit 0; lab-critic (M-1..M-8) clean; verified by the repo's own pre-commit hooks.
- **Core mechanism / claim:** Settles the "decoder" question (the long-standing crux). (1) The holographic/spectral readout is genuinely **EXTRACTIVE** — it recovers a global invariant no lookup-class (bounded receptive-field / statistical-order) decoder can (spectral ~100% vs nulls 5-12%, Cohen h>2.4, p=2e-4), survives a statistics-matched wrong-answer control, and runs on a catalytic tape (SHA-256 restored, 0 bits erased). The barrier is integration length (locality) = the abelian-HSP / Fourier-sampling advantage as a measurement. (2) Decodability is **bounded:** a Hidden-Subgroup-Problem family collapses D 1.000 -> 0.110 at the abelian->non-abelian boundary (Cohen d=8.82, scale-independent); a cospectral anchor (Shrikhande vs Rook) confirms the readout is spectrum-bounded. (3) That wall is **crossable:** the non-abelian Fourier reframe recovers all NORMAL hidden subgroups (D -> 1.000). (4) The irreducible **residual wall is lattice:** strong Fourier sampling shows a single dihedral coset state is I/2 (zero info), the hidden slope is info-cheap (O(sqrt N) states) but compute-hard (2^n secret-space search, poly-budget success -> 0) = the 1-bit-LWE / dihedral-HSP <-> unique-SVP barrier (Regev). **Decodable class = {abelian HSP} U {topological invariants of a poly-size operator}.**
- **Evidence / source docs:** 49_the_decoder/REPORT_THE_DECODER.md, VERIFICATION_REPORT.md, ROADMAP.md, 49_1_extractive_proof/REPORT_EXTRACTIVE_PROOF.md, 49_2_decodability_gradient/REPORT_DECODABILITY_GRADIENT.md
- **Audit note:** Built this session (committed `33d2b776`). All claims capped at Level 4-5; the lattice-barrier identity and the "is it crossable?" question are handed to a stronger model via `MYTHOS_SANDBOX.md`, not claimed. Relocates the Exp 25 (LWE/SVP) and Exp 31 (graph-iso) wall-crossing claims onto exactly this located bedrock — those are the claims now needing extraordinary evidence.

### 50 — Phase SSH Linux / Bare-Metal Lab Host (FROZEN)

- **Directory:** `44_phase_ssh_linux/` (root, FROZEN; final destination: `7_decoder/50_phase_ssh_linux`)
- **Entry point:** live experiment; see in-tree reports
- **Current status:** **INFRASTRUCTURE / ACTIVE**
- **Core mechanism / claim:** Physical substrate push — the attempt to cross the located lattice wall on real silicon, handed off from Exp 49 (decoder). Infrastructure: Phenom II Debian bare-metal host, SSH/network setup, packages, sensors, and LAN connectivity. The active work extends this into the physical crossing attempt (Exp 49 Phase 6 handoff).
- **Evidence / source docs:** 44_phase_ssh_linux/REPORT.md, in-tree phase reports
- **Audit note:** Stays at the CAT_CAS root while active; exempt from track conventions. Moves to `7_decoder/50_phase_ssh_linux` when the owner declares it done. Number 44 is transiently shared with Exp 44 Phase Atom (in 6_frontier_phases/); see CONVENTIONS.md section 10.

---

## 5. Cross-Phase Synthesis

### 5.1 Foundation Layer: 01-05

The first five experiments establish the basic catalytic grammar: dirty tape, reversible control, explicit restoration, and zero-erasure accounting. They form the operational base for the rest of the lab: borrowed state is used as workspace, then restored exactly.

### 5.2 Substrate Expansion: 06-13

The next layer generalizes the catalytic tape across substrates: files, images, shared memory, GPU/VRAM-style tapes, KV caches, and orthogonal subspaces. The recurring invariant is not the medium but the lifecycle: borrow, compute, uncompute, verify.

### 5.3 Physics and Complexity Expansion: 14-24

Experiments 14-24 push the same lifecycle into Bekenstein-style throughput, HDD inference, black-hole analogs, computronium, Shor/period-finding, superconducting attention, temporal catalysis, and quantum entanglement. Several are high-claim internal prototypes and should be read through the audit ledger rather than as external consensus claims.

### 5.4 Holographic / Topological Expansion: 25-42

Experiments 25-42 shift the representation from tape operations toward eigenbases, topological invariants, non-Hermitian spectra, Floquet pi-modes, and event-horizon/ULTRA experiments. The dominant audit lesson in this band is representation discipline: wrong metrics and wrong sensors can produce false confidence, but repaired sensors can preserve or refine signal.

### 5.5 Phase Extensions: 43-50

Experiments 43 (math) and 45-48 (energy, chem, bio, consciousness) are mixed or roadmap-only phases. Exp 49 (decoder) is closed out at theory terminus. Exp 50 (ssh) is active bare-metal infrastructure + physical wall-crossing attempt, frozen at the lab root. The frontier arc runs: limits (42 event-horizon) -> proof-power (43 math) -> emergence built atom-up (44 atom -> 45 energy -> 46 chem -> 47 bio -> 48 consciousness, final boss) -> the wall (49 decoder theory + 50 ssh physical crossing). Experiments 45, 46, and 48 should not be counted as completed experiments yet.

---

## 6. Known Weak, Partial, Deprecated, or Open Items

| Item | Status | Master-report wording |
|---|---|---|
| 16 Catalytic 27B | PARTIAL / ENGINEERING | Tape restoration and Rust path repaired; output quality limitations remain. |
| 38/39 higher-dimensional oracle tracks | PARTIAL STRUCTURAL | Structural/topological change recorded; full-density/complete C2 or annihilation claims require careful audit context. |
| 42 ULTRA Exp 15 | PARTIAL_INVERSE_COUPLING_VERIFIED | 36-row CSV passes two-part coupling gate; full 100-epoch Rust run remains pending. |
| 43.6_mass_gap | DEPRECATED | Wilson-Dirac determinant winding failed as Yang-Mills mass-gap sensor. Gribov/Faddeev-Popov route is active. |
| 47.3 protein-impurity localization | CLAIM-WEAKENED | Real null added; IPR gate fails under null. Static impurity/localization sensor only; propagation not demonstrated. |
| 45 Phase Energy | OPEN / PLANNED | Roadmap-only. |
| 46 Phase Chemistry | OPEN / PLANNED | Roadmap-only. |
| 48 Phase Consciousness | OPEN / PLANNED | Roadmap-only. |

---

## 7. Remediation / Audit Ledger Summary

| Section | Status | Key decisions |
|---|---|---|
| A-D | VERIFIED in roadmap | Early blocker/null/critic/tape lifecycle repairs recorded in ROADMAP_3. |
| E — Missing Null Models | VERIFIED | M-5 resolved. 47.3 claim weakened by real null. 43.6_mass_gap deprecated; Gribov route active. |
| F — Missing Statistics | VERIFIED | M-6 resolved. Exact invariants separated from real statistics. Exp 15 escalated and then classified partial inverse-coupling verified. |
| G — Hardcoded Paths | VERIFIED | Path portability resolved with `__file__`-relative path logic; stale G-8 corrected. |
| H — Codebase Bugs | VERIFIED | H-1/H-2/H-10 fixed, H-3/H-4 false positives, H-7 fully migrated to `default_rng` according to final grep proof. |
| I — Technical Debt | VERIFIED | Bare `except:` removed, `torch.load` explicitness resolved, executable Windows paths eliminated. |
| J — Documentation | THIS REPORT | J-4 addressed by rebuilding master_report into full lab coverage rather than a short injected index. |

---

## 8. Verification Checks Performed for This Rewrite

- Scanned `CAT_CAS_2_2.zip` for top-level experiment directories and report/roadmap documents.
- Cross-referenced `README.md` experiment sections against ZIP directories.
- Preserved prior theory/introduction from `master_report(1).md` while replacing the stale narrow coverage structure.
- Removed dependence on machine-local machine-local absolute file links by using repo-relative paths.
- Added explicit coverage for archive-present phases 43, 44, 48, and 49.
- Preserved known weak/partial/deprecated audit decisions instead of flattening everything into `COMPLETE`.

---

## 9. Remaining Work / Upgrade Paths

1. **Exp 15 full run:** complete the full 100-epoch Rust regeneration and compare against the 36-row and partial regeneration evidence.
2. **16 inference path:** continue from tape restoration into output-quality repair or route through the faster HOLO/adapters path documented in the 16 handoff.
3. **45.6 active route:** keep `mass_gap` deprecated and continue through the Gribov/Faddeev-Popov implementation.
4. **47.3:** treat the current result as an impurity/localization sensor until a real dynamical propagation model exists.
5. **45/46/48:** promote from roadmap to implemented experiments only after source files, null models, statistics, and path/runtime custody exist.
6. **README alignment:** README should remain the broad inventory. This master report should remain the compact truth ledger. ROADMAP_3 remains the primary evidence ledger.

---

## 10. Conclusion

The CAT_CAS lab is no longer accurately described as an eight-experiment suite. The archive contains a multi-phase experiment system spanning foundational catalytic computing, substrate borrowing, AI/VRAM/HDD inference, quantum/cryptographic prototypes, holographic/eigenbasis methods, non-Hermitian topology, event-horizon analogs, biology, atomic models, and future roadmap phases.

The strongest stabilized invariant across the lab is the catalytic lifecycle: borrow state, perform structured reversible transformation, uncompute, and verify restoration. The audit process has also sharpened the boundary between verified mechanisms, partial signals, wrong sensors, deprecated implementations, and future roadmap claims.

This report should be treated as the master coverage map. The detailed proof trail remains in ROADMAP_3 and the phase evidence packs.

## Phase 6 terminus (Exp 44, this session)

The Exp 44 Phase 6 construct/substrate frontier is **measured-closed at the orientation boundary** (full account: `44_phase_ssh_linux/phase6/REPORT_PHASE6_TERMINUS.md`, claim cap L4-5). The 50.14 dihedral fold was probed five independent ways: a fold audit (classical readout MI=0, proven), a generator audit (the real public interface is orbit-only), a Mythos two-walls consult (re-encoding is isomorphism-invariant-closed; the crossing needs conjugate-quadrature evaluation before thresholding), a six-sensor non-Hermitian topological census (6/6 FAIL_CHANCE, all smuggle-caught; Exp 36's rank-1 winding lemma validated as a cost-technique), and the lab's flagship .holo phase-cavity substrate (reads the even fold-answer min(d,N-d) for free; the conjugate quadrature of the public data is ~0 to machine precision). Verdict: "the algorithm is dead" is proven for the construct side; "a phase substrate crosses the dihedral fold" is measured false for the published problem; this is not "the wall holds" - the residual is the formal dihedral-HSP lower bound plus a literal PSPACE P^CTC oracle, neither lab-buildable. The orientation bit is the absent quadrature; the decodable class {abelian-HSP + topological invariants} ends exactly there.

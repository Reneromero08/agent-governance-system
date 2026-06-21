# CAT_CAS Map

The whole lab on one page. Experiments are grouped into eight thematic **tracks**
(see [CONVENTIONS.md](CONVENTIONS.md) §2). Each keeps its permanent number; the
track is its folder. Authoritative status lives in
[MASTER_REPORT.md](../MASTER_REPORT.md) and
[REPORTS/VIOLATIONS/ROADMAP_3_VERIFIED.md](REPORTS/VIOLATIONS/ROADMAP_3_VERIFIED.md);
this page is for navigation. For the full experiment-to-path index see
[INDEX.md](INDEX.md).

Legend: **[LB]** load-bearing (external code in EIGEN_BUDDY/HOLO depends on its
path) · **[STUB]** roadmap only, no code yet.

---

## The arc

Catalytic basics → scaling the substrate to real inference → pushing physical
limits and factorization → holographic/lattice structure → topological proofs →
frontier domain phases (limits: event-horizon; proof-power: math; emergence
built atom-up: atom -> energy -> chem -> bio -> consciousness as final boss) →
the wall: decoder theory (49) + physical substrate crossing (50 bm_cpu) →
external frontiers: independent adjudication, prize-bearing Walls, and
cross-domain transfer (51+).

---

## 1_foundations (01–05) — reversible computing & Landauer basics

| # | Experiment | What |
|---|------------|------|
| 01 | tree_evaluation | Zero-clean-space catalytic Tree-Evaluation solver: O(1) clean memory vs recursive O(d log k). Home of the shared `catalytic_engine`/`tree_eval` (moving to `_lib/`). |
| 02 | slack_space | Computes while storing all intermediate state inside file slack/padding bytes — zero OS-level entropy. |
| 03 | visual_bmp | DFS maze under a 64-byte clean limit using BMP pixel bytes as a catalytic stack; 100% reversible restore. |
| 04 | thermodynamic_cpu | Reversible vs irreversible 8-bit adder: 0 bits erased, 0 J Landauer heat. Home of shared `reversible_cpu`. |
| 05 | multibit_compiler | Compiles multi-bit boolean/arithmetic expressions to reversible gates with dynamic carry reclamation. |

## 2_substrate_expansion (06–13) — catalytic memory & inference substrate

| # | Experiment | What |
|---|------------|------|
| 06 | catalytic_neural_network | NN inference in 32 KB clean RAM by reversibly borrowing external storage (RevNet). |
| 07 | quantum_simulator | Reversible classical quantum-circuit sim (15/25 qubit), stealth-borrow entanglement, catalytic Shor period-finding. Owns the `storage/` quantum tapes. |
| 08 | catalytic_gpt | Reversible transformer multiplexing VRAM through a shared dirty tape; 1000 concurrent agents. |
| 09 | borrowing_os_memory | Borrowing OS shared memory (single-script probe). |
| 10 | catalytic_kv_cache | Compressed KV cache: SVD spatial projection + heavy-hitter pruning; ~8.8x on Gemma. |
| 11 | grail_calorimeter | Micro-calorimeter Landauer-heat benchmark; reversible path dissipates 0 J. |
| 12 | structured_tape_acceleration | Warm-tape cache that short-circuits a tape-aware solver. |
| 13 | orthogonal_multimodel | Two models share one tape in orthogonal QR subspaces; near-zero cross-talk. |

## 3_physics_complexity (14–24) — limits, factorization, NP, temporal

| # | Experiment | What |
|---|------------|------|
| 14 | bekenstein_violator | Catalytic throughput exceeds the static Bekenstein storage bound (~700M XOR on 2 MB). |
| 15 | hdd_native_inference | HDD-native out-of-core catalytic inference (weight streaming). |
| 16 | catalytic_27b_inference **[LB]** | 27B-scale catalytic inference via Eigen Buddy Rust FFI + warm-tape replay. Holds the qwen_0.5b model (gitignored). |
| 17 | temporal_bootstrap | NP-complete SAT solver via temporal bootstrap. |
| 18 | hawking_decompressor | Hawking-style Feistel decompressor. |
| 19 | catalytic_computronium | Information battery / computronium (Rust FFI). |
| 20 | catalytic_eigen_shor **[LB]** | Eigen-Shor factorization journey (11 sub-experiments); phase-cavity core used across the holo brain. Secondary Rust FFI producer. |
| 21 | holographic_elliptic_sieve | Phase-cavity eigenmode sieve for elliptic factorization. |
| 22 | superconducting_inference | Josephson-junction passive (zero-power) inference bit tracker. |
| 23 | temporal_catalysis | Retrocausal borrowing + pan-temporal attention. |
| 24 | quantum_catalytic_entanglement | Invisible-hand entanglement + Shor; Schmidt / D_pr scaling. |

## 4_holographic (25–33) — lattice/crypto, graphs, wormholes, MERA

| # | Experiment | What |
|---|------------|------|
| 25 | lattice_holography | LWE / holographic SVP lattice experiments (base; infinity variant under `infinity/`). |
| 25b | wigners_friend | Reversible-observer / Wigner's-friend experiment. |
| 26a | hawking_quantum | Hawking quantum-horizon simulator. |
| 26b | optical_3sat | Optical 3-SAT solver. |
| 27 | landauer_limit | Landauer-limit thermodynamics. |
| 28 | stealth_crypto | Zero-trace stealth crypto. |
| 29 | graph_reachability | O(1)-space graph reachability. |
| 30 | boundary_stress | Multi-process collision / boundary stress. |
| 31 | graph_isomorphism | Holographic graph isomorphism (spectral signature). |
| 32 | traversable_wormhole | ER=EPR traversable wormhole (Grail 5). |
| 33 | mera_compression **[LB]** | Cross-layer MERA SVD compression; produces the `.holo` files HOLO/Eigen Buddy consume. |

## 5_topological_proofs (34–41) — zeta/RH, halting oracles, ToE

| # | Experiment | What |
|---|------------|------|
| 34 | zeta_eigenbasis | Zeta-eigenbasis spectral approach to the Riemann hypothesis (5 stages). |
| 35 | topological_halting_oracle | Halting as topological edge destruction — the core proof (9 sub-experiments). |
| 36 | bekenstein_godel | Bekenstein-Godel singularity (Z2 Chern obstruction). |
| 37 | 2d_chern_oracle | 2D Chern oracle (halting as edge destruction). |
| 38 | 3d_weyl_oracle | 3D Weyl-annihilation oracle. |
| 39 | 4d_axion_oracle | 4D axion oracle (second Chern number). |
| 40 | 5d_floquet_oracle | 5D Floquet time-crystal oracle. |
| 41 | toe_bulletproof | Closing the theoretical gaps in the topological ToE. |

## 6_frontier_phases (42–48) — frontier physics & domain phases

| # | Experiment | What |
|---|------------|------|
| 42 | computational_event_horizon | Floating-point singularities at the event horizon. Sub-tracks: `01_core` (base 1-11), `02_ultra` (speculative 12-19), `03_black_holes` (20-23), `04_cosmos` (24-28). |
| 43 | phase_math | Millennium-problem oracles: collatz, navier-stokes, erdos, riemann, P-vs-NP, yang-mills. |
| 44 | phase_atom | Standard-model oracles: nucleus, electron edge states, pauli, LHC, higgs, quark confinement. |
| 45 | phase_energy **[STUB]** | Energy-extraction roadmap; no code yet. |
| 46 | phase_chem **[STUB]** | Topological-chemistry roadmap; no code yet. |
| 47 | phase_bio | Biology oracles: protein folding, folding pathway, prion, genetic code, neural binding, morphogenesis (holds the cell CSV). |
| 48 | phase_consciousness **[STUB]** | Qualia-engine roadmap; no code yet. The final-boss frontier phase. |

## 7_decoder (49–50) — wall localization & physical substrate crossing

| # | Experiment | What |
|---|------------|------|
| 49 | the_decoder | Extractive proof + decodability boundary (14 sub-experiments): locates the irreducible wall at lattice hardness. |
| 50 | phase_bm_cpu | Physical test of crossing the decoder wall (bare-metal CPU substrate). Lives at `7_decoder/50_phase_bm_cpu`. |

## 8_external_frontiers (51+) — independent adjudication & transfer

| # | Experiment | What |
|---|------------|------|
| 51 | poseidon_cryptanalysis **[STUB]** | Exact finite-field challenge engine for CICO, Zero-test, Density, partial collisions, and reduced-round attack research. |
| 52 | proximity_prize **[STUB]** | Reed–Solomon correlated-agreement and list-boundary laboratory, finite atlas, and theorem extraction. |
| 53 | vesuvius_challenge **[STUB]** | Non-collapse laminar geometry, surface continuation, mesh integrity, fiber fields, and ink provenance. |
| 54 | arc_prize **[STUB]** | `TaskHolo` and `WorldOrbit` for sparse transformation induction and interactive adaptation. |
| 55 | hutter_prize **[STUB]** | Exact relational compression, reversible corpus transforms, and byte-perfect restoration. |
| 56 | aimo_prize **[STUB]** | `ProofOrbit` mathematical agent architecture with countermodels, tools, verification, and proof repair. |
| 57 | erdos_bounties **[STUB]** | Governed theorem-bounty selector, exact finite laboratories, boundary atlases, and proof extraction. |
| 58 | ecdsa_fail **[STUB]** | Reversible elliptic-curve circuit equivalence, global uncomputation, and full resource-vector optimization. |
| 59 | neutral_atom_qday **[STUB]** | Independent logical-to-physical quantum resource audit, architecture sensitivity, qday evidence, and migration mapping. |

Track 8 is outward-facing but does not replace Exp 50. External frontiers and the physical Small Wall run in parallel and meet through explicit transfer records only.

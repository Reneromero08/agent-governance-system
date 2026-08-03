# MANIFEST — File Origins

Every file in `HOLO_WEIGHTS/` is a copy of a lab original. Origin paths are
relative to the repo root (`agent-governance-system`) unless noted. `(catvm)`
means the `codex/audio-frequency-wave-substrate` worktree at
`agent-governance-system-catvm`.

## pipeline/01_distill/ — SVD distillation to .holo

| File | Origin |
|---|---|
| distill_27b_holo.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ |
| distill_qwen.py | THOUGHT/LAB/EIGEN_BUDDY/distill/ |
| distill_catalytic.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ |
| distill_flash.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ |
| distill_gguf_catalytic.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ |
| distill_gguf_mtp.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ |
| distill_0_5b_holo.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ |
| distill_deepseek_flash.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ (archive ref) |
| distill_deepseek_flash_2.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ (archive ref) |
| holographic_cybernetic_engine.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ |
| load_holo_v2.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ |
| test_gguf_fast_distill.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ |

## pipeline/02_wormhole/ — wormhole rotation + residual correction (the fidelity fix)

| File | Origin |
|---|---|
| 1_cross_layer_mera.py ... 25_gguf_patcher.py | THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression/ |
| _residual_correct.py (rank-4, never executed) | THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression/ |
| _verify_fidelity.py, _k_sweep.py, _wormhole_loader.py, etc. | THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression/ |
| REPORT.md (Qwen 27B fidelity results 0.84-0.89) | THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression/ |
| PUSHED_REPORT_AUTOTUNE.md (analytic O(1) alignment) | THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression/ |

## pipeline/03_adapter/ — trained residual adapters + spectral compression

| File | Origin |
|---|---|
| train_adapter.py, flat_llm_adapter.py, gguf_backend.py, complex_gpt2_compress.py, gemma_complex_sweep.py, task3_diagnostics.py | THOUGHT/DEPRECATED/TINY_COMPRESS/extensions/03_flat_llm/ |
| REPORT.md, TRAIN_REPORT.md, REPORT_DIAGNOSTICS.md | THOUGHT/DEPRECATED/TINY_COMPRESS/extensions/03_flat_llm/ |
| spectral_compress.py, spectral_llm.py, eigen_gpt2.py, eigen_attention.py, compressed_inference.py, compress_and_finetune.py, activation_compress.py, run_eigen.py | THOUGHT/DEPRECATED/TINY_COMPRESS/llm-spectral/ |
| README.md (Df law: weights Df 500+, not SVD-compressible) | THOUGHT/DEPRECATED/TINY_COMPRESS/llm-spectral/ |

## pipeline/04_inference/ — engines

| File | Origin |
|---|---|
| attention.py, catalytic.py, catalytic_core.py, catalytic_inference.py, curvature.py, engine.py, gpu_distill.py, hybrid.py, nvme_harness.py, phase.py, phase_projection.py, position.py, __init__.py | THOUGHT/LAB/EIGEN_BUDDY/core/ |
| holo_pipeline/01_distill .. 05_inference/ | THOUGHT/LAB/HOLO/pipeline/ (extract_attn, cavity, wormhole, calibrate: autotune/correction_tape/residual_correct, inference: catalytic_27b/corrected_inference/holographic_engine/native_27b) |

## pipeline/05_validate/ — diagnostics

| File | Origin |
|---|---|
| phase/task1_plv.py, task2_dispersion.py, tasks_345.py, PHASE_REPORT.md | THOUGHT/DEPRECATED/TINY_COMPRESS/llm-spectral/phase/ |
| gemma/calibrate_gemma.py, step3_validate.py, FINAL_REPORT.md, GAMMA_REPORT.md | THOUGHT/DEPRECATED/TINY_COMPRESS/llm-spectral/gemma/ |
| results/REPORT_SPECTRAL_COMPRESSION.md, REPORT_COMPRESSION_BARRIER.md | THOUGHT/DEPRECATED/TINY_COMPRESS/llm-spectral/results/ |
| sweeps/sweep.py | THOUGHT/DEPRECATED/TINY_COMPRESS/llm-spectral/sweeps/ |

## native/phase_frontier/ — CAT_CAS phase machinery (the weapons)

114 C/H files + Makefiles copied from `(catvm)`:
`.../audio_frequency_wave_substrate/cat_cas_phase_frontier/`

Key items: phase-lock (unit(2z+conj(z)^2), drift 0.074 -> 1.6e-16),
streaming/parallel phase VMs, fredkin compiler, algebraic relation closures
(trees, cycles, series/parallel, TT quotients), exact cyclotomic carriers
(Q(zeta17), F103/F137), CATVM services (seccomp-enclosed custody), Kerr/SU(2)
wave carriers, F5 conics/Gauss kernels, F17 group-algebra charts.

## native/f17_frontier/ — the NEW .holo frontier (copied 2026-08-03, live work)

294 files copied from `(catvm)` `.../cat_cas_phase_frontier/`:

- pi-unit exact arithmetic: embedding_balance, exact_coordinate_descent,
  deferred_ledger_stream, horner_stream, lattice_center, height_reduction,
  pi_content_recurrence (log-free exact coordinate reduction in pi units)
- coherent charts: veronese_phase_chart, rank2_secant_coupling,
  rank4_double_secant_coupling, latent_basis_mismatch_grid,
  paired_phase_basis_holographic_matchgate, planar_free_fermion_phase_covariance
- rank laws: two_signature_separation_rank_diagnostic (LIVE, 2026-08-03),
  growing_shared_latent_no_go (2^k separator), grid_linear_separator_no_go,
  nonlinear_canonical_mps_separator_chart, variable_rank_nonseparable_tensor,
  symbolic_iterated_affine_reflection_secant_rank_growth
- chain family: cyclotomic_module, direct_real_hermitian, exact_lift, krylov,
  height_lower_bound, quadratic_extension_resident_carrier, real_subfield_horner,
  streamed_real_autocorrelation, transfer_closure, adaptive_gauge,
  gray_code_bond2, overlapping_cubic_bond3, two_latent_cubic_cycle_rank4
- closures: exchange_symmetric, global_shift_latent_ladder_convolution,
  shared_latent_cubic_hypertree, three_shear_relative_hermitian_trace_feedback,
  runtime_weighted_grid, shared_phase_parity_ledger_ladder,
  interacting_cubic_latent_trace
- per-experiment: *_oracle.py, RESULTS.json, PROVENANCE.json, REVIEW.md, qualify_*.sh

Two-signature verdict (live): two independent cubic Walsh derivative signatures
generate FULL 17x17 coefficient separation rank -> uniformly low separation-rank
factor charts rejected at family ceiling; canonical 578-cell C17^2 carrier
restores/reuses exactly; matched compact classical residue recurrences.

## native/unbounded_compute_v1/ — toroidal path-sum package

43 files from `(catvm)` `.../audio_cat_cas_unbounded_compute_v1/`:

- `CAT_CAS_COMPACT_TOROIDAL_PATH_SUM_REFERENCE_VERIFIED` (bounded mission stop)
- public `.holo` programs: programs/phase_path_{alpha,beta,gamma,delta}_{16..256}.holo
- Gamma vs explicit path-work: n=16 -> 2.19, n=32 -> 71913, n=64 -> 1.5e14,
  n=128 -> 1.4e33, n=256 -> 2.4e71 (compact classical DP still faster;
  no best-classical advantage claimed)
- ARCHITECTURE.md, FINAL_REPORT.md, EXTERNAL_VERIFICATION, INVARIANT_RESULTS,
  PACKAGE_MANIFEST.json (SHA-bound), development probes

## lib/

| File | Origin |
|---|---|
| load_holo_v2.py | THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/ (duplicate of pipeline/01_distill/) |

## docs/

| File | Origin |
|---|---|
| ENGINE_REPORT.md | THOUGHT/LAB/HOLO/docs/ (DeepSeek V4 engine: what worked, what didn't, the blockage list) |
| HOLO_HISTORY.md | THOUGHT/LAB/HOLO/ (all .holo generations, canonical definition) |
| MODELS_ARCHIVE.md | THOUGHT/LAB/HOLO/models/ARCHIVED.md (Seagate pointer) |

## config/

| File | Origin |
|---|---|
| paths.json | New — Seagate + local asset paths, single place to fix drive-path drift |

## NOT copied (by design)

- Model weights and large .holo binaries (Seagate only; see config/paths.json).
- EIGEN_ALIGNMENT/qgt_lib (C quantum-geometric library) — reference only.
- Neo3000 (llama.cpp fork) — separate repo; ggml dequant + baseline server.
- CAT_CAS lab governance/roadmap files — this workspace is self-contained.

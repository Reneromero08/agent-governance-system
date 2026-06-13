# CAT_CAS Storage Manifest

Large data artifacts (>1 MB binary / >100 MB anything) live **in their owning
experiment**, are **gitignored** (see [`../.gitignore`](../.gitignore)), and are
listed here as per-experiment requirements. Paths are relative to the experiment
directory so they stay correct across the track reorganization.

| Owner exp | Path (within exp) | Size | What it is | How to obtain / regenerate | Untracked because |
|-----------|-------------------|------|------------|----------------------------|-------------------|
| 06_catalytic_neural_network | `data/user_video.mp4` | 2 MB | Deterministic "dirty" data file used as the borrowable tape | `python generate_model_and_data.py` (seeded, reproducible) | regenerable |
| 07_quantum_simulator | `data/quantum_tape_25q.bin` | 1 GB | 1 GB catalytic tape for the 25-qubit reversible state sim | `python experiment.py` writes it | regenerable, 1 GB |
| 07_quantum_simulator | `data/quantum_tape_20q.bin`, `data/quantum_tape.bin` | 32 MB / 1 MB | Smaller-scale tapes (20-qubit / baseline) | written by the corresponding sim run | regenerable |
| 16_catalytic_27b_inference | `gemini_update/qwen_0.5b/model.safetensors` | 942 MB | Qwen-0.5B weights. **External contract** — EIGEN_BUDDY and HOLO hardcode this path. **Do not move.** | Hugging Face download (`qwen_0.5b`) | 942 MB, external |
| 33_mera_compression | `_analytic_merged.holo` | 199 MB | MERA cross-layer SVD-compressed weights; consumed by HOLO + Eigen Buddy | produced by the exp-33 MERA pipeline | 199 MB, derived |
| 47_phase_bio | `47_6_morphogenesis_oracle/cell_data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv` | 2.8 GB | HuBMAP CODEX single-cell dataset for the morphogenesis oracle | external download (Dryad); see exp 47_6 REPORT | 2.8 GB, external dataset |
| 42_computational_event_horizon | `04_cosmos/28_holographic_entropy_screen/telemetry/boundary_cloud_raw.csv` | 133 MB | Raw boundary-cloud telemetry from exp 28 | re-run exp_28 | regenerable telemetry |
| 42_computational_event_horizon | `02_ultra/14_boltzmann_brain/rust/mri_*.bin` | 39 MB x4 | Rust simulation state dumps (recursive/emergence/collision/discard) | re-run exp_14 rust build | regenerable |
| 50_phase_bm_cpu | `50_5_10_encoding_wall/_generated/*frozen*.json` | <10 KB | Frozen basin-threshold baseline for the phase 5.10 classifier | `python 50_5_10_encoding_wall/src/analyze_phase5_10.py` (seeded) | **TRACKED** (small JSON, deliberate reproducible baseline; do not gitignore) |

## Rules

- A new >100 MB artifact: place it inside its experiment, add a `.gitignore`
  pattern if not already covered, and add a row here.
- Never commit these; never relocate the exp-16 weights (load-bearing external
  path). See [CONVENTIONS.md](CONVENTIONS.md) §9.

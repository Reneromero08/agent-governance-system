Reproduce the Phase-3 full-mode stress gate (SciFact neighbor falsifier):
  python THOUGHT/LAB/FORMULA/experiments/open_questions/q32/q32_public_benchmarks.py --mode stress --dataset scifact --scoring crossencoder --wrong_checks neighbor --neighbor_k 10 --device cpu --threads 12 --stress_n 12 --stress_min_pass_rate 0.9 --stress_out LAW/CONTRACTS/_runs/q32_public/datatrail/stress_REPRO.json --empirical_receipt_out LAW/CONTRACTS/_runs/q32_public/datatrail/receipt_REPRO.json

Reproduce a Phase-3 multi-seed transfer run (example: SciFact -> SNLI):
  python THOUGHT/LAB/FORMULA/experiments/open_questions/q32/q32_public_benchmarks.py --mode transfer --scoring crossencoder --wrong_checks neighbor --neighbor_k 10 --device cpu --threads 12 --seed 0 --calibration_n 2 --verify_n 2 --calibrate_on scifact --apply_to snli --calibration_out LAW/CONTRACTS/_runs/q32_public/datatrail/cal_REPRO.json --empirical_receipt_out LAW/CONTRACTS/_runs/q32_public/datatrail/receipt_REPRO.json

Notes:
- This repo uses an on-disk HF cache under LAW/CONTRACTS/_runs/q32_public/hf_cache.
- Some dataset loading may warn about trust_remote_code; runs use the cached dataset artifacts.

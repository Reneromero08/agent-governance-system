Phase 5 replication bundle (Q32)
Generated: 20260110_100535

This folder captures environment + git identifiers, plus SHA256 hashes for the key Phase 5 receipts/logs under:
  LAW\\CONTRACTS\\_runs\\q32_public\\datatrail

Primary Phase 5 artifacts (see EVIDENCE_SHA256.txt for hashes):
  - Matrix (4-domain, cached calibration): empirical_receipt_p5_matrix4_full_cached_n2_20260110_073410.json
  - Stress (all datasets): empirical_receipt_p5_stress_all_full_n10_20260110_081613.json + stress_p5_all_full_n10_20260110_081613.json
  - Sweep-k (all datasets): empirical_receipt_p5_sweep_k_all_full_trials6_20260110_083905.json + sweep_k_p5_all_full_trials6_20260110_083905.json
  - Negative controls (bench/stream): inflation, paraphrase(overlap), shuffle(echo) receipts

Re-run configs are recorded inside each receipt under the JSON key: run
(Exact CLI invocations are derivable from those run fields + q32_public_benchmarks.py flags.)

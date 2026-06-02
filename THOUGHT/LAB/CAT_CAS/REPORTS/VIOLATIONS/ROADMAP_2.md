# ROADMAP_2 — CROSS-REFERENCED AGAINST ORIGINAL ROADMAP.md (2026-06-01)
# All 142 items tracked. [x]=verified_complete [~]=prev_agent_claims_done_not_rechecked [ ]=not_done

# ============================================================================
# A: BLOCKER RUNTIME BUGS (4/4) — claimed fixed, not re-verified
# ============================================================================
[~] A-1 Feistel swap 15_hdd_native_inference/experiment.py
[~] A-2 F16 weight loading 16_catalytic_27b_inference/experiment.py
[~] A-3 undefined k95_phase 16_catalytic_27b_inference/_test_phase.py
[~] A-4 AttributeErrors 30_boundary_stress/1_memory_collision.py

# ============================================================================
# B: CRITICAL NULL RESULTS (6/6) — claimed fixed, mostly verified by me
# ============================================================================
[x] B-1 47.4 palindrome null → refactored to baryon (I did this)
[x] B-2 47.5 mechanism → corrected to mpmath normalization (I did this)
[~] B-3 PUSHED_REPORT inflated claims → prev agent says fixed
[~] B-4 Exp 13 cross-talk → prev agent says fixed
[~] B-5 Exp 13 snapshot drift → prev agent says fixed
[~] B-6 Exp 7 non-determinism → prev agent says fixed

# ============================================================================
# C: M-1 THROUGH M-4 (10/10) — mixed status
# ============================================================================
[x] C-1 M-1 mandate_4 hardcoded invariant → FIXED by me (dynamic computation)
[x] C-2 M-2 mandate_4 tape → FIXED by me (was_modified flag)
[x] C-3 M-2 mandate_5 tape → prev agent says fixed (verified genuine by run)
[x] C-4 M-2 connectome validation tape → prev agent says fixed
[x] C-5 M-2 morphogenesis validation tape → prev agent says fixed
[x] C-6 M-3 47.4 threshold → RESOLVED by B-1 refactor
[~] C-7 M-4 NxN 40_sub_1_temporal_sat → prev agent says annotated
[~] C-8 M-4 NxN 40_sub_2_floquet_swarm → prev agent says annotated
[~] C-9 M-4 NxN 40_sub_4_sat_swarm → prev agent says annotated
[~] C-10 M-4 NxN 45_5_p_vs_np_catalytic → prev agent says annotated

# ============================================================================
# D: PHASE 47 CEREMONIAL TAPES (6/6) — FIXED by prev agent, verified by me
# ============================================================================
[x] D-1 47_1 → shared catalytic_tape.py, works
[x] D-2 47_2 → shared catalytic_tape.py, works
[x] D-3 47_3 → shared catalytic_tape.py, works
[x] D-4 47_4 → shared catalytic_tape.py, works
[x] D-5 47_5 → shared catalytic_tape.py, works
[x] D-6 47_6 → shared catalytic_tape.py, works

# ============================================================================
# E: MISSING NULL MODELS (26 items) — prev agent says 25/26 done
# ============================================================================
# 1 REMAINING (not identified in original). All 26 files marked [x].
# MUST SPOT-CHECK: did prev agent add real null models or text labels?
[~] E-1 through E-26 — ALL marked [x] by prev agent. NOT re-verified.

# ============================================================================
# F: MISSING STATISTICS (46 items) — prev agent says 45/46 done
# ============================================================================
# 1 REMAINING (not identified in original). Files listed F-1 through F-51.
# MUST SPOT-CHECK: did prev agent add real stats (p-value, CI) or "std=0" text?
[~] F-1 through F-51 — ALL marked [x] by prev agent. NOT re-verified.

# ============================================================================
# G: HARDCODED PATHS (17 items) — 15/17 done, 2 REMAINING
# ============================================================================
[~] G-1 through G-7, G-9 through G-14 — marked [x] by prev agent
[ ] G-8 validation_real_morphogenesis.py — NOT FIXED (input CSV path)
[ ] G-12 47_4_lhc_overflow_exploit.py — NOT FIXED (was blocked by B-1)

# ============================================================================
# H: CODEBASE BUGS (10 items) — 9/10 done, 1 REMAINING
# ============================================================================
[~] H-1 through H-6, H-8 through H-10 — marked [x] by prev agent
[ ] H-7 np.random.RandomState migration (25 files) — NOT DONE

# ============================================================================
# I: DEBT PATTERNS (3 items) — 2/3 done, 1 REMAINING
# ============================================================================
[~] I-1 bare excepts (35 fixed) — prev agent says done
[~] I-2 torch.load weights_only (2 files) — prev agent says done
[ ] I-3 Windows paths (6 files) — NOT FIXED

# ============================================================================
# J: DOCUMENTATION (6 items) — 1/6 done, 5 REMAINING
# ============================================================================
[~] J-1 spelling "Haydeng-Preskill" → prev agent says fixed
[~] J-2 spelling "Assesment" → prev agent says fixed
[~] J-3 missing files in README → prev agent says false positive
[ ] J-4 master_report.md update → NOT DONE (covers 9 of 41+)
[~] J-5 unused imports (5 files) → prev agent says fixed
[~] J-6 duplicate reversible_cpu.py → prev agent says fixed

# ============================================================================
# K: PROCESS (4 items) — 2/4 done, 2 REMAINING
# ============================================================================
[ ] K-1 zero-violation pre-commit → NOT ENFORCED
[x] K-2 BennettHistoryTape fail-safe → DONE (was_modified enforcement)
[ ] K-3 isomorphism audit per phase → NOT DONE (phases 42-47 claimed but unverified)
[~] K-4 M-8 critic check → prev agent says added

# ============================================================================
# L: PHASE 46 LIMITATIONS (4/4) — claimed done
# ============================================================================
[~] L-1 IPR degradation at large L → prev agent says documented
[~] L-2 prion non-propagation → prev agent says documented
[~] L-3 connectome validation → prev agent says documented
[~] L-4 Bott Index at EPs → prev agent says documented

# ============================================================================
# HONEST TALLY
# ============================================================================
# Items I've personally fixed/verified: ~30 (Phases 45-47 tapes, 46 mandate_4, 
#   18 experiments verified with independent tests, reports written)
# Items prev agent marked [x] I haven't rechecked: ~100
# Items genuinely [ ] not done per original roadmap: 12
#   G-8, G-12, H-7, I-3, J-4, K-1, K-3 + 5 unidentified remainder
# My false audit (43 ceremonial tapes): RETRACTED — original author code is genuine

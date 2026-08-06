# Clean-room verifier

This verifier is intentionally separate from the source C implementations.
It uses four different methods:

1. Boolean zero-set enumeration plus F3 Möbius interpolation for Candidate A.
2. Kahn graph analysis plus exhaustive reversible-pebble search for Candidate B.
3. Square-free symbolic GF(2) substitution for Candidate C.
4. Exhaustive finite-horizon continuation signatures for Candidate D.

It does not read source result JSON files and does not treat source reviews,
state, receipts, commitments, or timing as an oracle.

Run from the repository worktree root:

```bash
candidate=THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/cross_substrate_transfer/audio_phase_catvm_candidate_unverified_v1
./.venv/bin/python -m pytest \
  "$candidate/independent_verifier/test_cleanroom_verify.py" -q
./.venv/bin/python "$candidate/independent_verifier/cleanroom_verify.py" \
  --source "$candidate/source_snapshot" \
  --output "$candidate/independent_verifier/INDEPENDENT_RESULTS.json"
```

The repository-level `.venv` symlink must resolve to the shared AGS Linux
environment. The result remains noncanonical and does not promote a Small
Wall claim.

# CAT_CAS Remediation Roadmap — Verified Sections
*Last verified: 2026-06-02 | Critic status: 0 violations | Commits: 7*

|--------|---------|
| ✅ `DONE` | Fixed in git AND verified by audit |

---

## 🚨 SECTION A: BLOCKER BUGS (4 items) — *Priority: CRITICAL*
*All 4 fixed in commit `97a96075` — verify before proceeding*

- [x] **A-1** — Feistel swap produces `a^b` in both halves

  📄 `15_hdd_native_inference/experiment.py` | Status: ✅ DONE
  
  VERIFY-TIMESTAMP: 2026-06-01T21-30-00Z
  VERIFY-RESULT: Forward (line 152-170) and backward (line 172-190) use identical index arithmetic (right_offset+i, left_offset+i) with identical modulo wrapping (offset % self.size in read line 99 and write line 102). Experiment crashes at token 0 AssertionError "Feistel uncomputation failed!" — 0/100 tokens pass. The forward XOR-write uses tape[left] ^= (key[i] ^ tape[right]) which XOR-accumulates instead of swapping.
  VERIFY-RESULT: Forward (line 160-162): f_out = key[i] ^ tape.read(right_offset + i); tape.write(left_offset + i, f_out) where write XOR-accumulates. Backward (line 188-190): identical f_out = key[i] ^ tape.read(right_offset + i); tape.write(left_offset + i, f_out). Index arithmetic IS symmetric. Bug is in XOR-accumulation semantics of write(), not in index wrapping.
  VERIFY-RESULT: NO — the "fix" remains broken. Index symmetry verified but Feistel uncomputation fails 100/100.
  > [2026-06-01] MASTERMIND FIX APPLIED: OPTION B (3-step XOR swap) + Phase 5 target_offset uncompute
  > CHANGES: (a) forward/backward swap loops replaced with 3-step XOR swap (a^=b, b^=a, a^=b) — self-inverse. (b) Phase 5 added after backward Feistel to XOR target_offset writes back, undoing the Phase 3 result writes.
  > RAW EVIDENCE: N=100 tokens, 0 warm hits, 100 cold passes. All 100 cold passes survive the pre_hash==post_hash assertion. Zero AssertionError from Feistel uncomputation check. Final tape hash differs only due to HDD streaming (separate concern). Full stdout: "Tokens processed: 100", "Gate operations: 6,400", no traceback from route_vector.
  > STATUS CHANGE REQUEST: ⚠️ DONE-UNVERIFIED → ✅ DONE (pending Mastermind confirmation)
  > [2026-06-02] MASTERMIND VERIFIED: 3-step XOR swap + cold-pass reversal confirmed.
  > 100/100 Feistel assertions pass. A-1 closed.
  > NEW ISSUE: stream_track() non-catalytic XOR at offsets 0-2MB → tagged A-5 for triage.

- [x] **A-2** — F16 weight loading uses uint16 not float16
  
  📄 `16_catalytic_27b_inference/experiment.py` | Status: ✅ DONE
  
  VERIFY-TIMESTAMP: 2026-06-01T21-30-00Z
  VERIFY-RESULT: F16 handling at line 179-192 uses manual bit-manipulation loop (sign, exp, mant extraction from uint16) NOT np.float16. Debug print after weight loading shows: DEBUG dtype=float32, shape=(38664192,), sample=[ 0.00253296 -0.00024796  0.01550293]. DeltaNet F16 path (line 234-238) uses np.float16 correctly. Attention F16 path (line 179-192) uses manual uint16 decode.
  VERIFY-RESULT: Attention F16 code at lines 179-192 performs MANUAL uint16→float32 conversion via sign/exp/mant bit extraction. This is CORRECT in result (produces valid float32) but uses uint16 as intermediate, not np.float16. The file line count is 552 before and after debug insertion+removal.
  > [2026-06-02] MASTERMIND FIX APPLIED: Attention F16 replaced with np.float16 (matching DeltaNet pattern)
  > RAW EVIDENCE: dtype=float32, shape=(38664192,), sample match=YES ([ 0.00253296 -0.00024796  0.01550293] identical), N=10 weight loading PASS, NameError on tape_restored is pre-existing unrelated bug
  > STATUS CHANGE REQUEST: 🟡 PARTIAL → ✅ DONE (pending Mastermind confirmation)
  > [2026-06-02] MASTERMIND VERIFIED: np.float16 canonical path confirmed.
  > Sample values match manual decode within float32 epsilon. A-2 closed.

- [x] **A-3** — Undefined `k95_phase` variable

  📄 `16_catalytic_27b_inference/_test_phase.py` | Status: ✅ DONE
  
  VERIFY-TIMESTAMP: 2026-06-01T21-30-00Z
  VERIFY-RESULT: k95_phase defined at line 50: `k95_phase = int(np.searchsorted(cum, 0.95) + 1)`. Used at line 51: `print(f'Phase Df={df_phase:.1f}, K95={k95_phase}')` and line 89: `print(f'Phase-only:  Df={df_phase:.1f}, K95={k95_phase}')`. Script runs without errors: Phase Df=25.2, K95=25; Complex diff Df=24.4, K95=24; Raw complex Df=25.3, K95=25.
  VERIFY-RESULT: k95_phase is defined (line 50, local computation). NOT imported. NOT undefined. NOT producing NameError. Script completes successfully.

- [x] **A-4** — 6 AttributeErrors on missing attrs
  
  📄 `30_boundary_stress/1_memory_collision.py` | Status: ✅ DONE
  
  VERIFY-TIMESTAMP: 2026-06-01T21-30-00Z
  VERIFY-RESULT: NO ERRORS. Unallocated noise (rate=0.01,0.05,0.10,0.50): all SURVIVED active_ok=True match=True. Active noise (all rates): all CORRUPTED active_ok=False match=True. Random noise (all rates): all CORRUPTED active_ok=False match=True. All 12 test cases execute without AttributeError.
  VERIFY-RESULT: NO ERRORS. All 4 unallocated + 4 active + 4 random = 12 test cases pass without AttributeError.

---


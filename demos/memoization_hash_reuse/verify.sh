#!/usr/bin/env bash
# Verification script for memoization + hash-first dereference demo
# Verifies invariants from committed artifacts. Does NOT assume specific numeric values.

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")/../../CONTRACTS/_runs/_demos/memoization_hash_reuse" && pwd)"
cd "$DEMO_DIR"

py() { python -c "$1"; }

echo "=== Memoization Demo Verification ==="
echo ""

# Helpers
read_json_int() { py "import json; print(int(json.load(open('$1'))['$2']))"; }
read_json_str() { py "import json; print(str(json.load(open('$1'))['$2']))"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
jsonl_first_keys() { py "import json; print(sorted(json.loads(open('$1').readline()).keys()))"; }

# Check 1: Dereference count reduction (strict)
echo "✓ Check 1: Dereference count reduction"
baseline_count="$(read_json_int "baseline/DEREF_STATS.json" "deref_count")"
reuse_count="$(read_json_int "reuse/DEREF_STATS.json" "deref_count")"
echo "  Baseline: $baseline_count operations"
echo "  Reuse:    $reuse_count operations"
if [ "$reuse_count" -ge "$baseline_count" ]; then
  echo "  FAIL: Expected reuse_deref_count < baseline_deref_count"
  exit 1
fi
echo "  PASS: reuse performs fewer dereferences"
echo ""

# Check 2: Bytes read reduction (strict)
echo "✓ Check 2: Bytes read reduction"
baseline_bytes="$(read_json_int "baseline/DEREF_STATS.json" "bytes_read_total")"
reuse_bytes="$(read_json_int "reuse/DEREF_STATS.json" "bytes_read_total")"
echo "  Baseline: $baseline_bytes bytes"
echo "  Reuse:    $reuse_bytes bytes"
if [ "$reuse_bytes" -ge "$baseline_bytes" ]; then
  echo "  FAIL: Expected reuse_bytes_read_total < baseline_bytes_read_total"
  exit 1
fi
echo "  PASS: reuse reads fewer bytes"
echo ""

# Check 3: CAS hash identity (must match)
echo "✓ Check 3: CAS hash identity"
baseline_hash="$(read_json_str "baseline/DEREF_STATS.json" "hash")"
reuse_hash="$(read_json_str "reuse/DEREF_STATS.json" "hash")"
echo "  Baseline: $baseline_hash"
echo "  Reuse:    $reuse_hash"
if [ "$baseline_hash" != "$reuse_hash" ]; then
  echo "  FAIL: Expected baseline/reuse target hash to match"
  exit 1
fi
echo "  PASS: Same CAS object accessed"
echo ""

# Check 4: PROOF byte-identity (must match)
echo "✓ Check 4: PROOF byte-identity"
baseline_proof_hash="$(sha256_file "baseline/PROOF.json")"
reuse_proof_hash="$(sha256_file "reuse/PROOF.json")"
echo "  Baseline PROOF: $baseline_proof_hash"
echo "  Reuse PROOF:    $reuse_proof_hash"
if [ "$baseline_proof_hash" != "$reuse_proof_hash" ]; then
  echo "  FAIL: PROOF files differ"
  exit 1
fi
echo "  PASS: Byte-identical PROOF files"
echo ""

# Check 5: Memoization marker presence (baseline must NOT, reuse MUST)
echo "✓ Check 5: Memoization marker presence"
if grep -q 'memoization:hit' baseline/LEDGER.jsonl 2>/dev/null; then
  echo "  FAIL: Baseline should have no memoization marker"
  exit 1
fi
if ! grep -q 'memoization:hit' reuse/LEDGER.jsonl; then
  echo "  FAIL: Reuse should contain memoization marker"
  exit 1
fi
echo "  Baseline: no marker (expected)"
echo "  Reuse:    marker present (expected)"
echo "  PASS: Memoization marker present only in reuse"
echo ""

# Check 6: LEDGER required fields present on first JSONL record (schema-shape backstop)
echo "✓ Check 6: LEDGER required fields present"
required_py="import json; req=set(['JOBSPEC','RUN_INFO','PRE_MANIFEST','POST_MANIFEST','RESTORE_DIFF','OUTPUTS','STATUS']); keys=set(json.loads(open('%s').readline()).keys()); print(req.issubset(keys))"
baseline_ok="$(py "$(printf "$required_py" "baseline/LEDGER.jsonl")")"
reuse_ok="$(py "$(printf "$required_py" "reuse/LEDGER.jsonl")")"
if [ "$baseline_ok" != "True" ]; then
  echo "  FAIL: Baseline missing required keys"
  echo "  Keys: $(jsonl_first_keys baseline/LEDGER.jsonl)"
  exit 1
fi
if [ "$reuse_ok" != "True" ]; then
  echo "  FAIL: Reuse missing required keys"
  echo "  Keys: $(jsonl_first_keys reuse/LEDGER.jsonl)"
  exit 1
fi
echo "  PASS: Both LEDGERs contain required fields"
echo ""

echo "=== ALL CHECKS PASSED ==="
echo ""
echo "Summary:"
echo "  • reuse performs fewer dereferences and reads fewer bytes"
echo "  • PROOF files are byte-identical (correctness preserved)"
echo "  • memoization:hit is present only in reuse"
echo "  • LEDGER records contain required fields"
echo ""
echo "All claims are artifact-verifiable."

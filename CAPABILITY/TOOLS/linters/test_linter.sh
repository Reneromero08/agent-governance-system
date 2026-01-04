#!/usr/bin/env bash
# Quick test: Create a minimal test prompt to verify linter behavior

set -euo pipefail

TEST_DIR="/tmp/ags_lint_test_$$"
mkdir -p "$TEST_DIR/PHASE_01"

# Create minimal manifest
cat > "$TEST_DIR/PROMPT_PACK_MANIFEST.json" <<'EOF'
{
  "tasks": [
    {
      "task_id": "1.1",
      "prompt_path": "PHASE_01/1.1_test-prompt.md",
      "receipt_path": "receipts/1.1.json",
      "report_path": "reports/1.1.md",
      "depends_on": []
    }
  ]
}
EOF

# Create minimal INDEX
cat > "$TEST_DIR/INDEX.md" <<'EOF'
# Test Index
- Phase 1: **1.1** test-prompt -> `PHASE_01/1.1_test-prompt.md`
EOF

# Create canon files with known hashes
cat > "$TEST_DIR/1_PROMPT_POLICY_CANON.md" <<'EOF'
---
title: Test Policy
---
Test policy content.
EOF

cat > "$TEST_DIR/2_PROMPT_GENERATOR_GUIDE_FINAL.md" <<'EOF'
---
title: Test Guide
---
Test guide content.
EOF

# Compute hashes
if command -v sha256sum &>/dev/null; then
  POLICY_HASH=$(sha256sum "$TEST_DIR/1_PROMPT_POLICY_CANON.md" | awk '{print $1}')
  GUIDE_HASH=$(sha256sum "$TEST_DIR/2_PROMPT_GENERATOR_GUIDE_FINAL.md" | awk '{print $1}')
else
  POLICY_HASH=$(shasum -a 256 "$TEST_DIR/1_PROMPT_POLICY_CANON.md" | awk '{print $1}')
  GUIDE_HASH=$(shasum -a 256 "$TEST_DIR/2_PROMPT_GENERATOR_GUIDE_FINAL.md" | awk '{print $1}')
fi

# Create valid test prompt
cat > "$TEST_DIR/PHASE_01/1.1_test-prompt.md" <<EOF
---
phase: 1
task_id: "1.1"
slug: "test-prompt"
policy_canon_sha256: "$POLICY_HASH"
guide_canon_sha256: "$GUIDE_HASH"
depends_on: []
primary_model: "test-model"
fallback_chain: ["fallback-1", "fallback-2"]
receipt_path: "receipts/1.1.json"
report_path: "reports/1.1.md"
max_report_lines: 100
---

## REQUIRED FACTS
- Fact 1: FILL_ME__EXAMPLE
- Fact 2: Known value

## GOAL
Test prompt content.
EOF

echo "=== Test 1: Valid Prompt (should PASS) ==="
bash "$(dirname "$0")/lint_prompt_pack.sh" "$TEST_DIR"
echo "Exit code: $?"
echo ""

# Test 2: Add forbidden term
cat > "$TEST_DIR/PHASE_01/1.1_test-prompt.md" <<EOF
---
phase: 1
task_id: "1.1"
slug: "test-prompt"
policy_canon_sha256: "$POLICY_HASH"
guide_canon_sha256: "$GUIDE_HASH"
depends_on: []
primary_model: "test-model"
fallback_chain: ["fallback-1", "fallback-2"]
receipt_path: "receipts/1.1.json"
report_path: "reports/1.1.md"
max_report_lines: 100
---

## GOAL
We assume this will work.
EOF

echo "=== Test 2: Forbidden Term (should FAIL with exit 1) ==="
bash "$(dirname "$0")/lint_prompt_pack.sh" "$TEST_DIR" || echo "Exit code: $?"
echo ""

# Test 3: Empty bullet (warning)
cat > "$TEST_DIR/PHASE_01/1.1_test-prompt.md" <<EOF
---
phase: 1
task_id: "1.1"
slug: "test-prompt"
policy_canon_sha256: "$POLICY_HASH"
guide_canon_sha256: "$GUIDE_HASH"
depends_on: []
primary_model: "test-model"
fallback_chain: ["fallback-1", "fallback-2"]
receipt_path: "receipts/1.1.json"
report_path: "reports/1.1.md"
max_report_lines: 100
---

## GOAL
Test content.

## PLAN
- Step 1
- 
- Step 3
EOF

echo "=== Test 3: Empty Bullet (should WARN with exit 2) ==="
bash "$(dirname "$0")/lint_prompt_pack.sh" "$TEST_DIR" || echo "Exit code: $?"
echo ""

# Cleanup
rm -rf "$TEST_DIR"
echo "✓ Tests complete"

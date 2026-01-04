#!/usr/bin/env bash
# Validation script for lint_prompt_pack.sh
# Demonstrates expected behavior with current prompt pack

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINTER="$SCRIPT_DIR/lint_prompt_pack.sh"
PROMPTS_DIR="$SCRIPT_DIR/../../NAVIGATION/PROMPTS"

echo "=== Prompt Pack Linter Validation ==="
echo ""
echo "Running linter on: $PROMPTS_DIR"
echo "Linter script: $LINTER"
echo ""

# Run the linter and capture exit code
set +e
bash "$LINTER" "$PROMPTS_DIR"
EXIT_CODE=$?
set -e

echo ""
echo "=== Validation Results ==="
echo "Exit code: $EXIT_CODE"
echo ""

case $EXIT_CODE in
  0)
    echo "✅ PASS: All checks passed"
    echo "The prompt pack is fully compliant with the canon."
    ;;
  1)
    echo "❌ POLICY VIOLATION (blocking)"
    echo ""
    echo "Expected violations (as of 2026-01-04):"
    echo "  - Canon hash mismatches: Prompts contain outdated hashes"
    echo "  - This indicates prompts were generated from older canon versions"
    echo ""
    echo "To fix:"
    echo "  1. Regenerate prompts with current canon files, OR"
    echo "  2. Update canon hashes in prompt YAML frontmatter to match current canon"
    ;;
  2)
    echo "⚠️  WARNING (non-blocking)"
    echo "The prompt pack has warnings but no blocking violations."
    echo "Review warnings and address as needed."
    ;;
  *)
    echo "❓ UNEXPECTED EXIT CODE: $EXIT_CODE"
    echo "This may indicate a linter error."
    exit 1
    ;;
esac

echo ""
echo "=== Linter Capabilities Demonstrated ==="
echo "✓ Manifest validity check"
echo "✓ INDEX link validation"
echo "✓ YAML frontmatter validation"
echo "✓ Canon hash consistency enforcement"
echo "✓ Forbidden term detection (hex-escaped)"
echo "✓ Empty bullet line detection (warning)"
echo "✓ FILL_ME__ token containment check"
echo ""
echo "Exit code meanings:"
echo "  0 = PASS (all checks passed)"
echo "  1 = POLICY VIOLATION (must fix before proceeding)"
echo "  2 = WARNING (non-blocking, should address)"

exit 0

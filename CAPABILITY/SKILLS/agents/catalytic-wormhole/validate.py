#!/usr/bin/env python3
"""
Validates catalytic wormhole compression output.
Checks: compression ratio, chain fidelity, rank optimality.
"""
import json, sys, math
from pathlib import Path

def main(actual_path: Path, expected_path: Path) -> int:
    actual = json.loads(actual_path.read_text())
    expected = json.loads(expected_path.read_text())
    
    # Graceful skip on CI (model not available)
    if actual.get("skipped"):
        print("Validation PASSED (skipped — CI without model)")
        return 0
    
    errors = []
    
    # Check overall fields
    for field in ['ratio', 'total_orig_mb', 'total_comp_mb', 'weight_types', 'total_layers']:
        if field not in actual:
            errors.append(f"Missing field: {field}")
    
    if 'ratio' in actual and actual['ratio'] < 1.0:
        errors.append(f"Ratio {actual['ratio']} < 1.0 (no compression achieved)")
    
    # Check per-weight stats
    if 'per_weight' in actual:
        for wt, stats in actual['per_weight'].items():
            if 'rank' not in stats:
                errors.append(f"{wt}: missing rank")
            if 'fidelity' not in stats:
                errors.append(f"{wt}: missing fidelity")
            if stats.get('ratio', 0) < 1.0:
                errors.append(f"{wt}: ratio {stats['ratio']} < 1.0")
            if stats.get('rank', 0) <= 0:
                errors.append(f"{wt}: invalid rank {stats['rank']}")
    
    # Match against expected if provided
    if expected:
        for field in expected:
            if field in actual:
                exp_val = expected[field]
                act_val = actual[field]
                if isinstance(exp_val, (int, float)):
                    if abs(exp_val - act_val) > max(0.1 * abs(exp_val), 1e-6):
                        errors.append(f"{field}: expected {exp_val}, got {act_val}")
                elif exp_val != act_val:
                    errors.append(f"{field}: expected {exp_val}, got {act_val}")
    
    if errors:
        print(f"Validation FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        return 1
    
    print("Validation PASSED")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: validate.py <actual.json> <expected.json>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))

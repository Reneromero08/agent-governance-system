#!/usr/bin/env python3
"""
Canon validation script for CI checks.

Validates:
1. No duplicate rule numbers within canon files
2. Canon files approaching or exceeding line limits (250 warning, 300 hard limit per INV-009)
3. Authority gradient consistency with canon.json

Exit codes:
- 0: All checks passed
- 1: Validation errors found
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configure UTF-8 output for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def check_duplicate_rule_numbers(canon_path: Path) -> Tuple[bool, List[str]]:
    """
    Check for duplicate rule numbers within the same section of canon files.

    This validates that within a single numbered list section (between headers),
    there are no duplicate numbers. Multiple sections can restart numbering from 1.

    Returns: (passed, error_messages)
    """
    print("Checking for duplicate rule numbers within sections...")
    errors = []
    passed = True

    # Pattern for markdown headers (## or ###)
    header_pattern = re.compile(r'^#{2,3}\s+', re.MULTILINE)

    # Pattern for top-level numbered items (at start of line)
    numbered_item_pattern = re.compile(r'^(\d+)\.\s+\*\*', re.MULTILINE)

    md_files = list(canon_path.glob('*.md')) + list(canon_path.glob('CATALYTIC/*.md'))

    for md_file in sorted(md_files):
        try:
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Split content into sections by headers
            current_section_numbers = []
            current_section_start = 0

            for i, line in enumerate(lines):
                # Check if this line is a header
                if header_pattern.match(line):
                    # Check previous section for duplicates
                    if current_section_numbers:
                        seen: Dict[str, List[int]] = {}
                        for num, line_no in current_section_numbers:
                            if num not in seen:
                                seen[num] = []
                            seen[num].append(line_no)

                        # Find duplicates
                        for num, line_nos in seen.items():
                            if len(line_nos) > 1:
                                passed = False
                                rel_path = md_file.relative_to(canon_path)
                                errors.append(
                                    f"  {rel_path}: Rule number '{num}' appears {len(line_nos)} times "
                                    f"in section starting at line {current_section_start + 1}"
                                )

                    # Start new section
                    current_section_numbers = []
                    current_section_start = i

                # Check if this line has a numbered item
                match = numbered_item_pattern.match(line)
                if match:
                    current_section_numbers.append((match.group(1), i + 1))

            # Check final section
            if current_section_numbers:
                seen_final: Dict[str, List[int]] = {}
                for num, line_no in current_section_numbers:
                    if num not in seen_final:
                        seen_final[num] = []
                    seen_final[num].append(line_no)

                for num, line_nos in seen_final.items():
                    if len(line_nos) > 1:
                        passed = False
                        rel_path = md_file.relative_to(canon_path)
                        errors.append(
                            f"  {rel_path}: Rule number '{num}' appears {len(line_nos)} times "
                            f"in section starting at line {current_section_start + 1}"
                        )

        except Exception as e:
            passed = False
            errors.append(f"  Error reading {md_file.name}: {e}")

    if passed:
        print(f"  ✓ No duplicate rule numbers within sections ({len(md_files)} files checked)")
    else:
        print(f"  ✗ Duplicate rule numbers found:")
        for error in errors:
            print(error)

    return passed, errors


def check_line_counts(canon_path: Path) -> Tuple[bool, List[str]]:
    """
    Check canon files for line count compliance per INV-009:
    - Warn at 250 lines (approaching limit)
    - Error at 300 lines (hard limit)

    Returns: (passed, messages)
    """
    print("Checking canon file line counts (INV-009 compliance)...")

    messages = []
    has_errors = False
    has_warnings = False

    md_files = list(canon_path.glob('*.md')) + list(canon_path.glob('CATALYTIC/*.md'))

    for md_file in sorted(md_files):
        try:
            lines = md_file.read_text(encoding='utf-8').splitlines()
            line_count = len(lines)
            rel_path = md_file.relative_to(canon_path)

            if line_count >= 300:
                has_errors = True
                messages.append(f"  ✗ {rel_path}: {line_count} lines (EXCEEDS 300 line limit)")
            elif line_count >= 250:
                has_warnings = True
                messages.append(f"  ⚠ {rel_path}: {line_count} lines (approaching 300 line limit)")

        except Exception as e:
            has_errors = True
            messages.append(f"  ✗ Error reading {md_file.name}: {e}")

    if not has_errors and not has_warnings:
        print(f"  ✓ All {len(md_files)} canon files within line limits")
    else:
        if has_warnings and not has_errors:
            print("  ⚠ Some files approaching line limits:")
        elif has_errors:
            print("  ✗ Line count violations detected:")

        for msg in messages:
            print(msg)

    # Only fail on errors (>= 300), not warnings (>= 250)
    return not has_errors, messages


def check_authority_gradient(canon_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate authority gradient consistency with canon.json.

    Checks:
    1. canon.json exists and is valid JSON
    2. All files in authority_order exist
    3. Authority order matches CONTRACT.md specification

    Returns: (passed, messages)
    """
    print("Checking authority gradient consistency...")

    messages = []
    canon_json = canon_path / 'canon.json'

    if not canon_json.exists():
        print("  ⓘ canon.json not found, skipping authority gradient check")
        return True, ["  ⓘ canon.json not found (optional check)"]

    try:
        data = json.loads(canon_json.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        messages.append(f"  ✗ canon.json is not valid JSON: {e}")
        print("  ✗ canon.json parsing failed")
        return False, messages

    # Check authority_order exists and files are present
    if 'authority_order' not in data:
        messages.append("  ✗ canon.json missing 'authority_order' field")
        print("  ✗ Missing authority_order field")
        return False, messages

    authority_order = data['authority_order']
    expected_order = [
        "AGREEMENT.md",
        "CONTRACT.md",
        "INVARIANTS.md",
        "VERSIONING.md"
    ]

    # Check all authority files exist
    missing_files = []
    for filename in authority_order:
        if not (canon_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        messages.append(f"  ✗ Authority files missing: {missing_files}")
        print(f"  ✗ Missing authority files: {missing_files}")
        return False, messages

    # Validate order matches expected
    if authority_order != expected_order:
        messages.append(f"  ✗ Authority order mismatch")
        messages.append(f"    Expected: {expected_order}")
        messages.append(f"    Got:      {authority_order}")
        print("  ✗ Authority order does not match specification")
        return False, messages

    print("  ✓ Authority gradient consistent with canon.json")
    return True, messages


def main():
    """Run all canon validation checks."""
    # Determine canon path from input.json or use default
    script_dir = Path(__file__).parent
    input_file = script_dir / 'input.json'

    if input_file.exists():
        try:
            config = json.loads(input_file.read_text(encoding='utf-8'))
            canon_path = Path(config.get('canon_path', 'LAW/CANON'))
        except Exception as e:
            print(f"Error reading input.json: {e}")
            canon_path = Path('LAW/CANON')
    else:
        canon_path = Path('LAW/CANON')

    # Make canon_path absolute if relative
    if not canon_path.is_absolute():
        # Assume it's relative to the repo root
        # Script is in LAW/CONTRACTS/fixtures/governance/canon-validators
        # So we need to go up 5 levels: canon-validators -> governance -> fixtures -> CONTRACTS -> LAW -> repo_root
        repo_root = script_dir.parent.parent.parent.parent.parent
        canon_path = (repo_root / canon_path).resolve()
    else:
        canon_path = canon_path.resolve()

    if not canon_path.exists():
        print(f"Error: Canon path does not exist: {canon_path}")
        sys.exit(1)

    print(f"Validating canon files at: {canon_path}")
    print("=" * 70)

    # Run all checks
    checks = {
        'duplicate_numbers': check_duplicate_rule_numbers(canon_path),
        'line_count': check_line_counts(canon_path),
        'authority_gradient': check_authority_gradient(canon_path)
    }

    print("=" * 70)

    # Determine overall result
    all_passed = all(result[0] for result in checks.values())

    if all_passed:
        print("\n✓ All canon validation checks passed")

        # Write success output
        output = {
            "status": "passed",
            "checks": list(checks.keys())
        }

        output_file = script_dir / 'output.json'
        output_file.write_text(json.dumps(output, indent=2), encoding='utf-8')

        sys.exit(0)
    else:
        print("\n✗ Canon validation failed")

        # Write failure output with details
        output = {
            "status": "failed",
            "checks": list(checks.keys()),
            "failures": {
                check_name: messages
                for check_name, (passed, messages) in checks.items()
                if not passed
            }
        }

        output_file = script_dir / 'output.json'
        output_file.write_text(json.dumps(output, indent=2), encoding='utf-8')

        sys.exit(1)


if __name__ == '__main__':
    main()

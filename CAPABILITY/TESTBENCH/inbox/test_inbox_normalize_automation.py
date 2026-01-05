#!/usr/bin/env python3
"""
Governance Safety Check: INBOX Normalization Automation

Verifies that the weekly INBOX normalization automation is properly configured.
This test fails if the automation is removed or disabled.

Run: python CAPABILITY/TESTBENCH/inbox/test_inbox_normalize_automation.py
"""
import os
import sys
import json
import re
from pathlib import Path

# Paths
INBOX_ROOT = Path("INBOX")
WEEKLY_SCRIPT = Path("CAPABILITY/TOOLS/governance/weekly_normalize.py")
NORMALIZE_SCRIPT = Path("CAPABILITY/TOOLS/governance/inbox_normalize.py")
GOVERNANCE_FILE = Path("LAW/CANON/INBOX_POLICY.md")

def check_weekly_automation_exists() -> tuple[bool, str]:
    """Verify weekly automation script exists."""
    if not WEEKLY_SCRIPT.exists():
        return False, f"Weekly automation script not found: {WEEKLY_SCRIPT}"
    if not os.access(WEEKLY_SCRIPT, os.R_OK):
        return False, f"Cannot read weekly automation script: {WEEKLY_SCRIPT}"
    return True, "Weekly automation script exists"

def check_normalize_script_exists() -> tuple[bool, str]:
    """Verify core normalization script exists."""
    if not NORMALIZE_SCRIPT.exists():
        return False, f"Core normalization script not found: {NORMALIZE_SCRIPT}"
    return True, "Core normalization script exists"

def check_weekly_script_references_normalizer() -> tuple[bool, str]:
    """Verify weekly script references the core normalization runner."""
    try:
        content = WEEKLY_SCRIPT.read_text(encoding='utf-8')
        # Check for imports or references to normalize logic
        has_schema = "SCHEMA" in content
        has_timestamp_policy = "TIMESTAMP_POLICY" in content
        has_dry_run = "run_dry_run" in content or "DRY RUN" in content
        has_execution = "run_execution" in content or "EXECUTION" in content
        
        if has_schema and has_timestamp_policy and has_dry_run and has_execution:
            return True, "Weekly script references normalization runner"
        else:
            missing = []
            if not has_schema:
                missing.append("SCHEMA")
            if not has_timestamp_policy:
                missing.append("TIMESTAMP_POLICY")
            if not has_dry_run:
                missing.append("dry_run")
            if not has_execution:
                missing.append("execution")
            return False, f"Weekly script missing normalization components: {', '.join(missing)}"
    except Exception as e:
        return False, f"Error reading weekly script: {e}"

def check_governance_has_normalization_rules() -> tuple[bool, str]:
    """Verify governance documentation includes INBOX normalization rules."""
    try:
        content = GOVERNANCE_FILE.read_text(encoding='utf-8')
        required_patterns = [
            ("YYYY-MM/Week-XX", "Folder schema"),
            ("INBOX normalization", "Normalization rule"),
            ("weekly_normalize.py", "Automation script reference"),
            ("FAIL CLOSED", "Failure mode"),
            ("receipt", "Receipt requirement"),
        ]
        missing = []
        for pattern, desc in required_patterns:
            if pattern not in content:
                missing.append(f"{desc} ({pattern})")
        
        if missing:
            return False, f"Governance missing: {', '.join(missing)}"
        return True, "Governance includes all required normalization rules"
    except Exception as e:
        return False, f"Error reading governance file: {e}"

def check_weekly_script_has_version() -> tuple[bool, str]:
    """Verify weekly script has proper versioning."""
    try:
        content = WEEKLY_SCRIPT.read_text(encoding='utf-8')
        version_match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
        version_hash_match = re.search(r'VERSION_HASH\s*=\s*hashlib\.sha256.*\.hexdigest\(\)\[:16\]', content)
        
        if version_match and version_hash_match:
            return True, f"Weekly script has version: {version_match.group(1)}"
        return False, "Weekly script missing VERSION or VERSION_HASH"
    except Exception as e:
        return False, f"Error checking version: {e}"

def check_receipts_dir_exists() -> tuple[bool, str]:
    """Verify receipts directory exists or can be created."""
    receipts_dir = Path("LAW/CONTRACTS/_runs")
    if receipts_dir.exists():
        if os.access(receipts_dir, os.W_OK):
            return True, "Receipts directory exists and is writable"
        return False, "Receipts directory exists but is not writable"
    return True, "Receipts directory will be created on first run"

def run_safety_checks() -> dict:
    """Run all safety checks and return results."""
    checks = [
        ("weekly_automation_exists", check_weekly_automation_exists()),
        ("normalize_script_exists", check_normalize_script_exists()),
        ("weekly_references_normalizer", check_weekly_script_references_normalizer()),
        ("governance_has_rules", check_governance_has_normalization_rules()),
        ("weekly_has_version", check_weekly_script_has_version()),
        ("receipts_dir_accessible", check_receipts_dir_exists()),
    ]
    
    results = {
        "operation": "INBOX_NORMALIZE_AUTOMATION_SAFETY_CHECK",
        "timestamp": f"{__import__('datetime').datetime.utcnow().isoformat()}Z",
        "all_passed": all(r[1][0] for r in checks),
        "checks": {name: {"passed": passed, "message": msg} for name, (passed, msg) in checks}
    }
    
    return results

def main():
    """Main entry point."""
    print("=" * 60)
    print("INBOX Normalization Automation Safety Check")
    print("=" * 60)
    print()
    
    results = run_safety_checks()
    
    print("[@] Safety Check Results:")
    print()
    for check_name, result in results["checks"].items():
        status = "[PASS]" if result["passed"] else "[FAIL]"
        print(f"   {status} {check_name}")
        print(f"        {result['message']}")
        print()
    
    if results["all_passed"]:
        print("[+] All safety checks passed")
        print("    INBOX normalization automation is properly configured")
        return 0
    else:
        print("[!] Safety checks failed")
        print("    INBOX normalization automation may be misconfigured or missing")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
AGS Emergency CLI

Provides concrete CLI modes for crisis handling as defined in CANON/CRISIS.md.

Usage:
    python TOOLS/emergency.py --status          # Check current status
    python TOOLS/emergency.py --mode=validate   # Full validation
    python TOOLS/emergency.py --mode=rollback   # Rollback last commit
    python TOOLS/emergency.py --mode=quarantine # Enter quarantine mode
    python TOOLS/emergency.py --mode=restore    # Exit quarantine
    python TOOLS/emergency.py --mode=constitutional-reset --tag=v1.0.0
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Correctly resolving PROJECT_ROOT from CAPABILITY/TOOLS/utilities/emergency.py
# parents[0]=utilities, parents[1]=TOOLS, parents[2]=CAPABILITY, parents[3]=repo_root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUARANTINE_FILE = PROJECT_ROOT / ".quarantine"
LOGS_DIR = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "emergency_logs"
EMERGENCY_LOG = LOGS_DIR / "emergency.log"
CRITIC_PATH = PROJECT_ROOT / "CAPABILITY" / "TOOLS" / "governance" / "critic.py"
RUNNER_PATH = PROJECT_ROOT / "LAW" / "CONTRACTS" / "runner.py"

def log_event(event_type: str, message: str):
    """Log an emergency event."""
    # Ensure parent directories exist
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().isoformat()
    with open(EMERGENCY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {event_type}: {message}\n")
    print(f"[{event_type}] {message}")


def check_status():
    """Check current system status."""
    print("=" * 60)
    print("AGS EMERGENCY STATUS")
    print("=" * 60)
    
    # Check quarantine
    if QUARANTINE_FILE.exists():
        quarantine = json.loads(QUARANTINE_FILE.read_text())
        print(f"\n⚠️  QUARANTINE ACTIVE since {quarantine['entered']}")
        print(f"   Reason: {quarantine['reason']}")
        print(f"   Git hash: {quarantine['git_hash']}")
        return 3
    
    print("\n✓ Quarantine: Not active")
    
    # Run critic
    print("\n--- Running critic ---")
    if not CRITIC_PATH.exists():
         print(f"✗ Critic file not found at: {CRITIC_PATH}")
         return 1

    critic_result = subprocess.run(
        [sys.executable, str(CRITIC_PATH)],
        capture_output=True,
        text=True
    )
    if critic_result.returncode == 0:
        print("✓ Critic: PASSED")
    else:
        print("✗ Critic: FAILED")
        print(critic_result.stderr or critic_result.stdout)
        return 1
    
    # Run contract runner
    print("\n--- Running fixtures ---")
    if not RUNNER_PATH.exists():
         print(f"✗ Runner file not found at: {RUNNER_PATH}")
         return 1
         
    runner_result = subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        capture_output=True,
        text=True
    )
    if runner_result.returncode == 0:
        print("✓ Fixtures: PASSED")
    else:
        print("✗ Fixtures: FAILED")
        print(runner_result.stderr or runner_result.stdout)
        return 2
    
    print("\n" + "=" * 60)
    print("STATUS: Level 0 - Normal operations")
    print("=" * 60)
    return 0


def validate():
    """Run full validation suite."""
    print("Running full validation...")
    log_event("VALIDATE", "Full validation initiated")
    
    # Critic
    print("\n--- Critic ---")
    critic_result = subprocess.run(
        [sys.executable, str(CRITIC_PATH)],
        cwd=str(PROJECT_ROOT)
    )
    
    # Fixtures
    print("\n--- Contract Runner ---")
    runner_result = subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        cwd=str(PROJECT_ROOT)
    )
    
    # Summary
    print("\n--- Summary ---")
    if critic_result.returncode == 0 and runner_result.returncode == 0:
        print("✓ All validations PASSED")
        log_event("VALIDATE", "All validations passed")
        return 0
    else:
        print("✗ Validation FAILED")
        log_event("VALIDATE", "Validation failed")
        return 1


def rollback():
    """Rollback to the last known-good commit."""
    print("=" * 60)
    print("ROLLBACK PROCEDURE")
    print("=" * 60)
    
    # Check for uncommitted changes
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    
    if status_result.stdout.strip():
        print("\n⚠️  Uncommitted changes detected:")
        print(status_result.stdout)
        print("\nStashing changes before rollback...")
        subprocess.run(["git", "stash", "push", "-m", "emergency-rollback-stash"],
                       cwd=str(PROJECT_ROOT))
    
    # Get current commit
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    ).stdout.strip()
    
    print(f"\nCurrent commit: {current}")
    
    # Revert to previous commit
    print("Reverting to previous commit...")
    result = subprocess.run(
        ["git", "revert", "HEAD", "--no-edit"],
        cwd=str(PROJECT_ROOT)
    )
    
    if result.returncode == 0:
        log_event("ROLLBACK", f"Reverted commit {current}")
        print("\n✓ Rollback complete")
        print("\nNext steps:")
        print("1. Review what went wrong")
        print("2. Create an ADR documenting the issue")
        print("3. Fix and re-apply changes with proper ceremony")
        return 0
    else:
        log_event("ROLLBACK", f"Rollback failed for {current}")
        print("\n✗ Rollback failed - manual intervention required")
        return 1


def quarantine(reason: str = "Manual quarantine triggered"):
    """Enter quarantine mode - blocks all write operations."""
    if QUARANTINE_FILE.exists():
        print("⚠️  System is already in quarantine mode")
        return 1
    
    print("=" * 60)
    print("ENTERING QUARANTINE MODE")
    print("=" * 60)
    
    # Get current git hash
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    ).stdout.strip()
    
    # Create quarantine file
    quarantine_data = {
        "entered": datetime.now().isoformat(),
        "reason": reason,
        "triggered_by": "emergency.py",
        "git_hash": git_hash,
        "steward_notified": False
    }
    
    QUARANTINE_FILE.write_text(json.dumps(quarantine_data, indent=2))
    log_event("QUARANTINE", f"Entered: {reason}")
    
    print(f"\n✓ Quarantine file created: {QUARANTINE_FILE}")
    print(f"  Reason: {reason}")
    print(f"  Git hash: {git_hash}")
    print("\n⚠️  ALL WRITE OPERATIONS ARE NOW BLOCKED")
    print("\nTo exit quarantine:")
    print("  python TOOLS/emergency.py --mode=restore")
    
    return 0


def restore():
    """Exit quarantine mode."""
    if not QUARANTINE_FILE.exists():
        print("System is not in quarantine mode")
        return 1
    
    print("=" * 60)
    print("EXITING QUARANTINE MODE")
    print("=" * 60)
    
    quarantine_data = json.loads(QUARANTINE_FILE.read_text())
    print(f"\nQuarantine was entered: {quarantine_data['entered']}")
    print(f"Reason: {quarantine_data['reason']}")
    
    # Confirm
    print("\n⚠️  Confirm quarantine exit by running validation:")
    
    # Run validation
    if validate() != 0:
        print("\n✗ Validation failed - cannot exit quarantine")
        print("  Fix issues before attempting restore")
        return 1
    
    # Remove quarantine file
    QUARANTINE_FILE.unlink()
    log_event("QUARANTINE", "Exited - system restored to normal operation")
    
    print("\n✓ Quarantine lifted")
    print("  System is now in normal operation")
    
    return 0


def constitutional_reset(tag: str):
    """Reset to a known-good tagged release."""
    print("=" * 60)
    print("CONSTITUTIONAL RESET")
    print("=" * 60)
    print(f"\n⚠️  This will reset ALL CANON files to tag: {tag}")
    print("    CONTEXT will be preserved (decisions are history)")
    print("    This is a Level 4 crisis response")
    
    # Verify tag exists
    tag_check = subprocess.run(
        ["git", "rev-parse", tag],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    
    if tag_check.returncode != 0:
        print(f"\n✗ Tag '{tag}' does not exist")
        print("  Available tags:")
        subprocess.run(["git", "tag", "-l"], cwd=str(PROJECT_ROOT))
        return 1
    
    # Create backup
    backup_branch = f"backup-before-constitutional-reset-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"\nCreating backup branch: {backup_branch}")
    subprocess.run(["git", "branch", backup_branch], cwd=str(PROJECT_ROOT))
    
    # Checkout CANON files from tag
    print(f"\nRestoring CANON files from {tag}...")
    subprocess.run(
        ["git", "checkout", tag, "--", "CANON/"],
        cwd=str(PROJECT_ROOT)
    )
    
    log_event("CONSTITUTIONAL-RESET", f"Reset CANON to {tag}, backup: {backup_branch}")
    
    # Rebuild cortex
    print("\nRebuilding cortex...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "CORTEX" / "cortex.build.py")],
        cwd=str(PROJECT_ROOT)
    )
    
    print("\n" + "=" * 60)
    print("CONSTITUTIONAL RESET COMPLETE")
    print("=" * 60)
    print(f"\n✓ CANON reset to: {tag}")
    print(f"✓ Backup branch: {backup_branch}")
    print("\nNext steps:")
    print("1. Review all changes since the tagged release")
    print("2. Re-apply valid changes with proper ceremony")
    print("3. Document this incident in an ADR")
    print("4. Notify stakeholders")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="AGS Emergency CLI - Crisis handling for governance failures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python TOOLS/emergency.py --status
  python TOOLS/emergency.py --mode=validate
  python TOOLS/emergency.py --mode=quarantine --reason="Suspected canon corruption"
  python TOOLS/emergency.py --mode=constitutional-reset --tag=v1.0.0
        """
    )
    parser.add_argument("--status", action="store_true", help="Check current system status")
    parser.add_argument("--mode", choices=["validate", "rollback", "quarantine", "restore", "constitutional-reset"],
                        help="Emergency mode to execute")
    parser.add_argument("--reason", default="Manual trigger", help="Reason for quarantine")
    parser.add_argument("--tag", help="Tag for constitutional reset")
    
    args = parser.parse_args()
    
    if args.status:
        return check_status()
    
    if not args.mode:
        parser.print_help()
        return 0
    
    if args.mode == "validate":
        return validate()
    elif args.mode == "rollback":
        return rollback()
    elif args.mode == "quarantine":
        return quarantine(args.reason)
    elif args.mode == "restore":
        return restore()
    elif args.mode == "constitutional-reset":
        if not args.tag:
            print("Error: --tag is required for constitutional-reset")
            return 1
        return constitutional_reset(args.tag)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

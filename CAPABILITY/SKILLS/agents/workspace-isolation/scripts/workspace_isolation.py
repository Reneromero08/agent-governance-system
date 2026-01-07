#!/usr/bin/env python3
"""Workspace Isolation - Git worktree/branch management for parallel agent work.

This module provides mechanical, safe workspace isolation using git worktrees.
Agents can work in parallel without conflicting with each other or the main workspace.

Lifecycle:
  1. create  - Create worktree + branch for a task
  2. work    - Agent works in isolated worktree
  3. merge   - Merge branch to main (only after validation passes)
  4. cleanup - Remove worktree + delete branch

Usage:
  python workspace_isolation.py create <task_id>
  python workspace_isolation.py status
  python workspace_isolation.py merge <task_id>
  python workspace_isolation.py cleanup <task_id>
  python workspace_isolation.py cleanup-stale
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repo root detection
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]  # Up from scripts/ -> workspace-isolation/ -> agents/ -> SKILLS/ -> CAPABILITY/


class WorkspaceIsolationError(Exception):
    """Raised when workspace isolation operations fail."""
    pass


def run_git(args: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    cmd = ["git"] + args
    cwd = cwd or REPO_ROOT
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if check and result.returncode != 0:
        raise WorkspaceIsolationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def get_current_branch() -> str:
    """Get the current branch name."""
    result = run_git(["branch", "--show-current"])
    return result.stdout.strip()


def is_on_branch() -> bool:
    """Check if we're on a branch (not detached HEAD)."""
    result = run_git(["symbolic-ref", "-q", "HEAD"], check=False)
    return result.returncode == 0


def is_workspace_dirty() -> bool:
    """Check if the workspace has uncommitted changes."""
    result = run_git(["status", "--porcelain"])
    return bool(result.stdout.strip())


def branch_exists(branch_name: str) -> bool:
    """Check if a branch exists."""
    result = run_git(["branch", "--list", branch_name])
    return bool(result.stdout.strip())


def worktree_exists(path: Path) -> bool:
    """Check if a worktree exists at the given path."""
    return path.exists() and (path / ".git").exists()


def get_worktrees() -> List[Dict[str, str]]:
    """Get list of all worktrees with their details."""
    result = run_git(["worktree", "list", "--porcelain"])
    worktrees = []
    current = {}

    for line in result.stdout.strip().split("\n"):
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue

        if line.startswith("worktree "):
            current["path"] = line[9:]
        elif line.startswith("HEAD "):
            current["head"] = line[5:]
        elif line.startswith("branch "):
            current["branch"] = line[7:].replace("refs/heads/", "")
        elif line == "detached":
            current["detached"] = True

    if current:
        worktrees.append(current)

    return worktrees


def get_task_worktree_path(task_id: str) -> Path:
    """Get the standard worktree path for a task."""
    # Worktrees go in parent directory of repo
    return REPO_ROOT.parent / f"wt-{task_id}"


def get_task_branch_name(task_id: str) -> str:
    """Get the standard branch name for a task."""
    return f"task/{task_id}"


def create_worktree(task_id: str, base_branch: str = "main") -> Dict[str, str]:
    """Create an isolated worktree for a task.

    Args:
        task_id: Unique identifier for the task (e.g., "2.4.1C.5")
        base_branch: Branch to base the new branch on (default: main)

    Returns:
        Dict with worktree details (path, branch)

    Raises:
        WorkspaceIsolationError: If creation fails
    """
    # SAFETY: Verify we're in the main repo root, not a worktree
    cwd = Path.cwd().resolve()
    if cwd != REPO_ROOT:
        raise WorkspaceIsolationError(
            f"Must run create from main repo root.\n"
            f"  Current: {cwd}\n"
            f"  Expected: {REPO_ROOT}\n"
            f"  Fix: cd \"{REPO_ROOT}\""
        )

    if not is_on_branch():
        raise WorkspaceIsolationError("Cannot create worktree from detached HEAD. Checkout a branch first.")

    branch_name = get_task_branch_name(task_id)
    wt_path = get_task_worktree_path(task_id)

    # Check if worktree already exists
    if worktree_exists(wt_path):
        raise WorkspaceIsolationError(f"Worktree already exists at {wt_path}")

    # Create branch if it doesn't exist
    if not branch_exists(branch_name):
        run_git(["branch", branch_name, base_branch])

    # Create worktree
    run_git(["worktree", "add", str(wt_path), branch_name])

    return {
        "task_id": task_id,
        "branch": branch_name,
        "path": str(wt_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "base_branch": base_branch
    }


def get_worktree_status(task_id: Optional[str] = None) -> Dict:
    """Get status of worktrees.

    Args:
        task_id: Optional task ID to filter by

    Returns:
        Dict with worktree status information
    """
    worktrees = get_worktrees()
    current_branch = get_current_branch()
    is_dirty = is_workspace_dirty()

    # Filter if task_id provided
    if task_id:
        branch_name = get_task_branch_name(task_id)
        worktrees = [wt for wt in worktrees if wt.get("branch") == branch_name]

    # Separate main worktree from task worktrees
    main_worktree = None
    task_worktrees = []

    for wt in worktrees:
        path = Path(wt.get("path", ""))
        if path == REPO_ROOT:
            main_worktree = wt
        elif "wt-" in path.name:
            # Extract task_id from path
            task_id_from_path = path.name.replace("wt-", "")
            wt["task_id"] = task_id_from_path
            task_worktrees.append(wt)

    return {
        "current_branch": current_branch,
        "is_dirty": is_dirty,
        "main_worktree": main_worktree,
        "task_worktrees": task_worktrees,
        "total_worktrees": len(worktrees)
    }


def merge_task(task_id: str, delete_branch: bool = False) -> Dict:
    """Merge a task branch into main.

    IMPORTANT: This should only be called after validation passes!

    Args:
        task_id: Task ID to merge
        delete_branch: Whether to delete the branch after merge

    Returns:
        Dict with merge result

    Raises:
        WorkspaceIsolationError: If merge fails
    """
    # SAFETY: Verify we're in the main repo root, not a worktree
    cwd = Path.cwd().resolve()
    if cwd != REPO_ROOT:
        raise WorkspaceIsolationError(
            f"Must run merge from main repo root, not from worktree.\n"
            f"  Current: {cwd}\n"
            f"  Expected: {REPO_ROOT}\n"
            f"  Fix: cd \"{REPO_ROOT}\""
        )

    branch_name = get_task_branch_name(task_id)
    wt_path = get_task_worktree_path(task_id)

    # Must be on main to merge
    current = get_current_branch()
    if current != "main":
        raise WorkspaceIsolationError(
            f"Must be on main branch to merge (currently on {current}).\n"
            f"  Fix: git checkout main"
        )

    # Check main is clean
    if is_workspace_dirty():
        raise WorkspaceIsolationError("Main workspace is dirty. Commit or stash changes first.")

    # Check branch exists
    if not branch_exists(branch_name):
        raise WorkspaceIsolationError(f"Branch {branch_name} does not exist")

    # Perform merge (no-ff to preserve branch history)
    try:
        run_git(["merge", "--no-ff", branch_name, "-m", f"Merge {branch_name} into main"])
    except WorkspaceIsolationError as e:
        raise WorkspaceIsolationError(f"Merge failed: {e}. Resolve conflicts manually.")

    result = {
        "task_id": task_id,
        "branch": branch_name,
        "merged_into": "main",
        "merged_at": datetime.utcnow().isoformat() + "Z"
    }

    if delete_branch:
        # Can only delete branch if worktree is removed first
        if worktree_exists(wt_path):
            raise WorkspaceIsolationError(
                f"Cannot delete branch while worktree exists. Run cleanup first."
            )
        run_git(["branch", "-d", branch_name])
        result["branch_deleted"] = True

    return result


def cleanup_worktree(task_id: str, force: bool = False, delete_branch: bool = True) -> Dict:
    """Remove a worktree and optionally delete its branch.

    Args:
        task_id: Task ID to clean up
        force: Force removal even if worktree has uncommitted changes
        delete_branch: Also delete the branch (default: True)

    Returns:
        Dict with cleanup result

    Raises:
        WorkspaceIsolationError: If cleanup fails
    """
    # SAFETY: Verify we're in the main repo root, not the worktree being cleaned
    cwd = Path.cwd().resolve()
    if cwd != REPO_ROOT:
        raise WorkspaceIsolationError(
            f"Must run cleanup from main repo root, not from worktree.\n"
            f"  Current: {cwd}\n"
            f"  Expected: {REPO_ROOT}\n"
            f"  Fix: cd \"{REPO_ROOT}\""
        )

    branch_name = get_task_branch_name(task_id)
    wt_path = get_task_worktree_path(task_id)

    result = {
        "task_id": task_id,
        "branch": branch_name,
        "worktree_path": str(wt_path),
        "cleaned_at": datetime.utcnow().isoformat() + "Z"
    }

    # Remove worktree if it exists
    if worktree_exists(wt_path):
        # Safety: don't remove if we're inside it
        cwd = Path.cwd().resolve()
        wt_resolved = wt_path.resolve()
        if str(cwd).startswith(str(wt_resolved)):
            raise WorkspaceIsolationError(
                f"Cannot remove worktree while inside it. cd to main repo first."
            )

        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(wt_path))

        run_git(args)
        result["worktree_removed"] = True
    else:
        result["worktree_removed"] = False
        result["note"] = "Worktree did not exist"

    # Delete branch if requested
    if delete_branch and branch_exists(branch_name):
        try:
            # Use -D to force delete if branch is not fully merged
            run_git(["branch", "-D", branch_name])
            result["branch_deleted"] = True
        except WorkspaceIsolationError:
            # Try with -d first (safe delete)
            result["branch_deleted"] = False
            result["branch_note"] = "Branch not fully merged. Use force=True or merge first."

    return result


def cleanup_stale_worktrees(dry_run: bool = True) -> Dict:
    """Find and remove stale worktrees (task branches already merged to main).

    Args:
        dry_run: If True, only report what would be cleaned (default: True)

    Returns:
        Dict with cleanup results
    """
    worktrees = get_worktrees()
    stale = []

    # Get merged branches
    result = run_git(["branch", "--merged", "main"])
    merged_branches = set(b.strip().lstrip("* ") for b in result.stdout.strip().split("\n") if b.strip())

    for wt in worktrees:
        path = Path(wt.get("path", ""))
        branch = wt.get("branch", "")

        # Skip main worktree
        if path == REPO_ROOT:
            continue

        # Check if it's a task worktree with merged branch
        if "wt-" in path.name and branch in merged_branches:
            task_id = path.name.replace("wt-", "")
            stale.append({
                "task_id": task_id,
                "path": str(path),
                "branch": branch
            })

    results = {
        "dry_run": dry_run,
        "stale_count": len(stale),
        "stale_worktrees": stale,
        "cleaned": []
    }

    if not dry_run:
        for s in stale:
            try:
                cleanup_result = cleanup_worktree(s["task_id"], force=True, delete_branch=True)
                results["cleaned"].append(cleanup_result)
            except WorkspaceIsolationError as e:
                results.setdefault("errors", []).append({
                    "task_id": s["task_id"],
                    "error": str(e)
                })

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Workspace Isolation - Git worktree/branch management for parallel agent work",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  create <task_id>     Create isolated worktree for a task
  status [task_id]     Show worktree status
  merge <task_id>      Merge task branch into main
  cleanup <task_id>    Remove worktree and delete branch
  cleanup-stale        Find and remove all stale worktrees

Examples:
  python workspace_isolation.py create 2.4.1C.5
  python workspace_isolation.py status
  python workspace_isolation.py merge 2.4.1C.5
  python workspace_isolation.py cleanup 2.4.1C.5 --force
  python workspace_isolation.py cleanup-stale --apply
"""
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create isolated worktree for a task")
    create_parser.add_argument("task_id", help="Task identifier (e.g., 2.4.1C.5)")
    create_parser.add_argument("--base", default="main", help="Base branch (default: main)")

    # status command
    status_parser = subparsers.add_parser("status", help="Show worktree status")
    status_parser.add_argument("task_id", nargs="?", help="Optional task ID to filter")

    # merge command
    merge_parser = subparsers.add_parser("merge", help="Merge task branch into main")
    merge_parser.add_argument("task_id", help="Task identifier to merge")
    merge_parser.add_argument("--delete-branch", action="store_true", help="Delete branch after merge")

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove worktree and delete branch")
    cleanup_parser.add_argument("task_id", help="Task identifier to clean up")
    cleanup_parser.add_argument("--force", action="store_true", help="Force removal even with uncommitted changes")
    cleanup_parser.add_argument("--keep-branch", action="store_true", help="Keep the branch after removing worktree")

    # cleanup-stale command
    stale_parser = subparsers.add_parser("cleanup-stale", help="Find and remove stale worktrees")
    stale_parser.add_argument("--apply", action="store_true", help="Actually remove stale worktrees (default: dry-run)")

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create_worktree(args.task_id, args.base)
            print(json.dumps(result, indent=2))
            print(f"\nWorktree created. Next steps:")
            print(f"  cd \"{result['path']}\"")
            print(f"  # Do your work")
            print(f"  # Run tests")
            print(f"  # When done: python workspace_isolation.py merge {args.task_id}")

        elif args.command == "status":
            result = get_worktree_status(args.task_id)
            print(json.dumps(result, indent=2))

        elif args.command == "merge":
            result = merge_task(args.task_id, args.delete_branch)
            print(json.dumps(result, indent=2))

        elif args.command == "cleanup":
            result = cleanup_worktree(args.task_id, args.force, not args.keep_branch)
            print(json.dumps(result, indent=2))

        elif args.command == "cleanup-stale":
            result = cleanup_stale_worktrees(dry_run=not args.apply)
            print(json.dumps(result, indent=2))
            if result["dry_run"] and result["stale_count"] > 0:
                print(f"\nFound {result['stale_count']} stale worktrees. Run with --apply to remove them.")

        return 0

    except WorkspaceIsolationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

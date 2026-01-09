#!/usr/bin/env python3
"""Validate workspace-isolation skill compliance."""
import json
import sys
from pathlib import Path


def validate(input_path: Path) -> dict:
    """
    Validate that an agent session followed workspace-isolation rules.
    
    Input should contain:
    - git_operations: list of git commands executed in order
    - approvals: list of user approvals received
    
    Returns validation result with any violations.
    """
    with open(input_path) as f:
        data = json.load(f)
    
    git_ops = data.get("git_operations", [])
    approvals = data.get("approvals", [])
    
    violations = []
    commit_count = 0
    work_complete = False
    has_approval_for_commit = False
    has_approval_for_merge = False
    committed_before_approval = False
    
    approval_set = set(approvals)
    
    for op in git_ops:
        cmd = op.get("command", "")
        
        # Track work completion signal
        if op.get("type") == "work_complete":
            work_complete = True
            continue
            
        # Check commits
        if cmd.startswith("git commit"):
            if not work_complete:
                violations.append({
                    "rule": "Phase 1",
                    "violation": "Committed before all work was complete",
                    "command": cmd
                })
            
            if "commit" not in approval_set:
                violations.append({
                    "rule": "Approval Required",
                    "violation": "Committed without user approval",
                    "command": cmd
                })
                committed_before_approval = True
            
            commit_count += 1
            
            if commit_count > 1 and "--amend" not in cmd:
                violations.append({
                    "rule": "Phase 2",
                    "violation": "Multiple commits detected - should be single commit",
                    "command": cmd
                })
        
        # Check merges
        if cmd.startswith("git merge"):
            if "merge" not in approval_set:
                violations.append({
                    "rule": "Approval Required",
                    "violation": "Merged without user approval",
                    "command": cmd
                })
        
        # Check pushes
        if cmd.startswith("git push"):
            if "push" not in approval_set:
                violations.append({
                    "rule": "Approval Required",
                    "violation": "Pushed without user approval",
                    "command": cmd
                })
        
        # Check amends
        if "--amend" in cmd:
            if "amend" not in approval_set:
                violations.append({
                    "rule": "Approval Required",
                    "violation": "Amended without user approval",
                    "command": cmd
                })
            
            # Check if amend was safe
            if op.get("pushed_to_remote", False):
                violations.append({
                    "rule": "Phase 4",
                    "violation": "Amended a commit that was already pushed",
                    "command": cmd
                })
    
    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "summary": {
            "total_commits": commit_count,
            "work_completed_before_commit": work_complete,
            "approvals_received": list(approval_set)
        }
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: validate.py <input.json> <output.json>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    result = validate(input_path)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    sys.exit(0 if result["valid"] else 1)

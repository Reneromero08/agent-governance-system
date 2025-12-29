#!/usr/bin/env python3
"""
Validate CI trigger policy: workflows must not trigger on push to main.

Usage: python validate_ci_triggers.py <workflow_file>
"""

import sys
import yaml
from pathlib import Path

def check_workflow_triggers(workflow_path: Path) -> dict:
    """Check if workflow has forbidden push triggers to main."""
    try:
        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)
        
        violations = []
        
        # Check 'on' triggers
        triggers = workflow.get('on', {})
        
        # If 'on' is a list (shorthand), it's likely safe
        if isinstance(triggers, list):
            if 'push' in triggers:
                violations.append("Workflow uses shorthand 'push' trigger without branch restriction")
        
        # If 'on' is a dict, check push.branches
        elif isinstance(triggers, dict):
            push_config = triggers.get('push')
            if push_config is not None:
                # push exists - check if it targets main
                if isinstance(push_config, dict):
                    branches = push_config.get('branches', [])
                    if 'main' in branches or 'master' in branches:
                        violations.append(f"Workflow triggers on push to main/master: {branches}")
                else:
                    # push: without config means all branches
                    violations.append("Workflow has unrestricted 'push' trigger")
        
        return {
            "ok": len(violations) == 0,
            "policy": "no_push_to_main",
            "violations": violations
        }
    
    except Exception as e:
        return {
            "ok": False,
            "policy": "no_push_to_main",
            "violations": [f"Error reading workflow: {e}"]
        }

def main():
    if len(sys.argv) != 2:
        print("Usage: validate_ci_triggers.py <workflow_file>")
        return 1
    
    workflow_file = Path(sys.argv[1])
    result = check_workflow_triggers(workflow_file)
    
    import json
    print(json.dumps(result, indent=2))
    
    return 0 if result["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())

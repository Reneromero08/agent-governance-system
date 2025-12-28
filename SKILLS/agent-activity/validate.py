#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def main(input_path: Path, expected_path: Path) -> int:
    try:
        # We ignore expected_path content because it varies by time
        actual = json.loads(input_path.read_text())
        
        if actual.get("count") != 2:
            print(f"FAILURE: Expected count 2, got {actual.get('count')}")
            return 1
            
        agents = {a["session_id"]: a for a in actual["active_agents"]}
        
        # Verify Session 1
        s1 = agents.get("session-1")
        if not s1:
            print("FAILURE: Missing session-1")
            return 1
        if s1["working_on"] != "INVARIANTS":
            print(f"FAILURE: session-1 expected 'INVARIANTS', got '{s1.get('working_on')}'")
            return 1
        if s1["tool"] != "canon_read":
            print(f"FAILURE: session-1 expected tool 'canon_read', got '{s1.get('tool')}'")
            return 1
            
        # Verify Session 2
        s2 = agents.get("session-2")
        if not s2:
            print("FAILURE: Missing session-2")
            return 1
        if s2["working_on"] != "Query: packer":
            print(f"FAILURE: session-2 expected 'Query: packer', got '{s2.get('working_on')}'")
            return 1
        if s2["tool"] != "cortex_query":
            print(f"FAILURE: session-2 expected tool 'cortex_query', got '{s2.get('tool')}'")
            return 1

        print("Validation passed")
        return 0
        
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))

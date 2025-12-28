
import sys
import json
import time

def main():
    # Read input line
    line = sys.stdin.read()
    if not line:
        return

    try:
        req = json.loads(line)
    except Exception:
        # If input isn't JSON, we just return (or could write stderr)
        return

    method = req.get("method")
    
    # Happy path
    if method == "echo":
        res = {
            "jsonrpc": "2.0",
            "result": req.get("params"),
            "id": req.get("id")
        }
        print(json.dumps(res))
        return

    # Fail cases controlled by params
    params = req.get("params", {})
    
    if method == "crash":
        sys.exit(1)
        
    if method == "stderr":
        sys.stderr.write("Errors happened")
        # Also print valid json to stdout to confuse things?
        print("{}")
        return

    if method == "sleep":
        ms = params.get("ms", 1000)
        time.sleep(ms / 1000.0)
        print("{}")
        return
        
    if method == "bloat":
        size = params.get("size", 100000)
        print("a" * size)
        return

    if method == "malformed":
        print("{not json")
        return

if __name__ == "__main__":
    main()

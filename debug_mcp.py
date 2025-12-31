import subprocess
import sys
import json
import time

def run_probe():
    cmd = [sys.executable, "LAW/CONTRACTS/ags_mcp_entrypoint.py"]
    print(f"Launching: {cmd}")
    
    # Force binary streams
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="D:\\CCC 2.0\\AI\\agent-governance-system",
        # bufsize=0 is fine, but let's rely on explicit file handles
    )

    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "debug_probe", "version": "1.0"}
        }
    }
    
    msg = json.dumps(req).encode("utf-8")
    header = f"Content-Length: {len(msg)}\r\n\r\n".encode("ascii")
    
    print(f"Sending {len(header) + len(msg)} bytes...")
    proc.stdin.write(header + msg)
    proc.stdin.flush()
    
    print("Reading response header...")
    line = proc.stdout.readline()
    print(f"HEADER RAW: {line}")
    
    # Read rest if header ok
    if b"Content-Length" in line:
        try:
           val = int(line.split(b":")[1].strip())
           proc.stdout.readline() # \r\n
           body = proc.stdout.read(val)
           print(f"BODY: {body.decode('utf-8')}")
           
           # Call tool loop
           print("\nCalling read_canon...")
           
           # Notify initialized
           notify = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode("utf-8")
           proc.stdin.write(f"Content-Length: {len(notify)}\r\n\r\n".encode("ascii") + notify)
           proc.stdin.flush()

           # Call tool
           tool = json.dumps({
               "jsonrpc": "2.0", 
               "id": 2, 
               "method": "tools/call", 
               "params": {"name": "read_canon", "arguments": {"doc": "GENESIS.md"}}
           }).encode("utf-8")
           proc.stdin.write(f"Content-Length: {len(tool)}\r\n\r\n".encode("ascii") + tool)
           proc.stdin.flush()
           
           # Read tool response
           print("Reading tool response...")
           while True:
                h = proc.stdout.readline()
                if h and b"Content-Length" in h:
                    l = int(h.split(b":")[1].strip())
                    proc.stdout.readline()
                    b = proc.stdout.read(l)
                    print(f"TOOL RES: {b.decode('utf-8')}")
                    break
                    
        except Exception as e:
            print(f"Error: {e}")

    proc.terminate()

if __name__ == "__main__":
    run_probe()

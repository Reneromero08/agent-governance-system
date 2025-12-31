import subprocess
import sys
import json
import time

def run_probe():
    cmd = [sys.executable, "LAW/CONTRACTS/ags_mcp_entrypoint.py"]
    print(f"Launching: {cmd}")
    
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="D:\\CCC 2.0\\AI\\agent-governance-system",
        bufsize=0  # Unbuffered
    )

    # Initialize request
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
    
    print("Sending initialize...")
    try:
        proc.stdin.write(header + msg)
        proc.stdin.flush()
    except Exception as e:
        print(f"Write failed: {e}")
        return

    # Read response
    print("Reading response...")
    start = time.time()
    
    # Read header
    headers = {}
    while True:
        if time.time() - start > 5:
            print("TIMEOUT waiting for headers")
            break
            
        raw_line = proc.stdout.readline()
        if not raw_line:
            break
        line = raw_line.decode("ascii", errors="ignore").strip()
        if not line:
            break
        print(f"Header received: {line}")
        try:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
        except:
            pass

    if "content-length" in headers:
        length = int(headers["content-length"])
        body = proc.stdout.read(length)
        print(f"Body: {body.decode('utf-8')}")
    else:
        print("No content-length header found or timeout")
        # Check stderr
        err = proc.stderr.read()
        if err:
            print(f"STDERR: {err.decode('utf-8')}")

    proc.terminate()

if __name__ == "__main__":
    run_probe()

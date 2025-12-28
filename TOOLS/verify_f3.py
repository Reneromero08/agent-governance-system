import os
import subprocess
import sys
import shutil
from pathlib import Path

# Paths
ROOT = Path("d:/CCC 2.0/AI/agent-governance-system")
PROTOTYPE = ROOT / "CATALYTIC-DPT/LAB/PROTOTYPES/f3_cas_prototype.py"
TEMP = Path(os.getenv("TEMP")) / "f3_verify"

def run_cmd(args):
    return subprocess.run([sys.executable, str(PROTOTYPE)] + args, capture_output=True, text=True)

def main():
    if TEMP.exists(): shutil.rmtree(TEMP)
    TEMP.mkdir()

    src = TEMP / "src"
    src.mkdir()
    (src / "test.txt").write_text("Hello CAS")
    
    pack = TEMP / "pack"
    dst = TEMP / "dst"

    print("--- Build ---")
    r = run_cmd(["build", "--src", str(src), "--out", str(pack)])
    print(r.stdout or r.stderr)
    if r.returncode != 0: sys.exit(1)

    print("--- Reconstruct ---")
    r = run_cmd(["reconstruct", "--pack", str(pack), "--dst", str(dst)])
    print(r.stdout or r.stderr)
    if r.returncode != 0: sys.exit(1)

    print("--- Verify ---")
    r = run_cmd(["verify", "--src", str(src), "--dst", str(dst)])
    print(r.stdout or r.stderr)
    if r.returncode != 0: sys.exit(1)

    print("SUCCESS: F3 Prototype Verified.")

if __name__ == "__main__":
    main()

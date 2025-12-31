
import json
import subprocess
import sys
import re
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "SWARM_MANIFEST.json"
RUN_PY_PATH = REPO_ROOT / "CAPABILITY" / "SKILLS" / "agents" / "qwen-cli" / "run.py"
MODEL = "qwen2.5:7b"

def run_agent(file_path: str, instruction: str) -> str:
    """Run the Qwen agent on a file with instructions."""
    cmd = [
        sys.executable,
        str(RUN_PY_PATH),
        instruction,
        "--file", str(REPO_ROOT / file_path),
        "--model", MODEL,
        "--no-stream"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, encoding='utf-8')
        if result.returncode != 0:
            print(f"Agent failed for {file_path}: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        print(f"Exception running agent for {file_path}: {e}")
        return None

def extract_code(response: str) -> str:
    """Extract code block from response."""
    if not response:
        return None
    
    # Look for python code blocks
    matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[-1] # Take the last one (often the final fixed version)
    
    matches = re.findall(r'```\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[-1]
        
    return None

def main():
    if not MANIFEST_PATH.exists():
        print("SWARM_MANIFEST.json not found!")
        return

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    # Filter out files we already fixed manually
    SKIP_FILES = {
        "CAPABILITY/TESTBENCH/spectrum/test_spectrum02_resume.py",
        "CAPABILITY/TESTBENCH/spectrum/test_spectrum03_chain.py",
        "CAPABILITY/TESTBENCH/test_ags_phase6_adapter_contract.py",
        "CAPABILITY/TESTBENCH/test_runtime_guard.py",
        "CAPABILITY/TESTBENCH/test_schemas.py",
        "CAPABILITY/TESTBENCH/test_restore_runner.py",
        "CAPABILITY/TESTBENCH/test_cortex_integration.py",
        "CAPABILITY/TESTBENCH/test_governance_coverage.py",
        "CAPABILITY/TESTBENCH/test_memoization.py",
        "CAPABILITY/TESTBENCH/test_packing_hygiene.py",
        "CAPABILITY/TESTBENCH/test_pipeline_chain.py",
        "CAPABILITY/TESTBENCH/test_pipeline_verify_cli.py",
        "CAPABILITY/TESTBENCH/test_pipelines.py",
        "CAPABILITY/TESTBENCH/test_swarm_reuse.py",
        "CAPABILITY/TESTBENCH/test_swarm_runtime.py",
        "CAPABILITY/TESTBENCH/test_pipeline_dag.py",
        "CAPABILITY/TESTBENCH/test_verify_bundle.py"
    }
    
    tasks = [t for t in manifest if t["file"] not in SKIP_FILES]
    
    print(f"Processing {len(tasks)} remaining tasks via Swarm Orchestrator...")
    
    success_count = 0
    fail_count = 0
    
    for task in tqdm(tasks):
        file_path = task["file"]
        print(f"\nProcessing: {file_path}")
        
        response = run_agent(file_path, task["instruction"])
        if not response:
            print(f"  ❌ No response from agent")
            fail_count += 1
            continue
            
        fixed_code = extract_code(response)
        if not fixed_code:
            print(f"  ❌ No code block found in response")
            fail_count += 1
            continue
            
        # Basic validation (check if it's not empty and resembles code)
        if len(fixed_code) < 50:
            print(f"  ❌ Code block too short")
            fail_count += 1
            continue
            
        # Write back
        try:
            target_path = REPO_ROOT / file_path
            target_path.write_text(fixed_code, encoding='utf-8')
            print(f"  ✅ Fixed code written to {file_path}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ Failed to write file: {e}")
            fail_count += 1

    print(f"\nSwarm Run Complete: {success_count} success, {fail_count} failed")

if __name__ == "__main__":
    main()

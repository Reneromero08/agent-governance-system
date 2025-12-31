
import json
import subprocess
import sys
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "SWARM_MANIFEST.json"
RUN_PY_PATH = REPO_ROOT / "CAPABILITY" / "SKILLS" / "agents" / "qwen-cli" / "run.py"

# Use the tiny model as requested for speed and parallelism
MODEL = "qwen2.5:1.5b" 
WORKER_COUNT = 4

def run_agent(file_path: str, instruction: str) -> str:
    """Run the Qwen agent on a file with instructions."""
    # Add explicit instruction to be concise and code-only to help tiny models
    enhanced_instruction = instruction + "\nIMPORTANT: Return ONLY the fixed python code block. Do not confirm. Do not explain."
    
    cmd = [
        sys.executable,
        str(RUN_PY_PATH),
        enhanced_instruction,
        "--file", str(REPO_ROOT / file_path),
        "--model", MODEL,
        "--no-stream"
    ]
    
    try:
        # Shorter timeout for tiny models
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding='utf-8')
        if result.returncode != 0:
            return None
        return result.stdout
    except Exception as e:
        return None

def extract_code(response: str) -> str:
    """Extract code block from response."""
    if not response:
        return None
    
    # Look for python code blocks
    matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[-1] 
    
    matches = re.findall(r'```\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[-1]
    
    # Tiny models sometimes just dump the code without blocks if told strictly
    # But usually strictly following the run.py format warrants blocks.
    # We'll stick to blocks for safety.
    return None

def process_task(task):
    file_path = task["file"]
    instruction = task["instruction"]
    
    response = run_agent(file_path, instruction)
    if not response:
        return False, file_path, "No response"
        
    fixed_code = extract_code(response)
    if not fixed_code:
        return False, file_path, "No code block"
        
    if len(fixed_code) < 50:
        return False, file_path, "Code too short"
        
    try:
        target_path = REPO_ROOT / file_path
        target_path.write_text(fixed_code, encoding='utf-8')
        return True, file_path, "Success"
    except Exception as e:
        return False, file_path, str(e)

def main():
    if not MANIFEST_PATH.exists():
        print("SWARM_MANIFEST.json not found!")
        return

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    # Filter out files we already fixed manually (Cohort B)
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
        "CAPABILITY/TESTBENCH/test_verify_bundle.py",
        # Previous run successes (optimistic check)
        "CAPABILITY/TESTBENCH/test_adversarial_pipeline_resume.py",
        "CAPABILITY/TESTBENCH/test_ags_phase6_capability_revokes.py"
    }
    
    tasks = [t for t in manifest if t["file"] not in SKIP_FILES]
    
    print(f"Deploying {WORKER_COUNT} Ants with {MODEL} for {len(tasks)} tasks...")
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks)):
            success, path, msg = future.result()
            if success:
                print(f"  ✅ {path}")
                success_count += 1
            else:
                print(f"  ❌ {path}: {msg}")
                fail_count += 1

    print(f"\nTurbo Swarm Complete: {success_count} success, {fail_count} failed")

if __name__ == "__main__":
    main()

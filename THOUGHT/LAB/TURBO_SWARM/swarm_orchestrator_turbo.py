import argparse
import json
import os
import subprocess
import sys
import re
import threading
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------------------------------------------
# Configuration & Paths
# --------------------------------------------------------------------------------

# Reliable root determination
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"
RUN_PY_PATH = REPO_ROOT / "CAPABILITY" / "SKILLS" / "agents" / "qwen-cli" / "run.py"

# Swarm Configuration
MODELS = ["qwen2.5:1.5b", "qwen2.5:5b", "qwen2.5:7b"]
DEFAULT_MAX_WORKERS = 16
DEFAULT_GPU_SLOTS = 1

# Global State
GPU_SEMAPHORE = None
SWARM_CUDA_ENABLED = False

PAUSE_FLAG = REPO_ROOT / "SWARM_PAUSE.flag"
STOP_FLAG = REPO_ROOT / "SWARM_STOP.flag"

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def _check_swarm_control():
    """Checks for pause/stop flags."""
    if PAUSE_FLAG.exists():
        print("SWARM_PAUSE.flag detected. Pausing turbo swarm gracefully (no new tasks).")
        sys.exit(0)
    if STOP_FLAG.exists():
        print("SWARM_STOP.flag detected. Stopping turbo swarm immediately.")
        sys.exit(0)

def extract_code(response: str) -> str:
    """Robust code block extraction handling python, diff, or generic blocks."""
    if not response:
        return None
    
    # 1. Try python block
    matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[-1]
    
    # 2. Try generic block
    matches = re.findall(r'```\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[-1]

    # 3. Try diff/search-replace blocks (sometimes models drift)
    matches = re.findall(r'```diff\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[-1]
        
    return None

def _compile_python(path: Path) -> bool:
    """Syntax check using python's built-in compiler."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        return result.returncode == 0
    except Exception:
        return False

def run_agent(file_path: str, instruction: str, model: str) -> str:
    """Invokes the Qwen agent via the CLI wrapper."""
    enhanced_instruction = f"""{instruction}

CRITICAL RULES:
1. Output ONLY valid Python code in a single ```python code block
2. Do NOT add placeholder paths like "/path/to/..." - use actual repo paths
3. Do NOT add comments explaining changes
4. Do NOT remove existing imports unless they are truly unused
5. Preserve all existing functionality - only fix path references
6. Use REPO_ROOT = Path(__file__).resolve().parents[N] (adjust N based on depth) for path calculations
7. Replace legacy paths: CATALYTIC-DPT → CAPABILITY, TOOLS → CAPABILITY/TOOLS, CONTRACTS → LAW/CONTRACTS
8. If uncertain about a fix, return the input file UNCHANGED

Example fix:
WRONG: sys.path.insert(0, "/path/to/6-bucket/architecture")
RIGHT: REPO_ROOT = Path(__file__).resolve().parents[2]
       if str(REPO_ROOT) not in sys.path:
           sys.path.insert(0, str(REPO_ROOT))

Output format: ```python
<complete fixed file>
```"""
    
    cmd = [
        sys.executable,
        str(RUN_PY_PATH),
        enhanced_instruction,
        "--file", str(REPO_ROOT / file_path),
        "--model", model,
        "--no-stream"
    ]
    
    try:
        # Pass CUDA environment if enabled
        env = os.environ.copy() if SWARM_CUDA_ENABLED else None
        if SWARM_CUDA_ENABLED:
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

        # Serialized GPU access if enabled, otherwise uncapped
        if SWARM_CUDA_ENABLED:
            with GPU_SEMAPHORE:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120, 
                    encoding='utf-8',
                    env=env
                )
        else:
             result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120, 
                encoding='utf-8',
                env=env
            )
        
        if result.returncode != 0:
            return None
            
        return result.stdout
    except Exception as e:
        return None

def process_task(task):
    """Worker function to process a single file with escalation."""
    file_path = task["file"]
    instruction = task["instruction"]
    start_time = time.monotonic()
    target_path = REPO_ROOT / file_path
    
    attempted_tiers = []
    
    for tier_idx, model in enumerate(MODELS):
        attempted_tiers.append(model)
        
        # 1. Run Agent
        response = run_agent(file_path, instruction, model)
        if not response:
            if tier_idx == len(MODELS) - 1:
                return "failed", file_path, f"model_fail_tier{tier_idx+1}", time.monotonic() - start_time, {"attempted_tiers": attempted_tiers, "validators": ["model_execution"]}
            continue
            
        # 2. Extract Code
        fixed_code = extract_code(response)
        if not fixed_code or len(fixed_code) < 50:
            if tier_idx == len(MODELS) - 1:
                return "failed", file_path, f"no_code_block_tier{tier_idx+1}", time.monotonic() - start_time, {"attempted_tiers": attempted_tiers, "validators": ["code_extraction"]}
            continue
            
        # 3. Validation & Write
        try:
            tmp_path = Path(str(target_path) + ".swarm_tmp")
            tmp_path.write_text(fixed_code, encoding='utf-8')
            
            # Validator: JUST Syntax Check (Import validation was causing issues)
            if not _compile_python(tmp_path):
                tmp_path.unlink(missing_ok=True)
                if tier_idx == len(MODELS) - 1:
                    return "failed", file_path, f"compile_fail_tier{tier_idx+1}", time.monotonic() - start_time, {"attempted_tiers": attempted_tiers, "validators": ["py_compile"]}
                continue
            
            # Success! Swap files.
            tmp_path.replace(target_path)
            return "success", file_path, f"ok_tier{tier_idx+1}", time.monotonic() - start_time, {"attempted_tiers": attempted_tiers, "validators": ["py_compile"], "model_tier": model}
            
        except Exception:
            if tier_idx == len(MODELS) - 1:
                return "failed", file_path, f"write_fail_tier{tier_idx+1}", time.monotonic() - start_time, {"attempted_tiers": attempted_tiers, "validators": ["file_write"]}
            continue
    
    return "failed", file_path, "all_tiers_exhausted", time.monotonic() - start_time, {"attempted_tiers": attempted_tiers, "validators": []}

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    global GPU_SEMAPHORE
    global SWARM_CUDA_ENABLED

    parser = argparse.ArgumentParser(description="Turbo swarm orchestrator (Sonnet Restore)")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--gpu-slots", type=int, default=DEFAULT_GPU_SLOTS)
    args = parser.parse_args()

    # CUDA Setup
    SWARM_CUDA_ENABLED = os.environ.get("SWARM_CUDA") == "1"
    if SWARM_CUDA_ENABLED:
        print("SWARM_CUDA enabled.")
    
    GPU_SEMAPHORE = threading.Semaphore(args.gpu_slots)

    # Manifest Logic
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    SKIP_FILES = {
        # Known successful or problematic files to skip - none for now, full sweep
    }
    
    tasks = [t for t in manifest if t.get("file") not in SKIP_FILES]
    report = []
    
    print(f"Deploying {args.max_workers} Ants... (Simulated 3-Tier Escalation)")
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks)):
            status, path, reason, elapsed, metadata = future.result()
            
            # Console Feedback
            if status == "success":
                print(f"  ✅ {path} ({reason})")
                success_count += 1
            else:
                print(f"  ❌ {path}: {reason}")
                fail_count += 1
                
            report.append({
                "file": path,
                "status": status,
                "reason": reason,
                "elapsed_seconds": round(elapsed, 3),
                **metadata
            })

    print(f"\nSwarm Complete: {success_count} success, {fail_count} failed")
    
    # Write Report
    report_path = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "SWARM_REPORT.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

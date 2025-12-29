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

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "SWARM_MANIFEST.json"
RUN_PY_PATH = REPO_ROOT / "CAPABILITY" / "SKILLS" / "agents" / "qwen-cli" / "run.py"

# Use the tiny model as requested for speed and parallelism
MODEL = "qwen2.5:1.5b" 
DEFAULT_MAX_WORKERS = 16
DEFAULT_GPU_SLOTS = 1
GPU_SEMAPHORE = None
SWARM_CUDA_ENABLED = False

# Pause/Stop mechanism for graceful pause
PAUSE_FLAG = REPO_ROOT / "SWARM_PAUSE.flag"
STOP_FLAG = REPO_ROOT / "SWARM_STOP.flag"


def _check_swarm_control():
    if PAUSE_FLAG.exists():
        print("SWARM_PAUSE.flag detected. Pausing turbo swarm gracefully (no new tasks).")
        sys.exit(0)
    if STOP_FLAG.exists():
        print("SWARM_STOP.flag detected. Stopping turbo swarm immediately.")
        sys.exit(0)


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
        env = os.environ.copy() if SWARM_CUDA_ENABLED else None
        with GPU_SEMAPHORE:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8',
                env=env
            )
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
    # Fallback: nothing found
    return None


def _compile_python(path: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    return result.returncode == 0


def process_task(task):
    file_path = task["file"]
    instruction = task["instruction"]
    start_time = time.monotonic()
    
    response = run_agent(file_path, instruction)
    if not response:
        return "failed", file_path, "model_fail", time.monotonic() - start_time
        
    fixed_code = extract_code(response)
    if not fixed_code:
        return "failed", file_path, "no_code_block", time.monotonic() - start_time
        
    if len(fixed_code) < 50:
        return "failed", file_path, "code_too_short", time.monotonic() - start_time
        
    try:
        target_path = REPO_ROOT / file_path
        tmp_path = Path(str(target_path) + ".swarm_tmp")
        tmp_path.write_text(fixed_code, encoding='utf-8')
        if not _compile_python(tmp_path):
            tmp_path.unlink(missing_ok=True)
            return "failed", file_path, "compile_fail", time.monotonic() - start_time
        tmp_path.replace(target_path)
        return "success", file_path, "ok", time.monotonic() - start_time
    except Exception as e:
        return "failed", file_path, "write_fail", time.monotonic() - start_time


def main():
    global GPU_SEMAPHORE
    global SWARM_CUDA_ENABLED

    parser = argparse.ArgumentParser(description="Turbo swarm orchestrator")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--gpu-slots", type=int, default=DEFAULT_GPU_SLOTS)
    args = parser.parse_args()

    SWARM_CUDA_ENABLED = os.environ.get("SWARM_CUDA") == "1"
    if SWARM_CUDA_ENABLED:
        print("SWARM_CUDA enabled (env passthrough to qwen-cli).")
    else:
        print("SWARM_CUDA disabled.")

    # Pause/Stop check early
    _check_swarm_control()

    GPU_SEMAPHORE = threading.Semaphore(args.gpu_slots)

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
    report = []
    for task in manifest:
        if task["file"] in SKIP_FILES:
            report.append({
                "file": task["file"],
                "status": "skipped",
                "reason": "skip_list",
                "elapsed_seconds": 0.0
            })
    
    print(f"Deploying {args.max_workers} Ants with {MODEL} for {len(tasks)} tasks...")
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks)):
            status, path, reason, elapsed = future.result()
            report.append({
                "file": path,
                "status": status,
                "reason": reason,
                "elapsed_seconds": round(elapsed, 3)
            })
            if status == "success":
                print(f"  ✅ {path}")
                success_count += 1
            else:
                print(f"  ❌ {path}: {reason}")
                fail_count += 1

    print(f"\nTurbo Swarm Complete: {success_count} success, {fail_count} failed")
    report_path = REPO_ROOT / "SWARM_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

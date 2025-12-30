import argparse
import json
import subprocess
import sys
import re
import time
import threading
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------------------------------------------
# Configuration & Paths
# --------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"

# 3-Tier Escalation Models
MODELS = ["qwen2.5:1.5b", "qwen2.5:7b", "qwen3:8b"]
DEFAULT_MAX_WORKERS = 12
OLLAMA_SLOTS = 4
OLLAMA_API = "http://localhost:11434/api/generate"

# ROLES
TIER_1_ANT = "qwen2.5:1.5b"
TIER_2_FOREMAN = "qwen2.5:7b"
TIER_3_ARCHITECT = "qwen3:8b"

# Global state
OLLAMA_SEM = threading.Semaphore(OLLAMA_SLOTS)
SESSION = requests.Session()

# FILE CACHE - Read once, reuse across tiers
FILE_CACHE = {}
CACHE_LOCK = threading.Lock()

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def preload_models():
    """Pre-load all models into memory."""
    print("🔥 Pre-loading models...")
    for model in MODELS:
        print(f"   Loading {model}...", end='', flush=True)
        try:
            SESSION.post(
                OLLAMA_API,
                json={
                    "model": model,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=30
            )
            print(" ✅")
        except:
            print(" ⚠️")
    print()

def get_live_model(requested):
    """Fallback to Tier-2 if requested model (Architecture) is missing."""
    try:
        r = SESSION.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            existing = [m["name"] for m in r.json().get("models", [])]
            if requested in existing: return requested
            for e in existing:
                if e.startswith(requested.split(':')[0]): return e
        return TIER_2_FOREMAN
    except:
        return TIER_2_FOREMAN

def get_cached_file(file_path: str) -> str:
    """Thread-safe file caching - read once per file."""
    with CACHE_LOCK:
        if file_path not in FILE_CACHE:
            target_path = REPO_ROOT / file_path
            if target_path.exists():
                FILE_CACHE[file_path] = target_path.read_text(encoding='utf-8')
            else:
                FILE_CACHE[file_path] = ""
        return FILE_CACHE[file_path]

def extract_code(response: str) -> str:
    """Extract code block from response."""
    if not response:
        return None
    
    matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    matches = re.findall(r'```\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    return None

def _compile_python(path: Path) -> bool:
    """Syntax check."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def run_agent_raw(model: str, prompt: str, timeout: int = 90) -> str:
    """Optimized HTTP call with shorter default timeout."""
    try:
        with OLLAMA_SEM:
            response = SESSION.post(
                OLLAMA_API,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048,  # Reduced from 3072
                        "num_ctx": 3072       # Reduced from 4096
                    }
                },
                timeout=timeout
            )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        return ""
    except:
        return ""

# --------------------------------------------------------------------------------
# Tiered Logic (OPTIMIZED)
# --------------------------------------------------------------------------------

def architect_plan(file_path: str, instruction: str) -> str:
    """Tier-3: SHORTER planning prompt - just the essentials."""
    content = get_cached_file(file_path)
    
    # OPTIMIZATION: Send only first 1000 chars for planning
    snippet = content[:1000] if len(content) > 1000 else content
    
    prompt = f"""Analyze and create a fix plan:

Task: {instruction}

File (first 1000 chars):
{snippet}

Output bullet points ONLY. No code. No backticks."""
    
    plan = run_agent_raw(get_live_model(TIER_3_ARCHITECT), prompt, timeout=30)
    
    if "```" in plan or len(plan) < 20:
        return None
    
    return plan

def ant_execute(file_path: str, plan: str, instruction: str) -> str:
    """Tier-1: Ant execution with CACHED file content."""
    content = get_cached_file(file_path)
    
    prompt = f"""Fix this file following the plan:

PLAN:
{plan}

TASK: {instruction}

FILE:
```python
{content}
```

Return ONLY fixed code in ```python block."""
    
    return run_agent_raw(TIER_1_ANT, prompt, timeout=60)

def fast_verify(fixed_code: str) -> bool:
    """FAST verification - skip foreman LLM call, just syntax check."""
    # OPTIMIZATION: Skip the foreman LLM verification for speed
    # Just do basic checks
    if not fixed_code or len(fixed_code) < 50:
        return False
    if "REPO_ROOT" not in fixed_code:  # Basic sanity check
        return False
    return True

# --------------------------------------------------------------------------------
# Worker function (OPTIMIZED)
# --------------------------------------------------------------------------------

def process_task(task):
    """OPTIMIZED: Try Ant first, escalate to Architect only on failure."""
    file_path = task["file"]
    instruction = task["instruction"]
    start_time = time.monotonic()
    target_path = REPO_ROOT / file_path
    
    # OPTIMIZATION 1: Try ANT FIRST (skip planning for simple fixes)
    response = ant_execute(file_path, "Apply the fix directly", instruction)
    fixed_code = extract_code(response)
    
    if fixed_code and len(fixed_code) > 50:
        tmp_path = Path(str(target_path) + ".swarm_tmp")
        try:
            tmp_path.write_text(fixed_code, encoding='utf-8')
            
            if _compile_python(tmp_path) and fast_verify(fixed_code):
                tmp_path.replace(target_path)
                return "success", file_path, "ant_direct", time.monotonic() - start_time, {"model": TIER_1_ANT}
            
            tmp_path.unlink(missing_ok=True)
        except:
            if tmp_path.exists(): tmp_path.unlink()
    
    # ESCALATION: Ant failed, use Architect planning
    plan = architect_plan(file_path, instruction)
    if not plan:
        return "failed", file_path, "architect_fail", time.monotonic() - start_time, {}

    # Execute with plan
    response = ant_execute(file_path, plan, instruction)
    fixed_code = extract_code(response)
    if not fixed_code:
        return "failed", file_path, "ant_no_code", time.monotonic() - start_time, {}
        
    # Verify
    tmp_path = Path(str(target_path) + ".swarm_tmp")
    try:
        tmp_path.write_text(fixed_code, encoding='utf-8')
        
        if not _compile_python(tmp_path):
            tmp_path.unlink(missing_ok=True)
            return "failed", file_path, "syntax_fail", time.monotonic() - start_time, {}
        
        if not fast_verify(fixed_code):
            tmp_path.unlink(missing_ok=True)
            return "failed", file_path, "verify_fail", time.monotonic() - start_time, {}
            
        tmp_path.replace(target_path)
        return "success", file_path, "ant_planned", time.monotonic() - start_time, {"model": TIER_1_ANT}
        
    except Exception as e:
        if tmp_path.exists(): tmp_path.unlink()
        return "failed", file_path, f"error", time.monotonic() - start_time, {}

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Turbo Swarm v3.1 (Optimized)")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--ollama-slots", type=int, default=OLLAMA_SLOTS)
    args = parser.parse_args()

    global OLLAMA_SEM
    OLLAMA_SEM = threading.Semaphore(args.ollama_slots)

    print("\n" + "="*60)
    print("🚀 TURBO SWARM v3.1 (OPTIMIZED) 🚀".center(60))
    print("="*60 + "\n")

    try:
        test = SESSION.get("http://localhost:11434/api/tags", timeout=2)
        if test.status_code != 200:
            print("❌ Ollama not responding")
            return
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return

    preload_models()

    if not MANIFEST_PATH.exists():
        print(f"❌ Manifest not found")
        return

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    if not tasks:
        print("❌ No tasks")
        return
    
    print(f"🐜 {args.max_workers} workers × {args.ollama_slots} ollama slots")
    print(f"📊 Ant-first strategy with Architect escalation")
    print(f"📁 {len(tasks)} files\n")
    
    report = []
    success = 0
    failed = 0
    start_time = time.time()
    
    with tqdm(
        total=len(tasks),
        desc="🐜 Swarm",
        bar_format="{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ncols=100,
        colour="green"
    ) as pbar:
        try:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {executor.submit(process_task, t): t for t in tasks}
                
                for future in as_completed(futures):
                    status, path, reason, elapsed, meta = future.result()
                    
                    if status == "success":
                        success += 1
                    else:
                        failed += 1
                    
                    pbar.set_postfix_str(f"✅ {success} ❌ {failed}")
                    pbar.update(1)
                        
                    report.append({
                        "file": path,
                        "status": status,
                        "reason": reason,
                        "elapsed": round(elapsed, 2),
                        **meta
                    })
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted")
        finally:
            SESSION.close()

    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"✅ Success: {success}  ❌ Failed: {failed}")
    print(f"⏱️  Total: {total_time:.1f}s  Avg: {total_time/len(tasks):.1f}s/file")
    print("="*60)
    
    out = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "SWARM_REPORT.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"📋 {out}")

if __name__ == "__main__":
    main()
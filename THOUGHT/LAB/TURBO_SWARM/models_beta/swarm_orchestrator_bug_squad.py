#!/usr/bin/env python3
"""
🐞 BUG SQUAD: The Council of Experts 🐞
------------------------------------------------
A dedicated swarm mode for fixing stubborn failures.
It reads SWARM_REPORT.json, finds failures, and uses a
voting/consensus mechanism to solve them.

Architecture:
- The Academic: Correctness focus.
- The Pragmatist: Passing focus.
- The Security: Safety focus.
- THE CHAIRMAN: Integration and Final Decision.
"""

import sys
import json
import time
import argparse
import logging
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class SquadConfig:
    max_workers: int = 1  # Sequential focus
    model_member: str = "qwen2.5-coder:7b"
    model_chairman: str = "qwen2.5-coder:7b"
    keep_alive: str = "20m"
    max_ctx: int = 8192
    
    # Reports
    input_report: str = "BUG_SQUAD_TARGETS.json"
    output_report: str = "SQUAD_REPORT.json"

OLLAMA_API = "http://localhost:11434/api/generate"
# Reuse global locks from previous swarm design if needed, but we are sequential here mostly.
_PRINT_LOCK = threading.Lock()

# --------------------------------------------------------------------------------
# Logging & Utils
# --------------------------------------------------------------------------------

def _setup_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("BugSquad")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run_agent_raw(model: str, prompt: str, timeout: int, logger: logging.Logger) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "20m",
        "options": {
            "temperature": 0.3,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
    }
    try:
        cmd = ["curl", "-s", "-X", "POST", OLLAMA_API, "-H", "Content-Type: application/json", "-d", json.dumps(payload), "--max-time", str(timeout)]
        # logger.debug(f"Calling {model}...")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
        if res.returncode != 0:
            return ""
        resp = json.loads(res.stdout)
        return resp.get("response", "")
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return ""

def extract_code(text: str) -> Optional[str]:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    # Fallback: if mostly code, return whole thing? No, unsafe.
    return None

# --------------------------------------------------------------------------------
# Council Members
# --------------------------------------------------------------------------------

def academic_opinion(cfg: SquadConfig, file_path: str, content: str, error: str, instruction: str, logger: logging.Logger) -> str:
    prompt = (
        "[THE ACADEMIC]\n"
        "Analyze the failure. Focus on theoretical correctness and best practices.\n"
        "Do not write code yet. Just specific technical steps to fix it.\n\n"
        f"TASK: {instruction}\n"
        f"ERROR: {error}\n"
        f"FILE CONTENT:\n{content}\n"
    )
    return run_agent_raw(cfg.model_member, prompt, 120, logger)

def pragmatist_opinion(cfg: SquadConfig, file_path: str, content: str, error: str, instruction: str, logger: logging.Logger) -> str:
    prompt = (
        "[THE PRAGMATIST]\n"
        "Focus on getting the test to PASS. Ignore 'clean code' if necessary.\n"
        "What is the simplest, ugliest hack that fixes this error?\n\n"
        f"TASK: {instruction}\n"
        f"ERROR: {error}\n"
        f"FILE CONTENT:\n{content}\n"
    )
    return run_agent_raw(cfg.model_member, prompt, 120, logger)

def security_opinion(cfg: SquadConfig, file_path: str, content: str, error: str, instruction: str, logger: logging.Logger) -> str:
    prompt = (
        "[THE SECURITY OFFICER]\n"
        "Review the file and error. Ensure we don't break permissions or delete wrong files.\n"
        "Are there any safety concerns with the proposed task?\n\n"
        f"TASK: {instruction}\n"
        f"ERROR: {error}\n"
        f"FILE CONTENT:\n{content}\n"
    )
    return run_agent_raw(cfg.model_member, prompt, 120, logger)

def chairman_decision(cfg: SquadConfig, file_path: str, content: str, opinions: Dict[str, str], instruction: str, logger: logging.Logger) -> str:
    prompt = (
        "[THE CHAIRMAN]\n"
        "You are the final decision maker. Review the opinions of the council.\n"
        "Synthesize a FINAL, CORRECT Python file.\n"
        "Strictly follow the pragmatist if it solves the error, but listen to security.\n"
        "Return ONLY the full python file in a ```python block.\n\n"
        f"OPINION ACADEMIC: {opinions['academic']}\n\n"
        f"OPINION PRAGMATIST: {opinions['pragmatist']}\n\n"
        f"OPINION SECURITY: {opinions['security']}\n\n"
        f"TASK: {instruction}\n"
        f"ORIGINAL FILE:\n{content}\n"
    )
    res = run_agent_raw(cfg.model_chairman, prompt, 300, logger) # Long timeout for synthesis
    return extract_code(res)

# --------------------------------------------------------------------------------
# Main Loop
# --------------------------------------------------------------------------------

def process_failure(cfg: SquadConfig, item: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    file_path = item["file"]
    last_error = item.get("last_error", "Unknown Error")
    instruction = item.get("instruction", "Fix this file.") # We might need to fetch instruction if not in report
    
    # Try to fetch instruction from MANIFEST if missing (hacky)
    # Ideally SWARM_REPORT should contain instruction. It doesn't currently. 
    # We will assume the file existence check + read is enough context for "Fix this".
    
    logger.info(f"🐞 Convening Council for: {file_path}")
    
    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Cannot read file: {e}")
        return {"file": file_path, "status": "failed", "reason": "read_error"}

    # 1. Gather Opinions
    logger.info("  - Gathering opinions...")
    opinions = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(academic_opinion, cfg, file_path, content, last_error, instruction, logger)
        f2 = executor.submit(pragmatist_opinion, cfg, file_path, content, last_error, instruction, logger)
        f3 = executor.submit(security_opinion, cfg, file_path, content, last_error, instruction, logger)
        
        opinions["academic"] = f1.result()
        opinions["pragmatist"] = f2.result()
        opinions["security"] = f3.result()

    logger.debug(f"  - Academic: {opinions['academic'][:100]}...")
    logger.debug(f"  - Pragmatist: {opinions['pragmatist'][:100]}...")

    # 2. Chairman Decides
    logger.info("  - Chairman deliberating...")
    fixed_code = chairman_decision(cfg, file_path, content, opinions, instruction, logger)
    
    if not fixed_code:
        logger.error("  - Chairman failed to produce code.")
        return {"file": file_path, "status": "failed", "reason": "chairman_failed"}

    # 3. Write
    logger.info("  - Solution ratified. Writing to disk.")
    try:
        Path(file_path).write_text(fixed_code, encoding="utf-8")
        return {"file": file_path, "status": "fixed", "reason": "council_consensus"}
    except Exception as e:
        return {"file": file_path, "status": "failed", "reason": str(e)}

def main():
    logger = _setup_logging(True)
    cfg = SquadConfig()
    
    # Load Report
    report_path = Path(cfg.input_report)
    if not report_path.exists():
        # Try finding it in swarm dir
        report_path = Path(__file__).parent / "SWARM_REPORT.json"
        
    if not report_path.exists():
        logger.error("No SWARM_REPORT.json found.")
        sys.exit(1)
        
    data = json.loads(report_path.read_text(encoding="utf-8"))
    failures = [x for x in data if x.get("status") != "success"]
    
    logger.info(f"Loaded {len(data)} results. Found {len(failures)} failures to squash.")
    
    results = []
    
    # Needs manifest for instruction lookup?
    manifest_path = Path(__file__).parent / "SWARM_MANIFEST.json"
    manifest = {}
    if manifest_path.exists():
        m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for t in m_data:
            manifest[t["file"]] = t.get("instruction", "")

    for fail in failures:
        # Patch instruction if missing
        if "instruction" not in fail:
            fail["instruction"] = manifest.get(fail["file"], "Fix the errors in this file.")
            
        res = process_failure(cfg, fail, logger)
        results.append(res)
        
    # Save squad report
    Path(cfg.output_report).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("🐞 Council Adjourned.")

if __name__ == "__main__":
    main()

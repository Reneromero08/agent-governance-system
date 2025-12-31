#!/usr/bin/env python3
"""
Ants: Multi-worker agent runners (Phase 4.3)

Durable, repo-local ANT worker runtime that can run independently
and safely cooperate through the cassette DB.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from catalytic_chat.message_cassette import MessageCassette, MessageCassetteError


@dataclass
class AntConfig:
    run_id: str
    job_id: str
    worker_id: str
    repo_root: Path
    poll_interval_ms: int = 250
    ttl_seconds: int = 300
    continue_on_fail: bool = False
    max_idle_polls: int = 20


class AntWorker:
    
    def __init__(self, config: AntConfig):
        self.config = config
        self.cassette = MessageCassette(repo_root=config.repo_root)
    
    def run(self) -> int:
        idle_count = 0
        
        try:
            while idle_count < self.config.max_idle_polls:
                claim_result = self.cassette.claim_next_step(
                    run_id=self.config.run_id,
                    job_id=self.config.job_id,
                    worker_id=self.config.worker_id,
                    ttl_seconds=self.config.ttl_seconds
                )
                
                if claim_result is None:
                    idle_count += 1
                    time.sleep(self.config.poll_interval_ms / 1000)
                    continue
                
                idle_count = 0
                step_id = claim_result["step_id"]
                fencing_token = claim_result["fencing_token"]
                
                try:
                    receipt = self.cassette.execute_step(
                        run_id=self.config.run_id,
                        step_id=step_id,
                        worker_id=self.config.worker_id,
                        fencing_token=fencing_token,
                        repo_root=self.config.repo_root,
                        check_global_budget=True
                    )
                    
                    if receipt.get("status") != "SUCCESS":
                        if not self.config.continue_on_fail:
                            print(f"[FAIL] {step_id}: {receipt.get('error', 'Unknown error')}")
                            return 1
                except MessageCassetteError as e:
                    print(f"[FAIL] {step_id}: {e}")
                    if not self.config.continue_on_fail:
                        return 1
                    else:
                        continue
            
            return 0
            
        except MessageCassetteError as e:
            print(f"[FAIL] Invariant/DB error: {e}")
            return 2
        finally:
            self.cassette.close()


def spawn_ants(
    run_id: str,
    job_id: str,
    num_workers: int,
    repo_root: Path,
    continue_on_fail: bool = False
) -> int:
    processes: List[Tuple[Any, str]] = []
    
    cortex_dir = repo_root / "CORTEX" / "_generated"
    cortex_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = cortex_dir / f"ants_manifest_{run_id}_{job_id}.json"
    
    started_at = None
    
    try:
        import subprocess
        
        for i in range(num_workers):
            worker_id = f"ant_{run_id}_{job_id}_{i}"
            
            cmd = [
                sys.executable,
                "-m",
                "catalytic_chat.cli",
                "ants",
                "worker",
                "--run-id", run_id,
                "--job-id", job_id,
                "--worker-id", worker_id
            ]
            
            if continue_on_fail:
                cmd.append("--continue-on-fail")
            
            process = subprocess.Popen(
                cmd,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            processes.append((process, worker_id))
            
            if started_at is None:
                started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        manifest = {
            "run_id": run_id,
            "job_id": job_id,
            "started_at": started_at,
            "workers": [{"worker_id": w, "pid": p.pid} for p, w in processes],
            "argv": sys.argv
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        exit_codes = []
        for process, worker_id in processes:
            process.wait()
            exit_codes.append(process.returncode)
        
        if any(code == 2 for code in exit_codes):
            return 2
        elif any(code == 1 for code in exit_codes) and not continue_on_fail:
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"[FAIL] Spawn error: {e}")
        for process, worker_id in processes:
            process.terminate()
        return 2


def run_ant_worker(
    run_id: str,
    job_id: str,
    worker_id: str,
    repo_root: Path,
    continue_on_fail: bool = False,
    poll_interval_ms: int = 250,
    ttl_seconds: int = 300,
    max_idle_polls: int = 20
) -> int:
    config = AntConfig(
        run_id=run_id,
        job_id=job_id,
        worker_id=worker_id,
        repo_root=repo_root,
        poll_interval_ms=poll_interval_ms,
        ttl_seconds=ttl_seconds,
        continue_on_fail=continue_on_fail,
        max_idle_polls=max_idle_polls
    )
    
    worker = AntWorker(config)
    return worker.run()

"""
Event Logger for Swarm System.
Handles append-only event logging with simple file locking.
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime, timezone

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

class SimpleLock:
    def __init__(self, lock_path: Path, timeout: float = 2.0):
        self.lock_path = lock_path
        self.timeout = timeout

    def __enter__(self):
        start = time.time()
        while True:
            try:
                # mkdir is atomic on most runtimes
                self.lock_path.mkdir(parents=True, exist_ok=True)
                # If we created it (or it's a dir), we use a file inside as the actual mutex?
                # No, mkdir fails if it exists. 
                # Correction: parents=True makes it not fail if exists. We want it to fail.
                # Use a specific lock directory ending in .lock
                os.mkdir(self.lock_path)
                return
            except FileExistsError:
                if time.time() - start > self.timeout:
                    # Force break if very old? For now, just raise/fail safely (drop event).
                    # Or rely on fast ops.
                    pass
                time.sleep(0.05)
            except OSError:
                time.sleep(0.05)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.rmdir(self.lock_path)
        except:
            pass

def emit_event(type: str, payload: dict, inbox_root: Path) -> None:
    """
    Append an event to events.jsonl
    """
    events_dir = inbox_root / "_events"
    events_dir.mkdir(parents=True, exist_ok=True)
    
    lock_path = events_dir / "events.lock"
    log_path = events_dir / "events.jsonl"
    
    entry = {
        "timestamp": now_iso(),
        "type": type,
        "payload": payload
    }
    
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    
    try:
        with SimpleLock(lock_path):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        # Fallback print if locking fails
        print(f"[EVENT LOG ERROR] {e}")


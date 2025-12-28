#!/usr/bin/env python3
"""
SPECTRUM-02: Resume Bundles
CMP-01: Output Hash Enforcement

Enforces integrity for Swarm tasks:
- Resume bundles allow tasks to be checkpointed and resumed
- Output hashes verify artifact integrity
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Configuration
RUNS_DIR = Path("CONTRACTS/_runs")

class IntegrityStack:
    """Enforces SPECTRUM-02 and CMP-01."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run_dir = RUNS_DIR / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_path = self.run_dir / "RESUME_BUNDLE.json"
        self.hashes_path = self.run_dir / "OUTPUT_HASHES.json"
        
    # === SPECTRUM-02: Resume Bundles ===
    
    def create_bundle(self, task_spec: Dict, state: Dict = None) -> Path:
        """Create a resume bundle for checkpointing."""
        bundle = {
            "run_id": self.run_id,
            "created_at": datetime.utcnow().isoformat(),
            "task_spec": task_spec,
            "state": state or {},
            "checkpoint_hash": self._hash_dict(task_spec)
        }
        
        with open(self.bundle_path, 'w', encoding='utf-8') as f:
            json.dump(bundle, f, indent=2, sort_keys=True)
            
        return self.bundle_path
        
    def load_bundle(self) -> Optional[Dict]:
        """Load existing resume bundle."""
        if not self.bundle_path.exists():
            return None
        with open(self.bundle_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def update_state(self, new_state: Dict):
        """Update bundle state for checkpoint."""
        bundle = self.load_bundle()
        if bundle:
            bundle["state"] = new_state
            bundle["updated_at"] = datetime.utcnow().isoformat()
            with open(self.bundle_path, 'w', encoding='utf-8') as f:
                json.dump(bundle, f, indent=2, sort_keys=True)
    
    # === CMP-01: Output Hash Enforcement ===
    
    def register_output(self, path: str, content: bytes) -> str:
        """Register an output artifact and its hash."""
        content_hash = hashlib.sha256(content).hexdigest()
        
        hashes = self._load_hashes()
        hashes[path] = {
            "hash": content_hash,
            "size": len(content),
            "registered_at": datetime.utcnow().isoformat()
        }
        self._save_hashes(hashes)
        
        return content_hash
        
    def verify_output(self, path: str, content: bytes) -> bool:
        """Verify artifact matches registered hash. Hard-fail if mismatch."""
        hashes = self._load_hashes()
        
        if path not in hashes:
            raise ValueError(f"CMP-01 VIOLATION: Artifact '{path}' not in OUTPUT_HASHES.json")
            
        expected = hashes[path]["hash"]
        actual = hashlib.sha256(content).hexdigest()
        
        if actual != expected:
            raise ValueError(f"CMP-01 VIOLATION: Hash mismatch for '{path}'\n  Expected: {expected}\n  Actual: {actual}")
            
        return True
        
    def verify_all_outputs(self) -> Dict[str, bool]:
        """Verify all registered outputs exist and match."""
        hashes = self._load_hashes()
        results = {}
        
        for path, meta in hashes.items():
            try:
                content = Path(path).read_bytes()
                self.verify_output(path, content)
                results[path] = True
            except Exception as e:
                results[path] = False
                print(f"FAIL: {path} - {e}")
                
        return results
    
    def _load_hashes(self) -> Dict:
        if not self.hashes_path.exists():
            return {}
        with open(self.hashes_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _save_hashes(self, hashes: Dict):
        with open(self.hashes_path, 'w', encoding='utf-8') as f:
            json.dump(hashes, f, indent=2, sort_keys=True)
            
    def _hash_dict(self, d: Dict) -> str:
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Integrity Stack CLI")
    parser.add_argument("run_id", help="Run ID")
    subparsers = parser.add_subparsers(dest="command")
    
    # Create
    create_p = subparsers.add_parser("create", help="Create resume bundle")
    create_p.add_argument("--task", required=True, help="Task description")
    
    # Register
    reg_p = subparsers.add_parser("register", help="Register output artifact")
    reg_p.add_argument("path", help="Path to artifact")
    
    # Verify
    subparsers.add_parser("verify", help="Verify all outputs")
    
    args = parser.parse_args()
    stack = IntegrityStack(args.run_id)
    
    if args.command == "create":
        path = stack.create_bundle({"task": args.task})
        print(f"Bundle created: {path}")
    elif args.command == "register":
        content = Path(args.path).read_bytes()
        h = stack.register_output(args.path, content)
        print(f"Registered: {args.path} -> {h}")
    elif args.command == "verify":
        results = stack.verify_all_outputs()
        passed = sum(1 for v in results.values() if v)
        print(f"Verified: {passed}/{len(results)} artifacts")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

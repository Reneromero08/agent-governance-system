#!/usr/bin/env python3
"""
F3 Prototype: Catalytic Context Compression (CAS Proof of Concept)

Demonstrates how to turn a directory into a "Manifest" (LITE pack)
and reconstruct the full directory from a Content-Addressed Store (CAS).
"""

import os
import hashlib
import json
import shutil
from pathlib import Path

CAS_ROOT = Path("mock_cas_store")
SOURCE_DIR = Path("mock_source_data")
RESTORE_DIR = Path("mock_restored_data")

def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def setup_data():
    """Create random data to verify bit-perfect restoration."""
    if SOURCE_DIR.exists(): shutil.rmtree(SOURCE_DIR)
    SOURCE_DIR.mkdir()
    
    # diverse files
    (SOURCE_DIR / "config.json").write_text('{"key": "value"}')
    (SOURCE_DIR / "script.py").write_text('print("hello")')
    (SOURCE_DIR / "image.bin").write_bytes(os.urandom(1024)) # random binary
    
    print(f"[Setup] Created source data at {SOURCE_DIR}")

def ingest_to_cas(source_dir, cas_root):
    """
    Ingest files into CAS and generate a LITE manifest.
    Returns: manifest dict {path: hash}
    """
    if not cas_root.exists(): cas_root.mkdir()
    
    manifest = {}
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(source_dir).as_posix()
            
            content = file_path.read_bytes()
            file_hash = compute_sha256(content)
            
            # Store in CAS (sharded by first 2 chars)
            shard = cas_root / file_hash[:2]
            shard.mkdir(exist_ok=True)
            blob_path = shard / file_hash
            
            if not blob_path.exists():
                blob_path.write_bytes(content)
                
            manifest[rel_path] = file_hash
            
    print(f"[Ingest] Ingested {len(manifest)} files to CAS.")
    return manifest

def restore_from_manifest(manifest, cas_root, target_dir):
    """Hydrate a directory from a manifest + CAS."""
    if target_dir.exists(): shutil.rmtree(target_dir)
    target_dir.mkdir()
    
    print(f"[Restore] Hydrating {len(manifest)} files to {target_dir}...")
    
    for rel_path, file_hash in manifest.items():
        shard = cas_root / file_hash[:2]
        blob_path = shard / file_hash
        
        if not blob_path.exists():
            raise FileNotFoundError(f"CAS Corruption: Missing blob {file_hash}")
        
        # Determine target location
        item_path = target_dir / rel_path
        item_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy bytes
        content = blob_path.read_bytes()
        
        # Verify hash on read (Navajo Bridge principle: verify as you build)
        if compute_sha256(content) != file_hash:
             raise ValueError(f"Integrity Error: Blob {file_hash} corrupted!")
             
        item_path.write_bytes(content)

def verify_identity(dir1, dir2):
    """Prove byte-for-byte identity."""
    # Simple recursive diff
    hash1 = compute_hash_dir(dir1) # reusing helper logic
    hash2 = compute_hash_dir(dir2)
    
    if hash1 == hash2:
        print("[Verify] SUCCESS: Restored state matches Original state exactly.")
    else:
        print(f"[Verify] FAIL: Hash mismatch! {hash1} vs {hash2}")
        exit(1)

def compute_hash_dir(directory):
    sha = hashlib.sha256()
    for root, _, files in os.walk(directory):
        for file in sorted(files):
            path = Path(root) / file
            sha.update(path.relative_to(directory).as_posix().encode())
            sha.update(path.read_bytes())
    return sha.hexdigest()

def main():
    setup_data()
    
    # 1. Compress (Source -> CAS + Manifest)
    manifest = ingest_to_cas(SOURCE_DIR, CAS_ROOT)
    
    # Display LITE Manifest (Token Savings Proof)
    print(f"[Manifest] {json.dumps(manifest, indent=2)}")
    
    # 2. Reconstruct (Manifest + CAS -> Restore)
    restore_from_manifest(manifest, CAS_ROOT, RESTORE_DIR)
    
    # 3. Prove Integrity
    verify_identity(SOURCE_DIR, RESTORE_DIR)
    
    # Cleanup
    shutil.rmtree(SOURCE_DIR)
    shutil.rmtree(RESTORE_DIR)
    shutil.rmtree(CAS_ROOT)
    print("[Cleanup] All temporary artifacts removed.")

if __name__ == "__main__":
    main()

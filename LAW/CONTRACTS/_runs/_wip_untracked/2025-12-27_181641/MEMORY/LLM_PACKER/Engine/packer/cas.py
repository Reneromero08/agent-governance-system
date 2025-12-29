"""
Content-Addressed Storage (CAS) logic for LLM Packer.
Implements Lane F3 strategy: Catalytic Context Compression.
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional


class ContentAddressedStore:
    """
    Manages a content-addressed blob store.
    
    Layout:
      root/
        ab/
          abcd123... (blob)
    """
    
    def __init__(self, store_root: Path):
        self.root = store_root
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

    def ingest(self, file_path: Path) -> str:
        """
        Ingest a file into the CAS.
        Returns the SHA-256 hash of the content.
        """
        # 1. Calculate Hash
        sha256 = hashlib.sha256()
        try:
            sha256.update(file_path.read_bytes())
        except OSError:
            # If file is unreadable (e.g. lock, permission), let caller handle or skip
            raise
            
        file_hash = sha256.hexdigest()
        
        # 2. Store (Deduplicated)
        # Use first 2 chars for bucketing to avoid huge dirs
        bucket = self.root / file_hash[:2]
        blob_path = bucket / file_hash
        
        if not blob_path.exists():
            bucket.mkdir(exist_ok=True)
            shutil.copy2(file_path, blob_path)
            # Make read-only to simulate immutable store
            os.chmod(blob_path, 0o444)
            
        return file_hash

    def restore(self, file_hash: str, dest_path: Path) -> None:
        """
        Restore a file from CAS to dest_path.
        Raises FileNotFoundError if blob is missing.
        """
        bucket = self.root / file_hash[:2]
        blob_path = bucket / file_hash
        
        if not blob_path.exists():
            raise FileNotFoundError(f"CAS blob missing: {file_hash}")
            
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure destination is writable if it exists (overwrite)
        if dest_path.exists():
            os.chmod(dest_path, 0o666)
            
        shutil.copy2(blob_path, dest_path)
        os.chmod(dest_path, 0o666)
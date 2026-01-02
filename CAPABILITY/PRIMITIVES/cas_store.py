#!/usr/bin/env python3
"""
F3 Prototype: Catalytic Context Compression (CAS)

CLI for Content-Addressed Storage with build/reconstruct/verify.
Deterministic, fail-closed, with safety caps.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_VERIFY_MISMATCH = 2
EXIT_UNSAFE_PATH = 3
EXIT_BOUNDS_EXCEEDED = 4

# Safety caps
MAX_FILES = 5000
MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512MB
MAX_FILE_BYTES = 64 * 1024 * 1024    # 64MB
MAX_PATH_LENGTH = 260


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha.update(chunk)
    return sha.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, separators=(',', ':'), sort_keys=True).encode('utf-8')


def normalize_path(rel: str) -> str:
    """
    Normalizes a repo-relative path:
    - Replaces backslashes with forward slashes.
    - Collapses '.' and '..' components.
    - Rejects absolute paths or paths that escape the root via '..'.
    """
    if not isinstance(rel, str):
        # Handle Path objects if passed
        rel = str(rel)
    
    # Standardize separators
    rel = rel.replace('\\', '/')
    
    # Reject absolute paths
    if rel.startswith('/') or (len(rel) > 1 and rel[1] == ':'):
        raise ValueError(f"Absolute path not allowed: {rel}")
    
    # Normalize components
    parts = []
    for part in rel.split('/'):
        if not part or part == '.':
            continue
        if part == '..':
            raise ValueError(f"Path traversal ('..') not allowed: {rel}")
        parts.append(part)
    
    return '/'.join(parts)

normalize_relpath = normalize_path


def validate_path(rel: str, src_root: Path) -> bool:
    if len(rel) > MAX_PATH_LENGTH:
        return False
    try:
        normalized = normalize_path(rel)
        if normalized != rel.replace('\\', '/').strip('/'):
             # If normalization changed it significantly (like stripping dots), 
             # we might want to check if it's still considered "valid" in the caller's eyes.
             # But usually validate_path is used to check BEFORE normalization or AS part of it.
             pass
    except ValueError:
        return False
    return True


def build(src: Path, out: Path, ignores: list):
    if not src.is_dir():
        print(f"Error: Source is not a directory: {src}")
        sys.exit(EXIT_ERROR)

    cas_dir = out / 'cas'
    cas_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    file_count = 0
    total_bytes = 0

    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in ignores]
        for fname in files:
            if fname in ignores:
                continue

            fpath = Path(root) / fname
            rel = normalize_path(str(fpath.relative_to(src)))

            if not validate_path(rel, src):
                print(f"Error: Unsafe path detected: {rel}")
                sys.exit(EXIT_UNSAFE_PATH)

            fsize = fpath.stat().st_size
            if fsize > MAX_FILE_BYTES:
                print(f"Error: File exceeds max size ({MAX_FILE_BYTES}): {rel}")
                sys.exit(EXIT_BOUNDS_EXCEEDED)

            total_bytes += fsize
            if total_bytes > MAX_TOTAL_BYTES:
                print(f"Error: Total bytes exceed limit ({MAX_TOTAL_BYTES})")
                sys.exit(EXIT_BOUNDS_EXCEEDED)

            file_count += 1
            if file_count > MAX_FILES:
                print(f"Error: File count exceeds limit ({MAX_FILES})")
                sys.exit(EXIT_BOUNDS_EXCEEDED)

            content = fpath.read_bytes()
            h = sha256_bytes(content)

            # Store in CAS (sharded by first 2 chars)
            shard = cas_dir / h[:2]
            shard.mkdir(exist_ok=True)
            blob = shard / h
            if not blob.exists():
                blob.write_bytes(content)

            manifest[rel] = {'sha256': h, 'size': fsize}

    # Write manifest
    manifest_bytes = canonical_json(manifest)
    (out / 'manifest.json').write_bytes(manifest_bytes)

    # Write root hash
    root_hash = sha256_bytes(manifest_bytes)
    (out / 'root.sha256').write_text(root_hash)

    print(f"Build complete: {file_count} files, {total_bytes} bytes")
    print(f"Root hash: {root_hash}")
    sys.exit(EXIT_SUCCESS)


def reconstruct(pack: Path, dst: Path):
    manifest_path = pack / 'manifest.json'
    cas_dir = pack / 'cas'

    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {pack}")
        sys.exit(EXIT_ERROR)

    manifest = json.loads(manifest_path.read_bytes())

    dst.mkdir(parents=True, exist_ok=True)

    for rel, meta in manifest.items():
        h = meta['sha256']
        shard = cas_dir / h[:2]
        blob = shard / h

        if not blob.exists():
            print(f"Error: Missing blob {h} for {rel}")
            sys.exit(EXIT_ERROR)

        content = blob.read_bytes()
        if sha256_bytes(content) != h:
            print(f"Error: Blob corruption for {rel}")
            sys.exit(EXIT_VERIFY_MISMATCH)

        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)

    print(f"Reconstruct complete: {len(manifest)} files to {dst}")
    sys.exit(EXIT_SUCCESS)


def verify(src: Path, dst: Path):
    src_manifest = {}
    dst_manifest = {}

    for root, _, files in os.walk(src):
        for f in files:
            p = Path(root) / f
            rel = normalize_path(str(p.relative_to(src)))
            src_manifest[rel] = sha256_file(p)

    for root, _, files in os.walk(dst):
        for f in files:
            p = Path(root) / f
            rel = normalize_path(str(p.relative_to(dst)))
            dst_manifest[rel] = sha256_file(p)

    if src_manifest == dst_manifest:
        print(f"Verify SUCCESS: {len(src_manifest)} files match exactly.")
        sys.exit(EXIT_SUCCESS)
    else:
        print("Verify FAILED: Mismatch detected.")
        for k in set(src_manifest.keys()) | set(dst_manifest.keys()):
            s = src_manifest.get(k)
            d = dst_manifest.get(k)
            if s != d:
                print(f"  {k}: src={s} dst={d}")
        sys.exit(EXIT_VERIFY_MISMATCH)


def main():
    parser = argparse.ArgumentParser(description='F3 CAS Prototype')
    sub = parser.add_subparsers(dest='cmd')

    build_p = sub.add_parser('build')
    build_p.add_argument('--src', required=True, type=Path)
    build_p.add_argument('--out', required=True, type=Path)
    build_p.add_argument('--ignore', nargs='*', default=[])

    recon_p = sub.add_parser('reconstruct')
    recon_p.add_argument('--pack', required=True, type=Path)
    recon_p.add_argument('--dst', required=True, type=Path)

    verify_p = sub.add_parser('verify')
    verify_p.add_argument('--src', required=True, type=Path)
    verify_p.add_argument('--dst', required=True, type=Path)

    args = parser.parse_args()

    if args.cmd == 'build':
        build(args.src, args.out, args.ignore)
    elif args.cmd == 'reconstruct':
        reconstruct(args.pack, args.dst)
    elif args.cmd == 'verify':
        verify(args.src, args.dst)
    else:
        parser.print_help()
        sys.exit(EXIT_ERROR)



class CatalyticStore:
    """Wrapper class for CAS operations."""

    def __init__(self, objects_dir: Path):
        # objects_dir should point to the 'cas' subdirectory if using build() structure,
        # or the root of the CAS storage.
        self.objects_dir = Path(objects_dir)

    def object_path(self, hash_hex: str) -> Path:
        """Returns the deterministic path for a hash."""
        if not hash_hex or len(hash_hex) < 2:
             raise ValueError(f"Invalid hash: {hash_hex}")
        return self.objects_dir / hash_hex[:2] / hash_hex

    def put_bytes(self, data: bytes) -> str:
        """Writes bytes to CAS and returns the hash."""
        h = sha256_bytes(data)
        path = self.object_path(h)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # Use atomic write to avoid partial files
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return h

    def put_stream(self, stream) -> str:
        """Writes content of a stream to CAS and returns the hash."""
        h = hashlib.sha256()
        # Use a temporary file to store the stream until we have the full hash
        import tempfile
        fd, tmp_path = tempfile.mkstemp(dir=str(self.objects_dir) if self.objects_dir.exists() else None)
        try:
            with os.fdopen(fd, 'wb') as f:
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
            
            digest = h.hexdigest()
            target = self.object_path(digest)
            if target.exists():
                os.remove(tmp_path)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp_path, str(target))
            return digest
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get_bytes(self, hash_hex: str) -> bytes:
        """Reads bytes from CAS."""
        path = self.object_path(hash_hex)
        if not path.exists():
            raise FileNotFoundError(f"Hash not found in CAS: {hash_hex}")
        data = path.read_bytes()
        if sha256_bytes(data) != hash_hex:
            raise ValueError(f"CAS corruption detected for {hash_hex}")
        return data

    @staticmethod
    def build(src: Path, out: Path, ignores: list = None):
        return build(src, out, ignores or [])

    @staticmethod
    def reconstruct(pack: Path, dst: Path):
        return reconstruct(pack, dst)

    @staticmethod
    def verify(src: Path, dst: Path):
        return verify(src, dst)

if __name__ == '__main__':
    main()

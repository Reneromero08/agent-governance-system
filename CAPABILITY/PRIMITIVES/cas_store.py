from __future__ import annotations

import hashlib
import os
import errno
import re
import tempfile
from pathlib import Path
from typing import BinaryIO


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_relpath(path: str | Path) -> str:
    """
    Normalize a path into a repo-relative, POSIX-style relative path.

    Rules:
    - Converts Windows backslashes to '/'
    - Collapses '.' segments
    - Rejects absolute paths
    - Rejects traversal ('..' anywhere)
    - Returns normalized posix relative path
    """
    raw = str(path).replace("\\", "/")

    # Reject POSIX absolute and UNC-like roots.
    if raw.startswith("/") or raw.startswith("//"):
        raise ValueError(f"absolute paths are not allowed: {path!r}")

    # Reject Windows drive absolute/anchored paths (e.g. C:\ or C:/).
    if re.match(r"^[A-Za-z]:", raw):
        raise ValueError(f"absolute paths are not allowed: {path!r}")

    parts: list[str] = []
    for segment in raw.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            raise ValueError(f"path traversal is not allowed: {path!r}")
        parts.append(segment)

    return "/".join(parts) if parts else "."


class CatalyticStore:
    """
    Content Addressable Store (CAS) with deterministic on-disk layout.

    Objects are stored by SHA-256 hash (lowercase hex) under:
      <root_dir>/objects/<h[0:2]>/<h[2:4]>/<h>
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.objects_dir = self.root_dir / "objects"
        self.tmp_dir = self.root_dir / "tmp"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        final_path = self._object_path(digest)
        if final_path.exists():
            return digest

        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._temp_path(final_path.parent)
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            self._commit_temp(tmp_path, final_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return digest

    def get_bytes(self, hash_hex: str) -> bytes:
        hash_hex = self._validate_hash(hash_hex)
        path = self._object_path(hash_hex)
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path.read_bytes()

    def put_stream(self, stream: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        staging_path = self._temp_path(self.tmp_dir)
        hasher = hashlib.sha256()
        try:
            with open(staging_path, "wb") as staging:
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    staging.write(chunk)
                staging.flush()
                os.fsync(staging.fileno())

            digest = hasher.hexdigest()
            final_path = self._object_path(digest)
            if final_path.exists():
                return digest

            final_path.parent.mkdir(parents=True, exist_ok=True)

            self._commit_temp(staging_path, final_path)
            return digest
        finally:
            staging_path.unlink(missing_ok=True)

    def get_stream(self, hash_hex: str, out: BinaryIO, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        hash_hex = self._validate_hash(hash_hex)
        path = self._object_path(hash_hex)
        if not path.exists():
            raise FileNotFoundError(str(path))
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)

    def object_path(self, hash_hex: str) -> Path:
        """
        Return the deterministic on-disk path for a stored object.

        This is a read-only helper for bounded tooling (expand-by-hash).
        """
        hash_hex = self._validate_hash(hash_hex)
        return self.objects_dir / hash_hex[0:2] / hash_hex[2:4] / hash_hex

    def _object_path(self, digest: str) -> Path:
        digest = self._validate_hash(digest)
        return self.objects_dir / digest[0:2] / digest[2:4] / digest

    def _validate_hash(self, hash_hex: str) -> str:
        if not isinstance(hash_hex, str) or _HASH_RE.fullmatch(hash_hex) is None:
            raise ValueError(f"invalid hash: {hash_hex!r}")
        return hash_hex

    def _temp_path(self, dir_path: Path) -> Path:
        dir_path.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="cas_tmp_", dir=str(dir_path))
        os.close(fd)
        return Path(name)

    def _commit_temp(self, tmp_path: Path, final_path: Path) -> None:
        # Idempotent: never overwrite an existing object.
        if final_path.exists():
            return

        try:
            os.link(tmp_path, final_path)
        except FileExistsError:
            return
        except OSError as e:
            # Fallback path for platforms/filesystems that don't support hardlinks.
            if e.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP):
                raise
            fd = os.open(final_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with open(fd, "wb", closefd=True) as dst, open(tmp_path, "rb") as src:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                    dst.flush()
                    os.fsync(dst.fileno())
            except Exception:
                try:
                    os.unlink(final_path)
                finally:
                    raise
            return
        else:
            tmp_path.unlink(missing_ok=True)

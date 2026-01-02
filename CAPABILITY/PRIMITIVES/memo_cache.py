from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from CAPABILITY.PRIMITIVES.cas_store import normalize_relpath
from CAPABILITY.PRIMITIVES.restore_proof import canonical_json_bytes


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_job_cache_key(
    *,
    jobspec: Dict[str, Any],
    input_domain_roots: Dict[str, str],
    validator_semver: str,
    validator_build_id: str,
    strict: bool,
) -> str:
    """
    Compute deterministic job cache key.

    Key inputs (in deterministic order):
    1) JobSpec canonical JSON bytes
    2) Input domain roots canonical JSON bytes
    3) Toolchain identity canonical JSON bytes (validator_semver, validator_build_id)
    4) Strictness byte (b"1" or b"0")

    Hash: SHA-256 over the concatenation of the above components with NUL delimiters.
    """
    jobspec_bytes = canonical_json_bytes(jobspec)
    roots_bytes = canonical_json_bytes(dict(sorted(input_domain_roots.items(), key=lambda kv: kv[0])))
    toolchain_bytes = canonical_json_bytes(
        {"validator_semver": validator_semver, "validator_build_id": validator_build_id}
    )
    strict_byte = b"1" if strict else b"0"
    preimage = b"\0".join([jobspec_bytes, roots_bytes, toolchain_bytes, strict_byte])
    return _sha256_hex(preimage)


@dataclass(frozen=True)
class CacheHit:
    key: str
    cache_dir: Path


class JobMemoCache:
    """
    Deterministic, content-addressed cache for completed jobs.

    Layout:
      <root_dir>/<job_cache_key>/
        - JOBSPEC.json
        - INPUT_DOMAIN_ROOTS.json
        - OUTPUT_HASHES.json
        - DOMAIN_ROOTS.json
        - PROOF.json
        - VALIDATOR_ID.json
        - metadata.json
        - OUTPUTS/<relpath>  (materialized durable output bytes; files only)
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def cache_dir(self, key: str) -> Path:
        return self.root_dir / key

    def try_hit(
        self,
        *,
        key: str,
        require_files: Optional[Iterable[str]] = None,
    ) -> Optional[CacheHit]:
        cache_dir = self.cache_dir(key)
        if not cache_dir.exists():
            return None

        required = list(require_files) if require_files is not None else [
            "JOBSPEC.json",
            "INPUT_DOMAIN_ROOTS.json",
            "OUTPUT_HASHES.json",
            "DOMAIN_ROOTS.json",
            "PROOF.json",
            "VALIDATOR_ID.json",
            "metadata.json",
        ]
        for name in required:
            if not (cache_dir / name).exists():
                return None
        return CacheHit(key=key, cache_dir=cache_dir)

    def populate(
        self,
        *,
        key: str,
        run_dir: Path,
        durable_outputs: Dict[str, Path],
        input_domain_roots: Dict[str, str],
        jobspec: Dict[str, Any],
        validator_id: Dict[str, Any],
    ) -> Path:
        cache_dir = self.cache_dir(key)
        cache_dir.mkdir(parents=True, exist_ok=True)

        (cache_dir / "JOBSPEC.json").write_bytes(canonical_json_bytes(jobspec))
        (cache_dir / "INPUT_DOMAIN_ROOTS.json").write_bytes(
            canonical_json_bytes(dict(sorted(input_domain_roots.items(), key=lambda kv: kv[0])))
        )

        for name in ["OUTPUT_HASHES.json", "DOMAIN_ROOTS.json", "PROOF.json", "VALIDATOR_ID.json"]:
            src = run_dir / name
            if not src.exists():
                raise FileNotFoundError(f"missing run artifact required for cache: {src}")
            (cache_dir / name).write_bytes(src.read_bytes())

        outputs_dir = cache_dir / "OUTPUTS"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        for rel, abs_path in sorted(durable_outputs.items(), key=lambda kv: kv[0]):
            rel_norm = normalize_relpath(rel)
            dest = outputs_dir / rel_norm
            dest.parent.mkdir(parents=True, exist_ok=True)
            if abs_path.is_dir():
                # Deterministic directory materialization: copy files by normalized relative path.
                items: list[tuple[str, Path]] = []
                for file_path in abs_path.rglob("*"):
                    if file_path.is_file():
                        rel_file = normalize_relpath(file_path.relative_to(abs_path))
                        items.append((rel_file, file_path))
                for rel_file, file_path in sorted(items, key=lambda t: t[0]):
                    out_path = (outputs_dir / rel_norm / rel_file)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(file_path.read_bytes())
                continue

            dest.write_bytes(abs_path.read_bytes())

        metadata = {
            "job_cache_key": key,
            "jobspec_sha256": _sha256_hex(canonical_json_bytes(jobspec)),
            "input_domain_roots": dict(sorted(input_domain_roots.items(), key=lambda kv: kv[0])),
            "validator_id": {
                "validator_semver": validator_id.get("validator_semver", ""),
                "validator_build_id": validator_id.get("validator_build_id", ""),
            },
        }
        (cache_dir / "metadata.json").write_bytes(canonical_json_bytes(metadata))

        return cache_dir

    def restore_outputs(
        self,
        *,
        hit: CacheHit,
        durable_outputs: Dict[str, Path],
    ) -> None:
        outputs_dir = hit.cache_dir / "OUTPUTS"
        if not outputs_dir.exists():
            raise FileNotFoundError(f"cache missing OUTPUTS/: {outputs_dir}")

        for rel, abs_path in durable_outputs.items():
            rel_norm = normalize_relpath(rel)
            cached_path = outputs_dir / rel_norm
            if not cached_path.exists():
                raise FileNotFoundError(f"cache missing output bytes: {cached_path}")

            abs_path.parent.mkdir(parents=True, exist_ok=True)
            if cached_path.is_dir():
                abs_path.mkdir(parents=True, exist_ok=True)
                items: list[tuple[str, Path]] = []
                for file_path in cached_path.rglob("*"):
                    if file_path.is_file():
                        rel_file = normalize_relpath(file_path.relative_to(cached_path))
                        items.append((rel_file, file_path))
                for rel_file, file_path in sorted(items, key=lambda t: t[0]):
                    out_file = abs_path / rel_file
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_bytes(file_path.read_bytes())
                continue

            abs_path.write_bytes(cached_path.read_bytes())

    def restore_artifacts(self, *, hit: CacheHit, run_dir: Path) -> None:
        for name in ["OUTPUT_HASHES.json", "DOMAIN_ROOTS.json", "PROOF.json", "VALIDATOR_ID.json"]:
            (run_dir / name).write_bytes((hit.cache_dir / name).read_bytes())

#!/usr/bin/env python3
"""
Phase 1.5B: Deterministic Repo Digest, Purity Scan, and Restore Proof

Implements deterministic repo-state proofs that make catalysis measurable:
- Pre/post repo digest (tree hash with declared exclusions)
- RESTORE_PROOF receipt (PASS/FAIL with diff summary)
- PURITY_SCAN receipt (no new/modified files outside durable roots; tmp roots empty)

HARD INVARIANTS:
- Never mutate original user content as part of the scan
- Fail closed: if digest or scan cannot be computed deterministically, emit error receipt and exit nonzero
- Canonical ordering everywhere (paths, lists, diffs)
- No crypto sealing here (handled by CRYPTO_SAFE phase)

ALLOWED WRITES:
- Only to implement digest/scan/proofs + tests + docs
- Receipts output path must follow existing repo conventions
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Module version for deterministic tracking
MODULE_VERSION = "1.5b.0"


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON serialization with sorted keys and no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def normalize_path(path: Path, repo_root: Path) -> str:
    """
    Normalize path to repo-relative forward-slash format.

    Args:
        path: Absolute or relative path
        repo_root: Repository root path

    Returns:
        Normalized relative path with forward slashes
    """
    if path.is_absolute():
        rel = path.relative_to(repo_root)
    else:
        rel = path
    return str(rel).replace("\\", "/")


@dataclass
class DigestSpec:
    """Specification for digest computation."""
    repo_root: Path
    exclusions: List[str]  # Repo-relative paths to exclude (normalized)
    durable_roots: List[str]  # Paths to durable output roots (excluded from digest)
    tmp_roots: List[str]  # Paths to temporary roots (excluded from digest, verified empty for purity)


class RepoDigest:
    """Deterministic repository digest computation."""

    def __init__(self, spec: DigestSpec):
        """
        Initialize with digest specification.

        Args:
            spec: DigestSpec with repo root, exclusions, durable roots, tmp roots
        """
        self.spec = spec
        self.module_version_hash = hashlib.sha256(MODULE_VERSION.encode("utf-8")).hexdigest()

        # Compute exclusions spec hash (deterministic)
        exclusions_canonical = sorted(set(spec.exclusions + spec.durable_roots + spec.tmp_roots))
        self.exclusions_spec_hash = hashlib.sha256(
            canonical_json_bytes(exclusions_canonical)
        ).hexdigest()

    def compute_digest(self) -> Dict[str, Any]:
        """
        Compute deterministic digest of repository state.

        Returns:
            Digest receipt with digest, file_count, exclusions_spec_hash, module_version_hash, file_manifest

        Raises:
            ValueError: If digest cannot be computed deterministically
        """
        try:
            # Enumerate all files in repo
            file_records = self._enumerate_files()

            # Compute tree digest from canonical records
            tree_digest = self._compute_tree_digest(file_records)

            # Build file manifest {path: hash} for diff computation
            file_manifest = {r["path"]: r["hash"] for r in file_records}

            return {
                "digest": tree_digest,
                "file_count": len(file_records),
                "file_manifest": file_manifest,
                "exclusions_spec_hash": self.exclusions_spec_hash,
                "module_version_hash": self.module_version_hash,
                "module_version": MODULE_VERSION,
            }
        except Exception as e:
            raise ValueError(f"DIGEST_COMPUTATION_FAILED: {e}")

    def _should_exclude(self, path: Path) -> bool:
        """
        Check if path should be excluded from digest.

        Args:
            path: Absolute path to check

        Returns:
            True if path should be excluded
        """
        try:
            norm_path = normalize_path(path, self.spec.repo_root)
        except ValueError:
            # Path is not under repo root
            return True

        # Check if path is under any exclusion, durable root, or tmp root
        all_exclusions = set(self.spec.exclusions + self.spec.durable_roots + self.spec.tmp_roots)

        for exclusion in all_exclusions:
            # Normalize exclusion for comparison
            if norm_path == exclusion or norm_path.startswith(exclusion + "/"):
                return True

        return False

    def _enumerate_files(self) -> List[Dict[str, str]]:
        """
        Enumerate all files in repo with hashes, excluding specified paths.

        Returns:
            List of {path: normalized_relpath, hash: sha256} in canonical order
        """
        file_records = []

        for root, dirs, files in os.walk(self.spec.repo_root):
            root_path = Path(root)

            # Filter out excluded directories (modify dirs in-place to prune walk)
            dirs_to_remove = []
            for d in dirs:
                dir_path = root_path / d
                if self._should_exclude(dir_path):
                    dirs_to_remove.append(d)
            for d in dirs_to_remove:
                dirs.remove(d)

            # Process files
            for f in files:
                file_path = root_path / f

                if self._should_exclude(file_path):
                    continue

                # Hash file bytes
                try:
                    with open(file_path, "rb") as fh:
                        file_hash = hashlib.sha256(fh.read()).hexdigest()
                except Exception as e:
                    raise ValueError(f"HASH_FAILED: {file_path}: {e}")

                norm_path = normalize_path(file_path, self.spec.repo_root)
                file_records.append({"path": norm_path, "hash": file_hash})

        # Sort by normalized path for canonical ordering
        file_records.sort(key=lambda r: r["path"])

        return file_records

    def _compute_tree_digest(self, file_records: List[Dict[str, str]]) -> str:
        """
        Compute tree digest from canonical file records.

        Args:
            file_records: Sorted list of {path, hash} records

        Returns:
            SHA-256 digest of canonical records
        """
        # Create canonical representation: one line per file "path:hash\n"
        lines = []
        for record in file_records:
            lines.append(f"{record['path']}:{record['hash']}\n")

        canonical_text = "".join(lines)
        return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


class PurityScan:
    """Scan for purity violations (files outside durable roots, tmp residue)."""

    def __init__(self, spec: DigestSpec):
        """
        Initialize with digest specification.

        Args:
            spec: DigestSpec with repo root, durable roots, tmp roots
        """
        self.spec = spec
        self.module_version_hash = hashlib.sha256(MODULE_VERSION.encode("utf-8")).hexdigest()

    def scan(self, pre_digest_receipt: Dict[str, Any], post_digest_receipt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform purity scan comparing pre and post digests.

        Args:
            pre_digest_receipt: PRE_DIGEST receipt
            post_digest_receipt: POST_DIGEST receipt

        Returns:
            PURITY_SCAN receipt with verdict, violations, tmp_residue
        """
        try:
            violations = []
            tmp_residue = []

            # Check tmp roots are empty
            for tmp_root in self.spec.tmp_roots:
                tmp_path = self.spec.repo_root / tmp_root
                if tmp_path.exists():
                    # Enumerate files in tmp root
                    for root, dirs, files in os.walk(tmp_path):
                        root_path = Path(root)
                        for f in files:
                            file_path = root_path / f
                            norm_path = normalize_path(file_path, self.spec.repo_root)
                            tmp_residue.append(norm_path)

            # Sort tmp_residue for canonical ordering
            tmp_residue.sort()

            # Determine verdict
            if tmp_residue:
                verdict = "FAIL"
            elif pre_digest_receipt["digest"] == post_digest_receipt["digest"]:
                verdict = "PASS"
            else:
                # Digests differ: files changed outside durable roots
                verdict = "FAIL"
                # Note: Detailed diff is in RESTORE_PROOF, not here

            return {
                "verdict": verdict,
                "violations": violations,  # Reserved for future use
                "tmp_residue": tmp_residue,
                "scan_module_version_hash": self.module_version_hash,
                "module_version": MODULE_VERSION,
            }
        except Exception as e:
            raise ValueError(f"PURITY_SCAN_FAILED: {e}")


class RestoreProof:
    """Generate restore proof binding pre/post digests with verdict."""

    def __init__(self, spec: DigestSpec):
        """
        Initialize with digest specification.

        Args:
            spec: DigestSpec with repo root, durable roots, tmp roots, exclusions
        """
        self.spec = spec
        self.module_version_hash = hashlib.sha256(MODULE_VERSION.encode("utf-8")).hexdigest()

    def generate_proof(
        self,
        pre_digest_receipt: Dict[str, Any],
        post_digest_receipt: Dict[str, Any],
        purity_scan_receipt: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate RESTORE_PROOF binding pre/post digests with verdict.

        Args:
            pre_digest_receipt: PRE_DIGEST receipt
            post_digest_receipt: POST_DIGEST receipt
            purity_scan_receipt: PURITY_SCAN receipt

        Returns:
            RESTORE_PROOF receipt with verdict, diff summary
        """
        try:
            # Determine verdict
            if pre_digest_receipt["digest"] == post_digest_receipt["digest"] and purity_scan_receipt["verdict"] == "PASS":
                verdict = "PASS"
                diff_summary = None
            else:
                verdict = "FAIL"
                diff_summary = self._compute_diff_summary(pre_digest_receipt, post_digest_receipt)

            proof = {
                "verdict": verdict,
                "pre_digest": pre_digest_receipt["digest"],
                "post_digest": post_digest_receipt["digest"],
                "tmp_roots": sorted(self.spec.tmp_roots),
                "durable_roots": sorted(self.spec.durable_roots),
                "exclusions": sorted(self.spec.exclusions),
                "exclusions_spec_hash": pre_digest_receipt["exclusions_spec_hash"],
                "proof_module_version_hash": self.module_version_hash,
                "module_version": MODULE_VERSION,
            }

            if diff_summary is not None:
                proof["diff_summary"] = diff_summary

            return proof
        except Exception as e:
            raise ValueError(f"RESTORE_PROOF_GENERATION_FAILED: {e}")

    def _compute_diff_summary(
        self,
        pre_digest_receipt: Dict[str, Any],
        post_digest_receipt: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """
        Compute deterministic diff summary (added, removed, changed paths only).

        Args:
            pre_digest_receipt: PRE_DIGEST receipt (must contain file_manifest)
            post_digest_receipt: POST_DIGEST receipt (must contain file_manifest)

        Returns:
            {added: [paths], removed: [paths], changed: [paths]} in canonical order
        """
        # Extract file manifests from receipts
        pre_files = pre_digest_receipt.get("file_manifest", {})
        post_files = post_digest_receipt.get("file_manifest", {})

        added = sorted(set(post_files.keys()) - set(pre_files.keys()))
        removed = sorted(set(pre_files.keys()) - set(post_files.keys()))
        changed = sorted([
            path for path in set(pre_files.keys()) & set(post_files.keys())
            if pre_files[path] != post_files[path]
        ])

        return {
            "added": added,
            "removed": removed,
            "changed": changed,
        }


def write_receipt(path: Path, receipt: Dict[str, Any]) -> None:
    """
    Write receipt to path as canonical JSON.

    Args:
        path: Output path for receipt
        receipt: Receipt dictionary
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_bytes(canonical_json_bytes(receipt))
    tmp_path.replace(path)


def main() -> int:
    """
    CLI entry point for digest/scan/proof generation.

    Usage:
        python repo_digest.py --repo-root <path> --pre-digest <out>
        python repo_digest.py --repo-root <path> --post-digest <out>
        python repo_digest.py --repo-root <path> --purity-scan <pre> <post> <out>
        python repo_digest.py --repo-root <path> --restore-proof <pre> <post> <purity> <out>
    """
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1.5B: Deterministic Repo Digest & Restore Proof")
    parser.add_argument("--repo-root", required=True, help="Repository root path")
    parser.add_argument("--pre-digest", metavar="OUT", help="Generate PRE_DIGEST receipt")
    parser.add_argument("--post-digest", metavar="OUT", help="Generate POST_DIGEST receipt")
    parser.add_argument("--purity-scan", nargs=3, metavar=("PRE", "POST", "OUT"), help="Generate PURITY_SCAN receipt")
    parser.add_argument("--restore-proof", nargs=4, metavar=("PRE", "POST", "PURITY", "OUT"), help="Generate RESTORE_PROOF receipt")
    parser.add_argument("--exclusions", default="", help="Comma-separated exclusion paths")
    parser.add_argument("--durable-roots", default="", help="Comma-separated durable root paths")
    parser.add_argument("--tmp-roots", default="", help="Comma-separated tmp root paths")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        print(f"ERROR: repo_root not a directory: {repo_root}", file=sys.stderr)
        return 2

    # Parse comma-separated lists
    exclusions = [x.strip() for x in args.exclusions.split(",") if x.strip()]
    durable_roots = [x.strip() for x in args.durable_roots.split(",") if x.strip()]
    tmp_roots = [x.strip() for x in args.tmp_roots.split(",") if x.strip()]

    spec = DigestSpec(
        repo_root=repo_root,
        exclusions=exclusions,
        durable_roots=durable_roots,
        tmp_roots=tmp_roots,
    )

    try:
        if args.pre_digest:
            digest = RepoDigest(spec)
            receipt = digest.compute_digest()
            write_receipt(Path(args.pre_digest), receipt)
            print(f"OK: wrote PRE_DIGEST to {args.pre_digest}")
            return 0

        if args.post_digest:
            digest = RepoDigest(spec)
            receipt = digest.compute_digest()
            write_receipt(Path(args.post_digest), receipt)
            print(f"OK: wrote POST_DIGEST to {args.post_digest}")
            return 0

        if args.purity_scan:
            pre_path, post_path, out_path = args.purity_scan
            pre_receipt = json.loads(Path(pre_path).read_text())
            post_receipt = json.loads(Path(post_path).read_text())

            scanner = PurityScan(spec)
            receipt = scanner.scan(pre_receipt, post_receipt)
            write_receipt(Path(out_path), receipt)
            print(f"OK: wrote PURITY_SCAN to {out_path}")
            return 0

        if args.restore_proof:
            pre_path, post_path, purity_path, out_path = args.restore_proof
            pre_receipt = json.loads(Path(pre_path).read_text())
            post_receipt = json.loads(Path(post_path).read_text())
            purity_receipt = json.loads(Path(purity_path).read_text())

            prover = RestoreProof(spec)
            receipt = prover.generate_proof(pre_receipt, post_receipt, purity_receipt)
            write_receipt(Path(out_path), receipt)
            print(f"OK: wrote RESTORE_PROOF to {out_path}")

            # Exit nonzero if proof failed
            if receipt["verdict"] == "FAIL":
                print(f"FAIL: restore proof verdict=FAIL", file=sys.stderr)
                return 1

            return 0

        parser.print_help()
        return 2

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())

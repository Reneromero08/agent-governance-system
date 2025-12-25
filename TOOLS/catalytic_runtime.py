#!/usr/bin/env python3

"""
Catalytic Runtime: Enforce catalytic space constraints during execution.

Implements CMP-01 (Catalytic Mutation Protocol) by:
1. Taking pre-snapshots of catalytic domains
2. Executing a wrapped subprocess/function (or multi-step pipeline)
3. Verifying restoration of catalytic domains
4. Recording run ledger with proof

Usage (single command):
  python catalytic_runtime.py \
    --run-id cortex-build-2025-12-23 \
    --catalytic-domains CORTEX/_generated/_tmp \
    --durable-outputs CORTEX/_generated/cortex.json \
    --intent "Build cortex index" \
    -- python CORTEX/cortex.build.py

Usage (multi-step pipeline):
  python catalytic_runtime.py \
    --run-id pipeline-2025-12-23 \
    --catalytic-domains CORTEX/_generated/_tmp \
    --durable-outputs CORTEX/_generated/cortex.json \
    --step "Build scratch index::python build_index.py" \
    --step "Transform data::python transform.py" \
    --step "Validate output::python validate.py"

The run ledger is stored in CONTRACTS/_runs/<run_id>/
"""

import json
import sys
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add CATALYTIC-DPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "CATALYTIC-DPT"))
from PRIMITIVES.restore_proof import RestorationProofValidator
from PRIMITIVES.restore_proof import canonical_json_bytes
from PRIMITIVES.preflight import PreflightValidator
from PRIMITIVES.fs_guard import FilesystemGuard
from PRIMITIVES.cas_store import CatalyticStore, normalize_relpath
from PRIMITIVES.merkle import build_manifest_root
from PRIMITIVES.ledger import Ledger

PROJECT_ROOT = Path(__file__).parent.parent
DETERMINISTIC_TIMESTAMP_SENTINEL = "CATALYTIC-DPT-02_CONFIG"


class CatalyticSnapshot:
    """Snapshot the state of a filesystem domain."""

    def __init__(self, domain_path: Path):
        self.domain_path = domain_path
        self.files: Dict[str, str] = {}  # path -> sha256

    def capture(self, *, cas: Optional[CatalyticStore] = None) -> None:
        """Recursively snapshot all files in domain."""
        if not self.domain_path.exists():
            return

        # rglob ordering is not specified; enforce deterministic ordering by normalized relative path.
        items = []
        for file_path in self.domain_path.rglob("*"):
            if file_path.is_file():
                rel_path = normalize_relpath(file_path.relative_to(self.domain_path))
                items.append((rel_path, file_path))

        for rel_path, file_path in sorted(items, key=lambda t: t[0]):
            try:
                if cas is None:
                    sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
                else:
                    with open(file_path, "rb") as f:
                        sha = cas.put_stream(f)
                self.files[rel_path] = sha
            except Exception as e:
                print(f"[catalytic] Warning: Could not snapshot {file_path}: {e}", file=sys.stderr)

    def to_dict(self) -> Dict[str, str]:
        """Export snapshot as dict."""
        return self.files.copy()

    @staticmethod
    def from_dict(data: Dict[str, str]) -> "CatalyticSnapshot":
        """Import snapshot from dict."""
        snapshot = CatalyticSnapshot(Path("."))
        snapshot.files = data
        return snapshot

    def diff(self, other: "CatalyticSnapshot") -> Dict[str, any]:
        """Compare two snapshots. Return empty dict if identical."""
        added = {k: v for k, v in other.files.items() if k not in self.files}
        removed = {k: v for k, v in self.files.items() if k not in other.files}
        changed = {
            k: {"before": self.files[k], "after": other.files[k]}
            for k in self.files.keys() & other.files.keys()
            if self.files[k] != other.files[k]
        }

        return {"added": added, "removed": removed, "changed": changed}


class CatalyticRuntime:
    """Manage a catalytic run with pre/post snapshots and restoration proof."""

    def __init__(
        self,
        run_id: str,
        catalytic_domains: List[str],
        durable_outputs: List[str],
        intent: str,
        determinism: str = "deterministic",
        timestamp: Optional[str] = None,
    ):
        self.run_id = run_id
        self.catalytic_domains = [PROJECT_ROOT / d for d in catalytic_domains]
        self.durable_outputs = [PROJECT_ROOT / d for d in durable_outputs]
        self.intent = intent
        self.determinism = determinism
        # Determinism: runtime does not generate timestamps. Caller supplies deterministic timestamp.
        self.timestamp = timestamp or DETERMINISTIC_TIMESTAMP_SENTINEL

        # Run ledger directory
        self.ledger_dir = PROJECT_ROOT / "CONTRACTS" / "_runs" / run_id
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.cas = CatalyticStore(self.ledger_dir / "CAS")
        self.ledger = Ledger(self.ledger_dir / "LEDGER.jsonl")

        # Snapshots
        self.pre_snapshots: Dict[str, CatalyticSnapshot] = {}
        self.post_snapshots: Dict[str, CatalyticSnapshot] = {}

        # Initialize filesystem guard (Layer 2 runtime enforcement)
        self.fs_guard = FilesystemGuard(
            allowed_roots=[
                "CONTRACTS/_runs",
                "CORTEX/_generated",
                "MEMORY/LLM_PACKER/_packs",
                "CATALYTIC-DPT/_scratch",
            ],
            forbidden_paths=["CANON", "AGENTS.md", "BUILD", ".git"],
            project_root=PROJECT_ROOT,
        )

    def build_jobspec(self) -> Dict:
        """Build JobSpec dict from runtime parameters."""
        return {
            "job_id": self.run_id,
            "phase": 0,
            "task_type": "primitive_implementation",
            "intent": self.intent,
            "inputs": {},
            "outputs": {
                "durable_paths": [str(p.relative_to(PROJECT_ROOT)) for p in self.durable_outputs],
                "validation_criteria": {},
            },
            "catalytic_domains": [str(d.relative_to(PROJECT_ROOT)) for d in self.catalytic_domains],
            "determinism": self.determinism,
        }

    def validate_config(self) -> bool:
        """
        Pre-flight: validate that domains and outputs are legal.

        DEPRECATED: This is legacy validation. Use preflight_validate() instead.
        Kept for backwards compatibility.
        """
        allowed_roots = [
            PROJECT_ROOT / "CONTRACTS" / "_runs",
            PROJECT_ROOT / "CORTEX" / "_generated",
            PROJECT_ROOT / "MEMORY" / "LLM_PACKER" / "_packs",
        ]

        forbidden_paths = [
            PROJECT_ROOT / "CANON",
            PROJECT_ROOT / "AGENTS.md",
            PROJECT_ROOT / "BUILD",
        ]

        # Check catalytic domains don't overlap forbidden (bidirectional)
        for domain in self.catalytic_domains:
            # Check if domain is inside forbidden
            for forbidden in forbidden_paths:
                try:
                    domain.relative_to(forbidden)
                    print(
                        f"[catalytic] ERROR: Catalytic domain {domain} is inside forbidden path {forbidden}",
                        file=sys.stderr,
                    )
                    return False
                except ValueError:
                    pass

            # Check if forbidden is inside domain (the dangerous case)
            for forbidden in forbidden_paths:
                try:
                    forbidden.relative_to(domain)
                    print(
                        f"[catalytic] ERROR: Forbidden path {forbidden} would be inside catalytic domain {domain}",
                        file=sys.stderr,
                    )
                    return False
                except ValueError:
                    pass

        # Check durable outputs are under allowed roots
        for output in self.durable_outputs:
            is_allowed = False
            for root in allowed_roots:
                try:
                    output.relative_to(root)
                    is_allowed = True
                    break
                except ValueError:
                    pass

            if not is_allowed:
                print(
                    f"[catalytic] ERROR: Durable output {output} not under allowed roots",
                    file=sys.stderr,
                )
                return False

        return True

    def preflight_validate(self) -> Tuple[bool, List[Dict]]:
        """
        Preflight validation: fail closed if JobSpec violates contract.

        Returns:
            Tuple of (valid, errors)
        """
        # Load jobspec schema
        jobspec_schema_path = PROJECT_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "jobspec.schema.json"

        # Initialize preflight validator
        preflight = PreflightValidator(jobspec_schema_path)

        # Build JobSpec from runtime parameters
        jobspec = self.build_jobspec()

        # Validate
        valid, errors = preflight.validate(jobspec, PROJECT_ROOT)

        return valid, errors

    def snapshot_domains(self) -> None:
        """Take pre-snapshots of all catalytic domains."""
        for domain in self.catalytic_domains:
            snapshot = CatalyticSnapshot(domain)
            snapshot.capture(cas=self.cas)
            domain_key = str(domain.relative_to(PROJECT_ROOT))
            self.pre_snapshots[domain_key] = snapshot
            print(f"[catalytic] Snapshots pre: {domain_key} ({len(snapshot.files)} files)")

    def execute(self, cmd: List[str]) -> int:
        """Execute a command. Return exit code."""
        try:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            return result.returncode
        except Exception as e:
            print(f"[catalytic] ERROR executing command: {e}", file=sys.stderr)
            return 1

    def snapshot_after(self) -> None:
        """Take post-snapshots after execution."""
        for domain in self.catalytic_domains:
            snapshot = CatalyticSnapshot(domain)
            snapshot.capture(cas=self.cas)
            domain_key = str(domain.relative_to(PROJECT_ROOT))
            self.post_snapshots[domain_key] = snapshot
            print(f"[catalytic] Snapshots post: {domain_key} ({len(snapshot.files)} files)")

    def verify_restoration(self) -> Tuple[bool, Dict]:
        """Verify catalytic domains were restored. Return (success, diff_report)."""
        all_diffs = {}
        all_restored = True

        for domain_key, pre_snapshot in self.pre_snapshots.items():
            post_snapshot = self.post_snapshots.get(domain_key, CatalyticSnapshot(Path(".")))
            diff = pre_snapshot.diff(post_snapshot)

            if diff["added"] or diff["removed"] or diff["changed"]:
                all_restored = False

            all_diffs[domain_key] = diff

        return all_restored, all_diffs

    def collect_outputs(self) -> List[Dict]:
        """List durable outputs that exist."""
        outputs = []
        for output_path in self.durable_outputs:
            if output_path.exists():
                if output_path.is_file():
                    sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
                    outputs.append(
                        {
                            "path": str(output_path.relative_to(PROJECT_ROOT)),
                            "type": "file",
                            "sha256": sha,
                        }
                    )
                elif output_path.is_dir():
                    outputs.append(
                        {
                            "path": str(output_path.relative_to(PROJECT_ROOT)),
                            "type": "directory",
                        }
                    )
        return outputs

    def write_canonical_artifacts(
        self, exit_code: int, restoration_success: bool, diffs: Dict, status_state: str
    ) -> bool:
        """
        Write the complete canonical artifact set.

        Canonical artifact set (Phase 0.2):
        - JOBSPEC.json
        - STATUS.json (state machine)
        - INPUT_HASHES.json
        - OUTPUT_HASHES.json
        - DOMAIN_ROOTS.json
        - LEDGER.jsonl
        - VALIDATOR_ID.json
        - PROOF.json (written last)

        Returns True if all artifacts written successfully, False otherwise.
        """
        try:
            # 1. JOBSPEC.json (stub for now - would normally be passed in)
            jobspec = {
                "job_id": self.run_id,
                "phase": 0,
                "task_type": "primitive_implementation",
                "intent": self.intent,
                "inputs": {},
                "outputs": {
                    "durable_paths": [str(d.relative_to(PROJECT_ROOT)) for d in self.durable_outputs],
                    "validation_criteria": {"restoration_verified": True},
                },
                "catalytic_domains": [str(d.relative_to(PROJECT_ROOT)) for d in self.catalytic_domains],
                "determinism": self.determinism,
            }
            self.fs_guard.guarded_write_text(self.ledger_dir / "JOBSPEC.json", json.dumps(jobspec, indent=2))

            # 2. STATUS.json (state machine: started/failed/succeeded/verified)
            status = {
                "status": status_state,
                "restoration_verified": restoration_success,
                "exit_code": exit_code,
                "validation_passed": restoration_success and exit_code == 0,
            }
            self.fs_guard.guarded_write_text(self.ledger_dir / "STATUS.json", json.dumps(status, indent=2))

            # 3. INPUT_HASHES.json (from pre-snapshots)
            input_hashes = {}
            for domain, snapshot in self.pre_snapshots.items():
                for path, sha in snapshot.to_dict().items():
                    full_path = f"{domain}/{path}"
                    input_hashes[full_path] = sha
            self.fs_guard.guarded_write_text(self.ledger_dir / "INPUT_HASHES.json", json.dumps(input_hashes, indent=2, sort_keys=True))

            # 4. OUTPUT_HASHES.json (from outputs list)
            outputs_list = self.collect_outputs()
            output_hashes = {o["path"]: o.get("sha256", "") for o in outputs_list}
            self.fs_guard.guarded_write_text(self.ledger_dir / "OUTPUT_HASHES.json", json.dumps(output_hashes, indent=2, sort_keys=True))

            # 5. DOMAIN_ROOTS.json (Merkle roots per domain)
            pre_manifest = {domain: snapshot.to_dict() for domain, snapshot in self.pre_snapshots.items()}
            post_manifest = {domain: snapshot.to_dict() for domain, snapshot in self.post_snapshots.items()}
            domain_roots = {}
            for domain, files in sorted(post_manifest.items(), key=lambda kv: kv[0]):
                if files:
                    root_hash = build_manifest_root(files)
                else:
                    root_hash = hashlib.sha256(b"").hexdigest()
                domain_roots[domain] = root_hash
            self.fs_guard.guarded_write_text(self.ledger_dir / "DOMAIN_ROOTS.json", canonical_json_bytes(domain_roots).decode("utf-8"))

            # 6. LEDGER.jsonl (append-only receipts, schema-valid)
            def _restore_diff(pre_m: dict, post_m: dict) -> dict:
                out = {}
                for domain in sorted(set(pre_m.keys()) | set(post_m.keys())):
                    pre_files = pre_m.get(domain, {})
                    post_files = post_m.get(domain, {})
                    added = {p: post_files[p] for p in sorted(set(post_files) - set(pre_files))}
                    removed = {p: pre_files[p] for p in sorted(set(pre_files) - set(post_files))}
                    changed = {p: post_files[p] for p in sorted(set(pre_files) & set(post_files)) if pre_files[p] != post_files[p]}
                    out[domain] = {"added": added, "removed": removed, "changed": changed}
                return out

            ledger_record = {
                "JOBSPEC": jobspec,
                "RUN_INFO": {
                    "run_id": self.run_id,
                    "timestamp": self.timestamp,
                    "intent": self.intent,
                    "catalytic_domains": [str(d.relative_to(PROJECT_ROOT)).replace("\\", "/") for d in self.catalytic_domains],
                    "exit_code": exit_code,
                    "restoration_verified": restoration_success,
                },
                "PRE_MANIFEST": pre_manifest,
                "POST_MANIFEST": post_manifest,
                "RESTORE_DIFF": _restore_diff(pre_manifest, post_manifest),
                "OUTPUTS": outputs_list,
                "STATUS": status,
                "VALIDATOR_ID": {
                    "validator_semver": "0.1.0",
                    "validator_build_id": "phase0-canonical",
                },
            }
            self.ledger.append(ledger_record)

            # 7. VALIDATOR_ID.json
            validator_id = {
                "validator_semver": "0.1.0",
                "validator_build_id": "phase0-canonical",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            }
            self.fs_guard.guarded_write_text(self.ledger_dir / "VALIDATOR_ID.json", json.dumps(validator_id, indent=2))

            # 8. PROOF.json (written LAST after all other artifacts exist)
            proof_schema_path = PROJECT_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "proof.schema.json"
            validator = RestorationProofValidator(proof_schema_path)

            jobspec_hash = hashlib.sha256((self.ledger_dir / "JOBSPEC.json").read_bytes()).hexdigest()
            ledger_hash = hashlib.sha256((self.ledger_dir / "LEDGER.jsonl").read_bytes()).hexdigest()
            proof = validator.generate_proof(
                run_id=self.run_id,
                catalytic_domains=[str(d.relative_to(PROJECT_ROOT)) for d in self.catalytic_domains],
                pre_state=pre_manifest,
                post_state=post_manifest,
                timestamp=self.timestamp,
                referenced_artifacts={
                    "ledger_hash": ledger_hash,
                    "jobspec_hash": jobspec_hash,
                    "validator_id": {
                        "validator_semver": validator_id["validator_semver"],
                        "validator_build_id": validator_id["validator_build_id"],
                    },
                },
            )
            self.fs_guard.guarded_write_text(self.ledger_dir / "PROOF.json", canonical_json_bytes(proof).decode("utf-8"))

            # Legacy artifacts (keep for backwards compatibility)
            ledger_entry = {
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "intent": self.intent,
                "catalytic_domains": [str(d.relative_to(PROJECT_ROOT)) for d in self.catalytic_domains],
                "exit_code": exit_code,
                "restoration_verified": restoration_success,
            }
            self.fs_guard.guarded_write_text(self.ledger_dir / "RUN_INFO.json", json.dumps(ledger_entry, indent=2))
            self.fs_guard.guarded_write_text(self.ledger_dir / "PRE_MANIFEST.json", json.dumps(pre_manifest, indent=2))
            self.fs_guard.guarded_write_text(self.ledger_dir / "POST_MANIFEST.json", json.dumps(post_manifest, indent=2))
            self.fs_guard.guarded_write_text(self.ledger_dir / "RESTORE_DIFF.json", json.dumps(diffs, indent=2))
            self.fs_guard.guarded_write_text(self.ledger_dir / "OUTPUTS.json", json.dumps(outputs_list, indent=2))

            print(f"[catalytic] Canonical artifact set written to {self.ledger_dir}")
            print(f"[catalytic] PROOF.json: verified={proof['restoration_result']['verified']}, condition={proof['restoration_result']['condition']}")

            return True

        except Exception as e:
            print(f"[catalytic] ERROR writing canonical artifacts: {e}", file=sys.stderr)
            # Attempt to write failed STATUS
            try:
                failed_status = {
                    "status": "failed",
                    "restoration_verified": False,
                    "exit_code": 1,
                    "validation_passed": False,
                    "error": str(e),
                }
                self.fs_guard.guarded_write_text(self.ledger_dir / "STATUS.json", json.dumps(failed_status, indent=2))
            except:
                pass
            return False

    def run(self, cmd: List[str]) -> int:
        """
        Execute the full catalytic lifecycle with canonical artifact writing.

        State machine:
        - started: execution begun
        - succeeded: command succeeded, restoration verified
        - failed: command failed OR restoration failed OR artifact writing failed
        - verified: (future) external validation passed
        """
        print(f"[catalytic] Starting catalytic run: {self.run_id}")

        # Phase 0: Preflight validation (FAIL CLOSED)
        print("[catalytic] Phase 0: Preflight validation...")
        preflight_valid, preflight_errors = self.preflight_validate()

        if not preflight_valid:
            print("[catalytic] PREFLIGHT REJECTION: JobSpec violates contract", file=sys.stderr)
            for error in preflight_errors:
                print(f"  [{error['code']}] {error['message']}", file=sys.stderr)
            # Write failed artifacts
            self.write_canonical_artifacts(exit_code=1, restoration_success=False, diffs={}, status_state="failed")
            return 1

        print(f"[catalytic] Preflight passed: JobSpec is valid")

        # Legacy validation (for backwards compatibility)
        if not self.validate_config():
            # Write failed artifacts
            self.write_canonical_artifacts(exit_code=1, restoration_success=False, diffs={}, status_state="failed")
            return 1

        # Write started status
        try:
            started_status = {"status": "started", "restoration_verified": False, "exit_code": None, "validation_passed": False}
            self.fs_guard.guarded_write_text(self.ledger_dir / "STATUS.json", json.dumps(started_status, indent=2))
        except Exception as e:
            print(f"[catalytic] ERROR: Cannot write STATUS.json: {e}", file=sys.stderr)
            return 1

        # Phase 1: Snapshot
        print("[catalytic] Phase 1: Capturing pre-snapshots...")
        self.snapshot_domains()

        # Phase 2: Execute
        print(f"[catalytic] Phase 2: Executing command: {' '.join(cmd)}")
        exit_code = self.execute(cmd)
        print(f"[catalytic] Command exited with code {exit_code}")

        # Phase 3: Post-snapshot
        print("[catalytic] Phase 3: Capturing post-snapshots...")
        self.snapshot_after()

        # Phase 4: Verify restoration
        print("[catalytic] Phase 4: Verifying restoration...")
        restored, diffs = self.verify_restoration()

        if restored:
            print("[catalytic] SUCCESS: Catalytic domains fully restored")
        else:
            print("[catalytic] FAILURE: Catalytic domains were not fully restored", file=sys.stderr)
            print("[catalytic] Differences:", file=sys.stderr)
            print(json.dumps(diffs, indent=2), file=sys.stderr)

        # Determine final status state
        if exit_code == 0 and restored:
            status_state = "succeeded"
        else:
            status_state = "failed"

        # Phase 5: Write canonical artifact set
        print("[catalytic] Phase 5: Writing canonical artifact set...")
        artifacts_written = self.write_canonical_artifacts(exit_code, restored, diffs, status_state)

        if not artifacts_written:
            print(
                "[catalytic] HARD FAIL: Failed to write canonical artifact set. Run is invalid.",
                file=sys.stderr,
            )
            return 1

        # Final decision: hard fail if restoration failed
        if not restored:
            print(
                "[catalytic] HARD FAIL: Restoration proof failed. Run is invalid.",
                file=sys.stderr,
            )
            return 1

        if exit_code != 0:
            print(
                f"[catalytic] FAIL: Command exited with code {exit_code}.",
                file=sys.stderr,
            )
            return exit_code

        print("[catalytic] Run completed successfully with verified restoration.")
        return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Catalytic runtime executor (CMP-01)")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument(
        "--catalytic-domains",
        required=True,
        nargs="+",
        help="Paths allowed to be mutated (space-separated)",
    )
    parser.add_argument(
        "--durable-outputs",
        required=True,
        nargs="+",
        help="Paths where outputs may remain (space-separated)",
    )
    parser.add_argument("--intent", required=True, help="One-sentence description of the run")
    parser.add_argument(
        "--determinism",
        default="deterministic",
        choices=["deterministic", "bounded_nondeterministic", "nondeterministic"],
        help="Determinism level",
    )
    parser.add_argument("--timestamp", default=None, help="Deterministic timestamp string (if omitted, uses sentinel)")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute")

    args = parser.parse_args()

    if not args.cmd or args.cmd[0] != "--":
        print("ERROR: Command must be preceded by --", file=sys.stderr)
        return 1

    runtime = CatalyticRuntime(
        run_id=args.run_id,
        catalytic_domains=args.catalytic_domains,
        durable_outputs=args.durable_outputs,
        intent=args.intent,
        determinism=args.determinism,
        timestamp=args.timestamp,
    )

    return runtime.run(args.cmd[1:])


if __name__ == "__main__":
    sys.exit(main())

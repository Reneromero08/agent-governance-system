#!/usr/bin/env python3

"""
Catalytic Runtime: Enforce catalytic space constraints during execution.

Implements CMP-01 (Catalytic Mutation Protocol) by:
1. Taking pre-snapshots of catalytic domains
2. Executing a wrapped subprocess/function
3. Verifying restoration of catalytic domains
4. Recording run ledger with proof

Usage:
  python catalytic_runtime.py \
    --run-id cortex-build-2025-12-23 \
    --catalytic-domains CORTEX/_generated/_tmp \
    --durable-outputs CORTEX/_generated/cortex.json \
    --intent "Build cortex index" \
    -- python CORTEX/cortex.build.py

The run ledger is stored in CONTRACTS/_runs/<run_id>/
"""

import json
import sys
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent.parent


class CatalyticSnapshot:
    """Snapshot the state of a filesystem domain."""

    def __init__(self, domain_path: Path):
        self.domain_path = domain_path
        self.files: Dict[str, str] = {}  # path -> sha256

    def capture(self) -> None:
        """Recursively snapshot all files in domain."""
        if not self.domain_path.exists():
            return

        for file_path in self.domain_path.rglob("*"):
            if file_path.is_file():
                try:
                    sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    rel_path = str(file_path.relative_to(self.domain_path))
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
    ):
        self.run_id = run_id
        self.catalytic_domains = [PROJECT_ROOT / d for d in catalytic_domains]
        self.durable_outputs = [PROJECT_ROOT / d for d in durable_outputs]
        self.intent = intent
        self.determinism = determinism
        self.timestamp = datetime.utcnow().isoformat()

        # Run ledger directory
        self.ledger_dir = PROJECT_ROOT / "CONTRACTS" / "_runs" / run_id
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

        # Snapshots
        self.pre_snapshots: Dict[str, CatalyticSnapshot] = {}
        self.post_snapshots: Dict[str, CatalyticSnapshot] = {}

    def validate_config(self) -> bool:
        """Pre-flight: validate that domains and outputs are legal."""
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

        # Check catalytic domains don't overlap forbidden
        for domain in self.catalytic_domains:
            for forbidden in forbidden_paths:
                try:
                    domain.relative_to(forbidden)
                    print(
                        f"[catalytic] ERROR: Catalytic domain {domain} overlaps forbidden path {forbidden}",
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

    def snapshot_domains(self) -> None:
        """Take pre-snapshots of all catalytic domains."""
        for domain in self.catalytic_domains:
            snapshot = CatalyticSnapshot(domain)
            snapshot.capture()
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
            snapshot.capture()
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

    def save_ledger(self, exit_code: int, restoration_success: bool, diffs: Dict) -> None:
        """Save run ledger to disk."""
        run_info = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "determinism": self.determinism,
            "exit_code": exit_code,
            "catalytic_domains": [str(d.relative_to(PROJECT_ROOT)) for d in self.catalytic_domains],
            "durable_output_roots": [str(d.relative_to(PROJECT_ROOT)) for d in self.durable_outputs],
        }

        (self.ledger_dir / "RUN_INFO.json").write_text(json.dumps(run_info, indent=2))

        pre_manifest = {
            domain: snapshot.to_dict() for domain, snapshot in self.pre_snapshots.items()
        }
        (self.ledger_dir / "PRE_MANIFEST.json").write_text(json.dumps(pre_manifest, indent=2))

        post_manifest = {
            domain: snapshot.to_dict() for domain, snapshot in self.post_snapshots.items()
        }
        (self.ledger_dir / "POST_MANIFEST.json").write_text(json.dumps(post_manifest, indent=2))

        restore_diff = diffs
        (self.ledger_dir / "RESTORE_DIFF.json").write_text(json.dumps(restore_diff, indent=2))

        outputs_list = self.collect_outputs()
        (self.ledger_dir / "OUTPUTS.json").write_text(json.dumps(outputs_list, indent=2))

        status = {
            "status": "restored" if restoration_success else "dirty",
            "restoration_verified": restoration_success,
        }
        (self.ledger_dir / "STATUS.json").write_text(json.dumps(status, indent=2))

        print(f"[catalytic] Run ledger saved to {self.ledger_dir}")

    def run(self, cmd: List[str]) -> int:
        """Execute the full catalytic lifecycle."""
        print(f"[catalytic] Starting catalytic run: {self.run_id}")

        # Phase 0: Validate config
        if not self.validate_config():
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

        # Phase 5: Save ledger
        print("[catalytic] Phase 5: Saving run ledger...")
        self.save_ledger(exit_code, restored, diffs)

        # Final decision: hard fail if restoration failed
        if not restored:
            print(
                "[catalytic] HARD FAIL: Restoration proof failed. Run is invalid.",
                file=sys.stderr,
            )
            return 1

        return exit_code


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
    )

    return runtime.run(args.cmd[1:])


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

"""
Catalytic Run Ledger Validator: Verify CMP-01 compliance for a completed run.

Validates:
1. Run info schema is present and valid
2. PRE and POST manifests exist
3. PROOF.json exists and has verified=true (restoration proof-gated)
4. Durable outputs are listed and exist
5. No outputs appear outside durable roots

Acceptance is STRICTLY proof-driven: a run is accepted iff PROOF.json.verified == true.

Usage:
  python catalytic_validator.py --run-dir CONTRACTS/_runs/<run_id>
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add CATALYTIC-DPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "CATALYTIC-DPT"))
from PRIMITIVES.restore_proof import RestorationProofValidator


class CatalyticLedgerValidator:
    """Validate a run ledger against CMP-01 requirements."""

    def __init__(self, ledger_dir: Path):
        self.ledger_dir = Path(ledger_dir)
        self.errors: list = []
        self.warnings: list = []

    def validate_structure(self) -> bool:
        """Check required files exist."""
        required_files = ["RUN_INFO.json", "PRE_MANIFEST.json", "POST_MANIFEST.json", "RESTORE_DIFF.json", "OUTPUTS.json", "STATUS.json", "PROOF.json"]

        for filename in required_files:
            path = self.ledger_dir / filename
            if not path.exists():
                self.errors.append(f"Missing required file: {filename}")
                return False

        return True

    def validate_schemas(self) -> bool:
        """Validate JSON schemas."""
        try:
            self.run_info = json.loads((self.ledger_dir / "RUN_INFO.json").read_text())
            self.pre_manifest = json.loads((self.ledger_dir / "PRE_MANIFEST.json").read_text())
            self.post_manifest = json.loads((self.ledger_dir / "POST_MANIFEST.json").read_text())
            self.restore_diff = json.loads((self.ledger_dir / "RESTORE_DIFF.json").read_text())
            self.outputs = json.loads((self.ledger_dir / "OUTPUTS.json").read_text())
            self.status = json.loads((self.ledger_dir / "STATUS.json").read_text())
            self.proof = json.loads((self.ledger_dir / "PROOF.json").read_text())
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {e}")
            return False

        # Validate RUN_INFO structure
        required_fields = ["run_id", "timestamp", "intent", "catalytic_domains", "exit_code"]
        for field in required_fields:
            if field not in self.run_info:
                self.errors.append(f"RUN_INFO missing field: {field}")
                return False

        return True

    def validate_restoration(self) -> bool:
        """
        PROOF-GATED ACCEPTANCE: Check PROOF.json has verified=true.

        This is the ONLY source of truth for restoration acceptance.
        No heuristics, no logs, no RESTORE_DIFF inspection.
        """
        # Validate PROOF.json structure
        if not isinstance(self.proof, dict):
            self.errors.append("PROOF.json is not a dict")
            return False

        if "restoration_result" not in self.proof:
            self.errors.append("PROOF.json missing restoration_result")
            return False

        restoration_result = self.proof["restoration_result"]

        if not isinstance(restoration_result, dict):
            self.errors.append("restoration_result is not a dict")
            return False

        if "verified" not in restoration_result:
            self.errors.append("restoration_result missing 'verified' field")
            return False

        if "condition" not in restoration_result:
            self.errors.append("restoration_result missing 'condition' field")
            return False

        # HARD GATE: verified must be exactly true
        if restoration_result["verified"] is not True:
            condition = restoration_result.get("condition", "UNKNOWN")
            self.errors.append(
                f"PROOF-GATED REJECTION: verified={restoration_result['verified']}, condition={condition}"
            )

            # Surface mismatches if present
            if "mismatches" in restoration_result:
                mismatches = restoration_result["mismatches"]
                self.errors.append(f"Restoration failures: {len(mismatches)} mismatches")
                for mismatch in mismatches[:5]:  # Show first 5
                    self.errors.append(f"  - {mismatch.get('path')}: {mismatch.get('type')}")

            return False

        # SUCCESS: proof verified
        return True

    def validate_outputs(self) -> bool:
        """Check durable outputs are valid and exist."""
        if not isinstance(self.outputs, list):
            self.errors.append("OUTPUTS is not a list")
            return False

        allowed_roots = [
            "CONTRACTS/_runs",
            "CORTEX/_generated",
            "MEMORY/LLM_PACKER/_packs",
        ]

        project_root = Path(__file__).parent.parent

        for output in self.outputs:
            if not isinstance(output, dict) or "path" not in output:
                self.errors.append(f"Invalid output entry: {output}")
                return False

            # Normalize path separators (handle Windows backslashes)
            output_path = output["path"].replace("\\", "/")
            is_allowed = any(output_path.startswith(root) for root in allowed_roots)

            if not is_allowed:
                self.errors.append(f"Output {output_path} not under allowed roots")
                return False

            # Check that output actually exists
            full_path = project_root / output_path
            if not full_path.exists():
                self.errors.append(f"Output {output_path} does not exist (ledger claims it but it's missing)")
                return False

        return True

    def validate(self) -> Tuple[bool, Dict]:
        """Run all validations. Return (success, report)."""
        report = {
            "ledger_dir": str(self.ledger_dir),
            "valid": False,
            "errors": [],
            "warnings": [],
            "details": {},
        }

        if not self.validate_structure():
            report["errors"] = self.errors
            return False, report

        if not self.validate_schemas():
            report["errors"] = self.errors
            return False, report

        if not self.validate_restoration():
            report["errors"] = self.errors
            return False, report

        if not self.validate_outputs():
            report["errors"] = self.errors
            return False, report

        report["valid"] = True
        report["details"] = {
            "run_id": self.run_info.get("run_id"),
            "intent": self.run_info.get("intent"),
            "exit_code": self.run_info.get("exit_code"),
            "restoration_verified": self.status.get("restoration_verified"),
            "output_count": len(self.outputs),
        }

        return True, report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate catalytic run ledger (CMP-01)")
    parser.add_argument("--run-dir", required=True, help="Path to run ledger directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    validator = CatalyticLedgerValidator(Path(args.run_dir))
    success, report = validator.validate()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if report["valid"]:
            print(f"[catalytic-validator] PASS: {report['details']['run_id']}")
            print(f"  Intent: {report['details']['intent']}")
            print(f"  Exit code: {report['details']['exit_code']}")
            print(f"  Outputs: {report['details']['output_count']}")
        else:
            print("[catalytic-validator] FAIL")
            for error in report["errors"]:
                print(f"  ERROR: {error}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

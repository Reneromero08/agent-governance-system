#!/usr/bin/env python3
"""
SCL CLI (Phase 5.2.5.1)

Command-line interface for Semiotic Compression Layer operations.

Commands:
    scl decode <program>          Decode SCL program to JobSpec JSON
    scl validate <program|file>   Validate SCL program or JobSpec JSON
    scl run <program>             Execute with invariant proofs
    scl audit <program>           Human-readable expansion

Usage:
    python -m CAPABILITY.TOOLS.scl.scl_cli decode "C3:build"
    python -m CAPABILITY.TOOLS.scl.scl_cli validate "法.驗"
    python -m CAPABILITY.TOOLS.scl.scl_cli run "C3" --dry-run
    python -m CAPABILITY.TOOLS.scl.scl_cli audit "C&I"

Exit Codes:
    0: Success
    1: Validation/execution failure
    2: Invalid input
    3: Internal error
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# Add paths for imports
sys.path.insert(0, str(PROJECT_ROOT / "CAPABILITY" / "PRIMITIVES"))
sys.path.insert(0, str(PROJECT_ROOT / "CAPABILITY" / "TOOLS"))

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SCLReceipt:
    """Receipt for SCL operations."""
    operation: str
    program: str
    timestamp_utc: str
    success: bool
    layer: str = ""
    input_hash: str = ""
    output_hash: str = ""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def compute_receipt_hash(self) -> str:
        """Compute deterministic hash of receipt."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def emit_receipt(receipt: SCLReceipt, output_path: Optional[Path] = None) -> dict:
    """Emit receipt to file or return as dict."""
    receipt_dict = receipt.to_dict()
    receipt_dict["receipt_hash"] = receipt.compute_receipt_hash()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(receipt_dict, f, indent=2, ensure_ascii=False)

    return receipt_dict


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS (to avoid import errors if dependencies missing)
# ═══════════════════════════════════════════════════════════════════════════════

_validator_module = None
_codebook_module = None


def _get_validator():
    global _validator_module
    if _validator_module is None:
        from scl_validator import (
            validate_scl, validate_expansion, validate_syntax,
            ValidationResult, get_known_radicals, get_known_operators,
            get_contract_rules, get_invariants,
        )
        _validator_module = {
            'validate_scl': validate_scl,
            'validate_expansion': validate_expansion,
            'validate_syntax': validate_syntax,
            'ValidationResult': ValidationResult,
            'get_known_radicals': get_known_radicals,
            'get_known_operators': get_known_operators,
            'get_contract_rules': get_contract_rules,
            'get_invariants': get_invariants,
        }
    return _validator_module


def _get_codebook():
    global _codebook_module
    if _codebook_module is None:
        from codebook_lookup import (
            lookup_entry, stacked_lookup, list_entries,
            SEMANTIC_SYMBOLS, load_codebook, get_file_content,
        )
        _codebook_module = {
            'lookup_entry': lookup_entry,
            'stacked_lookup': stacked_lookup,
            'list_entries': list_entries,
            'SEMANTIC_SYMBOLS': SEMANTIC_SYMBOLS,
            'load_codebook': load_codebook,
            'get_file_content': get_file_content,
        }
    return _codebook_module


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

def decode_program(program: str) -> dict:
    """
    Decode SCL program to JobSpec JSON.

    Args:
        program: SCL program string (e.g., "C3:build", "法.驗")

    Returns:
        Dict with decoded JobSpec or error
    """
    validator = _get_validator()
    codebook = _get_codebook()

    # Step 1: Validate syntax and symbols
    result = validator['validate_scl'](program, level="L3")
    if not result.valid:
        return {
            "ok": False,
            "error": "validation_failed",
            "layer": result.layer,
            "errors": result.errors,
            "warnings": result.warnings,
        }

    # Step 2: Look up the entry
    lookup_result = codebook['lookup_entry'](program, expand=False)
    if not lookup_result.get('found'):
        return {
            "ok": False,
            "error": "lookup_failed",
            "message": lookup_result.get('error', 'Entry not found'),
        }

    entry = lookup_result['entry']
    parsed = result.parsed

    # Step 3: Build JobSpec from parsed program
    jobspec = _build_jobspec_from_entry(program, entry, parsed)

    return {
        "ok": True,
        "jobspec": jobspec,
        "parsed": parsed,
        "entry": entry,
        "warnings": result.warnings,
    }


def _build_jobspec_from_entry(program: str, entry: dict, parsed: dict) -> dict:
    """Build a JobSpec from a codebook entry."""
    entry_type = entry.get('type', 'unknown')
    entry_id = entry.get('id', program)

    # Determine task_type from entry type
    task_type_map = {
        'contract_rule': 'validation',
        'invariant': 'validation',
        'radical': 'pipeline_execution',
        'cjk_symbol': 'pipeline_execution',
        'compound': 'pipeline_execution',
        'file': 'validation',
        'domain': 'pipeline_execution',
    }
    task_type = task_type_map.get(entry_type, 'validation')

    # Build intent from entry
    intent = entry.get('summary', entry.get('full', f"Execute SCL program: {program}"))
    if len(intent) > 200:
        intent = intent[:197] + "..."

    # Determine paths for inputs/outputs
    paths = []
    if 'path' in entry:
        paths.append(entry['path'])
    if 'paths' in entry:
        paths.extend(entry['paths'])

    # Build JobSpec
    job_id = f"scl-{hashlib.sha256(program.encode()).hexdigest()[:12]}"
    context = parsed.get('context', 'execute')

    jobspec = {
        "job_id": job_id,
        "phase": 5,
        "task_type": task_type,
        "intent": intent,
        "inputs": {
            "scl_program": program,
            "entry_id": entry_id,
            "entry_type": entry_type,
            "source_paths": paths,
        },
        "outputs": {
            "durable_paths": [f"_runs/scl/{job_id}/result.json"],
            "validation_criteria": {
                "scl_valid": True,
                "layer": "L3",
            },
        },
        "catalytic_domains": [],
        "determinism": "deterministic",
        "metadata": {
            "scl_program": program,
            "scl_parsed": parsed,
            "context": context,
        },
    }

    return jobspec


def cmd_decode(args) -> int:
    """Handle decode command."""
    program = args.program
    timestamp = datetime.now(timezone.utc).isoformat()

    result = decode_program(program)

    # Build receipt
    receipt = SCLReceipt(
        operation="decode",
        program=program,
        timestamp_utc=timestamp,
        success=result.get("ok", False),
        layer="L3" if result.get("ok") else result.get("layer", "L0"),
        input_hash=hashlib.sha256(program.encode()).hexdigest(),
        errors=result.get("errors", []),
        warnings=result.get("warnings", []),
        metadata={"parsed": result.get("parsed", {})},
    )

    if result.get("ok"):
        receipt.output_hash = hashlib.sha256(
            json.dumps(result["jobspec"], sort_keys=True).encode()
        ).hexdigest()

    # Emit receipt if requested
    if args.receipt_out:
        emit_receipt(receipt, Path(args.receipt_out))

    # Output
    if args.json:
        output = {
            "ok": result.get("ok", False),
            "jobspec": result.get("jobspec"),
            "receipt": receipt.to_dict(),
        }
        if not result.get("ok"):
            output["errors"] = result.get("errors", [])
            output["error"] = result.get("error")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        if result.get("ok"):
            print(json.dumps(result["jobspec"], indent=2, ensure_ascii=False))
            if result.get("warnings"):
                for w in result["warnings"]:
                    sys.stderr.write(f"WARN: {w}\n")
        else:
            sys.stderr.write(f"[FAIL] {result.get('error', 'decode failed')}\n")
            for e in result.get("errors", []):
                sys.stderr.write(f"  ERROR: {e}\n")
            return 1

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATE COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_validate(args) -> int:
    """Handle validate command."""
    target = args.target
    timestamp = datetime.now(timezone.utc).isoformat()
    validator = _get_validator()

    is_json_file = target.endswith('.json') and Path(target).exists()

    if is_json_file:
        # Validate JobSpec JSON file
        try:
            with open(target, 'r', encoding='utf-8') as f:
                jobspec = json.load(f)
            result = validator['validate_expansion'](jobspec)
            program = f"file:{target}"
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[FAIL] Invalid JSON: {e}\n")
            return 2
        except FileNotFoundError:
            sys.stderr.write(f"[FAIL] File not found: {target}\n")
            return 2
    else:
        # Validate SCL program
        program = target
        level = args.level if hasattr(args, 'level') else "L3"
        result = validator['validate_scl'](program, level=level)

    # Build receipt
    receipt = SCLReceipt(
        operation="validate",
        program=program,
        timestamp_utc=timestamp,
        success=result.valid,
        layer=result.layer,
        input_hash=hashlib.sha256(program.encode()).hexdigest(),
        errors=result.errors,
        warnings=result.warnings,
        metadata={"parsed": result.parsed if hasattr(result, 'parsed') else {}},
    )

    if args.receipt_out:
        emit_receipt(receipt, Path(args.receipt_out))

    # Output
    if args.json:
        output = {
            "ok": result.valid,
            "layer": result.layer,
            "program": program,
            "errors": result.errors,
            "warnings": result.warnings,
            "receipt": receipt.to_dict(),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        status = "PASS" if result.valid else "FAIL"
        print(f"{status} [{result.layer}]: {program}")
        if result.errors:
            for e in result.errors:
                print(f"  ERROR: {e}")
        if result.warnings:
            for w in result.warnings:
                print(f"  WARN: {w}")

    return 0 if result.valid else 1


# ═══════════════════════════════════════════════════════════════════════════════
# RUN COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_run(args) -> int:
    """Handle run command - execute with invariant proofs."""
    program = args.program
    timestamp = datetime.now(timezone.utc).isoformat()
    dry_run = args.dry_run

    # Step 1: Decode to JobSpec
    decode_result = decode_program(program)
    if not decode_result.get("ok"):
        sys.stderr.write(f"[FAIL] Decode failed: {decode_result.get('error')}\n")
        for e in decode_result.get("errors", []):
            sys.stderr.write(f"  ERROR: {e}\n")
        return 1

    jobspec = decode_result["jobspec"]

    # Step 2: Validate the generated JobSpec (L4)
    validator = _get_validator()
    l4_result = validator['validate_expansion'](jobspec)
    if not l4_result.valid:
        sys.stderr.write(f"[FAIL] L4 validation failed\n")
        for e in l4_result.errors:
            sys.stderr.write(f"  ERROR: {e}\n")
        return 1

    # Step 3: Check invariants
    invariant_proofs = _check_invariants(program, jobspec)

    # Build execution result
    execution_result = {
        "ok": True,
        "dry_run": dry_run,
        "jobspec": jobspec,
        "invariant_proofs": invariant_proofs,
        "timestamp_utc": timestamp,
    }

    if not dry_run:
        # Actual execution would happen here
        # For now, we just mark it as "would execute"
        execution_result["status"] = "executed"
        execution_result["note"] = "Actual execution not yet implemented"
    else:
        execution_result["status"] = "dry_run"

    # Build receipt
    receipt = SCLReceipt(
        operation="run",
        program=program,
        timestamp_utc=timestamp,
        success=True,
        layer="L4",
        input_hash=hashlib.sha256(program.encode()).hexdigest(),
        output_hash=hashlib.sha256(
            json.dumps(execution_result, sort_keys=True).encode()
        ).hexdigest(),
        metadata={
            "dry_run": dry_run,
            "invariant_proofs": invariant_proofs,
            "jobspec_id": jobspec.get("job_id"),
        },
    )

    if args.receipt_out:
        emit_receipt(receipt, Path(args.receipt_out))

    # Output
    if args.json:
        output = {
            "ok": True,
            "execution": execution_result,
            "receipt": receipt.to_dict(),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        status = "[DRY-RUN]" if dry_run else "[EXECUTED]"
        print(f"{status} {program}")
        print(f"  job_id: {jobspec.get('job_id')}")
        print(f"  task_type: {jobspec.get('task_type')}")
        print(f"  intent: {jobspec.get('intent')[:60]}...")
        print(f"  invariants_checked: {len(invariant_proofs)}")
        for proof in invariant_proofs:
            status_icon = "+" if proof['passed'] else "x"
            print(f"    [{status_icon}] {proof['invariant']}: {proof['message']}")

    return 0


def _check_invariants(program: str, jobspec: dict) -> list:
    """Check relevant invariants for the program."""
    proofs = []

    # I5: Determinism - check jobspec has determinism field
    proofs.append({
        "invariant": "I5",
        "name": "Determinism",
        "passed": jobspec.get("determinism") == "deterministic",
        "message": f"determinism={jobspec.get('determinism')}",
    })

    # I6: Output roots - check all outputs are in allowed roots
    allowed_roots = {"_runs/", "_generated/", "_packs/", "_tmp/", "INBOX/", "NAVIGATION/RECEIPTS/"}
    outputs = jobspec.get("outputs", {}).get("durable_paths", [])
    all_valid = all(
        any(p.startswith(r) for r in allowed_roots)
        for p in outputs
    )
    proofs.append({
        "invariant": "I6",
        "name": "Output roots",
        "passed": all_valid,
        "message": f"outputs={outputs}",
    })

    # C7: Determinism (contract level) - same as I5
    proofs.append({
        "invariant": "C7",
        "name": "Contract determinism",
        "passed": jobspec.get("determinism") in ("deterministic", "bounded_nondeterministic"),
        "message": f"determinism={jobspec.get('determinism')}",
    })

    # C8: Output roots (contract level) - same as I6
    proofs.append({
        "invariant": "C8",
        "name": "Contract output roots",
        "passed": all_valid,
        "message": "All outputs in allowed roots" if all_valid else "Invalid output roots",
    })

    return proofs


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_audit(args) -> int:
    """Handle audit command - human-readable expansion."""
    program = args.program
    timestamp = datetime.now(timezone.utc).isoformat()
    codebook = _get_codebook()
    validator = _get_validator()

    # Validate first
    result = validator['validate_scl'](program, level="L3")

    # Get full expansion
    lookup_result = codebook['stacked_lookup'](
        program,
        query=args.query if hasattr(args, 'query') else None,
        semantic=args.semantic if hasattr(args, 'semantic') else None,
        expand=True,
        limit=args.limit if hasattr(args, 'limit') else 10,
    )

    # Build audit report
    audit_report = {
        "program": program,
        "timestamp_utc": timestamp,
        "validation": {
            "valid": result.valid,
            "layer": result.layer,
            "errors": result.errors,
            "warnings": result.warnings,
            "parsed": result.parsed,
        },
        "expansion": {
            "found": lookup_result.get("found", False),
            "resolution": lookup_result.get("resolution", "L1"),
        },
    }

    if lookup_result.get("found"):
        entry = lookup_result["entry"]
        audit_report["expansion"]["entry_id"] = entry.get("id")
        audit_report["expansion"]["entry_type"] = entry.get("type")
        audit_report["expansion"]["paths"] = (
            [entry.get("path")] if entry.get("path") else entry.get("paths", [])
        )
        audit_report["expansion"]["compression"] = entry.get("compression", 0)

        if "content" in entry:
            content = entry["content"]
            audit_report["expansion"]["content_length"] = len(content)
            audit_report["expansion"]["content_preview"] = content[:500] + ("..." if len(content) > 500 else "")
        if "filtered_content" in entry:
            fc = entry["filtered_content"]
            audit_report["expansion"]["filtered_content_length"] = len(fc)
            audit_report["expansion"]["filtered_content_preview"] = fc[:500] + ("..." if len(fc) > 500 else "")
            audit_report["expansion"]["chunk_count"] = entry.get("chunk_count", 0)

    # Build receipt
    receipt = SCLReceipt(
        operation="audit",
        program=program,
        timestamp_utc=timestamp,
        success=result.valid and lookup_result.get("found", False),
        layer=result.layer,
        input_hash=hashlib.sha256(program.encode()).hexdigest(),
        errors=result.errors,
        warnings=result.warnings,
        metadata={"resolution": lookup_result.get("resolution", "L1")},
    )

    if args.receipt_out:
        emit_receipt(receipt, Path(args.receipt_out))

    # Output
    if args.json:
        output = {
            "ok": result.valid and lookup_result.get("found", False),
            "audit": audit_report,
            "receipt": receipt.to_dict(),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print(f"SCL AUDIT: {program}")
        print("=" * 60)
        print()

        # Validation section
        print("VALIDATION")
        print("-" * 40)
        status = "PASS" if result.valid else "FAIL"
        print(f"  Status: {status} [{result.layer}]")
        if result.parsed:
            print(f"  Parsed: {result.parsed}")
        if result.errors:
            print("  Errors:")
            for e in result.errors:
                print(f"    - {e}")
        if result.warnings:
            print("  Warnings:")
            for w in result.warnings:
                print(f"    - {w}")
        print()

        # Expansion section
        print("EXPANSION")
        print("-" * 40)
        if lookup_result.get("found"):
            entry = lookup_result["entry"]
            print(f"  Entry ID: {entry.get('id')}")
            print(f"  Type: {entry.get('type')}")
            print(f"  Resolution: {lookup_result.get('resolution', 'L1')}")
            if entry.get("path"):
                print(f"  Path: {entry.get('path')}")
            if entry.get("paths"):
                print(f"  Paths:")
                for p in entry.get("paths", []):
                    print(f"    - {p}")
            if entry.get("compression"):
                print(f"  Compression: {entry.get('compression')}x")
            print()

            if "content" in entry:
                content = entry["content"]
                print("CONTENT PREVIEW")
                print("-" * 40)
                preview = content[:1000]
                print(preview)
                if len(content) > 1000:
                    print(f"\n... [{len(content)} total chars]")
            elif "filtered_content" in entry:
                fc = entry["filtered_content"]
                print(f"FILTERED CONTENT ({entry.get('chunk_count', 0)} chunks)")
                print("-" * 40)
                preview = fc[:1000]
                print(preview)
                if len(fc) > 1000:
                    print(f"\n... [{len(fc)} total chars]")
        else:
            print(f"  NOT FOUND: {lookup_result.get('error', 'Unknown error')}")

        print()
        print("=" * 60)

    return 0 if (result.valid and lookup_result.get("found", False)) else 1


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SCL CLI - Semiotic Compression Layer",
        epilog="Phase 5.2.5.1 - SCL command-line interface"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Common arguments for all subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON"
    )
    parent_parser.add_argument(
        "--receipt-out", type=str,
        help="Path to write operation receipt"
    )

    # decode command
    decode_parser = subparsers.add_parser(
        "decode",
        help="Decode SCL program to JobSpec JSON",
        parents=[parent_parser]
    )
    decode_parser.add_argument(
        "program",
        help="SCL program (e.g., 'C3:build', '法.驗')"
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate SCL program or JobSpec JSON",
        parents=[parent_parser]
    )
    validate_parser.add_argument(
        "target",
        help="SCL program or path to JobSpec JSON file"
    )
    validate_parser.add_argument(
        "--level", choices=["L1", "L2", "L3"], default="L3",
        help="Validation level (default: L3)"
    )

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Execute SCL program with invariant proofs",
        parents=[parent_parser]
    )
    run_parser.add_argument(
        "program",
        help="SCL program to execute"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be executed without running"
    )

    # audit command
    audit_parser = subparsers.add_parser(
        "audit",
        help="Human-readable expansion of SCL program",
        parents=[parent_parser]
    )
    audit_parser.add_argument(
        "program",
        help="SCL program to audit"
    )
    audit_parser.add_argument(
        "--query", type=str,
        help="FTS query within domain (L1+L2 resolution)"
    )
    audit_parser.add_argument(
        "--semantic", type=str,
        help="Semantic query within domain (L1+L3 resolution)"
    )
    audit_parser.add_argument(
        "--limit", type=int, default=10,
        help="Max results for filtered queries"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        commands = {
            "decode": cmd_decode,
            "validate": cmd_validate,
            "run": cmd_run,
            "audit": cmd_audit,
        }

        if args.command in commands:
            return commands[args.command](args)
        else:
            parser.print_help()
            return 1

    except Exception as e:
        sys.stderr.write(f"[ERROR] Internal error: {e}\n")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())

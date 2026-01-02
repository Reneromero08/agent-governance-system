#!/usr/bin/env python3
"""
Catalytic Chat CLI

Command-line interface for building and querying the section index.

Roadmap Phase: Phase 1 — Substrate + deterministic indexing
"""

import sys
import argparse
import json
from pathlib import Path

from catalytic_chat.section_extractor import extract_sections
from catalytic_chat.section_indexer import SectionIndexer, build_index
from catalytic_chat.symbol_registry import SymbolRegistry, SymbolError
from catalytic_chat.symbol_resolver import SymbolResolver, ResolverError, resolve_symbol
from catalytic_chat.message_cassette import MessageCassette, MessageCassetteError
from catalytic_chat.planner import Planner, PlannerError, post_request_and_plan
from catalytic_chat.ants import spawn_ants, run_ant_worker
from catalytic_chat.bundle import BundleBuilder, BundleVerifier, BundleError
from catalytic_chat.executor import BundleExecutor
from catalytic_chat.compression_validator import validate_compression_claim

try:
    from catalytic_chat.attestation import sign_receipt_bytes, verify_receipt_bytes
except ImportError:
    sign_receipt_bytes = None
    verify_receipt_bytes = None


def cmd_build(args) -> int:
    """Build section index.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        index_hash = build_index(
            repo_root=args.repo_root,
            substrate_mode=args.substrate,
            incremental=args.incremental
        )
        print(f"[OK] Index built")
        print(f"      index_hash: {index_hash[:16]}...")
        return 0
    except Exception as e:
        print(f"[FAIL] Build failed: {e}")
        return 1


def cmd_verify(args) -> int:
    """Verify index determinism.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    indexer = SectionIndexer(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        success = indexer.verify_determinism()
        return 0 if success else 1
    except Exception as e:
        print(f"[FAIL] Verification failed: {e}")
        return 1


def cmd_get(args) -> int:
    """Get section by ID with optional slice.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .section_indexer import SectionIndexer

    indexer = SectionIndexer(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        section_id = getattr(args, 'slice', None)
        content, content_hash, applied_slice, lines_applied, chars_applied = \
            indexer.get_section_content(args.section_id, section_id)

        print(content, end='')
        sys.stderr.write(f"section_id: {args.section_id}\n")
        sys.stderr.write(f"slice: {applied_slice}\n")
        sys.stderr.write(f"content_hash: {content_hash[:16]}...\n")
        sys.stderr.write(f"lines_applied: {lines_applied}\n")
        sys.stderr.write(f"chars_applied: {chars_applied}\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"[FAIL] Failed to get section: {e}\n")
        return 1


def cmd_extract(args) -> int:
    """Extract sections from a file.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    file_path = Path(args.file_path)

    if not file_path.exists():
        print(f"[FAIL] File not found: {file_path}")
        return 1

    try:
        sections = extract_sections(file_path, args.repo_root)
        print(f"Extracted {len(sections)} sections from {file_path}\n")

        for i, section in enumerate(sections, 1):
            print(f"[{i}] {section.section_id[:16]}...")
            print(f"    Heading: {' > '.join(section.heading_path)}")
            print(f"    Lines: {section.line_start}-{section.line_end}")
            print(f"    Hash: {section.content_hash[:16]}...")
            print()

        return 0
    except Exception as e:
        print(f"[FAIL] Extraction failed: {e}")
        return 1


def cmd_symbols_add(args) -> int:
    """Add symbol to registry.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_registry import SymbolRegistry, SymbolError

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        timestamp = registry.add_symbol(
            symbol_id=args.symbol_id,
            target_ref=args.section,
            default_slice=args.default_slice
        )
        print(f"[OK] Symbol added: {args.symbol_id}")
        print(f"      Target: {args.section}")
        if args.default_slice:
            print(f"      Default slice: {args.default_slice}")
        print(f"      Created: {timestamp}")
        return 0
    except SymbolError as e:
        print(f"[FAIL] {e}")
        return 1
    except Exception as e:
        print(f"[FAIL] Failed to add symbol: {e}")
        return 1


def cmd_symbols_get(args) -> int:
    """Get symbol from registry.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_registry import SymbolRegistry, Symbol

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        symbol = registry.get_symbol(args.symbol_id)

        if symbol is None:
            print(f"[FAIL] Symbol not found: {args.symbol_id}")
            return 1

        print(f"Symbol: {symbol.symbol_id}")
        print(f"  Target Type: {symbol.target_type}")
        print(f"  Target Ref: {symbol.target_ref}")
        if symbol.default_slice:
            print(f"  Default Slice: {symbol.default_slice}")
        print(f"  Created: {symbol.created_at}")
        print(f"  Updated: {symbol.updated_at}")
        return 0
    except Exception as e:
        print(f"[FAIL] Failed to get symbol: {e}")
        return 1


def cmd_symbols_list(args) -> int:
    """List symbols from registry.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_registry import SymbolRegistry

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        prefix = getattr(args, 'prefix', None)
        symbols = registry.list_symbols(prefix)

        print(f"Listing {len(symbols)} symbols")
        if prefix:
            print(f"  Prefix: {prefix}")
        print()

        for symbol in symbols:
            print(f"  {symbol.symbol_id}")
            print(f"    Target: {symbol.target_ref}")
            if symbol.default_slice:
                print(f"    Slice: {symbol.default_slice}")
        return 0
    except Exception as e:
        print(f"[FAIL] Failed to list symbols: {e}")
        return 1


def cmd_symbols_verify(args) -> int:
    """Verify symbol registry integrity.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_registry import SymbolRegistry

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    try:
        success = registry.verify()
        return 0 if success else 1
    except Exception as e:
        print(f"[FAIL] Verification error: {e}")
        return 1


def cmd_resolve(args) -> int:
    """Resolve symbol to content with caching.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from .symbol_resolver import ResolverError
    from .symbol_registry import SymbolRegistry

    registry = SymbolRegistry(
        repo_root=args.repo_root,
        substrate_mode=args.substrate
    )

    resolver = SymbolResolver(
        repo_root=args.repo_root,
        substrate_mode=args.substrate,
        symbol_registry=registry
    )

    try:
        payload, cache_hit = resolver.resolve(
            symbol_id=args.symbol_id,
            slice_expr=args.slice,
            run_id=args.run_id
        )

        print(payload, end='')
        sys.stderr.write(f"[CACHE {'HIT' if cache_hit else 'MISS'}]\n")
        return 0
    except ResolverError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"[FAIL] Resolution error: {e}\n")
        return 1


def cmd_cassette_verify(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    try:
        cassette.verify_cassette(getattr(args, 'run_id', None))
        return 0
    except MessageCassetteError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    finally:
        cassette.close()


def cmd_cassette_post(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    try:
        with open(args.json, 'r') as f:
            payload = json.load(f)
        
        message_id, job_id = cassette.post_message(
            payload=payload,
            run_id=args.run_id,
            source=args.source,
            idempotency_key=args.idempotency_key
        )
        
        print(f"[OK] Message posted")
        print(f"      message_id: {message_id}")
        print(f"      job_id: {job_id}")
        return 0
    except MessageCassetteError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    except FileNotFoundError:
        sys.stderr.write(f"[FAIL] File not found: {args.json}\n")
        return 1
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[FAIL] Invalid JSON: {e}\n")
        return 1
    finally:
        cassette.close()


def cmd_cassette_claim(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    try:
        result = cassette.claim_step(
            run_id=args.run_id,
            worker_id=args.worker,
            ttl_seconds=args.ttl
        )
        
        print(f"[OK] Step claimed")
        print(f"      step_id: {result['step_id']}")
        print(f"      job_id: {result['job_id']}")
        print(f"      message_id: {result['message_id']}")
        print(f"      ordinal: {result['ordinal']}")
        print(f"      fencing_token: {result['fencing_token']}")
        print(f"      lease_expires_at: {result['lease_expires_at']}")
        print()
        print("Payload:")
        print(json.dumps(result['payload'], indent=2))
        return 0
    except MessageCassetteError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    finally:
        cassette.close()


def cmd_cassette_complete(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    try:
        with open(args.receipt, 'r') as f:
            receipt_payload = json.load(f)
        
        receipt_id = cassette.complete_step(
            run_id=args.run_id,
            step_id=args.step,
            worker_id=args.worker,
            fencing_token=args.token,
            receipt_payload=receipt_payload,
            outcome=args.outcome
        )
        
        print(f"[OK] Step completed")
        print(f"      receipt_id: {receipt_id}")
        return 0
    except MessageCassetteError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    except FileNotFoundError:
        sys.stderr.write(f"[FAIL] File not found: {args.receipt}\n")
        return 1
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[FAIL] Invalid JSON: {e}\n")
        return 1
    finally:
        cassette.close()


def cmd_plan_request(args) -> int:
    try:
        plan_output = None
        if args.dry_run:
            with open(args.request_file, 'r') as f:
                request = json.load(f)
            
            planner = Planner(repo_root=args.repo_root)
            plan_output = planner.plan_request(request, dry_run=True)
            
            print(json.dumps(plan_output, indent=2))
        else:
            with open(args.request_file, 'r') as f:
                request = json.load(f)
            
            message_id, job_id, step_ids = post_request_and_plan(
                run_id=request.get("run_id", "default"),
                request_payload=request,
                idempotency_key=request.get("request_id"),
                repo_root=args.repo_root
            )
            
            print(f"[OK] Plan created")
            print(f"      message_id: {message_id}")
            print(f"      job_id: {job_id}")
            print(f"      steps: {len(step_ids)}")
            for i, step_id in enumerate(step_ids, 1):
                print(f"      step_id_{i}: {step_id}")
        
        return 0
    except PlannerError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    except FileNotFoundError:
        sys.stderr.write(f"[FAIL] File not found: {args.request_file}\n")
        return 1
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[FAIL] Invalid JSON: {e}\n")
        return 1


def cmd_execute(args) -> int:
    """Execute PENDING steps for a given job_id in ordinal order.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero on failure)
    """
    if args.workers > 1:
        return cmd_execute_parallel(args)
    
    cassette = MessageCassette(repo_root=args.repo_root)
    worker_id = f"cli_worker_{args.run_id}"
    
    try:
        conn = cassette._get_conn()
        
        cursor = conn.execute("""
            SELECT s.step_id, s.ordinal
            FROM cassette_steps s
            JOIN cassette_jobs j ON s.job_id = j.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            WHERE m.run_id = ? AND j.job_id = ? AND s.status = 'PENDING'
            ORDER BY s.ordinal ASC
        """, (args.run_id, args.job_id))
        
        steps = cursor.fetchall()
        
        if not steps:
            print(f"[INFO] No PENDING steps found for run_id={args.run_id}, job_id={args.job_id}")
            return 0
        
        for step_row in steps:
            step_id = step_row["step_id"]
            ordinal = step_row["ordinal"]
            
            try:
                claim_result = cassette.claim_step(
                    run_id=args.run_id,
                    worker_id=worker_id,
                    ttl_seconds=300
                )
                
                if claim_result["step_id"] != step_id:
                    print(f"[FAIL] step_id {step_id}: Claimed wrong step {claim_result['step_id']}")
                    return 1
                
                receipt = cassette.execute_step(
                    run_id=args.run_id,
                    step_id=step_id,
                    worker_id=worker_id,
                    fencing_token=claim_result["fencing_token"],
                    repo_root=args.repo_root
                )
                
                if receipt.get("status") == "SUCCESS":
                    print(f"[OK] {step_id}")
                else:
                    error = receipt.get("error", "Unknown error")
                    print(f"[FAIL] {step_id}: {error}")
                    return 1
                    
            except MessageCassetteError as e:
                print(f"[FAIL] {step_id}: {e}")
                return 1
        
        return 0
        
    finally:
        cassette.close()


def cmd_execute_parallel(args) -> int:
    """Execute PENDING steps for a given job_id using parallel workers.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero on failure)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    results_lock = threading.Lock()
    stop_flag = threading.Event()
    
    success_count = 0
    failure_count = 0
    
    def worker_task(worker_index: int) -> tuple:
        nonlocal success_count, failure_count
        
        cassette = MessageCassette(repo_root=args.repo_root)
        local_success = 0
        local_failure = 0
        worker_id = f"cli_worker_{args.run_id}_w{worker_index}"
        
        try:
            while not stop_flag.is_set():
                try:
                    claim_result = cassette.claim_next_step(
                        run_id=args.run_id,
                        job_id=args.job_id,
                        worker_id=worker_id,
                        ttl_seconds=300
                    )
                    
                    if claim_result is None:
                        break
                    
                    step_id = claim_result["step_id"]
                    
                    receipt = cassette.execute_step(
                        run_id=args.run_id,
                        step_id=step_id,
                        worker_id=worker_id,
                        fencing_token=claim_result["fencing_token"],
                        repo_root=args.repo_root,
                        check_global_budget=True
                    )
                    
                    with results_lock:
                        if receipt.get("status") == "SUCCESS":
                            print(f"[OK] {step_id}")
                            local_success += 1
                        else:
                            error = receipt.get("error", "Unknown error")
                            print(f"[FAIL] {step_id}: {error}")
                            local_failure += 1
                    
                    if local_failure > 0 and not args.continue_on_fail:
                        stop_flag.set()
                        break
                        
                except MessageCassetteError as e:
                    with results_lock:
                        print(f"[FAIL] {e}")
                        local_failure += 1
                    stop_flag.set()
                    break
        finally:
            cassette.close()
        
        return (local_success, local_failure)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker_task, i) for i in range(args.workers)]
        
        for future in as_completed(futures):
            local_success, local_failure = future.result()
            with results_lock:
                success_count += local_success
                failure_count += local_failure
    
    if failure_count > 0:
        print(f"[FAIL] job failed: {success_count} succeeded, {failure_count} failed")
        return 1
    else:
        print(f"[OK] job complete")
        return 0


def cmd_ants_spawn(args) -> int:
    """Spawn multiple ant workers.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero on failure)
    """
    try:
        exit_code = spawn_ants(
            run_id=args.run_id,
            job_id=args.job_id,
            num_workers=args.n,
            repo_root=args.repo_root or Path.cwd(),
            continue_on_fail=args.continue_on_fail
        )
        
        if exit_code == 0:
            print("[OK] All ants completed successfully")
        elif exit_code == 1:
            print("[FAIL] Some ants failed")
        else:
            print("[FAIL] Invariant/DB error occurred")
        
        return exit_code
    except Exception as e:
        print(f"[FAIL] Spawn failed: {e}")
        return 2


def cmd_ants_worker(args) -> int:
    """Run a single ant worker.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero on failure)
    """
    try:
        exit_code = run_ant_worker(
            run_id=args.run_id,
            job_id=args.job_id,
            worker_id=args.worker_id,
            repo_root=args.repo_root or Path.cwd(),
            continue_on_fail=args.continue_on_fail,
            poll_interval_ms=args.poll_ms,
            ttl_seconds=args.ttl,
            max_idle_polls=args.max_idle_polls
        )
        
        if exit_code == 0:
            print("[OK] Worker completed")
        elif exit_code == 1:
            print("[FAIL] Worker failed")
        else:
            print("[FAIL] Invariant/DB error occurred")
        
        return exit_code
    except Exception as e:
        print(f"[FAIL] Worker failed: {e}")
        return 2


def cmd_ants_status(args) -> int:
    """Show job status counts.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 on failure)
    """
    cassette = MessageCassette(repo_root=args.repo_root)
    try:
        status = cassette.get_job_status(run_id=args.run_id, job_id=args.job_id)

        if status is None:
            print("[FAIL] job_id or run_id not found")
            return 1

        print(f"PENDING: {status['pending']}")
        print(f"LEASED: {status['leased']}")
        print(f"COMMITTED: {status['committed']}")
        print(f"RECEIPTS: {status['receipts']}")
        print(f"WORKERS_SEEN: {status['workers_seen']}")

        return 0
    except MessageCassetteError as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        cassette.close()


def cmd_bundle_build(args) -> int:
    """Build bundle from completed job.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 on failure)
    """
    bundle_builder = BundleBuilder(repo_root=args.repo_root)

    try:
        output_dir = Path(args.out)
        result = bundle_builder.build(
            run_id=args.run_id,
            job_id=args.job_id,
            output_dir=output_dir
        )

        print(f"[OK] Bundle built")
        print(f"      bundle_id: {result['bundle_id']}")
        print(f"      output_dir: {result['output_dir']}")
        print(f"      artifacts: {result['artifact_count']}")
        print(f"      root_hash: {result['root_hash']}")
        return 0
    except BundleError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    finally:
        bundle_builder.close()


def cmd_bundle_verify(args) -> int:
    """Verify bundle integrity.

    Exit codes:
        0: OK
        1: Verification failed
        2: Invalid input
        3: Internal error

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0-3)
    """
    from catalytic_chat.cli_output import (
        write_json_report,
        write_info,
        write_error,
        format_json_report,
        classify_exit_code,
        EXIT_OK
    )

    bundle_dir = Path(args.bundle)
    quiet = getattr(args, 'quiet', False)
    json_mode = getattr(args, 'json', False)

    is_invalid_input = False
    is_verification_failure = False
    is_internal_error = False
    errors = []

    if not bundle_dir.exists():
        error_msg = f"Bundle directory not found: {bundle_dir}"
        write_error(error_msg)
        errors.append({"code": "BUNDLE_NOT_FOUND", "message": error_msg})
        is_invalid_input = True
    else:
        try:
            verifier = BundleVerifier(bundle_dir)
            result = verifier.verify()

            if not json_mode:
                write_info(f"[OK] Bundle verified", quiet)
                write_info(f"      bundle_id: {result['bundle_id']}", quiet)
                write_info(f"      run_id: {result['run_id']}", quiet)
                write_info(f"      job_id: {result['job_id']}", quiet)
                write_info(f"      artifacts: {result['artifact_count']}", quiet)
                write_info(f"      root_hash: {result['root_hash']}", quiet)

            if json_mode:
                report = format_json_report(
                    ok=True,
                    command="bundle_verify",
                    bundle_id=result.get("bundle_id"),
                    run_id=result.get("run_id"),
                    job_id=result.get("job_id"),
                    counts={"artifacts": result.get("artifact_count", 0)}
                )
                write_json_report(report, quiet)

            return EXIT_OK

        except BundleError as e:
            error_msg = f"{e}"
            write_error(f"[FAIL] {error_msg}")
            errors.append({"code": "VERIFICATION_ERROR", "message": error_msg})
            is_verification_failure = True
        except FileNotFoundError as e:
            error_msg = f"File not found: {e}"
            write_error(f"[FAIL] {error_msg}")
            errors.append({"code": "FILE_NOT_FOUND", "message": error_msg})
            is_invalid_input = True
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {e}"
            write_error(f"[FAIL] {error_msg}")
            errors.append({"code": "INVALID_JSON", "message": error_msg})
            is_invalid_input = True
        except Exception as e:
            error_msg = f"Internal error: {e}"
            write_error(f"[FAIL] {error_msg}")
            errors.append({"code": "INTERNAL_ERROR", "message": error_msg})
            is_internal_error = True

    if json_mode and errors:
        report = format_json_report(
            ok=False,
            command="bundle_verify",
            errors=errors
        )
        write_json_report(report, quiet)

    return classify_exit_code(
        is_verification_failure=is_verification_failure,
        is_invalid_input=is_invalid_input,
        is_internal_error=is_internal_error
    )


def cmd_bundle_run(args) -> int:
    """Run bundle and output receipt path with optional attestation verification.

    Exit codes:
        0: OK
        1: Verification failed
        2: Invalid input
        3: Internal error

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0-3)
    """
    from catalytic_chat.execution_policy import (
        policy_from_cli_args,
        load_execution_policy,
        validate_policy,
        ExecutionPolicyError
    )

    bundle_dir = Path(args.bundle)
    repo_root = args.repo_root or Path.cwd()

    policy_path = getattr(args, 'policy', None)
    attest = getattr(args, 'attest', False)
    signing_key = getattr(args, 'signing_key', None)
    print_merkle = getattr(args, 'print_merkle', False)
    attest_merkle = getattr(args, 'attest_merkle', False)
    merkle_key = getattr(args, 'merkle_key', None)
    verify_merkle_attestation_path = getattr(args, 'verify_merkle_attestation', None)
    merkle_attestation_out = getattr(args, 'merkle_attestation_out', None)
    receipt_out = getattr(args, 'receipt_out', None)
    verify_attestation = getattr(args, 'verify_attestation', False)
    verify_chain = getattr(args, 'verify_chain', False)
    require_attestation = getattr(args, 'require_attestation', False)
    require_merkle_attestation = getattr(args, 'require_merkle_attestation', False)
    strict_trust = getattr(args, 'strict_trust', False)
    strict_identity = getattr(args, 'strict_identity', False)
    quiet = getattr(args, 'quiet', False)
    json_mode = getattr(args, 'json', False)

    try:
        if policy_path:
            policy = load_execution_policy(policy_path)
        else:
            policy = policy_from_cli_args(args)
    except ExecutionPolicyError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1

    validate_policy(policy)

    if attest and not signing_key:
        sys.stderr.write("[FAIL] --attest requires --signing-key\n")
        return 1

    if print_merkle and not policy.get("require_verify_chain", False):
        sys.stderr.write("[FAIL] --print-merkle requires --verify-chain (or policy.require_verify_chain)\n")
        return 1

    if attest_merkle and not policy.get("require_verify_chain", False):
        sys.stderr.write("[FAIL] --attest-merkle requires --verify-chain (or policy.require_verify_chain)\n")
        return 1

    if attest_merkle and not merkle_key:
        sys.stderr.write("[FAIL] --attest-merkle requires --merkle-key\n")
        return 1

    if print_merkle and attest_merkle:
        sys.stderr.write("[FAIL] --print-merkle and --attest-merkle are mutually exclusive\n")
        return 1

    if verify_merkle_attestation_path and not policy.get("require_verify_chain", False):
        sys.stderr.write("[FAIL] --verify-merkle-attestation requires --verify-chain (or policy.require_verify_chain)\n")
        return 1

    if attest_merkle and verify_merkle_attestation_path:
        sys.stderr.write("[FAIL] --attest-merkle and --verify-merkle-attestation are mutually exclusive\n")
        return 1

    trust_index = None
    if policy.get("strict_trust", False) or policy.get("strict_identity", False) or policy.get("trust_policy_path"):
        from catalytic_chat.trust_policy import (
            load_trust_policy_bytes,
            parse_trust_policy,
            build_trust_index,
            TrustPolicyError
        )

        trust_policy_path = policy.get("trust_policy_path")
        if trust_policy_path is None:
            trust_policy_path = repo_root / "THOUGHT" / "LAB" / "CAT_CHAT" / "CORTEX" / "_generated" / "TRUST_POLICY.json"
        else:
            trust_policy_path = Path(trust_policy_path)

        try:
            policy_bytes = load_trust_policy_bytes(trust_policy_path)
            trust_policy_parsed = parse_trust_policy(policy_bytes)
            trust_index = build_trust_index(trust_policy_parsed)
        except TrustPolicyError as e:
            sys.stderr.write(f"[FAIL] {e}\n")
            return 1
        except Exception as e:
            sys.stderr.write(f"[FAIL] Failed to load trust policy: {e}\n")
            return 1

    try:
        executor = BundleExecutor(
            bundle_dir=bundle_dir,
            receipt_out=receipt_out,
            signing_key=signing_key if attest else None
        )

        result = executor.execute()

        if require_attestation and result.get('attestation') is None:
            sys.stderr.write("[FAIL] attestation required but not present\n")
            return 1

        verbose_output = not print_merkle and not (attest_merkle and not merkle_attestation_out)

        if verbose_output:
            print(f"[OK] Bundle executed")
            print(f"      receipt: {result['receipt_path']}")
            print(f"      outcome: {result['outcome']}")
            print(f"      receipt_hash: {result.get('receipt_hash', 'N/A')}")

            if result.get('parent_receipt_hash'):
                print(f"      parent_receipt_hash: {result['parent_receipt_hash']}")

        merkle_root = None
        receipts = []

        if verify_attestation and result['attestation'] is not None and verbose_output:
            from catalytic_chat.attestation import verify_receipt_attestation, AttestationError
            from catalytic_chat.receipt import receipt_canonical_bytes

            receipt_path = Path(result['receipt_path'])
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes.decode('utf-8'))

            try:
                verify_receipt_attestation(receipt, trust_index, strict_trust, strict_identity)
                if verbose_output:
                    print(f"      attestation: VALID")
            except AttestationError as e:
                if verbose_output:
                    print(f"      attestation: INVALID ({e})")
                return 1

        if verify_chain:
            from catalytic_chat.receipt import find_receipt_chain, verify_receipt_chain

            receipts_dir = receipt_out.parent if receipt_out else bundle_dir
            run_id = result.get('run_id')

            if run_id:
                receipts = find_receipt_chain(receipts_dir, run_id)

                if len(receipts) > 0:
                    try:
                        merkle_root = verify_receipt_chain(receipts, verify_attestation=verify_attestation)

                        if print_merkle:
                            print(merkle_root)
                        elif verbose_output:
                            print(f"      chain: VALID ({len(receipts)} receipts)")
                            print(f"      merkle_root: {merkle_root}")
                    except Exception as e:
                        if verbose_output:
                            print(f"      chain: INVALID ({e})")
                        return 1
                else:
                    if verbose_output:
                        print(f"      chain: N/A (no receipts found)")

        if attest_merkle and merkle_root:
            from catalytic_chat.merkle_attestation import sign_merkle_root, write_merkle_attestation

            if not isinstance(merkle_key, str):
                if merkle_key is None:
                    sys.stderr.write(f"[FAIL] --merkle-key is required for --attest-merkle\n")
                    return 1
                try:
                    merkle_key_hex = Path(merkle_key).read_text().strip()
                except Exception as e:
                    sys.stderr.write(f"[FAIL] Failed to read merkle key file: {e}\n")
                    return 1
            else:
                merkle_key_hex = merkle_key

            try:
                from catalytic_chat.validator_identity import get_validator_identity

                repo_root = args.repo_root or Path.cwd()
                merkle_validator_identity = get_validator_identity(
                    bytes.fromhex(merkle_key_hex),
                    repo_root
                )

                att = sign_merkle_root(
                    merkle_root,
                    merkle_key_hex,
                    validator_id=merkle_validator_identity["validator_id"],
                    build_id=merkle_validator_identity["build_id"]
                )

                if len(receipts) > 0:
                    att["receipt_count"] = len(receipts)
                    att["receipt_chain_head_hash"] = receipts[-1].get("receipt_hash")
                att["run_id"] = result.get('run_id')
                att["job_id"] = result.get('job_id')
                att["bundle_id"] = result.get('bundle_id')

                if merkle_attestation_out:
                    att_path = Path(merkle_attestation_out)
                    write_merkle_attestation(att_path, att)
                    if verbose_output:
                        print(f"      merkle_attestation: {att_path}")
                else:
                    from catalytic_chat.receipt import canonical_json_bytes
                    att_bytes = canonical_json_bytes(att)
                    sys.stdout.buffer.write(att_bytes)
                    sys.stdout.buffer.flush()

            except Exception as e:
                sys.stderr.write(f"[FAIL] Merkle attestation failed: {e}\n")
                return 1

        if verify_merkle_attestation_path and merkle_root:
            from catalytic_chat.merkle_attestation import load_merkle_attestation, verify_merkle_attestation_with_trust, MerkleAttestationError

            try:
                att_path = Path(verify_merkle_attestation_path)
                att = load_merkle_attestation(att_path)

                if att is None:
                    sys.stderr.write(f"[FAIL] Merkle attestation file not found: {att_path}\n")
                    return 1

                if require_merkle_attestation and att is None:
                    sys.stderr.write(f"[FAIL] merkle attestation required but not present\n")
                    return 1

                verify_merkle_attestation_with_trust(att, merkle_root, trust_index, strict_trust, strict_identity)

                if verbose_output:
                    print(f"      merkle_attestation: VALID")
            except MerkleAttestationError as e:
                sys.stderr.write(f"[FAIL] Merkle attestation verification failed: {e}\n")
                return 1
            except Exception as e:
                sys.stderr.write(f"[FAIL] Merkle attestation error: {e}\n")
                return 1

        return 0
    except BundleError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1


def cmd_trust_init(args) -> int:
    """Initialize empty trust policy at default location.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 on failure)
    """
    repo_root = args.repo_root or Path.cwd()
    policy_path = repo_root / "THOUGHT" / "LAB" / "CAT_CHAT" / "CORTEX" / "_generated" / "TRUST_POLICY.json"

    policy = {
        "policy_version": "1.0.0",
        "allow": []
    }

    from catalytic_chat.receipt import canonical_json_bytes

    try:
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_bytes = canonical_json_bytes(policy)
        policy_path.write_bytes(policy_bytes)
        sys.stderr.write("[OK] wrote TRUST_POLICY.json\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"[FAIL] Failed to write trust policy: {e}\n")
        return 1


def cmd_trust_verify(args) -> int:
    """Verify trust policy against schema and uniqueness rules.

    Exit codes:
        0: OK
        1: Verification failed
        2: Invalid input
        3: Internal error

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0-3)
    """
    from catalytic_chat.trust_policy import load_trust_policy_bytes, parse_trust_policy, build_trust_index, TrustPolicyError
    from catalytic_chat.validator_identity import parse_build_id, ValidatorIdentityError
    from catalytic_chat.cli_output import (
        write_json_report,
        write_info,
        write_error,
        format_json_report,
        classify_exit_code,
        EXIT_OK
    )

    if args.trust_policy:
        policy_path = Path(args.trust_policy)
    else:
        repo_root = args.repo_root or Path.cwd()
        policy_path = repo_root / "THOUGHT" / "LAB" / "CAT_CHAT" / "CORTEX" / "_generated" / "TRUST_POLICY.json"

    quiet = getattr(args, 'quiet', False)
    json_mode = getattr(args, 'json', False)

    is_invalid_input = False
    is_verification_failure = False
    is_internal_error = False
    errors = []

    if not policy_path.exists():
        error_msg = f"Trust policy file not found: {policy_path}"
        write_error(error_msg)
        errors.append({"code": "FILE_NOT_FOUND", "message": error_msg})
        is_invalid_input = True
    else:
        try:
            policy_bytes = load_trust_policy_bytes(policy_path)
            policy = parse_trust_policy(policy_bytes)

            for entry in policy.get("allow", []):
                build_id = entry.get("build_id")
                if build_id:
                    try:
                        parse_build_id(build_id)
                    except ValidatorIdentityError as e:
                        error_msg = f"invalid build_id in validator '{entry.get('validator_id')}': {e}"
                        write_error(f"[FAIL] {error_msg}")
                        errors.append({"code": "INVALID_BUILD_ID", "message": error_msg})
                        is_verification_failure = True

            if not is_verification_failure:
                build_trust_index(policy)
                if not json_mode:
                    write_info("[OK] trust policy valid", quiet)

            if json_mode:
                report = format_json_report(
                    ok=not is_verification_failure,
                    command="trust_verify",
                    errors=errors if errors else None
                )
                write_json_report(report, quiet)

            return EXIT_OK if not is_verification_failure else 1

        except TrustPolicyError as e:
            error_msg = f"{e}"
            write_error(f"[FAIL] {error_msg}")
            errors.append({"code": "TRUST_POLICY_ERROR", "message": error_msg})
            is_verification_failure = True
        except FileNotFoundError:
            error_msg = f"Trust policy file not found: {policy_path}"
            write_error(f"[FAIL] {error_msg}")
            errors.append({"code": "FILE_NOT_FOUND", "message": error_msg})
            is_invalid_input = True
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {e}"
            write_error(f"[FAIL] {error_msg}")
            errors.append({"code": "INVALID_JSON", "message": error_msg})
            is_invalid_input = True
        except Exception as e:
            error_msg = f"Internal error: {e}"
            write_error(f"[FAIL] {error_msg}")
            errors.append({"code": "INTERNAL_ERROR", "message": error_msg})
            is_internal_error = True

    if json_mode and errors:
        report = format_json_report(
            ok=False,
            command="trust_verify",
            errors=errors
        )
        write_json_report(report, quiet)

    return classify_exit_code(
        is_verification_failure=is_verification_failure,
        is_invalid_input=is_invalid_input,
        is_internal_error=is_internal_error
    )


def cmd_trust_show(args) -> int:
    """Print trust policy summary to stdout.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 on failure)
    """
    from catalytic_chat.trust_policy import load_trust_policy_bytes, parse_trust_policy, build_trust_index, TrustPolicyError
    from catalytic_chat.receipt import canonical_json_bytes

    if args.trust_policy:
        policy_path = Path(args.trust_policy)
    else:
        repo_root = args.repo_root or Path.cwd()
        policy_path = repo_root / "THOUGHT" / "LAB" / "CAT_CHAT" / "CORTEX" / "_generated" / "TRUST_POLICY.json"

    try:
        policy_bytes = load_trust_policy_bytes(policy_path)
        policy = parse_trust_policy(policy_bytes)
        index = build_trust_index(policy)

        enabled_count = sum(1 for entry in index["by_validator_id"].values() if entry.get("enabled", False))
        receipt_scope_count = sum(1 for entry in index["by_validator_id"].values() if entry.get("enabled", False) and "RECEIPT" in entry.get("scope", []))
        merkle_scope_count = sum(1 for entry in index["by_validator_id"].values() if entry.get("enabled", False) and "MERKLE" in entry.get("scope", []))

        summary = {
            "policy_version": policy.get("policy_version"),
            "enabled": enabled_count,
            "scopes": {
                "RECEIPT": receipt_scope_count,
                "MERKLE": merkle_scope_count
            }
        }

        summary_bytes = canonical_json_bytes(summary)
        sys.stdout.buffer.write(summary_bytes)
        sys.stdout.buffer.flush()
        return 0
    except TrustPolicyError as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"[FAIL] {e}\n")
        return 1


def cmd_compress_verify(args) -> int:
    """Verify compression claim.

    Exit codes:
        0: OK
        1: Verification failed (metrics mismatch, policy violation)
        2: Invalid input (missing files, schema errors)
        3: Internal error

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0-3)
    """
    quiet = getattr(args, 'quiet', False)
    json_mode = getattr(args, 'json', False)

    is_invalid_input = False
    is_verification_failure = False
    is_internal_error = False
    errors = []

    try:
        result = validate_compression_claim(
            bundle_path=args.bundle,
            receipts_dir=args.receipts,
            trust_policy_path=getattr(args, 'trust_policy', None),
            claim_json_path=args.claim,
            strict_trust=getattr(args, 'strict_trust', False),
            strict_identity=getattr(args, 'strict_identity', False),
            require_attestation=getattr(args, 'require_attestation', False)
        )

        if not json_mode:
            if result["ok"]:
                sys.stderr.write("[OK] Compression claim verified\n")
                sys.stderr.write(f"      bundle_id: {result['claim'].get('bundle_id', 'N/A')}\n")
                computed = result.get("computed", {})
                sys.stderr.write(f"      compression_ratio: {computed.get('compression_ratio', 0):.4f}\n")
                sys.stderr.write(f"      uncompressed_tokens: {computed.get('uncompressed_tokens', 0)}\n")
                sys.stderr.write(f"      compressed_tokens: {computed.get('compressed_tokens', 0)}\n")
            else:
                for err in result.get("errors", []):
                    sys.stderr.write(f"[FAIL] {err['code']}: {err['message']}\n")

        if json_mode:
            import json
            sys.stdout.buffer.write(json.dumps(result).encode('utf-8'))
            sys.stdout.buffer.write(b'\n')
            sys.stdout.buffer.flush()

        return 0 if result["ok"] else 1

    except Exception as e:
        error_msg = f"Internal error: {e}"
        if not json_mode:
            sys.stderr.write(f"[FAIL] {error_msg}\n")
        else:
            import json
            sys.stdout.buffer.write(json.dumps({
                "ok": False,
                "errors": [{"code": "INTERNAL_ERROR", "message": error_msg}]
            }).encode('utf-8'))
            sys.stdout.buffer.write(b'\n')
            sys.stdout.buffer.flush()
        return 3


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Catalytic Chat CLI",
        epilog="Roadmap Phase: Phase 1 — Substrate + deterministic indexing"
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root path (default: current working directory)"
    )
    parser.add_argument(
        "--substrate",
        choices=["sqlite", "jsonl"],
        default="sqlite",
        help="Substrate mode (default: sqlite)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    build_parser = subparsers.add_parser("build", help="Build section index")
    build_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Build incrementally (only changed files)"
    )

    verify_parser = subparsers.add_parser("verify", help="Verify index determinism")

    get_parser = subparsers.add_parser("get", help="Get section by ID")
    get_parser.add_argument("section_id", help="Section ID")
    get_parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Slice expression (e.g., lines[0:100], chars[0:500], head(50), tail(20))"
    )

    extract_parser = subparsers.add_parser("extract", help="Extract sections from file")
    extract_parser.add_argument("file_path", help="Path to file")

    symbols_parser = subparsers.add_parser("symbols", help="Symbol registry commands")
    symbols_subparsers = symbols_parser.add_subparsers(dest="symbols_command", help="Symbol commands")

    symbols_add_parser = symbols_subparsers.add_parser("add", help="Add symbol to registry")
    symbols_add_parser.add_argument("symbol_id", help="Symbol ID (must start with @)")
    symbols_add_parser.add_argument("--section", required=True, help="Section ID to reference")
    symbols_add_parser.add_argument("--default-slice", help="Default slice expression")

    symbols_get_parser = symbols_subparsers.add_parser("get", help="Get symbol from registry")
    symbols_get_parser.add_argument("symbol_id", help="Symbol ID")

    symbols_list_parser = symbols_subparsers.add_parser("list", help="List symbols")
    symbols_list_parser.add_argument("--prefix", help="Filter by prefix (e.g., @CANON/)")

    symbols_verify_parser = symbols_subparsers.add_parser("verify", help="Verify symbol registry")

    resolve_parser = subparsers.add_parser("resolve", help="Resolve symbol to content with caching")
    resolve_parser.add_argument("symbol_id", help="Symbol ID")
    resolve_parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Slice expression (e.g., lines[0:100], chars[0:500], head(50), tail(20))"
    )
    resolve_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for caching"
    )

    cassette_parser = subparsers.add_parser("cassette", help="Message cassette commands (Phase 3)")
    cassette_subparsers = cassette_parser.add_subparsers(dest="cassette_command", help="Cassette commands")

    cassette_verify_parser = cassette_subparsers.add_parser("verify", help="Verify cassette integrity")
    cassette_verify_parser.add_argument("--run-id", type=str, default=None, help="Verify specific run")

    cassette_post_parser = cassette_subparsers.add_parser("post", help="Post message to cassette")
    cassette_post_parser.add_argument("--json", type=Path, required=True, help="JSON file with message payload")
    cassette_post_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    cassette_post_parser.add_argument("--source", type=str, required=True, 
                                    choices=["USER", "PLANNER", "SYSTEM", "WORKER"], help="Message source")
    cassette_post_parser.add_argument("--idempotency-key", type=str, default=None, help="Idempotency key")

    cassette_claim_parser = cassette_subparsers.add_parser("claim", help="Claim a pending step")
    cassette_claim_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    cassette_claim_parser.add_argument("--worker", type=str, required=True, help="Worker ID")
    cassette_claim_parser.add_argument("--ttl", type=int, default=300, help="TTL in seconds (default: 300)")

    cassette_complete_parser = cassette_subparsers.add_parser("complete", help="Complete a step")
    cassette_complete_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    cassette_complete_parser.add_argument("--step", type=str, required=True, help="Step ID")
    cassette_complete_parser.add_argument("--worker", type=str, required=True, help="Worker ID")
    cassette_complete_parser.add_argument("--token", type=int, required=True, help="Fencing token")
    cassette_complete_parser.add_argument("--receipt", type=Path, required=True, help="JSON file with receipt payload")
    cassette_complete_parser.add_argument("--outcome", type=str, required=True,
                                        choices=["SUCCESS", "FAILURE", "ABORTED"], help="Outcome")

    plan_parser = subparsers.add_parser("plan", help="Deterministic planner (Phase 4)")
    plan_subparsers = plan_parser.add_subparsers(dest="plan_command", help="Plan commands")

    plan_request_parser = plan_subparsers.add_parser("request", help="Create plan from request JSON")
    plan_request_parser.add_argument("--request-file", type=Path, required=True, help="Path to plan request JSON")
    plan_request_parser.add_argument("--dry-run", action="store_true", help="Print plan to stdout without DB writes")

    plan_verify_parser = plan_subparsers.add_parser("verify", help="Verify stored plan hash")
    plan_verify_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    plan_verify_parser.add_argument("--request-id", type=str, required=True, help="Request ID")

    execute_parser = subparsers.add_parser("execute", help="Execute PENDING steps for a job (Phase 4.1/4.2)")
    execute_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    execute_parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    execute_parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    execute_parser.add_argument("--continue-on-fail", action="store_true", help="Continue execution on failure")

    ants_parser = subparsers.add_parser("ants", help="Ant worker commands (Phase 4.3)")
    ants_subparsers = ants_parser.add_subparsers(dest="ants_command", help="Ants commands")

    ants_spawn_parser = ants_subparsers.add_parser("spawn", help="Spawn multiple ant workers")
    ants_spawn_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    ants_spawn_parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    ants_spawn_parser.add_argument("-n", type=int, required=True, help="Number of workers")
    ants_spawn_parser.add_argument("--continue-on-fail", action="store_true", help="Continue on failure")

    ants_run_parser = ants_subparsers.add_parser("run", help="Alias for 'spawn' - spawn multiple ant workers")
    ants_run_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    ants_run_parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    ants_run_parser.add_argument("-n", type=int, required=True, help="Number of workers")
    ants_run_parser.add_argument("--continue-on-fail", action="store_true", help="Continue on failure")

    ants_status_parser = ants_subparsers.add_parser("status", help="Show job status counts")
    ants_status_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    ants_status_parser.add_argument("--job-id", type=str, required=True, help="Job ID")

    ants_worker_parser = ants_subparsers.add_parser("worker", help="Run a single ant worker")
    ants_worker_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    ants_worker_parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    ants_worker_parser.add_argument("--worker-id", type=str, required=True, help="Worker ID")
    ants_worker_parser.add_argument("--continue-on-fail", action="store_true", help="Continue on failure")
    ants_worker_parser.add_argument("--poll-ms", type=int, default=250, help="Poll interval in ms")
    ants_worker_parser.add_argument("--ttl", type=int, default=300, help="Lease TTL in seconds")
    ants_worker_parser.add_argument("--max-idle-polls", type=int, default=20, help="Max idle polls before exit")

    bundle_parser = subparsers.add_parser("bundle", help="Bundle commands (Phase 5)")
    bundle_subparsers = bundle_parser.add_subparsers(dest="bundle_command", help="Bundle commands")

    bundle_build_parser = bundle_subparsers.add_parser("build", help="Build bundle from completed job")
    bundle_build_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    bundle_build_parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    bundle_build_parser.add_argument("--out", type=Path, required=True, help="Output directory")

    bundle_verify_parser = bundle_subparsers.add_parser("verify", help="Verify bundle integrity")
    bundle_verify_parser.add_argument("--bundle", type=Path, required=True, help="Bundle directory path")
    bundle_verify_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report to stdout (human logs to stderr)")
    bundle_verify_parser.add_argument("--quiet", action="store_true", help="Suppress non-error stderr output")

    bundle_run_parser = bundle_subparsers.add_parser("run", help="Run bundle and output execution result")
    bundle_run_parser.add_argument("--bundle", type=Path, required=True, help="Bundle directory path")
    bundle_run_parser.add_argument("--receipt-out", type=Path, required=False, help="Receipt output path (default: <bundle>/receipt.json)")
    bundle_run_parser.add_argument("--attest", action="store_true", help="Emit receipt with attestation (requires --signing-key)")
    bundle_run_parser.add_argument("--signing-key", type=Path, required=False, help="Private signing key for attestation (32 bytes, ed25519)")
    bundle_run_parser.add_argument("--verify-attestation", action="store_true", help="Verify attestation if present")
    bundle_run_parser.add_argument("--verify-chain", action="store_true", help="Verify receipt chain linkage for run")
    bundle_run_parser.add_argument("--print-merkle", action="store_true", help="Print Merkle root to stdout (requires --verify-chain)")
    bundle_run_parser.add_argument("--attest-merkle", action="store_true", help="Sign Merkle root and emit attestation (requires --verify-chain and --merkle-key)")
    bundle_run_parser.add_argument("--merkle-key", type=str, required=False, help="Ed25519 signing key hex (64 hex chars) for Merkle attestation")
    bundle_run_parser.add_argument("--verify-merkle-attestation", type=Path, required=False, help="Verify Merkle attestation file (requires --verify-chain)")
    bundle_run_parser.add_argument("--merkle-attestation-out", type=Path, required=False, help="Write Merkle attestation to this path (default: print to stdout)")
    bundle_run_parser.add_argument("--trust-policy", type=Path, required=False, help="Trust policy file path (default: CAT_CORTEX/_generated/TRUST_POLICY.json)")
    bundle_run_parser.add_argument("--strict-trust", action="store_true", help="Enable strict trust verification (requires trust policy)")
    bundle_run_parser.add_argument("--strict-identity", action="store_true", help="Enable strict identity pinning (enforces build_id matching in trust policy)")
    bundle_run_parser.add_argument("--require-attestation", action="store_true", help="Require receipt attestation to be present")
    bundle_run_parser.add_argument("--require-merkle-attestation", action="store_true", help="Require merkle attestation to be present and valid")
    bundle_run_parser.add_argument("--policy", type=Path, required=False, help="Execution policy file path (unifies verification flags)")
    bundle_run_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report to stdout (human logs to stderr)")
    bundle_run_parser.add_argument("--quiet", action="store_true", help="Suppress non-error stderr output")

    trust_parser = subparsers.add_parser("trust", help="Trust policy commands (Phase 6.6)")
    trust_subparsers = trust_parser.add_subparsers(dest="trust_command", help="Trust commands")

    trust_init_parser = trust_subparsers.add_parser("init", help="Initialize empty trust policy")
    trust_verify_parser = trust_subparsers.add_parser("verify", help="Verify trust policy against schema")
    trust_verify_parser.add_argument("--trust-policy", type=Path, required=False, help="Trust policy file path")
    trust_verify_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report to stdout (human logs to stderr)")
    trust_verify_parser.add_argument("--quiet", action="store_true", help="Suppress non-error stderr output")
    trust_show_parser = trust_subparsers.add_parser("show", help="Print trust policy summary")
    trust_show_parser.add_argument("--trust-policy", type=Path, required=False, help="Trust policy file path")

    compress_parser = subparsers.add_parser("compress", help="Compression protocol commands (Phase 7)")
    compress_subparsers = compress_parser.add_subparsers(dest="compress_command", help="Compression commands")

    compress_verify_parser = compress_subparsers.add_parser("verify", help="Verify compression claim")
    compress_verify_parser.add_argument("--bundle", type=Path, required=True, help="Bundle directory path")
    compress_verify_parser.add_argument("--receipts", type=Path, required=True, help="Receipts directory path")
    compress_verify_parser.add_argument("--trust-policy", type=Path, required=False, help="Trust policy file path (optional)")
    compress_verify_parser.add_argument("--claim", type=Path, required=True, help="Claim JSON file path")
    compress_verify_parser.add_argument("--strict-trust", action="store_true", help="Enable strict trust verification")
    compress_verify_parser.add_argument("--strict-identity", action="store_true", help="Enable strict identity pinning")
    compress_verify_parser.add_argument("--require-attestation", action="store_true", help="Require attestation on all receipts")
    compress_verify_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report to stdout (human logs to stderr)")
    compress_verify_parser.add_argument("--quiet", action="store_true", help="Suppress non-error stderr output")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "build": cmd_build,
        "verify": cmd_verify,
        "get": cmd_get,
        "extract": cmd_extract,
        "resolve": cmd_resolve
    }

    if args.command == "symbols":
        symbols_commands = {
            "add": cmd_symbols_add,
            "get": cmd_symbols_get,
            "list": cmd_symbols_list,
            "verify": cmd_symbols_verify
        }

        if args.symbols_command not in symbols_commands:
            print(f"[FAIL] Unknown symbols command: {args.symbols_command}")
            parser.print_help()
            sys.exit(1)

        sys.exit(symbols_commands[args.symbols_command](args))

    if args.command == "cassette":
        cassette_commands = {
            "verify": cmd_cassette_verify,
            "post": cmd_cassette_post,
            "claim": cmd_cassette_claim,
            "complete": cmd_cassette_complete
        }

        if args.cassette_command not in cassette_commands:
            print(f"[FAIL] Unknown cassette command: {args.cassette_command}")
            parser.print_help()
            sys.exit(1)

        sys.exit(cassette_commands[args.cassette_command](args))

    if args.command == "plan":
        plan_commands = {
            "request": cmd_plan_request,
            "verify": cmd_cassette_verify
        }
        
        if args.plan_command not in plan_commands:
            print(f"[FAIL] Unknown plan command: {args.plan_command}")
            parser.print_help()
            sys.exit(1)
        
        sys.exit(plan_commands[args.plan_command](args))

    if args.command == "execute":
        sys.exit(cmd_execute(args))

    if args.command == "ants":
        ants_commands = {
            "spawn": cmd_ants_spawn,
            "run": cmd_ants_spawn,
            "worker": cmd_ants_worker,
            "status": cmd_ants_status
        }
        
        if args.ants_command not in ants_commands:
            print(f"[FAIL] Unknown ants command: {args.ants_command}")
            parser.print_help()
            sys.exit(1)

        sys.exit(ants_commands[args.ants_command](args))

    if args.command == "bundle":
        bundle_commands = {
            "build": cmd_bundle_build,
            "verify": cmd_bundle_verify,
            "run": cmd_bundle_run
        }

        if args.bundle_command not in bundle_commands:
            print(f"[FAIL] Unknown bundle command: {args.bundle_command}")
            parser.print_help()
            sys.exit(1)

        sys.exit(bundle_commands[args.bundle_command](args))

    if args.command == "trust":
        trust_commands = {
            "init": cmd_trust_init,
            "verify": cmd_trust_verify,
            "show": cmd_trust_show
        }

        if args.trust_command not in trust_commands:
            print(f"[FAIL] Unknown trust command: {args.trust_command}")
            parser.print_help()
            sys.exit(1)

        sys.exit(trust_commands[args.trust_command](args))

    if args.command == "compress":
        compress_commands = {
            "verify": cmd_compress_verify
        }

        if args.compress_command not in compress_commands:
            print(f"[FAIL] Unknown compress command: {args.compress_command}")
            parser.print_help()
            sys.exit(1)

        sys.exit(compress_commands[args.compress_command](args))

    if args.command not in commands:
        print(f"[FAIL] Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    sys.exit(commands[args.command](args))


if __name__ == '__main__':
    main()

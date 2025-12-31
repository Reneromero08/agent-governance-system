#!/usr/bin/env python3
"""
Patch file to add cmd_execute function to cli.py
"""

import re

# Read the file
with open('catalytic_chat/cli.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find execute_parser and insert cmd_execute before plan_parser
execute_parser_pattern = r'(execute_parser = subparsers\.add_parser\("execute")'
match = re.search(execute_parser_pattern, content)
if not match:
    print("Could not find execute_parser")
    exit(0)

insert_point = match.start()

# Build the execute_parser and cmd_execute function
execute_parser_code = '''    execute_parser = subparsers.add_parser("execute", help="Execute plan steps (Phase 4.1)")
    execute_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    execute_parser.add_argument("--job-id", type=str, required=True, help="Job ID")

def cmd_execute(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    
    try:
        print(f"[INFO] Executing plan: run_id={args.run_id}, job_id={args.job_id}")
        
        step_count = 0
        success_count = 0
        
        while True:
            try:
                result = cassette.claim_step(
                    run_id=args.run_id,
                    worker_id="system",
                    ttl_seconds=300
                )
                
                step_count += 1
                print(f"[STEP {step_count}] Claimed: {result['step_id']}")
                
                receipt = cassette.execute_step(
                    run_id=args.run_id,
                    step_id=result["step_id"],
                    worker_id="system",
                    fencing_token=result["fencing_token"],
                    repo_root=args.repo_root
                )
                
                if receipt.get("status") == "SUCCESS":
                    success_count += 1
                    print(f"[STEP {step_count}] SUCCESS")
                    
                    if "section_id" in receipt:
                        print(f"      Section: {receipt['section_id']}")
                        if "slice" in receipt:
                            print(f"      Slice: {receipt['slice']}")
                        if "content_hash" in receipt:
                            print(f"      Hash: {receipt['content_hash'][:16]}...")
                        if "bytes_read" in receipt:
                            print(f"      Bytes: {receipt['bytes_read']}")
                    elif "symbol_id" in receipt:
                        print(f"      Symbol: {receipt['symbol_id']}")
                        if "section_id" in receipt:
                            print(f"      Section: {receipt['section_id']}")
                        if "slice" in receipt:
                            print(f"      Slice: {receipt['slice']}")
                else:
                    print(f"[STEP {step_count}] FAILED: {receipt.get('error', 'Unknown error')}")
                    return 1
                    
            except MessageCassetteError as e:
                if "No pending steps" in str(e):
                    print(f"[DONE] No more pending steps")
                    print(f"[SUMMARY] {step_count} steps processed, {success_count} succeeded")
                    return 0
                else:
                    print(f"[FAIL] {e}")
                    return 1
                    
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cassette.close()

'''

# Insert the new code
lines_before = content[:insert_point]
lines_after = content[insert_point:]

new_content = lines_before + execute_parser_code + "\n\n" + '''
def cmd_execute(args) -> int:
    cassette = MessageCassette(repo_root=args.repo_root)
    
    try:
        print(f"[INFO] Executing plan: run_id={args.run_id}, job_id={args.job_id}")
        
        step_count = 0
        success_count = 0
        
        while True:
            try:
                result = cassette.claim_step(
                    run_id=args.run_id,
                    worker_id="system",
                    ttl_seconds=300
                )
                
                step_count += 1
                print(f"[STEP {step_count}] Claimed: {result['step_id']}")
                
                receipt = cassette.execute_step(
                    run_id=args.run_id,
                    step_id=result["step_id"],
                    worker_id="system",
                    fencing_token=result["fencing_token"],
                    repo_root=args.repo_root
                )
                
                if receipt.get("status") == "SUCCESS":
                    success_count += 1
                    print(f"[STEP {step_count}] SUCCESS")
                    
                    if "section_id" in receipt:
                        print(f"      Section: {receipt['section_id']}")
                        if "slice" in receipt:
                            print(f"      Slice: {receipt['slice']}")
                        if "content_hash" in receipt:
                            print(f"      Hash: {receipt['content_hash'][:16]}...")
                        if "bytes_read" in receipt:
                            print(f"      Bytes: {receipt['bytes_read']}")
                    elif "symbol_id" in receipt:
                        print(f"      Symbol: {receipt['symbol_id']}")
                        if "section_id" in receipt:
                            print(f"      Section: {receipt['section_id']}")
                        if "slice" in receipt:
                            print(f"      Slice: {receipt['slice']}")
                else:
                    print(f"[STEP {step_count}] FAILED: {receipt.get('error', 'Unknown error')}")
                    return 1
                    
            except MessageCassetteError as e:
                if "No pending steps" in str(e):
                    print(f"[DONE] No more pending steps")
                    print(f"[SUMMARY] {step_count} steps processed, {success_count} succeeded")
                    return 0
                else:
                    print(f"[FAIL] {e}")
                    return 1
                    
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cassette.close()

''' + lines_after

# Write the patched file
with open('catalytic_chat/cli.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Patched successfully")

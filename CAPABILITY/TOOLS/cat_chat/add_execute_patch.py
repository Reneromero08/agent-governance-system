#!/usr/bin/env python3
"""
Patch file to add cmd_execute function to cli.py
"""

import re

# Read the file
with open('catalytic_chat/cli.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where execute_parser should be added (before plan_parser)
if 'execute_parser = subparsers.add_parser("execute"' in content:
    print("execute_parser already exists")
    exit(0)

# Find where to insert execute_parser (before plan_parser = ...)
insert_pattern = r'(plan_parser = subparsers\.add_parser\("plan")'
match = re.search(insert_pattern, content)
if not match:
    print("Could not find insertion point")
    exit(1)

insert_point = match.start()

# Find the line number for insertion
lines_before = content[:insert_point]
lines_after = content[insert_point:]

# Build the execute_parser definition
execute_parser_code = '''    execute_parser = subparsers.add_parser("execute", help="Execute plan steps (Phase 4.1)")
    execute_parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    execute_parser.add_argument("--job-id", type=str, required=True, help="Job ID")

'''

# Build the cmd_execute function
cmd_execute_code = '''

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

# Add execute command handling in the args.command == "cassette" block
# We need to add it before the plan command check

# Find where plan command handling starts and add execute before it
plan_command_pattern = r'(if args\.command == "plan":)'
plan_match = re.search(plan_command_pattern, content)
if not plan_match:
    print("Could not find plan command handler")
    exit(1)

plan_start = plan_match.start()

# Add cmd_execute call before plan command handler
new_main_code = lines_before[:plan_start]

# Add the execute command handler
new_main_code += '''
    if args.command == "execute":
        return cmd_execute(args)
    
'''

# Add the rest after the execute handler
new_main_code += lines_after[plan_start:]

# Write the patched file
with open('catalytic_chat/cli.py', 'w', encoding='utf-8') as f:
    f.write(new_main_code)

print("Patched successfully")

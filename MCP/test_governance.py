#!/usr/bin/env python3
"""Quick test for the new governance tools."""
import json
import sys
sys.path.insert(0, ".")

from MCP.server import AGSMCPServer

server = AGSMCPServer()

def _run_tool_test(name, args={}):
    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args}
    })
    result = response["result"]
    content = result["content"][0]["text"]
    is_error = result.get("isError", False)
    return content, is_error

print("=" * 60)
print("GOVERNANCE TOOLS TEST")
print("=" * 60)

# Test critic_run
print("\n--- critic_run ---")
content, error = _run_tool_test("critic_run")
try:
    result = json.loads(content)
    print(f"✓ Passed: {result['passed']}")
except:
    print(f"Output: {content[:200]}")

# Test commit_ceremony
print("\n--- commit_ceremony ---")
content, error = _run_tool_test("commit_ceremony")
try:
    result = json.loads(content)
    print(f"✓ Critic passed: {result['checklist']['1_failsafe_critic']['passed']}")
    print(f"✓ Runner passed: {result['checklist']['2_failsafe_runner']['passed']}")
    print(f"✓ Staged files: {result['staged_count']}")
    print(f"✓ Ready: {result['checklist']['4_ready_for_commit']}")
except:
    print(f"Output: {content[:200]}")

# Test adr_create (don't actually create one in test)
print("\n--- adr_create (validation only) ---")
content, error = _run_tool_test("adr_create", {"title": ""})  # Should fail
if error:
    print("✓ Correctly rejects empty title")
else:
    print("✗ Should have rejected empty title")

print("\n" + "=" * 60)
print("ALL GOVERNANCE TESTS COMPLETED")
print("=" * 60)

#!/usr/bin/env python3
"""
Built-in self test for the AGS MCP server (the --test mode body).

Extracted from server.py main(). Exercises initialize, tools/list,
resources/list, a representative set of tool calls, resources/read, and
prompts/get against an in-process server (no stdio transport).

For the CI-gating smoke test with drift guards, see verify_governance.py.
"""

import json

from .server import AGSMCPServer


def run_selftest() -> None:
    """Run sample requests for all implemented protocol surfaces."""
    server = AGSMCPServer()

    print("=" * 60)
    print("AGS MCP SERVER TEST")
    print("=" * 60)

    # Initialize
    init_response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    })
    print("\n[OK] Initialize:", init_response["result"]["serverInfo"])

    # List tools
    tools_response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    })
    tool_names = [t["name"] for t in tools_response["result"]["tools"]]
    print(f"\n[OK] Tools available: {tool_names}")

    # List resources
    resources_response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "resources/list",
        "params": {}
    })
    print(f"\n[OK] Resources available: {len(resources_response['result']['resources'])} resources")

    # Test session_info
    print("\n--- Testing session_info ---")
    session_response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "session_info", "arguments": {}}
    })
    content = session_response["result"]["content"][0]["text"]
    is_error = session_response["result"].get("isError", False)
    print(f"[OK] session_info(): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
    if not is_error and content:
        try:
            results = json.loads(content)
            print(f"  Session: {results.get('session_id', 'unknown')}")
        except Exception as e:
            print(f"  Error parsing JSON: {e}, Output: {content[:100]}...")

    # Test context_search
    print("\n--- Testing context_search ---")
    context_response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "context_search", "arguments": {"type": "decisions"}}
    })
    content = context_response["result"]["content"][0]["text"]
    is_error = context_response["result"].get("isError", False)
    print(f"[OK] context_search(type='decisions'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
    if not is_error and content:
        try:
            results = json.loads(content)
            print(f"  Found {len(results)} records")
        except Exception as e:
            print(f"  Error parsing JSON: {e}, Output: {content[:100]}...")

    # Test context_review
    print("\n--- Testing context_review ---")
    review_response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"name": "context_review", "arguments": {"days": 30}}
    })
    content = review_response["result"]["content"][0]["text"]
    is_error = review_response["result"].get("isError", False)
    print(f"[OK] context_review(days=30): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
    if not is_error and content:
        try:
            results = json.loads(content)
            overdue = len(results.get("overdue", []))
            upcoming = len(results.get("upcoming", []))
            print(f"  Overdue: {overdue}, Upcoming: {upcoming}")
        except Exception as e:
            print(f"  Error parsing JSON: {e}, Output: {content[:100]}...")

    # Test canon_read
    print("\n--- Testing canon_read ---")
    canon_response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "canon_read", "arguments": {"file": "CONTRACT"}}
    })
    content = canon_response["result"]["content"][0]["text"]
    is_error = canon_response["result"].get("isError", False)
    print(f"[OK] canon_read('CONTRACT'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")

    # Test resource reading
    print("\n--- Testing resources/read ---")
    read_response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "resources/read",
        "params": {"uri": "ags://canon/genesis"}
    })
    content = read_response["result"]["contents"][0]["text"]
    print(f"[OK] resources/read('ags://canon/genesis'): {len(content)} chars")

    # Test prompts/get
    print("\n--- Testing prompts/get ---")
    prompt_response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 9,
        "method": "prompts/get",
        "params": {"name": "genesis"}
    })
    messages = prompt_response["result"].get("messages", [])
    print(f"[OK] prompts/get('genesis'): {len(messages)} messages")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

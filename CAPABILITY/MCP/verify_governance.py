#!/usr/bin/env python3
"""Quick smoke test for MCP server tools and governance guards.

Exercises registered tools directly (no stdio transport). Exits non-zero
on failure so it can be used as a CI step or pre-commit check.
"""
import json
import sys
from pathlib import Path

# Add Root to path for imports
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.MCP.server import AGSMCPServer
from CAPABILITY.MCP.primitives import clamp_limit, MAX_RESULTS_PER_PAGE

server = AGSMCPServer()
failures = []
MCP_DIR = PROJECT_ROOT / "CAPABILITY" / "MCP"


def _run_tool_test(name, args=None):
    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args or {}}
    })
    result = response["result"]
    content = result["content"][0]["text"]
    is_error = result.get("isError", False)
    return content, is_error


def _check(label, ok):
    print(f"{'[OK]' if ok else '[FAIL]'} {label}")
    if not ok:
        failures.append(label)


print("=" * 60)
print("MCP SERVER GOVERNANCE SMOKE TEST")
print("=" * 60)

# canon_read: valid file returns content
content, error = _run_tool_test("canon_read", {"file": "CONTRACT"})
_check("canon_read('CONTRACT') returns content", not error and len(content) > 0)

# canon_read: traversal attempts must be rejected
for evil in ["../CONTEXT/X", "..\\..\\AGENTS", "CANON/../../README"]:
    content, error = _run_tool_test("canon_read", {"file": evil})
    _check(f"canon_read({evil!r}) rejected", error)

# context_search: returns parseable JSON
content, error = _run_tool_test("context_search", {"type": "decisions"})
try:
    json.loads(content)
    parsed = True
except (json.JSONDecodeError, TypeError):
    parsed = False
_check("context_search(type='decisions') returns JSON", not error and parsed)

# session_info: returns session metadata
content, error = _run_tool_test("session_info")
try:
    info = json.loads(content)
    has_session = bool(info.get("session_id"))
except (json.JSONDecodeError, TypeError):
    has_session = False
_check("session_info() returns session_id", not error and has_session)

# unknown tool: must error, not crash
response = server.handle_request({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "no_such_tool", "arguments": {}}
})
_check("unknown tool returns error response",
       response["result"].get("isError", False))

# --- Tool registry drift (schema <-> handlers <-> docs) ---
tools_schema = json.loads(
    (MCP_DIR / "schemas" / "tools.json").read_text(encoding="utf-8"))
definitions = tools_schema.get("definitions", {})
schema_names = {d.get("name") for d in definitions.values()}
_check("tools.json definitions keys match their name fields",
       all(k == d.get("name") for k, d in definitions.items()))
_check("tools.json names == server.tool_handlers keys",
       schema_names == set(server.tool_handlers.keys()))
readme = (MCP_DIR / "README.md").read_text(encoding="utf-8")
for name in sorted(server.tool_handlers):
    _check(f"README.md documents tool: {name}", f"`{name}`" in readme)

# --- Limit clamping ---
_check("clamp_limit caps huge values",
       clamp_limit(10**9) == MAX_RESULTS_PER_PAGE)
_check("clamp_limit defaults bad input", clamp_limit("junk", default=7) == 7)
_check("clamp_limit floors at 1", clamp_limit(-5) == 1)

# --- canon_read enum drift (schema <-> canon.json) ---
canon_index = json.loads(
    (PROJECT_ROOT / "LAW" / "CANON" / "canon.json").read_text(encoding="utf-8"))
canon_stems = {
    Path(rel).stem.upper()
    for bucket in canon_index.get("buckets", {}).values()
    for rel in bucket.get("files", [])
}
enum_entries = set(
    definitions["canon_read"]["inputSchema"]["properties"]["file"]["enum"])
_check("canon_read enum == canon.json stems", enum_entries == canon_stems)
for entry in sorted(enum_entries):
    _check(f"canon enum entry resolves: {entry}",
           server._resolve_canon_file(entry) is not None)

# hyphenated stem passes the name regex end-to-end
content, error = _run_tool_test("canon_read", {"file": "SPECTRUM-02_RESUME_BUNDLE"})
_check("canon_read accepts hyphenated stem", not error and len(content) > 0)

# --- context_review shape (implemented, no longer a stub) ---
content, error = _run_tool_test("context_review", {"days": 30})
try:
    review = json.loads(content)
    shape_ok = all(k in review for k in ("checked_days", "overdue", "upcoming"))
except (json.JSONDecodeError, TypeError):
    shape_ok = False
_check("context_review returns expected JSON shape", not error and shape_ok)

print("=" * 60)
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    sys.exit(1)
print("ALL GOVERNANCE CHECKS PASSED")

#!/usr/bin/env python3
"""
MCP stdio transport framing.

Extracted from server.py. Pure functions for reading and writing JSON-RPC
messages in either Content-Length framed mode (VS Code, Antigravity, most
MCP clients) or newline-delimited JSON mode (simpler clients). The mode is
auto-detected from the first message and then pinned for the session.
"""

import json
from typing import Dict, Optional, Tuple


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes from a buffered binary stream."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("EOF while reading message body")
        buf.extend(chunk)
    return bytes(buf)


def _read_message(stdin, mode: Optional[str]) -> Tuple[Optional[Dict], Optional[str]]:
    """Read one MCP message in either framed (Content-Length) or JSONL mode.

    Returns: (request_dict_or_none, detected_mode)
    """
    if mode == "jsonl":
        line = stdin.readline()
        if not line:
            return None, None
        # Skip blank lines
        while line in (b"\r\n", b"\n"):
            line = stdin.readline()
            if not line:
                return None, None
        return json.loads(line.decode("utf-8", errors="replace")), "jsonl"

    # framed or auto-detect
    first = stdin.readline()
    if not first:
        return None, None

    # Skip blank lines
    while first in (b"\r\n", b"\n"):
        first = stdin.readline()
        if not first:
            return None, None

    if not first.lower().startswith(b"content-length:"):
        # Auto-detect fallback: treat as JSONL.
        if mode == "framed":
            raise ValueError("Expected Content-Length header, got JSON line")
        return json.loads(first.decode("utf-8", errors="replace")), "jsonl"

    # Framed: read headers until blank line
    headers = [first]
    while True:
        line = stdin.readline()
        if not line:
            raise EOFError("EOF while reading headers")
        if line in (b"\r\n", b"\n"):
            break
        headers.append(line)

    content_length: Optional[int] = None
    for h in headers:
        if h.lower().startswith(b"content-length:"):
            content_length = int(h.split(b":", 1)[1].strip())
            break
    if content_length is None:
        raise ValueError("Missing Content-Length header")

    body = _read_exact(stdin, content_length)
    return json.loads(body.decode("utf-8", errors="replace")), "framed"


def _write_framed_json(stdout, message: Dict) -> None:
    body = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    stdout.write(header)
    stdout.write(body)
    stdout.flush()

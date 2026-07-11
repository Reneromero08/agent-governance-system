#!/usr/bin/env python3
"""Deterministically pack explicitly selected task context to a token budget."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, List

import tiktoken

MAX_CONTEXT_FILE_BYTES = 2 * 1024 * 1024
MAX_CONTEXT_TOTAL_BYTES = 8 * 1024 * 1024
MAX_CONTEXT_TOKEN_BUDGET = 131_072


def _inside_any(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def pack_context(
    workspace: str,
    read_roots: Iterable[str],
    context_files: Iterable[str],
    context_texts: Iterable[str],
    token_budget: int,
    tokenizer: str,
) -> tuple[str, dict[str, Any]]:
    files = list(context_files)
    texts = list(context_texts)
    if not files and not texts:
        return "", {
            "tokenizer": tokenizer,
            "token_budget": token_budget,
            "included_tokens": 0,
            "sources": [],
        }
    if token_budget < 1 or token_budget > MAX_CONTEXT_TOKEN_BUDGET:
        raise ValueError(f"context token budget must be 1-{MAX_CONTEXT_TOKEN_BUDGET}")
    try:
        encoding = tiktoken.get_encoding(tokenizer)
    except ValueError as exc:
        raise ValueError(f"unknown tiktoken encoding: {tokenizer}") from exc

    ws = Path(workspace).resolve()
    roots = [Path(root).resolve() for root in read_roots]
    sources: List[tuple[str, str, str]] = []
    total_bytes = 0
    for raw_path in files:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = ws / candidate
        candidate = candidate.resolve()
        if not _inside_any(candidate, roots):
            raise ValueError(f"context file escapes read scope: {raw_path}")
        if not candidate.is_file():
            raise ValueError(f"context file does not exist: {candidate}")
        size = candidate.stat().st_size
        if size > MAX_CONTEXT_FILE_BYTES:
            raise ValueError(f"context file exceeds {MAX_CONTEXT_FILE_BYTES} bytes: {candidate}")
        total_bytes += size
        if total_bytes > MAX_CONTEXT_TOTAL_BYTES:
            raise ValueError(f"context files exceed {MAX_CONTEXT_TOTAL_BYTES} total bytes")
        raw = candidate.read_bytes()
        sources.append((
            f"file:{candidate}",
            raw.decode("utf-8", errors="replace"),
            hashlib.sha256(raw).hexdigest(),
        ))
    for index, text in enumerate(texts, 1):
        total_bytes += len(text.encode("utf-8"))
        if total_bytes > MAX_CONTEXT_TOTAL_BYTES:
            raise ValueError(f"manual context exceeds {MAX_CONTEXT_TOTAL_BYTES} total bytes")
        sources.append((
            f"text:{index}",
            text,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ))

    remaining = token_budget
    parts: List[str] = []
    manifest_sources: List[dict[str, Any]] = []
    for label, content, source_hash in sources:
        header = f"[CONTEXT SOURCE: {label}]\n"
        header_tokens = encoding.encode(header)
        content_tokens = encoding.encode(content)
        original_tokens = len(header_tokens) + len(content_tokens)
        included_tokens = min(original_tokens, remaining)
        if included_tokens > 0:
            combined = header_tokens + content_tokens
            parts.append(encoding.decode(combined[:included_tokens]))
        manifest_sources.append({
            "source": label,
            "sha256": source_hash,
            "original_tokens": original_tokens,
            "included_tokens": included_tokens,
            "truncated": included_tokens < original_tokens,
        })
        remaining -= included_tokens

    included = token_budget - remaining
    return "\n\n".join(parts), {
        "tokenizer": tokenizer,
        "token_budget": token_budget,
        "included_tokens": included,
        "sources": manifest_sources,
    }

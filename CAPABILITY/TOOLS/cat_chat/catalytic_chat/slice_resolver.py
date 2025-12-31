#!/usr/bin/env python3
"""
Slice Resolver

Parses and applies slice expressions to section content with fail-closed validation.

Roadmap Phase: Phase 1 — Substrate + deterministic indexing
"""

import re
import hashlib
from typing import Tuple
from dataclasses import dataclass


@dataclass
class SliceResult:
    """Result of applying a slice to content."""
    content: str
    content_hash: str
    slice_expr: str
    lines_applied: int
    chars_applied: int


class SliceError(Exception):
    """Slice parsing or application error."""
    pass


class SliceResolver:
    """Resolves and applies slice expressions to content."""

    PATTERN_LINES = re.compile(r'^lines\[(\d+):(\d+)\]$')
    PATTERN_CHARS = re.compile(r'^chars\[(\d+):(\d+)\]$')
    PATTERN_HEAD = re.compile(r'^head\((\d+)\)$')
    PATTERN_TAIL = re.compile(r'^tail\((\d+)\)$')

    def __init__(self):
        """Initialize slice resolver."""

    def parse_slice(self, slice_expr: str, content_len_lines: int, content_len_chars: int) -> Tuple[str, int, int]:
        """Parse slice expression into normalized form and bounds.

        Args:
            slice_expr: Slice expression (e.g., "lines[0:100]", "head(50)")
            content_len_lines: Number of lines in content
            content_len_chars: Number of characters in content

        Returns:
            Tuple of (normalized_expr, start_idx, end_idx)

        Raises:
            SliceError: If slice is malformed or out of bounds
        """
        slice_expr = slice_expr.strip()

        if not slice_expr:
            raise SliceError("Slice expression is empty")

        if slice_expr.lower() == "all":
            raise SliceError("slice=ALL is forbidden (unbounded expansion)")

        match_lines = self.PATTERN_LINES.match(slice_expr)
        if match_lines:
            a = int(match_lines.group(1))
            b = int(match_lines.group(2))

            if a < 0 or b < 0:
                raise SliceError(f"Negative indices forbidden: lines[{a}:{b}]")
            if a >= b:
                raise SliceError(f"Start index must be less than end index: lines[{a}:{b}]")
            if b > content_len_lines:
                raise SliceError(f"Line index {b} exceeds content length {content_len_lines}")

            return ("lines", a, b)

        match_chars = self.PATTERN_CHARS.match(slice_expr)
        if match_chars:
            a = int(match_chars.group(1))
            b = int(match_chars.group(2))

            if a < 0 or b < 0:
                raise SliceError(f"Negative indices forbidden: chars[{a}:{b}]")
            if a >= b:
                raise SliceError(f"Start index must be less than end index: chars[{a}:{b}]")
            if b > content_len_chars:
                raise SliceError(f"Char index {b} exceeds content length {content_len_chars}")

            return ("chars", a, b)

        match_head = self.PATTERN_HEAD.match(slice_expr)
        if match_head:
            n = int(match_head.group(1))

            if n < 0:
                raise SliceError(f"Negative count forbidden: head({n})")
            if n == 0:
                raise SliceError(f"head(0) results in empty slice")

            if n > content_len_chars:
                n = content_len_chars

            return ("head", 0, n)

        match_tail = self.PATTERN_TAIL.match(slice_expr)
        if match_tail:
            n = int(match_tail.group(1))

            if n < 0:
                raise SliceError(f"Negative count forbidden: tail({n})")
            if n == 0:
                raise SliceError(f"tail(0) results in empty slice")

            if n > content_len_chars:
                n = content_len_chars

            start = content_len_chars - n
            return ("tail", start, content_len_chars)

        raise SliceError(f"Malformed slice expression: {slice_expr}")

    def apply_slice(self, content: str, slice_expr: str) -> SliceResult:
        """Apply slice expression to content and return result.

        Args:
            content: Original content (with normalized line endings)
            slice_expr: Slice expression

        Returns:
            SliceResult with content, hash, metadata

        Raises:
            SliceError: If slice is invalid or out of bounds
        """
        content_len_lines = content.count('\n') + 1
        content_len_chars = len(content)

        slice_type, start_idx, end_idx = self.parse_slice(
            slice_expr,
            content_len_lines,
            content_len_chars
        )

        if slice_type == "lines":
            lines = content.split('\n')
            sliced_lines = lines[start_idx:end_idx]
            result = '\n'.join(sliced_lines)
        elif slice_type == "chars":
            result = content[start_idx:end_idx]
        elif slice_type == "head":
            result = content[:end_idx]
        elif slice_type == "tail":
            result = content[start_idx:]
        else:
            raise SliceError(f"Unknown slice type: {slice_type}")

        result_hash = hashlib.sha256(result.encode('utf-8')).hexdigest()
        result_lines = result.count('\n') + 1 if result else 0
        result_chars = len(result)

        return SliceResult(
            content=result,
            content_hash=result_hash,
            slice_expr=slice_expr,
            lines_applied=result_lines,
            chars_applied=result_chars
        )

    def validate_content_hash(self, content: str, expected_hash: str) -> None:
        """Validate that content matches expected hash.

        Args:
            content: Content to validate
            expected_hash: Expected SHA-256 hash

        Raises:
            SliceError: If hash does not match
        """
        content = content.replace('\r\n', '\n')
        actual_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        if actual_hash != expected_hash:
            raise SliceError(
                f"Content hash mismatch: expected {expected_hash[:16]}..., "
                f"got {actual_hash[:16]}..."
            )


def resolve_slice(content: str, slice_expr: str) -> SliceResult:
    """Convenience function to resolve a slice on content.

    Args:
        content: Content to slice
        slice_expr: Slice expression

    Returns:
        SliceResult

    Raises:
        SliceError: If slice is invalid
    """
    resolver = SliceResolver()
    return resolver.apply_slice(content, slice_expr)


if __name__ == '__main__':
    import sys

    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

    resolver = SliceResolver()

    tests = [
        ("lines[0:2]", "First 2 lines"),
        ("lines[1:3]", "Lines 2-3"),
        ("chars[0:10]", "First 10 chars"),
        ("head(20)", "First 20 chars"),
        ("tail(10)", "Last 10 chars"),
    ]

    for slice_expr, description in tests:
        print(f"\nTest: {description}")
        print(f"  Slice: {slice_expr}")
        try:
            result = resolver.apply_slice(test_content, slice_expr)
            print(f"  Result: {repr(result.content)}")
            print(f"  Hash: {result.content_hash[:16]}...")
            print(f"  Lines: {result.lines_applied}, Chars: {result.chars_applied}")
        except SliceError as e:
            print(f"  ERROR: {e}")

    error_tests = ["lines[5:10]", "chars[-1:10]", "head(0)", "tail(0)", "ALL"]
    print(f"\n\nError tests (should fail):")
    for slice_expr in error_tests:
        print(f"\nTest: {slice_expr}")
        try:
            result = resolver.apply_slice(test_content, slice_expr)
            print(f"  UNEXPECTED SUCCESS: {result.content[:20]}...")
        except SliceError as e:
            print(f"  Expected error: {e}")

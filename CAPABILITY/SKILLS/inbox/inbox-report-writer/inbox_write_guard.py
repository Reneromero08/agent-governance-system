#!/usr/bin/env python3
"""
Runtime interceptor for INBOX writes.
Blocks unhashed writes to INBOX/ at the tool/API level.

This module provides a decorator and context manager to enforce
INBOX hash integrity at runtime, preventing tools from writing
unhashed files to the INBOX directory.
"""
import functools
import hashlib
import re
from pathlib import Path
from typing import Callable, Any


class InboxWriteError(Exception):
    """Raised when attempting to write an unhashed file to INBOX."""
    pass


def compute_content_hash(content: str) -> str:
    """
    Compute SHA256 hash of content, excluding any existing CONTENT_HASH comment.
    
    Args:
        content: The file content
        
    Returns:
        Hex-encoded SHA256 hash
    """
    # Remove any existing CONTENT_HASH comments for hash computation
    cleaned_content = re.sub(
        r'<!--\s*CONTENT_HASH:\s*[a-f0-9]+\s*-->\n*',
        '',
        content,
        flags=re.IGNORECASE
    )
    
    # Compute hash of cleaned content
    return hashlib.sha256(cleaned_content.encode('utf-8')).hexdigest()


def has_valid_hash(content: str) -> tuple[bool, str | None]:
    """
    Check if content has a valid CONTENT_HASH comment.
    
    Args:
        content: The file content
        
    Returns:
        Tuple of (has_valid_hash, hash_value)
    """
    # Extract stored hash
    hash_pattern = r'<!--\s*CONTENT_HASH:\s*([a-f0-9]+)\s*-->'
    match = re.search(hash_pattern, content, re.IGNORECASE)
    
    if not match:
        return False, None
    
    stored_hash = match.group(1)
    computed_hash = compute_content_hash(content)
    
    return stored_hash == computed_hash, stored_hash


def validate_inbox_write(filepath: Path | str, content: str) -> None:
    """
    Validate that a write to INBOX has a valid hash.
    
    Args:
        filepath: Path to the file being written
        content: Content being written
        
    Raises:
        InboxWriteError: If the file is in INBOX and lacks a valid hash
    """
    filepath = Path(filepath)
    
    # Check if this is an INBOX file
    try:
        # Normalize path to check if it's under INBOX/
        parts = filepath.parts
        if 'INBOX' in parts and filepath.suffix == '.md':
            # This is an INBOX markdown file, validate hash
            valid, hash_value = has_valid_hash(content)
            
            if not valid:
                computed_hash = compute_content_hash(content)
                raise InboxWriteError(
                    f"Attempted to write unhashed file to INBOX: {filepath}\n"
                    f"All INBOX/*.md files must have a valid CONTENT_HASH comment.\n"
                    f"Computed hash: {computed_hash}\n"
                    f"To fix, add this line after frontmatter:\n"
                    f"<!-- CONTENT_HASH: {computed_hash} -->\n"
                    f"Or use: python CAPABILITY/SKILLS/inbox/inbox-report-writer/hash_inbox_file.py update {filepath}"
                )
    except ValueError:
        # Path is not relative to current directory, allow
        pass


def inbox_write_guard(func: Callable) -> Callable:
    """
    Decorator to guard file write operations to INBOX.
    
    Usage:
        @inbox_write_guard
        def write_file(path, content):
            Path(path).write_text(content)
    
    Args:
        func: Function that writes files (should accept path and content args)
        
    Returns:
        Wrapped function that validates INBOX writes
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Try to extract filepath and content from args
        # This is a best-effort approach
        if len(args) >= 2:
            filepath = args[0]
            content = args[1]
            
            # Handle bytes content (decode for validation)
            if isinstance(content, bytes):
                try:
                    content_str = content.decode('utf-8')
                except UnicodeDecodeError:
                    # If it's not valid UTF-8, it's not a markdown file we care about
                    return func(*args, **kwargs)
            else:
                content_str = content
            
            # Validate before write
            validate_inbox_write(filepath, content_str)
        
        # Proceed with original function
        return func(*args, **kwargs)
    
    return wrapper


class InboxWriteGuard:
    """
    Context manager to guard INBOX writes.
    
    Usage:
        with InboxWriteGuard():
            # Any INBOX writes here will be validated
            # Example: Path.cwd().joinpath("INBOX", "reports", "my_report.md").write_text(content)
    """
    
    def __init__(self):
        self.original_write_text = None
        self.original_write_bytes = None
    
    def __enter__(self):
        """Patch Path.write_text and Path.write_bytes."""
        self.original_write_text = Path.write_text
        self.original_write_bytes = Path.write_bytes
        
        def guarded_write_text(self, data, encoding=None, errors=None, newline=None):
            # Validate INBOX writes
            if encoding is None:
                encoding = 'utf-8'
            validate_inbox_write(self, data)
            return InboxWriteGuard.original_write_text_static(
                self, data, encoding=encoding, errors=errors, newline=newline
            )
        
        # Store static reference
        InboxWriteGuard.original_write_text_static = self.original_write_text
        
        # Monkey patch
        Path.write_text = guarded_write_text
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original Path methods."""
        if self.original_write_text:
            Path.write_text = self.original_write_text
        if self.original_write_bytes:
            Path.write_bytes = self.original_write_bytes
        
        return False


# Convenience function for manual validation
def check_inbox_file(filepath: Path | str) -> tuple[bool, str]:
    """
    Check if an INBOX file has a valid hash.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Tuple of (is_valid, message)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return False, f"File not found: {filepath}"
    
    content = filepath.read_text(encoding='utf-8')
    valid, hash_value = has_valid_hash(content)
    
    if valid:
        return True, f"✅ Valid hash: {hash_value}"
    else:
        computed_hash = compute_content_hash(content)
        if hash_value:
            return False, f"❌ Hash mismatch. Stored: {hash_value}, Computed: {computed_hash}"
        else:
            return False, f"❌ Missing hash. Computed: {computed_hash}"

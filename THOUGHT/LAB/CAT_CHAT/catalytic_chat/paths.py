#!/usr/bin/env python3
"""
Canonical Path Helpers

Single source of truth for all artifact paths in CAT_CHAT sandbox.
Ensures consistency across all modules and prevents path mismatches.

Structure:
    THOUGHT/LAB/CAT_CHAT/
        _generated/
            cat_chat.db     # Single consolidated database
            bundles/        # Bundle outputs

On graduation, _generated/cat_chat.db moves to NAVIGATION/CORTEX/cassettes/cat_chat.db
"""

import sqlite3
from pathlib import Path
from typing import Optional


def get_generated_dir(repo_root: Optional[Path] = None) -> Path:
    """Get _generated directory for CAT_CHAT sandbox.

    Args:
        repo_root: Repository root path. Defaults to current working directory.

    Returns:
        Path to THOUGHT/LAB/CAT_CHAT/_generated
    """
    if repo_root is None:
        repo_root = Path.cwd()

    generated_dir = repo_root / "THOUGHT" / "LAB" / "CAT_CHAT" / "_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    return generated_dir


def get_cat_chat_db(repo_root: Optional[Path] = None) -> Path:
    """Get path to the consolidated cat_chat.db.

    All tables (index, cassette, session) are in this single database.

    Args:
        repo_root: Repository root path. Defaults to current working directory.

    Returns:
        Path to cat_chat.db
    """
    generated_dir = get_generated_dir(repo_root)
    db_path = generated_dir / "cat_chat.db"
    return db_path


# Legacy aliases - all point to consolidated DB
def get_cortex_dir(repo_root: Optional[Path] = None) -> Path:
    """Legacy alias for get_generated_dir."""
    return get_generated_dir(repo_root)


def get_db_path(repo_root: Optional[Path] = None, name: str = "cat_chat.db") -> Path:
    """Get path to database file. Now always returns cat_chat.db."""
    return get_cat_chat_db(repo_root)


def get_system1_db(repo_root: Optional[Path] = None) -> Path:
    """Legacy alias - now points to consolidated cat_chat.db."""
    return get_cat_chat_db(repo_root)


def get_system3_db(repo_root: Optional[Path] = None) -> Path:
    """Legacy alias - now points to consolidated cat_chat.db."""
    return get_cat_chat_db(repo_root)


def get_sqlite_connection(db_path: Path) -> sqlite3.Connection:
    """Get SQLite connection with standard settings.
    
    Args:
        db_path: Path to database file
    
    Returns:
        SQLite connection with foreign_keys and WAL enabled
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn

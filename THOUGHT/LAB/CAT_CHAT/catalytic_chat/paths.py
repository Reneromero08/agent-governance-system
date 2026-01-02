#!/usr/bin/env python3
"""
Canonical Path Helpers

Single source of truth for all artifact paths in CAT_CORTEX substrate.
Ensures consistency across all modules and prevents path mismatches.
"""

import sqlite3
from pathlib import Path
from typing import Optional


def get_cortex_dir(repo_root: Optional[Path] = None) -> Path:
    """Get CAT_CORTEX/_generated directory.
    
    Args:
        repo_root: Repository root path. Defaults to current working directory.
    
    Returns:
        Path to CAT_CORTEX/_generated
    """
    if repo_root is None:
        repo_root = Path.cwd()
    
    cortex_dir = repo_root / "THOUGHT" / "LAB" / "CAT_CHAT" / "CAT_CORTEX" / "_generated"
    cortex_dir.mkdir(parents=True, exist_ok=True)
    return cortex_dir


def get_db_path(repo_root: Optional[Path] = None, name: str = "system1.db") -> Path:
    """Get path to a database file in CAT_CORTEX/_generated.
    
    Args:
        repo_root: Repository root path. Defaults to current working directory.
        name: Database filename (e.g., "system1.db", "system3.db")
    
    Returns:
        Path to database file
    """
    cortex_dir = get_cortex_dir(repo_root)
    db_path = cortex_dir / name
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_system1_db(repo_root: Optional[Path] = None) -> Path:
    """Get path to system1.db (sections, symbols, expansion_cache).
    
    Args:
        repo_root: Repository root path. Defaults to current working directory.
    
    Returns:
        Path to system1.db
    """
    return get_db_path(repo_root, "system1.db")


def get_system3_db(repo_root: Optional[Path] = None) -> Path:
    """Get path to system3.db (cassette_* tables).
    
    Args:
        repo_root: Repository root path. Defaults to current working directory.
    
    Returns:
        Path to system3.db
    """
    return get_db_path(repo_root, "system3.db")


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

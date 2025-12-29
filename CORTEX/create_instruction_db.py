"""
Create Instruction Database for Tiny Models

Analyzes codebase_full.db and generates refactoring tasks
that tiny models can execute using hash references.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime


class InstructionDBCreator:
    """Create instruction database from mechanical index."""

    def __init__(self, codebase_db: str = "CORTEX/codebase_full.db",
                 instruction_db: str = "CORTEX/instructions.db"):
        self.codebase_conn = sqlite3.connect(codebase_db)
        self.instruction_conn = sqlite3.connect(instruction_db)
        self._create_schema()

    def _create_schema(self):
        """Create instruction database schema."""
        self.instruction_conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,  -- 'refactor', 'add_feature', 'fix_bug'
                target_hash TEXT NOT NULL,  -- @hash reference to codebase
                target_path TEXT,  -- Human-readable path
                instruction TEXT NOT NULL,
                context TEXT,  -- Additional context (JSON)
                priority INTEGER DEFAULT 1,
                status TEXT DEFAULT 'pending',  -- 'pending', 'in_progress', 'done', 'failed'
                created_at TEXT,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS analysis_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,  -- 'missing_error_handling', 'long_function', etc.
                description TEXT,
                files_affected TEXT,  -- JSON array of hashes
                detected_at TEXT
            );

            CREATE TABLE IF NOT EXISTS task_results (
                task_id TEXT PRIMARY KEY,
                original_hash TEXT,
                modified_hash TEXT,
                diff TEXT,  -- Code diff
                success BOOLEAN,
                error_message TEXT,
                completed_at TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks(task_id)
            );

            CREATE INDEX IF NOT EXISTS idx_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority DESC);
        """)
        self.instruction_conn.commit()

    def analyze_and_create_tasks(self):
        """Analyze codebase and generate refactoring tasks."""
        tasks_created = 0

        # Pattern 1: Python files without error handling
        print("[1/4] Analyzing error handling patterns...")
        tasks_created += self._create_error_handling_tasks()

        # Pattern 2: Long functions (>50 lines)
        print("[2/4] Analyzing function length...")
        tasks_created += self._create_function_length_tasks()

        # Pattern 3: Missing docstrings
        print("[3/4] Analyzing documentation coverage...")
        tasks_created += self._create_docstring_tasks()

        # Pattern 4: Duplicate code detection (simple version)
        print("[4/4] Detecting potential duplicates...")
        tasks_created += self._create_deduplication_tasks()

        self.instruction_conn.commit()
        return tasks_created

    def _create_error_handling_tasks(self) -> int:
        """Create tasks for adding error handling."""
        cursor = self.codebase_conn.execute("""
            SELECT f.hash, f.path, pm.functions
            FROM files f
            JOIN python_metadata pm ON f.hash = pm.file_hash
            WHERE f.content NOT LIKE '%try:%'
            AND f.content NOT LIKE '%except%'
            AND json_array_length(pm.functions) > 0
            LIMIT 20  -- Limit for demo
        """)

        count = 0
        for row in cursor:
            file_hash, path, functions_json = row
            functions = json.loads(functions_json)

            if not functions:
                continue

            task_id = f"error_handling_{file_hash[:8]}"
            self.instruction_conn.execute("""
                INSERT OR IGNORE INTO tasks
                (task_id, task_type, target_hash, target_path, instruction, context, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                "add_error_handling",
                f"@hash:{file_hash}",
                path,
                f"Add try/except error handling to functions: {', '.join(functions[:3])}",
                json.dumps({"functions": functions, "focus": "file I/O and external calls"}),
                2,
                datetime.now().isoformat()
            ))
            count += 1

        return count

    def _create_function_length_tasks(self) -> int:
        """Create tasks for refactoring long functions."""
        cursor = self.codebase_conn.execute("""
            SELECT hash, path, content
            FROM files
            WHERE extension = '.py'
            AND line_count > 100
            LIMIT 10
        """)

        count = 0
        for row in cursor:
            file_hash, path, content = row

            # Check if file has long functions (simple heuristic)
            if "def " in content:
                task_id = f"refactor_long_{file_hash[:8]}"
                self.instruction_conn.execute("""
                    INSERT OR IGNORE INTO tasks
                    (task_id, task_type, target_hash, target_path, instruction, priority, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_id,
                    "refactor_function_length",
                    f"@hash:{file_hash}",
                    path,
                    "Identify and refactor functions longer than 50 lines into smaller, focused functions",
                    1,
                    datetime.now().isoformat()
                ))
                count += 1

        return count

    def _create_docstring_tasks(self) -> int:
        """Create tasks for adding missing docstrings."""
        cursor = self.codebase_conn.execute("""
            SELECT f.hash, f.path, pm.functions, pm.classes, pm.docstring
            FROM files f
            JOIN python_metadata pm ON f.hash = pm.file_hash
            WHERE (pm.docstring IS NULL OR pm.docstring = '')
            AND (json_array_length(pm.functions) > 0 OR json_array_length(pm.classes) > 0)
            LIMIT 15
        """)

        count = 0
        for row in cursor:
            file_hash, path, functions_json, classes_json, docstring = row
            functions = json.loads(functions_json)
            classes = json.loads(classes_json)

            if not functions and not classes:
                continue

            task_id = f"add_docstrings_{file_hash[:8]}"
            self.instruction_conn.execute("""
                INSERT OR IGNORE INTO tasks
                (task_id, task_type, target_hash, target_path, instruction, context, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                "add_documentation",
                f"@hash:{file_hash}",
                path,
                "Add docstrings to functions and classes",
                json.dumps({"functions": functions, "classes": classes}),
                3,
                datetime.now().isoformat()
            ))
            count += 1

        return count

    def _create_deduplication_tasks(self) -> int:
        """Create tasks for identifying potential duplicate code."""
        # Simple approach: find files with similar names
        cursor = self.codebase_conn.execute("""
            SELECT hash, path, size
            FROM files
            WHERE extension = '.py'
            ORDER BY size DESC
            LIMIT 5
        """)

        count = 0
        for row in cursor:
            file_hash, path, size = row
            task_id = f"analyze_duplicates_{file_hash[:8]}"
            self.instruction_conn.execute("""
                INSERT OR IGNORE INTO tasks
                (task_id, task_type, target_hash, target_path, instruction, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                "analyze_duplication",
                f"@hash:{file_hash}",
                path,
                "Analyze for duplicate code blocks and suggest extraction into shared utilities",
                4,
                datetime.now().isoformat()
            ))
            count += 1

        return count

    def get_task_queue(self, limit: int = 10) -> list:
        """Get pending tasks sorted by priority."""
        cursor = self.instruction_conn.execute("""
            SELECT task_id, task_type, target_hash, target_path, instruction, priority
            FROM tasks
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
        """, (limit,))

        tasks = []
        for row in cursor:
            tasks.append({
                "task_id": row[0],
                "task_type": row[1],
                "target_hash": row[2],
                "target_path": row[3],
                "instruction": row[4],
                "priority": row[5]
            })
        return tasks

    def resolve_hash(self, hash_ref: str) -> str:
        """Resolve @hash reference to actual content."""
        if hash_ref.startswith("@hash:"):
            file_hash = hash_ref[6:]
        else:
            file_hash = hash_ref

        cursor = self.codebase_conn.execute(
            "SELECT content FROM files WHERE hash = ?", (file_hash,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def close(self):
        """Close database connections."""
        self.codebase_conn.close()
        self.instruction_conn.close()


def main():
    """Create instruction database."""
    print("=" * 60)
    print("INSTRUCTION DATABASE CREATOR")
    print("=" * 60)
    print()

    creator = InstructionDBCreator()

    print("Analyzing codebase patterns...")
    print()
    tasks_created = creator.analyze_and_create_tasks()

    print()
    print(f"Created {tasks_created} refactoring tasks")
    print()

    # Show task queue
    print("Top 10 tasks for tiny model:")
    print("-" * 60)
    tasks = creator.get_task_queue(10)
    for i, task in enumerate(tasks, 1):
        print(f"{i}. [{task['task_type']}] (Priority {task['priority']})")
        print(f"   Target: {task['target_hash']}")
        print(f"   File: {Path(task['target_path']).name}")
        print(f"   Task: {task['instruction']}")
        print()

    print("-" * 60)
    print(f"Database: CORTEX/instructions.db")
    print(f"Size: {Path('CORTEX/instructions.db').stat().st_size / 1024:.1f} KB")

    creator.close()

    print()
    print("=" * 60)
    print("INSTRUCTION DB READY FOR TINY MODELS")
    print("=" * 60)


if __name__ == "__main__":
    main()

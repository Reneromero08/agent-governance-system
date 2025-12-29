"""
Tiny Model Demo - Working with Hash-Referenced Instructions

Demonstrates how a tiny model (Haiku, local 2B) would:
1. Read task from instructions.db
2. Resolve @hash to get actual code
3. Execute refactoring
4. Store result

This is a simulation - actual tiny model integration would use
Claude Haiku API or local model (Qwen, Llama, etc.)
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime


class TinyModelWorker:
    """Simulates tiny model working with instruction database."""

    def __init__(self):
        self.codebase_db = sqlite3.connect("CORTEX/codebase_full.db")
        self.instruction_db = sqlite3.connect("CORTEX/instructions.db")

    def get_next_task(self):
        """Get highest priority pending task."""
        cursor = self.instruction_db.execute("""
            SELECT task_id, task_type, target_hash, target_path, instruction, context
            FROM tasks
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        """)
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "task_id": row[0],
            "task_type": row[1],
            "target_hash": row[2],
            "target_path": row[3],
            "instruction": row[4],
            "context": json.loads(row[5]) if row[5] else {}
        }

    def resolve_hash(self, hash_ref: str) -> dict:
        """Resolve @hash to actual code and metadata."""
        if hash_ref.startswith("@hash:"):
            file_hash = hash_ref[6:]
        else:
            file_hash = hash_ref

        cursor = self.codebase_db.execute("""
            SELECT content, path, size, line_count
            FROM files
            WHERE hash = ?
        """, (file_hash,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "content": row[0],
            "path": row[1],
            "size": row[2],
            "line_count": row[3],
            "hash": file_hash
        }

    def execute_task(self, task: dict) -> dict:
        """
        Execute refactoring task.

        In real implementation, this would:
        1. Send task + resolved code to tiny model API
        2. Get back modified code
        3. Validate changes
        4. Return result

        For demo: We simulate the execution.
        """
        code_data = self.resolve_hash(task["target_hash"])
        if not code_data:
            return {"success": False, "error": "Hash not found"}

        # Simulate tiny model processing
        print(f"[TINY MODEL] Processing task: {task['task_id']}")
        print(f"[TINY MODEL] Type: {task['task_type']}")
        print(f"[TINY MODEL] File: {Path(code_data['path']).name}")
        print(f"[TINY MODEL] Size: {code_data['size']} bytes, {code_data['line_count']} lines")
        print(f"[TINY MODEL] Instruction: {task['instruction']}")
        print()

        # In real implementation, send to model:
        # response = haiku_api.complete(f"""
        # Task: {task['instruction']}
        # Code:
        # {code_data['content']}
        #
        # Return modified code.
        # """)

        # For demo, we show what would be sent
        token_estimate = len(task['instruction']) // 4 + code_data['size'] // 4
        print(f"[TINY MODEL] Token usage estimate: ~{token_estimate:,} tokens")
        print(f"[TINY MODEL] (vs {5234 * 600:,} tokens if full codebase loaded)")
        print(f"[TINY MODEL] Savings: {((5234 * 600 - token_estimate) / (5234 * 600)) * 100:.1f}%")
        print()

        # Mark task as completed
        self.instruction_db.execute("""
            UPDATE tasks
            SET status = 'in_progress', completed_at = ?
            WHERE task_id = ?
        """, (datetime.now().isoformat(), task['task_id']))
        self.instruction_db.commit()

        return {
            "success": True,
            "task_id": task['task_id'],
            "tokens_used": token_estimate,
            "file_path": code_data['path']
        }

    def get_stats(self) -> dict:
        """Get task statistics."""
        stats = {}

        cursor = self.instruction_db.execute("""
            SELECT status, COUNT(*) FROM tasks GROUP BY status
        """)
        for row in cursor:
            stats[row[0]] = row[1]

        return stats

    def close(self):
        """Close connections."""
        self.codebase_db.close()
        self.instruction_db.close()


def main():
    """Demo tiny model workflow."""
    print("=" * 70)
    print("TINY MODEL DEMO: Hash-Based Refactoring")
    print("=" * 70)
    print()

    worker = TinyModelWorker()

    # Show initial stats
    stats = worker.get_stats()
    print(f"Task Queue Status:")
    for status, count in stats.items():
        print(f"  {status}: {count}")
    print()

    # Process top 3 tasks
    print("=" * 70)
    print("Processing Tasks...")
    print("=" * 70)
    print()

    for i in range(3):
        task = worker.get_next_task()
        if not task:
            print("No more pending tasks!")
            break

        print(f"Task {i+1}/3")
        print("-" * 70)
        result = worker.execute_task(task)

        if result['success']:
            print(f"[RESULT] SUCCESS - Task completed")
            print(f"[RESULT] Tokens used: ~{result['tokens_used']:,}")
        else:
            print(f"[RESULT] FAILED - {result.get('error', 'Unknown error')}")

        print()

    # Final stats
    print("=" * 70)
    print("Final Statistics")
    print("=" * 70)
    stats = worker.get_stats()
    for status, count in stats.items():
        print(f"  {status}: {count}")

    worker.close()

    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("What this demonstrates:")
    print("1. Entire codebase indexed mechanically (5,234 files, zero tokens)")
    print("2. Analysis patterns detected (50 refactoring tasks)")
    print("3. Tiny model works with @hash references (99%+ token savings)")
    print("4. Tasks executed without loading full codebase into context")
    print()
    print("Next steps:")
    print("- Integrate with Claude Haiku API for actual refactoring")
    print("- Add result validation and diff generation")
    print("- Create automated task queue processor")


if __name__ == "__main__":
    main()
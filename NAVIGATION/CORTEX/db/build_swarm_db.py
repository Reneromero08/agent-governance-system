#!/usr/bin/env python3
"""
Swarm Instructions Database Builder

Creates a SQLite database with:
1. Indexed codebase structure (6-bucket architecture)
2. Swarm task templates for ant workers
3. Skill registry integration
"""

import sqlite3
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add GuardedWriter for write firewall enforcement
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "swarm_instructions.db"

# 6-Bucket Architecture
BUCKETS = {
    "LAW": "Governance rules, contracts, schemas, context decisions",
    "CAPABILITY": "Skills, tools, primitives, pipelines, MCP integrations",
    "NAVIGATION": "CORTEX indexes, semantic maps, navigation aids",
    "DIRECTION": "Roadmaps, strategic planning documents",
    "THOUGHT": "LAB experiments, research, demos",
    "MEMORY": "Archives, LLM packer outputs, historical data"
}

class SwarmInstructionsDB:
    """Database for swarm task instructions and codebase index."""
    
    def __init__(self, db_path: Path = DB_PATH, writer: Optional[GuardedWriter] = None):
        self.db_path = db_path
        self.writer = writer
        
        # Use GuardedWriter for mkdir if available, otherwise fallback
        if self.writer is None:
            # Instantiate writer if not provided, with expanded roots
            self.writer = GuardedWriter(
                project_root=REPO_ROOT,
                durable_roots=[
                    "LAW/CONTRACTS/_runs",
                    "NAVIGATION/CORTEX/_generated",
                    "NAVIGATION/CORTEX/db"
                ]
            )

        # Use GuardedWriter for mkdir - must be durable for the DB
        self.writer.mkdir_durable("NAVIGATION/CORTEX/db")
        # Open commit gate immediately as this is a build script
        self.writer.open_commit_gate()
            
        self.conn = sqlite3.connect(str(db_path), timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            -- Metadata
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            
            -- Buckets (6-bucket architecture)
            CREATE TABLE IF NOT EXISTS buckets (
                bucket_id TEXT PRIMARY KEY,
                description TEXT,
                indexed_at TIMESTAMP
            );
            
            -- Files index
            CREATE TABLE IF NOT EXISTS files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                bucket_id TEXT NOT NULL,
                relative_path TEXT UNIQUE NOT NULL,
                file_type TEXT,
                size_bytes INTEGER,
                content_hash TEXT,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bucket_id) REFERENCES buckets(bucket_id)
            );
            
            -- Swarm task templates
            CREATE TABLE IF NOT EXISTS task_templates (
                task_id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                description TEXT,
                target_bucket TEXT,
                input_schema TEXT,
                output_schema TEXT,
                difficulty TEXT CHECK(difficulty IN ('trivial', 'easy', 'medium', 'hard')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Skill mappings (which skills can execute which tasks)
            CREATE TABLE IF NOT EXISTS skill_mappings (
                skill_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                capability_level TEXT,
                PRIMARY KEY (skill_id, task_id),
                FOREIGN KEY (task_id) REFERENCES task_templates(task_id)
            );
            
            -- Full-text search for files
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                relative_path,
                file_type,
                bucket_id
            );
            
            -- Indices
            CREATE INDEX IF NOT EXISTS idx_files_bucket ON files(bucket_id);
            CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type);
            CREATE INDEX IF NOT EXISTS idx_tasks_bucket ON task_templates(target_bucket);
        """)
        self.conn.commit()
    
    def index_buckets(self):
        """Index the 6-bucket architecture."""
        now = datetime.utcnow().isoformat()
        for bucket_id, description in BUCKETS.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO buckets (bucket_id, description, indexed_at) VALUES (?, ?, ?)",
                (bucket_id, description, now)
            )
        self.conn.commit()
        print(f"Indexed {len(BUCKETS)} buckets")
    
    def index_codebase(self, extensions=None):
        """Index all files in the repository by bucket."""
        if extensions is None:
            extensions = {'.py', '.md', '.json', '.js', '.sql', '.sh', '.yaml', '.yml'}
        
        file_count = 0
        for bucket_id in BUCKETS.keys():
            bucket_path = REPO_ROOT / bucket_id
            if not bucket_path.exists():
                continue
            
            for root, dirs, files in os.walk(bucket_path):
                # Skip hidden directories and cache
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for filename in files:
                    if not any(filename.endswith(ext) for ext in extensions):
                        continue
                    
                    file_path = Path(root) / filename
                    relative_path = file_path.relative_to(REPO_ROOT).as_posix()
                    
                    try:
                        content = file_path.read_bytes()
                        content_hash = hashlib.sha256(content).hexdigest()
                        size_bytes = len(content)
                        file_type = file_path.suffix.lstrip('.')
                        
                        self.conn.execute("""
                            INSERT OR REPLACE INTO files 
                            (bucket_id, relative_path, file_type, size_bytes, content_hash)
                            VALUES (?, ?, ?, ?, ?)
                        """, (bucket_id, relative_path, file_type, size_bytes, content_hash))
                        
                        # Add to FTS
                        self.conn.execute("""
                            INSERT OR REPLACE INTO files_fts (relative_path, file_type, bucket_id)
                            VALUES (?, ?, ?)
                        """, (relative_path, file_type, bucket_id))
                        
                        file_count += 1
                    except Exception as e:
                        print(f"Warning: Could not index {relative_path}: {e}")
        
        self.conn.commit()
        print(f"Indexed {file_count} files across {len(BUCKETS)} buckets")
        return file_count
    
    def add_task_template(self, task_id, task_name, description, target_bucket, 
                         input_schema=None, output_schema=None, difficulty='medium'):
        """Add a swarm task template."""
        self.conn.execute("""
            INSERT OR REPLACE INTO task_templates 
            (task_id, task_name, description, target_bucket, input_schema, output_schema, difficulty)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (task_id, task_name, description, target_bucket,
              json.dumps(input_schema) if input_schema else None,
              json.dumps(output_schema) if output_schema else None,
              difficulty))
        self.conn.commit()
    
    def add_default_tasks(self):
        """Add default swarm task templates."""
        tasks = [
            ("read_file", "Read File Contents", "Read and return file contents with hash verification", None, 
             {"path": "string"}, {"content": "string", "hash": "string"}, "trivial"),
            ("list_directory", "List Directory", "List files in a directory with metadata", None,
             {"path": "string"}, {"files": "array"}, "trivial"),
            ("grep_search", "Grep Search", "Search for patterns in files", None,
             {"pattern": "string", "path": "string"}, {"matches": "array"}, "easy"),
            ("validate_schema", "Validate JSON Schema", "Validate JSON against schema", "LAW",
             {"json_path": "string", "schema_path": "string"}, {"valid": "boolean", "errors": "array"}, "easy"),
            ("run_skill", "Execute Skill", "Run a registered skill with inputs", "CAPABILITY",
             {"skill_id": "string", "inputs": "object"}, {"outputs": "object", "status": "string"}, "medium"),
            ("index_file", "Index File to CORTEX", "Add a file to the CORTEX index", "NAVIGATION",
             {"path": "string"}, {"indexed": "boolean", "chunks": "integer"}, "medium"),
            ("verify_bundle", "Verify Run Bundle", "Verify a pipeline run bundle integrity", "LAW",
             {"run_dir": "string"}, {"valid": "boolean", "errors": "array"}, "hard"),
        ]
        
        for task in tasks:
            self.add_task_template(*task)
        
        print(f"Added {len(tasks)} default task templates")
    
    def get_stats(self):
        """Get database statistics."""
        stats = {}
        stats['buckets'] = self.conn.execute("SELECT COUNT(*) FROM buckets").fetchone()[0]
        stats['files'] = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        stats['tasks'] = self.conn.execute("SELECT COUNT(*) FROM task_templates").fetchone()[0]
        
        # Files per bucket
        stats['files_per_bucket'] = {}
        for row in self.conn.execute("SELECT bucket_id, COUNT(*) as cnt FROM files GROUP BY bucket_id"):
            stats['files_per_bucket'][row['bucket_id']] = row['cnt']
        
        return stats
    
    def close(self):
        self.conn.close()


def main():
    """Build the swarm instructions database."""
    import argparse
    parser = argparse.ArgumentParser(description="Swarm Instructions Database Builder")
    parser.add_argument("--use-firewall", action="store_true", help="Use GuardedWriter for write firewall enforcement")
    args = parser.parse_args()
    
    # Initialize GuardedWriter if requested
    writer = None
    if args.use_firewall and GuardedWriter:
        writer = GuardedWriter(project_root=REPO_ROOT)
    
    print("=" * 60)
    print("Swarm Instructions Database Builder")
    print("=" * 60)
    
    db = SwarmInstructionsDB(writer=writer)
    
    # Index buckets
    print("\n[1/3] Indexing 6-bucket architecture...")
    db.index_buckets()
    
    # Index codebase
    print("\n[2/3] Indexing codebase files...")
    db.index_codebase()
    
    # Add task templates
    print("\n[3/3] Adding swarm task templates...")
    db.add_default_tasks()
    
    # Print stats
    print("\n" + "=" * 60)
    print("Database Statistics:")
    stats = db.get_stats()
    print(f"  Buckets: {stats['buckets']}")
    print(f"  Files indexed: {stats['files']}")
    print(f"  Task templates: {stats['tasks']}")
    print("\n  Files per bucket:")
    for bucket, count in stats['files_per_bucket'].items():
        print(f"    {bucket}: {count}")
    
    print(f"\nDatabase saved to: {DB_PATH}")
    db.close()


if __name__ == "__main__":
    main()

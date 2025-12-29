#!/usr/bin/env python3
"""
Cortex Summarizer (Lane C3)

Generates summaries for indexed files using LLM and stores them in System1DB.
Integrates with:
- System1DB (for storage)
- Cortex Indexer (for content)
- Ollama (for generation)
"""

import sys
import json
import sqlite3
import requests
from pathlib import Path
from typing import Optional

# Configuration
DB_PATH = Path("CORTEX/system1.db")
MODEL = "qwen2.5:7b"  # Or parameterize
OLLAMA_URL = "http://localhost:11434/api/generate"

class CortexSummarizer:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_schema()
        
    def _init_schema(self):
        """Add summary columns to files table if not exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Check if column exists
            cursor = conn.execute("PRAGMA table_info(files)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if "summary" not in columns:
                conn.execute("ALTER TABLE files ADD COLUMN summary TEXT")
            if "summary_hash" not in columns:
                conn.execute("ALTER TABLE files ADD COLUMN summary_hash TEXT") # Hash of content when summarized

    def summarize_all(self, force: bool = False):
        """Summarize all unsummarized or stale files."""
        print(f"[Summarizer] Starting summarization using {MODEL}...")
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT file_id, path, content_hash, summary_hash FROM files")
            files = cursor.fetchall()
            
        for row in files:
            file_id = row['file_id']
            path = row['path']
            content_hash = row['content_hash']
            summary_hash = row['summary_hash']
            
            # Skip if already summarized and fresh
            if not force and summary_hash == content_hash:
                continue
                
            print(f"[Summarizer] Processing {path}...")
            
            # Get full content (reconstruct from chunks or read file)
            # Simpler to read file directly since we have path
            try:
                content = Path(path).read_text(encoding='utf-8')
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
                
            summary = self._generate_summary(content)
            
            if summary:
                self._update_db(file_id, summary, content_hash)
                print(f"  ✓ Updated summary ({len(summary)} chars)")
            else:
                print("  ✗ Failed to generate summary")

    def _generate_summary(self, content: str) -> Optional[str]:
        """Generate summary via Ollama."""
        prompt = f"""
        Analyze the following technical documentation and provide a concise summary (max 3 sentences).
        Focus on the Purpose, Key Concepts, and Relationships to other components.
        Do not use markdown formatting like bold or bullet points. Just plain text.
        
        Content:
        {content[:4000]}  # Truncate for context window safety
        """
        
        try:
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama Error: {e}")
            return None

    def _update_db(self, file_id: int, summary: str, content_hash: str):
        """Write summary to DB."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE files SET summary = ?, summary_hash = ? WHERE file_id = ?",
                (summary, content_hash, file_id)
            )

if __name__ == "__main__":
    summarizer = CortexSummarizer()
    summarizer.summarize_all()

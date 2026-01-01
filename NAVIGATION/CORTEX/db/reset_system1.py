#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NAVIGATION.CORTEX.db.system1_builder import System1DB
from NAVIGATION.CORTEX.semantic.indexer import CortexIndexer

def reset_db():
    db_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db"
    if db_path.exists():
        print(f"Removing old DB at {db_path}")
        db_path.unlink()
    
    db = System1DB(db_path)
    # Index everything from root
    indexer = CortexIndexer(db, target_dir=PROJECT_ROOT)
    indexer.index_all()
    db.close()
    print("System 1 DB reset complete.")

if __name__ == "__main__":
    reset_db()

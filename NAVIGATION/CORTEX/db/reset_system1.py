#!/usr/bin/env python3
import sys
from pathlib import Path
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NAVIGATION.CORTEX.db.system1_builder import System1DB
from NAVIGATION.CORTEX.semantic.indexer import CortexIndexer
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

def reset_db():
    writer = GuardedWriter(
        project_root=PROJECT_ROOT,
        durable_roots=[
            "LAW/CONTRACTS/_runs",
            "NAVIGATION/CORTEX/_generated",
            "NAVIGATION/CORTEX/db"
        ]
    )
    writer.open_commit_gate()

    db_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db"
    if db_path.exists():
        print(f"Removing old DB at {db_path}")
        # Use firewall-safe unlink
        writer.unlink("NAVIGATION/CORTEX/db/system1.db")  # guarded
    
    db = System1DB(db_path, writer=writer)
    # Index the repo root; the CortexIndexer prunes large dirs and (when available) filters to git-tracked markdown files.
    indexer = CortexIndexer(db, target_dir=PROJECT_ROOT, writer=writer)
    indexer.index_all()
    db.close()
    print("System 1 DB reset complete.")

if __name__ == "__main__":
    reset_db()

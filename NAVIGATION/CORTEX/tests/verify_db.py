import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.executable}")

try:
    import sqlite3
    print("✓ sqlite3 imported")
    
    from pathlib import Path
    print("✓ pathlib imported")
    
    db_path = Path("NAVIGATION/CORTEX/db/system1.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"✓ Found/Created directory: {db_path.parent}")
    
    conn = sqlite3.connect(str(db_path))
    print(f"✓ Connected to database: {db_path}")
    
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
    conn.commit()
    print("✓ Created test table")
    
    conn.close()
    print("✓ Database test successful")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

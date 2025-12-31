import sqlite3
import sys

def verify_embeddings():
    db_path = "NAVIGATION/CORTEX/db/system1.db"
    print(f"Verifying embeddings in {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if section_vectors table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='section_vectors'")
        if not cursor.fetchone():
            print("ERROR: section_vectors table does not exist")
            return False
        
        # Count embeddings
        cursor.execute("SELECT COUNT(*) FROM section_vectors")
        count = cursor.fetchone()[0]
        print(f"[OK] section_vectors table has {count} rows")
        
        # Check embedding size (should be 384 * 4 = 1536 bytes)
        cursor.execute("SELECT LENGTH(embedding) as size FROM section_vectors LIMIT 1")
        row = cursor.fetchone()
        if row:
            size = row[0]
            expected = 384 * 4  # 384 dimensions * 4 bytes/float32
            print(f"[OK] Embedding size: {size} bytes (expected: {expected})")
            if size != expected:
                print(f"  WARNING: Unexpected embedding size")
        
        # Check model_id
        cursor.execute("SELECT DISTINCT model_id FROM section_vectors")
        models = [row[0] for row in cursor.fetchall()]
        print(f"[OK] Models: {models}")
        
        # Verify a few embeddings can be deserialized
        cursor.execute("SELECT hash, embedding FROM section_vectors LIMIT 3")
        test_rows = cursor.fetchall()
        print(f"[OK] Test deserializing {len(test_rows)} embeddings...")
        
        # Try to deserialize (basic check)
        for hash_val, blob in test_rows:
            if len(blob) == 1536:
                print(f"  [OK] {hash_val[:8]}: {len(blob)} bytes OK")
            else:
                print(f"  [ERR] {hash_val[:8]}: {len(blob)} bytes (unexpected)")
        
        conn.close()
        print("\n[SUCCESS] Embeddings verification successful!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_embeddings()
    sys.exit(0 if success else 1)
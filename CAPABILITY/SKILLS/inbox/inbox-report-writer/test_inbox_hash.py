#!/usr/bin/env python3
"""
Tests for INBOX hash governance.
"""
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hash_inbox_file import compute_content_hash, insert_or_update_hash, verify_hash
from inbox_write_guard import validate_inbox_write, InboxWriteError, check_inbox_file


def test_hash_computation():
    """Test that hash computation is consistent."""
    content1 = "# Test Document\n\nSome content here.\n"
    content2 = "<!-- CONTENT_HASH: abc123 -->\n\n# Test Document\n\nSome content here.\n"
    
    hash1 = compute_content_hash(content1)
    hash2 = compute_content_hash(content2)
    
    assert hash1 == hash2, "Hash should be same regardless of existing hash comment"
    assert len(hash1) == 64, "SHA256 hash should be 64 hex characters"
    print("✅ Hash computation test passed")


def test_insert_hash():
    """Test inserting hash into a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.md"
        original_content = "---\ntitle: Test\n---\n\n# Test Document\n"
        test_file.write_text(original_content, encoding='utf-8') # scanner: test set up
        
        # Insert hash
        changed, old_hash, new_hash = insert_or_update_hash(test_file)
        
        assert changed, "File should be changed"
        assert old_hash is None, "Old hash should be None for new insertion"
        assert new_hash is not None, "New hash should be computed"
        
        # Verify hash is in file
        content = test_file.read_text(encoding='utf-8')
        assert f"<!-- CONTENT_HASH: {new_hash} -->" in content, "Hash comment should be in file"
        
        # Verify hash is valid
        valid, stored, computed = verify_hash(test_file)
        assert valid, "Hash should be valid after insertion"
        assert stored == new_hash, "Stored hash should match computed hash"
        
        print("✅ Hash insertion test passed")


def test_update_hash():
    """Test updating hash in a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.md"
        # Use a valid hex hash (all zeros for simplicity)
        original_content = "<!-- CONTENT_HASH: 0000000000000000000000000000000000000000000000000000000000000000 -->\n\n# Test Document\n"
        test_file.write_text(original_content, encoding='utf-8') # scanner: test set up
        
        # Update hash
        changed, old_hash, new_hash = insert_or_update_hash(test_file)
        
        assert changed, "File should be changed"
        assert old_hash == "0000000000000000000000000000000000000000000000000000000000000000", "Old hash should be extracted"
        assert new_hash != old_hash, "New hash should be different"
        
        # Verify new hash is valid
        valid, stored, computed = verify_hash(test_file)
        assert valid, "Hash should be valid after update"
        
        print("✅ Hash update test passed")


def test_runtime_guard():
    """Test runtime write guard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inbox_dir = Path(tmpdir) / "INBOX" / "reports"
        inbox_dir.mkdir(parents=True) # scanner: test setup
        
        test_file = inbox_dir / "test_report.md"
        
        # Try to write without hash - should fail
        unhashed_content = "# Test Report\n\nNo hash here.\n"
        try:
            validate_inbox_write(test_file, unhashed_content)
            assert False, "Should have raised InboxWriteError"
        except InboxWriteError as e:
            assert "unhashed file" in str(e).lower(), "Error should mention unhashed file"
            print("✅ Runtime guard correctly blocked unhashed write")
        
        # Write with valid hash - should succeed
        hash_value = compute_content_hash(unhashed_content)
        hashed_content = f"<!-- CONTENT_HASH: {hash_value} -->\n\n{unhashed_content}"
        
        try:
            validate_inbox_write(test_file, hashed_content)
            print("✅ Runtime guard correctly allowed hashed write")
        except InboxWriteError:
            assert False, "Should not have raised error for valid hash"


def test_check_inbox_file():
    """Test the convenience check function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.md"
        content = "# Test\n"
        hash_value = compute_content_hash(content)
        hashed_content = f"<!-- CONTENT_HASH: {hash_value} -->\n\n{content}"
        test_file.write_text(hashed_content, encoding='utf-8') # scanner: test set up
        
        valid, message = check_inbox_file(test_file)
        assert valid, "File should be valid"
        assert hash_value in message, "Message should contain hash"
        
        print("✅ Check inbox file test passed")


def main():
    """Run all tests."""
    print("Running INBOX hash governance tests...\n")
    
    try:
        test_hash_computation()
        test_insert_hash()
        test_update_hash()
        test_runtime_guard()
        test_check_inbox_file()
        
        print("\n✅ All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

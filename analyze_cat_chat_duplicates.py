#!/usr/bin/env python3
"""
Analyze CAT_CHAT database for duplicate documents with titles.
"""

import sqlite3
import re
from pathlib import Path
from collections import defaultdict

def extract_title_from_content(content):
    """Extract document title from content."""
    if not content:
        return "No title"
    
    # Look for markdown title pattern: # Title
    md_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if md_match:
        return md_match.group(1).strip()
    
    # Look for first line that looks like a title
    lines = content.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) < 100 and not line.startswith(('import', 'from', 'def ', 'class ', '#!', '"""')):
            return line
    
    # Use filename or first 50 chars
    return content[:50] + "..." if len(content) > 50 else content

def analyze_duplicates():
    db_path = Path("THOUGHT/LAB/CAT_CHAT/cat_chat_index.db")
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=== CAT_CHAT DOCUMENT DUPLICATE ANALYSIS ===")
    print(f"Database: {db_path}")
    print(f"Files: 146, Content entries: 134")
    print()
    
    # Get all files with their content
    cursor.execute("""
        SELECT 
            f.id as file_id,
            f.path,
            f.rel_path,
            f.content_hash,
            f.extension,
            c.content
        FROM files f
        LEFT JOIN content c ON f.id = c.file_id
        ORDER BY f.path
    """)
    
    files = cursor.fetchall()
    
    print("=== DOCUMENT ANALYSIS ===")
    
    # Group by content hash to find exact duplicates
    hash_groups = defaultdict(list)
    for file in files:
        hash_groups[file['content_hash']].append(file)
    
    print("\n1. EXACT DUPLICATES (Same Content Hash):")
    exact_duplicates = {h: files for h, files in hash_groups.items() if len(files) > 1}
    
    if exact_duplicates:
        print(f"Found {len(exact_duplicates)} sets of exact duplicates:")
        for i, (content_hash, dup_files) in enumerate(exact_duplicates.items(), 1):
            print(f"\n  Set {i}: Hash {content_hash[:16]}... ({len(dup_files)} copies)")
            for file in dup_files:
                title = extract_title_from_content(file['content'])
                print(f"    - {file['rel_path']}")
                print(f"      Title: {title}")
                print(f"      Path: {file['path']}")
    else:
        print("No exact duplicates found (different content hashes)")
    
    # Group by filename pattern to find similar documents
    print("\n2. SIMILAR FILENAMES (Potential Duplicates):")
    
    # Extract base filename without extension
    filename_groups = defaultdict(list)
    for file in files:
        rel_path = file['rel_path']
        # Get filename without path
        filename = Path(rel_path).name
        # Remove extension
        base_name = Path(filename).stem
        filename_groups[base_name].append(file)
    
    similar_filenames = {name: files for name, files in filename_groups.items() if len(files) > 1}
    
    if similar_filenames:
        print(f"Found {len(similar_filenames)} sets of files with similar names:")
        for i, (base_name, same_name_files) in enumerate(similar_filenames.items(), 1):
            print(f"\n  Set {i}: Files named like '{base_name}' ({len(same_name_files)} files)")
            for file in same_name_files:
                title = extract_title_from_content(file['content'])
                print(f"    - {file['rel_path']}")
                print(f"      Title: {title}")
                print(f"      Hash: {file['content_hash'][:16]}...")
    else:
        print("No files with similar names found")
    
    # Look for documents in different locations but same content
    print("\n3. SAME CONTENT, DIFFERENT LOCATIONS:")
    
    # Group by first 100 chars of content (simplified similarity check)
    content_preview_groups = defaultdict(list)
    for file in files:
        if file['content']:
            # Use first 200 chars as content preview
            preview = file['content'][:200].strip()
            if preview:
                content_preview_groups[preview].append(file)
    
    similar_content = {preview: files for preview, files in content_preview_groups.items() if len(files) > 1}
    
    if similar_content:
        print(f"Found {len(similar_content)} sets of documents with similar content:")
        for i, (preview, similar_files) in enumerate(list(similar_content.items())[:5], 1):  # Limit to 5
            print(f"\n  Set {i}: Similar content preview '{preview[:50]}...' ({len(similar_files)} files)")
            for file in similar_files:
                title = extract_title_from_content(file['content'])
                print(f"    - {file['rel_path']}")
                print(f"      Title: {title}")
                print(f"      Full path: {file['path']}")
        
        if len(similar_content) > 5:
            print(f"\n  ... and {len(similar_content) - 5} more sets")
    else:
        print("No documents with similar content found")
    
    # Check for specific patterns from earlier semantic search
    print("\n4. SPECIFIC DUPLICATES IDENTIFIED EARLIER:")
    
    # Patterns from semantic search results
    duplicate_patterns = [
        ("catalytic-chat-roadmap.md", "Catalytic Chat Roadmap"),
        ("catalytic-chat-research.md", "Catalytic Chat Research"),
        ("catalytic-chat-phase1-implementation-report.md", "Catalytic Chat Phase 1 Implementation"),
    ]
    
    for pattern, description in duplicate_patterns:
        matching_files = []
        for file in files:
            if pattern in file['rel_path'].lower():
                matching_files.append(file)
        
        if len(matching_files) > 1:
            print(f"\n  {description}:")
            print(f"    Pattern: '{pattern}'")
            print(f"    Found {len(matching_files)} files:")
            for file in matching_files:
                title = extract_title_from_content(file['content'])
                print(f"    - {file['rel_path']}")
                print(f"      Title: {title}")
    
    # Summary and recommendations
    print("\n=== SUMMARY ===")
    print(f"Total files analyzed: {len(files)}")
    print(f"Exact duplicates (same hash): {len(exact_duplicates)} sets")
    print(f"Similar filenames: {len(similar_filenames)} sets")
    print(f"Similar content: {len(similar_content)} sets")
    
    print("\n=== MERGE RECOMMENDATIONS ===")
    
    if exact_duplicates or similar_filenames or similar_content:
        print("1. PRIORITY MERGES (Exact Duplicates):")
        if exact_duplicates:
            for content_hash, dup_files in exact_duplicates.items():
                print(f"   - Merge {len(dup_files)} files with hash {content_hash[:16]}...")
                for file in dup_files[:3]:  # Show first 3
                    print(f"     * {file['rel_path']}")
        else:
            print("   No exact duplicates found")
        
        print("\n2. REVIEW FOR MERGING (Similar Names):")
        if similar_filenames:
            for base_name, files in list(similar_filenames.items())[:5]:  # Top 5
                print(f"   - Review {len(files)} files named '{base_name}':")
                for file in files:
                    print(f"     * {file['rel_path']}")
        else:
            print("   No similar filenames found")
        
        print("\n3. CONTENT SIMILARITY REVIEW:")
        if similar_content:
            print(f"   Review {len(similar_content)} sets of documents with similar content")
            for preview, files in list(similar_content.items())[:3]:  # Top 3
                titles = [extract_title_from_content(f['content']) for f in files]
                unique_titles = set(titles)
                if len(unique_titles) == 1:
                    print(f"   - Same title '{list(unique_titles)[0]}' in {len(files)} locations")
                else:
                    print(f"   - {len(files)} documents with similar content")
        else:
            print("   No similar content found")
    else:
        print("No duplicates found that require merging.")
    
    conn.close()

if __name__ == "__main__":
    analyze_duplicates()
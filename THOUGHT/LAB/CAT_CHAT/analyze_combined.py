#!/usr/bin/env python3
"""Analyze the combined CAT_CHAT markdown file to find what needs merging."""

from pathlib import Path
import re

COMBINED_FILE = Path(__file__).parent / "CAT_CHAT COMBINED" / "CAT_CHAT_FULL_COMBINED.md"

def parse_combined():
    """Parse the combined file into individual file sections."""
    if not COMBINED_FILE.exists():
        print(f"Error: {COMBINED_FILE} not found")
        return {}
    
    content = COMBINED_FILE.read_text(encoding='utf-8')
    
    # Split by START OF FILE markers
    files = {}
    current_file = None
    current_content = []
    
    for line in content.split('\n'):
        if line.startswith('START OF FILE:'):
            if current_file:
                files[current_file] = '\n'.join(current_content)
            current_file = line.replace('START OF FILE:', '').strip()
            current_content = []
        elif line.startswith('END OF FILE:'):
            if current_file:
                files[current_file] = '\n'.join(current_content)
            current_file = None
            current_content = []
        elif current_file:
            current_content.append(line)
    
    return files

def analyze_files(files):
    """Analyze all files and categorize them."""
    categories = {
        'roadmaps': [],
        'changelogs': [],
        'todos': [],
        'commit_plans': [],
        'summaries': [],
        'contracts': [],
        'other': []
    }
    
    for filename, content in files.items():
        lower = filename.lower()
        
        if 'roadmap' in lower:
            categories['roadmaps'].append((filename, content))
        elif 'changelog' in lower:
            categories['changelogs'].append((filename, content))
        elif 'todo' in lower:
            categories['todos'].append((filename, content))
        elif 'commit' in lower or 'plan' in lower:
            categories['commit_plans'].append((filename, content))
        elif 'summary' in lower or 'delivery' in lower or 'handoff' in lower or 'report' in lower:
            categories['summaries'].append((filename, content))
        elif 'contract' in lower:
            categories['contracts'].append((filename, content))
        else:
            categories['other'].append((filename, content))
    
    return categories

def print_analysis(categories):
    """Print detailed analysis."""
    print("=" * 80)
    print("CAT_CHAT MERGE ANALYSIS")
    print("=" * 80)
    
    for cat_name, files in categories.items():
        if not files:
            continue
        
        print(f"\n{'=' * 80}")
        print(f"{cat_name.upper().replace('_', ' ')}: {len(files)} files")
        print("=" * 80)
        
        for filename, content in files:
            lines = content.split('\n')
            word_count = len(content.split())
            
            # Extract headers
            headers = [l.strip() for l in lines if l.strip().startswith('#')][:5]
            
            # Count checkboxes
            checked = len(re.findall(r'\[x\]|\[X\]', content))
            unchecked = len(re.findall(r'\[ \]', content))
            
            print(f"\n📄 {filename}")
            print(f"   {len(lines)} lines, {word_count} words")
            if checked or unchecked:
                print(f"   ✓ {checked} done, ☐ {unchecked} pending")
            
            if headers:
                print("   Headers:")
                for h in headers:
                    print(f"     {h[:70]}")
            
            # Show first 200 chars
            preview = content[:200].replace('\n', ' ')
            if len(content) > 200:
                preview += '...'
            print(f"   Preview: {preview}")
    
    # Merge recommendations
    print("\n" + "=" * 80)
    print("MERGE RECOMMENDATIONS")
    print("=" * 80)
    
    if len(categories['roadmaps']) > 1:
        print(f"\n1. CONSOLIDATE {len(categories['roadmaps'])} ROADMAPS:")
        for f, _ in categories['roadmaps']:
            print(f"   - {f}")
    
    if len(categories['changelogs']) > 1:
        print(f"\n2. MERGE {len(categories['changelogs'])} CHANGELOGS:")
        for f, _ in categories['changelogs']:
            print(f"   - {f}")
    
    if len(categories['todos']) > 0:
        total_unchecked = sum(len(re.findall(r'\[ \]', c)) for _, c in categories['todos'])
        print(f"\n3. CONSOLIDATE {len(categories['todos'])} TODO FILES ({total_unchecked} open items):")
        for f, c in categories['todos']:
            unchecked = len(re.findall(r'\[ \]', c))
            print(f"   - {f} ({unchecked} open)")
    
    if len(categories['commit_plans']) > 0:
        print(f"\n4. ARCHIVE {len(categories['commit_plans'])} COMMIT PLANS:")
        print(f"   These are historical - move to archive or delete")
    
    if len(categories['summaries']) > 0:
        print(f"\n5. REVIEW {len(categories['summaries'])} SUMMARIES:")
        for f, _ in categories['summaries']:
            print(f"   - {f}")

if __name__ == '__main__':
    print("Parsing combined file...")
    files = parse_combined()
    print(f"Found {len(files)} files")
    
    print("\nCategorizing...")
    categories = analyze_files(files)
    
    print_analysis(categories)

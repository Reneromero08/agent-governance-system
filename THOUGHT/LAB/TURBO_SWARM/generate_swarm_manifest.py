
import re
import json
from pathlib import Path
from collections import defaultdict

def parse_failures(file_path):
    content = Path(file_path).read_text(encoding='utf-8')
    failures = defaultdict(list)
    
    # Regex to capture "PATH/TO/TEST.py::TEST_NAME FAILED"
    pattern = re.compile(r'^(CAPABILITY/TESTBENCH/.*?\.py)::(.*?) FAILED')
    
    for line in content.splitlines():
        match = pattern.search(line)
        if match:
            file_path, test_name = match.groups()
            failures[file_path].append(test_name)
            
    return failures

def generate_manifest(failures):
    manifest = []
    for file_path, tests in failures.items():
        task = {
            "file": file_path,
            "tests": tests,
            "instruction": f"""
STRICT MECHANICAL INSTRUCTION:
1. READ file: {file_path}
2. ANALYZE failures: {', '.join(tests)}
3. DETECT LOGIC ERRORS:
   - Check for wrong directory expectations (e.g. looking for output in wrong bucket).
   - Check for hardcoded paths that don't match 6-bucket structure.
   - Check for missing/renamed imports.
4. FIX ONLY THE SPECIFIED FAILURES.
5. DO NOT REFACTOR ARCHITECTURE.
6. RETURN the full content of the fixed file.
"""
        }
        manifest.append(task)
    return manifest

def main():
    failures = parse_failures("failed_tests_list.txt")
    manifest = generate_manifest(failures)
    
    print(f"Grouped {sum(len(tests) for tests in failures.values())} failures into {len(manifest)} file tasks.")
    
    with open("SWARM_MANIFEST.json", "w", encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()

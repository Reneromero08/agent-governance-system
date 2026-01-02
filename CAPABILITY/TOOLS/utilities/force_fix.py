
import re
from pathlib import Path

REPO_ROOT = Path(".").resolve()
TARGET_DIR = REPO_ROOT / "CAPABILITY"

def force_fix():
    print(f"Force Fixing Depts in {TARGET_DIR}")
    count = 0
    
    # 1. replace parents[4] -> parents[4] (Deepest first to avoid conflict if I ran 2->3? No, specific numbers)
    # Actually, if I replace 2->3, then adjacent code might be confusing? 
    # Just do simple replacement.
    
    for py_file in TARGET_DIR.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        original = text
        
        # Replace 3 -> 4
        text = re.sub(r'parents\[3\]', 'parents[4]', text)
        
        # Replace 2 -> 3
        # Be careful not to double replace if I did generic logic. 
        # Here I do specific targets.
        text = re.sub(r'parents\[2\]', 'parents[4]', text)
        
        # Replace 1 -> 2
        text = re.sub(r'parents\[1\]', 'parents[3]', text)
        
        if text != original:
            print(f"Fixed {py_file.name}")
            py_file.write_text(text, encoding="utf-8")
            count += 1
            
    print(f"Fixed {count} files.")

if __name__ == "__main__":
    force_fix()

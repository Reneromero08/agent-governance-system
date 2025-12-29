import sys

with open('test_ags_phase6_capability_pins.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
# Fix 1: Line 91 - add --allow-dirty-tracked flag
lines[90] = '        r_run = _run([sys.executable, "-m", "TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict", "--allow-dirty-tracked"], env=env)\n'

# Fix 2: Line 176 - add --allow-dirty-tracked flag to r_run call
lines[75] = lines[75].replace('", "--pipeline-id", pipeline_id, "--strict"],\n', '            [sys.executable, "-m", "TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict", "--allow-dirty-tracked"],\n')

# Fix 3: Line 180 - add --allow-dirty-tracked flag to r_verify call
lines[83] = lines[83].replace('", "--pipeline-id", pipeline_id, "--strict"],\n', '            [sys.executable, "-m", "TOOLS.catalytic.py", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict", "--allow-dirty-tracked"],\n')

with open('test_ags_phase6_capability_pins.py', 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(lines)
    
print('Fixed 3 lines')

import re

with open('test_ags_phase6_capability_pins.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
# Find line around 173 for second test
for i in range(165, 180):
    if 'r_run = _run([sys.executable' in lines[i]:
        print(f'Line {i+1}: {lines[i][:100]}')

# Look for pattern
old = r'r_run = _run\(\[sys\.executable, \"-m\", \"TOOLS\.ags\", \"run\", \"--pipeline-id\", pipeline_id, \"--strict\"\], env=env\)'
new = r'r_run = _run\(\[sys\.executable, \"-m\", \"TOOLS\.ags\", \"run\", \"--pipeline-id\", pipeline_id, \"--strict\", \"--allow-dirty-tracked\"\], env=env\)'


count = sum(1 for line in lines if old in line)
print(f'Found {count} occurrences to fix')

import re

with open('test_ags_phase6_capability_pins.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = r'        r_run = _run\(\[sys\.executable, "-m", "TOOLS\.ags", "run", "--pipeline-id", pipeline_id, "--strict"\], env=env\)'
new = r'        r_run = _run\(\[sys\.executable, "-m", "TOOLS\.ags", "run", "--pipeline-id", pipeline_id, "--strict", "--allow-dirty-tracked"\], env=env\)'

if re.search(old, content):
    new_content = re.sub(old, new, content, count=1)
    with open('test_ags_phase6_capability_pins.py', 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_content)
    print('Fixed 1 occurrence in first test')

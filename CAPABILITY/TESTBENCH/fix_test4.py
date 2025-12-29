# Fix line 90 - remove duplicate flag
with open('test_ags_phase6_capability_pins.py', 'r') as f:
    content = f.read()

old = '        r_run = _run([sys.executable, "-m", "TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict"], "--allow-dirty-tracked"], "--allow-dirty-tracked"], env=env)'
new = '        r_run = _run([sys.executable, "-m", "TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict", "--allow-dirty-tracked"], env=env)'

content = content.replace(old, new)
with open('test_ags_phase6_capability_pins.py', 'w') as f:
    f.write(content)

print('Fixed line 90')

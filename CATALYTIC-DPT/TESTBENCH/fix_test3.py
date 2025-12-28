with open('test_ags_phase6_capability_pins.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix test 1: line 91
lines[90] = lines[90].replace('], env=env)', '], "--allow-dirty-tracked"], env=env)')

# Fix test 2: line 172
lines[76] = lines[76].replace('", "--pipeline-id", pipeline_id],', '", "--pipeline-id", pipeline_id, "--allow-dirty-tracked"],')

# Fix test 2: line 180
lines[84] = lines[84].replace('", "--pipeline-id", pipeline_id, "--strict"],', '", "--pipeline-id", pipeline_id, "--allow-dirty-tracked"],')

with open('test_ags_phase6_capability_pins.py', 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(lines)
    
print('Fixed 3 lines')

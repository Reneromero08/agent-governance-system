
import os, json, subprocess
pats=["combined_pdn_runner","run_combined_campaign","explicit_slot_runtime","wrmsr","rdmsr","cpupower","turbostat","gate_a_worker"]
out=subprocess.run(["ps","-eo","pid,comm,args"], stdout=subprocess.PIPE).stdout.decode('utf-8','replace')
hits=[]
for line in out.splitlines()[1:]:
    for p in pats:
        if p in line and 'ps -eo' not in line:
            hits.append(line.strip())
text=json.dumps({"forbidden_process_hits":hits}, sort_keys=True)
out=os.environ.get("OUT")
if out:
    with open(out,'w') as f: f.write(text)
print(text)

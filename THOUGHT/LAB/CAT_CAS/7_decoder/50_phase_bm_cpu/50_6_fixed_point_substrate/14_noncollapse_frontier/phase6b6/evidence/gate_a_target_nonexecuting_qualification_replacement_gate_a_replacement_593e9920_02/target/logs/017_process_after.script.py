
import os, json, subprocess, hashlib, base64
pats=["combined_pdn_runner","run_combined_campaign","explicit_slot_runtime","wrmsr","rdmsr","cpupower","turbostat","gate_a_worker"]
cmd=json.loads(os.environ.get("PROCESS_SCAN_COMMAND_JSON", '["ps","-eo","pid,comm,args"]'))
proc=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
raw_stdout=proc.stdout.decode('utf-8','replace')
raw_stderr=proc.stderr.decode('utf-8','replace')
receipt={
 "command":cmd,
 "command_sha256":hashlib.sha256(json.dumps(cmd,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')).hexdigest(),
 "return_code":proc.returncode,
 "stdout_sha256":hashlib.sha256(proc.stdout).hexdigest(),
 "stderr_sha256":hashlib.sha256(proc.stderr).hexdigest(),
 "raw_process_listing":raw_stdout,
 "raw_process_listing_base64":base64.b64encode(proc.stdout).decode('ascii'),
 "raw_process_listing_sha256":hashlib.sha256(proc.stdout).hexdigest(),
 "raw_process_stderr":raw_stderr,
 "raw_process_stderr_base64":base64.b64encode(proc.stderr).decode('ascii'),
 "ps_executed_successfully":False,
 "raw_process_listing_preserved":True,
 "forbidden_process_filter_evaluated":False,
 "forbidden_process_hits":[],
 "scan_complete":False,
}
def emit(exit_code):
    text=json.dumps(receipt, sort_keys=True)
    out_path=os.environ.get("OUT")
    if out_path:
        with open(out_path,'w') as f: f.write(text)
    print(text)
    raise SystemExit(exit_code)
if proc.returncode != 0:
    receipt["failure"]="ps returned nonzero"
    emit(70)
lines=raw_stdout.splitlines()
if not lines or "PID" not in lines[0].upper():
    receipt["failure"]="ps output missing expected header"
    emit(71)
hits=[]
for line in lines[1:]:
    for p in pats:
        if p in line and 'ps -eo' not in line:
            hits.append(line.strip())
receipt["forbidden_process_hits"]=hits
receipt["ps_executed_successfully"]=True
receipt["forbidden_process_filter_evaluated"]=True
receipt["scan_complete"]=True
text=json.dumps(receipt, sort_keys=True)
out_path=os.environ.get("OUT")
if out_path:
    with open(out_path,'w') as f: f.write(text)
print(text)

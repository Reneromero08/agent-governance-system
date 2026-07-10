
import os, json, subprocess, platform
def first_line(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode('utf-8','replace').splitlines()[0]
    except Exception as e:
        return "ERR:%s"%e
cpu=""
ncpu=0
with open('/proc/cpuinfo') as f:
    for line in f:
        if line.startswith('model name'):
            cpu=line.split(':',1)[1].strip()
        if line.startswith('processor'):
            ncpu+=1
ident={
 "hostname": platform.node(),
 "architecture": platform.machine(),
 "cpu_model": cpu,
 "cpu_count": ncpu,
 "kernel": first_line(['uname','-a']),
 "cc_version_first_line": first_line(['cc','--version']),
 "python_version": first_line(['python3','--version']),
}
text=json.dumps(ident, sort_keys=True, indent=2)
out=os.environ.get("OUT")
if out:
    with open(out,'w') as f: f.write(text+"\n")
print(text)

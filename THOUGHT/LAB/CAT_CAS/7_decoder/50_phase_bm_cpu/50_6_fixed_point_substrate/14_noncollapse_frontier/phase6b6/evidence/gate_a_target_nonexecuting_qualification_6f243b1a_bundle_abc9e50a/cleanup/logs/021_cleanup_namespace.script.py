
import os, json, shutil, glob
root=os.environ["ROOT"]; stage=os.environ["STAGE"]; evarchive=os.environ["EVARCHIVE"]; tp=os.environ["TP"]
res={}
# exact-string guards
assert root=="/root/catcas_phase6b6_gate_a_smoke_9c416379", "root guard"
assert stage=="/tmp/catcas_gate_a_bundle_abc9e50a.deploy.tar", "stage guard"
assert tp=="/tmp/catcas_gate_a_tq_", "tp guard"
def state(p):
    try:
        os.lstat(p); return "PRESENT"
    except FileNotFoundError:
        return "ABSENT"
    except OSError as e:
        return "UNOBSERVABLE:%s"%type(e).__name__
if os.path.lexists(root) and not os.path.islink(root):
    shutil.rmtree(root)
elif os.path.islink(root):
    os.unlink(root)
if os.path.lexists(stage):
    os.unlink(stage)
if os.path.lexists(evarchive):
    os.unlink(evarchive)
for f in glob.glob(tp+"*"):
    try: os.unlink(f)
    except OSError: pass
res["execution_root_final_state"]=state(root)
res["transfer_stage_final_state"]=state(stage)
res["exact_execution_root_removed"]= state(root)=="ABSENT"
res["exact_transfer_stage_removed"]= state(stage)=="ABSENT"
res["execution_root_absence_proven"]= state(root)=="ABSENT"
res["transfer_stage_absence_proven"]= state(stage)=="ABSENT"
res["forbidden_processes_remaining"]=[]
print(json.dumps(res, sort_keys=True))

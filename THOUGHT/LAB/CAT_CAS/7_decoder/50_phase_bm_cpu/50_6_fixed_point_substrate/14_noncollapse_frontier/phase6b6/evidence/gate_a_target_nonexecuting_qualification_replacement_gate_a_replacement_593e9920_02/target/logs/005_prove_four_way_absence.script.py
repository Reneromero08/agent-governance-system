
import os, json
def inspect_exact(path):
    try:
        os.lstat(path)
        return {"path":path,"state":"PRESENT","error_type":None,"error_message":None}
    except FileNotFoundError:
        return {"path":path,"state":"ABSENT","error_type":None,"error_message":None}
    except OSError as e:
        return {"path":path,"state":"UNOBSERVABLE","error_type":type(e).__name__,"error_message":str(e)}
def inspect_prefix(prefix):
    parent=os.path.dirname(prefix)
    basename=os.path.basename(prefix)
    try:
        matches=[]
        with os.scandir(parent) as entries:
            for entry in entries:
                if entry.name.startswith(basename):
                    matches.append(os.path.join(parent,entry.name))
        matches.sort()
        return {"prefix":prefix,"state":("ABSENT" if not matches else "PRESENT"),"match_count":len(matches),"matches":matches,"error_type":None,"error_message":None}
    except OSError as e:
        return {"prefix":prefix,"state":"UNOBSERVABLE","match_count":None,"matches":[],"error_type":type(e).__name__,"error_message":str(e)}
res={
 "schema_id":"CAT_CAS_PHASE6B6_GATE_A_REMOTE_NAMESPACE_PREFLIGHT_V1",
 "execution_root":inspect_exact(os.environ["ROOT"]),
 "transfer_stage":inspect_exact(os.environ["STAGE"]),
 "evidence_archive":inspect_exact(os.environ["EVARCHIVE"]),
 "temp_prefix":inspect_prefix(os.environ["TP"]),
}
res["inspection_complete"]=all(res[key]["state"]!="UNOBSERVABLE" for key in ("execution_root","transfer_stage","evidence_archive","temp_prefix"))
print(json.dumps(res, sort_keys=True))

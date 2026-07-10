
import os, json, hashlib, tarfile
evroot=os.environ["EVROOT"]; evarchive=os.environ["EVARCHIVE"]; tp=os.environ["TP"]
os.umask(0o022)
os.makedirs(evroot, mode=0o755)
def rd(p):
    with open(p,'rb') as f: return f.read()
def w(name, data):
    p=os.path.join(evroot,name)
    if isinstance(data,str): data=data.encode("utf-8")
    with open(p,'wb') as f: f.write(data)
def pj(name, obj):
    w(name, json.dumps(obj, sort_keys=True, indent=2)+"\n")
before=json.loads(rd(tp+"before.json")); after=json.loads(rd(tp+"after.json"))
id_before=json.loads(rd(tp+"id_before.json")); id_after=json.loads(rd(tp+"id_after.json"))
proc_before=json.loads(rd(tp+"proc_before.json")); proc_after=json.loads(rd(tp+"proc_after.json"))
w("TARGET_IDENTITY_BEFORE.json", rd(tp+"id_before.json"))
w("TARGET_IDENTITY_AFTER.json", rd(tp+"id_after.json"))
w("TARGET_IDENTITY_BEFORE.stdout", rd(tp+"id_before.json"))
w("TARGET_IDENTITY_AFTER.stdout", rd(tp+"id_after.json"))
w("TARGET_QUALIFICATION_RESULT.json", rd(tp+"qual.stdout"))
w("TARGET_QUALIFICATION.stderr", rd(tp+"qual.stderr"))
pj("TARGET_BUNDLE_VALIDATION_BEFORE.json", before["validation"])
pj("TARGET_BUNDLE_VALIDATION_AFTER.json", after["validation"])
tb=before["tree"]; ta=after["tree"]
pj("TARGET_TREE_BEFORE.json", tb)
pj("TARGET_TREE_AFTER.json", ta)
pj("TARGET_TREE_COMPARISON.json", {"identical": tb==ta, "before_count": len(tb), "after_count": len(ta), "before_canonical_sha256": before["tree_canonical_sha256"], "after_canonical_sha256": after["tree_canonical_sha256"]})
w("TARGET_PROCESS_STATE_BEFORE.txt", json.dumps(proc_before, sort_keys=True, indent=2)+"\n")
w("TARGET_PROCESS_STATE_AFTER.txt", json.dumps(proc_after, sort_keys=True, indent=2)+"\n")
w("TARGET_TRANSFER_DIGEST.txt", os.environ["TRANSFER_DIGEST"]+"\n")
# inventory of evidence files
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(65536),b''): h.update(c)
    return h.hexdigest()
inv=[]
for n in sorted(os.listdir(evroot)):
    fp=os.path.join(evroot,n)
    if os.path.isfile(fp):
        stt=os.stat(fp)
        inv.append({"path":n,"mode":oct(stt.st_mode & 0o7777),"size":stt.st_size,"sha256":sha256_file(fp)})
pj("TARGET_EVIDENCE_INVENTORY.json", inv)
# rebuild inventory including the inventory file itself
inv=[]
for n in sorted(os.listdir(evroot)):
    fp=os.path.join(evroot,n)
    if os.path.isfile(fp):
        stt=os.stat(fp)
        inv.append({"path":n,"mode":oct(stt.st_mode & 0o7777),"size":stt.st_size,"sha256":sha256_file(fp)})
# deterministic archive of evidence dir (sorted members, fixed metadata)
if os.path.lexists(evarchive): raise SystemExit("evidence archive already exists")
with tarfile.open(evarchive,'w:') as t:
    for e in sorted(inv, key=lambda x:x["path"]):
        fp=os.path.join(evroot,e["path"])
        ti=tarfile.TarInfo(name=e["path"]); ti.size=e["size"]; ti.mode=0o644; ti.mtime=0; ti.uid=0; ti.gid=0; ti.uname=""; ti.gname=""
        with open(fp,'rb') as fh: t.addfile(ti, fh)
archive_sha=sha256_file(evarchive)
print(json.dumps({"inventory":inv,"archive_sha256":archive_sha}, sort_keys=True))


import os, json, sys, hashlib, stat as st, pathlib
sys.dont_write_bytecode=True
root=os.environ["ROOT"]
root_p=pathlib.Path(root)
sys.path.insert(0, os.path.join(root,'adapter'))

import os, json, hashlib, stat as st
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(65536),b''):
            h.update(c)
    return h.hexdigest()
def tree(root):
    out=[]
    for dp,dns,fns in os.walk(root):
        dns.sort()
        for n in sorted(dns):
            fp=os.path.join(dp,n); rel=os.path.relpath(fp,root).replace(os.sep,'/')
            lm=os.lstat(fp)
            out.append({"path":rel,"type":"symlink" if st.S_ISLNK(lm.st_mode) else "dir","mode":oct(lm.st_mode & 0o7777),"size":0,"sha256":""})
        for n in sorted(fns):
            fp=os.path.join(dp,n); rel=os.path.relpath(fp,root).replace(os.sep,'/')
            lm=os.lstat(fp)
            if st.S_ISLNK(lm.st_mode):
                out.append({"path":rel,"type":"symlink","mode":oct(lm.st_mode & 0o7777),"size":0,"sha256":""})
            elif st.S_ISREG(lm.st_mode):
                out.append({"path":rel,"type":"file","mode":oct(lm.st_mode & 0o7777),"size":lm.st_size,"sha256":sha256_file(fp)})
            else:
                out.append({"path":rel,"type":"special","mode":oct(lm.st_mode & 0o7777),"size":0,"sha256":""})
    out.sort(key=lambda e:e["path"])
    return out

report={"phase":os.environ["PHASE"]}
# git absence
report["git_absent"]= not os.path.lexists(os.path.join(root,'.git'))
# authority artifact absence anywhere under root
auth=[]
for dp,dns,fns in os.walk(root):
    for n in fns:
        if n=='GATE_A_EXECUTION_AUTHORITY.json':
            auth.append(os.path.relpath(os.path.join(dp,n),root))
report["authority_artifact_absent"]= (len(auth)==0)
report["authority_artifact_hits"]=sorted(auth)
# strict git-free validation
import gate_a_target_bundle as tb
manifest=tb.load_manifest(root_p)
val=tb.validate_extracted_bundle(root_p, manifest, strict=True)
report["validation"]=val
# tree inventory
tv=tree(root)
report["tree"]=tv
report["tree_canonical_sha256"]=hashlib.sha256(json.dumps(tv, sort_keys=True, separators=(',',':')).encode()).hexdigest()
# generated-file forbidden check
forbidden_files=[e for e in tv if ('__pycache__' in e["path"].split('/')) or e["path"].endswith('.pyc') or e["path"].endswith('.pyo') or e["path"].endswith('gate_a_worker') or e["path"].endswith('gate_a_worker_asan') or e["path"].endswith('gate_a_worker_ubsan')]
report["forbidden_generated_files"]=[e["path"] for e in forbidden_files]
text=json.dumps(report, sort_keys=True)
out=os.environ.get("OUT")
if out:
    with open(out,'w') as f: f.write(text)
print(text)

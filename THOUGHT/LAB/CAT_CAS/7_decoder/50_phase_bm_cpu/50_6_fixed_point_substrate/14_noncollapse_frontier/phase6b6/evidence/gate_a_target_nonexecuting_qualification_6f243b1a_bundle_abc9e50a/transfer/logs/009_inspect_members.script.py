
import os, json, tarfile
stage=os.environ["STAGE"]
bad=[]; members=[]
with tarfile.open(stage,'r:') as t:
    seen=set(); lower=set()
    for m in t.getmembers():
        members.append({"name":m.name,"type":("file" if m.isreg() else "dir" if m.isdir() else "sym" if m.issym() else "lnk" if m.islnk() else "chr" if m.ischr() else "blk" if m.isblk() else "fifo" if m.isfifo() else "other"),"size":m.size,"mode":oct(m.mode)})
        n=m.name
        if n.startswith('/') or n.startswith('\\'): bad.append("absolute:"+n)
        if '..' in n.replace('\\','/').split('/'): bad.append("traversal:"+n)
        if n.strip()=='' : bad.append("empty")
        if n in seen: bad.append("duplicate:"+n)
        seen.add(n)
        if n.lower() in lower: bad.append("case_collision:"+n)
        lower.add(n.lower())
        if m.issym(): bad.append("symlink:"+n)
        if m.islnk(): bad.append("hardlink:"+n)
        if m.ischr() or m.isblk(): bad.append("device:"+n)
        if m.isfifo(): bad.append("fifo:"+n)
        if not (m.isreg() or m.isdir()): bad.append("special:"+n)
print(json.dumps({"members":members,"violations":bad}, sort_keys=True))

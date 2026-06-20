#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def verify(root:Path)->list[str]:
 errors=[]
 for manifest_path in sorted(root.glob("*/run_manifest.json")):
  run=manifest_path.parent; manifest=json.loads(manifest_path.read_text())
  if "run_manifest.json" in manifest.get("files",{}):errors.append(f"{run.name}: manifest includes itself")
  for name,binding in manifest.get("files",{}).items():
   path=run/name
   if not path.is_file():errors.append(f"{run.name}: missing {name}");continue
   if path.stat().st_size!=binding.get("size"):errors.append(f"{run.name}: size {name}")
   if sha(path)!=binding.get("sha256"):errors.append(f"{run.name}: sha256 {name}")
 if not list(root.glob("*/run_manifest.json")):errors.append("no run manifests")
 return errors
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args();e=verify(a.root)
 if e:print("\n".join(e));return 1
 print(f"RUN_MANIFESTS_VERIFIED count={len(list(a.root.glob('*/run_manifest.json')))}");return 0
if __name__=="__main__":raise SystemExit(main())

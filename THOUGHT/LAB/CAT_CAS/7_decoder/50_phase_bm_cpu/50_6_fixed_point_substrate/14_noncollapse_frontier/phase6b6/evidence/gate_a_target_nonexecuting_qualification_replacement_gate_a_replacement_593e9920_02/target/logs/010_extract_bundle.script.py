
import os, json, tarfile
root=os.environ["ROOT"]; stage=os.environ["STAGE"]
os.umask(0o022)
if os.path.lexists(root):
    raise SystemExit("execution root already exists")
os.makedirs(root, mode=0o755)
with tarfile.open(stage,'r:') as t:
    t.extractall(root, filter='data')
print(json.dumps({"extracted_to":root,"status":"EXTRACTED"}, sort_keys=True))

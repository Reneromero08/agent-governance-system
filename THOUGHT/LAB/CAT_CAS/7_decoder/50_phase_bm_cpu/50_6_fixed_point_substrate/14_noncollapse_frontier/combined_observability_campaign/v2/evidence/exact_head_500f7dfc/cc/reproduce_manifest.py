import hashlib
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
entries = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(root).as_posix()
    entries.append((rel, hashlib.sha256(path.read_bytes()).hexdigest()))
entries.sort(key=lambda item: item[0].encode("utf-8"))
out.write_text("".join(f"{digest}  {rel}\n" for rel, digest in entries), encoding="utf-8", newline="")
print(json.dumps({"file_count": len(entries), "manifest": str(out)}, sort_keys=True))

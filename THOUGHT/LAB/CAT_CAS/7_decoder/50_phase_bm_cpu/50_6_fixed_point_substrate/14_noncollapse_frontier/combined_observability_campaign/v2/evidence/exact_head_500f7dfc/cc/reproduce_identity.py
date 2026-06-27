import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone
tar_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
identity_path = pathlib.Path(sys.argv[3])
commit = sys.argv[4]
tree = sys.argv[5]
archive_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
file_count = sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line)
identity = {
    "schema_id": "CAT_CAS_PHASE6_V2_SEALED_GIT_ARCHIVE_SNAPSHOT_REPRODUCTION_V1",
    "git_archive_commit": commit,
    "git_tree_sha": tree,
    "archive_filename": tar_path.name,
    "archive_size_bytes": tar_path.stat().st_size,
    "archive_sha256": archive_sha,
    "recursive_manifest_filename": manifest_path.name,
    "recursive_file_count": file_count,
    "recursive_manifest_sha256": manifest_sha,
    "created_timestamp": datetime.now(timezone.utc).isoformat(),
    "source_transport": "sealed git-archive snapshot reproduction",
    "generated_from": "fresh local git archive reproduction, not mutable working tree"
}
identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="ascii")
print(json.dumps(identity, sort_keys=True))

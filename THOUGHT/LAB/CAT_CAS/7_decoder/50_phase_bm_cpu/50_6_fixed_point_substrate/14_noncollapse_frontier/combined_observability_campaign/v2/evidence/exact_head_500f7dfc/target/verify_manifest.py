import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
expected = {}
for line in manifest.read_text(encoding='utf-8').splitlines():
    if not line:
        continue
    digest, rel = line.split('  ', 1)
    expected[rel] = digest
actual = {}
for path in root.rglob('*'):
    if path.is_file():
        rel = path.relative_to(root).as_posix()
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        actual[rel] = h.hexdigest()
missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
changed = sorted(rel for rel in set(expected) & set(actual) if expected[rel] != actual[rel])
print(f"expected_files={len(expected)}")
print(f"actual_files={len(actual)}")
print(f"missing_files={len(missing)}")
print(f"extra_files={len(extra)}")
print(f"changed_files={len(changed)}")
if missing:
    print("MISSING", missing[:20])
if extra:
    print("EXTRA", extra[:20])
if changed:
    print("CHANGED", changed[:20])
if missing or extra or changed:
    raise SystemExit(1)
print("MANIFEST_VERIFICATION_PASS")

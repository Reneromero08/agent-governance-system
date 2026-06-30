from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qualification.portable_package import (  # noqa: E402
    EXPECTED_SCOPED_TREE,
    PACKAGE_ROOT_DIR,
    export_portable_target_package,
)
from qualification import portable_target_qualification as portable  # noqa: E402


class PortableTargetQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.temp = Path(cls.tempdir.name)
        cls.left = cls.temp / "left.tar"
        cls.right = cls.temp / "right.tar"
        cls.left_result = export_portable_target_package(cls.left)
        cls.right_result = export_portable_target_package(cls.right)
        cls.extract_root = cls.temp / "extract"
        with tarfile.open(cls.left, "r") as archive:
            cls._assert_tar_members_safe(archive)
            archive.extractall(cls.extract_root)
        cls.package_root = cls.extract_root / PACKAGE_ROOT_DIR

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    @staticmethod
    def _assert_tar_members_safe(archive: tarfile.TarFile) -> None:
        names: list[str] = []
        for member in archive.getmembers():
            names.append(member.name)
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts:
                raise AssertionError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise AssertionError(f"unsupported archive member: {member.name}")
        folded: dict[str, str] = {}
        for name in names:
            key = name.casefold()
            previous = folded.get(key)
            if previous is not None and previous != name:
                raise AssertionError(f"case collision in archive: {previous} {name}")
            folded[key] = name

    def _copy_package(self) -> Path:
        target = self.temp / f"pkg-{len(list(self.temp.glob('pkg-*')))}"
        shutil.copytree(self.package_root, target, copy_function=shutil.copy2)
        return target

    def test_export_is_deterministic(self) -> None:
        self.assertEqual(self.left.read_bytes(), self.right.read_bytes())
        self.assertEqual(self.left_result["archive_sha256"], self.right_result["archive_sha256"])
        self.assertEqual(self.left_result["portable_manifest_sha256"], self.right_result["portable_manifest_sha256"])

    def test_package_contains_no_git_content(self) -> None:
        with tarfile.open(self.left, "r") as archive:
            names = [member.name for member in archive.getmembers()]
        self.assertFalse(any(name.endswith(".bundle") for name in names))
        self.assertFalse(any("/.git/" in f"/{name}/" or name.endswith("/.git") for name in names))

    def test_runner_source_does_not_call_git(self) -> None:
        source = (ROOT / "qualification" / "portable_target_qualification.py").read_text(encoding="utf-8")
        forbidden = ("git clone", "git checkout", "git bundle", "git archive", "git rev-parse", "git ls-tree", "git cat-file")
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertNotIn("subprocess.run([\"git\"", source)
        self.assertNotIn("import jsonschema", source)
        self.assertNotIn("from jsonschema", source)

    def test_copied_file_inventory_and_scoped_tree_match(self) -> None:
        manifest = portable.load_json(self.package_root / "PORTABLE_PACKAGE_MANIFEST.json")
        binding = portable.load_json(self.package_root / "TRUSTED_SNAPSHOT_BINDING.json")
        portable.validate_manifest(manifest)
        portable.validate_binding(binding)
        portable.verify_copied_files(self.package_root, manifest)
        result = portable.verify_snapshot(self.package_root, binding)
        self.assertEqual(result["observed_inventory_sha256"], portable.EXPECTED_INVENTORY_SHA256)
        self.assertEqual(result["calculated_scoped_tree"], EXPECTED_SCOPED_TREE)
        self.assertEqual(result["calculated_phase6b6_subtree_inventory_sha256"], portable.EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256)
        self.assertEqual(result["calculated_v2_source_sha256"], portable.EXPECTED_V2_SOURCE_SHA256)

    def test_blob_identities_match_manifest(self) -> None:
        manifest = portable.load_json(self.package_root / "PORTABLE_PACKAGE_MANIFEST.json")
        for item in manifest["copied_files"]:
            data = (self.package_root / item["path"]).read_bytes()
            self.assertEqual(portable.blob_sha1(data), item["git_blob_sha1"])

    @unittest.skipUnless(shutil.which("gcc") is not None, "native gcc is required for full portable verifier test")
    def test_runner_succeeds_with_git_hidden_and_jsonschema_unavailable(self) -> None:
        bindir = self.temp / "nogit-bin"
        bindir.mkdir(exist_ok=True)
        for tool in ("gcc", "cc", "as", "ld", "ar"):
            source = shutil.which(tool)
            if source:
                target = bindir / tool
                try:
                    target.symlink_to(source)
                except OSError:
                    shutil.copy2(source, target)
        env = os.environ.copy()
        env["PATH"] = str(bindir)
        hook = self.temp / "block-jsonschema"
        hook.mkdir(exist_ok=True)
        (hook / "sitecustomize.py").write_text(
            "import builtins\n"
            "_orig=__import__\n"
            "def guarded(name,*args,**kwargs):\n"
            "    if name == 'jsonschema' or name.startswith('jsonschema.'):\n"
            "        raise ImportError('jsonschema blocked for portable target test')\n"
            "    return _orig(name,*args,**kwargs)\n"
            "builtins.__import__=guarded\n",
            encoding="utf-8",
        )
        env["PYTHONPATH"] = str(hook)
        result = subprocess.run(
            [sys.executable, str(ROOT / "qualification" / "portable_target_qualification.py"), "--package-root", str(self.package_root)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "PHASE6B6_PORTABLE_TARGET_QUALIFICATION_PASS")
        self.assertEqual(payload["calculated_scoped_tree"], EXPECTED_SCOPED_TREE)

    def test_changed_copied_byte_fails(self) -> None:
        root = self._copy_package()
        target = next((root / "snapshot").rglob("contract.py"))
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        manifest = portable.load_json(root / "PORTABLE_PACKAGE_MANIFEST.json")
        with self.assertRaises(portable.PortableQualificationError):
            portable.verify_copied_files(root, manifest)

    def test_missing_file_fails(self) -> None:
        root = self._copy_package()
        target = next((root / "snapshot").rglob("contract.py"))
        target.unlink()
        manifest = portable.load_json(root / "PORTABLE_PACKAGE_MANIFEST.json")
        with self.assertRaises(portable.PortableQualificationError):
            portable.verify_copied_files(root, manifest)

    def test_extra_file_fails(self) -> None:
        root = self._copy_package()
        extra = root / "snapshot" / "extra.txt"
        extra.write_text("extra\n", encoding="utf-8")
        manifest = portable.load_json(root / "PORTABLE_PACKAGE_MANIFEST.json")
        with self.assertRaises(portable.PortableQualificationError):
            portable.verify_copied_files(root, manifest)

    def test_changed_mode_fails(self) -> None:
        root = self._copy_package()
        manifest = portable.load_json(root / "PORTABLE_PACKAGE_MANIFEST.json")
        target_record = next(item for item in manifest["copied_files"] if item["mode"] == "100644")
        target_record["mode"] = "100755"
        with self.assertRaises(portable.PortableQualificationError):
            portable.verify_copied_files(root, manifest)

    def test_symlink_fails(self) -> None:
        root = self._copy_package()
        target = next((root / "snapshot").rglob("contract.py"))
        backup = target.with_suffix(".bak")
        target.rename(backup)
        try:
            target.symlink_to(backup)
        except OSError:
            self.skipTest("symlink creation is unavailable")
        manifest = portable.load_json(root / "PORTABLE_PACKAGE_MANIFEST.json")
        with self.assertRaises(portable.PortableQualificationError):
            portable.verify_copied_files(root, manifest)

    def test_path_escape_fails(self) -> None:
        root = self._copy_package()
        manifest = portable.load_json(root / "PORTABLE_PACKAGE_MANIFEST.json")
        manifest["copied_files"][0]["path"] = "../escape"
        with self.assertRaises(portable.PortableQualificationError):
            portable.verify_copied_files(root, manifest)

    def test_duplicate_path_fails(self) -> None:
        manifest = portable.load_json(self.package_root / "PORTABLE_PACKAGE_MANIFEST.json")
        manifest["copied_files"].append(dict(manifest["copied_files"][0]))
        with self.assertRaises(portable.PortableQualificationError):
            portable.verify_copied_files(self.package_root, manifest)

    def test_case_collision_fails(self) -> None:
        root = self._copy_package()
        existing = next((root / "snapshot").rglob("contract.py"))
        collision = existing.with_name(existing.name.upper())
        if collision == existing:
            self.skipTest("case collision cannot be represented on this filesystem")
        collision.write_bytes(existing.read_bytes())
        manifest = portable.load_json(root / "PORTABLE_PACKAGE_MANIFEST.json")
        with self.assertRaises(portable.PortableQualificationError):
            portable.verify_copied_files(root, manifest)

    def test_changed_trusted_binding_fails(self) -> None:
        root = self._copy_package()
        binding_path = root / "TRUSTED_SNAPSHOT_BINDING.json"
        binding = portable.load_json(binding_path)
        binding["expected_scoped_tree"] = "0" * 40
        binding_path.write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")
        with self.assertRaises(portable.PortableQualificationError):
            portable.validate_binding(portable.load_json(binding_path))

    def test_changed_archive_manifest_fails(self) -> None:
        root = self._copy_package()
        manifest_path = root / "PORTABLE_PACKAGE_MANIFEST.json"
        manifest = portable.load_json(manifest_path)
        manifest["target_executes_git"] = True
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        with self.assertRaises(portable.PortableQualificationError):
            portable._read_manifest_sha(root)

    @unittest.skipUnless(shutil.which("gcc") is not None, "native gcc is required for C mismatch test")
    def test_changed_c_table_fails(self) -> None:
        root = self._copy_package()
        source = root / "emit_v2_reference_table.c"
        text = source.read_text(encoding="utf-8")
        mutated = text.replace('printf("\\"tone_count\\":12,");', 'printf("\\"tone_count\\":11,");')
        self.assertNotEqual(mutated, text)
        source.write_text(mutated, encoding="utf-8")
        manifest = portable.load_json(root / "PORTABLE_PACKAGE_MANIFEST.json")
        data = source.read_bytes()
        record = next(item for item in manifest["copied_files"] if item["path"] == "emit_v2_reference_table.c")
        record["size"] = len(data)
        record["sha256"] = hashlib.sha256(data).hexdigest()
        record["git_blob_sha1"] = portable.blob_sha1(data)
        binding = portable.load_json(root / "TRUSTED_SNAPSHOT_BINDING.json")
        portable.verify_copied_files(root, manifest)
        portable.verify_snapshot(root, binding)
        with self.assertRaises(portable.PortableQualificationError):
            portable.compare_reference(portable.compile_and_emit(root))

    def test_hardware_options_fail_closed(self) -> None:
        with self.assertRaises(portable.PortableQualificationError):
            portable.hardware_rejection(["--hardware"])


if __name__ == "__main__":
    unittest.main()

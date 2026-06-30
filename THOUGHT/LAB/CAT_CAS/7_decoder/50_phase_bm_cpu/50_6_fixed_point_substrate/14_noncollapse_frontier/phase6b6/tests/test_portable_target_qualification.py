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
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qualification import portable_package  # noqa: E402
from qualification import portable_target_qualification as portable  # noqa: E402
from qualification.portable_package import (  # noqa: E402
    EXPECTED_SCOPED_TREE,
    PACKAGE_ROOT_DIR,
    export_portable_target_package,
)


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

    def _rewrite_manifest(self, root: Path, manifest: dict[str, object]) -> None:
        manifest_path = root / "PORTABLE_PACKAGE_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (root / "PORTABLE_PACKAGE_MANIFEST.sha256").write_text(
            f"{digest}  PORTABLE_PACKAGE_MANIFEST.json\n",
            encoding="ascii",
        )

    def _expect_manifest_rejected(self, mutator) -> None:
        manifest = portable.load_json(self.package_root / "PORTABLE_PACKAGE_MANIFEST.json")
        mutator(manifest)
        with self.assertRaises(portable.PortableQualificationError):
            portable.validate_manifest(manifest)

    def _expect_binding_rejected(self, mutator) -> None:
        binding = portable.load_json(self.package_root / "TRUSTED_SNAPSHOT_BINDING.json")
        mutator(binding)
        with self.assertRaises(portable.PortableQualificationError):
            portable.validate_binding(binding)

    def _expect_contract_rejected(self, mutator) -> None:
        contract = portable.load_json(self.package_root / "QUALIFICATION_CONTRACT.json")
        mutator(contract)
        with self.assertRaises(portable.PortableQualificationError):
            portable.validate_contract(contract)

    def _assert_dirty_export_rejected(self, source: Path) -> None:
        original = source.read_bytes()
        try:
            source.write_bytes(original + b"\n")
            with self.assertRaises(Exception):
                export_portable_target_package(self.temp / f"dirty-{source.name}.tar")
        finally:
            source.write_bytes(original)

    def test_export_is_deterministic(self) -> None:
        self.assertEqual(self.left.read_bytes(), self.right.read_bytes())
        self.assertEqual(self.left_result["archive_sha256"], self.right_result["archive_sha256"])
        self.assertEqual(self.left_result["portable_manifest_sha256"], self.right_result["portable_manifest_sha256"])

    def test_package_contains_no_git_content(self) -> None:
        with tarfile.open(self.left, "r") as archive:
            names = [member.name for member in archive.getmembers()]
        self.assertFalse(any(name.endswith(".bundle") for name in names))
        self.assertFalse(any("/.git/" in f"/{name}/" or name.endswith("/.git") for name in names))

    def test_export_binds_exact_head_and_support_blobs(self) -> None:
        manifest = portable.load_json(self.package_root / "PORTABLE_PACKAGE_MANIFEST.json")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parents[7], text=True).strip()
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT.parents[7], text=True).strip()
        self.assertEqual(manifest["portable_export_commit"], head)
        self.assertEqual(manifest["portable_export_tree"], tree)
        support = manifest["portable_support_blob_bindings"]
        self.assertEqual([item["role"] for item in support], ["c_reference_emitter", "portable_target_runner"])
        for item in support:
            blob = subprocess.check_output(["git", "rev-parse", f"HEAD:{item['source_path']}"], cwd=ROOT.parents[7], text=True).strip()
            self.assertEqual(item["source_blob_sha1"], blob)
            self.assertEqual(item["source_commit"], head)
            self.assertEqual(item["source_tree"], tree)

    def test_tracked_portable_runner_modified_in_worktree_blocks_export(self) -> None:
        self._assert_dirty_export_rejected(ROOT / "qualification" / "portable_target_qualification.py")

    def test_tracked_emitter_modified_in_worktree_blocks_export(self) -> None:
        self._assert_dirty_export_rejected(ROOT / "qualification" / "emit_v2_reference_table.c")

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
            self.assertEqual(portable.blob_sha1(data), item["source_blob_sha1"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["content_sha256"])

    def test_manifest_rejects_forged_portable_source_commit(self) -> None:
        self._expect_manifest_rejected(lambda manifest: manifest.update({"portable_export_commit": "0" * 40}))

    def test_manifest_rejects_forged_portable_source_tree(self) -> None:
        self._expect_manifest_rejected(lambda manifest: manifest.update({"portable_export_tree": "0" * 40}))

    def test_manifest_rejects_support_blob_not_belonging_to_claimed_commit(self) -> None:
        def mutate(manifest: dict[str, object]) -> None:
            manifest["copied_files"][0]["source_commit"] = manifest["portable_export_commit"]  # type: ignore[index]

        self._expect_manifest_rejected(mutate)

    def test_manifest_rejects_stale_base_qualification_identity(self) -> None:
        self._expect_manifest_rejected(lambda manifest: manifest.update({"base_qualification_merge": "0" * 40}))

    def test_manifest_rejects_changed_snapshot_subject_commit(self) -> None:
        self._expect_manifest_rejected(lambda manifest: manifest.update({"snapshot_subject_commit": "0" * 40}))

    def test_manifest_rejects_changed_snapshot_subject_tree(self) -> None:
        self._expect_manifest_rejected(lambda manifest: manifest.update({"snapshot_subject_tree": "0" * 40}))

    def test_manifest_rejects_changed_package_root(self) -> None:
        self._expect_manifest_rejected(lambda manifest: manifest.update({"package_root": "other"}))

    def test_manifest_rejects_changed_qualification_scope(self) -> None:
        def mutate(manifest: dict[str, object]) -> None:
            manifest["portable_qualification_scope"] = list(manifest["portable_qualification_scope"])[:-1]  # type: ignore[index]

        self._expect_manifest_rejected(mutate)

    def test_manifest_rejects_changed_snapshot_file_count(self) -> None:
        self._expect_manifest_rejected(lambda manifest: manifest.update({"snapshot_file_count": 31}))

    def test_manifest_rejects_unknown_copied_file_field(self) -> None:
        def mutate(manifest: dict[str, object]) -> None:
            manifest["copied_files"][0]["extra"] = True  # type: ignore[index]

        self._expect_manifest_rejected(mutate)

    def test_manifest_rejects_invalid_role_path_combination(self) -> None:
        def mutate(manifest: dict[str, object]) -> None:
            record = next(item for item in manifest["copied_files"] if item["role"] == "portable_target_runner")  # type: ignore[index]
            record["path"] = "wrong.py"

        self._expect_manifest_rejected(mutate)

    def test_binding_rejects_changed_subject_commit_and_tree(self) -> None:
        self._expect_binding_rejected(lambda binding: binding.update({"snapshot_subject_commit": "0" * 40}))
        self._expect_binding_rejected(lambda binding: binding.update({"snapshot_subject_tree": "0" * 40}))

    def test_contract_rejects_changed_digests_and_authority_shape(self) -> None:
        self._expect_contract_rejected(lambda contract: contract.update({"qualification_contract_sha256": "0" * 64}))
        self._expect_contract_rejected(lambda contract: contract.update({"schedule_digest": "0" * 64}))
        self._expect_contract_rejected(lambda contract: contract.update({"mock_custody_digest": "0" * 64}))

        def mutate_authority(contract: dict[str, object]) -> None:
            contract["authority_state"]["unexpected"] = False  # type: ignore[index]

        self._expect_contract_rejected(mutate_authority)

        def mutate_nested(contract: dict[str, object]) -> None:
            contract["unexpected_nested"] = True

        self._expect_contract_rejected(mutate_nested)

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
        self.assertEqual(payload["runtime_validate_only"]["status"], "PHASE6B6_PORTABLE_RUNTIME_VALIDATE_ONLY_OK")
        self.assertEqual(payload["runtime_validate_only"]["schedule_digest"], portable.SCHEDULE_DIGEST)
        self.assertEqual(payload["runtime_validate_only"]["mock_custody_digest"], portable.MOCK_CUSTODY_DIGEST)
        self.assertEqual(payload["target_executed_git"], False)
        self.assertEqual(payload["jsonschema_required"], False)
        self.assertRegex(payload["raw_c_emission_sha256"], r"^[0-9a-f]{64}$")
        portable.validate_final_result(payload)

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
        record["content_sha256"] = hashlib.sha256(data).hexdigest()
        record["source_blob_sha1"] = portable.blob_sha1(data)
        binding = portable.load_json(root / "TRUSTED_SNAPSHOT_BINDING.json")
        portable.verify_copied_files(root, manifest)
        portable.verify_snapshot(root, binding)
        with self.assertRaises(portable.PortableQualificationError):
            portable.compare_reference(portable.compile_and_emit(root)["payload"])

    def test_c_reference_rejects_missing_extra_wrong_and_nonfinite_tones(self) -> None:
        payload = portable.python_reference_table()
        missing = json.loads(json.dumps(payload))
        missing["tones"] = missing["tones"][:-1]
        with self.assertRaises(portable.PortableQualificationError):
            portable.validate_c_reference(missing)

        extra = json.loads(json.dumps(payload))
        extra["tones"].append(dict(extra["tones"][-1]))
        with self.assertRaises(portable.PortableQualificationError):
            portable.validate_c_reference(extra)

        wrong = json.loads(json.dumps(payload))
        wrong["tones"][3]["codeword_source_index"] = 9
        with self.assertRaises(portable.PortableQualificationError):
            portable.validate_c_reference(wrong)

        unknown = json.loads(json.dumps(payload))
        unknown["unexpected"] = True
        with self.assertRaises(portable.PortableQualificationError):
            portable.validate_c_reference(unknown)

        with self.assertRaises(portable.PortableQualificationError):
            portable.parse_json_bytes(b'{"schema_id":"x","frequency_hz":NaN}')
        with self.assertRaises(portable.PortableQualificationError):
            portable.parse_json_bytes(b'{"schema_id":"x","frequency_hz":Infinity}')

    def test_canonical_equal_but_byte_different_c_emissions_fail(self) -> None:
        payload = portable.python_reference_table()
        raw_a = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        raw_b = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
        self.assertEqual(portable.canonical_json(portable.parse_json_bytes(raw_a)), portable.canonical_json(portable.parse_json_bytes(raw_b)))
        fake_a = {
            "status": "PHASE6B6_PORTABLE_C_REFERENCE_EMIT_OK",
            "mode": "strict",
            "raw_stdout_sha256": hashlib.sha256(raw_a).hexdigest(),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "payload": portable.parse_json_bytes(raw_a),
            "raw_stdout": raw_a,
        }
        fake_b = dict(fake_a)
        fake_b["raw_stdout"] = raw_b
        fake_b["raw_stdout_sha256"] = hashlib.sha256(raw_b).hexdigest()
        with mock.patch.object(portable, "compile_and_emit", side_effect=[fake_a, fake_b]):
            with self.assertRaises(portable.PortableQualificationError):
                portable.run(self.package_root, [])

    @unittest.skipUnless(shutil.which("gcc") is not None, "native gcc is required for temp cleanup test")
    def test_temporary_compiler_artifacts_are_removed_on_pass_and_fail(self) -> None:
        root = self._copy_package()
        temp_root = self.temp / "compiler-temp-root"
        temp_root.mkdir(exist_ok=True)
        before = set(temp_root.iterdir())
        old = os.environ.get("PHASE6B6_PORTABLE_C_TEMP_ROOT")
        os.environ["PHASE6B6_PORTABLE_C_TEMP_ROOT"] = str(temp_root)
        try:
            portable.compile_and_emit(root)
            self.assertEqual(set(temp_root.iterdir()), before)
            source = root / "emit_v2_reference_table.c"
            source.write_text("not c\n", encoding="utf-8")
            with self.assertRaises(portable.PortableQualificationError):
                portable.compile_and_emit(root)
            self.assertEqual(set(temp_root.iterdir()), before)
        finally:
            if old is None:
                os.environ.pop("PHASE6B6_PORTABLE_C_TEMP_ROOT", None)
            else:
                os.environ["PHASE6B6_PORTABLE_C_TEMP_ROOT"] = old

    def test_option_rejection_is_closed(self) -> None:
        for option in (
            "--hardware",
            "--hardware=true",
            "--acquire",
            "--acquire=yes",
            "--calibrate",
            "--calibrate=1",
            "--run-campaign",
            "--run-campaign=true",
            "--unknown",
        ):
            with self.subTest(option=option):
                with self.assertRaises(portable.PortableQualificationError):
                    portable.hardware_rejection([option])

    @unittest.skipUnless(Path("/proc").is_dir(), "process absence proof requires /proc")
    def test_forbidden_process_argument_is_detected(self) -> None:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "portable_target_qualification_forbidden_probe"])
        try:
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    with self.assertRaises(portable.PortableQualificationError):
                        portable.sender_absence_probe()
                    return
                except AssertionError:
                    time.sleep(0.1)
            self.fail("forbidden process was not detected")
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_hardware_options_fail_closed(self) -> None:
        with self.assertRaises(portable.PortableQualificationError):
            portable.hardware_rejection(["--hardware"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class CapturedFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temp.name) / "captured_file_fixture"
        result = subprocess.run(
            ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
             str(HERE / "captured_file_fixture.c"), str(HERE / "captured_file.c"),
             "-o", str(cls.binary)],
            text=True, capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def run_fixture(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run([str(self.binary), *args], capture_output=True)

    def test_sha256_known_vectors(self) -> None:
        vectors = (
            "",
            "abc",
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        )
        for value in vectors:
            with self.subTest(value=value):
                result = self.run_fixture("hash", value)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(result.stdout.decode().strip(), hashlib.sha256(value.encode()).hexdigest())

    def test_capture_survives_original_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "input"
            replacement = root / "replacement"
            original.write_bytes(b"before-custody")
            replacement.write_bytes(b"after-replacement")
            result = self.run_fixture("capture-replace", str(original), str(replacement))
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            digest, size, payload = result.stdout.split(b"\n", 2)
            self.assertEqual(int(size), len(b"before-custody"))
            self.assertEqual(payload, b"before-custody")
            self.assertEqual(digest.decode(), hashlib.sha256(payload).hexdigest())
            self.assertEqual(original.read_bytes(), b"after-replacement")

    @unittest.skipIf(os.name == "nt", "O_NOFOLLOW is a Linux custody requirement")
    def test_symlink_capture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "target"
            link = root / "link"
            target.write_bytes(b"bound")
            link.symlink_to(target.name)
            result = self.run_fixture("capture", str(link))
            self.assertNotEqual(result.returncode, 0)

    def test_exclusive_writer_preserves_bytes_and_rejects_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            destination = root / "destination"
            source.write_bytes(b"captured-output")
            first = self.run_fixture("write", str(source), str(destination))
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            second = self.run_fixture("write", str(source), str(destination))
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(destination.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)

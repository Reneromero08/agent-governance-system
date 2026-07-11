#!/usr/bin/env python3
"""Exact local/remote equivalence tests for Gate A process custody."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import types
import unittest
from typing import Any

import gate_a_process_custody as process_custody


def local_receipt(stdout: bytes) -> dict[str, Any]:
    completed = subprocess.CompletedProcess(
        list(process_custody.PROCESS_COMMAND),
        0,
        stdout=stdout,
        stderr=b"",
    )
    return process_custody.scan_processes(
        "post_cleanup",
        runner=lambda *_args, **_kwargs: completed,
    )


def remote_receipt(stdout: bytes) -> dict[str, Any]:
    original = sys.modules.get("subprocess")
    fake = types.ModuleType("subprocess")
    fake.PIPE = subprocess.PIPE
    fake.TimeoutExpired = subprocess.TimeoutExpired
    fake.run = lambda *_args, **_kwargs: subprocess.CompletedProcess(
        list(process_custody.PROCESS_COMMAND),
        0,
        stdout=stdout,
        stderr=b"",
    )
    sys.modules["subprocess"] = fake
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(process_custody.render_remote_scan_script("post_cleanup"), {})
    finally:
        if original is None:
            del sys.modules["subprocess"]
        else:
            sys.modules["subprocess"] = original
    return json.loads(output.getvalue())


class ProcessCustodyEquivalenceTests(unittest.TestCase):
    def test_local_and_remote_receipts_are_exactly_equivalent(self) -> None:
        cases = (
            (b"1 init /sbin/init\n", None),
            (b"", "PROCESS_STDOUT_EMPTY"),
            (b"\xff", "PROCESS_STDOUT_NOT_UTF8"),
            (b"malformed\n", "PROCESS_STDOUT_MALFORMED"),
        )
        for stdout, expected_failure in cases:
            with self.subTest(stdout=stdout, expected_failure=expected_failure):
                local = local_receipt(stdout)
                remote = remote_receipt(stdout)
                self.assertEqual(local, remote)
                self.assertEqual(local["failure"], expected_failure)
                self.assertEqual(local["scan_complete"], expected_failure is None)
                self.assertEqual(local["forbidden_filter_evaluated"], expected_failure is None)


if __name__ == "__main__":
    unittest.main()

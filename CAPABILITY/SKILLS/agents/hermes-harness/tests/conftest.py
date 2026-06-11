"""Test-suite safety net: NO test may ever reach the live Hermes agent.

Background: the Worker API control plane talks to Hermes through
hermes_harness.call_hermes_responses -> urllib.request.urlopen. If a test
constructs a WorkerController without injecting a fake caller (or monkeypatches
the wrong module), and HERMES_API_KEY/API_SERVER_KEY is set in the environment,
it would make a REAL agent call -- burning tokens and producing unwanted repo
artifacts (e.g. THOUGHT/LAB/CAT_CAS/phaseX/AUDIT_*.md).

This autouse fixture blocks urllib.request.urlopen for every test. Tests that
legitimately exercise the HTTP layer (e.g. test_session_key_header_sent) set
their own urlopen via monkeypatch AFTER this fixture runs, so their fake wins.
Any accidental live call instead fails loudly with a clear message.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(autouse=True)
def block_live_hermes(monkeypatch):
    import urllib.request

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "BLOCKED: a test attempted a live network call (Hermes agent). "
            "Tests must inject a fake caller via WorkerController(caller=...) or "
            "monkeypatch the transport. No test may spend real tokens."
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    yield

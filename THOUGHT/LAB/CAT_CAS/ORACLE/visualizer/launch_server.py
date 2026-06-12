"""Detached server launcher: starts server.py in a fully-detached subprocess."""
import subprocess
import sys
import os
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = str(next(p for p in Path(__file__).resolve().parents if (p / ".git").exists()) / ".venv" / "Scripts" / "python.exe")
SERVER_PY = os.path.join(HERE, "server.py")
LOG = os.path.join(HERE, "server.log")
ERR = os.path.join(HERE, "server.err")

# DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
DETACHED = 0x00000008
NEW_PG = 0x00000200

log_f = open(LOG, "ab")
err_f = open(ERR, "ab")
p = subprocess.Popen(
    [VENV_PY, "-u", SERVER_PY],
    stdout=log_f, stderr=err_f, stdin=subprocess.DEVNULL,
    creationflags=DETACHED | NEW_PG,
    cwd=HERE,
    close_fds=True,
)
print(f"server pid: {p.pid}")

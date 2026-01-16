---
name: launch-terminal
version: "0.1.0"
description: Opens a named terminal tab in VSCode via Antigravity Bridge (port 4001). Use when spawning agent terminals for swarm workers.
compatibility: VSCode, Antigravity Bridge extension
---

<!-- CONTENT_HASH: 8f370f438786a09460b00a58ad9a1b524a6e33256cc90003640fbae121049dac -->



# Launch Terminal

Spawns VSCode terminals via Antigravity Bridge.

## Usage

```bash
python scripts/run.py input.json output.json
```

## Input Schema

```json
{
  "name": "Governor",
  "command": "python poll_and_execute.py --role Governor",
  "cwd": "D:/path/to/CATALYTIC-DPT"
}
```

## Requirements

Antigravity Bridge must be running on `http://127.0.0.1:4000/terminal`.

If bridge is down, inform user - do not bypass via OS-level spawning (INV-014).
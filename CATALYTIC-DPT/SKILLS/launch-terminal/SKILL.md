---
name: launch-terminal
description: Opens a named terminal tab directly in the user's IDE bottom panel using the Antigravity Bridge.
---

# Launch Terminal Skill

This skill allows the agent to spawn interactive terminals directly in the user's IDE panel (not the chat area).

## 🚀 How it Works
It uses the `antigravity-bridge` extension (listening on local port 4000) to trigger VS Code's native `createTerminal` API.

## Usage
This skill is executed via the standard runner pattern:

```bash
python SKILLS/launch-terminal/run.py <input.json> <output.json>
```

### Input Schema
```json
{
  "name": "Terminal Title",
  "command": "Initial command to run",
  "cwd": "Absolute working directory"
}
```

### Parameters
You can customize the JSON body:
- `name`: Title of the terminal tab
- `cwd`: Working directory (absolute path)
- `initialCommand`: Optional command to run immediately on startup

## 📵 Governance Constraint
As of Canon v2.7.1, the **Legacy Fallback** (external window spawning) is **PROHIBITED** by [INV-014]. If the bridge is down, the agent must inform the user and request a bridge restart, rather than bypassing security via OS-level window spawning.

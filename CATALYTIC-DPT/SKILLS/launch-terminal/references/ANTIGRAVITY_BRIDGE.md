# Antigravity Bridge Extension

> Local copy of extension from `D:\CCC 2.0\AI\AGI\EXTENSIONS\antigravity-bridge`

## Purpose

Exposes VSCode IDE capabilities to local agents via HTTP. Enables Claude and other agents to spawn terminals **inside VSCode** rather than external windows.

## Installation

The extension is pre-built as a VSIX:
```
D:\CCC 2.0\AI\AGI\EXTENSIONS\antigravity-bridge\antigravity-bridge-0.1.0.vsix
```

Install via:
```
code --install-extension antigravity-bridge-0.1.0.vsix
```

## How It Works

1. Extension starts HTTP server on **port 4000** (localhost only)
2. Agents POST to `http://127.0.0.1:4000/terminal`
3. Extension calls `vscode.window.createTerminal()`
4. Terminal appears in VSCode panel

## API

### POST /terminal

Create a new terminal in VSCode:

```json
{
  "name": "Terminal Title",
  "cwd": "/absolute/working/directory",
  "initialCommand": "command to run on startup"
}
```

**Response**:
```json
{"status": "success", "message": "Terminal 'Terminal Title' created"}
```

## Usage from Python

```python
import requests

response = requests.post("http://127.0.0.1:4000/terminal", json={
    "name": "Governor",
    "cwd": "d:/CCC 2.0/AI/agent-governance-system/CATALYTIC-DPT",
    "initialCommand": "gemini --prompt 'You are the Governor...'"
})
print(response.json())
```

## Source Location

Original source: `D:\CCC 2.0\AI\AGI\EXTENSIONS\antigravity-bridge\src\extension.ts`

## Governance

> **INV-014**: External window spawning is PROHIBITED
> 
> All agent terminals MUST use this bridge. No `Start-Process wt`, no external windows.

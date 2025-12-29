# AGI Hardener Skill

**Version:** 1.0.0

**Status:** Experimental

**Required_Canon_Version:** >=2.0.0

## Purpose

Hardens the external AGI repository (`D:/CCC 2.0/AI/AGI`) to follow AGS engineering standards.

## What It Does

Applies automated fixes to Python files:

1. **Bare Excepts**: Converts `except:` to `except Exception as e:` with logging
2. **UTF-8 Encoding**: Adds `encoding='utf-8'` to `open()` calls
3. **Headless Execution**: Replaces `input()` calls with batch mode responses
4. **Atomic Writes**: All file modifications use temp-file + atomic rename

## Usage

```bash
python SKILLS/agi-hardener/run.py
```

## Target Files

- `AGI/SKILLS/ant/run.py`
- `AGI/SKILLS/swarm-governor/run.py`
- `AGI/SKILLS/research-ingest/run.py`
- `AGI/SKILLS/launch-terminal/run.py`
- `AGI/MCP/server.py`

## Dependencies

- External AGI repository at `D:/CCC 2.0/AI/AGI`

## Limitations

- Targets hardcoded file paths (external repo specific)
- Uses raw Path access (known governance deviation for external repos)

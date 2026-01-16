# Oracle Tool - AI-to-AI Communication

## What Is It?

The **Oracle tool** lets your local model ask another AI for help when it encounters questions it can't answer. Think of it as "phone a friend" for AI models.

## How It Works

Your local model (Nemotron 3 Nano 30B) can call:
```python
oracle("What are the latest developments in quantum computing?")
```

This forwards the question to a more powerful AI and returns the answer.

## Configuration Options

### Option 1: DuckDuckGo AI Chat (FREE, Recommended)

**Pros:**
- No API key needed
- Free to use
- Access to Claude or GPT

**Setup:**
```bash
pip install -U duckduckgo-search
```

**Status:** Configured as default, but may hit rate limits

### Option 2: OpenAI API (Paid)

**Pros:**
- Reliable, fast
- High rate limits
- GPT-4 access

**Setup:**
```bash
export OPENAI_API_KEY="your-key-here"
```

Edit `oracle_bridge.py`:
```python
class OracleConfig:
    PREFERRED_ORACLE = "openai"
```

### Option 3: Anthropic API (Paid)

**Pros:**
- Direct Claude access
- Reliable
- High quality

**Setup:**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Edit `oracle_bridge.py`:
```python
class OracleConfig:
    PREFERRED_ORACLE = "anthropic"
```

### Option 4: Local LLM (Free)

**Pros:**
- Completely local
- No API costs
- Private

**Setup:**
Run another LLM (like Claude via LM Studio) on port 1234.

Edit `oracle_bridge.py`:
```python
class OracleConfig:
    LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"
    LOCAL_LLM_MODEL = "your-model-name"
    PREFERRED_ORACLE = "local"
```

## When Should The Model Use Oracle?

The system prompt tells the model to use oracle for:
- Questions beyond its knowledge cutoff
- Deep reasoning it's not confident about
- Second opinion or verification
- Tasks outside its core competencies

**Example use cases:**
- "What happened in the 2024 elections?" (beyond knowledge cutoff)
- "Verify this proof of Fermat's Last Theorem" (needs verification)
- "What's the latest research on AGI safety?" (current developments)

## Current Status

**Integrated:** Yes ✓
**Default:** DuckDuckGo AI Chat
**Fallback:** OpenAI > Anthropic > Local LLM

The model now has `oracle()` as tool #5 in its available tools.

## Testing

Test the oracle connection:
```bash
cd THOUGHT/LAB/MODEL_TESTS
python oracle_bridge.py
```

Expected output:
- DuckDuckGo: May hit rate limits ("ERR_BN_LIMIT")
- OpenAI/Anthropic: Requires API key
- Local: Requires LLM server running

## Rate Limits & Workarounds

**DuckDuckGo:**
- Rate limited after a few requests
- Workaround: Set up OpenAI/Anthropic API or local LLM

**OpenAI:**
- Tier-based rate limits (depends on your plan)
- Workaround: Upgrade API tier or use local LLM

**Anthropic:**
- Similar to OpenAI
- Workaround: Use OpenAI or local LLM

**Local LLM:**
- Only limited by your hardware
- Recommended for high-volume usage

## Architecture

```
Nemotron 3 Nano 30B (Local)
         |
         | oracle("question")
         v
   oracle_bridge.py
         |
    /----+----\
    |    |    |
    v    v    v
  DDG  API  Local
 Chat       LLM
```

## Files

- **tool_executor_v2.py** - Main framework with oracle integration
- **oracle_bridge.py** - Oracle routing logic (DDG, OpenAI, Anthropic, Local)
- **ORACLE_SETUP.md** - This file

## Example Session

```
User: "Ask the oracle what the latest developments in quantum computing are"

Nemotron: *recognizes this needs current info*
oracle("What are the latest developments in quantum computing?")

Oracle (via DuckDuckGo/Claude): *provides 2024/2025 developments*

Nemotron: *incorporates oracle's response into final answer*
```

## Recommendations

**For casual use:** Stick with DuckDuckGo (free, occasionally rate-limited)

**For serious work:** Set up OpenAI API key ($5/month minimum usually covers it)

**For privacy/high-volume:** Run local LLM (Claude 3.5 Sonnet via LM Studio)

**Best setup:** Configure OpenAI as fallback when DDG hits rate limit:
```python
class OracleConfig:
    PREFERRED_ORACLE = "duckduckgo"  # Try DDG first
    OPENAI_API_KEY = "sk-..."  # Fall back to OpenAI if DDG fails
```

This gives you free usage most of the time, with paid fallback for reliability.

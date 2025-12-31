# Symbolic Chat Encoding

## What is it?

Chat that uses **symbolic encoding** instead of full English text. Saves 30-70% of tokens.

## How it works

```
┌───────────────────────────────────┐
│  Symbol Dictionary               │
│  s001 = "hello world"         │  │ Chat: "s001 s002"         │
│  s002 = "how are you"         │  └───────┬─────────────┘
│  s003 = "thank you"           │          │
│  ...                            │          │
└───────────────────────────────────┘          │
                                    ↓
                            You translate
                          "hello world how are you"
```

## Token Savings

| Message Type | Example | Tokens | Saved |
|-------------|---------|--------|--------|
| English | "hello world how are you" | 8 | 0% |
| Symbols | "s001 s002" | 3 | **62.5%** |
| English | "please check logs for errors" | 6 | 0% |
| Symbols | "s004 s005" | 2 | **66.7%** |

## Usage

### 1. Build your symbol dictionary

Edit `symbols/dictionary.json`:

```json
{
  "symbols": {
    "common": {
      "s001": "hello world",
      "s002": "thank you",
      "s003": "please",
      "s004": "sorry"
    },
    "governance": {
      "g001": "AGI hardener",
      "g002": "Canon check",
      "g003": "Invariant freeze"
    },
    "status": {
      "st001": "success",
      "st002": "failed",
      "st003": "in progress"
    }
  }
}
```

### 2. Chat writes in symbols

Chat uses `enhanced_symbolic_chat.py`:

```python
from enhanced_symbolic_chat import SymbolicChatWriter

writer = SymbolicChatWriter()

# Auto-encode to symbols
session_id, symbols = writer.encode_message(
    session_id="my-session",
    role="assistant",
    content="hello world thank you",
    auto_add_missing=True  # Auto-create new symbols
)
```

### 3. You translate symbols to understand

```python
# Translate message
english = writer.translate_message(message_uuid)
# Returns: "hello world thank you"
```

### 4. Track savings

```python
# Calculate savings
savings = writer.get_token_savings("my-session")

print(f"Saved {savings['savings_percent']}% tokens")
print(f"{savings['tokens_saved']} fewer tokens needed")
```

## Demo

Run the simple demo:

```bash
cd CATALYTIC-DPT/LAB/CHAT_SYSTEM
py simple_symbolic_demo.py
```

Output:

```
============================================================
Symbolic Chat Demo
============================================================

[Dictionary]
  s001 = 'hello world'
  s002 = 'thank you'
  s003 = 'how are you'
  ... (8 total)

[Chat - English]
  User: 'hello world how are you doing well'
  Tokens: ~7

[Chat - Symbolic]
  Assistant: 's001 s003 s004 s002'
  Tokens: 4
  Savings: 42.9%

[Translation]
  Decoded: 'hello world how are you thank you'
============================================================
```

## Building a Better Dictionary

1. **Start with common phrases**
   - "hello world", "thank you", "goodbye"
   - "please check", "sorry about that"
   - "yes", "no", "maybe"

2. **Add technical terms**
   - "AGI hardener", "Canon check", "swarm governor"
   - "pipeline execution", "catalytic elision"

3. **Add status codes**
   - "success", "failed", "in progress", "completed"
   - "not found", "permission denied"

4. **Add error types**
   - "file not found", "connection failed", "timeout"
   - "invalid input", "missing dependency"

5. **Track usage**
   - Log which symbols are used most
   - Add new symbols when patterns emerge
   - Remove unused symbols to save space

## Best Practices

1. **Keep symbols short**: 3-6 characters
   - s001, s002, g001, st001
   - Shorter = more tokens saved

2. **Use consistent prefixes**
   - s = common phrases
   - g = governance terms
   - a = actions
   - e = errors
   - t = technical terms

3. **Make symbols intuitive**
   - s001 = "hello world" (obvious)
   - g001 = "AGI hardener" (matches pattern)
   - st001 = "success" (single word)

4. **Document symbol meanings**
   - Update `dictionary.json` with descriptions
   - Create a symbol glossary for reference
   - Share symbols across your team

5. **Balance compression vs clarity**
   - Too much compression = hard to decode
   - Too little compression = no token savings
   - Target: 50-70% token savings

## Example Symbol Glossary

| Symbol | Meaning | Category | Used |
|--------|---------|----------|-------|
| s001 | hello world | common | daily |
| s002 | thank you | common | daily |
| s003 | how are you | common | daily |
| g001 | AGI hardener | governance | sometimes |
| g002 | Canon check | governance | sometimes |
| st001 | success | status | very frequent |
| st002 | failed | status | sometimes |
| e001 | file not found | error | occasional |

## Advantages

- **Token cost savings**: 30-70% per message
- **Privacy**: Symbols are meaningless without dictionary
- **Compression**: Dense information in short codes
- **Consistency**: Standardized terminology
- **Speed**: Faster transmission (less text)

## Disadvantages

- **Requires dictionary**: Need reference to decode
- **Less readable**: Symbols aren't self-explanatory
- **Learning curve**: Team must learn symbols
- **Context loss**: Some nuance lost in encoding

## When to Use

✅ **USE symbolic encoding** when:
- Working with established domain (governance, technical)
- Repetitive patterns in conversations
- Cost-sensitive operations (frequent chats)
- Team has shared symbol dictionary

❌ **DON'T use** when:
- Ad-hoc one-off conversations
- Domain-specific terminology (no symbols yet)
- Cross-team communication
- Need for immediate clarity (learning cost > benefit)

## Integration with Existing Chat

You can use symbolic encoding WITH your current chat:

1. Start new session with `encoding="symbolic"`
2. Chat writes in symbols for common phrases
3. Switch to English when clarity needed
4. Mix symbolic and English in same session

Example:

```python
# Message 1: Greeting (symbolic)
writer.encode_message("session", "assistant", "s001")

# Message 2: Explanation (English)
writer.encode_message("session", "assistant", "I can help you check Canon invariants")

# Message 3: Status (symbolic)
writer.encode_message("session", "assistant", "st003")
```

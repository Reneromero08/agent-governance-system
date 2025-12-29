# Qwen CLI Skill

**Status**: ✓ Ready to use
**Model**: Qwen 2.5 (7B and 1.5B installed)
**Interface**: Batch file + Python CLI

## Quick Start

```bash
# From project root
cd "d:\CCC 2.0\AI\agent-governance-system"

# Ask a question
qwen "How do I parse JSON in Python?"

# Interactive mode
qwen
```

## What You Get

- **Fast Local AI**: No API costs, works offline
- **Multiple Interfaces**: Batch file, Python CLI, interactive REPL
- **File Analysis**: Can read and analyze code files
- **Conversation Memory**: Multi-turn conversations
- **Customizable**: JSON config for defaults

## Files

```
SKILLS/qwen-cli/
├── qwen.bat           - Windows batch launcher
├── qwen_cli.py        - Main Python CLI
├── config.json        - Configuration
├── SKILL.md           - Full documentation
├── QUICKSTART.md      - Quick reference
└── README.md          - This file
```

Plus: `qwen.bat` in project root for global access

## Usage Examples

### Simple Question
```bash
qwen "What is a Python generator?"
```

### Code Analysis
```bash
qwen "Explain this code" CORTEX\embeddings.py
```

### Interactive Mode
```bash
qwen

You: How do I create a class in Python?
[Qwen explains]

You: Show me an example
[Qwen provides code]

You: /quit
```

### Python API
```python
from qwen_cli import QwenCLI

cli = QwenCLI(model="qwen2.5:7b")
response = cli.ask("How do I sort a dictionary?")
print(response)
```

## Models Available

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| qwen2.5:1.5b | 986 MB | Fast | Quick questions |
| qwen2.5:7b | 4.7 GB | Balanced | **Default** |

Switch models:
```bash
python qwen_cli.py "question" --model qwen2.5:1.5b
```

## Interactive Commands

```
/file <path>  - Include file in next question
/clear        - Clear conversation history
/save         - Save conversation to JSON
/quit         - Exit
```

## Configuration

Edit `config.json`:
```json
{
  "model": "qwen2.5:7b",
  "temperature": 0.7,
  "max_tokens": 2000,
  "system_prompt": "You are a helpful coding assistant.",
  "stream": true
}
```

## Command Line Options

```bash
python qwen_cli.py --help

Options:
  question              Question to ask
  --file, -f           File to include
  --model, -m          Model to use
  --interactive, -i    Start REPL mode
  --system             System prompt
  --config, -c         Config file path
  --save               Save conversation
  --no-stream          Disable streaming
```

## Integration with AGS

The CLI understands AGS structure:
- Can read CORTEX database
- Knows CANON/CONTEXT/SKILLS layout
- Follows governance principles
- Assists with skill development

Example:
```bash
qwen "How do I add a new ADR to this project?"
```

## Performance

- **Startup**: ~1 second
- **Response**: ~0.5-2 seconds per token (CPU)
- **Memory**: ~6 GB for 7B model
- **Offline**: Works without internet

## Comparison to Cloud APIs

| Feature | Qwen Local | Claude/GPT-4 |
|---------|------------|--------------|
| Cost | Free | ~$0.015 per 1K tokens |
| Speed | 0.5-2 sec/token | 0.1-0.5 sec/token |
| Privacy | 100% local | Sent to cloud |
| Offline | ✓ Yes | ✗ No |
| Quality | Good for code | Excellent |
| Context | 32K tokens | 200K tokens |

**Use Qwen for**: Quick questions, code snippets, local/private work
**Use Claude for**: Complex tasks, large codebases, advanced reasoning

## Troubleshooting

**Ollama not running**:
```bash
ollama list  # Should show installed models
```

**Model not found**:
```bash
ollama pull qwen2.5:7b
```

**Slow responses**:
- Use smaller model: `--model qwen2.5:1.5b`
- Close other apps to free memory

**Import errors**:
```bash
pip install ollama
```

## Documentation

- [QUICKSTART.md](./QUICKSTART.md) - Quick reference
- [SKILL.md](./SKILL.md) - Full documentation
- [Ollama Docs](https://ollama.com/docs)

## Examples

### Debug Help
```bash
qwen "Why does Python say 'NameError: name 'x' is not defined'?"
```

### Code Generation
```bash
qwen "Write a function to calculate factorial"
```

### File Review
```bash
qwen "Review this for bugs" --file demo_semantic_dispatch.py
```

### Learning
```bash
qwen "Explain list comprehensions with examples"
```

## Tips

1. **Be specific**: "How do I X?" vs "Tell me about X"
2. **Include context**: Use `--file` for code questions
3. **Try both models**: 1.5b for speed, 7b for quality
4. **Save sessions**: Use `/save` for important conversations
5. **System prompts**: Use `--system` to set role/style

## Status

✓ CLI working
✓ Models installed (7b + 1.5b)
✓ Batch file ready
✓ Configuration available
✓ Documentation complete

Ready to use! Try: `qwen "Hello, can you help me code?"`

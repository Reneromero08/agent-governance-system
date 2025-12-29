# Qwen CLI - Local AI Assistant

**Version:** 1.0.0

**Status:** Active

**Required_Canon_Version:** >=2.0.0

**Purpose**: Provides a local CLI interface to Qwen 7B via Ollama for fast, offline AI assistance.

**Model**: Qwen2.5 7B (via Ollama)

**Use Cases**:
- Quick code questions without cloud API costs
- Offline development assistance
- Fast prototyping and testing
- Private/sensitive code analysis

## Features

- **Multiple Interfaces**: Batch file, Python CLI, interactive REPL
- **Context-Aware**: Can read files and provide code assistance
- **Streaming Output**: Real-time responses
- **Conversation Memory**: Multi-turn conversations
- **File Operations**: Read and analyze code files

## Usage

### Quick Start (Batch File)

```bash
# Ask a question
qwen.bat "How do I parse JSON in Python?"

# Analyze a file
qwen.bat "Explain this code" CORTEX/embeddings.py

# Interactive mode
qwen.bat
```

### Python CLI

```bash
# Direct question
python qwen_cli.py "What is a generator in Python?"

# With file context
python qwen_cli.py "Review this code" --file CORTEX/semantic_search.py

# Interactive REPL
python qwen_cli.py --interactive
```

### Advanced Options

```bash
# Specify model
python qwen_cli.py "question" --model qwen2.5:14b

# Control output length
python qwen_cli.py "question" --max-tokens 1000

# System prompt
python qwen_cli.py "question" --system "You are a Python expert"

# Save conversation
python qwen_cli.py --interactive --save conversation.json
```

## Installation

1. **Install Ollama** (if not already):
   - Download from https://ollama.com
   - Run installer

2. **Pull Qwen Model**:
   ```bash
   ollama pull qwen2.5:7b
   ```

3. **Verify Installation**:
   ```bash
   ollama list
   ```

## Model Options

Available Qwen models via Ollama:
- `qwen2.5:0.5b` - Fastest, smallest (500MB)
- `qwen2.5:1.5b` - Fast, lightweight (1.5GB)
- `qwen2.5:7b` - **Default**, balanced (4.7GB)
- `qwen2.5:14b` - More capable (8.9GB)
- `qwen2.5:32b` - Most capable (19GB)
- `qwen2.5-coder:7b` - Code-specialized

## Configuration

Edit `config.json` to set defaults:
```json
{
  "model": "qwen2.5:7b",
  "temperature": 0.7,
  "max_tokens": 2000,
  "system_prompt": "You are a helpful coding assistant.",
  "stream": true
}
```

## Examples

### Code Explanation
```bash
qwen.bat "Explain what this function does" CORTEX/embeddings.py
```

### Debug Help
```bash
qwen.bat "Why am I getting AttributeError: 'Row' object has no attribute 'get'?"
```

### Code Generation
```bash
qwen.bat "Write a function to calculate Fibonacci numbers in Python"
```

### Code Review
```bash
python qwen_cli.py "Review this for bugs" --file demo_semantic_dispatch.py
```

## Integration with AGS

The Qwen CLI integrates with the Agent Governance System:
- Can read CORTEX database
- Understands AGS file structure
- Follows CANON principles
- Can assist with skill development

## Performance

- **Startup**: ~1-2 seconds
- **Response time**: ~0.5-2 seconds per token (CPU)
- **Memory**: ~6GB for 7B model
- **Offline**: Works without internet

## Limitations

- Local model, smaller than Claude/GPT-4
- Best for focused questions, not large codebases
- Requires ~8GB RAM for 7B model
- CPU inference is slower than GPU

## Tips

1. **Be Specific**: "How do I X in Python?" vs "Tell me about Python"
2. **Provide Context**: Include file paths or code snippets
3. **Use System Prompts**: Set role for better responses
4. **Right-Size Model**: Use 1.5b for speed, 14b for quality
5. **Save Conversations**: Use `--save` for important sessions

## Troubleshooting

**Ollama not found**:
```bash
# Check if running
ollama list

# Start service (Windows)
# Ollama runs as a system service after install
```

**Model not available**:
```bash
ollama pull qwen2.5:7b
```

**Slow responses**:
- Try smaller model: `qwen2.5:1.5b`
- Reduce max_tokens
- Close other applications

**Connection refused**:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags
```

## Files

- `qwen.bat` - Windows batch wrapper
- `qwen_cli.py` - Main Python CLI
- `config.json` - Configuration
- `SKILL.md` - This file

## See Also

- [Ollama Documentation](https://ollama.com/docs)
- [Qwen Model Card](https://ollama.com/library/qwen2.5)
- AGS CORTEX for semantic search integration

<!-- CONTENT_HASH: 2e67d7e80be200c326ea291104a9c26ecc68f4cfaa01f4d454048a2cfbf03372 -->

# Qwen CLI - Quick Start

You're all set! The Qwen models are already installed.

## Usage

### From Project Root

```bash
# Ask a question
qwen "How do I use list comprehensions in Python?"

# Analyze a file
qwen "Explain this code" CORTEX\embeddings.py

# Interactive mode
qwen
```

### From SKILLS Directory

```bash
cd SKILLS\qwen-cli

# Direct Python
python qwen_cli.py "What is a decorator?"

# With file
python qwen_cli.py "Review this" --file ..\..\demo_semantic_dispatch.py

# Interactive REPL
python qwen_cli.py --interactive
```

## Available Models

You have these installed:
- `qwen2.5:1.5b` (986 MB) - Fast, for simple questions
- `qwen2.5:7b` (4.7 GB) - **Default**, best balance

Switch models:
```bash
python qwen_cli.py "question" --model qwen2.5:1.5b
```

## Examples

### Code Explanation
```bash
qwen "What does this do?" CORTEX\semantic_search.py
```

### Quick Help
```bash
qwen "How do I read a JSON file in Python?"
```

### Debug Assistance
```bash
qwen "Why am I getting 'list index out of range' error?"
```

### Code Generation
```bash
qwen "Write a function to merge two sorted lists"
```

### Interactive Session
```bash
qwen

You: How do I create a virtual environment?
[Qwen responds]

You: /file requirements.txt
Next question will include: requirements.txt

You: What packages are here?
[Qwen analyzes requirements.txt]

You: /quit
```

## Interactive Commands

- `/file <path>` - Include a file in next question
- `/clear` - Clear conversation history
- `/save` - Save conversation to JSON
- `/quit` - Exit

## Tips

1. **Be specific**: "How do I parse JSON?" vs "Tell me about Python"
2. **Use file context**: Include files for code reviews
3. **Try smaller model**: Use `qwen2.5:1.5b` for faster responses
4. **Save important sessions**: Use `/save` in interactive mode

## Performance

- **Startup**: ~1 second
- **Response**: ~0.5-2 seconds per token (CPU)
- **Memory**: ~5-6 GB for 7B model
- **Works offline**: No internet needed

## Configuration

Edit `config.json` to customize:
```json
{
  "model": "qwen2.5:7b",
  "temperature": 0.7,
  "max_tokens": 2000,
  "system_prompt": "You are a helpful coding assistant.",
  "stream": true
}
```

## Troubleshooting

**"Error: ollama package not installed"**
```bash
pip install ollama
```

**"Connection refused"**
- Ollama service should auto-start
- Check: `ollama list`

**Slow responses**
- Try faster model: `--model qwen2.5:1.5b`
- Reduce tokens: edit `max_tokens` in config.json

## Next Steps

- Try the demo: `qwen "Hello, can you help me code?"`
- Analyze AGS code: `qwen "Explain this" README.md`
- Use in workflow: `qwen "How do I add a new skill to AGS?"`

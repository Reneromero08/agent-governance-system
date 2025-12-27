# Model Setup Guide for MCP Startup

This guide helps you get a working model loaded in Ollama for your MCP network.

## Current Status

Your system has:
- ✅ Ollama server running
- ✅ MCP startup skill installed
- ⚠️ LFM2 model has compatibility issue ("missing tensor 'output_norm'")

## Option 1: Fix LFM2 (Currently Downloading)

A fresh LFM2 GGUF model is being downloaded from Hugging Face. Once complete:

```bash
cd d:\CCC\ 2.0\AI\agent-governance-system

# Check if download completed
ls -lh models_cache/models--LiquidAI--LFM2-2.6B-Exp-GGUF/

# Create new Modelfile pointing to fresh download
cat > Modelfile.lfm2-fresh << 'EOF'
FROM ./models_cache/models--LiquidAI--LFM2-2.6B-Exp-GGUF/snapshots/*/LFM2-2.6B-Exp-Q4_K_M.gguf
EOF

# Create Ollama model
ollama create lfm2 -f Modelfile.lfm2-fresh

# Test
python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "2+2"
```

## Option 2: Use Mistral (Quickest, Guaranteed to Work)

Mistral is lightweight and battle-tested with Ollama:

```bash
cd d:\CCC\ 2.0\AI\agent-governance-system

# Pull Mistral model
ollama pull mistral:latest

# Test it
python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "2+2"
```

Output should look like:
```
Helper: Using model 'mistral:latest'...
Helper: Sending prompt to Ollama (lfm2.gguf)...
2 + 2 = 4
```

(Note: It says "lfm2.gguf" in the debug message, but it's actually using whichever model is available)

## Option 3: Use Neural Chat (Balanced)

Good balance of size and quality:

```bash
ollama pull neural-chat:latest
```

## Available Models

List what you have:
```bash
ollama list
```

To see what's available in Ollama:
- **mistral:latest** - Fast, reliable, 7B params
- **neural-chat:latest** - Balanced, ~7B params
- **llama2:latest** - Popular, ~7B params
- **orca-mini:latest** - Smaller, ~3B params
- **lfm2:latest** - Your LFM2 (if fixed)

## Verification

After loading a model, verify it works:

```bash
# Check model is loaded
ollama list

# Test with your MCP startup script
python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --ollama-only

# Test inference
python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "Hello, how are you?"
```

## Troubleshooting

### Download Too Slow?
Models are 4-13GB. If slow:
1. Check internet connection
2. Try alternative: `ollama pull orca-mini` (smallest, ~2GB)
3. Download later, test with smaller model now

### Model Still Not Working?
```bash
# Clear Ollama and start fresh
ollama rm lfm2        # Remove problematic model
ollama list           # See what's left
ollama pull mistral   # Get a working one
```

### Test Direct Ollama API
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [{"role": "user", "content": "2+2"}],
    "stream": false
  }'
```

## Next Steps

1. **Wait for LFM2 Download to Complete** (Option 1)
   - Check: `ls models_cache/ | grep -i lfm`
   - Or proceed with Option 2 if impatient

2. **Load a Model** (Pick one option above)

3. **Verify** with: `python CATALYTIC-DPT/SKILLS/ant-worker/scripts/lfm2_runner.py "test"`

4. **Start Your MCP Network**
   ```bash
   python CATALYTIC-DPT/SKILLS/mcp-startup/scripts/startup.py --all
   ```

## Performance Notes

| Model | Size | Speed | Quality | RAM |
|-------|------|-------|---------|-----|
| orca-mini | 2GB | Fast | Good | 2.5GB |
| mistral | 4GB | Good | Good | 5GB |
| neural-chat | 7GB | Good | Great | 8GB |
| llama2 | 7GB | Good | Great | 8GB |
| lfm2 | 1.6GB | Fast | Decent | 2.2GB |

## Common Commands

```bash
# List loaded models
ollama list

# Pull new model
ollama pull [model-name]

# Test model directly
ollama run [model-name] "2+2"

# Delete model
ollama rm [model-name]

# View model info
ollama show [model-name]
```

## Integration with MCP

Once a model is working:

1. Your `lfm2_runner.py` automatically detects available models
2. The MCP startup skill can dispatch tasks
3. Ant workers poll for tasks and send to Ollama
4. Results come back through the MCP ledger

Everything is already configured - just need a working model!

---

**Status**: Awaiting fresh LFM2 download or user selection of alternative model.

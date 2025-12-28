#!/usr/bin/env python3
"""
Qwen CLI - Local AI Assistant via Ollama

Usage:
    python qwen_cli.py "your question"
    python qwen_cli.py "explain this code" --file path/to/file.py
    python qwen_cli.py --interactive
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    import ollama
except ImportError:
    print("Error: ollama package not installed")
    print("Install with: pip install ollama")
    sys.exit(1)


class QwenCLI:
    """Command-line interface for Qwen via Ollama."""

    def __init__(self, model="qwen2.5:7b", config_path=None):
        self.model = model
        self.config = self.load_config(config_path)
        self.conversation = []

    def load_config(self, config_path):
        """Load configuration from JSON file."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)

        # Default config
        return {
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 2000,
            "system_prompt": "You are a helpful coding assistant.",
            "stream": True
        }

    def read_file(self, file_path):
        """Read a file and return its contents."""
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"File: {file_path}\n\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"

    def ask(self, question, file_path=None, system_prompt=None):
        """Ask Qwen a question."""
        # Build prompt
        prompt = question
        if file_path:
            file_content = self.read_file(file_path)
            prompt = f"{question}\n\n{file_content}"

        # Add to conversation
        messages = []

        # System prompt
        if system_prompt or self.config.get("system_prompt"):
            messages.append({
                "role": "system",
                "content": system_prompt or self.config["system_prompt"]
            })

        # Add conversation history
        messages.extend(self.conversation)

        # Add current question
        messages.append({
            "role": "user",
            "content": prompt
        })

        # Call Ollama
        try:
            if self.config.get("stream", True):
                return self._stream_response(messages)
            else:
                return self._batch_response(messages)
        except Exception as e:
            return f"Error: {e}\n\nIs Ollama running? Try: ollama list"

    def _stream_response(self, messages):
        """Get streaming response from Ollama."""
        print(f"\n[Qwen {self.model}]:\n", flush=True)

        full_response = ""
        try:
            stream = ollama.chat(
                model=self.config["model"],
                messages=messages,
                stream=True,
                options={
                    "temperature": self.config.get("temperature", 0.7),
                    "num_predict": self.config.get("max_tokens", 2000)
                }
            )

            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                full_response += content

            print("\n")

            # Add to conversation history
            self.conversation.append({"role": "user", "content": messages[-1]["content"]})
            self.conversation.append({"role": "assistant", "content": full_response})

            return full_response

        except Exception as e:
            print(f"\nError: {e}")
            return None

    def _batch_response(self, messages):
        """Get batch response from Ollama."""
        print(f"\n[Qwen {self.model}]: ", end='', flush=True)

        try:
            response = ollama.chat(
                model=self.config["model"],
                messages=messages,
                options={
                    "temperature": self.config.get("temperature", 0.7),
                    "num_predict": self.config.get("max_tokens", 2000)
                }
            )

            content = response['message']['content']
            print(content)

            # Add to conversation history
            self.conversation.append({"role": "user", "content": messages[-1]["content"]})
            self.conversation.append({"role": "assistant", "content": content})

            return content

        except Exception as e:
            print(f"Error: {e}")
            return None

    def interactive(self, save_path=None):
        """Start interactive REPL mode."""
        print(f"Qwen Interactive Mode (model: {self.config['model']})")
        print("Type your question, or:")
        print("  /file <path>  - Include a file in next question")
        print("  /clear        - Clear conversation history")
        print("  /save         - Save conversation")
        print("  /quit         - Exit")
        print()

        current_file = None

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Commands
                if user_input.startswith('/'):
                    cmd_parts = user_input.split(maxsplit=1)
                    cmd = cmd_parts[0].lower()

                    if cmd == '/quit' or cmd == '/exit':
                        print("Goodbye!")
                        break

                    elif cmd == '/clear':
                        self.conversation = []
                        current_file = None
                        print("Conversation cleared.")
                        continue

                    elif cmd == '/file':
                        if len(cmd_parts) > 1:
                            current_file = cmd_parts[1]
                            print(f"Next question will include: {current_file}")
                        else:
                            print("Usage: /file <path>")
                        continue

                    elif cmd == '/save':
                        path = save_path or f"conversation_{datetime.now():%Y%m%d_%H%M%S}.json"
                        self.save_conversation(path)
                        print(f"Conversation saved to: {path}")
                        continue

                    else:
                        print(f"Unknown command: {cmd}")
                        continue

                # Ask question
                self.ask(user_input, file_path=current_file)
                current_file = None  # Reset after use

            except KeyboardInterrupt:
                print("\n\nUse /quit to exit")
            except EOFError:
                print("\nGoodbye!")
                break

    def save_conversation(self, path):
        """Save conversation to JSON file."""
        data = {
            "model": self.config["model"],
            "timestamp": datetime.now().isoformat(),
            "messages": self.conversation
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen CLI - Local AI Assistant via Ollama"
    )
    parser.add_argument(
        "question",
        nargs='?',
        help="Question to ask Qwen"
    )
    parser.add_argument(
        "--file", "-f",
        help="File to include in the question"
    )
    parser.add_argument(
        "--model", "-m",
        default="qwen2.5:7b",
        help="Ollama model to use (default: qwen2.5:7b)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive mode"
    )
    parser.add_argument(
        "--system",
        help="System prompt to use"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--save",
        help="Save conversation to file (interactive mode only)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )

    args = parser.parse_args()

    # Create CLI instance
    cli = QwenCLI(model=args.model, config_path=args.config)

    if args.no_stream:
        cli.config["stream"] = False

    # Interactive mode
    if args.interactive:
        cli.interactive(save_path=args.save)
        return 0

    # Single question mode
    if args.question:
        response = cli.ask(
            args.question,
            file_path=args.file,
            system_prompt=args.system
        )

        if response is None:
            return 1
        return 0

    # No arguments - show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

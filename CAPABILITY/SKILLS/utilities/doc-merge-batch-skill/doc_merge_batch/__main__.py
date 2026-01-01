"""Entry point for python -m doc_merge_batch"""
from .cli import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli())

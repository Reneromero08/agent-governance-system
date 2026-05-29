"""Entry point for running eigen-alignment as a module.

Usage:
    python -m eigen_alignment anchors build words.txt
    python -m eigen_alignment signature compute anchors.json --model all-MiniLM-L6-v2
"""

from cli.main import main

if __name__ == '__main__':
    main()

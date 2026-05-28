#!/usr/bin/env pythonw
"""Console-free .holo opener for Windows."""

from pathlib import Path
import sys

from holo_native_viewer import HoloNativeViewer


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: holo_open.pyw <file.holo>")
    path = Path(sys.argv[1]).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    HoloNativeViewer(path).run()


if __name__ == "__main__":
    main()

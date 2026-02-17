"""
Entry point for the CLI: run from any directory.
Usage: safety-vest
"""
import os
import sys
from pathlib import Path

# Project root (where this file lives)
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.main import main as cli_main
    sys.exit(cli_main())


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

# Make the `llm_platform` package importable when tests run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

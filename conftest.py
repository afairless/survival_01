"""Ensure the repository root is importable so ``from src...`` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Shared pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

for directory in (PROJECT_ROOT, SOURCE_DIRECTORY):
    directory_string = str(directory)

    if directory_string not in sys.path:
        sys.path.insert(0, directory_string)
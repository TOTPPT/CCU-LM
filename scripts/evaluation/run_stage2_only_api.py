#!/usr/bin/env python3
"""Stage2-only API entry (generation + point evaluation)."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from stage2_only_fewshot_api import main
if __name__ == "__main__":
    main()

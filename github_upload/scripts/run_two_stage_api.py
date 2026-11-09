#!/usr/bin/env python3
"""Two-stage API entry (generation + evaluation)."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vllm_lora_example_api import main
if __name__ == "__main__":
    main()

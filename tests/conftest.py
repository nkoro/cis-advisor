"""Test fixtures.

Puts the ``app/`` directory on ``sys.path`` so ``import main`` resolves exactly
the way ``python app/main.py`` resolves it in production.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

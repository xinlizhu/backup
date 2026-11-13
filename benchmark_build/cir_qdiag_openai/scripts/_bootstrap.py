"""Make direct script execution import this project's src/cir_qdiag first."""

from __future__ import annotations

import sys
from pathlib import Path


LOCAL_SRC = str(Path(__file__).resolve().parents[1] / "src")
if sys.path[0] != LOCAL_SRC:
    sys.path.insert(0, LOCAL_SRC)

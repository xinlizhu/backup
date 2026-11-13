from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cir_qdiag.utils.hashing import sha256_text

def load_prompt(prompt_dir: str | Path, stem: str, version: str = "v5") -> tuple[str, str]:
    path = Path(prompt_dir) / f"{stem}_{version}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    return text, sha256_text(text)


def dump_prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

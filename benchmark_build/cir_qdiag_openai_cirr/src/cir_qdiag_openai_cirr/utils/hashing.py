from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(payload: Any, image_paths: list[str] | None = None) -> str:
    images = []
    for raw in image_paths or []:
        path = Path(raw)
        try:
            stat = path.stat()
            images.append(
                {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            )
        except FileNotFoundError:
            images.append({"path": str(path), "missing": True})
    return sha256_text(stable_json({"payload": payload, "images": images}))


def deterministic_seed(*parts: Any, modulo: int = 2**31 - 1) -> int:
    digest = sha256_text(stable_json(parts))
    return int(digest[:16], 16) % modulo

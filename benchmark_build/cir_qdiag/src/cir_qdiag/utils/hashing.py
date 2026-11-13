from __future__ import annotations

import hashlib
import json
from functools import lru_cache
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


def local_model_fingerprint(path: str | Path) -> str | None:
    """Fingerprint local model metadata and weight-file identity cheaply.

    Multi-gigabyte weights are represented by path, size, and nanosecond mtime;
    small configuration/tokenizer files are content-hashed. This catches the
    normal case of replacing a local checkpoint in place without reading every
    weight byte for each pipeline invocation.
    """

    return _local_model_fingerprint_cached(str(Path(path).resolve()))


@lru_cache(maxsize=16)
def _local_model_fingerprint_cached(resolved_path: str) -> str | None:
    root = Path(resolved_path)
    if not root.exists():
        return None
    if root.is_file():
        stat = root.stat()
        return sha256_text(
            stable_json(
                {
                    "path": str(root),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        )

    metadata_suffixes = {".json", ".jinja", ".model", ".py", ".txt"}
    weight_suffixes = {".bin", ".pt", ".safetensors"}
    manifest: list[dict[str, Any]] = []
    for item in sorted(root.iterdir(), key=lambda value: value.name):
        if not item.is_file():
            continue
        stat = item.stat()
        row: dict[str, Any] = {
            "name": item.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if item.suffix.casefold() in metadata_suffixes:
            row["sha256"] = sha256_file(item)
        elif item.suffix.casefold() not in weight_suffixes:
            continue
        manifest.append(row)
    return sha256_text(stable_json({"root": str(root), "files": manifest}))


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
